# ==============================================================================
# Qwen3-VL-4B-Instruct + LocateAnything-3B — 单镜像双服务 CUDA 推理镜像
#
# 基础镜像 : nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04（本地已缓存，无需拉取；
#            runtime 基底在 docker.io 直连会超时，故 final 阶段沿用 devel 基底，
#            与 qwen3.8-27B-deploy 同一取舍）
# 构建内容 : llama-server（qwen3-vl-4b）+ locate-anything-server / -cli（locate-anything-3b）
# 源码获取 : llama.cpp 构建时从 GitHub 浅取固定 commit；locate-anything.cpp 使用
#            本目录内 vendored 的源码（locate-anything.cpp/，含新增的 examples/server），
#            ggml 在构建期按固定 commit 浅取（与原 submodule pin 一致）
# 使用方式 : 见 docker-compose.yml；模型权重通过 volume 挂载，不打进镜像
#
# 说明：
#   * 构建上下文即本部署目录（context: .），由本目录 .dockerignore 裁剪。
#   * 容器构建阶段没有可见的 GPU，必须显式指定 -DCMAKE_CUDA_ARCHITECTURES；
#     链接 CUDA driver 时用 stubs/libcuda.so.1 + -rpath-link 技巧（详见 la-build
#     阶段注释），运行时由 nvidia-container-toolkit 注入真实驱动库。
# ==============================================================================

# ---------- stage 1: llama.cpp llama-server ----------
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS llama-build

# 固定 llama.cpp commit（该 commit 已支持 qwen3vl / mtmd qwen3vl）
# 注意：git fetch 按 SHA 拉取需要完整 40 位（缩写 SHA 会报 "couldn't find remote ref"）
ARG LLAMA_CPP_COMMIT=70aff25250075bf23b533c207b55168a4f926350
# CUDA 架构（分号分隔的多架构列表；单卡环境可覆盖如 86 以加速构建。
# 注意：传给 cmake 时须加引号，否则 shell 会把分号当作命令分隔符）
ARG CUDA_ARCHS=70;75;80;86;89;90

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_PARALLEL_LEVEL=16

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake git pkg-config curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/llama.cpp
RUN git init -q . \
 && git remote add origin https://github.com/ggml-org/llama.cpp \
 && git fetch --depth 1 origin "${LLAMA_CPP_COMMIT}" \
 && git checkout -q FETCH_HEAD

# 驱动 stub 说明：构建期容器内没有 GPU，libggml-cuda.so 链接 CUDA::cuda_driver 时会
# 记录 DT_NEEDED libcuda.so.1（driver 只提供 stubs/libcuda.so，其 SONAME 就是 .so.1）。
# 下游链接可执行文件时 ld 找不到 libcuda.so.1 会报 "undefined reference to cuMem*"。
# 注意：-L 对传递依赖（shared lib 的 DT_NEEDED）无效，必须用 -rpath-link。
# 故在 stubs 目录补一个 libcuda.so.1 符号链接，并通过 -rpath-link 指向 stubs 目录。
# 运行时无需担心：nvidia-container-toolkit 会把宿主真实 libcuda.so.1 挂进容器并通过
# LD_LIBRARY_PATH 优先加载，stub 不会被用到。
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DGGML_CUDA=ON \
      -DGGML_CUDA_FA_ALL_QUANTS=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DGGML_NATIVE=OFF \
      -DLLAMA_CURL=OFF \
      -DGGML_CCACHE=OFF \
      -DLLAMA_BUILD_TESTS=OFF \
      -DLLAMA_BUILD_SERVER=ON \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
 && cmake --build build --target llama-server

# ---------- stage 2: locate-anything.cpp server + cli ----------
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 AS la-build

# ggml 子模块固定 commit（与 locate-anything.cpp/third_party/ggml 的 submodule pin 一致）
ARG GGML_COMMIT=7142aa6bf9fcaeec0fef8d80fcd90afe4268adf1
ARG CUDA_ARCHS=70;75;80;86;89;90

ENV DEBIAN_FRONTEND=noninteractive \
    CMAKE_BUILD_PARALLEL_LEVEL=16

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake git pkg-config curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# 本地源码（含新增的 examples/server）随构建上下文带入；.git 已被 .dockerignore 排除，
# 因此 ggml 子模块无法走 git submodule update，改为按固定 commit 浅取（与 submodule pin 一致）
WORKDIR /opt/locate-anything.cpp
COPY locate-anything.cpp/ /opt/locate-anything.cpp/
RUN rm -rf third_party/ggml \
 && git init -q third_party/ggml \
 && git -C third_party/ggml remote add origin https://github.com/ggml-org/ggml \
 && git -C third_party/ggml fetch --depth 1 origin "${GGML_COMMIT}" \
 && git -C third_party/ggml checkout -q FETCH_HEAD

# ggml 静态链接进 locate_anything（BUILD_SHARED_LIBS=OFF），但 libggml-cuda 同样
# 需要 driver stub 链接技巧（见 llama-build 阶段注释）
RUN ln -sf libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1 \
 && cmake -B build \
      -DCMAKE_BUILD_TYPE=Release \
      -DLA_GGML_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES="${CUDA_ARCHS}" \
      -DLA_BUILD_CLI=ON \
      -DLA_BUILD_SERVER=ON \
      -DLA_BUILD_TESTS=OFF \
      -DCMAKE_EXE_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
      -DCMAKE_SHARED_LINKER_FLAGS="-Wl,-rpath-link,/usr/local/cuda/lib64/stubs" \
 && cmake --build build --target locate-anything-server locate-anything-cli

# ---------- final: 运行镜像（单镜像双二进制） ----------
FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

ARG LLAMA_CPP_COMMIT=70aff25250075bf23b533c207b55168a4f926350
LABEL org.opencontainers.image.source="https://github.com/ggml-org/llama.cpp + locate-anything.cpp" \
      llama.cpp.commit="${LLAMA_CPP_COMMIT}" \
      description="llama-server (Qwen3-VL-4B) + locate-anything-server (LocateAnything-3B), CUDA multi-arch"

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# llama-server 及其共享库（libggml*.so 等都在 build/bin 下）
COPY --from=llama-build /opt/llama.cpp/build/bin/ /opt/llama.cpp/build/bin/
# locate-anything 二进制（ggml 静态链接，无外部 libggml 依赖）
COPY --from=la-build /opt/locate-anything.cpp/build/examples/server/locate-anything-server /opt/locate-anything/bin/
COPY --from=la-build /opt/locate-anything.cpp/build/examples/cli/locate-anything-cli       /opt/locate-anything/bin/

ENV PATH="/opt/locate-anything/bin:/opt/llama.cpp/build/bin:${PATH}" \
    LD_LIBRARY_PATH="/opt/llama.cpp/build/bin:${LD_LIBRARY_PATH:-}" \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility

WORKDIR /opt
# 不设 ENTRYPOINT，启动命令由 compose / docker run 显式传入，便于调试
CMD ["bash", "-c", "llama-server --help | head -5; echo ---; locate-anything-server --help"]
