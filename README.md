# Qwen3-VL-4B + LocateAnything-3B 多模态部署

单个 Docker 镜像、两个推理服务：

| 服务 | 二进制 | 模型 | 容器端口 | 宿主默认端口 |
| ---- | ------ | ---- | -------- | ------------ |
| `qwen3-vl` | `llama-server`（llama.cpp @ `70aff25`，OpenAI 兼容） | `models/qwen3-vl-4b/Qwen3-VL-4B-Instruct-Q8_0.gguf` + `mmproj-F16.gguf` | 8080 | 8090（`VL_PORT`） |
| `locate-anything` | `locate-anything-server`（locate-anything.cpp，新增 HTTP 前端） | `models/locate-anything-3b/locate-anything-q8_0.gguf` | 8081 | 8091（`LOCATE_PORT`） |

- 镜像：`qwen3-vl-4b-locate-3b:latest`，多阶段构建（`Dockerfile`），默认多 CUDA 架构（`CUDA_ARCHS=70;75;80;86;89;90`，可覆盖如 `86` 加速构建）。
- 模型通过 volume 只读挂载（默认 `./models`，即本部署目录内），不打进镜像。
- 两个服务默认共用 `GPU_ID=0`；RTX 3090 24GB 可同时容纳两者（VL Q8_0 ≈4.5GB + mmproj 0.8GB + KV，locate q8_0 ≈3GB）。

## 前置条件

- NVIDIA 驱动 + nvidia-container-toolkit；Docker（含 compose 插件）。
- 模型文件（可用脚本一键下载，需 `pip install modelscope`，可按目标分别下载）：

  ```bash
  bash scripts/download-models.sh                    # 全部 → ./models/（支持断点续传）
  bash scripts/download-models.sh qwen3-vl           # 只下 VL 模型 + mmproj
  bash scripts/download-models.sh locate-anything    # 只下 locate 模型
  ```

  等价于：

  ```bash
  modelscope download --model unsloth/Qwen3-VL-4B-Instruct-GGUF \
      Qwen3-VL-4B-Instruct-Q8_0.gguf mmproj-F16.gguf --local_dir ./models/qwen3-vl-4b
  modelscope download --model mudler/locate-anything.cpp-gguf \
      locate-anything-q8_0.gguf --local_dir ./models/locate-anything-3b
  ```

  目标文件：
  - `models/qwen3-vl-4b/Qwen3-VL-4B-Instruct-Q8_0.gguf`
  - `models/qwen3-vl-4b/mmproj-F16.gguf`
  - `models/locate-anything-3b/locate-anything-q8_0.gguf`
- 构建期需要访问 GitHub（浅取 llama.cpp 与 ggml 的固定 commit）。

## 启动 / 停止

```bash
bash scripts/start.sh --build     # 首次：构建镜像并启动（CUDA 编译约 20–80 分钟）
bash scripts/start.sh             # 之后：直接启动全部服务
bash scripts/start.sh --service qwen3-vl         # 只启动 VL 服务（别名 vl）
bash scripts/start.sh --service locate-anything  # 只启动检测服务（别名 locate）
bash scripts/start.sh --smoke     # 启动后自动冒烟（只测本次启动的服务）
docker compose down               # 停止
docker compose logs -f            # 日志
```

可选覆盖：`GPU_ID=1 VL_PORT=9090 LOCATE_PORT=9091 MODELS_DIR=/path/models VL_CTX=65536 bash scripts/start.sh`。

> 显存说明：`qwen3-vl` 显式限定 `-c 32768`（`VL_CTX` 可调）。llama-server 的 `--fit on`
> 若不配 `-c` 会把上下文撑到吃满整张卡——24GB 卡上曾规划出 26 万 token 的 KV，
> 权重 4.7GB 的模型实测占到 17GB。限定 32k 后 VL 服务约 7GB、locate 服务约 6.5GB，
> 两服务合计约 13.5GB，一张 24GB 卡还可余出空间跑 `parity-test.sh` 的 CLI 基准。

## API

### Qwen3-VL-4B（llama-server，OpenAI 兼容）

```bash
curl http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3-VL-4B-Instruct",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 img.png)"'"}},
    {"type": "text", "text": "描述这张图"}
  ]}]
}'
```

### LocateAnything-3B（locate-anything-server）

```bash
curl http://127.0.0.1:8091/health
# {"status":"ok"}

curl http://127.0.0.1:8091/v1/detect \
  -F image=@street.jpg \
  -F 'prompt=Locate all the instances that matches the following description: person</c>car.' \
  -F mode=hybrid
# {"detections":[{"label":"person","box":[x1,y1,x2,y2]}, ...],
#  "prompt":"...", "mode":"hybrid", "image":{"width":W,"height":H}, "elapsed_ms":N}
```

- 也接受 JSON：`{"image_base64": "...", "prompt": "...", "mode": "hybrid", "max_new": 256}`。
- `mode`：`hybrid`（默认，Parallel Box Decoding + AR 回退）/ `slow`（纯自回归）/ `fast`（仅 MTP）。
- **限制**：引擎有状态，推理请求经互斥锁串行执行，不支持并发；`box` 为原图像素坐标。
- 服务端实现：`locate-anything.cpp/examples/server/main.cpp`（cpp-httplib，vendored 于 `third_party/cpp-httplib`）。

## 测试

```bash
bash scripts/smoke-test.sh      # VL 带图问答 + locate detect 冒烟（用 benchmarks/media/bus_in.png）
bash scripts/parity-test.sh     # 对 media 三张图比对容器内 CLI 与 server 输出（应逐位一致），并覆盖 slow/fast
```

两个脚本默认连 `127.0.0.1:8090/8091`，可传参覆盖；`MEDIA_DIR` 可替换测试图片目录
（默认 `../locate-anything.cpp/benchmarks/media`）。

## 目录结构

```
qwen3-vl-4b-locate-anything-3b-deploy/
├── Dockerfile              # 单镜像多阶段：llama-server + locate-anything-server/-cli
├── docker-compose.yml      # 一个 image、两个 service
├── locate-anything.cpp/    # vendored 源码（含新增的 examples/server，即 HTTP server）
│   └── benchmarks/media/   # 测试图片（smoke/parity 脚本与容器 /media 挂载使用）
├── models/                 # 模型权重（download-models.sh 下载；已被 .gitignore/.dockerignore 排除）
└── scripts/                # download-models.sh / start.sh / smoke-test.sh / parity-test.sh
```

`locate-anything.cpp/` 是上游 [mudler/locate-anything.cpp](https://github.com/mudler/locate-anything.cpp)
的 vendored 副本（剔除 `.git`、`build/`、`models/`；`third_party/ggml` 为完整源码），新增的
`examples/server`（`locate-anything-server`）就在其中。本目录整体可纳入 GitHub 管理
（`.gitignore` 已排除本地构建产物）。

## 构建上下文的说明

compose 的 `build.context` 即本目录，由本目录 `.dockerignore` 剔除
`locate-anything.cpp/build` 等本地产物。llama.cpp 与 ggml 不随 context 打包，
而是在构建期按固定 commit 从 GitHub 浅取（`Dockerfile` 中的 `LLAMA_CPP_COMMIT` /
`GGML_COMMIT`），因此 context 仅约 36MB。
