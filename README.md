# Qwen3-VL-4B + LocateAnything-3B 多模态部署

单个 Docker 镜像、两个推理服务，外加一个宿主机安全检测流水线（大小模型结合）：

| 组件 | 二进制/脚本 | 模型 | 宿主端口 |
| --- | --- | --- | --- |
| `qwen3-vl` | `llama-server`（llama.cpp，OpenAI 兼容） | `Qwen3-VL-4B-Instruct-Q8_0.gguf` + `mmproj-F16.gguf` | 8090 |
| `locate-anything` | `locate-anything-server` | `locate-anything-q8_0.gguf` | 8091 |
| `detect-pipeline/` | `yoloworld_vlm_safety.py` / `locate_vlm_safety.py` | YOLO-World 或 LocateAnything 检测人形 + Qwen3-VL 判定吸烟/安全帽/反光衣 | —（宿主机运行） |

- 镜像 `qwen3-vl-4b-locate-3b:latest` 多阶段构建（`Dockerfile`，默认多 CUDA 架构，可用 `CUDA_ARCHS` 覆盖加速，如单卡 `75`）。
- 模型通过 volume 只读挂载（默认 `./models`），不打进镜像。
- 开发与运维细节（构建注意事项、网络镜像、验证方法等）见 [`AGENTS.md`](AGENTS.md)。

## 前置条件

- NVIDIA 驱动 + nvidia-container-toolkit；Docker。
- 模型权重（需 `pip install modelscope`，支持断点续传，也可按目标分别下载）：

  ```bash
  bash scripts/download-models.sh                    # 全部 → ./models/
  bash scripts/download-models.sh qwen3-vl           # 只下 VL 模型 + mmproj
  bash scripts/download-models.sh locate-anything    # 只下 locate 模型
  ```

- 构建期需要访问 GitHub（浅取 llama.cpp 与 ggml 固定 commit）。

## 启动 / 停止

```bash
bash scripts/start.sh --build                 # 首次：构建镜像并启动全部服务（CUDA 编译约 20–80 分钟）
bash scripts/start.sh                         # 之后：直接启动
bash scripts/start.sh --service qwen3-vl      # 只启动 VL 服务（别名 vl）
bash scripts/start.sh --service locate-anything
bash scripts/start.sh --smoke                 # 启动后自动冒烟
docker compose down                           # 停止
docker compose logs -f                        # 日志
```

可选覆盖：`GPU_ID=1 VL_PORT=9090 LOCATE_PORT=9091 MODELS_DIR=/path/models VL_CTX=65536 CUDA_ARCHS=75`。

> 本机无 compose 插件时，用 `docker run` 等价命令（见 AGENTS.md「常用命令」）。

## API

### Qwen3-VL（llama-server，OpenAI 兼容）

```bash
curl http://127.0.0.1:8090/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "Qwen3-VL-4B-Instruct",
  "messages": [{"role": "user", "content": [
    {"type": "image_url", "image_url": {"url": "data:image/png;base64,'"$(base64 -w0 img.png)"'"}},
    {"type": "text", "text": "描述这张图"}
  ]}]
}'
```

### LocateAnything（locate-anything-server）

```bash
curl http://127.0.0.1:8091/health        # {"status":"ok"}

curl http://127.0.0.1:8091/v1/detect \
  -F image=@street.jpg \
  -F 'prompt=Locate all the instances that matches the following description: person</c>car.' \
  -F mode=hybrid
# {"detections":[{"label":"person","box":[x1,y1,x2,y2]}, ...]}
```

- 也接受 JSON：`{"image_base64": "...", "prompt": "...", "mode": "hybrid", "max_new": 256}`；`mode`：`hybrid`（默认）/ `slow` / `fast`。
- **限制**：引擎有状态，推理请求互斥串行，不支持并发；`box` 为原图像素坐标；超大图会撑爆显存，客户端需限制送图边长（见 AGENTS.md）。

## 测试

```bash
bash scripts/smoke-test.sh     # VL 带图问答 + locate detect 冒烟
bash scripts/parity-test.sh    # 容器内 CLI 与 server 输出逐位一致性 + slow/fast 覆盖
```

两脚本默认连 `127.0.0.1:8090/8091`，可传参覆盖；`MEDIA_DIR` 可换测试图片目录。

## 目录结构

```
qwen3-vl-4b-locate-anything-3b-deploy/
├── Dockerfile              # 单镜像多阶段：llama-server + locate-anything-server/-cli
├── docker-compose.yml      # 一个 image、两个 service
├── AGENTS.md               # 开发/运维指南（构建、网络镜像、验证方法、已知坑）
├── detect-pipeline/        # 安全检测流水线（宿主机）：小模型检测 + VLM 判定
│   ├── yoloworld_vlm_safety.py   # YOLO-World 版（本地实时）
│   ├── locate_vlm_safety.py      # LocateAnything 版（开放式类别）
│   ├── requirements.txt
│   └── README.md                 # 流水线部署/配置/排障
├── LICENSE                 # 本仓库自身代码：MIT
├── locate-anything.cpp/    # vendored 源码（含新增的 examples/server）
│   └── benchmarks/media/   # 测试图片（smoke/parity 脚本与容器 /media 挂载使用）
├── models/                 # 模型权重（download-models.sh 下载；已被 gitignore/dockerignore 排除）
└── scripts/                # download-models.sh / start.sh / smoke-test.sh / parity-test.sh
```

## License

本仓库自身代码采用 **MIT**（见根目录 `LICENSE`）；权重与引擎由各上游项目单独授权：

| 组件 | 许可证 |
| --- | --- |
| llama.cpp / ggml / locate-anything.cpp / cpp-httplib | MIT |
| Qwen3-VL-4B-Instruct 权重 | Apache-2.0 |
| LocateAnything-3B 权重 | NVIDIA 许可证（非 OSI 标准，商用前请阅读其 LICENSE） |
| nvidia/cuda 基础镜像 | NVIDIA CUDA EULA + Ubuntu 组件各自许可证 |
