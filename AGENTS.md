# AGENTS.md — 开发/运维指南

面向在本仓库工作的 agent/开发者。用户-facing 的项目介绍、API 示例、许可证见
`README.md`；检测流水线的部署与配置见 `detect-pipeline/README.md`。本文只放
运维操作与环境要点，与上述文件不重复。

## 组件与端口

| 组件 | 容器名 | 端口 | GPU（当前） | 健康检查 |
| --- | --- | --- | --- | --- |
| Qwen3-VL-4B（llama-server） | `qwen3-vl-server` | 8090→8080 | 0 | `curl :8090/health` |
| LocateAnything-3B | `locate-anything-server` | 8091→8081 | 1 | `curl :8091/health` |
| 检测流水线（宿主机进程） | — | — | 0（YOLO 版 torch） | 日志 `[heartbeat]` |

模型文件（volume 挂载 `models/`，只读）：
- `models/qwen3-vl-4b/Qwen3-VL-4B-Instruct-Q8_0.gguf`（4.28GB）+ `mmproj-F16.gguf`（836MB）
- `models/locate-anything-3b/locate-anything-q8_0.gguf`
- YOLO 版检测权重 `models/yolov8s-worldv2.pt`（26MB， ultralytics 首次运行可自动下载）

## 本机环境要点（重要）

- **4× RTX 2080 Ti 11GB**；小显存卡：VLM 与 locate 服务**必须分卡**（同卡共 10GB 会 OOM）。
- **docker.io 直连超时**：`docker.1ms.run` 镜像站可用。基础镜像需本地缓存，Dockerfile 假设
  `nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04` 已存在：
  ```bash
  docker pull docker.1ms.run/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
  docker tag  docker.1ms.run/nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04 \
              nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04
  ```
- **无 docker compose 插件**（Ubuntu docker.io 包）。`scripts/start.sh` 里
  `docker compose up -d` 会失败；构建可用（脚本里的 `docker build` 正常），启动服务用
  下面的 `docker run` 等价命令。
- **pip 下载大 wheel 会中途停滞**（download.pytorch.org 与阿里云镜像都出现 0 速度僵死）。
  已验证方案：解析 simple 索引拿 URL → `curl -C - --speed-limit 102400 --speed-time 20
  --retry 1000 --retry-all-errors` 断点续传 → `pip install --no-index --find-links` 离线装。
  解析器脚本在 `/tmp/resolve.py`（会话产物，可参考重建）。
- **无 sudo**（nvidia-container-toolkit 为管理员手工安装，已验证 GPU 容器可用）。
- 宿主机 Python 环境：`.venv/`（3.10，已装 torch 2.7.0+cu126 / ultralytics 8.4.184 /
  opencv-python-headless 5.0 / edge-tts 7.3.1）。ultralytics 会自动从 GitHub 装 CLIP，
  权重下到 `weights/clip/`（已 gitignore）。
- 构建镜像务必带 `CUDA_ARCHS=75`（单架构加速，默认 6 架构 20-80 分钟）。

## 常用命令

### 构建镜像

```bash
CUDA_ARCHS=75 bash scripts/start.sh --build --service qwen3-vl   # start.sh 只用到其中的 docker build
```

### 启动/停止服务（docker run 等价，compose 缺失时）

```bash
# Qwen3-VL（GPU 0）
docker run -d --name qwen3-vl-server --restart unless-stopped \
  --gpus device=0 -e NVIDIA_VISIBLE_DEVICES=0 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$PWD/models:/models:ro" -p 8090:8080 \
  qwen3-vl-4b-locate-3b:latest \
  llama-server --model /models/qwen3-vl-4b/Qwen3-VL-4B-Instruct-Q8_0.gguf \
    --mmproj /models/qwen3-vl-4b/mmproj-F16.gguf \
    --host 0.0.0.0 --port 8080 --alias Qwen3-VL-4B-Instruct,qwen3-vl-4b \
    --jinja --tools all --metrics --parallel 1 --flash-attn on \
    -c 32768 --fit on --fit-target 768 -b 2048 -ub 256 \
    --cache-type-k q4_0 --cache-type-v q4_0

# LocateAnything（GPU 1）
docker run -d --name locate-anything-server --restart unless-stopped \
  --gpus device=1 -e NVIDIA_VISIBLE_DEVICES=1 \
  -e NVIDIA_DRIVER_CAPABILITIES=compute,utility \
  -v "$PWD/models:/models:ro" -p 8091:8081 \
  qwen3-vl-4b-locate-3b:latest \
  locate-anything-server --model /models/locate-anything-3b/locate-anything-q8_0.gguf \
    --host 0.0.0.0 --port 8081 --threads 4

docker stop/start <容器名>；docker logs -f <容器名>
```

### 检测流水线（宿主机）

```bash
SAVE_VIDEO=1 SAVE_JSONL=1 .venv/bin/python detect-pipeline/yoloworld_vlm_safety.py
LOCATE_BASE_URL=http://127.0.0.1:8091 FRAME_INTERVAL=10 \
  .venv/bin/python detect-pipeline/locate_vlm_safety.py
# 常用环境变量见 detect-pipeline/README.md；启动会打印 [CFG] 全量配置横幅
```

### 测试

```bash
bash scripts/smoke-test.sh        # 两服务冒烟（需 curl + python3）
bash scripts/parity-test.sh       # CLI vs server 一致性
```

流水线验证手法：从 test.mp4 用 cv2 剪一段含人的片段存 /tmp，短帧数跑通全链路；
VLM 是否触发取决于 `VLM_MIN_AREA` 与画面中人形面积（test.mp4 里人形占比 1-9%）。

## 已知坑

- **locate-anything 大图 OOM**：3B 检测模型处理 2160×1440 原图需 ~16GB 显存
  （11GB 卡必崩，容器反复重启）。客户端必须限制送图边长：脚本里
  `LOCATE_MAX_SIDE=1280`，框坐标已换算回原图。
- **locate 引擎有状态**：请求互斥串行，不要并发调用；响应**无置信度**（conf 置 1.0）。
- **VLM prompt 设计**（误报教训，详见 detect-pipeline/README.md）：必须正向提问
  （helmet/vest 0-4）+ 代码取反；必须带可见性守卫（头/躯干出画答 2）；解析兼容
  `<0>` 尖括号。改动 prompt 后务必用有人/无头两类图回归。
- 无头环境 `cv2.imshow/destroyAllWindows` 会报错——脚本已用 `SHOW_DISPLAY` 门控，勿移除。
- 音频文件缺失是**设计内的静默行为**（只画面渲染），不是 bug。
- llama-server 冷启动加载 Q8_0 + mmproj 约 20-60s；`start.sh` 的健康等待上限 300s。
- **勿移除 compose 里的 `-c 32768`（VL_CTX）**：`--fit on` 不配 `-c` 会把 KV 缓存
  撑到吃满整卡（24GB 卡实测权重 4.7GB 的模型占到 17GB）；当前 q4_0 KV + 32k 上下文
  下 VL 服务约 7GB。
- 构建上下文依赖 `.dockerignore` 保持在 ~36MB（排除 `.venv/`、`models/`、`test.mp4`、
  `outputs/`、`locate-anything.cpp/build/`）；新增大产物目录时同步两个 ignore 文件。

## 维护约定

- 大文件一律不进 git（已 gitignore）：`models/`、`.venv/`、`outputs/`、`weights/`、
  `test.mp4`、`locate-anything.cpp/build/`。新增产物目录时同步 `.gitignore` 与 `.dockerignore`。
- 修改 `locate-anything.cpp/` 源码后需重新构建镜像才影响容器服务。
- 改 Dockerfile/compose 时同步检查 `scripts/start.sh` 的兼容性（compose 插件缺失问题）。
- 提交前 `git status` 确认无大文件混入；作者身份沿用仓库已有 `zjd1988 <942002795@qq.com>`。
