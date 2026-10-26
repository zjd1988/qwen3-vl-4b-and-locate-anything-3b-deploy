# 安全检测流水线：小模型检测 + 大模型判定

本目录演示**大小模型结合**的落地方式：小模型做实时/廉价的目标检测，把"有人、在哪、
框有多大"交给小模型；大模型（Qwen3-VL-4B）只做小模型做不了的语义级安全判定
（吸烟/安全帽/反光衣）。两者通过"检测框触发 + 限频 + 异步送检"结合，大模型的耗时
完全不阻塞小模型检测主循环。

提供两个小模型版本，流程完全一致，只换检测环节：

| 脚本 | 小模型 | 特点 |
|---|---|---|
| `yoloworld_vlm_safety.py` | ultralytics YOLO-World v8s（本地 PyTorch） | ~26MB，每帧 20-50ms，真正的实时；类别固定（person） |
| `locate_vlm_safety.py` | LocateAnything-3B（HTTP 服务，文本提示开放式检测） | 每帧 1-3s（远程调用）；换提示词即可检任意类别，如 `person</c>helmet</c>smoke` |

```
RTSP / 本地视频
      │
      ▼
┌───────────────────┐   人形框(面积过滤/聚组/裁剪)   ┌────────────────────┐
│ 小模型: YOLO-World │  ─────────────────────────▶ │ 大模型: Qwen3-VL-4B │
│ 或 LocateAnything │   限频5s + 后台线程异步       │ llama-server 服务化  │
│ 检测画面中所有 person│ ◀─────────────────────────  │ 0-4 档位细粒度判定    │
└───────────────────┘   判定结果(TTL内生效)        └────────────────────┘
      │                                                (本仓库 docker 部署)
      ▼
画面渲染告警(红框/文字) + 可选落盘 outputs/ + 可选音频
```

**为什么这样分工**

| | 小模型（检测） | 大模型（Qwen3-VL） |
|---|---|---|
| 职责 | "画面里有没有人在哪" | "这个人在不在吸烟/有没有戴帽穿衣" |
| 成本 | YOLO-World 每帧毫秒级；LocateAnything 每帧 1-3s | 每次 ~3s，只送裁剪小图 |
| 做不到 | 无法判断吸烟/PPE 等语义 | 不能实时跑全视频 |

若用大模型逐帧看全视频，11.8 分钟的片子要几小时；小模型先筛出"有人"的时刻并裁好
图，大模型只做 0-1 次/5 秒的细判定，全片几分钟跑完。

## 文件

- `yoloworld_vlm_safety.py` — YOLO-World 版（本地小模型，实时）
- `locate_vlm_safety.py` — LocateAnything 版（远程检测服务，开放式类别）
- `requirements.txt` — Python 依赖（Python 3.10，两个脚本通用）

两个脚本共享同一套流程代码（取流/抽帧/VLM 判定/告警/落盘/配置横幅），
只有检测环节和对应配置不同。

## 部署

### 1. 启动两个模型服务（本仓库 docker 部署）

```bash
cd <仓库根目录>
# 大模型 Qwen3-VL（构建见根 README，20-80 分钟）
bash scripts/start.sh --build --service qwen3-vl        # -> 127.0.0.1:8090
# 小模型 LocateAnything（镜像相同；无 compose 插件时等价 docker run）
docker run -d --name locate-anything-server --restart unless-stopped \
  --gpus device=1 \
  -v "$PWD/models:/models:ro" -p 8091:8081 \
  qwen3-vl-4b-locate-3b:latest \
  locate-anything-server --model /models/locate-anything-3b/locate-anything-q8_0.gguf \
    --host 0.0.0.0 --port 8081 --threads 4

curl http://127.0.0.1:8090/health && curl http://127.0.0.1:8091/health
```

建议 LocateAnything 与大模型分卡部署（如 VLM 用 GPU 0、Locate 用 GPU 1）；
单张 11GB 卡同时跑两者会 OOM。也可用任意 OpenAI 兼容的多模态服务替代 VLM
（设 `VLM_BASE_URL`），或跳过 LocateAnything 只用 YOLO-World 版（无需 8091）。

### 2. 准备宿主机 Python 环境

```bash
python3.10 -m venv .venv && source .venv/bin/activate
pip install -r detect-pipeline/requirements.txt
# 网络不佳时建议用镜像 + 超时重试：
# pip install -r detect-pipeline/requirements.txt \
#     -i https://mirrors.aliyun.com/pypi/simple/ --timeout 30 --retries 100
```

YOLO-World 权重 `yolov8s-worldv2.pt` 首次运行自动下载，也可手动放到 `models/`。

### 3. 运行

```bash
# YOLO-World 版（实时，本地视频）
SAVE_VIDEO=1 SAVE_JSONL=1 python detect-pipeline/yoloworld_vlm_safety.py

# LocateAnything 版（开放式检测，建议抽帧间隔调大）
LOCATE_BASE_URL=http://127.0.0.1:8091 FRAME_INTERVAL=10 \
  SAVE_VIDEO=1 SAVE_JSONL=1 python detect-pipeline/locate_vlm_safety.py

# RTSP 摄像头
RTSP_URL='rtsp://user:pass@host:554/...' VIDEO_FILE='' python detect-pipeline/yoloworld_vlm_safety.py
```

启动时打印 `[CFG]` 配置横幅（实际生效值全量输出，含两个小模型/音频文件存在性、
Locate 服务可达性），运行中每 30 秒一条 `[heartbeat]` 进度日志
（处理帧数/源帧位置/百分比/当前状态）。

## 配置（环境变量，均有默认值；未列出的见脚本内注释）

| 变量 | 默认 | 说明 |
|---|---|---|
| `VIDEO_FILE` | `test.mp4` | 本地视频路径；文件不存在则回退 RTSP |
| `RTSP_URL` | （见脚本） | RTSP 地址（日志中密码自动打码） |
| `LOOP_VIDEO` | `1` | 本地视频播完是否循环 |
| `FRAME_INTERVAL` | `1`（locate 版 `10`） | 抽帧间隔：每 N 个源帧处理 1 帧（位置严格确定） |
| `MAX_FRAMES` | `0` | >0 时只处理前 N 帧（调试） |
| `SAVE_VIDEO` / `SAVE_JSONL` | `0` / `0` | 落盘开关（默认关闭）：`outputs/annotated_*.mp4`、`outputs/detections_*.jsonl` |
| `OUTPUT_DIR` | `outputs` | 输出目录 |
| `VLM_ENABLED` | `true` | 大模型安全判定总开关 |
| `VLM_BASE_URL` | `http://127.0.0.1:8090/v1` | OpenAI 兼容接口 |
| `VLM_MIN_AREA` | `0.01` | 人形框面积占比低于此值不送检 |
| `VLM_INTERVAL` | `5.0` | 两次送检最小间隔（秒） |
| `VLM_DEBUG` | `1` | VLM 调试日志（触发/跳过/过滤原因） |
| `LOCATE_BASE_URL` | `http://127.0.0.1:8091` | LocateAnything 服务（仅 locate 版） |
| `LOCATE_PROMPT` | `...person.` | 检测提示词；`</c>` 分隔多类别 |
| `LOCATE_MODE` | `hybrid` | `hybrid`(默认)/`slow`/`fast` |
| `LOCATE_MAX_SIDE` | `1280` | 送检图最长边。**务必限制**：3B 检测模型处理 2160×1440 原图需 ~16GB 显存，直接 OOM；脚本已把框坐标换算回原图 |
| `SHOW_DISPLAY` | 代码内配置 | `True` 时开窗口（需 GUI 环境） |

代码内可调：`PERSON_AREA_THRES`（"有人"告警面积门槛）、`ALARM_LEVEL_THRES`
（告警档位，0-4 制，≥3 告警，敏感场景可调 4 只报"肯定有"）。

## 输出

- **画面**：实时画框 + 顶部状态文字 + 告警红色边框（SMOKING! / NO HELMET! / NO VEST!）
- **落盘**（可选）：`annotated_*.mp4`（标注视频，帧率按抽帧折算保持实时节奏）、
  `detections_*.jsonl`（逐帧：人形框/面积占比/告警文本/VLM 三档状态/src_frame 源帧号）
- **音频**：`audio_files/*.wav` 存在即播放（缺失时静默、只画面渲染，启动横幅会标明）；
  可用 edge-tts 生成语音：
  `edge-tts --voice zh-CN-YunxiaNeural --text "警告，发现有人吸烟" --write-media audio_files/smoking_alarm.mp3`

## 大模型判定 prompt 的设计要点（误报排查经验）

1. **正向提问、代码取反**：问"是否每个人都戴了安全帽"（helmet 0-4），再算
   `no_helmet = 4 - helmet`。否定式提问（"给未戴帽程度打分"）对 4B 小模型极易误报。
2. **可见性守卫**：头部/躯干不在画面内时要求回答 2（不确定）而不是 0——否则
   "工人紧贴摄像头、头在画外"会被结构性误判为未戴帽。
3. **格式兜底**：兼容 `<0>` 尖括号并要求纯数字输出；解析失败保留旧值不误报。
4. **聚合策略**：多组裁剪"任一违规即违规"；限频 + 后台线程，主循环零阻塞。

VLM 相关日志：`[VLM] 过滤`（面积不足）/ `跳过`（限频或占用）/ `触发` / `组i/N 尺寸 回答` /
`聚合 -> 吸烟=.. 未戴安全帽=.. 未穿反光衣=..`，排障时先对启动 `[CFG]` 横幅。

## 与主部署的关系

本流水线运行在**宿主机**：YOLO-World 版占显存 ~2-3GB（本地 torch + 视频读写），
LocateAnything 版不占本地显存（推理在容器里，~3GB）。docker 内是两个模型服务
（VLM ~7GB、Locate ~3GB），11GB 小卡建议一卡一个服务；24GB 卡可同卡。
LocateAnything 与 YOLO-World 是两种小模型路线：前者开放式（文本提示任意类别），
后者固定类别但实时，按场景选择或级联（YOLO 触发 → Locate 精检）。
