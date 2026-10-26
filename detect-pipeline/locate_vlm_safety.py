"""locate-anything-3B (开放式检测服务) 人形检测 + RTSP/本地视频取流 + Qwen3-VL 安全判定(吸烟/安全帽/反光衣) + 音频告警
检测结果可选落盘到 outputs/（标注视频 + 逐帧检测 jsonl，默认关闭，SAVE_VIDEO/SAVE_JSONL=1 开启）

与 yoloworld_vlm_safety.py 的区别：小模型从本地 ultralytics YOLO-World 换成
locate-anything-server 的 HTTP 检测接口（文本提示开放式检测，本脚本只提示 person）。
检测为远程调用（3B 模型，每帧约 1-3s），建议 FRAME_INTERVAL 适当调大。

流程:
  1. locate-anything 检测画面中所有 person（框为原图像素坐标）
  2. 任一 person 的框面积占比 >= PERSON_AREA_THRES 才触发"有人"告警
  3. 画面中出现 person 即裁剪送 Qwen3-VL 判定吸烟/安全帽/反光衣
     (限频 + 后台线程, 不阻塞检测)

注意: locate-anything 对超大图显存需求高(2160x1440 约需 16GB),
     本脚本送检前缩放到 LOCATE_MAX_SIDE(默认 1280), 框坐标已换算回原图。
"""

import os
import time
import json
import base64
import re
import threading
import subprocess

import cv2
import requests


# ==================== 配置 ====================
# ---- 视频源: 本地视频文件 或 RTSP（文件存在则优先用文件，便于离线测试） ----
VIDEO_FILE = os.environ.get("VIDEO_FILE", "test.mp4")
RTSP_URL = os.environ.get(
    "RTSP_URL",
    "rtsp://admin:glodon111@192.168.1.2:554/Streaming/Channels/101")
LOOP_VIDEO = os.environ.get("LOOP_VIDEO", "1").lower() in ("1", "true", "yes")  # 本地视频播完是否循环
MAX_FRAMES = int(os.environ.get("MAX_FRAMES", "0"))  # >0 时仅处理前 N 帧（调试用，按处理帧计）
FRAME_INTERVAL = max(1, int(os.environ.get("FRAME_INTERVAL", "10")))  # 抽帧间隔: 每 N 个源帧处理 1 帧(长视频建议 3~10)

# ---- 检测结果落盘（默认关闭; SAVE_VIDEO/SAVE_JSONL=1 开启） ----
OUTPUT_DIR = os.environ.get("OUTPUT_DIR", "outputs")  # 输出目录：标注视频 + 检测日志
SAVE_VIDEO = os.environ.get("SAVE_VIDEO", "0").lower() in ("1", "true", "yes")  # 标注视频（annotated_*.mp4）
SAVE_JSONL = os.environ.get("SAVE_JSONL", "0").lower() in ("1", "true", "yes")  # 逐帧检测日志（detections_*.jsonl）

# ---- 小模型: locate-anything-server（HTTP 检测接口） ----
LOCATE_ENABLED = os.environ.get("LOCATE_ENABLED", "1").lower() in ("1", "true", "yes")
LOCATE_BASE_URL = os.environ.get("LOCATE_BASE_URL", "http://127.0.0.1:8091")
LOCATE_PROMPT = os.environ.get(
    "LOCATE_PROMPT",
    "Locate all the instances that matches the following description: person.")
LOCATE_MODE = os.environ.get("LOCATE_MODE", "hybrid")     # hybrid(默认, PBD+AR回退)/slow(纯AR)/fast(仅MTP)
LOCATE_TIMEOUT = int(os.environ.get("LOCATE_TIMEOUT", "120"))
LOCATE_MAX_SIDE = int(os.environ.get("LOCATE_MAX_SIDE", "1280"))  # 送检图最长边(过大显存爆炸), 框坐标自动换算回原图

ALARM_FILE = "audio_files/alarm.wav"                    # 有人告警音频
SMOKING_ALARM_FILE = "audio_files/smoking_alarm.wav"    # 吸烟告警音频（没有则用 ALARM_FILE 兜底）
HELMET_ALARM_FILE = "audio_files/helmet_alarm.wav"      # 未戴安全帽告警音频（没有则用 ALARM_FILE 兜底）
VEST_ALARM_FILE = "audio_files/vest_alarm.wav"          # 未穿反光衣告警音频（没有则用 ALARM_FILE 兜底）
ALARM_COOLDOWN = 5.0                        # 音频最小间隔(秒)

# ---- 人形告警条件 ----
PERSON_AREA_THRES = 0.20   # 人体框面积占画面比例 >= 此值才告警

# ---- Qwen3-VL 服务（llama.cpp OpenAI 兼容接口）----
VLM_ENABLED = os.environ.get("VLM_ENABLED", "true").lower() not in ("0", "false", "no")
VLM_BASE_URL = os.environ.get("VLM_BASE_URL", "http://127.0.0.1:8090/v1")
VLM_API_KEY = "EMPTY"
VLM_MODEL = "Qwen3-VL-4B-Instruct"
VLM_INTERVAL = 5.0               # 两次送检的最小间隔(秒)
VLM_TIMEOUT = 20                 # VLM 单次请求超时(秒)
SMOKING_STATE_TTL = 10.0         # 判定结果的有效期(秒)
VLM_MIN_AREA = float(os.environ.get("VLM_MIN_AREA", "0.1"))  # 人体框占画面面积 < 此值不送检(太小看不清, 易误检)
VLM_GROUP_MARGIN = 0.35          # 框外扩比例; 外扩后相交的框合并为一组一起送检
VLM_MAX_GROUPS = 3               # 单次触发最多调用 VLM 的次数(按组面积从大到小取前N组)
VLM_MAX_SIDE = 1024              # 送检裁剪图最长边(像素), 超过则缩放, 控制 VLM 耗时
VLM_BATCH_MAX_SECS = 12          # 单次触发的 VLM 总耗时上限(秒), 超时则跳过剩余组
ALARM_LEVEL_THRES = 3            # 预警阈值: 档位 >= 此值才告警 (0=肯定无 1=大概无 2=不确定 3=大概有 4=肯定有)

# ---- 可视化 ----
#SHOW_DISPLAY = True   # False = 无头后台运行（不建窗口，Ctrl+C 退出，日志看终端/文件）
SHOW_DISPLAY = False


# ==================== 小模型: locate-anything 远程检测 ====================
def detect_persons(frame_bgr):
    """调 locate-anything-server /v1/detect 检测 person,
    返回 [(conf, (x1,y1,x2,y2)), ...] 原图坐标(按面积降序); 请求失败返回 []
    注: 服务端响应无置信度, conf 固定 1.0"""
    if not LOCATE_ENABLED:
        return []
    h, w = frame_bgr.shape[:2]
    scale = 1.0
    send = frame_bgr
    if max(h, w) > LOCATE_MAX_SIDE:
        scale = LOCATE_MAX_SIDE / max(h, w)
        send = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", send, [cv2.IMWRITE_JPEG_QUALITY, 85])
    try:
        resp = requests.post(
            f"{LOCATE_BASE_URL.rstrip('/')}/v1/detect",
            files={"image": ("frame.jpg", buf.tobytes(), "image/jpeg")},
            data={"prompt": LOCATE_PROMPT, "mode": LOCATE_MODE},
            timeout=LOCATE_TIMEOUT)
        resp.raise_for_status()
        dets = resp.json().get("detections", [])
    except Exception as e:
        print(f"[WARN] locate-anything 请求失败: {str(e)[:120]}", flush=True)
        return []
    out = []
    for d in dets:
        if d.get("label") != "person":
            continue
        x1, y1, x2, y2 = (float(v) for v in d["box"])
        if scale != 1.0:
            x1, x2 = x1 / scale, x2 / scale
            y1, y2 = y1 / scale, y2 / scale
        out.append((1.0, (x1, y1, x2, y2)))
    out.sort(key=lambda t: -((t[1][2] - t[1][0]) * (t[1][3] - t[1][1])))
    return out


def locate_health():
    try:
        return requests.get(f"{LOCATE_BASE_URL.rstrip('/')}/health",
                            timeout=3).ok
    except Exception:
        return False


# ==================== 告警音频 ====================
_last_alarm = 0.0
_player_warned = False
_ppe_toggle = {"v": False}   # 安全帽/反光衣同时违规时轮流播报用

def play_alarm(file=ALARM_FILE):
    global _last_alarm, _player_warned
    path = file if os.path.exists(file) else ALARM_FILE
    if not os.path.exists(path):
        return                    # 无音频文件: 仅画面渲染告警, 静默跳过
    now = time.time()
    if now - _last_alarm < ALARM_COOLDOWN:
        return
    _last_alarm = now
    for player in ("aplay", "paplay"):
        try:
            subprocess.Popen([player, "-q", path],
                             stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL)
            return
        except FileNotFoundError:
            continue
    if not _player_warned:
        _player_warned = True
        print("[WARN] 找不到 aplay/paplay，请安装 alsa-utils")


# ==================== Qwen3-VL 安全判定(吸烟/安全帽/反光衣) ====================
safety_state = {
    "smoking": 0,          # 有人吸烟 (0~4 档)
    "no_helmet": 0,        # 有人未戴安全帽 (0~4 档)
    "no_vest": 0,          # 有人未穿反光衣 (0~4 档)
    "ts": 0.0, "busy": False, "last_check": 0.0,
}

# ---- VLM 调试日志（2s 节流, VLM_DEBUG=0 关闭） ----
DEBUG_VLM = os.environ.get("VLM_DEBUG", "1").lower() in ("1", "true", "yes")
_last_dbg = 0.0

def _dbg(msg):
    global _last_dbg
    if not DEBUG_VLM:
        return
    now = time.time()
    if now - _last_dbg < 2.0:
        return
    _last_dbg = now
    print(msg, flush=True)

VLM_PROMPT = (
    "You are a construction-site safety inspector. The image is a crop from a "
    "surveillance camera (it may contain screen UI; judge only the real workers).\n"
    "Rate each item on a scale of 0-4:\n"
    "0 = definitely no, 1 = probably no, 2 = unsure, 3 = probably yes, 4 = definitely yes.\n"
    "Items:\n"
    "1. helmet: EVERY worker's head is protected by a helmet or hard hat "
    "(red, yellow, white or blue hard hats all count as helmets).\n"
    "2. vest: EVERY worker's torso wears a reflective vest or reflective-striped workwear "
    "(yellow/orange fluorescent vest, or a work jacket with silver reflective stripes; "
    "ordinary clothes without reflective stripes do NOT count).\n"
    "3. smoking: ANY worker is holding a lit cigarette at the mouth "
    "(unlit packs, pens, fingers, tools do NOT count).\n"
    "Only judge body parts that are visible: if no worker's head is visible in the "
    "image, answer helmet:2; if no worker's torso is visible, answer vest:2.\n"
    "Reply with EXACTLY one line in this format (plain numbers only, no angle brackets):\n"
    "helmet:<0-4> vest:<0-4> smoking:<0-4>"
)


def _parse_level(answer, key):
    """解析某项档位 0~4; 解析失败返回 None（保留旧状态, 不误报）"""
    m = re.search(key + r"\s*[:：]\s*<?\s*([0-4])", answer, re.IGNORECASE)
    return None if m is None else int(m.group(1))


def _query_vlm(crop_bgr, remaining=VLM_TIMEOUT):
    """单张裁剪图送检，返回(原文, 解析dict); timeout 不超过 remaining"""
    h, w = crop_bgr.shape[:2]
    scale = VLM_MAX_SIDE / max(h, w)
    if scale < 1.0:
        crop_bgr = cv2.resize(crop_bgr, (int(w * scale), int(h * scale)))
    ok, buf = cv2.imencode(".jpg", crop_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
    b64 = base64.b64encode(buf.tobytes()).decode()
    resp = requests.post(
        f"{VLM_BASE_URL.rstrip('/')}/chat/completions",
        headers={"Authorization": f"Bearer {VLM_API_KEY}"},
        json={
            "model": VLM_MODEL,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "image_url",
                     "image_url": {"url": f"data:image/jpeg;base64,{b64}"}},
                    {"type": "text", "text": VLM_PROMPT},
                ],
            }],
            "temperature": 0.0,
            "max_tokens": 30,
        },
        timeout=min(VLM_TIMEOUT, max(2, int(remaining))),
    )
    resp.raise_for_status()
    answer = resp.json()["choices"][0]["message"]["content"].strip()
    helmet = _parse_level(answer, "helmet")   # 正向提问, 代码取反
    vest = _parse_level(answer, "vest")
    return answer, {
        "smoking": _parse_level(answer, "smoking"),
        "no_helmet": None if helmet is None else 4 - helmet,
        "no_vest": None if vest is None else 4 - vest,
    }


def vlm_worker(crops):
    """后台线程：逐组送检(串行, 受总耗时上限约束)，多组结果按
    "任一违规即违规, 全部否才否"聚合"""
    merged = {"smoking": None, "no_helmet": None, "no_vest": None}
    deadline = time.time() + VLM_BATCH_MAX_SECS
    try:
        for i, crop in enumerate(crops):
            remaining = deadline - time.time()
            if remaining <= 0:
                print(f"[VLM] 已达本批最长耗时 {VLM_BATCH_MAX_SECS}s, "
                      f"剩余 {len(crops) - i} 组跳过", flush=True)
                break
            answer, flags = _query_vlm(crop, remaining)
            print(f"[VLM] 组{i+1}/{len(crops)} 尺寸{crop.shape[1]}x{crop.shape[0]} "
                  f"剩余{remaining:.1f}s 回答: {answer!r}", flush=True)
            for k, v in flags.items():
                if v is None:
                    continue                       # 该项解析失败, 不覆盖
                merged[k] = v if merged[k] is None else max(merged[k], v)
        final = {k: v for k, v in merged.items() if v is not None}
        if final:
            safety_state.update(final)
            safety_state["ts"] = time.time()
        print(f"[VLM] 聚合 -> 吸烟={safety_state['smoking']} "
              f"未戴安全帽={safety_state['no_helmet']} "
              f"未穿反光衣={safety_state['no_vest']}"
              + ("" if len(final) == 3 else "  [部分项解析失败, 保留旧值]"), flush=True)
    except Exception as e:
        print(f"[VLM] 请求失败: {e}")
    finally:
        safety_state["busy"] = False


def _group_boxes(boxes, margin, frame_w, frame_h):
    """外扩 margin 后相交的框合并为一组(迭代至稳定), 返回 [(面积, 合并框)]
    外扩结果收敛在图像边界内"""
    groups = []   # 元素: [x1,y1,x2,y2]
    for x1, y1, x2, y2 in boxes:
        eb = [max(0.0, x1 - (x2 - x1) * margin), max(0.0, y1 - (y2 - y1) * margin),
              min(float(frame_w), x2 + (x2 - x1) * margin),
              min(float(frame_h), y2 + (y2 - y1) * margin)]
        groups.append(eb)
    changed = True
    while changed:
        changed = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = groups[i], groups[j]
                inter = not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])
                if inter:
                    groups[i] = [min(a[0], b[0]), min(a[1], b[1]),
                                 max(a[2], b[2]), max(a[3], b[3])]
                    del groups[j]
                    changed = True
                    break
            if changed:
                break
    out = []
    for g in groups:
        out.append(((g[2] - g[0]) * (g[3] - g[1]), g))
    out.sort(key=lambda t: -t[0])
    return out


def maybe_check_safety(frame, person_dets, frame_area):
    """过滤小框 -> 邻近框聚类 -> 按组裁剪送检（限频 + 防重入），不影响主循环"""
    if not VLM_ENABLED or not person_dets:
        return
    now = time.time()
    if safety_state["busy"] or (now - safety_state["last_check"] < VLM_INTERVAL):
        _dbg(f"[VLM] 跳过: busy={safety_state['busy']} "
             f"距上次{now - safety_state['last_check']:.1f}s < 间隔{VLM_INTERVAL}s")
        return

    # 1. 过滤太小的框(看不清, 易误检)
    boxes = []
    for _, (x1, y1, x2, y2) in person_dets:
        if max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / frame_area >= VLM_MIN_AREA:
            boxes.append((x1, y1, x2, y2))
    if not boxes:
        _dbg(f"[VLM] 过滤: {len(person_dets)} 个人形框面积均 < {VLM_MIN_AREA}, 不送检")
        return

    # 2. 邻近框合并成组, 每组裁一张图
    fh, fw = frame.shape[:2]
    crops = []
    for _, (gx1, gy1, gx2, gy2) in _group_boxes(boxes, VLM_GROUP_MARGIN, fw, fh)[:VLM_MAX_GROUPS]:
        gx1 = int(max(0, gx1)); gy1 = int(max(0, gy1))
        gx2 = int(min(fw, gx2)); gy2 = int(min(fh, gy2))
        if gx2 > gx1 and gy2 > gy1:
            crops.append(frame[gy1:gy2, gx1:gx2])
    if not crops:
        return

    print(f"[VLM] 触发: {len(person_dets)}框 -> 过滤后{len(boxes)}框 -> 合并{len(crops)}组送检", flush=True)
    safety_state["busy"] = True
    safety_state["last_check"] = now
    threading.Thread(target=vlm_worker, args=(crops,), daemon=True).start()


# ==================== 视频源取流（RTSP 后台线程 / 本地文件按需顺序读） ====================
class VideoReader:
    """
    - 本地文件: 主线程按需顺序解码读取(解码不抢跑, 抽帧位置严格确定),
      播完后按 LOOP_VIDEO 循环或置 eof 结束
    - RTSP: 后台线程持续取流, 始终读最新帧(避免缓冲堆积延迟), 断流自动重连
    """
    def __init__(self, url, is_file=False):
        self.url = url
        self.is_file = is_file
        self.cap = None
        self.frame = None
        self.lock = threading.Lock()
        self.running = True
        self.eof = False          # 本地视频且不循环时, 播完置 True
        self.frames_read = 0      # 已读取的源帧计数(含被抽帧跳过的)
        self.frames_total = 0     # 本地视频总帧数(取流为 0)
        self._open()
        if not is_file:
            self.thread = threading.Thread(target=self._grab_rtsp, daemon=True)
            self.thread.start()

    def _open(self):
        self.cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if self.is_file:
            self.frames_total = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    @property
    def fps(self):
        if self.cap is None:
            return 0.0
        return float(self.cap.get(cv2.CAP_PROP_FPS) or 0.0)

    def _grab_rtsp(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                print("[WARN] RTSP 读取失败，3 秒后重连...")
                self.cap.release()
                time.sleep(3)
                self._open()
                continue
            with self.lock:
                self.frame = frame
                self.frames_read += 1

    def read(self):
        """本地文件: 顺序读下一帧(仅主线程调用); RTSP: 取后台线程最新帧"""
        if self.is_file:
            ret, frame = self.cap.read()
            if not ret:
                if LOOP_VIDEO:
                    self._open()                # 循环: 重头开始
                    ret, frame = self.cap.read()
                    if not ret:
                        return False, None
                else:
                    self.eof = True             # 不循环: 标记播完
                    return False, None
            self.frames_read += 1
            return True, frame
        with self.lock:
            if self.eof:
                return False, None
            return self.frame is not None, self.frame

    def release(self):
        self.running = False
        if self.cap:
            self.cap.release()


# ==================== 主循环 ====================
def _open_outputs(orig_w, orig_h, fps):
    """创建输出目录/标注视频/检测日志，返回 (writer, log_f)"""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    writer = None
    if SAVE_VIDEO:
        out_path = os.path.join(OUTPUT_DIR, f"annotated_{stamp}.mp4")
        eff_fps = fps / FRAME_INTERVAL if fps > 0 else 0   # 抽帧后按实时节奏写
        out_fps = eff_fps if 0 < eff_fps <= 60 else 25.0 / FRAME_INTERVAL
        writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                                 out_fps, (orig_w, orig_h))
        if writer.isOpened():
            print(f"[INFO] 标注视频 -> {out_path} ({orig_w}x{orig_h}@{out_fps:g}fps)")
        else:
            print(f"[WARN] 无法创建标注视频: {out_path}")
            writer = None
    log_f = None
    if SAVE_JSONL:
        log_path = os.path.join(OUTPUT_DIR, f"detections_{stamp}.jsonl")
        log_f = open(log_path, "w", encoding="utf-8")
        print(f"[INFO] 检测日志 -> {log_path}")
    return writer, log_f


def _print_config(source, is_file):
    """启动时打印全部有效配置，便于定位分析"""
    def has(path):
        return "有" if os.path.exists(path) else "无(静默跳过)"
    url = re.sub(r"(://[^:/]+:)[^@]+@", r"\1****@", source) if not is_file else source
    print("=" * 64, flush=True)
    print(f"[CFG] 视频源: {'本地文件' if is_file else 'RTSP'} -> {url}", flush=True)
    print(f"[CFG]   LOOP_VIDEO={LOOP_VIDEO} FRAME_INTERVAL={FRAME_INTERVAL} "
          f"MAX_FRAMES={MAX_FRAMES or '不限'}", flush=True)
    print(f"[CFG] 落盘: OUTPUT_DIR={OUTPUT_DIR} SAVE_VIDEO={SAVE_VIDEO} "
          f"SAVE_JSONL={SAVE_JSONL}", flush=True)
    print(f"[CFG] 检测(locate-anything): {'可达' if locate_health() else '不可达!'} "
          f"{LOCATE_BASE_URL} mode={LOCATE_MODE} "
          f"max_side={LOCATE_MAX_SIDE} timeout={LOCATE_TIMEOUT}s", flush=True)
    print(f"[CFG]   LOCATE_PROMPT={LOCATE_PROMPT!r}", flush=True)
    print(f"[CFG]   PERSON_AREA_THRES={PERSON_AREA_THRES}", flush=True)
    print(f"[CFG] 告警音频: 有人={has(ALARM_FILE)} 吸烟={has(SMOKING_ALARM_FILE)} "
          f"安全帽={has(HELMET_ALARM_FILE)} 反光衣={has(VEST_ALARM_FILE)} "
          f"COOLDOWN={ALARM_COOLDOWN}s", flush=True)
    print(f"[CFG] VLM: enabled={VLM_ENABLED} url={VLM_BASE_URL} model={VLM_MODEL}", flush=True)
    print(f"[CFG]   min_area={VLM_MIN_AREA} interval={VLM_INTERVAL}s "
          f"timeout={VLM_TIMEOUT}s TTL={SMOKING_STATE_TTL}s "
          f"告警档位>={ALARM_LEVEL_THRES}", flush=True)
    print(f"[CFG]   聚合: group_margin={VLM_GROUP_MARGIN} max_groups={VLM_MAX_GROUPS} "
          f"max_side={VLM_MAX_SIDE} batch_max={VLM_BATCH_MAX_SECS}s", flush=True)
    print(f"[CFG] 显示: SHOW_DISPLAY={SHOW_DISPLAY} (False=无头)", flush=True)
    print("=" * 64, flush=True)


def main():
    if VIDEO_FILE and os.path.exists(VIDEO_FILE):
        source, is_file = VIDEO_FILE, True
    else:
        source, is_file = RTSP_URL, False
    _print_config(source, is_file)
    reader = VideoReader(source, is_file=is_file)
    time.sleep(1)   # 等第一帧到达
    try:
        _main_loop(reader)
    finally:
        reader.release()


def _main_loop(reader):
    writer, log_f = None, None
    outputs_ready = False
    frame_idx = 0
    last_src = 0           # 上次已处理的源帧号(防止重复处理同一帧)
    next_at = 1            # 下一个抽帧目标(源帧号)
    try:
        while True:
            ret, frame = reader.read()
            if not ret:
                if reader.eof:
                    print("[INFO] 视频播放完毕")
                    break
                time.sleep(0.01)
                continue

            src_frame = reader.frames_read
            if reader.is_file:
                if (src_frame - 1) % FRAME_INTERVAL != 0:
                    continue            # 抽帧: 顺序读到目标帧再处理(位置严格确定)
            else:
                if src_frame < next_at or src_frame == last_src:
                    time.sleep(0.002)   # 等待到达抽帧目标 / 等待新帧到来
                    continue
                last_src = src_frame
                next_at = src_frame + FRAME_INTERVAL

            if not outputs_ready:
                oh, ow = frame.shape[:2]
                writer, log_f = _open_outputs(ow, oh, reader.fps)
                outputs_ready = True
            frame_idx += 1

            orig_h, orig_w = frame.shape[:2]
            frame_area = orig_h * orig_w

            person_dets = detect_persons(frame)   # 小模型: locate-anything 远程检测

            annotated = frame.copy()              # 自行画框(蓝色)
            for _, (x1, y1, x2, y2) in person_dets:
                cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 2)
                cv2.putText(annotated, "person", (int(x1), max(20, int(y1) - 5)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            # 计算最大人体框面积占比
            max_area_ratio = 0.0
            for _, (x1, y1, x2, y2) in person_dets:
                area_ratio = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1)) / frame_area
                max_area_ratio = max(max_area_ratio, area_ratio)

            # ---- VLM 安全判定：过滤小框 + 邻近合并成组 + 按组裁剪送检（后台）----
            if person_dets:
                maybe_check_safety(frame, person_dets, frame_area)

            state_fresh = (time.time() - safety_state["ts"]) < SMOKING_STATE_TTL
            smoking_active = state_fresh and safety_state["smoking"] >= ALARM_LEVEL_THRES
            helmet_active = state_fresh and safety_state["no_helmet"] >= ALARM_LEVEL_THRES
            vest_active = state_fresh and safety_state["no_vest"] >= ALARM_LEVEL_THRES

            if smoking_active:
                play_alarm(SMOKING_ALARM_FILE)
                text = "SMOKING!"
                color = (0, 0, 255)
                cv2.rectangle(annotated, (0, 0), (orig_w - 1, orig_h - 1), (0, 0, 255), 6)
            elif helmet_active or vest_active:
                bad = []
                if helmet_active: bad.append("NO HELMET")
                if vest_active:   bad.append("NO VEST")
                text = " ".join(bad) + "!"
                color = (0, 165, 255)
                cv2.rectangle(annotated, (0, 0), (orig_w - 1, orig_h - 1), (0, 165, 255), 6)
                # 选择对应告警音; 两项同时违规时轮流播放(音频冷却 5s 决定轮换节奏)
                if helmet_active and vest_active:
                    _ppe_toggle["v"] = not _ppe_toggle["v"]
                    ppe_file = HELMET_ALARM_FILE if _ppe_toggle["v"] else VEST_ALARM_FILE
                else:
                    ppe_file = HELMET_ALARM_FILE if helmet_active else VEST_ALARM_FILE
                play_alarm(ppe_file)
            elif person_dets and max_area_ratio >= PERSON_AREA_THRES:
                play_alarm()
                text = f"PERSON ({max_area_ratio*100:.0f}%)"
                color = (0, 165, 255)
            elif person_dets:
                text = f"person {max_area_ratio*100:.0f}% (too far)"
                color = (200, 200, 200)
            else:
                text = "NO PERSON"
                color = (0, 255, 0)

            cv2.putText(annotated, text, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)

            # ---- 结果落盘: 标注帧视频 + 逐帧检测记录 ----
            if writer is not None:
                writer.write(annotated)
            if log_f is not None:
                rec = {
                    "frame": frame_idx,
                    "src_frame": src_frame,
                    "persons": [
                        {"conf": round(float(c), 4),
                         "box": [round(float(v), 1) for v in box]}
                        for c, box in person_dets
                    ],
                    "max_area_ratio": round(max_area_ratio, 4),
                    "text": text,
                }
                if VLM_ENABLED:
                    rec["vlm"] = {k: safety_state[k]
                                  for k in ("smoking", "no_helmet", "no_vest")}
                log_f.write(json.dumps(rec, ensure_ascii=False) + "\n")

            if MAX_FRAMES and frame_idx >= MAX_FRAMES:
                print(f"[INFO] 达到 MAX_FRAMES={MAX_FRAMES}, 结束")
                break

            if SHOW_DISPLAY:
                cv2.imshow("LocateAnything Person/Smoking Detector", annotated)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q") or key == 27:
                    break
            else:
                # 无头模式: 慢速心跳日志, 便于确认进程活着
                now = time.time()
                if now - getattr(main, "_last_beat", 0) > 30:
                    main._last_beat = now
                    total = getattr(reader, "frames_total", 0)
                    prog = (f" src {src_frame}/{total} ({src_frame*100//total}%)"
                            if total else f" src {src_frame}")
                    print(f"[heartbeat] {time.strftime('%H:%M:%S')} | {text} | "
                          f"proc {frame_idx}{prog}", flush=True)
    finally:
        if writer is not None:
            writer.release()
        if log_f is not None:
            log_f.close()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[INFO] 收到 Ctrl+C, 退出")
    finally:
        if SHOW_DISPLAY:
            cv2.destroyAllWindows()
