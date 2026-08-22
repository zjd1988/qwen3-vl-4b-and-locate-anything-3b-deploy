#!/usr/bin/env bash
# ==============================================================================
# 冒烟测试：Qwen3-VL-4B（带图问答）+ LocateAnything-3B（目标检测）
#
# 用法:
#   bash scripts/smoke-test.sh [VL_BASE_URL] [LOCATE_BASE_URL] [服务过滤...]
# 默认: http://127.0.0.1:8090  http://127.0.0.1:8091  全部服务
# 服务过滤: qwen3-vl（别名 vl）/ locate-anything（别名 locate），可传多个
# ==============================================================================
set -euo pipefail

VL_URL="${1:-http://127.0.0.1:8090}"
LOCATE_URL="${2:-http://127.0.0.1:8091}"
shift $(( $# >= 1 ? 1 : 0 )) 2>/dev/null || true
shift $(( $# >= 1 ? 1 : 0 )) 2>/dev/null || true
FILTERS=("$@")

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MEDIA_DIR="${MEDIA_DIR:-${SCRIPT_DIR}/../locate-anything.cpp/benchmarks/media}"
TEST_IMAGE="${TEST_IMAGE:-${MEDIA_DIR}/bus_in.png}"
VL_MODEL="${VL_MODEL:-Qwen3-VL-4B-Instruct}"
DETECT_PROMPT='Locate all the instances that matches the following description: person</c>car</c>bus.'

# 无过滤参数 = 全测；支持别名 vl / locate
want() {
  [[ ${#FILTERS[@]} -eq 0 ]] && return 0
  local f
  for f in "${FILTERS[@]}"; do
    case "$f" in
      qwen3-vl|vl)             [[ "$1" == qwen3-vl ]] && return 0 ;;
      locate-anything|locate)  [[ "$1" == locate-anything ]] && return 0 ;;
      all)                     return 0 ;;
    esac
  done
  return 1
}

[[ -f "${TEST_IMAGE}" ]] || { echo "!!! 测试图片不存在: ${TEST_IMAGE}"; exit 1; }

# ---------- LocateAnything-3B：detect ----------
if want locate-anything; then
  echo "==> [locate-anything] 健康检查: ${LOCATE_URL}/health"
  curl -fsS "${LOCATE_URL}/health" && echo

  echo "==> [locate-anything] 检测请求: POST ${LOCATE_URL}/v1/detect (image=$(basename "${TEST_IMAGE}"))"
  curl -fsS "${LOCATE_URL}/v1/detect" \
    -F "image=@${TEST_IMAGE}" \
    -F "prompt=${DETECT_PROMPT}" \
    -F "mode=hybrid" | python3 -c "
import sys, json
d = json.load(sys.stdin)
dets = d.get('detections')
assert dets is not None, '响应缺少 detections 字段: %s' % d
w, h = d['image']['width'], d['image']['height']
print('image: %dx%d, elapsed: %sms' % (w, h, d.get('elapsed_ms')))
print('detections: %d' % len(dets))
for det in dets:
    x1, y1, x2, y2 = det['box']
    assert 0 <= x1 <= w and 0 <= x2 <= w and 0 <= y1 <= h and 0 <= y2 <= h, 'box 越界: %s' % det
    print('  %-10s [%.1f, %.1f, %.1f, %.1f]' % (det['label'], x1, y1, x2, y2))
assert len(dets) > 0, '未检测到任何目标（bus 图上应有 person/car/bus）'
print('OK')
"
fi

# ---------- Qwen3-VL-4B：带图问答 ----------
if want qwen3-vl; then
  echo
  echo "==> [qwen3-vl] 健康检查: ${VL_URL}/health"
  curl -fsS "${VL_URL}/health" && echo

  echo "==> [qwen3-vl] 带图问答: POST ${VL_URL}/v1/chat/completions (model=${VL_MODEL})"
  IMG_B64="$(base64 -w0 "${TEST_IMAGE}")"
  printf '%s' "${IMG_B64}" | python3 -c "
import sys, json, urllib.request
b64 = sys.stdin.read()
body = json.dumps({
    'model': '${VL_MODEL}',
    'messages': [{'role': 'user', 'content': [
        {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + b64}},
        {'type': 'text', 'text': '用一句话描述这张图片里有什么。'},
    ]}],
    'max_tokens': 128,
    'temperature': 0,
}).encode()
req = urllib.request.Request('${VL_URL}/v1/chat/completions', data=body,
                             headers={'Content-Type': 'application/json'})
d = json.load(urllib.request.urlopen(req, timeout=300))
content = d['choices'][0]['message'].get('content') or ''
assert content.strip(), '回复为空: %s' % d
print('content:', content)
print('OK')
"
fi

echo
echo "✔ 冒烟测试全部通过"
