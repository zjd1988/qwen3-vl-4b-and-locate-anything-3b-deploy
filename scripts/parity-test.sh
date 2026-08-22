#!/usr/bin/env bash
# ==============================================================================
# 一致性测试：locate-anything-cli（容器内）vs locate-anything-server（HTTP）
# 对 benchmarks/media/ 下的输入图逐一比对，同一引擎同一参数应产出逐位一致的检测框。
# 覆盖三种解码 mode：三张图默认 hybrid，另对 bus_in.png 追加 slow / fast。
#
# 前置: 服务已通过 scripts/start.sh 启动（locate-anything 容器挂载了 /media）。
# 用法: bash scripts/parity-test.sh [LOCATE_BASE_URL]
#       LA_DEVICE=cpu bash scripts/parity-test.sh   # 显存放不下第二份权重时 CLI 走 CPU
#                                                   # （CPU/GPU 数值差异，允许亚像素级漂移）
# ==============================================================================
set -euo pipefail

LOCATE_URL="${1:-http://127.0.0.1:8091}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
COMPOSE=(docker compose -f "${REPO_ROOT}/docker-compose.yml")
LA_MODEL="/models/locate-anything-3b/locate-anything-q8_0.gguf"
MEDIA_HOST="${MEDIA_DIR:-${SCRIPT_DIR}/../locate-anything.cpp/benchmarks/media}"

PROMPT='Locate all the instances that matches the following description: person</c>car</c>bus.'

echo "==> 健康检查: ${LOCATE_URL}/health"
curl -fsS "${LOCATE_URL}/health" >/dev/null && echo OK

run_case() {
  local image="$1" mode="$2"
  local cli_json srv_json
  echo "==> 用例: $(basename "${image}") mode=${mode}"

  # 基准：容器内 CLI（/media 由 compose 挂载，只读），默认走 GPU 以便与 server 逐位一致。
  # 若同卡显存放不下第二份权重（约 4.75GB），可用 LA_DEVICE=cpu 跑 CPU——
  # 此时 CPU/GPU 数值差异会导致个别坐标亚像素漂移，甚至影响 hybrid 的贪心解码路径。
  cli_json="$("${COMPOSE[@]}" exec -T -e "LA_DEVICE=${LA_DEVICE:-}" locate-anything \
      locate-anything-cli detect --model "${LA_MODEL}" \
      --input "/media/$(basename "${image}")" --prompt "${PROMPT}" --mode "${mode}" 2>/dev/null)"

  # 被测：HTTP server
  srv_json="$(curl -fsS "${LOCATE_URL}/v1/detect" \
      -F "image=@${image}" -F "prompt=${PROMPT}" -F "mode=${mode}")"

  CLI_JSON="${cli_json}" SRV_JSON="${srv_json}" python3 -c "
import json, os, sys
cli = json.loads(os.environ['CLI_JSON'])['detections']
srv = json.loads(os.environ['SRV_JSON'])['detections']
if cli != srv:
    print('MISMATCH'); print('cli:', cli); print('srv:', srv); sys.exit(1)
print('  %d detections 一致' % len(cli))
for det in srv:
    print('    %-10s [%.1f, %.1f, %.1f, %.1f]' % (det['label'], *det['box']))
"
}

FAIL=0
for img in bus_in.png coco_kitchen_in.png coco_skater_in.png; do
  [[ -f "${MEDIA_HOST}/${img}" ]] || { echo "!! 跳过（文件不存在）: ${img}"; continue; }
  run_case "${MEDIA_HOST}/${img}" hybrid || FAIL=1
done
for mode in slow fast; do
  run_case "${MEDIA_HOST}/bus_in.png" "${mode}" || FAIL=1
done

if [[ "${FAIL}" -ne 0 ]]; then echo "✘ 一致性测试失败"; exit 1; fi
echo
echo "✔ 一致性测试全部通过（CLI 与 server 输出逐位一致）"
