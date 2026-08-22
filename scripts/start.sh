#!/usr/bin/env bash
# ==============================================================================
# Qwen3-VL-4B-Instruct + LocateAnything-3B — 一键启动脚本（单镜像双服务）
#
# 用法（在仓库任意目录执行均可）:
#   bash scripts/start.sh                            # 启动全部服务（无镜像则自动构建）
#   bash scripts/start.sh --service qwen3-vl         # 只启动 VL 服务（别名 vl）
#   bash scripts/start.sh --service locate-anything  # 只启动检测服务（别名 locate）
#   bash scripts/start.sh --build                    # 强制重新构建镜像后启动
#   bash scripts/start.sh --gpu 1                    # 使用 GPU 1（默认 0）
#   bash scripts/start.sh --smoke                    # 启动完成后自动跑对应服务的冒烟测试
#
# 环境变量覆盖（与命令行等价）:
#   SERVICE=vl GPU_ID=1 VL_PORT=8090 LOCATE_PORT=8091 MODELS_DIR=/path/models CUDA_ARCHS=86 bash scripts/start.sh
# ==============================================================================
set -euo pipefail

# ---- 定位部署仓库根目录（脚本可放在 scripts/ 下任意位置运行） ----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="qwen3-vl-4b-locate-3b:latest"
CUDA_ARCHS="${CUDA_ARCHS:-70;75;80;86;89;90}"   # 默认多架构；单卡环境可覆盖如 86 以加速构建

GPU_ID="${GPU_ID:-0}"
VL_PORT="${VL_PORT:-8090}"
LOCATE_PORT="${LOCATE_PORT:-8091}"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"
SERVICE="${SERVICE:-all}"
DO_BUILD=0
DO_SMOKE=0

# ---- 解析命令行参数 ----
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build|--rebuild)  DO_BUILD=1 ;;
    --gpu)   GPU_ID="$2"; shift ;;
    --service) SERVICE="$2"; shift ;;
    --smoke) DO_SMOKE=1 ;;
    *) echo "未知参数: $1（支持 --build / --gpu N / --service all|qwen3-vl|locate-anything / --smoke）"; exit 1 ;;
  esac
  shift
done

# ---- 服务选择（all / qwen3-vl(别名 vl) / locate-anything(别名 locate)） ----
case "${SERVICE}" in
  all)             SERVICES=(qwen3-vl locate-anything) ;;
  qwen3-vl|vl)     SERVICES=(qwen3-vl) ;;
  locate-anything|locate) SERVICES=(locate-anything) ;;
  *) echo "未知服务: ${SERVICE}（支持 all / qwen3-vl / locate-anything）"; exit 1 ;;
esac
has_service() { local s; for s in "${SERVICES[@]}"; do [[ "$s" == "$1" ]] && return 0; done; return 1; }

# ---- 0. 检查模型文件（仅检查待启动服务需要的） ----
VL_MODEL="${MODELS_DIR}/qwen3-vl-4b/Qwen3-VL-4B-Instruct-Q8_0.gguf"
VL_MMPROJ="${MODELS_DIR}/qwen3-vl-4b/mmproj-F16.gguf"
LA_MODEL="${MODELS_DIR}/locate-anything-3b/locate-anything-q8_0.gguf"
MISSING=0
has_service qwen3-vl && for f in "${VL_MODEL}" "${VL_MMPROJ}"; do
  [[ ! -f "$f" ]] && { echo "!!! 缺少模型文件: $f"; MISSING=1; }
done
has_service locate-anything && [[ ! -f "${LA_MODEL}" ]] && { echo "!!! 缺少模型文件: ${LA_MODEL}"; MISSING=1; }
if [[ "$MISSING" -eq 1 ]]; then
  echo "    请先下载模型: bash ${SCRIPT_DIR}/download-models.sh"
  exit 1
fi

# ---- 1. 构建镜像（--build 强制，或镜像缺失时自动构建） ----
if [[ "$DO_BUILD" -eq 1 ]] || ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  echo "==> 构建镜像 ${IMAGE}（CUDA 编译约 20–80 分钟，取决于 CUDA_ARCHS=${CUDA_ARCHS}）..."
  docker build -f "${REPO_ROOT}/Dockerfile" \
    --build-arg CUDA_ARCHS="${CUDA_ARCHS}" \
    -t "${IMAGE}" "${REPO_ROOT}"
else
  echo "==> 使用已有镜像 ${IMAGE}"
fi

# ---- 2. 启动服务 ----
echo "==> 启动服务 ${SERVICES[*]} (GPU ${GPU_ID}, VL 端口 ${VL_PORT}, locate 端口 ${LOCATE_PORT}, models=${MODELS_DIR})..."
cd "${REPO_ROOT}"
GPU_ID="${GPU_ID}" VL_PORT="${VL_PORT}" LOCATE_PORT="${LOCATE_PORT}" \
  MODELS_DIR="${MODELS_DIR}" CUDA_ARCHS="${CUDA_ARCHS}" docker compose up -d "${SERVICES[@]}"

# ---- 3. 等待服务健康 ----
wait_health() {
  local name="$1" url="$2" i
  echo -n "==> 等待 ${name} 就绪 ${url}/health "
  for i in $(seq 1 100); do
    if curl -fsS "${url}/health" >/dev/null 2>&1; then echo "OK（${i}×3s）"; return 0; fi
    echo -n "."; sleep 3
  done
  echo
  echo "!!! ${name} 在 300s 内未就绪，请查看日志："
  echo "    docker compose -f ${REPO_ROOT}/docker-compose.yml logs -f ${name}"
  return 1
}
has_service qwen3-vl        && wait_health "qwen3-vl"        "http://127.0.0.1:${VL_PORT}"
has_service locate-anything && wait_health "locate-anything" "http://127.0.0.1:${LOCATE_PORT}"

# ---- 4. 状态汇总 ----
echo
has_service qwen3-vl        && echo "==> qwen3-vl 健康:        $(curl -fsS "http://127.0.0.1:${VL_PORT}/health")"
has_service locate-anything && echo "==> locate-anything 健康: $(curl -fsS "http://127.0.0.1:${LOCATE_PORT}/health")"
echo "==> GPU ${GPU_ID} 状态:"
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv -i "${GPU_ID}"

# ---- 5. 可选冒烟测试（只测本次启动的服务） ----
if [[ "$DO_SMOKE" -eq 1 ]]; then
  echo
  echo "==> 冒烟测试:"
  bash "${SCRIPT_DIR}/smoke-test.sh" "http://127.0.0.1:${VL_PORT}" "http://127.0.0.1:${LOCATE_PORT}" "${SERVICES[@]}"
fi

echo
echo "✔ 启动完成（${SERVICES[*]}）."
has_service qwen3-vl        && echo "  Qwen3-VL-4B (OpenAI 兼容): http://127.0.0.1:${VL_PORT}/v1"
has_service locate-anything && echo "  LocateAnything-3B (检测):  http://127.0.0.1:${LOCATE_PORT}/v1/detect"
echo "  冒烟: bash scripts/smoke-test.sh"
echo "  一致性: bash scripts/parity-test.sh"
