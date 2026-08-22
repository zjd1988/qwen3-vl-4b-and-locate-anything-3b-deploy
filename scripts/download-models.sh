#!/usr/bin/env bash
# ==============================================================================
# 模型下载脚本（ModelScope）— Qwen3-VL-4B-Instruct + LocateAnything-3B
#
# 用法:
#   bash scripts/download-models.sh                    # 下载全部（等价于 all）
#   bash scripts/download-models.sh qwen3-vl           # 只下载 VL 模型 + mmproj
#   bash scripts/download-models.sh locate-anything    # 只下载 locate 模型
#   bash scripts/download-models.sh qwen3-vl locate-anything
#   MODELS_DIR=/path/models bash scripts/download-models.sh all
#
# 前置: pip install modelscope
# 已存在的文件会跳过（modelscope 支持断点续传，重复执行安全）。
# ==============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
MODELS_DIR="${MODELS_DIR:-${REPO_ROOT}/models}"

command -v modelscope >/dev/null 2>&1 || {
  echo "!!! 未找到 modelscope CLI，请先安装: pip install modelscope"; exit 1;
}

dl_qwen3_vl() {
  echo "==> 下载 Qwen3-VL-4B-Instruct（unsloth/Qwen3-VL-4B-Instruct-GGUF）→ ${MODELS_DIR}/qwen3-vl-4b/"
  modelscope download --model unsloth/Qwen3-VL-4B-Instruct-GGUF \
    Qwen3-VL-4B-Instruct-Q8_0.gguf mmproj-F16.gguf \
    --local_dir "${MODELS_DIR}/qwen3-vl-4b"
}

dl_locate_anything() {
  echo "==> 下载 LocateAnything-3B（mudler/locate-anything.cpp-gguf）→ ${MODELS_DIR}/locate-anything-3b/"
  modelscope download --model mudler/locate-anything.cpp-gguf \
    locate-anything-q8_0.gguf \
    --local_dir "${MODELS_DIR}/locate-anything-3b"
}

# 无参数 = all；支持别名 vl / locate
TARGETS=("${@:-all}")
for t in "${TARGETS[@]}"; do
  case "${t}" in
    all)                    dl_qwen3_vl; dl_locate_anything ;;
    qwen3-vl|vl)            dl_qwen3_vl ;;
    locate-anything|locate) dl_locate_anything ;;
    *) echo "未知目标: ${t}（支持 all / qwen3-vl / locate-anything）"; exit 1 ;;
  esac
done

echo
echo "✔ 下载完成:"
for t in "${TARGETS[@]}"; do
  case "${t}" in
    all)                    ls -lh "${MODELS_DIR}/qwen3-vl-4b/" "${MODELS_DIR}/locate-anything-3b/" ;;
    qwen3-vl|vl)            ls -lh "${MODELS_DIR}/qwen3-vl-4b/" ;;
    locate-anything|locate) ls -lh "${MODELS_DIR}/locate-anything-3b/" ;;
  esac
done
