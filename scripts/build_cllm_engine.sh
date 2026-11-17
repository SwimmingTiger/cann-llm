#!/bin/sh
# 构建 cann-local-lm 引擎库 libcann_lm_engine.so —— "缺库就编" 的手动入口。
#
# 它只做一件事：**找一个能用的 python，把参数原样转发给 cann_llm.engine_build**。
# 全部逻辑（clone 源码 / 引导三方库 / 交叉编译 / 签名 / 安装）都在 Python 里：
#   src/cann_llm/engine_build.py（设计说明见 docs/maintainer-notes.md §167.8）
#
#   scripts/build_cllm_engine.sh                                  # auto：有 SDK 用 SDK，否则用 PATH 的 clang
#   scripts/build_cllm_engine.sh --clang                          # 不用 SDK，纯 clang 本机编译
#   scripts/build_cllm_engine.sh --sdk /path/to/ohos-sdk/native   # 显式给 SDK
#   scripts/build_cllm_engine.sh --force --refresh                 # 强制重编/重 clone
#   scripts/build_cllm_engine.sh --skip-sign                       # 仅本机检查（设备上加载会失败）
#   scripts/build_cllm_engine.sh --src ~/cann-local-lm             # 用本地源码树，跳过 clone
#
# 产物：<cann-llm>/lib/libcann_lm_engine.so（已签名）+ libc++_shared.so
# 临时：<cann-llm>/.tmp/engine-build/{src,sign-tool,build-engine.log}
#
# 环境变量：
#   PYTHON                  指定解释器（跳过自动发现）
#   OHOS_SDK_NATIVE         OHOS SDK native 子树（可选；不给就用 PATH 上的 clang）
#   CANN_LLM_CLLM_*         见 docs/cann-local-lm.md §4.1.1
set -eu

ROOT=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)

# 随包 python3 优先（与 scripts/start_chat.sh 一致：发布态资产在 python3/ 下）
PY=${PYTHON:-}
if [ -z "$PY" ]; then
    for _cand in "$ROOT/python3/bin/python3" python3 python; do
        if command -v "$_cand" >/dev/null 2>&1; then
            PY=$_cand
            break
        fi
    done
fi
if [ -z "$PY" ]; then
    echo "找不到 python3；用 PYTHON=/path/to/python3 指定" >&2
    exit 1
fi

# 让 python 能 import cann_llm（源码树跑法）
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
exec "$PY" -m cann_llm.engine_build "$@"
