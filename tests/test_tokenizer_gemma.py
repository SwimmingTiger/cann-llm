import os
"""与 HF 的 Gemma tokenizer 对拍（需装有 tokenizers 库的开发环境 ✓）

路径经环境变量提供，缺省路径不存在时本文件自动跳过 ✓
"""
import sys, time
sys.path.insert(0, os.environ.get("CANN_LLM_TMP", ""))
if not os.path.isdir(os.environ.get("CANN_LLM_TMP", "")):
    pytest.skip("未设置 CANN_LLM_TMP（对拍脚本目录），跳过", allow_module_level=True)
from tok_gemma import GemmaTokenizer
from transformers import AutoTokenizer
M = os.environ.get("GEMMA_MODEL_DIR", "")
if not M or not os.path.isdir(M):
    pytest.skip("未设置 GEMMA_MODEL_DIR，跳过", allow_module_level=True)
t0 = time.time(); mine = GemmaTokenizer(M); print("  我的加载 %.1fs" % (time.time()-t0), flush=True)
hf = AutoTokenizer.from_pretrained(M)
tests = ["Hello, world!", "The capital of France is", "1+1=", "你好，世界",
         "  leading spaces", "def f(x):\n    return x+1", "Gemma 4 is a model."]
ok = 0
for s in tests:
    a = mine.encode(s); b = hf.encode(s, add_special_tokens=False)
    same = a == b
    ok += same
    print("  %-28r %s" % (s[:26], "✓" if same else "✗ 我=%s hf=%s" % (a[:12], b[:12])), flush=True)
    if same:
        print("      decode: %r" % mine.decode(a)[:60], flush=True)
print("  ★ %d/%d 一致" % (ok, len(tests)), flush=True)
