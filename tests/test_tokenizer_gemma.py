"""与 HF 的 Gemma tokenizer 对拍（x570 上有 tokenizers 库 ✓）"""
import sys, time
sys.path.insert(0, "/home/hu60/work/llm/.tmp")
from tok_gemma import GemmaTokenizer
from transformers import AutoTokenizer
M = "/home/hu60/work/llm/ddk-llm/models/gemma-4-E2B-it"
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
