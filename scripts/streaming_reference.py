"""★ 已验证可用的 hiai 流式参考实现（逐 token，实测 30 token → 30 个分片）。

⚠ 这份脚本是**当时**的形态，只有"流式"那部分仍然权威。两处已经过时，别照抄：
  * 建 Executor 用的是 `Executor_CreateFromJson` —— 后端现在走官方服务的
    `InitOption_*` + `Executor_Init_Use_Option` 那条路（见
    docs/hiai-backend-handoff.md 最后一节、以及 backends/hiai.py 的 `_create_executor`）
  * `PUSH` / `RUN` 那两个 `base + 偏移` 是**内部函数**，后端已全部弃用
    （那套东西随系统升级就失效）
  输入方式也从 `Prompt_SetTokenIds` 换成了 `Context_SetPrefixPrompt(文本)`。

在设备上运行：
    PYTHONPATH=<repo>/src <设备自带 python3.12> scripts/streaming_reference.py

要点（全部实测确认）：
  * SetOnSomeTokenGenerateDoneFunc 每生成一个 token 回调一次
  * 回调里只能用【轻量】的 GetOneTokenGeneration 取单 token 并 append 到 list
  * 回调里【不能】：GetAllTokenGeneration（全量拷贝 → 重入引擎 → segfault）、
    在回调里解码、用 queue.put
  * 读取增量由【主线程】轮询 list 长度后解码产出
  * 绝不能在生成期间从外部读 Context（会与工作线程竞态 → abort / segfault）
"""
import ctypes, os, sys, threading, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
D = os.environ.get("MODEL_DIR", "")
from cann_llm.backends.hiai_tokenizer import QwenTokenizer
L = ctypes.CDLL("/system/lib64/libhiai_llm_engine.so", mode=ctypes.RTLD_LOCAL)
def b(n, r, a):
    f = getattr(L, n); f.restype = r; f.argtypes = a; return f
V, S = ctypes.c_void_p, ctypes.c_char_p
I32P = ctypes.POINTER(ctypes.c_int32); U64 = ctypes.POINTER(ctypes.c_uint64)
cfromjson = b("HIAI_LLMEngine_Executor_CreateFromJson", V, [S])
ctx_new = b("HIAI_LLMEngine_Context_Create", V, [])
setmax = b("HIAI_LLMEngine_Context_SetMaxGenTokens", ctypes.c_int, [V, ctypes.c_int])
pr_new = b("HIAI_LLMEngine_Prompt_Create", V, [])
pr_ids = b("HIAI_LLMEngine_Prompt_SetTokenIds", ctypes.c_int, [V, I32P, ctypes.c_uint])
seta = b("HIAI_LLMEngine_Context_SetOnAllTokensGenerateDoneFunc", ctypes.c_int, [V, V])
setf = b("HIAI_LLMEngine_Context_SetOnGenerateAsyncFailed", ctypes.c_int, [V, V])
sets = b("HIAI_LLMEngine_Context_SetOnSomeTokenGenerateDoneFunc", ctypes.c_int, [V, V])
one = b("HIAI_LLMEngine_Context_GetOneTokenGeneration", ctypes.c_int, [V, S, ctypes.c_int])
base = None
for line in open("/proc/self/maps"):
    if "libhiai_llm_engine.so" in line:
        a = int(line.split("-")[0], 16); base = a if base is None else min(base, a)
PUSH = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p)(base + 0xfda28)
RUN = ctypes.CFUNCTYPE(ctypes.c_uint32, ctypes.c_void_p, ctypes.c_void_p,
                       ctypes.c_void_p)(base + 0x118704)
CB = ctypes.CFUNCTYPE(None, ctypes.c_void_p)
ids, done, failed = [], threading.Event(), threading.Event()
def on_some(p):
    one_ = ctypes.c_int32(0)
    if one(p, ctypes.cast(ctypes.byref(one_), ctypes.c_char_p), 4) == 0:
        ids.append(int(one_.value))
def on_a(p): done.set()
def on_f(p): failed.set()
os.chdir(D)
ex = cfromjson(open("executor_super.json").read().encode())
tok = QwenTokenizer(os.path.join(D, "tokenizer.json"))
ctx = ctx_new(); setmax(ctx, 30)
pr = pr_new()
i0 = [2] + tok.encode("def add(a, b): return a + b")
arr = (ctypes.c_int32 * len(i0))(*i0); pr_ids(pr, arr, len(i0))
cb_s, cb_a, cb_f = CB(on_some), CB(on_a), CB(on_f)
sets(ctx, ctypes.cast(cb_s, V)); seta(ctx, ctypes.cast(cb_a, V)); setf(ctx, ctypes.cast(cb_f, V))
vec = (ctypes.c_uint64 * 3)(0, 0, 0)
PUSH(ctypes.byref(vec), pr)
print(f"  RUN rc={RUN(ex, ctx, ctypes.byref(vec))}", flush=True)
t0 = time.time(); seen = 0; emitted = ""; n = 0
while not done.is_set() and not failed.is_set():
    if len(ids) > seen:
        seen = len(ids)
        txt = tok.decode(list(ids))
        if len(txt) > len(emitted):
            n += 1
            print(f"    [{n:>2}] {time.time()-t0:5.2f}s ids={seen} +{txt[len(emitted):]!r}", flush=True)
            emitted = txt
    time.sleep(0.02)
time.sleep(0.2)
print(f"  ★ done={done.is_set()} failed={failed.is_set()} 分片={n} token={len(ids)}", flush=True)
print(f"  ★ 全文: {emitted[:180]!r}", flush=True)
