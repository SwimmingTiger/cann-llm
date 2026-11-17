"""按需构建 cann-local-lm 引擎库（``libcann_lm_engine.so``）。

`cllm` 后端的引擎库是**交叉编译产物**，系统里没有；用户常常只有源码仓库地址，
忘了先编库 ⇒ `-b cllm` 报"找不到 libcann_lm_engine.so" ✗。本模块把"缺了就编"做掉。

## 工具链：SDK 或"普通 clang"，两种都行

* ``sdk``（交叉）：用 ``OHOS_SDK_NATIVE`` 的 ``llvm/bin/clang`` + ``sysroot``，
  走上游 ``build_ohos.sh``（``--no-signed``）；
* ``clang``（本机编译）：**不要求任何 SDK 目录**，直接用 PATH 上的 ``clang/clang++``
  （它们本身就以 ``aarch64-unknown-linux-ohos`` 为目标），cmake 直配
  ``cann_local_lm/engine/ohos``。设备上编译目标就是本机 ⇒ 天然可跑 ✓。

选择：``CANN_LLM_CLLM_TOOLCHAIN=clang|sdk``（**默认 clang**）。
只有显式指定 SDK（``--sdk`` / ``OHOS_SDK_NATIVE``）时才走 SDK 交叉编译；
**不会**从 PATH 或常见目录去猜 SDK ✗。

## 构建目录（全部在 cann-llm 仓库内）

```
<cann-llm>/.tmp/engine-build/          # 构建临时根目录（不入库，见 .gitignore）
├── src/                               # 从上游 clone 的 cann-local-lm（--depth 1）
│   ├── scripts/cann_lm_engine/build_ohos.sh
│   └── build_ohos/                    # 构建产物（libcann_lm_engine.so 等）
└── ...
```

**不在 /tmp 里做**：本机（鸿蒙设备）的 /tmp 不可用，且仓库内方便事后排查 ✓。
安装位置是 ``<模型目录>/lib/``（运行时找库的地方）。

## 环境变量

```
CANN_LLM_CLLM_REPO       上游仓库（默认 https://gitcode.com/cann/cann-local-lm.git）
CANN_LLM_CLLM_REF        分支/标签（默认取默认分支）
CANN_LLM_CLLM_SRC        直接用本地源码树（跳过 clone，便于离线/固定版本）
CANN_LLM_CLLM_BUILD_DIR  构建根目录（默认 <cann-llm>/.tmp/engine-build）
CANN_LLM_CLLM_REFRESH    1=强制重新 clone（默认复用已 clone 的）
CANN_LLM_CLLM_AUTOBUILD  1=允许自动构建 / 0=禁止 / 缺省=条件齐了才编
CANN_LLM_CLLM_SIGN       auto(默认) / skip / 指定签名器路径
CANN_LLM_CLLM_SIGN_REPO  签名工具仓库（默认 https://gitcode.com/hu60cn/ohos-bst-light）
CANN_LLM_CLLM_LIB_DIR    最终库安装目录（默认 <cann-llm>/lib）
OHOS_SDK_NATIVE          OHOS SDK 的 native 子树（含 sysroot/ 与 llvm/bin/clang）
```

最终库落在 **`<cann-llm>/lib/`**（与仓库里其它随包库同处，加载器也优先从这里找），
构建日志在 `<cann-llm>/.tmp/engine-build/build-engine.log`。

## 为什么不能只编不签

设备上未签名的 ``.so`` 会被 ``dlopen`` 拒绝（``Permission denied`` ✗），所以构建完
必须签名再安装。

签名用 **ohos-bst-light 的 ``selfsign.py``**：该仓库同样在首次构建时 clone 到
``<scratch>/sign-tool``，调用方式 ``python3 selfsign.py <elf> --force``（就地签）。
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

from .errors import BackendUnavailableError

__all__ = [
    "BuildError",
    "BuildPlan",
    "REPO_URL",
    "repo_root",
    "scratch_root",
    "resolve_src",
    "resolve_sdk",
    "ensure_source",
    "make_plan",
    "is_fresh",
    "build",
    "install",
    "ensure_engine_lib",
    "main",
]

#: 上游仓库（cann-local-lm）
REPO_URL = "https://gitcode.com/cann/cann-local-lm.git"

ENV_REPO = "CANN_LLM_CLLM_REPO"
ENV_REF = "CANN_LLM_CLLM_REF"
ENV_SRC = "CANN_LLM_CLLM_SRC"
ENV_BUILD_DIR = "CANN_LLM_CLLM_BUILD_DIR"
ENV_REFRESH = "CANN_LLM_CLLM_REFRESH"
ENV_AUTOBUILD = "CANN_LLM_CLLM_AUTOBUILD"
ENV_SIGN = "CANN_LLM_CLLM_SIGN"
ENV_SIGN_REPO = "CANN_LLM_CLLM_SIGN_REPO"
ENV_LIB_DIR = "CANN_LLM_CLLM_LIB_DIR"
ENV_SDK = "OHOS_SDK_NATIVE"
ENV_SDK_ALT = "CANN_LLM_CLLM_SDK"
ENV_TOOLCHAIN = "CANN_LLM_CLLM_TOOLCHAIN"   # auto | sdk | clang
ENV_CC = "CANN_LLM_CLLM_CC"
ENV_CXX = "CANN_LLM_CLLM_CXX"

#: 构建脚本（相对源码树）
BUILD_SCRIPT = os.path.join("scripts", "cann_lm_engine", "build_ohos.sh")
#: 构建产物目录（相对源码树，由 build_ohos.sh 决定）
BUILD_DIR = "build_ohos"
#: 签名工具仓库（ohos-bst-light，里面有 selfsign.py）
SIGN_REPO_URL = "https://gitcode.com/hu60cn/ohos-bst-light"
#: 工具名
SIGN_TOOL = "selfsign.py"
#: 需要安装/签名两个库
LIB_NAMES = ("libcann_lm_engine.so", "libc++_shared.so")
#: 判断"源码比产物新"时扫的扩展名
SRC_EXTS = (".cpp", ".cc", ".c", ".h", ".hpp", ".cmake", ".txt")

Log = Callable[[str], None]


class BuildError(BackendUnavailableError):
    """构建失败（或无法开始构建）。"""


@dataclass
class BuildPlan:
    """一次构建需要的全部信息。"""

    src: str
    sdk: str                 # 空 = 不用 SDK（clang 本机模式）
    script: str
    build_dir: str
    model_dir: str
    toolchain: str = "sdk"   # "sdk" | "clang"

    @property
    def built_lib(self) -> str:
        return os.path.join(self.build_dir, LIB_NAMES[0])

    @property
    def install_dir(self) -> str:
        """最终库目录：``<cann-llm>/lib``（可用 CANN_LLM_CLLM_LIB_DIR 覆盖）。"""
        env = os.environ.get(ENV_LIB_DIR)
        if env:
            return os.path.abspath(os.path.expanduser(env))
        return os.path.join(repo_root(), "lib")

    @property
    def log_path(self) -> str:
        """构建日志放在 .tmp 下（别污染 lib/）。"""
        return os.path.join(scratch_root(), "build-engine.log")


def _is_elf(path: str) -> bool:
    """是不是 ELF（安装/签名前必查：坏产物别拿去 sign，报错会很难懂 ✗）。"""
    try:
        with open(path, "rb") as fh:
            return fh.read(4) == b"\x7fELF"
    except OSError:
        return False


def _stderr(text: str) -> None:
    print(f"[cllm/build] {text}", file=sys.stderr, flush=True)


def repo_root() -> str:
    """cann-llm 仓库根目录（本文件在 ``<repo>/src/cann_llm/`` 下）。"""
    return os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))


def scratch_root() -> str:
    """构建临时根目录（默认 ``<cann-llm>/.tmp/engine-build``，不用 /tmp）。"""
    env = os.environ.get(ENV_BUILD_DIR)
    if env:
        return os.path.abspath(os.path.expanduser(env))
    return os.path.join(repo_root(), ".tmp", "engine-build")


def resolve_src(model_dir: Optional[str] = None, explicit: Optional[str] = None) -> Optional[str]:
    """**已在本地**的源码树：显式 → ``CANN_LLM_CLLM_SRC`` → 模型目录下的 src/。

    注意：这里只找现成的；要"没有就 clone"请用 :func:`ensure_source`。
    """
    cands: List[str] = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get(ENV_SRC)
    if env:
        cands.append(env)
    if model_dir:
        cands.append(os.path.join(model_dir, "src", "cann-local-lm"))
        cands.append(os.path.join(model_dir, "cann-local-lm"))
    for c in cands:
        path = os.path.abspath(os.path.expanduser(c))
        if os.path.isfile(os.path.join(path, BUILD_SCRIPT)):
            return path
    return None


def _is_sdk_native(path: str) -> bool:
    return (os.path.isdir(os.path.join(path, "sysroot"))
            and os.path.isfile(os.path.join(path, "llvm", "bin", "clang")))


def sdk_sources() -> "List[str]":
    """给报错用：SDK **只在显式指定时**才用（不猜 ✗）。"""
    return [ENV_SDK, ENV_SDK_ALT, "--sdk <path>"]


def resolve_sdk(explicit: Optional[str] = None) -> Optional[str]:
    """取**用户显式指定**的 OHOS SDK native 子树（含 ``sysroot/`` 与 ``llvm/bin/clang``）。

    ★默认不猜★：只有 ``--sdk`` / ``OHOS_SDK_NATIVE`` / ``CANN_LLM_CLLM_SDK``
    给了才用 SDK；**不**从 PATH、也**不**扫常见目录推断。
    没指定 ⇒ 走普通 clang（见 :func:`pick_toolchain`）。
    """
    for c in (explicit, os.environ.get(ENV_SDK), os.environ.get(ENV_SDK_ALT)):
        if not c:
            continue
        path = os.path.abspath(os.path.expanduser(c))
        if not _is_sdk_native(path):
            raise BuildError(
                f"指定的 SDK 不可用：{path}（需要 sysroot/ 与 llvm/bin/clang）。")
        return path
    return None


def _git(args: Sequence[str], log: Log) -> None:
    log("git " + " ".join(args))
    proc = subprocess.run(["git", *args], capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(
            "git 失败：git " + " ".join(args) + "\n  "
            + (proc.stderr or proc.stdout).strip()[-500:])


def ensure_source(scratch: Optional[str] = None, *, log: Log = _stderr,
                  explicit: Optional[str] = None, refresh: bool = False) -> str:
    """拿到可构建的源码树：

    1. ``CANN_LLM_CLLM_SRC`` / 显式路径（已在本地）⇒ 直接用；
    2. 否则从上游 **clone 到 ``<scratch>/src``**（已存在且不 refresh 就复用）。
    """
    local = resolve_src(None, explicit)
    if local:
        return local
    root = scratch or scratch_root()
    dest = os.path.join(root, "src")
    if refresh or not os.path.isfile(os.path.join(dest, BUILD_SCRIPT)):
        fresh = dest + ".new"
        if os.path.isdir(fresh):
            shutil.rmtree(fresh)
        os.makedirs(root, exist_ok=True)
        args = ["clone", "--depth", "1"]
        ref = os.environ.get(ENV_REF, "").strip()
        if ref:
            args += ["--branch", ref]
        args += [os.environ.get(ENV_REPO, REPO_URL).strip() or REPO_URL, fresh]
        try:
            _git(args, log)
        except BuildError:
            if os.path.isfile(os.path.join(dest, BUILD_SCRIPT)):
                log("clone 失败，改用已缓存的源码树（可用 CANN_LLM_CLLM_REFRESH=1 强制重来）")
                return dest
            raise
        if os.path.isdir(dest):
            shutil.rmtree(dest)
        os.replace(fresh, dest)
        log(f"源码树已就绪：{dest}")
    return dest


def _autobuild_enabled(has_src: bool, has_sdk: bool) -> bool:
    flag = os.environ.get(ENV_AUTOBUILD, "").strip()
    if flag == "0":
        return False
    if flag == "1":
        return True
    # 缺省：能拿到 SDK 才自动做（源码可以 clone）；SDK 不能克隆，缺了必然失败
    return has_sdk


def _find_compiler(name: str, alt_env: str, fallback: str) -> Optional[str]:
    """找编译器：``CANN_LLM_CLLM_CC/CXX`` → ``CC/CXX`` → PATH 上的 clang/clang++。"""
    for cand in (os.environ.get(alt_env), os.environ.get(name), fallback):
        if not cand:
            continue
        got = shutil.which(cand) or (cand if os.path.isfile(cand) else None)
        if got:
            return got
    return None


def pick_toolchain(explicit: Optional[str] = None,
                   sdk_path: Optional[str] = None) -> str:
    """决定工具链：**默认普通 clang**（用 PATH 上的 clang/clang++）。

    只有用户显式给 SDK（``--sdk`` / ``OHOS_SDK_NATIVE`` / ``CANN_LLM_CLLM_SDK``）
    或显式 ``--toolchain sdk`` 时才走 SDK 交叉编译；不做任何推断 ✓。
    """
    want = (explicit or os.environ.get(ENV_TOOLCHAIN, "")).strip().lower()
    if want == "sdk":
        return "sdk"
    if want == "clang":
        return "clang"
    return "sdk" if sdk_path else "clang"


def make_plan(model_dir: str, *, src: Optional[str] = None, sdk: Optional[str] = None,
              toolchain: Optional[str] = None, log: Log = _stderr,
              refresh: bool = False) -> BuildPlan:
    """组装构建计划（必要时 clone 源码）。

    ★不再强制要求 SDK★：没 SDK 时自动走 ``clang`` 本机模式（PATH 上的 clang）。
    只有显式 ``toolchain="sdk"`` 且找不到 SDK 才报错。
    """
    model_dir = os.path.abspath(os.path.expanduser(model_dir)) if model_dir else ""
    got_sdk = resolve_sdk(sdk)
    mode = pick_toolchain(toolchain, got_sdk)
    # 先把工具链敲定，再 clone（否则配置错了还要白等一次 clone ✗）
    if mode == "clang":
        if not (_find_compiler("CC", ENV_CC, "clang")
                and _find_compiler("CXX", ENV_CXX, "clang++")):
            raise BuildError(
                "clang 工具链不可用：PATH 上找不到 clang/clang++"
                f"（可用 {ENV_CC}/{ENV_CXX} 或 CC/CXX 指定）。\n"
                "  也可以给 --sdk <native> 走 SDK 交叉编译。")
    got_src = ensure_source(None, log=log, explicit=src, refresh=refresh)

    if mode == "sdk":
        if not got_sdk:
            raise BuildError(
                "指定了 sdk 工具链，但找不到 OHOS SDK native 子树"
                "（要含 sysroot/ 与 llvm/bin/clang）。\n"
                "  已尝试：" + "；".join(sdk_sources()) + "。\n"
                f"  显式给：{ENV_SDK}=/path/to/ohos-sdk/native 或 --sdk <path>；"
                "或改用 --toolchain clang（用 PATH 上的 clang，不需要 SDK）。")
        return BuildPlan(src=got_src, sdk=got_sdk,
                         script=os.path.join(got_src, BUILD_SCRIPT),
                         build_dir=os.path.join(got_src, BUILD_DIR),
                         model_dir=model_dir, toolchain="sdk")

    cc = _find_compiler("CC", ENV_CC, "clang")
    cxx = _find_compiler("CXX", ENV_CXX, "clang++")
    if not cc or not cxx:
        raise BuildError(
            "clang 工具链不可用：PATH 上找不到 clang/clang++"
            f"（可用 {ENV_CC}/{ENV_CXX} 或 CC/CXX 指定）。\n"
            "  也可以给 --sdk <native> 走 SDK 交叉编译。")
    return BuildPlan(src=got_src, sdk="", script=os.path.join(got_src, BUILD_SCRIPT),
                     build_dir=os.path.join(scratch_root(), "clang-build"),
                     model_dir=model_dir, toolchain="clang")


def _newest_source(plan: BuildPlan) -> float:
    """源码树里最新一个源文件的 mtime（只看 engine 子树 + 顶层 cmake）。"""
    root = os.path.join(plan.src, "cann_local_lm", "engine")
    newest = 0.0
    for base, _dirs, files in os.walk(root):
        for name in files:
            if name.endswith(SRC_EXTS):
                try:
                    newest = max(newest, os.path.getmtime(os.path.join(base, name)))
                except OSError:
                    pass
    for extra in ("CMakeLists.txt", "build.sh"):
        p = os.path.join(plan.src, extra)
        if os.path.isfile(p):
            newest = max(newest, os.path.getmtime(p))
    return newest


def is_fresh(plan: BuildPlan) -> bool:
    """已有产物且不比源码旧 ⇒ 可以跳过构建。"""
    lib = plan.built_lib
    if not os.path.isfile(lib):
        return False
    try:
        return os.path.getmtime(lib) >= _newest_source(plan)
    except OSError:
        return False


def _ensure_third_party(plan: BuildPlan, log: Log) -> None:
    """首次构建先引导 C++ 三方库（``bash build.sh third_party``）。

    不引导的话 cmake 会停在
    ``[nlohmann_json] nlohmann_json not found`` ✗（fresh clone 里没有 output/third_party ✓）。
    """
    marker = os.path.join(plan.src, "output", "third_party", "json")
    if os.path.isdir(marker):
        return
    script = os.path.join(plan.src, "build.sh")
    if not os.path.isfile(script):
        log("警告：找不到 build.sh，跳过三方库引导（构建可能因缺 nlohmann_json 失败）")
        return
    log("引导 C++ 三方库：bash build.sh third_party（首次需要，会下载依赖）…")
    env = dict(os.environ)
    env.setdefault("PYTHON", sys.executable)
    os.makedirs(os.path.dirname(plan.log_path), exist_ok=True)
    with open(plan.log_path, "ab") as logf:
        logf.write(b"\n==== third_party bootstrap ====\n")
        logf.flush()
        proc = subprocess.run(["bash", script, "third_party"], cwd=plan.src,
                              env=env, stdout=logf, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        raise BuildError(
            f"三方库引导失败（rc={proc.returncode}）。完整日志：{plan.log_path}\n"
            "  常见原因：网络无法下载依赖（可设 CANN_3RD_LIB_PATH 指向现成三方库目录）。")


def _run(cmd: "Sequence[str]", cwd: str, logf, timeout: float, log: Log,
         env: "Optional[dict]" = None) -> None:
    """跑一条命令：日志进文件、失败带命令；可传环境变量；超时也报清楚。"""
    log("$ " + " ".join(cmd))
    try:
        proc = subprocess.run(cmd, cwd=cwd, env=env, stdout=logf,
                              stderr=subprocess.STDOUT, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        raise BuildError(f"命令超时（>{timeout:.0f}s）：{' '.join(cmd)}") from exc
    if proc.returncode != 0:
        raise BuildError(f"命令失败（rc={proc.returncode}）：{' '.join(cmd)}")


def _ensure_cxx_shared(plan: BuildPlan, log: Log) -> None:
    """本机 clang 模式没有 SDK 目录 ⇒ 自己找一份 ``libc++_shared.so`` 放到产物旁。"""
    dst = os.path.join(plan.build_dir, "libc++_shared.so")
    if os.path.isfile(dst):
        if _is_elf(dst):
            return
        os.remove(dst)          # 坏产物（非 ELF）⇒ 丢掉重找 ✓
    cands: List[str] = []
    cxx = _find_compiler("CXX", ENV_CXX, "clang++")
    if cxx:
        try:
            out = subprocess.run([cxx, "-print-file-name=libc++_shared.so"],
                                 capture_output=True, text=True, timeout=15).stdout.strip()
            if out and os.path.isfile(out):
                cands.append(out)
        except (OSError, subprocess.SubprocessError):
            pass
    cands += ["/system/lib64/libc++_shared.so", "/system/lib64/ndk/libc++_shared.so"]
    if plan.sdk:
        for base, _dirs, files in os.walk(plan.sdk):
            if "libc++_shared.so" in files:
                cands.append(os.path.join(base, "libc++_shared.so"))
                break
    for c in cands:
        if os.path.isfile(c) and _is_elf(c):
            shutil.copy2(c, dst)
            log(f"libc++_shared.so ← {c}")
            return
    log("警告：没找到 libc++_shared.so（设备上它可能在 /system/lib64 里，加载时再解析）")


def _build_with_clang(plan: BuildPlan, logf, timeout: float, log: Log) -> None:
    """★不用 SDK★：直接用 PATH 上的 clang/clang++ 配 cmake（目标就是本机 ohos-arm64）。"""
    cc = _find_compiler("CC", ENV_CC, "clang")
    cxx = _find_compiler("CXX", ENV_CXX, "clang++")
    if not cc or not cxx:
        raise BuildError("找不到 clang/clang++（可用 --cc/--cxx 或 CC/CXX 指定）")
    src_dir = os.path.join(plan.src, "cann_local_lm", "engine", "ohos")
    if not os.path.isdir(src_dir):
        raise BuildError(f"源码树里缺 {src_dir}")
    os.makedirs(plan.build_dir, exist_ok=True)
    log(f"工具链：clang（{cc} / {cxx}）— 不使用 SDK 目录")
    _run(["cmake", "-G", "Unix Makefiles",
          "-DCMAKE_BUILD_TYPE=Release", "-DOHOS_PLATFORM=OHOS",
          f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}",
          "-B", plan.build_dir, "-S", src_dir], plan.src, logf, timeout, log)
    _run(["cmake", "--build", plan.build_dir, "-j", str(os.cpu_count() or 1)],
         plan.src, logf, timeout, log)
    _ensure_cxx_shared(plan, log)


def _log_tail(path: str, lines: int = 12) -> str:
    """构建日志最后几行（失败时直接打出来，别让用户自己去翻文件 ✓）。"""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            tail = fh.read().strip().splitlines()[-lines:]
        return "\n".join("    " + ln for ln in tail)
    except OSError:
        return ""


def build(plan: BuildPlan, *, force: bool = False, log: Log = _stderr) -> str:
    """构建引擎库（``clang`` 直接 cmake；``sdk`` 走 build_ohos.sh），返回库路径。

    默认 ``clang``；只有显式给了 SDK/工具链才是 ``sdk``。**失败不偷偷换工具链** ✗。
    """
    if not force and is_fresh(plan):
        log(f"产物已是最新，跳过构建：{plan.built_lib}")
        return plan.built_lib
    os.makedirs(plan.install_dir, exist_ok=True)
    _ensure_third_party(plan, log)
    t0 = time.monotonic()
    os.makedirs(os.path.dirname(plan.log_path), exist_ok=True)
    timeout = float(os.environ.get("CANN_LLM_CLLM_BUILD_TIMEOUT", "3600"))
    log(f"开始构建引擎库（toolchain={plan.toolchain}）…")
    with open(plan.log_path, "ab") as logf:
        logf.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} build start ====\n".encode())
        logf.write(f"src={plan.src}\nsdk={plan.sdk or '<无>'}\ntoolchain={plan.toolchain}\n".encode())
        logf.flush()
        try:
            if plan.toolchain == "clang":
                _build_with_clang(plan, logf, timeout, log)
            else:
                env = dict(os.environ)
                env[ENV_SDK] = plan.sdk          # ★必须传★ build_ohos.sh 用 set -u ✓
                env.setdefault("PYTHON", sys.executable)
                # --no-signed：脚本自带 jar 签名器多数环境没有 ✗；
                #   签名由 install() 用 ohos-bst-light 的 selfsign.py 统一做 ✓。
                _run(["bash", plan.script, "--no-signed"], plan.src,
                     logf, timeout, log, env=env)
        except BuildError as exc:
            raise BuildError(
                f"{exc}\n  日志末尾：\n{_log_tail(plan.log_path)}\n"
                f"  完整日志：{plan.log_path}") from exc
    took = time.monotonic() - t0
    if not os.path.isfile(plan.built_lib):
        raise BuildError(
            f"构建命令成功但没产出 {plan.built_lib}。看日志：{plan.log_path}")
    log(f"构建完成（{took:.0f}s，toolchain={plan.toolchain}）：{plan.built_lib}")
    return plan.built_lib


def ensure_sign_tool(scratch: Optional[str] = None, *, log: Log = _stderr,
                     refresh: bool = False) -> str:
    """拿到 ``selfsign.py``：没有就从 ohos-bst-light **clone 到 ``<scratch>/sign-tool``**。

    与源码同理：已 clone 且不 refresh 就复用；clone 失败但有缓存时用缓存并告警。
    """
    root = scratch or scratch_root()
    dest = os.path.join(root, "sign-tool")
    tool = os.path.join(dest, SIGN_TOOL)
    if not refresh and os.path.isfile(tool):
        return tool
    fresh = dest + ".new"
    if os.path.isdir(fresh):
        shutil.rmtree(fresh)
    os.makedirs(root, exist_ok=True)
    url = os.environ.get(ENV_SIGN_REPO, SIGN_REPO_URL).strip() or SIGN_REPO_URL
    try:
        _git(["clone", "--depth", "1", url, fresh], log)
    except BuildError:
        if os.path.isfile(tool):
            log("签名工具 clone 失败，改用已缓存的 selfsign.py")
            return tool
        raise
    if os.path.isdir(dest):
        shutil.rmtree(dest)
    os.replace(fresh, dest)
    if not os.path.isfile(tool):
        raise BuildError(f"clone 完成但没找到 {SIGN_TOOL}（仓库={url}）")
    log(f"签名工具已就绪：{tool}")
    return tool


def _sign(files: Sequence[str], log: Log) -> None:
    """给库签名（设备上未签名会被 dlopen 拒绝 ✗）。

    * 默认（``CANN_LLM_CLLM_SIGN`` 空/auto）⇒ clone ohos-bst-light，用 ``selfsign.py``；
    * ``skip`` ⇒ 跳过（仅本机检查）；
    * 其它值 ⇒ 当作签名器路径（自定义工具按 ``<tool> <elf>`` 调用）。
    """
    mode = os.environ.get(ENV_SIGN, "auto").strip()
    if mode == "skip":
        log("按配置跳过签名（仅本机检查可用；设备上会 Permission denied ✗）")
        return
    if mode not in ("", "auto"):
        tool = mode
        if not os.path.isfile(tool):
            raise BuildError(f"{ENV_SIGN}={tool} 不存在")
    else:
        tool = ensure_sign_tool(log=log)
    log(f"签名：{tool}")
    use_force = os.path.basename(tool) == SIGN_TOOL
    for path in files:
        cmd = [sys.executable, tool, path] if tool.endswith(".py") else [tool, path]
        if use_force:
            cmd.append("--force")          # selfsign.py：已有 .codesign 段时先剥离再重签 ✓
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if proc.returncode != 0:
            raise BuildError(
                f"签名失败（{path}，rc={proc.returncode}）："
                f"{(proc.stderr or proc.stdout)[-400:]}")


def install(plan: BuildPlan, *, log: Log = _stderr) -> str:
    """把构建产物装到 ``<模型目录>/lib/`` 并签名，返回最终库路径。"""
    os.makedirs(plan.install_dir, exist_ok=True)
    target = ""
    for name in LIB_NAMES:
        src = os.path.join(plan.build_dir, name)
        if not os.path.isfile(src):
            if name == LIB_NAMES[0]:
                raise BuildError(f"构建目录里缺 {src}")
            continue
        if not _is_elf(src):
            if name == LIB_NAMES[0]:
                raise BuildError(f"产物不是 ELF：{src}（看日志 {plan.log_path}）")
            log(f"跳过非 ELF 的附带库：{src}（设备上有系统版 libc++_shared.so 可用）")
            continue
        dst = os.path.join(plan.install_dir, name)
        # ★原子替换★：不能直接 open(dst,'wb') —— 已签名/已落盘的 .so 在 OHOS 上
        #   会带特殊属性，直接覆盖报 "Operation not permitted" ✗（实测）；
        #   写临时文件再 rename 则不受影响 ✓（selfsign.py 也是这么干的）。
        tmp = dst + ".tmp"
        shutil.copy2(src, tmp)
        os.chmod(tmp, 0o755)
        os.replace(tmp, dst)
        if name == LIB_NAMES[0]:
            target = dst
        log(f"安装：{dst}")
    _sign([os.path.join(plan.install_dir, n) for n in LIB_NAMES
           if os.path.isfile(os.path.join(plan.install_dir, n))], log)
    return target


def ensure_engine_lib(model_dir: str, *, log: Log = _stderr, src: Optional[str] = None,
                      sdk: Optional[str] = None, toolchain: Optional[str] = None,
                      force: bool = False, refresh: bool = False,
                      require_enabled: bool = True) -> str:
    """缺库就编（必要时先 clone 源码）：返回可直接加载的库路径。

    ``require_enabled``：``-b cllm`` 的**自动**路径为 True（尊重
    ``CANN_LLM_CLLM_AUTOBUILD=0``）；用户**显式**跑构建脚本/CLI 时为 False
    （意图明确，不该被开关拦住 ✗）。

    ★默认不需要 SDK★：用 PATH 上的 clang 编；要给 SDK 就说一声。
    """
    if require_enabled and os.environ.get(ENV_AUTOBUILD, "").strip() == "0":
        raise BackendUnavailableError(
            f"自动构建被 {ENV_AUTOBUILD}=0 关掉了。\n"
            "  去掉该变量，或直接跑 scripts/build_cllm_engine.sh 显式构建，"
            "或给现成的库 CANN_LLM_CLLM_LIB=/path/libcann_lm_engine.so。")
    plan = make_plan(model_dir, src=src, sdk=sdk, toolchain=toolchain,
                     log=log, refresh=refresh)
    build(plan, force=force, log=log)
    return install(plan, log=log)


# ---------------------------------------------------------------- 命令行入口

def main(argv: "Optional[Sequence[str]]" = None) -> int:
    """``python -m cann_llm.engine_build`` —— 手动触发一次"clone + 构建 + 签名 + 安装"。

    ``scripts/build_cllm_engine.sh`` 只是找解释器 + 转发参数，逻辑都在这里。
    """
    import argparse

    ap = argparse.ArgumentParser(
        prog="build_cllm_engine",
        description="构建 cann-local-lm 引擎库 libcann_lm_engine.so"
                    "（缺失则 clone 源码 → 编 → 签名 → 装到 <cann-llm>/lib）")
    ap.add_argument("-d", "--model-dir", default="",
                    help="模型目录（可选；默认 <cann-llm>/.tmp/model）")
    ap.add_argument("--sdk", default="", help="OHOS SDK native 子树（等同 OHOS_SDK_NATIVE）")
    ap.add_argument("--toolchain", choices=("sdk", "clang"), default="",
                    help="工具链：clang(默认，用 PATH 上的 clang) / "
                         "sdk(需 --sdk 或 OHOS_SDK_NATIVE)")
    ap.add_argument("--clang", action="store_true",
                    help="等价于 --toolchain clang：用 PATH 上的 clang，不需要 SDK")
    ap.add_argument("--cc", default="", help="C 编译器（clang 模式，默认 PATH 的 clang）")
    ap.add_argument("--cxx", default="", help="C++ 编译器（clang 模式，默认 PATH 的 clang++）")
    ap.add_argument("--src", default="", help="本地源码树（跳过 clone）")
    ap.add_argument("--repo", default="", help="源码仓库（默认 gitcode cann-local-lm）")
    ap.add_argument("--ref", default="", help="分支/标签")
    ap.add_argument("--lib-dir", default="", help="安装目录（默认 <cann-llm>/lib）")
    ap.add_argument("--build-dir", default="", help="构建临时根（默认 <cann-llm>/.tmp/engine-build）")
    ap.add_argument("--sign-tool", default="", help="自定义签名器（默认 clone ohos-bst-light 用 selfsign.py）")
    ap.add_argument("--skip-sign", action="store_true", help="跳过签名（仅本机检查，设备上加载会 Permission denied）")
    ap.add_argument("--force", action="store_true", help="即使产物比源码新也重编")
    ap.add_argument("--refresh", action="store_true", help="强制重新 clone 源码与签名工具")
    args = ap.parse_args(list(argv) if argv is not None else None)

    if args.sdk:
        os.environ[ENV_SDK] = args.sdk
    toolchain = "clang" if args.clang else (args.toolchain or None)
    if toolchain:
        os.environ[ENV_TOOLCHAIN] = toolchain
    if args.cc:
        os.environ[ENV_CC] = args.cc
    if args.cxx:
        os.environ[ENV_CXX] = args.cxx
    if args.repo:
        os.environ[ENV_REPO] = args.repo
    if args.ref:
        os.environ[ENV_REF] = args.ref
    if args.lib_dir:
        os.environ[ENV_LIB_DIR] = args.lib_dir
    if args.build_dir:
        os.environ[ENV_BUILD_DIR] = args.build_dir
    if args.skip_sign:
        os.environ[ENV_SIGN] = "skip"
    elif args.sign_tool:
        os.environ[ENV_SIGN] = args.sign_tool

    model_dir = args.model_dir or os.path.join(scratch_root(), "model")
    try:
        lib = ensure_engine_lib(model_dir, src=args.src or None, sdk=args.sdk or None,
                                toolchain=toolchain,
                                force=args.force, refresh=args.refresh,
                                require_enabled=False)
    except BackendUnavailableError as exc:
        print(f"构建失败：{exc}", file=sys.stderr)
        return 1
    print(f"✓ 引擎库就绪：{lib}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
