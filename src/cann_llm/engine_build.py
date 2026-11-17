"""按需构建 cann-local-lm 引擎库（``libcann_lm_engine.so``）。

`cllm` 后端的引擎库是**交叉编译产物**，系统里没有；用户常常只有源码仓库地址，
忘了先编库 ⇒ `-b cllm` 报"找不到 libcann_lm_engine.so" ✗。本模块把"缺了就编"做掉。

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
CANN_LLM_CLLM_SIGN       auto(默认找 PATH 上的 ohos-sign-elf.py) / skip / 指定路径
CANN_LLM_CLLM_LIB_DIR    最终库安装目录（默认 <cann-llm>/lib）
OHOS_SDK_NATIVE          OHOS SDK 的 native 子树（含 sysroot/ 与 llvm/bin/clang）
```

最终库落在 **`<cann-llm>/lib/`**（与仓库里其它随包库同处，加载器也优先从这里找），
构建日志在 `<cann-llm>/.tmp/engine-build/build-engine.log`。

## 为什么不能只编不签

设备上未签名的 ``.so`` 会被 ``dlopen`` 拒绝（``Permission denied`` ✗），所以构建完
必须用 ``ohos-sign-elf.py`` 签名再安装。
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
ENV_LIB_DIR = "CANN_LLM_CLLM_LIB_DIR"
ENV_SDK = "OHOS_SDK_NATIVE"

#: 构建脚本（相对源码树）
BUILD_SCRIPT = os.path.join("scripts", "cann_lm_engine", "build_ohos.sh")
#: 构建产物目录（相对源码树，由 build_ohos.sh 决定）
BUILD_DIR = "build_ohos"
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
    sdk: str
    script: str
    build_dir: str
    model_dir: str

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


def resolve_sdk(explicit: Optional[str] = None) -> Optional[str]:
    """找 OHOS SDK 的 native 子树（必须含 sysroot/ 与 llvm/bin/clang）。"""
    cands: List[str] = []
    if explicit:
        cands.append(explicit)
    env = os.environ.get(ENV_SDK)
    if env:
        cands.append(env)
    for c in cands:
        path = os.path.abspath(os.path.expanduser(c))
        if os.path.isdir(os.path.join(path, "sysroot")) and os.path.isfile(
                os.path.join(path, "llvm", "bin", "clang")):
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


def make_plan(model_dir: str, *, src: Optional[str] = None, sdk: Optional[str] = None,
              log: Log = _stderr, refresh: bool = False) -> BuildPlan:
    """组装构建计划（必要时 clone 源码）；条件不足时抛 :class:`BuildError`。"""
    model_dir = os.path.abspath(os.path.expanduser(model_dir)) if model_dir else ""
    got_sdk = resolve_sdk(sdk)
    if not got_sdk:
        raise BuildError(
            "找不到 OHOS SDK native 子树（要含 sysroot/ 与 llvm/bin/clang）。\n"
            f"  设 {ENV_SDK}=/path/to/ohos-sdk/native（SDK 不能自动 clone，需自行下载）。")
    got_src = ensure_source(None, log=log, explicit=src, refresh=refresh)
    return BuildPlan(src=got_src, sdk=got_sdk,
                     script=os.path.join(got_src, BUILD_SCRIPT),
                     build_dir=os.path.join(got_src, BUILD_DIR),
                     model_dir=model_dir)


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


def build(plan: BuildPlan, *, force: bool = False, log: Log = _stderr) -> str:
    """跑 ``build_ohos.sh``（在 clone 出来的源码树里），返回构建出的库路径。"""
    if not force and is_fresh(plan):
        log(f"产物已是最新，跳过构建：{plan.built_lib}")
        return plan.built_lib
    os.makedirs(plan.install_dir, exist_ok=True)
    _ensure_third_party(plan, log)
    env = dict(os.environ)
    env[ENV_SDK] = plan.sdk
    env.setdefault("PYTHON", sys.executable)
    t0 = time.monotonic()
    log(f"开始构建引擎库：{plan.script}（SDK={plan.sdk}）…")
    os.makedirs(os.path.dirname(plan.log_path), exist_ok=True)
    with open(plan.log_path, "ab") as logf:
        logf.write(f"\n==== {time.strftime('%Y-%m-%d %H:%M:%S')} build start ====\n".encode())
        logf.write(f"src={plan.src}\nsdk={plan.sdk}\n".encode())
        logf.flush()
        timeout = float(os.environ.get("CANN_LLM_CLLM_BUILD_TIMEOUT", "3600"))
        # ★--no-signed★：脚本自带的 jar 签名器在多数环境里没有 ✗（报了就给 rc=1 ✗），
        #   签名由本模块 install() 用 ohos-sign-elf.py 统一做 ✓。
        try:
            proc = subprocess.run(["bash", plan.script, "--no-signed"], cwd=plan.src, env=env,
                                  stdout=logf, stderr=subprocess.STDOUT, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            raise BuildError(
                f"构建超时（>{timeout:.0f}s）。日志：{plan.log_path}") from exc
    took = time.monotonic() - t0
    if proc.returncode != 0:
        raise BuildError(
            f"构建失败（rc={proc.returncode}，耗时 {took:.0f}s）。完整日志：{plan.log_path}\n"
            "  常见原因：cmake/make 不在 PATH、SDK 与源码不配套、SDK 缺 builtins。")
    if not os.path.isfile(plan.built_lib):
        raise BuildError(f"构建脚本成功但没产出 {plan.built_lib}。看日志：{plan.log_path}")
    log(f"构建完成（{took:.0f}s）：{plan.built_lib}")
    return plan.built_lib


def _sign(files: Sequence[str], log: Log) -> None:
    """给库签名（设备上未签名会被 dlopen 拒绝 ✗）。"""
    mode = os.environ.get(ENV_SIGN, "auto").strip()
    if mode == "skip":
        log("按配置跳过签名（仅本机检查可用；设备上会 Permission denied ✗）")
        return
    tool = mode if mode not in ("", "auto") else shutil.which("ohos-sign-elf.py")
    if not tool or not os.path.isfile(tool):
        raise BuildError(
            "找不到 ohos-sign-elf.py（设备上未签名的 .so 会被 dlopen 拒绝：Permission denied）。\n"
            "  装好签名工具后重试，或设 CANN_LLM_CLLM_SIGN=/path/to/ohos-sign-elf.py；\n"
            "  只在本机检查、不上设备时可设 CANN_LLM_CLLM_SIGN=skip。")
    log(f"签名：{tool}")
    cmd = ([sys.executable, tool, *files] if tool.endswith(".py") else [tool, *files])
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise BuildError(f"签名失败（rc={proc.returncode}）：{(proc.stderr or proc.stdout)[-400:]}")


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
        dst = os.path.join(plan.install_dir, name)
        shutil.copy2(src, dst)
        os.chmod(dst, 0o755)
        if name == LIB_NAMES[0]:
            target = dst
        log(f"安装：{dst}")
    _sign([os.path.join(plan.install_dir, n) for n in LIB_NAMES
           if os.path.isfile(os.path.join(plan.install_dir, n))], log)
    return target


def ensure_engine_lib(model_dir: str, *, log: Log = _stderr, src: Optional[str] = None,
                      sdk: Optional[str] = None, force: bool = False,
                      refresh: bool = False) -> str:
    """缺库就编（必要时先 clone 源码）：返回可直接加载的库路径。"""
    got_sdk = resolve_sdk(sdk)
    if not _autobuild_enabled(True, bool(got_sdk)):
        raise BackendUnavailableError(
            "找不到 cann-local-lm 引擎库 libcann_lm_engine.so，且未启用自动构建。\n"
            f"  自动构建需要 OHOS SDK：设 {ENV_SDK}=/path/to/ohos-sdk/native"
            f"（设好后缺源码会自动 clone 到 {os.path.join(scratch_root(), 'src')}；库装到 {os.path.join(repo_root(), 'lib')}）；\n"
            f"  或显式设 {ENV_AUTOBUILD}=1；也可以直接给现成的库："
            "CANN_LLM_CLLM_LIB=/path/libcann_lm_engine.so。")
    plan = make_plan(model_dir, src=src, sdk=sdk, log=log, refresh=refresh)
    build(plan, force=force, log=log)
    return install(plan, log=log)
