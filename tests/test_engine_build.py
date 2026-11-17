"""``engine_build``：缺引擎库时自动构建/签名/安装 的单测。

用**假的源码树 + 假 SDK + 假签名器**跑真实流程（不碰设备、不碰真交叉编译）：
``scripts/cann_lm_engine/build_ohos.sh`` 只负责写两个假 ``.so`` ✓
"""

from __future__ import annotations

import os
import shutil
import tempfile
import unittest

from cann_llm import engine_build as eb
from cann_llm.errors import BackendUnavailableError

#: ★测试临时目录放在仓库内★（本机 /tmp 不可用；与自动构建的 .tmp 同源）
_TMP_BASE = os.path.join(eb.repo_root(), ".tmp", "unit-engine-build")


class _RepoDir:
    """仓库内临时目录（用完删除）。"""

    def __enter__(self) -> str:
        os.makedirs(_TMP_BASE, exist_ok=True)
        self.path = tempfile.mkdtemp(dir=_TMP_BASE)
        return self.path

    def __exit__(self, *exc) -> None:
        shutil.rmtree(self.path, ignore_errors=True)

_ENV_KEYS = (eb.ENV_SRC, eb.ENV_SDK, eb.ENV_AUTOBUILD, eb.ENV_SIGN,
             eb.ENV_LIB_DIR, eb.ENV_REPO, eb.ENV_REF, eb.ENV_REFRESH,
             eb.ENV_BUILD_DIR)


class _Env(unittest.TestCase):
    """保存/恢复相关环境变量，避免测试之间互相污染。"""

    def setUp(self) -> None:
        self._saved = {k: os.environ.get(k) for k in _ENV_KEYS}
        for k in _ENV_KEYS:
            os.environ.pop(k, None)

    def tearDown(self) -> None:
        for k, v in self._saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


def _write(path: str, text: str = "", mode: int = 0o644) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)
    os.chmod(path, mode)


def _fake_tree(root: str) -> "tuple[str, str]":
    """造源码树 + SDK；返回 (src, sdk)。"""
    src = os.path.join(root, "cann-local-lm")
    sdk = os.path.join(root, "sdk-native")
    _write(os.path.join(src, eb.BUILD_SCRIPT),
           "#!/bin/bash\nset -e\n"
           "mkdir -p build_ohos\n"
           "echo LIB-$(date +%s%N) > build_ohos/libcann_lm_engine.so\n"
           "echo CPP > build_ohos/libc++_shared.so\n",
           mode=0o755)
    _write(os.path.join(src, "cann_local_lm", "engine", "main.cpp"), "int main(){}")
    _write(os.path.join(src, "CMakeLists.txt"), "project(x)")
    _write(os.path.join(src, "output", "third_party", "json", ".keep"))
    _write(os.path.join(sdk, "sysroot", ".keep"))
    _write(os.path.join(sdk, "llvm", "bin", "clang"), "", mode=0o755)
    return src, sdk


def _fake_signer(root: str) -> str:
    path = os.path.join(root, "ohos-sign-elf.py")
    _write(path, "import sys\n"
                 "for p in sys.argv[1:]:\n"
                 "    open(p, 'a').write('SIGNED\\n')\n"
                 "print('SIGNED', *sys.argv[1:])\n", mode=0o755)
    return path


class TestResolve(_Env):
    def test_resolve_src_env(self):
        with _RepoDir() as d:
            src, _sdk = _fake_tree(d)
            os.environ[eb.ENV_SRC] = src
            self.assertEqual(eb.resolve_src(), src)

    def test_resolve_src_from_model_dir(self):
        with _RepoDir() as d:
            src, _sdk = _fake_tree(d)
            model = os.path.join(d, "model")
            os.makedirs(os.path.join(model, "src"))
            os.symlink(src, os.path.join(model, "src", "cann-local-lm"))
            self.assertEqual(os.path.realpath(eb.resolve_src(model)), os.path.realpath(src))

    def test_resolve_sdk_requires_sysroot_and_clang(self):
        with _RepoDir() as d:
            _src, sdk = _fake_tree(d)
            os.environ[eb.ENV_SDK] = sdk
            self.assertEqual(eb.resolve_sdk(), sdk)
            os.remove(os.path.join(sdk, "llvm", "bin", "clang"))
            self.assertIsNone(eb.resolve_sdk())

    def test_plan_requires_sdk_first(self):
        """SDK 不能克隆 ⇒ 缺它必须直接报，且消息里说清"SDK 需自行下载" ✓"""
        with _RepoDir() as d:
            src, _sdk = _fake_tree(d)
            os.environ[eb.ENV_SRC] = src
            with self.assertRaises(eb.BuildError) as cm:
                eb.make_plan(os.path.join(d, "m"))
            self.assertIn(eb.ENV_SDK, str(cm.exception))


def _materialize_src(dest: str, sdk: str) -> None:
    """在 dest 造一棵能构建的源码树（模拟 clone 结果）。"""
    _write(os.path.join(dest, eb.BUILD_SCRIPT),
           "#!/bin/bash\nset -e\nmkdir -p build_ohos\n"
           "echo LIB > build_ohos/libcann_lm_engine.so\n"
           "echo CPP > build_ohos/libc++_shared.so\n", mode=0o755)
    _write(os.path.join(dest, "cann_local_lm", "engine", "main.cpp"), "int main(){}")
    _write(os.path.join(dest, "CMakeLists.txt"), "project(x)")
    _write(os.path.join(dest, "output", "third_party", "json", ".keep"))


class TestSourceAcquire(_Env):
    """源码获取：没有就 clone 到 <cann-llm>/.tmp/engine-build/src ✓（不用 /tmp ✓）。"""

    def test_scratch_root_is_inside_repo(self):
        root = eb.scratch_root()
        self.assertTrue(root.startswith(eb.repo_root() + os.sep))
        self.assertIn(os.sep + ".tmp" + os.sep, root)
        self.assertNotEqual(os.path.dirname(root), "/tmp")

    def test_scratch_root_env_override(self):
        with _RepoDir() as d:
            os.environ[eb.ENV_BUILD_DIR] = d
            self.assertEqual(eb.scratch_root(), d)

    def test_clone_when_missing_and_reuse(self):
        with _RepoDir() as d:
            sdk_dir = os.path.join(d, "sdk")
            calls = []

            def fake_git(args, log):
                calls.append(list(args))
                dest = args[-1]
                _materialize_src(dest, sdk_dir)

            real = eb._git
            eb._git = fake_git
            try:
                scratch = os.path.join(d, "scratch")
                src = eb.ensure_source(scratch, log=lambda _m: None)
                self.assertEqual(src, os.path.join(scratch, "src"))
                self.assertEqual(len(calls), 1)
                self.assertIn("--depth", calls[0])
                # 再调一次：复用，不再 clone ✓
                self.assertEqual(eb.ensure_source(scratch, log=lambda _m: None), src)
                self.assertEqual(len(calls), 1)
                # refresh=1：重新 clone ✓
                eb.ensure_source(scratch, log=lambda _m: None, refresh=True)
                self.assertEqual(len(calls), 2)
            finally:
                eb._git = real

    def test_clone_failure_falls_back_to_cache(self):
        with _RepoDir() as d:
            scratch = os.path.join(d, "scratch")
            _materialize_src(os.path.join(scratch, "src"), "")

            def boom(_args, _log):
                raise eb.BuildError("网络不通")

            real = eb._git
            eb._git = boom
            try:
                logs = []
                src = eb.ensure_source(scratch, log=logs.append, refresh=True)
                self.assertEqual(src, os.path.join(scratch, "src"))
                self.assertTrue(any("缓存" in line for line in logs))
            finally:
                eb._git = real

    def test_clone_uses_env_repo_and_ref(self):
        with _RepoDir() as d:
            seen = {}

            def fake_git(args, _log):
                seen["args"] = list(args)
                _materialize_src(args[-1], "")

            real = eb._git
            eb._git = fake_git
            os.environ[eb.ENV_REPO] = "https://example.invalid/x.git"
            os.environ[eb.ENV_REF] = "v1.2.3"
            try:
                eb.ensure_source(os.path.join(d, "s"), log=lambda _m: None)
                self.assertIn("https://example.invalid/x.git", seen["args"])
                self.assertIn("v1.2.3", seen["args"])
            finally:
                eb._git = real


def _fake_sign_tool_repo(dest: str) -> str:
    """在 dest 造一个 ohos-bst-light 的最小替身（只有 selfsign.py）。"""
    argv_dump = os.path.join(os.path.dirname(dest), "sign_argv.txt")
    _write(os.path.join(dest, eb.SIGN_TOOL),
           "import sys, os\n"
           f"open({argv_dump!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
           "open(sys.argv[1], 'a').write('SIGNED\\n')\n", mode=0o755)
    return argv_dump


class TestSigning(_Env):
    """签名改用 ohos-bst-light 的 selfsign.py（clone 到 .tmp/sign-tool ✓）。"""

    def test_auto_clones_selfsign_and_signs_with_force(self):
        with _RepoDir() as d:
            os.environ[eb.ENV_BUILD_DIR] = os.path.join(d, "scratch")
            argv_dump = os.path.join(d, "scratch", "sign_argv.txt")

            def fake_git(args, _log):
                self.assertEqual(args[0], "clone")
                self.assertTrue(args[-1].endswith("sign-tool.new"))
                _write(os.path.join(args[-1], eb.SIGN_TOOL),
                       "import sys\n"
                       f"open({argv_dump!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
                       "open(sys.argv[1], 'a').write('SIGNED\\n')\n", mode=0o755)

            real = eb._git
            eb._git = fake_git
            try:
                tool = eb.ensure_sign_tool(log=lambda _m: None)
                self.assertTrue(tool.endswith(eb.SIGN_TOOL))
                self.assertTrue(os.path.isfile(tool))
                elf = os.path.join(d, "x.so")
                _write(elf, "ELF-HEADER")
                eb._sign([elf], lambda _m: None)
                with open(elf, encoding="utf-8") as fh:
                    self.assertIn("SIGNED", fh.read())
                with open(argv_dump, encoding="utf-8") as fh:
                    self.assertIn("--force", fh.read())     # 已有段时先剥离再签 ✓
            finally:
                eb._git = real

    def test_skip_mode_does_not_clone(self):
        with _RepoDir() as d:
            os.environ[eb.ENV_BUILD_DIR] = os.path.join(d, "s")
            os.environ[eb.ENV_SIGN] = "skip"
            called = []
            real = eb._git
            eb._git = lambda *_a, **_k: called.append(1)
            try:
                elf = os.path.join(d, "y.so")
                _write(elf, "X")
                eb._sign([elf], lambda _m: None)
                self.assertEqual(called, [])
            finally:
                eb._git = real

    def test_custom_signer_path_is_used_without_force(self):
        with _RepoDir() as d:
            dump = os.path.join(d, "argv.txt")
            tool = os.path.join(d, "my-sign.py")
            _write(tool, "import sys\n"
                         f"open({dump!r}, 'a').write(' '.join(sys.argv[1:]) + '\\n')\n"
                         "open(sys.argv[1], 'a').write('OK\\n')\n", mode=0o755)
            os.environ[eb.ENV_SIGN] = tool
            elf = os.path.join(d, "z.so")
            _write(elf, "X")
            eb._sign([elf], lambda _m: None)
            with open(dump, encoding="utf-8") as fh:
                line = fh.read().strip()
            self.assertIn(os.path.basename(elf), line)
            self.assertNotIn("--force", line)               # 自定义工具不乱加参数 ✓

    def test_clone_failure_falls_back_to_cached_tool(self):
        with _RepoDir() as d:
            os.environ[eb.ENV_BUILD_DIR] = os.path.join(d, "s")
            cached = os.path.join(d, "s", "sign-tool")
            _fake_sign_tool_repo(cached)

            def boom(_args, _log):
                raise eb.BuildError("网络不通")

            real = eb._git
            eb._git = boom
            try:
                logs = []
                tool = eb.ensure_sign_tool(log=logs.append, refresh=True)
                self.assertEqual(tool, os.path.join(cached, eb.SIGN_TOOL))
                self.assertTrue(any("缓存" in line for line in logs))
            finally:
                eb._git = real


class TestBuildAndInstall(_Env):
    def test_build_install_sign_end_to_end(self):
        with _RepoDir() as d:
            src, sdk = _fake_tree(d)
            model = os.path.join(d, "model")
            os.makedirs(model)
            signer = _fake_signer(d)
            os.environ[eb.ENV_SRC] = src
            os.environ[eb.ENV_SDK] = sdk
            os.environ[eb.ENV_SIGN] = signer
            logs = []
            lib_dir = os.path.join(d, "cann-llm-lib")
            os.environ[eb.ENV_LIB_DIR] = lib_dir
            out = eb.ensure_engine_lib(model, log=logs.append)
            self.assertTrue(os.path.isfile(out))
            self.assertEqual(out, os.path.join(lib_dir, "libcann_lm_engine.so"))
            with open(out, encoding="utf-8") as fh:
                self.assertIn("SIGNED", fh.read())      # 签名器被调用 ✓
            self.assertTrue(os.path.isfile(os.path.join(lib_dir, "libc++_shared.so")))
            self.assertTrue(os.path.isfile(os.path.join(eb.scratch_root(), "build-engine.log")))
            self.assertTrue(any("开始构建" in line for line in logs))

    def test_default_install_dir_is_repo_lib(self):
        """★最终库默认装到 <cann-llm>/lib★（不写模型目录、更不写 /tmp）"""
        with _RepoDir() as d:
            src, sdk = _fake_tree(d)
            os.environ[eb.ENV_SRC] = src
            os.environ[eb.ENV_SDK] = sdk
            plan = eb.make_plan(os.path.join(d, "model"))
            self.assertEqual(plan.install_dir, os.path.join(eb.repo_root(), "lib"))
            self.assertTrue(plan.log_path.startswith(eb.scratch_root() + os.sep))

    def test_second_call_skips_rebuild(self):
        with _RepoDir() as d:
            src, sdk = _fake_tree(d)
            model = os.path.join(d, "model")
            os.makedirs(model)
            os.environ[eb.ENV_SRC] = src
            os.environ[eb.ENV_SDK] = sdk
            os.environ[eb.ENV_SIGN] = "skip"
            os.environ[eb.ENV_LIB_DIR] = os.path.join(d, "lib")
            first = eb.ensure_engine_lib(model, log=lambda _m: None)
            mtime = os.path.getmtime(first)
            logs = []
            eb.ensure_engine_lib(model, log=logs.append)
            self.assertEqual(os.path.getmtime(first), mtime)   # 没重编 ✓
            self.assertTrue(any("跳过构建" in line for line in logs))

    def test_source_touched_forces_rebuild(self):
        with _RepoDir() as d:
            src, sdk = _fake_tree(d)
            model = os.path.join(d, "model")
            os.makedirs(model)
            os.environ[eb.ENV_SRC] = src
            os.environ[eb.ENV_SDK] = sdk
            os.environ[eb.ENV_SIGN] = "skip"
            os.environ[eb.ENV_LIB_DIR] = os.path.join(d, "lib")
            first = eb.ensure_engine_lib(model, log=lambda _m: None)
            mtime = os.path.getmtime(first)
            src_file = os.path.join(src, "cann_local_lm", "engine", "main.cpp")
            os.utime(src_file, (mtime + 10, mtime + 10))
            eb.ensure_engine_lib(model, log=lambda _m: None)
            self.assertGreater(os.path.getmtime(first), mtime)

    def test_disabled_autobuild_raises_with_hint(self):
        with _RepoDir() as d:
            src, sdk = _fake_tree(d)
            os.environ[eb.ENV_AUTOBUILD] = "0"
            os.environ[eb.ENV_SRC] = src
            os.environ[eb.ENV_SDK] = sdk
            with self.assertRaises(BackendUnavailableError) as cm:
                eb.ensure_engine_lib(os.path.join(d, "model"))
            self.assertIn(eb.ENV_AUTOBUILD, str(cm.exception))

    def test_explicit_autobuild_without_sdk_still_errors(self):
        """显式 AUTOBUILD=1 但没 SDK：仍然要报"缺 SDK"而不是静默跳过。"""
        with _RepoDir() as d:
            src, _sdk = _fake_tree(d)
            os.environ[eb.ENV_AUTOBUILD] = "1"
            os.environ[eb.ENV_SRC] = src
            with self.assertRaises(eb.BuildError) as cm:
                eb.ensure_engine_lib(os.path.join(d, "model"))
            self.assertIn(eb.ENV_SDK, str(cm.exception))

    def test_missing_signer_is_actionable(self):
        with _RepoDir() as d:
            src, sdk = _fake_tree(d)
            model = os.path.join(d, "model")
            os.makedirs(model)
            os.environ[eb.ENV_SRC] = src
            os.environ[eb.ENV_SDK] = sdk
            os.environ[eb.ENV_SIGN] = os.path.join(d, "no-such-signer.py")
            os.environ[eb.ENV_LIB_DIR] = os.path.join(d, "lib")
            with self.assertRaises(eb.BuildError) as cm:
                eb.ensure_engine_lib(model, log=lambda _m: None)
            self.assertIn(eb.ENV_SIGN, str(cm.exception))
            self.assertIn("不存在", str(cm.exception))

    def test_build_failure_reports_log_path(self):
        with _RepoDir() as d:
            src, sdk = _fake_tree(d)
            _write(os.path.join(src, eb.BUILD_SCRIPT), "#!/bin/bash\nexit 3\n", mode=0o755)
            model = os.path.join(d, "model")
            os.makedirs(model)
            os.environ[eb.ENV_SRC] = src
            os.environ[eb.ENV_SDK] = sdk
            with self.assertRaises(eb.BuildError) as cm:
                eb.ensure_engine_lib(model, log=lambda _m: None)
            self.assertIn("build-engine.log", str(cm.exception))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
