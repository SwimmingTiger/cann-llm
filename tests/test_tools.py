import json
import unittest

from cann_llm.tools import ToolError, ToolRegistry, default_registry, validate
from cann_llm.tools.builtin import register_into
from cann_llm.tools.schema import SchemaError


class TestSchema(unittest.TestCase):
    def test_types(self):
        for value, t, ok in [(1, "integer", True), (1.5, "integer", False),
                             (1, "number", True), (True, "integer", False),
                             ("a", "string", True), ([1], "array", True),
                             ({"a": 1}, "object", True), (None, "null", True),
                             (True, "boolean", True), (1, "boolean", False)]:
            r = validate(value, {"type": t})
            self.assertEqual(r.ok, ok, f"{value!r} as {t}")

    def test_required_and_unknown(self):
        schema = {"type": "object", "properties": {"a": {"type": "integer"}},
                  "required": ["a"], "additionalProperties": False}
        self.assertFalse(validate({}, schema).ok)
        self.assertTrue(validate({"a": 1}, schema).ok)
        self.assertIn("不认识的参数", validate({"a": 1, "b": 2}, schema).message())

    def test_constraints(self):
        cases = [
            ({"n": 0}, {"minimum": 1}, False),
            ({"n": 5}, {"minimum": 1, "maximum": 10}, True),
            ({"n": 11}, {"maximum": 10}, False),
            ({"n": 4}, {"multipleOf": 2}, True),
            ({"s": "ab"}, {"minLength": 3}, False),
            ({"s": "abc"}, {"pattern": "^a"}, True),
            ({"a": [1, 2]}, {"minItems": 3}, False),
        ]
        for value, extra, ok in cases:
            key = list(value)[0]
            schema = {"type": "object", "properties": {key: {"type": (
                "array" if isinstance(value[key], list) else
                "string" if isinstance(value[key], str) else "integer"), **extra}}}
            self.assertEqual(validate(value, schema).ok, ok, f"{value} {extra}")

    def test_enum_and_anyof(self):
        self.assertFalse(validate("c", {"enum": ["a", "b"]}).ok)
        self.assertTrue(validate("a", {"enum": ["a", "b"]}).ok)
        self.assertTrue(validate(1, {"anyOf": [{"type": "integer"}, {"type": "string"}]}).ok)
        self.assertFalse(validate(1.5, {"anyOf": [{"type": "integer"}, {"type": "string"}]}).ok)

    def test_nested_array_items(self):
        r = validate([1, "a"], {"type": "array", "items": {"type": "integer"}})
        self.assertFalse(r.ok)
        self.assertIn("[1]", r.message())

    def test_collects_all_errors(self):
        schema = {"type": "object", "required": ["a", "b"],
                  "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}}}
        r = validate({}, schema)
        self.assertEqual(len(r.errors), 2)

    def test_raise_if_bad(self):
        with self.assertRaises(SchemaError):
            validate({}, {"type": "object", "required": ["x"]}).raise_if_bad()


class TestRegistry(unittest.TestCase):
    def _reg(self):
        r = ToolRegistry()

        @r.register("echo", "回显", {"type": "object",
                                     "properties": {"s": {"type": "string"}},
                                     "required": ["s"]})
        def _echo(s):
            return s

        @r.register("boom", "总是失败", {"type": "object"}, dangerous=True)
        def _boom():
            raise ToolError("炸了")

        return r

    def test_schema_shape(self):
        s = self._reg().openai_schemas()[0]
        self.assertEqual(s["type"], "function")
        self.assertEqual(s["function"]["name"], "echo")

    def test_call_validates(self):
        r = self._reg()
        self.assertEqual(r.call("echo", {"s": "hi"}), "hi")
        with self.assertRaises(SchemaError):
            r.call("echo", {})
        with self.assertRaises(ToolError):
            r.call("nope", {})

    def test_dangerous_not_selected_by_default(self):
        r = self._reg()
        self.assertEqual(r.select().names(), ["echo"])
        self.assertEqual(r.select(["boom"], include_dangerous=True).names(), ["boom"])
        with self.assertRaises(PermissionError):
            r.select(["boom"])

    def test_select_unknown(self):
        with self.assertRaises(KeyError):
            self._reg().select(["nope"])

    def test_register_rejects_empty_name(self):
        with self.assertRaises(ValueError):
            ToolRegistry().add(__import__("cann_llm.tools", fromlist=["Tool"]).Tool(
                "", "", {}, lambda: None))


class TestBuiltin(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry()
        register_into(self.reg)

    def test_names(self):
        self.assertEqual(self.reg.names(),
                         ["calculator", "get_current_time", "http_get"])

    def test_default_registry_has_builtins(self):
        self.assertIn("calculator", default_registry().names())

    def test_time(self):
        out = json.loads(self.reg.call("get_current_time", {}))
        self.assertIn("datetime", out)
        out0 = json.loads(self.reg.call("get_current_time", {"timezone_offset_hours": 0}))
        self.assertNotEqual(out["iso8601"], out0["iso8601"])

    def test_calculator_ok(self):
        for expr, want in [("(23*7+11)/4", 43.0), ("2**10", 1024),
                           ("round(3.14159, 2)", 3.14), ("max(1,5,3)", 5)]:
            got = json.loads(self.reg.call("calculator", {"expression": expr}))["result"]
            self.assertEqual(got, want, expr)

    def test_calculator_blocks_dangerous(self):
        for bad in ["__import__('os').system('ls')", "open('f')", "x+1",
                    "().__class__", "1/0", "2**9999", "f'{1}'"]:
            with self.assertRaises(ToolError, msg=bad):
                self.reg.call("calculator", {"expression": bad})

    def test_calculator_length_limit(self):
        with self.assertRaises(SchemaError):
            self.reg.call("calculator", {"expression": "1+" * 400 + "1"})

    def test_http_get_ssrf(self):
        for url in ["file:///etc/passwd", "http://127.0.0.1/x", "http://10.0.0.1/",
                    "http://localhost/", "http://[::1]/", "ftp://x/"]:
            with self.assertRaises(ToolError, msg=url):
                self.reg.call("http_get", {"url": url})

    def test_http_get_schema(self):
        with self.assertRaises(SchemaError):
            self.reg.call("http_get", {})

    def test_http_get_is_dangerous(self):
        self.assertTrue(self.reg.get("http_get").dangerous)
        self.assertFalse(self.reg.get("calculator").dangerous)


if __name__ == "__main__":
    unittest.main()
