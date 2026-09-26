"""JSON Schema 的一个实用子集校验器。

为什么不直接用 ``jsonschema``：本项目的核心约束是**零第三方依赖**（目标环境
是鸿蒙设备，装包困难）。而工具参数校验只需要一个很小的子集，自己实现反而
能给出更贴合场景的错误信息。

支持的关键字：
    type / enum / const / properties / required / additionalProperties /
    minimum / maximum / exclusiveMinimum / exclusiveMaximum / multipleOf /
    minLength / maxLength / pattern / minItems / maxItems / items /
    anyOf / oneOf
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

__all__ = ["SchemaError", "validate", "ValidationResult"]


class SchemaError(ValueError):
    """参数不符合 schema。"""


class ValidationResult:
    """校验结果：把错误收集起来一次性返回，便于回给模型让它自我修正。"""

    def __init__(self) -> None:
        self.errors: List[str] = []

    @property
    def ok(self) -> bool:
        return not self.errors

    def add(self, path: str, msg: str) -> None:
        self.errors.append(f"{path or '<root>'}: {msg}")

    def message(self) -> str:
        return "; ".join(self.errors)

    def raise_if_bad(self) -> None:
        if self.errors:
            raise SchemaError(self.message())


_TYPES = {
    "object": dict,
    "array": list,
    "string": str,
    "boolean": bool,
    "null": type(None),
}


def _type_ok(value: Any, expected: str) -> bool:
    if expected == "integer":
        # bool 是 int 的子类，必须排除
        if isinstance(value, bool):
            return False
        return isinstance(value, int)
    if expected == "number":
        if isinstance(value, bool):
            return False
        return isinstance(value, (int, float))
    if expected == "boolean":
        return isinstance(value, bool)
    cls = _TYPES.get(expected)
    if cls is None:
        return True           # 未知类型不拦，保持宽松
    if cls is str:
        return isinstance(value, str)
    return isinstance(value, cls)


def _check(value: Any, schema: Dict[str, Any], path: str, res: ValidationResult) -> None:
    if not isinstance(schema, dict):
        return

    # anyOf / oneOf：至少/恰好满足一个分支
    for key in ("anyOf", "oneOf"):
        if key in schema and isinstance(schema[key], list):
            passed = 0
            for sub in schema[key]:
                sub_res = ValidationResult()
                _check(value, sub, path, sub_res)
                if sub_res.ok:
                    passed += 1
            if key == "anyOf" and passed == 0:
                res.add(path, f"不满足 anyOf 中任何一个分支")
                return
            if key == "oneOf" and passed != 1:
                res.add(path, f"需恰好满足 oneOf 中一个分支，实际 {passed} 个")
                return

    if "const" in schema and value != schema["const"]:
        res.add(path, f"必须等于 {schema['const']!r}")
        return

    if "enum" in schema and value not in schema["enum"]:
        res.add(path, f"必须是 {schema['enum']} 之一，实际 {value!r}")
        return

    stype: Optional[str] = schema.get("type")
    if isinstance(stype, list):
        if not any(_type_ok(value, t) for t in stype):
            res.add(path, f"类型应为 {'/'.join(stype)}，实际 {type(value).__name__}")
            return
    elif isinstance(stype, str):
        if not _type_ok(value, stype):
            res.add(path, f"类型应为 {stype}，实际 {type(value).__name__}（{value!r}）")
            return

    if isinstance(value, bool):
        return
    if isinstance(value, (int, float)):
        if "minimum" in schema and value < schema["minimum"]:
            res.add(path, f"不能小于 {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            res.add(path, f"不能大于 {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            res.add(path, f"必须大于 {schema['exclusiveMinimum']}")
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            res.add(path, f"必须小于 {schema['exclusiveMaximum']}")
        mult = schema.get("multipleOf")
        if mult:
            try:
                if abs((value / mult) - round(value / mult)) > 1e-9:
                    res.add(path, f"必须是 {mult} 的倍数")
            except ZeroDivisionError:
                pass
        return
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            res.add(path, f"长度不能小于 {schema['minLength']}")
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            res.add(path, f"长度不能大于 {schema['maxLength']}")
        if "pattern" in schema:
            try:
                if not re.search(schema["pattern"], value):
                    res.add(path, f"不匹配 pattern {schema['pattern']!r}")
            except re.error:
                pass
        return
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            res.add(path, f"元素个数不能少于 {schema['minItems']}")
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            res.add(path, f"元素个数不能多于 {schema['maxItems']}")
        items = schema.get("items")
        if isinstance(items, dict):
            for i, item in enumerate(value):
                _check(item, items, f"{path}[{i}]", res)
        return
    if isinstance(value, dict):
        props: Dict[str, Any] = schema.get("properties") or {}
        for req in schema.get("required") or []:
            if req not in value:
                res.add(path, f"缺少必需参数 {req!r}")
        for k, v in value.items():
            if k in props:
                _check(v, props[k], f"{path}.{k}" if path else k, res)
            else:
                extra = schema.get("additionalProperties", True)
                if extra is False:
                    res.add(path, f"不认识的参数 {k!r}（allowed: {sorted(props)}）")
                elif isinstance(extra, dict):
                    _check(v, extra, f"{path}.{k}" if path else k, res)


def validate(value: Any, schema: Optional[Dict[str, Any]]) -> ValidationResult:
    """按 schema 校验值，返回结果（不抛异常）。"""
    res = ValidationResult()
    if schema:
        _check(value, schema, "", res)
    return res
