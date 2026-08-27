"""Mechanical JSON-schema and decoded-value helpers without domain policy."""

from __future__ import annotations

import inspect
from typing import Any, Literal, Mapping, Union, get_args, get_origin, get_type_hints, is_typeddict


def annotation_schema(annotation: Any) -> tuple[dict[str, Any], bool]:
    """Return a JSON schema and whether the annotation is nullable."""
    if annotation is Any or annotation is inspect.Parameter.empty:
        raise TypeError("structured values need supported annotations")
    if is_typeddict(annotation):
        hints = get_type_hints(annotation)
        properties = {}
        for name, child_annotation in hints.items():
            child_schema, _nullable = annotation_schema(child_annotation)
            properties[name] = child_schema
        required_keys = sorted(getattr(annotation, "__required_keys__", hints))
        return {
            "type": "object",
            "properties": properties,
            "required": required_keys,
            "additionalProperties": False,
        }, False
    origin = get_origin(annotation)
    args = get_args(annotation)
    if origin in (Union, getattr(__import__("types"), "UnionType", object)):
        if len(args) != 2 or type(None) not in args:
            raise TypeError("only T | None annotations are supported")
        other = args[0] if args[1] is type(None) else args[1]
        schema, _ = annotation_schema(other)
        return {"anyOf": [schema, {"type": "null"}]}, True
    if origin is Literal:
        values = list(args)
        if not values:
            raise TypeError("Literal annotations cannot be empty")
        type_name = (
            "string"
            if all(isinstance(value, str) for value in values)
            else "integer"
            if all(isinstance(value, int) and not isinstance(value, bool) for value in values)
            else "number"
            if all(isinstance(value, (int, float)) and not isinstance(value, bool) for value in values)
            else "boolean"
            if all(isinstance(value, bool) for value in values)
            else None
        )
        if type_name is None:
            raise TypeError("Literal values must share a JSON scalar type")
        return {"type": type_name, "enum": values}, False
    if annotation is str:
        return {"type": "string"}, False
    if annotation is bool:
        return {"type": "boolean"}, False
    if annotation is int:
        return {"type": "integer"}, False
    if annotation is float:
        return {"type": "number"}, False
    if annotation in (list, dict):
        return (
            ({"type": "array", "items": {}}, False)
            if annotation is list
            else ({"type": "object", "additionalProperties": True}, False)
        )
    if origin is list:
        item = args[0] if args else Any
        if item is Any:
            return {"type": "array", "items": {}}, False
        item_schema, _ = annotation_schema(item)
        return {"type": "array", "items": item_schema}, False
    if origin is dict:
        if len(args) != 2 or args[0] is not str:
            raise TypeError("structured mappings must be dict[str, T]")
        if args[1] is Any:
            return {"type": "object", "additionalProperties": True}, False
        value_schema, _ = annotation_schema(args[1])
        return {"type": "object", "additionalProperties": value_schema}, False
    raise TypeError(f"unsupported structured annotation {annotation!r}")


def validate_value(annotation: Any, value: Any, path: str) -> None:
    """Validate one decoded JSON value against a supported annotation."""
    if is_typeddict(annotation):
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise TypeError(f"structured value '{path}' must be an object")
        hints = get_type_hints(annotation)
        required_keys = set(getattr(annotation, "__required_keys__", hints))
        unknown = set(value) - set(hints)
        missing = required_keys - set(value)
        if unknown:
            raise TypeError(f"structured value '{path}' has unknown keys {sorted(unknown)}")
        if missing:
            raise TypeError(f"structured value '{path}' is missing keys {sorted(missing)}")
        for key, child in value.items():
            validate_value(hints[key], child, f"{path}.{key}")
        return
    origin = get_origin(annotation)
    args = get_args(annotation)
    union_type = getattr(__import__("types"), "UnionType", object)
    if origin in (Union, union_type):
        if type(None) in args and value is None:
            return
        non_null = [candidate for candidate in args if candidate is not type(None)]
        if len(non_null) == 1:
            validate_value(non_null[0], value, path)
            return
        raise TypeError(f"unsupported union for structured value '{path}'")
    if origin is Literal:
        if not any(type(value) is type(item) and value == item for item in args):
            raise TypeError(f"structured value '{path}' is not one of the declared literal values")
        return
    if annotation is str:
        if not isinstance(value, str):
            raise TypeError(f"structured value '{path}' must be a string")
        return
    if annotation is bool:
        if not isinstance(value, bool):
            raise TypeError(f"structured value '{path}' must be a boolean")
        return
    if annotation is int:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"structured value '{path}' must be an integer")
        return
    if annotation is float:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(f"structured value '{path}' must be a number")
        return
    if annotation in (list, dict):
        if not isinstance(value, annotation):
            raise TypeError(f"structured value '{path}' has the wrong container type")
        return
    if origin is list:
        if not isinstance(value, list):
            raise TypeError(f"structured value '{path}' must be a list")
        if args and args[0] is not Any:
            for index, item in enumerate(value):
                validate_value(args[0], item, f"{path}[{index}]")
        return
    if origin is dict:
        if not isinstance(value, dict) or any(not isinstance(key, str) for key in value):
            raise TypeError(f"structured value '{path}' must be a string-keyed object")
        if len(args) == 2 and args[1] is not Any:
            for key, item in value.items():
                validate_value(args[1], item, f"{path}.{key}")
        return
    if annotation is Any:
        return
    raise TypeError(f"unsupported structured annotation for '{path}'")


def json_compatible(value: Any) -> Any:
    """Convert a value to the supported JSON-compatible surface."""
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return {str(key): json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_compatible(item) for item in value]
    raise TypeError(f"value of type {type(value).__name__} is not JSON-compatible")


__all__ = [
    "annotation_schema",
    "json_compatible",
    "validate_value",
]
