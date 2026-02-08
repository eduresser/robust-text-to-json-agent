from __future__ import annotations

import copy
import json
import math
import re
from datetime import datetime
from typing import Any, Optional
from urllib.parse import urlparse


class SchemaPatchChecker:
    _ANY_SCHEMA: dict[str, Any] = {"__any": True}

    @staticmethod
    def _is_object(x: Any) -> bool:
        return isinstance(x, dict)

    @staticmethod
    def _clone(obj: Any) -> Any:
        return copy.deepcopy(obj)

    @classmethod
    def _deep_equal(cls, a: Any, b: Any) -> bool:
        if a is b:
            return True
        if type(a) is not type(b):
            # special: int/float comparison
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                return a == b
            return False
        if a is None:
            return b is None
        if isinstance(a, list) and isinstance(b, list):
            if len(a) != len(b):
                return False
            return all(cls._deep_equal(a[i], b[i]) for i in range(len(a)))
        if isinstance(a, dict) and isinstance(b, dict):
            ak = set(a.keys())
            bk = set(b.keys())
            if ak != bk:
                return False
            return all(cls._deep_equal(a[k], b[k]) for k in ak)
        return a == b

    @staticmethod
    def _decode_pointer_token(token: str) -> str:
        return token.replace("~1", "/").replace("~0", "~")

    @classmethod
    def _parse_json_pointer(cls, path: str) -> list[str]:
        if path == "" or path == "/":
            return []
        if not path.startswith("/"):
            raise ValueError(
                f'Invalid JSON Pointer (must start with "/"): {path}'
            )
        return [cls._decode_pointer_token(t) for t in path.split("/")[1:]]

    @classmethod
    def _get_at(cls, doc: Any, tokens: list[str]) -> dict[str, Any]:
        cur = doc
        for t in tokens:
            if cur is None:
                return {"exists": False, "value": None}
            if isinstance(cur, list):
                if t == "-":
                    return {"exists": False, "value": None}
                try:
                    idx = int(t)
                except (ValueError, TypeError):
                    return {"exists": False, "value": None}
                if idx < 0 or idx >= len(cur):
                    return {"exists": False, "value": None}
                cur = cur[idx]
            elif cls._is_object(cur):
                if t not in cur:
                    return {"exists": False, "value": None}
                cur = cur[t]
            else:
                return {"exists": False, "value": None}
        return {"exists": True, "value": cur}

    @classmethod
    def _get_parent_and_key(
        cls, doc: Any, tokens: list[str]
    ) -> dict[str, Any]:
        if len(tokens) == 0:
            return {"parent": None, "key": None}
        key = tokens[-1]
        parent_tokens = tokens[:-1]
        res = cls._get_at(doc, parent_tokens)
        if not res["exists"]:
            return {"parent": None, "key": key}
        return {"parent": res["value"], "key": key}

    @staticmethod
    def _resolve_ref(schema: Any, root_schema: Any) -> Any:
        """Resolve JSON Schema $ref references against the root schema.

        Handles nested $ref chains (up to 10 levels to prevent infinite loops).
        """
        if not isinstance(schema, dict) or "$ref" not in schema:
            return schema
        seen: set[str] = set()
        current = schema
        for _ in range(10):  # max depth to prevent infinite loops
            if not isinstance(current, dict) or "$ref" not in current:
                return current
            ref = current["$ref"]
            if ref in seen:
                return current  # circular ref, stop
            seen.add(ref)
            if not ref.startswith("#/"):
                return current  # can't resolve external refs
            parts = ref[2:].split("/")
            resolved = root_schema
            for part in parts:
                part = part.replace("~1", "/").replace("~0", "~")
                if isinstance(resolved, dict):
                    resolved = resolved.get(part)
                else:
                    return current  # can't navigate further
            if resolved is None:
                return current
            current = resolved
        return current

    @classmethod
    def _inline_refs(cls, schema: Any, root_schema: Any, _seen: Optional[set] = None) -> Any:
        """Recursively resolve all $ref in a schema, inlining definitions.

        Handles circular references by stopping after the first encounter.
        """
        if _seen is None:
            _seen = set()
        if not isinstance(schema, dict):
            return schema

        # Resolve $ref first
        if "$ref" in schema:
            ref = schema["$ref"]
            if ref in _seen:
                # Circular ref — return a permissive schema to avoid infinite loop
                return cls._ANY_SCHEMA
            resolved = cls._resolve_ref(schema, root_schema)
            if resolved is schema:
                return schema  # couldn't resolve
            new_seen = _seen | {ref}
            return cls._inline_refs(resolved, root_schema, new_seen)

        # Recurse into known schema keywords
        result = {}
        for key, val in schema.items():
            if key == "properties" and isinstance(val, dict):
                result[key] = {
                    k: cls._inline_refs(v, root_schema, _seen)
                    for k, v in val.items()
                }
            elif key == "items" and isinstance(val, dict):
                result[key] = cls._inline_refs(val, root_schema, _seen)
            elif key == "additionalProperties" and isinstance(val, dict):
                result[key] = cls._inline_refs(val, root_schema, _seen)
            elif key in ("anyOf", "oneOf", "allOf") and isinstance(val, list):
                result[key] = [
                    cls._inline_refs(item, root_schema, _seen)
                    for item in val
                ]
            elif key == "definitions" and isinstance(val, dict):
                # Don't recurse into definitions (they get resolved via $ref)
                result[key] = val
            else:
                result[key] = val
        return result

    @staticmethod
    def _normalize_type(type_val: Any) -> Optional[list[str]]:
        if not type_val:
            return None
        if isinstance(type_val, list):
            return type_val
        return [type_val]

    @staticmethod
    def _type_of_instance(x: Any) -> str:
        if x is None:
            return "null"
        if isinstance(x, bool):
            return "boolean"
        if isinstance(x, list):
            return "array"
        if isinstance(x, dict):
            return "object"
        if isinstance(x, int):
            return "integer"
        if isinstance(x, float):
            # A float with no fractional part is also an integer
            if x == int(x) and not (math.isinf(x) or math.isnan(x)):
                return "integer"
            return "number"
        if isinstance(x, str):
            return "string"
        return type(x).__name__

    @classmethod
    def _validate_format(cls, fmt: str, value: Any) -> bool:
        if not isinstance(value, str):
            return False

        if fmt == "email":
            return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", value))

        if fmt == "idn-email":
            return bool(re.match(r"^[^\s@]+@[^\s@]+\.[^\s@]+$", value))

        if fmt == "date":
            m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", value)
            if not m:
                return False
            y, mo, d = int(m.group(1)), int(m.group(2)), int(m.group(3))
            if mo < 1 or mo > 12:
                return False
            if d < 1 or d > 31:
                return False
            try:
                dt = datetime(y, mo, d)
                return dt.month == mo and dt.day == d and y >= 0
            except (ValueError, OverflowError):
                return False

        if fmt == "time":
            m = re.match(
                r"^(\d{2}):(\d{2}):(\d{2})(\.\d+)?(Z|[+-]\d{2}:\d{2})?$",
                value,
            )
            if not m:
                return False
            h, mn, sec = int(m.group(1)), int(m.group(2)), int(m.group(3))
            return 0 <= h <= 23 and 0 <= mn <= 59 and 0 <= sec <= 60

        if fmt == "date-time":
            m = re.match(
                r"^(\d{4})-(\d{2})-(\d{2})[Tt](\d{2}):(\d{2}):(\d{2})"
                r"(\.\d+)?([Zz]|[+-]\d{2}:\d{2})$",
                value,
            )
            if not m:
                return False
            y = int(m.group(1))
            mo = int(m.group(2))
            d = int(m.group(3))
            h = int(m.group(4))
            mn = int(m.group(5))
            sec = int(m.group(6))
            if mo < 1 or mo > 12:
                return False
            if d < 1 or d > 31:
                return False
            if h < 0 or h > 23:
                return False
            if mn < 0 or mn > 59:
                return False
            if sec < 0 or sec > 60:
                return False
            try:
                datetime.fromisoformat(value.replace("Z", "+00:00").replace("z", "+00:00"))
            except (ValueError, OverflowError):
                return False
            return y >= 0

        if fmt == "duration":
            m = re.match(
                r"^P(?:(\d+)Y)?(?:(\d+)M)?(?:(\d+)D)?"
                r"(?:T(?:(\d+)H)?(?:(\d+)M)?(?:(\d+(?:\.\d+)?)S)?)?$"
                r"|^P(\d+)W$",
                value,
            )
            if not m:
                return False
            if value in ("P", "PT"):
                return False
            return True

        if fmt == "uri":
            try:
                parsed = urlparse(value)
                return bool(parsed.scheme) and len(parsed.scheme) >= 1
            except Exception:
                return False

        if fmt == "uri-reference":
            if value == "" or value.startswith("/") or value.startswith("#") or value.startswith("?"):
                return True
            try:
                urlparse(value)
                return True
            except Exception:
                return bool(
                    re.match(r"^[^:/?#]*(?:/[^?#]*)?(?:\?[^#]*)?(?:#.*)?$", value)
                )

        if fmt == "uri-template":
            if re.search(r"\{[^}]*\{|\}[^{]*\}", value):
                return False
            template_var_re = (
                r"\{[+#./;?&]?[a-zA-Z0-9_]+"
                r"(?::[1-9][0-9]*|\*)?"
                r"(?:,[a-zA-Z0-9_]+(?::[1-9][0-9]*|\*)?)*\}"
            )
            without_templates = re.sub(template_var_re, "", value)
            return not re.search(r"[{}]", without_templates)

        if fmt in ("iri", "iri-reference"):
            if fmt == "iri-reference":
                if value == "" or value.startswith("/") or value.startswith("#") or value.startswith("?"):
                    return True
            try:
                parsed = urlparse(value)
                return bool(parsed.scheme) and len(parsed.scheme) >= 1
            except Exception:
                return False

        if fmt == "hostname":
            if len(value) > 253:
                return False
            labels = value.split(".")
            for label in labels:
                if len(label) == 0 or len(label) > 63:
                    return False
                if not re.match(r"^[a-zA-Z0-9]([a-zA-Z0-9-]*[a-zA-Z0-9])?$", label) and not re.match(
                    r"^[a-zA-Z0-9]$", label
                ):
                    return False
            return True

        if fmt == "idn-hostname":
            if len(value) > 253:
                return False
            labels = value.split(".")
            for label in labels:
                if len(label) == 0 or len(label) > 63:
                    return False
                if label.startswith("-") or label.endswith("-"):
                    return False
            return True

        if fmt == "ipv4":
            parts = value.split(".")
            if len(parts) != 4:
                return False
            for part in parts:
                if not re.match(r"^\d{1,3}$", part):
                    return False
                num = int(part)
                if num < 0 or num > 255:
                    return False
                if len(part) > 1 and part.startswith("0"):
                    return False
            return True

        if fmt == "ipv6":
            ipv6_re = (
                r"^(?:(?:[0-9a-fA-F]{1,4}:){7}[0-9a-fA-F]{1,4}"
                r"|(?:[0-9a-fA-F]{1,4}:){1,7}:"
                r"|(?:[0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}"
                r"|(?:[0-9a-fA-F]{1,4}:){1,5}(?::[0-9a-fA-F]{1,4}){1,2}"
                r"|(?:[0-9a-fA-F]{1,4}:){1,4}(?::[0-9a-fA-F]{1,4}){1,3}"
                r"|(?:[0-9a-fA-F]{1,4}:){1,3}(?::[0-9a-fA-F]{1,4}){1,4}"
                r"|(?:[0-9a-fA-F]{1,4}:){1,2}(?::[0-9a-fA-F]{1,4}){1,5}"
                r"|[0-9a-fA-F]{1,4}:(?::[0-9a-fA-F]{1,4}){1,6}"
                r"|:(?:(?::[0-9a-fA-F]{1,4}){1,7}|:)"
                r"|fe80:(?::[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]+"
                r"|::(?:ffff(?::0{1,4})?:)?(?:(?:25[0-5]|(?:2[0-4]|1?[0-9])?[0-9])\.){3}"
                r"(?:25[0-5]|(?:2[0-4]|1?[0-9])?[0-9])"
                r"|(?:[0-9a-fA-F]{1,4}:){1,4}:(?:(?:25[0-5]|(?:2[0-4]|1?[0-9])?[0-9])\.){3}"
                r"(?:25[0-5]|(?:2[0-4]|1?[0-9])?[0-9]))$"
            )
            return bool(re.match(ipv6_re, value))

        if fmt == "uuid":
            return bool(
                re.match(
                    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[1-5][0-9a-fA-F]{3}"
                    r"-[89abAB][0-9a-fA-F]{3}-[0-9a-fA-F]{12}$",
                    value,
                )
            )

        if fmt == "json-pointer":
            if value == "":
                return True
            if not value.startswith("/"):
                return False
            return not re.search(r"~(?![01])", value)

        if fmt == "relative-json-pointer":
            m = re.match(r"^(0|[1-9][0-9]*)(#|/.*)?\Z", value)
            if not m:
                return False
            suffix = m.group(2) or ""
            if suffix and suffix != "#" and suffix.startswith("/"):
                return not re.search(r"~(?![01])", suffix)
            return True

        if fmt == "regex":
            try:
                re.compile(value)
                return True
            except re.error:
                return False

        # Unknown format: pass
        return True

    @classmethod
    def _validate_instance(
        cls,
        schema: Any,
        instance: Any,
        at_pointer: str = "",
    ) -> list[dict[str, str]]:
        errors: list[dict[str, str]] = []

        def push_err(msg: str) -> None:
            errors.append({"pointer": at_pointer or "/", "message": msg})

        if schema is None or schema is True:
            return errors
        if schema is False:
            push_err('schema "false" does not accept any value')
            return errors
        if isinstance(schema, dict) and schema.get("__any"):
            return errors

        # anyOf
        if isinstance(schema, dict) and "anyOf" in schema:
            any_pass = any(
                len(cls._validate_instance(s, instance, at_pointer)) == 0
                for s in schema["anyOf"]
            )
            if not any_pass:
                push_err("failed in anyOf (no alternative accepted the value)")
            return errors

        # enum
        if isinstance(schema, dict) and "enum" in schema:
            ok = any(cls._deep_equal(v, instance) for v in schema["enum"])
            if not ok:
                push_err(f"value is not in enum: {json.dumps(instance)}")

        # type check
        allowed_types = cls._normalize_type(schema.get("type") if isinstance(schema, dict) else None)
        if allowed_types:
            inst_type = cls._type_of_instance(instance)
            # JSON Schema: "integer" is a subtype of "number"
            type_matches = (
                inst_type in allowed_types
                or (inst_type == "integer" and "number" in allowed_types)
            )
            if not type_matches:
                push_err(
                    f"invalid type: expected {' | '.join(allowed_types)}, "
                    f"received {inst_type}"
                )
                return errors

        inst_type = cls._type_of_instance(instance)

        # string validations
        if inst_type == "string" and isinstance(schema, dict):
            if schema.get("pattern"):
                if not re.search(schema["pattern"], instance):
                    push_err(
                        f"string does not match pattern: {schema['pattern']}"
                    )
            if schema.get("format"):
                if not cls._validate_format(schema["format"], instance):
                    push_err(
                        f"string does not respect format: {schema['format']}"
                    )

        # number/integer validations
        if inst_type in ("number", "integer") and isinstance(schema, dict):
            if isinstance(schema.get("minimum"), (int, float)):
                if not (instance >= schema["minimum"]):
                    push_err(f"number < minimum ({schema['minimum']})")
            if isinstance(schema.get("maximum"), (int, float)):
                if not (instance <= schema["maximum"]):
                    push_err(f"number > maximum ({schema['maximum']})")

        # array validations
        if inst_type == "array" and isinstance(schema, dict):
            if schema.get("items"):
                for i, item in enumerate(instance):
                    child_ptr = f"{at_pointer}/{i}"
                    errors.extend(
                        cls._validate_instance(schema["items"], item, child_ptr)
                    )

        # object validations
        if inst_type == "object" and isinstance(schema, dict):
            props = schema.get("properties", {})
            req = schema.get("required", [])

            for r in req:
                if r not in instance:
                    errors.append(
                        {
                            "pointer": at_pointer or "/",
                            "message": f"required field missing: {r}",
                        }
                    )

            for k, v in instance.items():
                if k in props:
                    errors.extend(
                        cls._validate_instance(
                            props[k], v, f"{at_pointer}/{k}"
                        )
                    )
                else:
                    ap = schema.get("additionalProperties")
                    if ap is False:
                        errors.append(
                            {
                                "pointer": f"{at_pointer}/{k}" or "/",
                                "message": (
                                    f"property not allowed "
                                    f"(additionalProperties=false): {k}"
                                ),
                            }
                        )
                    elif ap and ap is not True:
                        errors.extend(
                            cls._validate_instance(
                                ap, v, f"{at_pointer}/{k}"
                            )
                        )

        return errors

    @classmethod
    def _build_base_doc_from_schema(cls, schema: Any) -> dict[str, Any]:
        if not schema or not isinstance(schema, dict) or schema.get("type") != "object":
            return {}
        props = schema.get("properties", {})
        required = schema.get("required", [])
        base: dict[str, Any] = {}

        for key in required:
            prop_schema = props.get(key, {})
            type_val = prop_schema.get("type")
            if isinstance(type_val, list):
                type_val = type_val[0] if type_val else None
            if type_val == "array":
                base[key] = []
            elif type_val == "object":
                base[key] = {}
            else:
                base[key] = None

        return base

    @classmethod
    def _schema_candidates_for_property(
        cls, schema: Any, prop_name: str
    ) -> list[Any]:
        if schema is None or schema is True:
            return [cls._ANY_SCHEMA]
        if schema is False:
            return []
        if isinstance(schema, dict) and schema.get("__any"):
            return [cls._ANY_SCHEMA]

        if isinstance(schema, dict) and "anyOf" in schema:
            all_candidates: list[Any] = []
            for s in schema["anyOf"]:
                all_candidates.extend(
                    cls._schema_candidates_for_property(s, prop_name)
                )
            return all_candidates if all_candidates else []

        if not isinstance(schema, dict):
            return [cls._ANY_SCHEMA]

        props = schema.get("properties", {})
        if prop_name in props:
            return [props[prop_name]]

        ap = schema.get("additionalProperties")
        if ap is False:
            return []
        if ap is True or ap is None:
            return [cls._ANY_SCHEMA]
        return [ap]

    @classmethod
    def _schema_candidates_for_index(cls, schema: Any) -> list[Any]:
        if schema is None or schema is True:
            return [cls._ANY_SCHEMA]
        if schema is False:
            return []
        if isinstance(schema, dict) and schema.get("__any"):
            return [cls._ANY_SCHEMA]

        if isinstance(schema, dict) and "anyOf" in schema:
            all_candidates: list[Any] = []
            for s in schema["anyOf"]:
                all_candidates.extend(cls._schema_candidates_for_index(s))
            return all_candidates if all_candidates else []

        if isinstance(schema, dict) and schema.get("items"):
            return [schema["items"]]
        return [cls._ANY_SCHEMA]

    @classmethod
    def _schema_at_pointer_candidates(
        cls, root_schema: Any, tokens: list[str]
    ) -> list[Any]:
        candidates = [root_schema]
        for t in tokens:
            next_candidates: list[Any] = []
            for s in candidates:
                s = cls._resolve_ref(s, root_schema)
                types = cls._normalize_type(
                    s.get("type") if isinstance(s, dict) else None
                )
                could_be_array = s and (
                    (isinstance(s, dict) and s.get("__any"))
                    or not types
                    or "array" in types
                )
                could_be_object = s and (
                    (isinstance(s, dict) and s.get("__any"))
                    or not types
                    or "object" in types
                )

                if t != "-" and re.match(r"^[0-9]+$", t) and could_be_array:
                    resolved = [
                        cls._resolve_ref(c, root_schema)
                        for c in cls._schema_candidates_for_index(s)
                    ]
                    next_candidates.extend(resolved)
                    continue
                if could_be_object:
                    resolved = [
                        cls._resolve_ref(c, root_schema)
                        for c in cls._schema_candidates_for_property(s, t)
                    ]
                    next_candidates.extend(resolved)

            candidates = next_candidates if next_candidates else [cls._ANY_SCHEMA]
        # Final resolution of candidates
        return [cls._resolve_ref(c, root_schema) for c in candidates]

    @classmethod
    def _apply_json_patch(
        cls, doc: Any, patch_ops: list[dict[str, Any]]
    ) -> Any:
        current = cls._clone(doc)

        def ensure_container_for_add(
            parent: Any, key: str, value: Any
        ) -> Any:
            nonlocal current
            if parent is None:
                return cls._clone(value)
            if isinstance(parent, list):
                idx = len(parent) if key == "-" else int(key)
                if not isinstance(idx, int) or idx < 0 or idx > len(parent):
                    raise ValueError(f"add in array: invalid index: {key}")
                parent.insert(idx, value)
                return current
            if not cls._is_object(parent):
                raise ValueError("add: parent is not object/array at path")
            parent[key] = value
            return current

        def set_value(
            parent: Any, key: str, value: Any, op_path: str
        ) -> Any:
            nonlocal current
            if parent is None:
                return cls._clone(value)
            if isinstance(parent, list):
                idx = int(key)
                if idx < 0 or idx >= len(parent):
                    raise ValueError(
                        f"replace failed: {op_path} does not exist"
                    )
                parent[idx] = value
                return current
            if not cls._is_object(parent):
                raise ValueError(
                    "replace: parent is not object/array at path"
                )
            if key not in parent:
                raise ValueError(
                    f"replace failed: {op_path} does not exist"
                )
            parent[key] = value
            return current

        def remove_value(
            parent: Any, key: str, op_path: str
        ) -> Any:
            nonlocal current
            if parent is None:
                return None
            if isinstance(parent, list):
                idx = int(key)
                if idx < 0 or idx >= len(parent):
                    raise ValueError(
                        f"remove failed: {op_path} does not exist"
                    )
                parent.pop(idx)
                return current
            if not cls._is_object(parent):
                raise ValueError(
                    "remove: parent is not object/array at path"
                )
            if key not in parent:
                raise ValueError(
                    f"remove failed: {op_path} does not exist"
                )
            del parent[key]
            return current

        for op in patch_ops:
            tokens = cls._parse_json_pointer(op["path"])
            pk = cls._get_parent_and_key(current, tokens)
            parent, key = pk["parent"], pk["key"]

            op_name = op["op"]

            if op_name == "add":
                current = ensure_container_for_add(
                    parent, key, cls._clone(op.get("value"))
                )
            elif op_name == "replace":
                current = set_value(
                    parent, key, cls._clone(op.get("value")), op["path"]
                )
            elif op_name == "remove":
                current = remove_value(parent, key, op["path"])
            elif op_name == "test":
                result = cls._get_at(current, tokens)
                if not result["exists"]:
                    raise ValueError(
                        f"test failed: {op['path']} does not exist"
                    )
                if not cls._deep_equal(result["value"], op.get("value")):
                    raise ValueError(
                        f"test failed: value differs at {op['path']}"
                    )
            elif op_name == "copy":
                from_tokens = cls._parse_json_pointer(op["from"])
                src = cls._get_at(current, from_tokens)
                if not src["exists"]:
                    raise ValueError(
                        f"copy failed: from={op['from']} does not exist"
                    )
                current = ensure_container_for_add(
                    parent, key, cls._clone(src["value"])
                )
            elif op_name == "move":
                from_tokens = cls._parse_json_pointer(op["from"])
                src = cls._get_at(current, from_tokens)
                if not src["exists"]:
                    raise ValueError(
                        f"move failed: from={op['from']} does not exist"
                    )
                from_pk = cls._get_parent_and_key(current, from_tokens)
                current = remove_value(
                    from_pk["parent"], from_pk["key"], op["from"]
                )
                dst_tokens = cls._parse_json_pointer(op["path"])
                dst_pk = cls._get_parent_and_key(current, dst_tokens)
                current = ensure_container_for_add(
                    dst_pk["parent"], dst_pk["key"], cls._clone(src["value"])
                )
            else:
                raise ValueError(f"Operation not supported: {op_name}")

        return current

    @classmethod
    def validate_patch_ops_against_schema(
        cls,
        root_schema: Any,
        initial_doc: Any,
        patch_ops: list[dict[str, Any]],
    ) -> dict[str, Any]:
        errors: list[dict[str, Any]] = []
        # Pre-resolve all $ref in the schema so all validation works correctly
        if root_schema and isinstance(root_schema, dict):
            root_schema = cls._inline_refs(root_schema, root_schema)
        base_doc = cls._build_base_doc_from_schema(root_schema) if root_schema else {}
        merged = {**base_doc, **(initial_doc or {})} if isinstance(initial_doc, dict) else (initial_doc or {})
        doc = cls._clone(merged)

        def add_err(
            op_index: int,
            op: Any,
            pointer: str,
            message: str,
        ) -> None:
            errors.append(
                {
                    "opIndex": op_index,
                    "op": op,
                    "pointer": pointer or (op.get("path", "/") if isinstance(op, dict) else "/"),
                    "message": message,
                }
            )

        def op_needs_value(op_name: str) -> bool:
            return op_name in ("add", "replace", "test")

        if root_schema is None:

            def ensure_parent_chain_for_add(
                base: Any, tokens: list[str]
            ) -> Any:
                if len(tokens) <= 1:
                    return base
                cur = base
                for i in range(len(tokens) - 1):
                    t = tokens[i]
                    nxt = tokens[i + 1]
                    next_is_index = nxt == "-" or re.match(r"^[0-9]+$", nxt)

                    if isinstance(cur, list):
                        idx = len(cur) if t == "-" else int(t)
                        if not isinstance(idx, int) or idx < 0 or idx > len(cur):
                            raise ValueError(
                                f"add in array: invalid index: {t}"
                            )
                        if idx == len(cur):
                            cur.append([] if next_is_index else {})
                        elif (
                            cur[idx] is None
                            or not isinstance(cur[idx], (dict, list))
                        ):
                            cur[idx] = [] if next_is_index else {}
                        cur = cur[idx]
                        continue

                    if cls._is_object(cur):
                        if (
                            t not in cur
                            or cur[t] is None
                            or not isinstance(cur[t], (dict, list))
                        ):
                            cur[t] = [] if next_is_index else {}
                        cur = cur[t]
                        continue

                    cur = [] if next_is_index else {}
                return base

            for i, op in enumerate(patch_ops):
                if not isinstance(op, dict):
                    add_err(i, op, "/", "invalid operation (not an object)")
                    continue
                if not isinstance(op.get("op"), str) or not isinstance(
                    op.get("path"), str
                ):
                    add_err(i, op, "/", "invalid operation (missing op/path)")
                    continue
                if op_needs_value(op["op"]) and "value" not in op:
                    add_err(
                        i,
                        op,
                        op["path"],
                        f'operation "{op["op"]}" requires field "value"',
                    )
                    continue
                if op["op"] in ("move", "copy") and not isinstance(
                    op.get("from"), str
                ):
                    add_err(
                        i,
                        op,
                        op["path"],
                        f'operation "{op["op"]}" requires field "from"',
                    )
                    continue
                try:
                    if op["op"] == "add":
                        tokens = cls._parse_json_pointer(op["path"])
                        doc = ensure_parent_chain_for_add(doc, tokens)
                    doc = cls._apply_json_patch(doc, [op])
                except Exception as e:
                    add_err(i, op, op["path"], f"failed to apply patch: {e}")

            return {"ok": len(errors) == 0, "errors": errors, "finalDoc": doc}

        for i, op in enumerate(patch_ops):
            if not isinstance(op, dict):
                add_err(i, op, "/", "invalid operation (not an object)")
                continue
            if not isinstance(op.get("op"), str) or not isinstance(
                op.get("path"), str
            ):
                add_err(i, op, "/", "invalid operation (missing op/path)")
                continue
            if op_needs_value(op["op"]) and "value" not in op:
                add_err(
                    i,
                    op,
                    op["path"],
                    f'operation "{op["op"]}" requires field "value"',
                )
                continue
            if op["op"] in ("move", "copy") and not isinstance(
                op.get("from"), str
            ):
                add_err(
                    i,
                    op,
                    op["path"],
                    f'operation "{op["op"]}" requires field "from"',
                )
                continue

            try:
                tokens = cls._parse_json_pointer(op["path"])
            except ValueError as e:
                add_err(i, op, op["path"], str(e))
                continue

            target = cls._get_at(doc, tokens)
            pk = cls._get_parent_and_key(doc, tokens)
            parent, key = pk["parent"], pk["key"]
            parent_tokens = tokens[:-1] if tokens else []

            schema_at_target = cls._schema_at_pointer_candidates(
                root_schema, tokens
            )
            schema_at_parent = cls._schema_at_pointer_candidates(
                root_schema, parent_tokens
            )

            # existence check for replace/remove/test
            if op["op"] in ("replace", "remove", "test"):
                if not target["exists"]:
                    add_err(
                        i,
                        op,
                        op["path"],
                        f"{op['op']} failed: path does not exist in current document",
                    )
                    continue

            # add validation
            if op["op"] == "add":
                if len(tokens) > 0 and parent is None:
                    add_err(
                        i,
                        op,
                        op["path"],
                        "add failed: parent path does not exist. "
                        "Use inspect_keys to verify the parent path exists before adding.",
                    )
                    continue

                if len(tokens) > 0 and isinstance(parent, list):
                    valid_idx = key == "-" or (
                        re.match(r"^[0-9]+$", str(key))
                        and 0 <= int(key) <= len(parent)
                    )
                    if not valid_idx:
                        add_err(
                            i,
                            op,
                            op["path"],
                            f"add in array: invalid index '{key}'. "
                            f"Array has {len(parent)} items (valid indices: 0..{len(parent)}, or '-' to append). "
                            f"Use \"{op['path'].rsplit('/', 1)[0]}/-\" to append to the end.",
                        )
                        continue

                if len(tokens) > 0 and cls._is_object(parent):
                    prop_allowed = any(
                        cls._is_prop_allowed(s, key) for s in schema_at_parent
                    )
                    if not prop_allowed:
                        add_err(
                            i,
                            op,
                            op["path"],
                            f'add invalid: property "{key}" is not allowed by the parent schema. '
                            f"Check the TargetSchema to see which properties are allowed at this level.",
                        )
                        continue

                # ── Guard: detect destructive array overwrite ──
                # If the target path already holds an array and the value
                # being added is NOT an array, the model almost certainly
                # meant to append a single item — not replace the array.
                if target["exists"] and isinstance(target["value"], list):
                    val = op.get("value")
                    existing_len = len(target["value"])
                    if not isinstance(val, list):
                        add_err(
                            i,
                            op,
                            op["path"],
                            f"DESTRUCTIVE OVERWRITE BLOCKED: path \"{op['path']}\" currently holds "
                            f"an array with {existing_len} items. Your \"add\" would REPLACE the "
                            f"entire array with a single {cls._type_of_instance(val)}. "
                            f"To APPEND an item, use \"{op['path']}/-\" as the path instead. "
                            f'Example: {{"op":"add","path":"{op["path"]}/-","value":...}}',
                        )
                        continue
                    else:
                        add_err(
                            i,
                            op,
                            op["path"],
                            f"DESTRUCTIVE OVERWRITE BLOCKED: path \"{op['path']}\" currently holds "
                            f"an array with {existing_len} items. Your \"add\" would REPLACE all "
                            f"existing data with a new array of {len(val)} items. "
                            f"To APPEND items, use separate operations with \"{op['path']}/-\" for each item. "
                            f'Example: [{{"op":"add","path":"{op["path"]}/-","value":item1}}, ...]',
                        )
                        continue

            # remove validation
            if op["op"] == "remove":
                if len(tokens) == 0:
                    add_err(
                        i,
                        op,
                        op["path"],
                        "remove at root leaves the document undefined (incompatible with schema)",
                    )
                    continue
                if cls._is_object(parent):
                    would_remove_required = any(
                        cls._is_required_by_schema(s, key)
                        for s in schema_at_parent
                    )
                    if would_remove_required:
                        add_err(
                            i,
                            op,
                            op["path"],
                            f'remove invalid: "{key}" is required by parent schema',
                        )
                        continue

            # value validation for add/replace
            if op["op"] in ("add", "replace"):
                value_errors_list = [
                    cls._validate_instance(s, op.get("value"), op["path"])
                    for s in schema_at_target
                ]
                value_errors_list.sort(key=len)
                best = value_errors_list[0] if value_errors_list else []
                if len(best) > 0:
                    msgs = " | ".join(
                        f"{e['pointer']}: {e['message']}" for e in best
                    )
                    # Provide actionable hint for common type mismatches
                    hint = ""
                    val = op.get("value")
                    val_type = cls._type_of_instance(val)
                    for s in schema_at_target:
                        s_type = s.get("type") if isinstance(s, dict) else None
                        if s_type == "array" and val_type == "object":
                            hint = (
                                f' HINT: The schema expects an array at "{op["path"]}", '
                                f"but you provided a single object. "
                                f'To append this object to the array, use path "{op["path"]}/-" instead.'
                            )
                            break
                        elif s_type == "array" and val_type != "array":
                            hint = (
                                f' HINT: The schema expects an array at "{op["path"]}". '
                                f'To append an item, use path "{op["path"]}/-" with the item as value.'
                            )
                            break
                        elif s_type == "object" and val_type == "array":
                            hint = (
                                f' HINT: The schema expects an object at "{op["path"]}", '
                                f"but you provided an array. Pass a single object as the value."
                            )
                            break
                    add_err(
                        i,
                        op,
                        op["path"],
                        f"value incompatible with schema at path: {msgs}{hint}",
                    )
                    continue

            # test value validation
            if op["op"] == "test":
                value_errors_list = [
                    cls._validate_instance(s, op.get("value"), op["path"])
                    for s in schema_at_target
                ]
                value_errors_list.sort(key=len)
                best = value_errors_list[0] if value_errors_list else []
                if len(best) > 0:
                    msgs = " | ".join(
                        f"{e['pointer']}: {e['message']}" for e in best
                    )
                    add_err(
                        i,
                        op,
                        op["path"],
                        f"test value incompatible with schema: {msgs}",
                    )
                    continue

            # apply the patch
            try:
                doc = cls._apply_json_patch(doc, [op])
            except Exception as e:
                add_err(i, op, op["path"], f"failed to apply patch: {e}")
                continue

            # post-operation validation
            post_errors = cls._validate_instance(root_schema, doc, "")
            if post_errors:
                msgs_list = post_errors[:5]
                msgs = " | ".join(
                    f"{e['pointer']}: {e['message']}" for e in msgs_list
                )
                if len(post_errors) > 5:
                    msgs += " | ..."
                add_err(
                    i,
                    op,
                    op["path"],
                    f"post-operation document became invalid: {msgs}",
                )

        return {"ok": len(errors) == 0, "errors": errors, "finalDoc": doc}

    @classmethod
    def _is_prop_allowed(cls, schema: Any, key: str) -> bool:
        if schema is None or schema is True:
            return True
        if schema is False:
            return False
        if isinstance(schema, dict) and schema.get("__any"):
            return True
        if isinstance(schema, dict) and "anyOf" in schema:
            return True
        if not isinstance(schema, dict):
            return True
        ap = schema.get("additionalProperties")
        props = schema.get("properties", {})
        if key in props:
            return True
        if ap is False:
            return False
        return True

    @classmethod
    def _is_required_by_schema(cls, schema: Any, key: str) -> bool:
        if schema is None or schema is True:
            return False
        if isinstance(schema, dict) and schema.get("__any"):
            return False
        if isinstance(schema, dict) and "anyOf" in schema:
            return False
        if not isinstance(schema, dict):
            return False
        req = schema.get("required", [])
        return key in req


def apply_patches(
    document: dict[str, Any],
    patches: list[dict[str, Any]],
    target_schema: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """
    Apply JSON Patch operations to the document with schema validation.

    This is the public API matching the n8n SchemaPatchChecker behavior.

    Args:
        document: The current JSON document.
        patches: List of JSON Patch operations (RFC 6902).
        target_schema: Optional JSON Schema for validation.

    Returns:
        Result dict with ok (bool), errors (list), finalDoc (dict).
    """
    if not patches:
        return {"ok": True, "errors": [], "finalDoc": document}

    return SchemaPatchChecker.validate_patch_ops_against_schema(
        target_schema, document, patches
    )
