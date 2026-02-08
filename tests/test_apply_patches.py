"""Tests for tools/apply_patches.py — SchemaPatchChecker and apply_patches."""

from __future__ import annotations

import pytest

from tools.apply_patches import SchemaPatchChecker, apply_patches


# ======================================================================
# Basic operations without schema
# ======================================================================
class TestApplyPatchesNoSchema:
    """apply_patches with target_schema=None (schemaless mode)."""

    def test_add_to_empty_doc(self, empty_doc):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/name", "value": "Alice"},
        ])
        assert result["ok"] is True
        assert result["finalDoc"]["name"] == "Alice"

    def test_add_nested_auto_creates_parents(self, empty_doc):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/a/b/c", "value": 42},
        ])
        assert result["ok"] is True
        assert result["finalDoc"]["a"]["b"]["c"] == 42

    def test_add_to_array_with_dash(self, empty_doc):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/items", "value": []},
        ])
        assert result["ok"] is True
        result2 = apply_patches(result["finalDoc"], [
            {"op": "add", "path": "/items/-", "value": "first"},
            {"op": "add", "path": "/items/-", "value": "second"},
        ])
        assert result2["ok"] is True
        assert result2["finalDoc"]["items"] == ["first", "second"]

    def test_replace_existing(self, populated_doc):
        result = apply_patches(populated_doc, [
            {"op": "replace", "path": "/metadata/title", "value": "Updated"},
        ])
        assert result["ok"] is True
        assert result["finalDoc"]["metadata"]["title"] == "Updated"

    def test_replace_nonexistent_fails(self, empty_doc):
        result = apply_patches(empty_doc, [
            {"op": "replace", "path": "/missing", "value": "x"},
        ])
        assert result["ok"] is False

    def test_remove_existing(self, populated_doc):
        result = apply_patches(populated_doc, [
            {"op": "remove", "path": "/metadata/author"},
        ])
        assert result["ok"] is True
        assert "author" not in result["finalDoc"]["metadata"]

    def test_remove_nonexistent_fails(self, empty_doc):
        result = apply_patches(empty_doc, [
            {"op": "remove", "path": "/missing"},
        ])
        assert result["ok"] is False

    def test_move_operation(self, populated_doc):
        result = apply_patches(populated_doc, [
            {"op": "move", "from": "/metadata/author", "path": "/metadata/creator"},
        ])
        assert result["ok"] is True
        assert result["finalDoc"]["metadata"]["creator"] == "Bot"
        assert "author" not in result["finalDoc"]["metadata"]

    def test_copy_operation(self, populated_doc):
        result = apply_patches(populated_doc, [
            {"op": "copy", "from": "/metadata/title", "path": "/metadata/subtitle"},
        ])
        assert result["ok"] is True
        assert result["finalDoc"]["metadata"]["subtitle"] == "Test"
        assert result["finalDoc"]["metadata"]["title"] == "Test"  # original preserved

    def test_test_operation_passes(self, populated_doc):
        result = apply_patches(populated_doc, [
            {"op": "test", "path": "/metadata/title", "value": "Test"},
        ])
        assert result["ok"] is True

    def test_test_operation_fails(self, populated_doc):
        result = apply_patches(populated_doc, [
            {"op": "test", "path": "/metadata/title", "value": "Wrong"},
        ])
        assert result["ok"] is False

    def test_empty_patches_returns_ok(self, empty_doc):
        result = apply_patches(empty_doc, [])
        assert result["ok"] is True
        assert result["finalDoc"] == {}

    def test_invalid_op_format(self, empty_doc):
        result = apply_patches(empty_doc, [
            {"op": "add"},  # missing path
        ])
        assert result["ok"] is False

    def test_batch_of_operations(self, empty_doc):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/name", "value": "Alice"},
            {"op": "add", "path": "/age", "value": 30},
            {"op": "add", "path": "/tags", "value": []},
            {"op": "add", "path": "/tags/-", "value": "engineer"},
        ])
        assert result["ok"] is True
        assert result["finalDoc"]["name"] == "Alice"
        assert result["finalDoc"]["age"] == 30
        assert result["finalDoc"]["tags"] == ["engineer"]


# ======================================================================
# Schema validation
# ======================================================================
class TestApplyPatchesWithSchema:
    """apply_patches with a target schema."""

    def test_valid_add_against_schema(self, empty_doc, simple_schema):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/name", "value": "Alice"},
            {"op": "add", "path": "/age", "value": 30},
        ], simple_schema)
        assert result["ok"] is True
        assert result["finalDoc"]["name"] == "Alice"

    def test_type_mismatch_blocked(self, empty_doc, simple_schema):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/name", "value": "Alice"},
            {"op": "add", "path": "/age", "value": "not a number"},
        ], simple_schema)
        assert result["ok"] is False
        assert any("type" in e["message"] for e in result["errors"])

    def test_additional_properties_false(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}},
            "additionalProperties": False,
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/name", "value": "Alice"},
            {"op": "add", "path": "/extra", "value": "nope"},
        ], schema)
        assert result["ok"] is False
        assert any("not allowed" in e["message"] for e in result["errors"])

    def test_required_field_checked_post_op(self, empty_doc, simple_schema):
        # Adding only non-required fields — post-op check will notice
        # missing "name" after each op
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/age", "value": 30},
        ], simple_schema)
        # The required check happens post-operation
        assert result["ok"] is False

    def test_format_validation_email(self, empty_doc, simple_schema):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/name", "value": "Alice"},
            {"op": "add", "path": "/email", "value": "not-an-email"},
        ], simple_schema)
        assert result["ok"] is False
        assert any("format" in e["message"] for e in result["errors"])

    def test_format_validation_email_valid(self, empty_doc, simple_schema):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/name", "value": "Alice"},
            {"op": "add", "path": "/email", "value": "alice@example.com"},
        ], simple_schema)
        assert result["ok"] is True

    def test_array_schema_validation(self, empty_doc, array_schema):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/items", "value": []},
            {"op": "add", "path": "/items/-", "value": {"id": 1, "value": "first"}},
            {"op": "add", "path": "/items/-", "value": {"id": 2, "value": "second"}},
        ], array_schema)
        assert result["ok"] is True
        assert len(result["finalDoc"]["items"]) == 2

    def test_array_item_type_mismatch(self, empty_doc, array_schema):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/items", "value": []},
            {"op": "add", "path": "/items/-", "value": {"id": "not-int", "value": "x"}},
        ], array_schema)
        assert result["ok"] is False

    def test_remove_required_field_blocked(self):
        schema = {
            "type": "object",
            "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
            "required": ["name"],
        }
        doc = {"name": "Alice", "age": 30}
        result = apply_patches(doc, [
            {"op": "remove", "path": "/name"},
        ], schema)
        assert result["ok"] is False
        assert any("required" in e["message"] for e in result["errors"])


# ======================================================================
# $ref resolution
# ======================================================================
class TestRefResolution:
    """Test JSON Schema $ref resolution."""

    def test_ref_basic(self, empty_doc, nested_schema):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/name", "value": "Alice"},
            {"op": "add", "path": "/address", "value": {"street": "Main St", "city": "NY"}},
        ], nested_schema)
        assert result["ok"] is True

    def test_ref_missing_required(self, empty_doc, nested_schema):
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/name", "value": "Alice"},
            {"op": "add", "path": "/address", "value": {"street": "Main St"}},
        ], nested_schema)
        # city is required in Address
        assert result["ok"] is False

    def test_circular_ref_handled(self):
        schema = {
            "type": "object",
            "definitions": {
                "Node": {
                    "type": "object",
                    "properties": {
                        "value": {"type": "string"},
                        "child": {"$ref": "#/definitions/Node"},
                    },
                }
            },
            "properties": {
                "root": {"$ref": "#/definitions/Node"},
            },
        }
        doc = {}
        result = apply_patches(doc, [
            {"op": "add", "path": "/root", "value": {"value": "a", "child": {"value": "b"}}},
        ], schema)
        assert result["ok"] is True


# ======================================================================
# Destructive overwrite detection (in SchemaPatchChecker)
# ======================================================================
class TestDestructiveOverwriteDetection:
    """SchemaPatchChecker blocks add on existing arrays."""

    def test_add_on_existing_array_blocked(self):
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
        }
        doc = {"items": ["a", "b"]}
        result = apply_patches(doc, [
            {"op": "add", "path": "/items", "value": ["c"]},
        ], schema)
        assert result["ok"] is False
        assert any("DESTRUCTIVE" in e["message"] for e in result["errors"])

    def test_append_to_existing_array_ok(self):
        schema = {
            "type": "object",
            "properties": {"items": {"type": "array", "items": {"type": "string"}}},
        }
        doc = {"items": ["a", "b"]}
        result = apply_patches(doc, [
            {"op": "add", "path": "/items/-", "value": "c"},
        ], schema)
        assert result["ok"] is True
        assert result["finalDoc"]["items"] == ["a", "b", "c"]


# ======================================================================
# oneOf / allOf validation
# ======================================================================
class TestOneOfAllOfValidation:
    """Test oneOf and allOf schema validation."""

    def test_one_of_passes_with_single_match(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                }
            },
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/value", "value": "hello"},
        ], schema)
        assert result["ok"] is True

    def test_one_of_fails_with_no_match(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                }
            },
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/value", "value": [1, 2, 3]},
        ], schema)
        assert result["ok"] is False
        assert any("oneOf" in e["message"] for e in result["errors"])

    def test_one_of_fails_with_multiple_matches(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "oneOf": [
                        {"type": "number"},
                        {"type": "integer"},
                    ]
                }
            },
        }
        # integer matches both "number" and "integer"
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/value", "value": 42},
        ], schema)
        assert result["ok"] is False
        assert any("oneOf" in e["message"] for e in result["errors"])

    def test_all_of_passes_when_all_match(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "allOf": [
                        {"type": "integer"},
                        {"minimum": 0},
                        {"maximum": 100},
                    ]
                }
            },
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/value", "value": 50},
        ], schema)
        assert result["ok"] is True

    def test_all_of_fails_when_one_doesnt_match(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "allOf": [
                        {"type": "integer"},
                        {"minimum": 0},
                        {"maximum": 100},
                    ]
                }
            },
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/value", "value": 150},
        ], schema)
        assert result["ok"] is False
        assert any("maximum" in e["message"] for e in result["errors"])

    def test_all_of_with_object_properties(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                        {
                            "type": "object",
                            "properties": {"age": {"type": "integer"}},
                            "required": ["age"],
                        },
                    ]
                }
            },
        }
        # Both name and age are required by allOf
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/person", "value": {"name": "Alice", "age": 30}},
        ], schema)
        assert result["ok"] is True

    def test_all_of_missing_required_from_one_sub_schema(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "person": {
                    "allOf": [
                        {
                            "type": "object",
                            "properties": {"name": {"type": "string"}},
                            "required": ["name"],
                        },
                        {
                            "type": "object",
                            "properties": {"age": {"type": "integer"}},
                            "required": ["age"],
                        },
                    ]
                }
            },
        }
        # Missing "age" which is required by second sub-schema
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/person", "value": {"name": "Alice"}},
        ], schema)
        assert result["ok"] is False


# ======================================================================
# Format validation
# ======================================================================
class TestFormatValidation:
    """Test string format validators."""

    @pytest.mark.parametrize("value,expected", [
        ("user@example.com", True),
        ("user@sub.domain.com", True),
        ("not-email", False),
        ("@missing.user", False),
    ])
    def test_email_format(self, value, expected):
        ok = SchemaPatchChecker._validate_format("email", value)
        assert ok == expected

    @pytest.mark.parametrize("value,expected", [
        ("2024-01-15", True),
        ("2024-13-01", False),
        ("not-a-date", False),
    ])
    def test_date_format(self, value, expected):
        ok = SchemaPatchChecker._validate_format("date", value)
        assert ok == expected

    @pytest.mark.parametrize("value,expected", [
        ("2024-01-15T10:30:00Z", True),
        ("2024-01-15T10:30:00+03:00", True),
        ("not-datetime", False),
    ])
    def test_datetime_format(self, value, expected):
        ok = SchemaPatchChecker._validate_format("date-time", value)
        assert ok == expected

    @pytest.mark.parametrize("value,expected", [
        ("https://example.com", True),
        ("ftp://files.org/doc", True),
        ("not a uri", False),
    ])
    def test_uri_format(self, value, expected):
        ok = SchemaPatchChecker._validate_format("uri", value)
        assert ok == expected

    @pytest.mark.parametrize("value,expected", [
        ("192.168.1.1", True),
        ("0.0.0.0", True),
        ("256.1.1.1", False),
        ("1.2.3", False),
    ])
    def test_ipv4_format(self, value, expected):
        ok = SchemaPatchChecker._validate_format("ipv4", value)
        assert ok == expected

    @pytest.mark.parametrize("value,expected", [
        ("550e8400-e29b-41d4-a716-446655440000", True),
        ("not-a-uuid", False),
    ])
    def test_uuid_format(self, value, expected):
        ok = SchemaPatchChecker._validate_format("uuid", value)
        assert ok == expected

    def test_unknown_format_passes(self):
        assert SchemaPatchChecker._validate_format("custom-format", "anything") is True

    def test_non_string_fails(self):
        assert SchemaPatchChecker._validate_format("email", 123) is False


# ======================================================================
# Enum validation
# ======================================================================
class TestEnumValidation:

    def test_enum_valid(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/status", "value": "active"},
        ], schema)
        assert result["ok"] is True

    def test_enum_invalid(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["active", "inactive"]},
            },
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/status", "value": "unknown"},
        ], schema)
        assert result["ok"] is False
        assert any("enum" in e["message"] for e in result["errors"])


# ======================================================================
# Number constraints
# ======================================================================
class TestNumberConstraints:

    def test_minimum(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {"age": {"type": "integer", "minimum": 0}},
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/age", "value": -1},
        ], schema)
        assert result["ok"] is False

    def test_maximum(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {"score": {"type": "number", "maximum": 100}},
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/score", "value": 101},
        ], schema)
        assert result["ok"] is False

    def test_within_range(self, empty_doc):
        schema = {
            "type": "object",
            "properties": {
                "pct": {"type": "number", "minimum": 0, "maximum": 100}
            },
        }
        result = apply_patches(empty_doc, [
            {"op": "add", "path": "/pct", "value": 50.5},
        ], schema)
        assert result["ok"] is True
