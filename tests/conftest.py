"""Shared fixtures for the test suite."""

from __future__ import annotations

import pytest


@pytest.fixture
def simple_schema():
    """A simple JSON Schema with an object and required fields."""
    return {
        "type": "object",
        "properties": {
            "name": {"type": "string"},
            "age": {"type": "integer"},
            "email": {"type": "string", "format": "email"},
        },
        "required": ["name"],
    }


@pytest.fixture
def array_schema():
    """JSON Schema with an array of objects."""
    return {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "integer"},
                        "value": {"type": "string"},
                    },
                    "required": ["id"],
                },
            }
        },
    }


@pytest.fixture
def nested_schema():
    """Deeply nested schema with $ref."""
    return {
        "type": "object",
        "definitions": {
            "Address": {
                "type": "object",
                "properties": {
                    "street": {"type": "string"},
                    "city": {"type": "string"},
                },
                "required": ["street", "city"],
            }
        },
        "properties": {
            "name": {"type": "string"},
            "address": {"$ref": "#/definitions/Address"},
        },
        "required": ["name"],
    }


@pytest.fixture
def empty_doc():
    """Empty JSON document."""
    return {}


@pytest.fixture
def populated_doc():
    """A pre-populated document for testing patches."""
    return {
        "metadata": {"title": "Test", "author": "Bot"},
        "sections": [
            {
                "section_name": "Overview",
                "fields": [
                    {"label": "Revenue", "value": 1000},
                    {"label": "Profit", "value": 200},
                ],
            }
        ],
    }
