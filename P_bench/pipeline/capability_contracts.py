"""Dependency-light frozen authority-surface capability contracts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

from pipeline import dataset_contracts


CAPABILITY_SNAPSHOT_SCHEMA_V1 = \
    "egoconseq.authority-capability-snapshot.v1"
CAPABILITY_SNAPSHOT_SCHEMA_V2 = \
    "egoconseq.authority-capability-snapshot.v2"
CAPABILITY_SNAPSHOT_SCHEMA = CAPABILITY_SNAPSHOT_SCHEMA_V2
CAPABILITY_SCOPE = "static_authority_surface_capability"
METHODS = (
    "assign",
    "instance_points",
    "confirm_contact_instances",
    "instance_triangles",
)
# Presentation order for capability reports; collection scheduling uses
# dataset_contracts.main_collection_datasets() and intentionally differs.
DATASET_ORDER = ("r2r", "b1k", "gs")
FROZEN_METHOD_SURFACES = {
    "r2r": (True, True, True, True),
    "b1k": (True, True, True, True),
    "gs": (True, True, True, False),
}
TASK_REQUIREMENTS = {
    "A1": (),
    "A2": (),
    "A3": ("assign", "confirm_contact_instances"),
    "B1": ("assign",),
    "B2": ("assign",),
    "C1": (),
}
CERTIFICATION = {
    "runtime_evidence": "not_assessed",
    "source_binding": "not_assessed",
    "formal_collection": "not_assessed",
}

_METHODS_V1 = (
    "assign",
    "instance_points",
    "target_geometry_atom",
    "confirm_contact_instances",
    "instance_triangles",
)
_FROZEN_METHOD_SURFACES_V1 = {
    "r2r": (True, True, True, True, True),
    "b1k": (True, True, True, True, True),
    "gs": (True, True, True, True, False),
}
_TASK_REQUIREMENTS_V1 = {
    "A1": (),
    "A2": (),
    "A3": ("assign", "confirm_contact_instances"),
    "B1": ("instance_points", "target_geometry_atom"),
    "B2": ("instance_points", "target_geometry_atom"),
    "C1": (),
}


def _schema_contract(schema: str):
    value = str(schema)
    if value == CAPABILITY_SNAPSHOT_SCHEMA_V1:
        return _METHODS_V1, _FROZEN_METHOD_SURFACES_V1, \
            _TASK_REQUIREMENTS_V1
    if value == CAPABILITY_SNAPSHOT_SCHEMA_V2:
        return METHODS, FROZEN_METHOD_SURFACES, TASK_REQUIREMENTS
    raise ValueError(f"unsupported capability snapshot schema: {value!r}")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"),
        allow_nan=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _type_exact_equal(left, right) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, dict):
        return set(left) == set(right) and all(
            _type_exact_equal(left[key], right[key]) for key in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(
            _type_exact_equal(first, second)
            for first, second in zip(left, right))
    return left == right


def frozen_method_surface(
        source_dataset: str, *,
        schema: str = CAPABILITY_SNAPSHOT_SCHEMA) -> dict[str, bool]:
    """Return the Task 3 method surface for one registered dataset."""
    dataset_contracts.dataset_source_contract(source_dataset)
    methods, surfaces, _requirements = _schema_contract(schema)
    try:
        values = surfaces[source_dataset]
    except KeyError as error:
        raise ValueError(
            f"unsupported capability dataset: {source_dataset!r}") from error
    return dict(zip(methods, values))


def task_available(
        source_dataset: str, task_id: str, *,
        schema: str = CAPABILITY_SNAPSHOT_SCHEMA) -> bool:
    """Return whether one frozen dataset authority supports one task."""
    _methods, _surfaces, requirements = _schema_contract(schema)
    if task_id not in requirements:
        raise ValueError(f"unsupported capability task: {task_id!r}")
    method_surface = frozen_method_surface(source_dataset, schema=schema)
    return all(
        method_surface.get(method, False)
        for method in requirements[task_id]
    )


def snapshot_fields(
        source_dataset: str, *,
        schema: str = CAPABILITY_SNAPSHOT_SCHEMA) -> dict:
    """Build one structured capability body without its record hash."""
    source_contract = dataset_contracts.dataset_source_contract(source_dataset)
    _methods, _surfaces, requirements = _schema_contract(schema)
    method_surface = frozen_method_surface(source_dataset, schema=schema)
    statuses = {}
    for task, required_methods in requirements.items():
        missing = [method for method in required_methods
                   if not method_surface.get(method, False)]
        statuses[task] = (
            {
                "status": "unavailable",
                "reason": "capability_unavailable",
                "missing_methods": missing,
            }
            if missing else {"status": "available"}
        )
    return {
        "schema": str(schema),
        "scope": CAPABILITY_SCOPE,
        "source_dataset": source_dataset,
        "main_collection_enabled": source_contract.main_collection_enabled,
        "method_surface": method_surface,
        "task_statuses": statuses,
        "certification": dict(CERTIFICATION),
    }


def validate_snapshot(snapshot: Mapping, source_dataset: str) -> dict:
    """Authenticate a persisted snapshot under its declared frozen schema."""
    if not isinstance(snapshot, Mapping):
        raise ValueError("capability snapshot must be an object")
    schema = snapshot.get("schema")
    if not isinstance(schema, str):
        raise ValueError("capability snapshot schema is missing")
    body = snapshot_fields(source_dataset, schema=schema)
    expected = {**body, "sha256": _canonical_sha256(body)}
    observed = dict(snapshot)
    if not _type_exact_equal(observed, expected):
        raise ValueError("capability snapshot differs from its frozen schema")
    return observed
