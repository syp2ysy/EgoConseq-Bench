"""Dependency-light frozen authority-surface capability contracts."""

from __future__ import annotations

from pipeline import dataset_contracts


CAPABILITY_SNAPSHOT_SCHEMA = "egoconseq.authority-capability-snapshot.v1"
CAPABILITY_SCOPE = "static_authority_surface_capability"
METHODS = (
    "assign",
    "instance_points",
    "target_geometry_atom",
    "confirm_contact_instances",
    "instance_triangles",
)
# Presentation order for capability reports; collection scheduling uses
# dataset_contracts.main_collection_datasets() and intentionally differs.
DATASET_ORDER = ("r2r", "b1k", "gs")
FROZEN_METHOD_SURFACES = {
    "r2r": (True, True, True, True, True),
    "b1k": (True, True, True, True, True),
    "gs": (True, True, True, True, False),
}
TASK_REQUIREMENTS = {
    "A1": (),
    "A2": (),
    "A3": ("assign", "confirm_contact_instances"),
    "B1": ("instance_points", "target_geometry_atom"),
    "B2": ("instance_points", "target_geometry_atom"),
    "C1": (),
}
CERTIFICATION = {
    "runtime_evidence": "not_assessed",
    "source_binding": "not_assessed",
    "formal_collection": "not_assessed",
}


def frozen_method_surface(source_dataset: str) -> dict[str, bool]:
    """Return the Task 3 method surface for one registered dataset."""
    dataset_contracts.dataset_source_contract(source_dataset)
    try:
        values = FROZEN_METHOD_SURFACES[source_dataset]
    except KeyError as error:
        raise ValueError(
            f"unsupported capability dataset: {source_dataset!r}") from error
    return dict(zip(METHODS, values))


def task_available(source_dataset: str, task_id: str) -> bool:
    """Return whether one frozen dataset authority supports one task."""
    if task_id not in TASK_REQUIREMENTS:
        raise ValueError(f"unsupported capability task: {task_id!r}")
    method_surface = frozen_method_surface(source_dataset)
    return all(
        method_surface.get(method, False)
        for method in TASK_REQUIREMENTS[task_id]
    )


def snapshot_fields(source_dataset: str) -> dict:
    """Build one structured capability body without its record hash."""
    source_contract = dataset_contracts.dataset_source_contract(source_dataset)
    method_surface = frozen_method_surface(source_dataset)
    statuses = {}
    for task, required_methods in TASK_REQUIREMENTS.items():
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
        "schema": CAPABILITY_SNAPSHOT_SCHEMA,
        "scope": CAPABILITY_SCOPE,
        "source_dataset": source_dataset,
        "main_collection_enabled": source_contract.main_collection_enabled,
        "method_surface": method_surface,
        "task_statuses": statuses,
        "certification": dict(CERTIFICATION),
    }
