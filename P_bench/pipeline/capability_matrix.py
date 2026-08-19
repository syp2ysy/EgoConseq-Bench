"""Read-only static authority-surface capability diagnostics.

This module deliberately reports only methods exposed by the current semantic
authority classes.  It does not initialize a runtime, read source assets, or
claim source binding, evidence, or collection certification.
"""

from __future__ import annotations

from collections.abc import Iterable

from pipeline import (
    b1k_semantic, capability_contracts, dataset_contracts, gs_semantic,
    semantic,
)


CAPABILITY_REPORT_SCHEMA = "egoconseq.authority-capability-matrix.v1"
CAPABILITY_SCOPE = capability_contracts.CAPABILITY_SCOPE
METHODS = capability_contracts.METHODS
DATASET_ORDER = capability_contracts.DATASET_ORDER
_AUTHORITY_TYPES = {
    "r2r": semantic.MP3DSemanticIndex,
    "b1k": b1k_semantic.B1KSemanticAuthority,
    "gs": gs_semantic.VisibleDepthSemanticIndex,
}
_FROZEN_METHOD_SURFACES = capability_contracts.FROZEN_METHOD_SURFACES
_TASK_REQUIREMENTS = capability_contracts.TASK_REQUIREMENTS
_CERTIFICATION = capability_contracts.CERTIFICATION


def authority_method_surface(source_dataset: str) -> dict[str, bool]:
    """Inspect one current authority class, including inherited methods."""
    try:
        authority_type = _AUTHORITY_TYPES[source_dataset]
    except KeyError as error:
        raise ValueError(f"unsupported capability dataset: {source_dataset!r}") \
            from error
    return {method: callable(getattr(authority_type, method, None))
            for method in METHODS}


def task_statuses(method_surface: dict[str, bool]) -> dict[str, str]:
    """Derive task availability from the static object-surface requirements."""
    statuses = {}
    for task, required_methods in _TASK_REQUIREMENTS.items():
        missing = next(
            (method for method in required_methods
             if not method_surface.get(method, False)), None)
        statuses[task] = "available" if missing is None else f"missing {missing}"
    return statuses


def _dataset_row(source_dataset: str) -> dict:
    method_surface = authority_method_surface(source_dataset)
    observed = tuple(method_surface[method] for method in METHODS)
    if observed != _FROZEN_METHOD_SURFACES[source_dataset]:
        raise RuntimeError(
            f"{source_dataset} authority method surface drifted from the "
            "frozen capability matrix")
    return {
        "source_dataset": source_dataset,
        "main_collection_enabled": dataset_contracts.dataset_source_contract(
            source_dataset).main_collection_enabled,
        "method_surface": method_surface,
        "task_statuses": task_statuses(method_surface),
    }


def _normalized_scene_identities(
        scene_identities: Iterable[tuple[str, str]]) -> list[tuple[str, str]]:
    values = []
    seen = set()
    for source_dataset, scene_id in scene_identities:
        if source_dataset not in _AUTHORITY_TYPES:
            raise ValueError(
                f"unsupported capability dataset: {source_dataset!r}")
        normalized_scene_id = str(scene_id).strip()
        if not normalized_scene_id:
            raise ValueError("capability scene id must be nonempty")
        identity = (source_dataset, normalized_scene_id)
        if identity in seen:
            raise ValueError(
                "duplicate capability scene identity: "
                f"{source_dataset}:{normalized_scene_id}")
        seen.add(identity)
        values.append(identity)
    rank = {name: index for index, name in enumerate(DATASET_ORDER)}
    return sorted(values, key=lambda value: (rank[value[0]], value[1]))


def scene_identities_from_specs(scene_specs: Iterable[object]
                                ) -> list[tuple[str, str]]:
    """Normalize scene-catalog rows for a static per-scene expansion."""
    return _normalized_scene_identities(
        (str(getattr(spec, "source_dataset")),
         str(getattr(spec, "scene_id")))
        for spec in scene_specs)


def build_capability_report(
        *, scene_identities: Iterable[tuple[str, str]] = ()) -> dict:
    """Build a deterministic report from static class surfaces and scene IDs.

    ``scene_identities`` must be explicitly supplied by a caller or obtained
    from a scene catalog.  Rows are capability expansions only: they are not
    claims that a source bound runtime, evidence, or formal collector exists.
    """
    rows = {dataset: _dataset_row(dataset) for dataset in DATASET_ORDER}
    scene_rows = []
    for source_dataset, scene_id in _normalized_scene_identities(
            scene_identities):
        scene_rows.append({
            "scene_id": scene_id,
            **rows[source_dataset],
        })
    return {
        "schema": CAPABILITY_REPORT_SCHEMA,
        "scope": CAPABILITY_SCOPE,
        "certification": dict(_CERTIFICATION),
        "datasets": [rows[dataset] for dataset in DATASET_ORDER],
        "scenes": scene_rows,
    }
