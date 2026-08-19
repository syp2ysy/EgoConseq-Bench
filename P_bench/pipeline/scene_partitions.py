"""Frozen train-seen/test-unseen scene ownership for collection."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping, Sequence


SCHEMA = "egoconseq.scene-partitions.v1"
DEFAULT_PATH = Path(__file__).with_name("assets") / \
    "scene_partitions.v1.json"
COLLECTABLE_PARTITIONS = ("train_seen", "test_unseen")
_KNOWN_PARTITIONS = frozenset({
    *COLLECTABLE_PARTITIONS,
    "excluded_damaged",
    "excluded_unavailable",
})


@dataclass(frozen=True)
class ScenePartitions:
    """Complete scene-to-partition and scene-to-family assignment."""

    rows: Mapping[str, Mapping[str, Mapping[str, str]]]
    sha256: str | None = None

    def partition(self, dataset: str, scene_id: str) -> str:
        try:
            return str(self.rows[str(dataset)][str(scene_id)]["partition"])
        except KeyError as error:
            raise ValueError(
                f"scene partition has no {dataset} scene {scene_id!r}") \
                from error

    def family(self, dataset: str, scene_id: str) -> str:
        try:
            return str(self.rows[str(dataset)][str(scene_id)]["family"])
        except KeyError as error:
            raise ValueError(
                f"scene partition has no {dataset} scene {scene_id!r}") \
                from error

    def counts(self, dataset: str) -> dict[str, int]:
        try:
            values = self.rows[str(dataset)].values()
        except KeyError as error:
            raise ValueError(
                f"scene partition has no dataset {dataset!r}") from error
        return dict(sorted(Counter(
            str(row["partition"]) for row in values).items()))

    def select_catalog(
            self, dataset: str, catalog: Sequence,
            *, benchmark_partition: str) -> list:
        """Return the requested partition after proving every scene assigned."""
        requested = str(benchmark_partition)
        if requested not in COLLECTABLE_PARTITIONS:
            raise ValueError(
                f"benchmark partition is not collectable: {requested!r}")
        selected = []
        for scene in catalog:
            scene_dataset = str(getattr(scene, "source_dataset"))
            scene_id = str(getattr(scene, "scene_id"))
            if scene_dataset != str(dataset):
                raise ValueError(
                    f"scene catalog dataset mismatch: {scene_dataset!r}")
            if self.partition(dataset, scene_id) == requested:
                selected.append(scene)
        return selected


def from_rows(rows: Mapping, *, sha256: str | None = None) -> ScenePartitions:
    """Validate in-memory rows and return the immutable lookup surface."""
    if not isinstance(rows, Mapping) or not rows:
        raise ValueError("scene partitions must contain datasets")
    normalized = {}
    for dataset, scene_rows in rows.items():
        if not isinstance(scene_rows, Mapping) or not scene_rows:
            raise ValueError(
                f"scene partition dataset {dataset!r} has no scenes")
        families = {}
        normalized_rows = {}
        for scene_id, row in scene_rows.items():
            if (not isinstance(scene_id, str) or not scene_id or
                    not isinstance(row, Mapping)):
                raise ValueError("scene partition row is invalid")
            partition = str(row.get("partition") or "")
            family = str(row.get("family") or "")
            if partition not in _KNOWN_PARTITIONS or not family:
                raise ValueError(
                    f"scene partition row is invalid for {scene_id!r}")
            previous = families.setdefault(family, partition)
            if previous != partition:
                raise ValueError(
                    f"family {family!r} crosses partitions")
            normalized_rows[scene_id] = {
                "partition": partition,
                "family": family,
            }
        normalized[str(dataset)] = normalized_rows
    return ScenePartitions(rows=normalized, sha256=sha256)


def load(path: Path | str = DEFAULT_PATH) -> ScenePartitions:
    """Load the committed partition asset once at the collection boundary."""
    raw = Path(path).read_bytes()
    payload = json.loads(raw)
    if (not isinstance(payload, dict) or payload.get("schema") != SCHEMA or
            set(payload) != {"schema", "datasets"}):
        raise ValueError("scene partition asset header is invalid")
    return from_rows(
        payload["datasets"], sha256=hashlib.sha256(raw).hexdigest())
