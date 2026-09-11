"""Reviewed A3 category names and open-answer aliases."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import json
from pathlib import Path
import re
from typing import Mapping


_SYNSET_SUFFIX = re.compile(r"\.[anvr]\.\d+$", flags=re.IGNORECASE)


def normalize_category_name(value: object) -> str:
    text = re.sub(r"\s+", " ", str(value or "").strip().casefold())
    text = re.sub(r"^(?:a|an|the)\s+", "", text)
    text = re.sub(r"[^a-z0-9 ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def display_category_name(value: object) -> str:
    text = _SYNSET_SUFFIX.sub("", str(value or "").strip())
    return re.sub(r"\s+", " ", text.replace("_", " ")).strip().casefold()


@dataclass(frozen=True)
class CategoryCatalog:
    datasets: Mapping[str, Mapping[str, object]]

    @classmethod
    def load(cls, path: Path) -> "CategoryCatalog":
        document = json.loads(Path(path).read_text(encoding="utf-8"))
        if document.get("schema") != "egoconseq.category-names.v1":
            raise ValueError("category-name schema is invalid")
        datasets = document.get("datasets")
        if not isinstance(datasets, dict):
            raise ValueError("category-name datasets are invalid")
        for dataset, value in datasets.items():
            if dataset not in {"b1k", "r2r"} or not isinstance(value, dict):
                raise ValueError("category-name dataset is invalid")
            categories = value.get("categories")
            excluded = value.get("excluded_raw")
            if not isinstance(categories, dict) or not isinstance(excluded, list):
                raise ValueError("category-name table is invalid")
            owners = {}
            for raw, row in categories.items():
                if (not isinstance(raw, str) or not isinstance(row, dict) or
                        not isinstance(row.get("canonical"), str) or
                        not isinstance(row.get("aliases"), list)):
                    raise ValueError("category-name entry is invalid")
                canonical = normalize_category_name(row["canonical"])
                for answer in (row["canonical"], *row["aliases"]):
                    normalized = normalize_category_name(answer)
                    if not normalized or owners.setdefault(
                            normalized, canonical) != canonical:
                        raise ValueError(
                            f"shared category answer in {dataset}: {normalized}")
        return cls(datasets=datasets)

    def resolve(self, dataset: str, *, raw: str, machine: str) -> dict:
        table = self.datasets[str(dataset)]
        raw = str(raw).strip()
        if raw in set(table["excluded_raw"]):
            raise ValueError(f"category is excluded from A3: {dataset}/{raw}")
        value = table["categories"].get(raw)
        canonical = (
            str(value["canonical"]).strip() if value is not None else
            re.sub(r"\s+", " ", raw.replace("_", " ")).strip().casefold())
        return {
            "machine": str(machine).strip(),
            "raw": raw,
            "canonical": canonical,
            "aliases": sorted({
                str(alias).strip() for alias in (
                    value["aliases"] if value is not None else [])
                if str(alias).strip() and
                normalize_category_name(alias) !=
                normalize_category_name(canonical)
            }, key=str.casefold),
        }

    def accepted_answers(self, dataset: str, canonical: str) -> set[str]:
        table = self.datasets[str(dataset)]
        matches = [
            value for value in table["categories"].values()
            if normalize_category_name(value["canonical"]) ==
            normalize_category_name(canonical)
        ]
        # The alias catalog is not an exhaustive vocabulary of new scenes.
        # A certified GT category is always an accepted answer to itself.
        return {str(canonical).strip()} | {
            text for value in matches
            for text in (
                str(value["canonical"]).strip(),
                *(str(alias).strip() for alias in value["aliases"]),
            )
        }


@lru_cache(maxsize=1)
def category_catalog() -> CategoryCatalog:
    return CategoryCatalog.load(Path(__file__).with_name("category_names.json"))


def a3_answer(record: Mapping[str, object],
              output: Mapping[str, object]) -> str:
    """Resolve A3 from official source evidence without guessing from a synset."""
    machine = str(output.get("category") or "").strip()
    raw = str(output.get("source_category") or "").strip()
    if not raw and output.get("instance_id") is not None:
        target_id = str(output["instance_id"])
        for entity in record.get("visible_entities") or []:
            if str(entity.get("instance_id")) != target_id:
                continue
            raw = str(entity.get("source_category") or "").strip()
            candidate = str(entity.get("category") or "").strip()
            if not raw and candidate and not _SYNSET_SUFFIX.search(candidate):
                raw = candidate
            break
    if not raw and machine and not _SYNSET_SUFFIX.search(machine):
        raw = machine
    if raw:
        return category_catalog().resolve(
            str(record["dataset"]), raw=raw, machine=machine)["canonical"]
    return display_category_name(machine)
