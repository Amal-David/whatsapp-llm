"""Resumable run manifests and deterministic merging for the research council."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .corpus import (
    CorpusReport,
    load_jsonl,
    normalized_identifier,
    paper_identifier_keys,
    validate_corpus,
)

RUN_SCHEMA_VERSION = "digital_self_council_run.v1"
INSPECTION_RANK = {"metadata": 1, "abstract": 2, "full_text": 3}
EVIDENCE_RANK = {
    "limited": 1,
    "theoretical": 2,
    "moderate": 3,
    "strong": 4,
    "foundational": 5,
}


class CouncilRunError(RuntimeError):
    """Raised when council runtime state cannot be trusted or advanced."""


@dataclass(frozen=True)
class CouncilMergeReport:
    """Aggregate merge result without paper text."""

    input_records: int
    output_records: int
    duplicate_groups: int
    duplicate_records: int
    duplicate_group_sources: list[list[str]]
    corpus_report: CorpusReport
    retained_record_ids: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "input_records": self.input_records,
            "output_records": self.output_records,
            "duplicate_groups": self.duplicate_groups,
            "duplicate_records": self.duplicate_records,
            "duplicate_group_sources": self.duplicate_group_sources,
            "corpus_report": self.corpus_report.to_dict(),
            "retained_record_ids": self.retained_record_ids,
        }


def initialize_run(
    run_dir: str | Path,
    *,
    run_id: str,
    council_path: str | Path,
    schema_path: str | Path,
    created_at: datetime | None = None,
) -> dict[str, Any]:
    """Create or safely resume one council run pinned to its input contracts."""
    if not run_id.strip():
        raise ValueError("run_id cannot be empty")
    root = Path(run_dir)
    council_file = Path(council_path).resolve()
    schema_file = Path(schema_path).resolve()
    council = yaml.safe_load(council_file.read_text(encoding="utf-8"))
    if not isinstance(council, dict) or not isinstance(council.get("seats"), list):
        raise CouncilRunError("council configuration must define a seats list")
    seat_ids = [str(seat["id"]) for seat in council["seats"]]
    if len(seat_ids) != len(set(seat_ids)) or not seat_ids:
        raise CouncilRunError("council seat ids must be unique and non-empty")

    input_state = {
        "council_path": str(council_file),
        "council_sha256": _file_hash(council_file),
        "paper_schema_path": str(schema_file),
        "paper_schema_sha256": _file_hash(schema_file),
    }
    manifest_path = root / "manifest.json"
    if manifest_path.exists():
        existing = load_manifest(root)
        if existing.get("run_id") != run_id:
            raise CouncilRunError("run_id does not match the existing manifest")
        existing_hashes = {
            key: existing.get("inputs", {}).get(key)
            for key in ("council_sha256", "paper_schema_sha256")
        }
        requested_hashes = {
            key: input_state[key]
            for key in ("council_sha256", "paper_schema_sha256")
        }
        if existing_hashes != requested_hashes:
            raise CouncilRunError(
                "research contract input hashes differ from the existing run"
            )
        if set(existing.get("seats", {})) != set(seat_ids):
            raise CouncilRunError("council seats differ from the existing run")
        return existing

    root.mkdir(parents=True, exist_ok=True)
    timestamp = _as_utc(created_at or datetime.now(timezone.utc)).isoformat()
    manifest: dict[str, Any] = {
        "schema_version": RUN_SCHEMA_VERSION,
        "run_id": run_id,
        "created_at": timestamp,
        "updated_at": timestamp,
        "inputs": input_state,
        "seats": {
            seat_id: {
                "status": "pending",
                "agent_id": None,
                "started_at": None,
                "completed_at": None,
                "query_strategy": [],
                "artifact_path": None,
                "artifact_sha256": None,
                "record_count": 0,
                "inspection_depth_counts": {},
                "errors": [],
            }
            for seat_id in seat_ids
        },
        "merge": {
            "status": "pending",
            "output_path": None,
            "output_sha256": None,
            "report_path": None,
            "input_records": 0,
            "output_records": 0,
            "duplicate_groups": 0,
            "errors": [],
        },
    }
    _save_manifest(root, manifest)
    return manifest


def load_manifest(run_dir: str | Path) -> dict[str, Any]:
    manifest_path = Path(run_dir) / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(manifest_path)
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema_version") != RUN_SCHEMA_VERSION:
        raise CouncilRunError("unsupported or malformed council manifest")
    if not isinstance(data.get("seats"), dict) or not isinstance(
        data.get("merge"), dict
    ):
        raise CouncilRunError("council manifest is missing runtime state")
    return data


def record_seat_artifact(
    run_dir: str | Path,
    *,
    seat_id: str,
    artifact_path: str | Path,
    agent_id: str,
    query_strategy: list[str],
) -> dict[str, Any]:
    """Validate one seat and persist success or failure before returning."""
    root = Path(run_dir)
    artifact = Path(artifact_path).resolve()
    manifest = load_manifest(root)
    if seat_id not in manifest["seats"]:
        raise CouncilRunError(f"unknown council seat: {seat_id}")
    if not agent_id.strip() or not query_strategy or not all(
        query.strip() for query in query_strategy
    ):
        raise ValueError("agent_id and a non-empty query strategy are required")

    seat = manifest["seats"][seat_id]
    seat.update(
        {
            "status": "running",
            "agent_id": agent_id,
            "started_at": seat.get("started_at") or _utc_now(),
            "completed_at": None,
            "query_strategy": list(query_strategy),
            "artifact_path": str(artifact),
            "artifact_sha256": _file_hash(artifact) if artifact.is_file() else None,
            "record_count": 0,
            "inspection_depth_counts": {},
            "errors": [],
        }
    )
    _save_manifest(root, manifest)

    try:
        records = load_jsonl(artifact)
        report = validate_corpus(
            records,
            min_count=15,
            max_count=25,
            require_reviewed=False,
        )
        wrong_seats = sorted(
            {
                str(record.get("seat"))
                for record in records
                if record.get("seat") != seat_id
            }
        )
        if wrong_seats:
            raise CouncilRunError(
                f"seat artifact {seat_id} contains records for {wrong_seats}"
            )
    except Exception as error:
        seat.update(
            {
                "status": "failed",
                "completed_at": _utc_now(),
                "record_count": 0,
                "inspection_depth_counts": {},
                "errors": [str(error)],
            }
        )
        _save_manifest(root, manifest)
        raise

    seat.update(
        {
            "status": "complete",
            "completed_at": _utc_now(),
            "artifact_sha256": _file_hash(artifact),
            "record_count": report.counted_primary_papers,
            "inspection_depth_counts": report.inspection_depth_counts,
            "errors": [],
        }
    )
    _save_manifest(root, manifest)
    return manifest


def merge_run(
    run_dir: str | Path,
    *,
    output_path: str | Path,
) -> CouncilMergeReport:
    """Merge every complete, unchanged seat and fail on partial evidence."""
    root = Path(run_dir)
    output = Path(output_path).resolve()
    report_path = output.with_name(f"{output.stem}-report.json")
    manifest = load_manifest(root)
    incomplete = [
        seat_id
        for seat_id, state in manifest["seats"].items()
        if state.get("status") != "complete"
    ]
    if incomplete:
        raise CouncilRunError(f"{len(incomplete)} seats are not complete: {incomplete}")

    records = []
    try:
        artifact_paths = {
            Path(str(state["artifact_path"])).resolve()
            for state in manifest["seats"].values()
        }
        if {output, report_path} & artifact_paths:
            raise CouncilRunError("merge outputs cannot overwrite a seat artifact")

        for seat_id, state in manifest["seats"].items():
            artifact = Path(str(state["artifact_path"]))
            current_hash = _file_hash(artifact)
            if current_hash != state.get("artifact_sha256"):
                raise CouncilRunError(
                    f"seat artifact changed after validation: {seat_id}"
                )
            seat_records = load_jsonl(artifact)
            validate_corpus(
                seat_records,
                min_count=15,
                max_count=25,
                require_reviewed=True,
            )
            extractor_agent_id = state.get("agent_id")
            if any(
                record["provenance"].get("reviewer") == extractor_agent_id
                for record in seat_records
            ):
                raise CouncilRunError(
                    f"seat {seat_id} cannot review its own evidence"
                )
            records.extend(seat_records)

        merged, retained, duplicate_group_sources = _deduplicate_records(records)
        corpus_report = validate_corpus(
            merged,
            min_count=100,
            max_count=200,
            require_reviewed=True,
        )
        duplicate_records = len(records) - len(merged)
        report = CouncilMergeReport(
            input_records=len(records),
            output_records=len(merged),
            duplicate_groups=len(duplicate_group_sources),
            duplicate_records=duplicate_records,
            duplicate_group_sources=duplicate_group_sources,
            corpus_report=corpus_report,
            retained_record_ids=retained,
        )
        _write_jsonl(output, merged)
        _write_private_text(
            report_path,
            json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n",
        )
    except Exception as error:
        manifest["merge"].update(
            {
                "status": "failed",
                "errors": [str(error)],
            }
        )
        _save_manifest(root, manifest)
        raise

    manifest["merge"].update(
        {
            "status": "complete",
            "output_path": str(output.resolve()),
            "output_sha256": _file_hash(output),
            "report_path": str(report_path.resolve()),
            "input_records": report.input_records,
            "output_records": report.output_records,
            "duplicate_groups": report.duplicate_groups,
            "errors": [],
        }
    )
    _save_manifest(root, manifest)
    return report


def summarize_run(run_dir: str | Path) -> dict[str, Any]:
    manifest = load_manifest(run_dir)
    status_counts = {
        status: sum(
            state.get("status") == status for state in manifest["seats"].values()
        )
        for status in ("pending", "running", "complete", "failed")
    }
    return {
        "schema_version": "digital_self_council_status.v1",
        "run_id": manifest["run_id"],
        "seat_count": len(manifest["seats"]),
        "completed_seats": status_counts["complete"],
        "pending_seats": status_counts["pending"],
        "running_seats": status_counts["running"],
        "failed_seats": status_counts["failed"],
        "validated_records": sum(
            int(state.get("record_count", 0))
            for state in manifest["seats"].values()
            if state.get("status") == "complete"
        ),
        "merge_status": manifest["merge"].get("status"),
        "merge_output_records": manifest["merge"].get("output_records", 0),
    }


def _deduplicate_records(
    records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, str], list[list[str]]]:
    parent = list(range(len(records)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[max(left_root, right_root)] = min(left_root, right_root)

    identifier_owner: dict[str, int] = {}
    for index, record in enumerate(records):
        for identifier in paper_identifier_keys(record):
            existing = identifier_owner.get(identifier)
            if existing is not None:
                union(index, existing)
            else:
                identifier_owner[identifier] = index

    groups: dict[int, list[dict[str, Any]]] = {}
    for index, record in enumerate(records):
        groups.setdefault(find(index), []).append(record)

    merged = []
    retained: dict[str, str] = {}
    duplicate_group_sources = []
    for group in groups.values():
        winner = sorted(group, key=_record_rank)[0]
        combined = deepcopy(winner)
        winner_canonical = normalized_identifier(combined["canonical_id"])
        alternate_identifiers: dict[str, dict[str, Any]] = {}
        for record in group:
            for identifier in [record["canonical_id"], *record["alternate_ids"]]:
                key = normalized_identifier(identifier)
                if key != winner_canonical:
                    alternate_identifiers.setdefault(key, deepcopy(identifier))
        combined["alternate_ids"] = [
            alternate_identifiers[key] for key in sorted(alternate_identifiers)
        ]
        combined["taxonomy_tags"] = sorted(
            {tag for record in group for tag in record["taxonomy_tags"]}
        )
        combined["modalities"] = sorted(
            {modality for record in group for modality in record["modalities"]}
        )
        combined["architecture_implications"] = sorted(
            {
                implication
                for record in group
                for implication in record["architecture_implications"]
            }
        )
        combined["provenance"]["search_queries"] = sorted(
            {
                query
                for record in group
                for query in record["provenance"]["search_queries"]
            }
        )
        if len(group) > 1:
            contributing_seats = sorted({str(record["seat"]) for record in group})
            merged_record_ids = sorted({str(record["record_id"]) for record in group})
            combined["provenance"]["contributing_seats"] = contributing_seats
            combined["provenance"]["merged_record_ids"] = merged_record_ids
            duplicate_group_sources.append(
                sorted(
                    f"{record['seat']}/{record['record_id']}" for record in group
                )
            )
        merged.append(combined)
        for record in group:
            if record["record_id"] != combined["record_id"]:
                retained[record["record_id"]] = combined["record_id"]

    return (
        sorted(merged, key=lambda record: record["record_id"]),
        retained,
        sorted(duplicate_group_sources),
    )


def _record_rank(record: dict[str, Any]) -> tuple[int, int, int, str, str, str]:
    return (
        -INSPECTION_RANK[record["inspection_depth"]],
        -int(record["relevance_score"]),
        -EVIDENCE_RANK[record["evidence_strength"]],
        str(record["record_id"]),
        str(record["seat"]),
        json.dumps(record, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )


def _save_manifest(root: Path, manifest: dict[str, Any]) -> None:
    manifest["updated_at"] = _utc_now()
    _write_private_text(
        root / "manifest.json",
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
    )


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    content = "\n".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records
    )
    _write_private_text(path, content + ("\n" if content else ""))


def _write_private_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _file_hash(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _as_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


__all__ = [
    "CouncilMergeReport",
    "CouncilRunError",
    "initialize_run",
    "load_manifest",
    "merge_run",
    "record_seat_artifact",
    "summarize_run",
]
