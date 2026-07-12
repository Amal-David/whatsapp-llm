"""Validation and deduplication contracts for digital-self research evidence."""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

COUNCIL_SEATS = (
    "psychometrics",
    "narrative_identity",
    "cognitive_memory",
    "persona_dialogue",
    "social_relationships",
    "longitudinal_modeling",
    "values_decisions_emotion",
    "evaluation_fidelity",
    "privacy_identity_safety",
    "digital_twins_agents",
)

TAXONOMY_TAGS = (
    "active_learning",
    "affect",
    "autobiographical_memory",
    "behavioral_fidelity",
    "behavioral_inference",
    "calibration",
    "cognitive_architecture",
    "consent",
    "continual_learning",
    "decision_making",
    "digital_twin",
    "dialogue",
    "evaluation",
    "forgetting",
    "goals",
    "identity_rights",
    "lifelogging",
    "longitudinal_change",
    "memory_consolidation",
    "narrative_identity",
    "persona",
    "personality",
    "privacy",
    "relationship_modeling",
    "roleplay",
    "self_schema",
    "social_identity",
    "theory_of_mind",
    "uncertainty",
    "user_modeling",
    "values",
)

CANONICAL_ID_TYPES = {"doi", "arxiv", "acl", "pubmed", "title_year", "isbn"}
PRIMARY_PUBLICATION_TYPES = {"journal", "conference", "workshop", "preprint"}
PUBLICATION_TYPES = PRIMARY_PUBLICATION_TYPES | {
    "book",
    "chapter",
    "report",
    "thesis",
}
EVIDENCE_STRENGTHS = {"foundational", "strong", "moderate", "limited", "theoretical"}
INSPECTION_DEPTHS = {"full_text", "abstract", "metadata"}
REQUIRED_TEXT_FIELDS = (
    "title",
    "venue",
    "research_question",
    "method",
    "population_or_data",
    "core_finding",
    "limitations",
)


class CorpusValidationError(ValueError):
    """Raised when research evidence cannot be counted or trusted."""


@dataclass(frozen=True)
class CorpusReport:
    """Aggregate, text-free validation result for one corpus snapshot."""

    total_records: int
    counted_primary_papers: int
    unique_canonical_papers: int
    reviewed_records: int
    seat_counts: dict[str, int]
    publication_type_counts: dict[str, int]
    inspection_depth_counts: dict[str, int]

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_records": self.total_records,
            "counted_primary_papers": self.counted_primary_papers,
            "unique_canonical_papers": self.unique_canonical_papers,
            "reviewed_records": self.reviewed_records,
            "seat_counts": self.seat_counts,
            "publication_type_counts": self.publication_type_counts,
            "inspection_depth_counts": self.inspection_depth_counts,
        }


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Load JSON objects while preserving useful source line errors."""
    records = []
    for line_number, raw_line in enumerate(
        Path(path).read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not raw_line.strip():
            continue
        try:
            value = json.loads(raw_line)
        except json.JSONDecodeError as error:
            raise CorpusValidationError(
                f"invalid JSON on line {line_number}: {error.msg}"
            ) from error
        if not isinstance(value, dict):
            raise CorpusValidationError(
                f"line {line_number} must contain a JSON object"
            )
        records.append(value)
    return records


def normalized_identifier(identifier: Mapping[str, Any]) -> str:
    """Return the canonical comparison key for one paper identifier."""
    return _normalized_identifier(identifier)


def paper_identifier_keys(record: Mapping[str, Any]) -> set[str]:
    """Return canonical and alternate keys used for cross-seat deduplication."""
    return {
        normalized_identifier(identifier)
        for identifier in [record["canonical_id"], *record.get("alternate_ids", [])]
    }


def validate_corpus(
    records: Sequence[Mapping[str, Any]],
    *,
    min_count: int = 100,
    max_count: int = 200,
    require_reviewed: bool = True,
) -> CorpusReport:
    """Validate records, deduplicate every identifier, and enforce paper gates."""
    if min_count < 0 or max_count < min_count:
        raise ValueError("invalid corpus count bounds")

    seen_record_ids: set[str] = set()
    identifier_owner: dict[str, str] = {}
    canonical_papers: set[str] = set()
    counted_records = []
    seat_counts: Counter[str] = Counter()
    publication_counts: Counter[str] = Counter()
    inspection_counts: Counter[str] = Counter()
    reviewed_records = 0

    for index, original in enumerate(records, start=1):
        record = dict(original)
        _validate_record(record, index=index, require_reviewed=require_reviewed)

        record_id = str(record["record_id"])
        if record_id in seen_record_ids:
            raise CorpusValidationError(f"duplicate record_id: {record_id}")
        seen_record_ids.add(record_id)

        canonical = _normalized_identifier(record["canonical_id"])
        aliases = [
            _normalized_identifier(identifier)
            for identifier in record.get("alternate_ids", [])
        ]
        all_identifiers = [canonical, *aliases]
        if len(all_identifiers) != len(set(all_identifiers)):
            raise CorpusValidationError(
                f"record {record_id} repeats a canonical or alternate identifier"
            )
        for identifier in all_identifiers:
            existing = identifier_owner.get(identifier)
            if existing is not None:
                raise CorpusValidationError(
                    "duplicate canonical paper: "
                    f"{record_id} and {existing} share {identifier}"
                )
            identifier_owner[identifier] = record_id
        canonical_papers.add(canonical)

        if record["provenance"].get("reviewer"):
            reviewed_records += 1
        if record["counted"]:
            counted_records.append(record)
            seat_counts[str(record["seat"])] += 1
        publication_counts[str(record["publication_type"])] += 1
        inspection_counts[str(record["inspection_depth"])] += 1

    counted_count = len(counted_records)
    if counted_count < min_count:
        raise CorpusValidationError(
            f"counted primary corpus must contain at least {min_count} papers; "
            f"found {counted_count}"
        )
    if counted_count > max_count:
        raise CorpusValidationError(
            f"counted primary corpus may contain at most {max_count} papers; "
            f"found {counted_count}"
        )

    return CorpusReport(
        total_records=len(records),
        counted_primary_papers=counted_count,
        unique_canonical_papers=len(canonical_papers),
        reviewed_records=reviewed_records,
        seat_counts=dict(sorted(seat_counts.items())),
        publication_type_counts=dict(sorted(publication_counts.items())),
        inspection_depth_counts=dict(sorted(inspection_counts.items())),
    )


def _validate_record(
    record: Mapping[str, Any],
    *,
    index: int,
    require_reviewed: bool,
) -> None:
    prefix = f"record {index}"
    required = {
        "schema_version",
        "record_id",
        "canonical_id",
        "alternate_ids",
        "title",
        "authors",
        "year",
        "venue",
        "publication_type",
        "primary_url",
        "seat",
        "taxonomy_tags",
        "research_question",
        "method",
        "population_or_data",
        "modalities",
        "core_finding",
        "architecture_implications",
        "limitations",
        "evidence_strength",
        "relevance_score",
        "inspection_depth",
        "counted",
        "provenance",
    }
    missing = sorted(required - set(record))
    if missing:
        raise CorpusValidationError(f"{prefix} missing required fields: {missing}")

    if record["schema_version"] != "digital_self_paper.v1":
        raise CorpusValidationError(f"{prefix} has unsupported schema_version")
    if not _is_text(record["record_id"]) or not str(record["record_id"]).startswith(
        "paper:"
    ):
        raise CorpusValidationError(f"{prefix} record_id must start with paper:")

    _normalized_identifier(record["canonical_id"])
    alternate_ids = record["alternate_ids"]
    if not isinstance(alternate_ids, list):
        raise CorpusValidationError(f"{prefix} alternate_ids must be a list")
    for identifier in alternate_ids:
        _normalized_identifier(identifier)

    for field_name in REQUIRED_TEXT_FIELDS:
        if not _is_text(record[field_name]):
            raise CorpusValidationError(f"{prefix} {field_name} cannot be empty")

    authors = record["authors"]
    if not isinstance(authors, list) or not authors or not all(
        _is_text(author) for author in authors
    ):
        raise CorpusValidationError(f"{prefix} authors must be a non-empty text list")

    year = record["year"]
    if isinstance(year, bool) or not isinstance(year, int) or not 1900 <= year <= 2100:
        raise CorpusValidationError(f"{prefix} year must be between 1900 and 2100")

    publication_type = record["publication_type"]
    if publication_type not in PUBLICATION_TYPES:
        raise CorpusValidationError(
            f"{prefix} publication_type must be one of {sorted(PUBLICATION_TYPES)}"
        )

    if not _is_https_url(record["primary_url"]):
        raise CorpusValidationError(f"{prefix} primary_url must be an HTTPS URL")

    seat = record["seat"]
    if seat not in COUNCIL_SEATS:
        raise CorpusValidationError(f"{prefix} seat must be a known council seat")

    tags = record["taxonomy_tags"]
    if not isinstance(tags, list) or not tags:
        raise CorpusValidationError(f"{prefix} taxonomy_tags must be non-empty")
    unknown_tags = sorted(set(tags) - set(TAXONOMY_TAGS))
    if unknown_tags:
        raise CorpusValidationError(f"{prefix} has unknown taxonomy_tags: {unknown_tags}")
    if len(tags) != len(set(tags)):
        raise CorpusValidationError(f"{prefix} taxonomy_tags cannot contain duplicates")

    modalities = record["modalities"]
    if not isinstance(modalities, list) or not modalities or not all(
        _is_text(modality) for modality in modalities
    ):
        raise CorpusValidationError(f"{prefix} modalities must be a non-empty text list")

    implications = record["architecture_implications"]
    if not isinstance(implications, list) or not implications or not all(
        _is_text(implication) for implication in implications
    ):
        raise CorpusValidationError(
            f"{prefix} architecture_implications must be a non-empty text list"
        )

    if record["evidence_strength"] not in EVIDENCE_STRENGTHS:
        raise CorpusValidationError(f"{prefix} evidence_strength is invalid")
    score = record["relevance_score"]
    if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
        raise CorpusValidationError(f"{prefix} relevance_score must be between 1 and 5")
    if record["inspection_depth"] not in INSPECTION_DEPTHS:
        raise CorpusValidationError(f"{prefix} inspection_depth is invalid")
    if not isinstance(record["counted"], bool):
        raise CorpusValidationError(f"{prefix} counted must be boolean")
    if record["counted"] and publication_type not in PRIMARY_PUBLICATION_TYPES:
        raise CorpusValidationError(
            f"{prefix} publication_type {publication_type!r} cannot count as a primary paper"
        )

    provenance = record["provenance"]
    if not isinstance(provenance, Mapping):
        raise CorpusValidationError(f"{prefix} provenance must be an object")
    if not _is_text(provenance.get("extractor")):
        raise CorpusValidationError(f"{prefix} provenance.extractor is required")
    _validate_iso_datetime(provenance.get("extracted_at"), f"{prefix} extracted_at")
    queries = provenance.get("search_queries")
    if not isinstance(queries, list) or not queries or not all(
        _is_text(query) for query in queries
    ):
        raise CorpusValidationError(
            f"{prefix} provenance.search_queries must be a non-empty text list"
        )
    contributing_seats = provenance.get("contributing_seats")
    merged_record_ids = provenance.get("merged_record_ids")
    if (contributing_seats is None) != (merged_record_ids is None):
        raise CorpusValidationError(
            f"{prefix} merged provenance fields must be provided together"
        )
    if contributing_seats is not None:
        if (
            not isinstance(contributing_seats, list)
            or len(contributing_seats) < 2
            or not all(_is_text(seat_name) for seat_name in contributing_seats)
            or len(contributing_seats) != len(set(contributing_seats))
            or set(contributing_seats) - set(COUNCIL_SEATS)
        ):
            raise CorpusValidationError(
                f"{prefix} provenance.contributing_seats must contain unique council seats"
            )
        if (
            not isinstance(merged_record_ids, list)
            or not merged_record_ids
            or not all(_is_text(record_id) for record_id in merged_record_ids)
            or len(merged_record_ids) != len(set(merged_record_ids))
            or not all(
                str(record_id).startswith("paper:")
                for record_id in merged_record_ids
            )
        ):
            raise CorpusValidationError(
                f"{prefix} provenance.merged_record_ids must contain unique paper ids"
            )
    reviewer = provenance.get("reviewer")
    reviewed_at = provenance.get("reviewed_at")
    if require_reviewed and record["counted"]:
        if not _is_text(reviewer) or not _is_text(reviewed_at):
            raise CorpusValidationError(
                f"{prefix} counted records must be reviewed before final validation"
            )
    if reviewer is not None and not _is_text(reviewer):
        raise CorpusValidationError(f"{prefix} provenance.reviewer must be text or null")
    if reviewed_at is not None:
        _validate_iso_datetime(reviewed_at, f"{prefix} reviewed_at")


def _normalized_identifier(identifier: Any) -> str:
    if not isinstance(identifier, Mapping):
        raise CorpusValidationError("canonical and alternate identifiers must be objects")
    identifier_type = identifier.get("type")
    raw_value = identifier.get("value")
    if identifier_type not in CANONICAL_ID_TYPES or not _is_text(raw_value):
        raise CorpusValidationError(
            "identifier type or value is invalid; expected a supported non-empty id"
        )
    value = str(raw_value).strip()
    if identifier_type == "doi":
        value = re.sub(
            r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
            "",
            value,
            flags=re.IGNORECASE,
        ).lower()
    elif identifier_type == "arxiv":
        value = re.sub(
            r"^(?:https?://arxiv\.org/(?:abs|pdf)/|arxiv:\s*)",
            "",
            value,
            flags=re.IGNORECASE,
        )
        value = re.sub(r"\.pdf$", "", value, flags=re.IGNORECASE)
        value = re.sub(r"v\d+$", "", value, flags=re.IGNORECASE).lower()
    elif identifier_type in {"acl", "pubmed"}:
        value = value.lower().strip()
    elif identifier_type == "isbn":
        value = re.sub(r"[^0-9xX]", "", value).lower()
    else:
        value = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    if not value:
        raise CorpusValidationError("normalized canonical identifier cannot be empty")
    return f"{identifier_type}:{value}"


def _validate_iso_datetime(value: Any, label: str) -> None:
    if not _is_text(value):
        raise CorpusValidationError(f"{label} must be an ISO-8601 timestamp")
    try:
        datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError as error:
        raise CorpusValidationError(f"{label} must be an ISO-8601 timestamp") from error


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _is_https_url(value: Any) -> bool:
    if not _is_text(value):
        return False
    parsed = urlparse(str(value))
    return parsed.scheme == "https" and bool(parsed.netloc)


__all__ = [
    "COUNCIL_SEATS",
    "TAXONOMY_TAGS",
    "CorpusReport",
    "CorpusValidationError",
    "load_jsonl",
    "normalized_identifier",
    "paper_identifier_keys",
    "validate_corpus",
]
