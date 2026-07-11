"""Deterministic corpus summaries and grounded architecture-claim validation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .corpus import COUNCIL_SEATS, load_jsonl, validate_corpus

EVIDENCE_STATUSES = {"validated", "plausible", "speculative"}


class EvidenceMapValidationError(ValueError):
    """Raised when a synthesis claim is malformed or cites absent evidence."""


class CorpusManifestError(ValueError):
    """Raised when committed council artifacts are incomplete or inconsistent."""


def summarize_corpus(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return a deterministic, text-free evidence coverage summary."""
    report = validate_corpus(
        records,
        min_count=0,
        max_count=max(200, len(records)),
        require_reviewed=False,
    )
    evidence_strengths: Counter[str] = Counter()
    relevance_scores: Counter[str] = Counter()
    taxonomy_tags: Counter[str] = Counter()
    modalities: Counter[str] = Counter()
    years: Counter[str] = Counter()
    decades: Counter[str] = Counter()
    contributing_seats: Counter[str] = Counter()
    duplicate_merged_records = 0
    architecture_implication_count = 0

    for record in records:
        evidence_strengths[str(record["evidence_strength"])] += 1
        relevance_scores[str(record["relevance_score"])] += 1
        year = int(record["year"])
        years[str(year)] += 1
        decades[f"{year // 10 * 10}s"] += 1
        taxonomy_tags.update(str(tag) for tag in record["taxonomy_tags"])
        modalities.update(str(modality) for modality in record["modalities"])
        architecture_implication_count += len(record["architecture_implications"])

        merged_seats = record["provenance"].get("contributing_seats")
        if merged_seats:
            duplicate_merged_records += 1
            contributing_seats.update(str(seat) for seat in merged_seats)
        else:
            contributing_seats[str(record["seat"])] += 1

    return {
        "schema_version": "digital_self_corpus_summary.v1",
        "total_records": report.total_records,
        "counted_primary_papers": report.counted_primary_papers,
        "unique_canonical_papers": report.unique_canonical_papers,
        "reviewed_records": report.reviewed_records,
        "duplicate_merged_records": duplicate_merged_records,
        "architecture_implication_count": architecture_implication_count,
        "publication_type_counts": report.publication_type_counts,
        "inspection_depth_counts": report.inspection_depth_counts,
        "retained_seat_counts": report.seat_counts,
        "contributing_seat_counts": _complete_counts(
            contributing_seats,
            COUNCIL_SEATS,
        ),
        "evidence_strength_counts": _sorted_counts(evidence_strengths),
        "relevance_score_counts": _sorted_counts(relevance_scores),
        "taxonomy_tag_counts": _sorted_counts(taxonomy_tags),
        "modality_label_count": len(modalities),
        "top_modality_counts": _top_counts(modalities, limit=25),
        "year_counts": _sorted_counts(years),
        "decade_counts": _sorted_counts(decades),
    }


def build_corpus_manifest(root: str | Path) -> dict[str, Any]:
    """Pin the public council contracts, seats, reviews, and merged corpus."""
    council_root = Path(root)
    council_path = council_root / "council.yaml"
    schema_path = council_root / "paper.schema.json"
    corpus_path = council_root / "corpus.jsonl"
    corpus_records = load_jsonl(corpus_path)
    corpus_report = validate_corpus(corpus_records)

    seats = []
    for seat in COUNCIL_SEATS:
        seat_path = council_root / "seats" / f"{seat}.jsonl"
        note_path = council_root / "seat-notes" / f"{seat}.md"
        review_paths = sorted(
            (council_root / "reviews").glob(f"{seat}.reviewed-by-*.md")
        )
        if len(review_paths) != 1:
            raise CorpusManifestError(
                f"seat {seat} must have exactly one cross-review artifact"
            )
        review_path = review_paths[0]
        records = load_jsonl(seat_path)
        report = validate_corpus(
            records,
            min_count=15,
            max_count=25,
            require_reviewed=True,
        )
        if report.seat_counts != {seat: report.counted_primary_papers}:
            raise CorpusManifestError(f"seat artifact contains records outside {seat}")
        reviewers = sorted(
            {str(record["provenance"]["reviewer"]) for record in records}
        )
        extractors = sorted(
            {str(record["provenance"]["extractor"]) for record in records}
        )
        seats.append(
            {
                "seat": seat,
                "path": seat_path.relative_to(council_root).as_posix(),
                "sha256": _sha256_file(seat_path),
                "record_count": report.counted_primary_papers,
                "inspection_depth_counts": report.inspection_depth_counts,
                "extractors": extractors,
                "reviewers": reviewers,
                "note_path": note_path.relative_to(council_root).as_posix(),
                "note_sha256": _sha256_file(note_path),
                "review_path": review_path.relative_to(council_root).as_posix(),
                "review_sha256": _sha256_file(review_path),
            }
        )

    merged_corpus: dict[str, Any] = {
        "path": corpus_path.relative_to(council_root).as_posix(),
        "sha256": _sha256_file(corpus_path),
        "record_count": corpus_report.counted_primary_papers,
        "reviewed_records": corpus_report.reviewed_records,
        "inspection_depth_counts": corpus_report.inspection_depth_counts,
    }
    for key, filename in (
        ("report", "corpus-report.json"),
        ("summary", "corpus-summary.json"),
    ):
        artifact_path = council_root / filename
        if artifact_path.is_file():
            merged_corpus[f"{key}_path"] = filename
            merged_corpus[f"{key}_sha256"] = _sha256_file(artifact_path)

    evidence_map_path = council_root / "evidence-map.json"
    evidence_map: dict[str, Any] | None = None
    if evidence_map_path.is_file():
        value = json.loads(evidence_map_path.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            raise CorpusManifestError("evidence-map.json must contain an object")
        expected_corpus_hash = _sha256_file(corpus_path)
        if value.get("corpus_sha256") != expected_corpus_hash:
            raise CorpusManifestError("evidence map does not pin the merged corpus")
        validation = validate_evidence_map(value, corpus_records)
        evidence_map = {
            "path": "evidence-map.json",
            "sha256": _sha256_file(evidence_map_path),
            **validation,
        }

    return {
        "schema_version": "digital_self_corpus_manifest.v1",
        "contracts": {
            "council_path": "council.yaml",
            "council_sha256": _sha256_file(council_path),
            "paper_schema_path": "paper.schema.json",
            "paper_schema_sha256": _sha256_file(schema_path),
        },
        "merged_corpus": merged_corpus,
        "seats": seats,
        "evidence_map": evidence_map,
    }


def validate_evidence_map(
    evidence_map: Mapping[str, Any],
    corpus_records: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate claim grounding against one exact merged corpus snapshot."""
    if evidence_map.get("schema_version") != "digital_self_evidence_map.v1":
        raise EvidenceMapValidationError("unsupported evidence-map schema_version")
    corpus_sha256 = evidence_map.get("corpus_sha256")
    if (
        not isinstance(corpus_sha256, str)
        or len(corpus_sha256) != 64
        or any(character not in "0123456789abcdef" for character in corpus_sha256)
    ):
        raise EvidenceMapValidationError("corpus_sha256 must be a lowercase SHA-256")

    known_papers = {str(record["record_id"]) for record in corpus_records}
    claims = evidence_map.get("claims")
    if not isinstance(claims, list) or not claims:
        raise EvidenceMapValidationError("evidence map must contain claims")

    claim_ids: set[str] = set()
    cited_papers: set[str] = set()
    status_counts: Counter[str] = Counter()
    for index, value in enumerate(claims, start=1):
        if not isinstance(value, Mapping):
            raise EvidenceMapValidationError(f"claim {index} must be an object")
        claim_id = value.get("id")
        if not _is_text(claim_id) or not str(claim_id).startswith("claim:"):
            raise EvidenceMapValidationError(f"claim {index} has an invalid id")
        if claim_id in claim_ids:
            raise EvidenceMapValidationError(f"duplicate claim id: {claim_id}")
        claim_ids.add(str(claim_id))

        for field_name in ("domain", "claim", "confidence_rationale"):
            if not _is_text(value.get(field_name)):
                raise EvidenceMapValidationError(
                    f"claim {claim_id} requires {field_name}"
                )
        status = value.get("status")
        if status not in EVIDENCE_STATUSES:
            raise EvidenceMapValidationError(f"claim {claim_id} has invalid status")

        supporting = _paper_id_list(value, "supporting_paper_ids", claim_id)
        contradicting = _paper_id_list(value, "contradicting_paper_ids", claim_id)
        if status in {"validated", "plausible"} and not supporting:
            raise EvidenceMapValidationError(
                f"claim {claim_id} status {status} requires supporting papers"
            )
        overlap = set(supporting) & set(contradicting)
        if overlap:
            raise EvidenceMapValidationError(
                f"claim {claim_id} cites papers as both support and contradiction"
            )
        unknown = (set(supporting) | set(contradicting)) - known_papers
        if unknown:
            raise EvidenceMapValidationError(
                f"claim {claim_id} cites unknown paper ids: {sorted(unknown)}"
            )

        decisions = value.get("architecture_decisions")
        if not isinstance(decisions, list) or not decisions or not all(
            _is_text(decision) for decision in decisions
        ):
            raise EvidenceMapValidationError(
                f"claim {claim_id} requires architecture_decisions"
            )
        status_counts[str(status)] += 1
        cited_papers.update(supporting)
        cited_papers.update(contradicting)

    return {
        "claim_count": len(claims),
        "status_counts": _sorted_counts(status_counts),
        "cited_paper_count": len(cited_papers),
        "uncited_corpus_papers": len(known_papers - cited_papers),
    }


def _paper_id_list(
    claim: Mapping[str, Any],
    field_name: str,
    claim_id: Any,
) -> list[str]:
    value = claim.get(field_name)
    if not isinstance(value, list) or not all(
        _is_text(paper_id) and str(paper_id).startswith("paper:") for paper_id in value
    ):
        raise EvidenceMapValidationError(
            f"claim {claim_id} {field_name} must contain paper ids"
        )
    normalized = [str(paper_id) for paper_id in value]
    if len(normalized) != len(set(normalized)):
        raise EvidenceMapValidationError(
            f"claim {claim_id} {field_name} contains duplicates"
        )
    return normalized


def _complete_counts(counter: Counter[str], keys: Sequence[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(keys)}


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return dict(sorted(counter.items()))


def _top_counts(counter: Counter[str], *, limit: int) -> dict[str, int]:
    ranked = sorted(counter.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return dict(ranked)


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _is_text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


__all__ = [
    "EVIDENCE_STATUSES",
    "CorpusManifestError",
    "EvidenceMapValidationError",
    "build_corpus_manifest",
    "summarize_corpus",
    "validate_evidence_map",
]
