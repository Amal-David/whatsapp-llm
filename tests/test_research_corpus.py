import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

import pytest
import yaml

from living_brain.research.corpus import (
    COUNCIL_SEATS,
    TAXONOMY_TAGS,
    CorpusValidationError,
    load_jsonl,
    validate_corpus,
)

ROOT = Path(__file__).parents[1]
RESEARCH_ROOT = ROOT / "research" / "digital-self-council"


def _record(index, *, seat="psychometrics", counted=True):
    now = datetime(2026, 7, 11, 0, 0, tzinfo=timezone.utc).isoformat()
    return {
        "schema_version": "digital_self_paper.v1",
        "record_id": f"paper:test-{index:03d}",
        "canonical_id": {"type": "doi", "value": f"10.1000/test.{index}"},
        "alternate_ids": [],
        "title": f"Synthetic validation paper {index}",
        "authors": ["Researcher One", "Researcher Two"],
        "year": 2020 + (index % 6),
        "venue": "Validation Conference",
        "publication_type": "conference",
        "primary_url": f"https://doi.org/10.1000/test.{index}",
        "seat": seat,
        "taxonomy_tags": ["personality", "behavioral_inference"],
        "research_question": "Can this fixture exercise the corpus contract?",
        "method": "Synthetic test fixture; not real research evidence.",
        "population_or_data": "No human data; deterministic fixture.",
        "modalities": ["text"],
        "core_finding": "The validator accepts complete structurally valid records.",
        "architecture_implications": [
            "Require explicit evidence metadata before a source can be counted."
        ],
        "limitations": "This is a test fixture and must never enter the research corpus.",
        "evidence_strength": "limited",
        "relevance_score": 3,
        "inspection_depth": "full_text",
        "counted": counted,
        "provenance": {
            "extractor": "fixture-agent",
            "extracted_at": now,
            "reviewer": "fixture-reviewer",
            "reviewed_at": now,
            "search_queries": ["deterministic fixture"],
        },
    }


def test_schema_and_council_charter_match_runtime_contract():
    schema = json.loads((RESEARCH_ROOT / "paper.schema.json").read_text(encoding="utf-8"))
    council = yaml.safe_load((RESEARCH_ROOT / "council.yaml").read_text(encoding="utf-8"))

    schema_seats = set(schema["properties"]["seat"]["enum"])
    schema_tags = set(schema["properties"]["taxonomy_tags"]["items"]["enum"])
    charter_seats = {seat["id"] for seat in council["seats"]}

    assert schema["$id"].endswith("digital_self_paper.v1.schema.json")
    assert schema_seats == set(COUNCIL_SEATS) == charter_seats
    assert schema_tags == set(TAXONOMY_TAGS)
    assert len(council["seats"]) == 10
    assert all(seat["cross_review_seat"] in charter_seats for seat in council["seats"])


def test_validator_accepts_a_reviewed_100_paper_primary_corpus():
    records = [_record(index) for index in range(100)]

    report = validate_corpus(records)

    assert report.total_records == 100
    assert report.counted_primary_papers == 100
    assert report.unique_canonical_papers == 100
    assert report.reviewed_records == 100
    assert report.seat_counts == {"psychometrics": 100}


def test_validator_normalizes_identifiers_before_duplicate_detection():
    first = _record(1)
    duplicate = _record(2)
    duplicate["canonical_id"] = {
        "type": "doi",
        "value": "https://doi.org/10.1000/TEST.1",
    }

    with pytest.raises(CorpusValidationError, match="duplicate canonical paper"):
        validate_corpus([first, duplicate], min_count=1)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda record: record.pop("primary_url"), "primary_url"),
        (
            lambda record: record.update(publication_type="website"),
            "publication_type",
        ),
        (
            lambda record: record.update(relevance_score=6),
            "relevance_score",
        ),
        (
            lambda record: record["provenance"].update(reviewer=None, reviewed_at=None),
            "reviewed",
        ),
    ],
)
def test_validator_rejects_malformed_or_unreviewed_counted_records(mutation, message):
    record = _record(1)
    mutation(record)

    with pytest.raises(CorpusValidationError, match=message):
        validate_corpus([record], min_count=1)


def test_validator_enforces_the_requested_corpus_size_range():
    with pytest.raises(CorpusValidationError, match="at least 100"):
        validate_corpus([_record(index) for index in range(99)])

    with pytest.raises(CorpusValidationError, match="at most 200"):
        validate_corpus([_record(index) for index in range(201)])


def test_uncounted_theory_sources_do_not_satisfy_the_paper_floor():
    theory = _record(999, counted=False)
    theory.update(publication_type="book")

    with pytest.raises(CorpusValidationError, match="at least 1"):
        validate_corpus([theory], min_count=1)


def test_jsonl_loader_reports_line_numbers_and_keeps_input_immutable(tmp_path):
    records = [_record(1), _record(2)]
    original = deepcopy(records)
    valid_path = tmp_path / "valid.jsonl"
    valid_path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )

    loaded = load_jsonl(valid_path)
    validate_corpus(loaded, min_count=2)

    assert records == original

    invalid_path = tmp_path / "invalid.jsonl"
    invalid_path.write_text("{}\n{broken\n", encoding="utf-8")
    with pytest.raises(CorpusValidationError, match="line 2"):
        load_jsonl(invalid_path)
