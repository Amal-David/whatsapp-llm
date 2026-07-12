import json
import re
from hashlib import sha256
from pathlib import Path

import pytest

from living_brain.research.corpus import load_jsonl
from living_brain.research.synthesis import (
    EvidenceMapValidationError,
    build_corpus_manifest,
    summarize_corpus,
    validate_evidence_map,
)

ROOT = Path(__file__).parents[1]
RESEARCH_ROOT = ROOT / "research" / "digital-self-council"


def test_committed_corpus_summary_is_deterministic_and_tracks_merged_provenance():
    records = load_jsonl(RESEARCH_ROOT / "corpus.jsonl")

    summary = summarize_corpus(records)

    assert summary["schema_version"] == "digital_self_corpus_summary.v1"
    assert summary["total_records"] == 187
    assert summary["counted_primary_papers"] == 187
    assert summary["reviewed_records"] == 187
    assert summary["inspection_depth_counts"] == {
        "abstract": 87,
        "full_text": 100,
    }
    assert summary["duplicate_merged_records"] == 13
    assert summary["contributing_seat_counts"]["evaluation_fidelity"] == 20
    assert summary["taxonomy_tag_counts"]["evaluation"] > 0
    assert summary == summarize_corpus(records)


def test_evidence_map_requires_known_papers_and_grounded_statuses():
    records = load_jsonl(RESEARCH_ROOT / "corpus.jsonl")
    supporting_id = records[0]["record_id"]
    contradicting_id = records[1]["record_id"]
    evidence_map = {
        "schema_version": "digital_self_evidence_map.v1",
        "corpus_sha256": "a" * 64,
        "claims": [
            {
                "id": "claim:separate-evidence-from-inference",
                "domain": "epistemics",
                "claim": "Store observed evidence separately from inferred identity.",
                "status": "validated",
                "supporting_paper_ids": [supporting_id],
                "contradicting_paper_ids": [contradicting_id],
                "architecture_decisions": ["Use a provenance-bearing evidence ledger."],
                "confidence_rationale": "Directly constrained by reviewed evidence.",
            }
        ],
    }

    report = validate_evidence_map(evidence_map, records)

    assert report == {
        "claim_count": 1,
        "status_counts": {"validated": 1},
        "cited_paper_count": 2,
        "uncited_corpus_papers": 185,
    }

    unknown = json.loads(json.dumps(evidence_map))
    unknown["claims"][0]["supporting_paper_ids"] = ["paper:not-in-corpus"]
    with pytest.raises(EvidenceMapValidationError, match="unknown paper ids"):
        validate_evidence_map(unknown, records)

    unsupported = json.loads(json.dumps(evidence_map))
    unsupported["claims"][0]["supporting_paper_ids"] = []
    with pytest.raises(EvidenceMapValidationError, match="requires supporting papers"):
        validate_evidence_map(unsupported, records)


def test_corpus_manifest_pins_every_reviewed_seat_and_contract_input():
    manifest = build_corpus_manifest(RESEARCH_ROOT)

    assert manifest["schema_version"] == "digital_self_corpus_manifest.v1"
    assert manifest["merged_corpus"]["record_count"] == 187
    assert len(manifest["merged_corpus"]["sha256"]) == 64
    assert len(manifest["contracts"]["council_sha256"]) == 64
    assert len(manifest["contracts"]["paper_schema_sha256"]) == 64
    assert len(manifest["seats"]) == 10
    assert {seat["seat"] for seat in manifest["seats"]} == {
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
    }
    assert all(seat["record_count"] == 20 for seat in manifest["seats"])
    assert all(len(seat["reviewers"]) == 1 for seat in manifest["seats"])
    assert all(len(seat["review_sha256"]) == 64 for seat in manifest["seats"])
    assert manifest == build_corpus_manifest(RESEARCH_ROOT)


def test_committed_synthesis_artifacts_are_grounded_in_the_pinned_corpus():
    records = load_jsonl(RESEARCH_ROOT / "corpus.jsonl")
    known_paper_ids = {record["record_id"] for record in records}
    corpus_hash = sha256((RESEARCH_ROOT / "corpus.jsonl").read_bytes()).hexdigest()

    summary = json.loads(
        (RESEARCH_ROOT / "corpus-summary.json").read_text(encoding="utf-8")
    )
    manifest = json.loads(
        (RESEARCH_ROOT / "corpus-manifest.json").read_text(encoding="utf-8")
    )
    evidence_map = json.loads(
        (RESEARCH_ROOT / "evidence-map.json").read_text(encoding="utf-8")
    )

    assert summary == summarize_corpus(records)
    assert manifest == build_corpus_manifest(RESEARCH_ROOT)
    assert evidence_map["corpus_sha256"] == corpus_hash
    assert validate_evidence_map(evidence_map, records)["claim_count"] >= 30

    for name in (
        "taxonomy.md",
        "contradictions.md",
        "elicitation-map.md",
        "synthesis.md",
    ):
        content = (RESEARCH_ROOT / name).read_text(encoding="utf-8")
        cited = set(re.findall(r"paper:[a-z0-9][a-z0-9._-]+", content))
        assert cited
        assert cited <= known_paper_ids

    drafts = sorted((RESEARCH_ROOT / "synthesis-drafts").glob("*.md"))
    assert len(drafts) == 10
    for draft in drafts:
        content = draft.read_text(encoding="utf-8")
        cited = set(re.findall(r"paper:[a-z0-9][a-z0-9._-]+", content))
        assert cited
        assert cited <= known_paper_ids
        assert "VALIDATED" in content
        assert "PLAUSIBLE" in content
        assert "SPECULATIVE" in content
