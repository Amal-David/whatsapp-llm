import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

from living_brain.main import main
from living_brain.research.corpus import (
    COUNCIL_SEATS,
    CorpusValidationError,
    load_jsonl,
    normalized_identifier,
)
from living_brain.research.council import (
    CouncilRunError,
    _deduplicate_records,
    initialize_run,
    load_manifest,
    merge_run,
    record_seat_artifact,
    summarize_run,
)

ROOT = Path(__file__).parents[1]
RESEARCH_ROOT = ROOT / "research" / "digital-self-council"


def _record(index, seat, *, canonical_value=None):
    timestamp = datetime(2026, 7, 11, tzinfo=timezone.utc).isoformat()
    canonical_value = canonical_value or f"10.2000/{seat}.{index}"
    return {
        "schema_version": "digital_self_paper.v1",
        "record_id": f"paper:{seat}-{index:02d}",
        "canonical_id": {"type": "doi", "value": canonical_value},
        "alternate_ids": [],
        "title": f"Council fixture {seat} {index}",
        "authors": ["Fixture Researcher"],
        "year": 2025,
        "venue": "Council Test Proceedings",
        "publication_type": "conference",
        "primary_url": f"https://doi.org/{canonical_value}",
        "seat": seat,
        "taxonomy_tags": ["user_modeling", "evaluation"],
        "research_question": "Can a council artifact be tracked reproducibly?",
        "method": "Deterministic test fixture.",
        "population_or_data": "No human participants or private data.",
        "modalities": ["text"],
        "core_finding": "Complete artifacts can be hashed and merged deterministically.",
        "architecture_implications": ["Track every seat artifact by content hash."],
        "limitations": "Synthetic fixture; not research evidence.",
        "evidence_strength": "limited",
        "relevance_score": 3,
        "inspection_depth": "abstract",
        "counted": True,
        "provenance": {
            "extractor": f"council-{seat}",
            "extracted_at": timestamp,
            "reviewer": None,
            "reviewed_at": None,
            "search_queries": [f"fixture query {seat}"],
        },
    }


def _write_seat(path, seat, *, duplicate_canonical=None, reviewer=None):
    records = [
        _record(
            index,
            seat,
            canonical_value=duplicate_canonical if index == 0 else None,
        )
        for index in range(15)
    ]
    if reviewer:
        for record in records:
            record["provenance"]["reviewer"] = reviewer
            record["provenance"]["reviewed_at"] = datetime(
                2026,
                7,
                11,
                tzinfo=timezone.utc,
            ).isoformat()
    path.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    return records


def test_initialization_is_resumable_and_pins_research_contract_hashes(tmp_path):
    run_dir = tmp_path / "run"
    created_at = datetime(2026, 7, 11, tzinfo=timezone.utc)
    manifest = initialize_run(
        run_dir,
        run_id="council-test",
        council_path=RESEARCH_ROOT / "council.yaml",
        schema_path=RESEARCH_ROOT / "paper.schema.json",
        created_at=created_at,
    )

    assert manifest["schema_version"] == "digital_self_council_run.v1"
    assert manifest["run_id"] == "council-test"
    assert manifest["created_at"] == created_at.isoformat()
    assert set(manifest["seats"]) == set(COUNCIL_SEATS)
    assert {seat["status"] for seat in manifest["seats"].values()} == {"pending"}
    assert len(manifest["inputs"]["council_sha256"]) == 64
    assert len(manifest["inputs"]["paper_schema_sha256"]) == 64

    artifact = tmp_path / "psychometrics.jsonl"
    _write_seat(artifact, "psychometrics")
    record_seat_artifact(
        run_dir,
        seat_id="psychometrics",
        artifact_path=artifact,
        agent_id="agent-psychometrics",
        query_strategy=["personality inference language primary paper"],
    )

    resumed = initialize_run(
        run_dir,
        run_id="council-test",
        council_path=RESEARCH_ROOT / "council.yaml",
        schema_path=RESEARCH_ROOT / "paper.schema.json",
        created_at=created_at,
    )
    assert resumed["seats"]["psychometrics"]["status"] == "complete"
    assert resumed["seats"]["psychometrics"]["record_count"] == 15

    changed_schema = tmp_path / "changed.schema.json"
    changed_schema.write_text("{}\n", encoding="utf-8")
    with pytest.raises(CouncilRunError, match="input hashes"):
        initialize_run(
            run_dir,
            run_id="council-test",
            council_path=RESEARCH_ROOT / "council.yaml",
            schema_path=changed_schema,
            created_at=created_at,
        )


def test_failed_seat_validation_is_recorded_and_never_counted_complete(tmp_path):
    run_dir = tmp_path / "run"
    initialize_run(
        run_dir,
        run_id="failed-seat",
        council_path=RESEARCH_ROOT / "council.yaml",
        schema_path=RESEARCH_ROOT / "paper.schema.json",
    )
    artifact = tmp_path / "bad.jsonl"
    records = _write_seat(artifact, "psychometrics")
    records[1]["canonical_id"] = records[0]["canonical_id"]
    artifact.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(CorpusValidationError, match="duplicate canonical paper"):
        record_seat_artifact(
            run_dir,
            seat_id="psychometrics",
            artifact_path=artifact,
            agent_id="agent-bad",
            query_strategy=["bad fixture"],
        )

    manifest = load_manifest(run_dir)
    seat = manifest["seats"]["psychometrics"]
    assert seat["status"] == "failed"
    assert seat["record_count"] == 0
    assert seat["errors"]
    assert summarize_run(run_dir)["completed_seats"] == 0


def test_merge_requires_every_seat_and_reconciles_cross_seat_duplicates(tmp_path):
    run_dir = tmp_path / "run"
    initialize_run(
        run_dir,
        run_id="merge-test",
        council_path=RESEARCH_ROOT / "council.yaml",
        schema_path=RESEARCH_ROOT / "paper.schema.json",
    )

    first_canonical = None
    same_id_canonical = None
    second_canonical = None
    for seat_index, seat in enumerate(COUNCIL_SEATS):
        artifact = tmp_path / f"{seat}.jsonl"
        records = _write_seat(artifact, seat, reviewer=f"reviewer-{seat}")
        if seat_index == 0:
            first_canonical = records[0]["canonical_id"]["value"]
            same_id_canonical = records[1]["canonical_id"]["value"]
            records[0]["inspection_depth"] = "full_text"
            records[0]["relevance_score"] = 5
        elif seat_index == 1:
            second_canonical = records[0]["canonical_id"]["value"]
            records[0]["alternate_ids"] = [
                {"type": "doi", "value": first_canonical}
            ]
        elif seat_index == 2:
            records[1]["record_id"] = "paper:psychometrics-01"
            records[1]["canonical_id"] = {
                "type": "doi",
                "value": same_id_canonical,
            }
            records[1]["primary_url"] = f"https://doi.org/{same_id_canonical}"
        artifact.write_text(
            "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
            encoding="utf-8",
        )
        record_seat_artifact(
            run_dir,
            seat_id=seat,
            artifact_path=artifact,
            agent_id=f"agent-{seat}",
            query_strategy=[f"primary query {seat}"],
        )

    output_path = tmp_path / "corpus.draft.jsonl"
    report = merge_run(run_dir, output_path=output_path)
    merged = load_jsonl(output_path)

    assert report.input_records == 150
    assert report.output_records == 148
    assert report.duplicate_groups == 2
    assert report.duplicate_records == 2
    assert {frozenset(group) for group in report.duplicate_group_sources} == {
        frozenset(
            {
                "psychometrics/paper:psychometrics-00",
                "narrative_identity/paper:narrative_identity-00",
            }
        ),
        frozenset(
            {
                "psychometrics/paper:psychometrics-01",
                "cognitive_memory/paper:psychometrics-01",
            }
        ),
    }
    assert len(merged) == 148
    assert output_path.with_name("corpus.draft-report.json").is_file()
    retained_id = report.retained_record_ids["paper:narrative_identity-00"]
    retained = next(record for record in merged if record["record_id"] == retained_id)
    retained_identifiers = {
        normalized_identifier(identifier)
        for identifier in [retained["canonical_id"], *retained["alternate_ids"]]
    }
    assert normalized_identifier({"type": "doi", "value": second_canonical}) in (
        retained_identifiers
    )
    assert retained["provenance"]["contributing_seats"] == [
        "narrative_identity",
        "psychometrics",
    ]
    assert summarize_run(run_dir)["completed_seats"] == 10
    assert summarize_run(run_dir)["merge_status"] == "complete"
    assert load_manifest(run_dir)["merge"]["output_sha256"] == hashlib.sha256(
        output_path.read_bytes()
    ).hexdigest()


def test_duplicate_winner_is_invariant_to_input_order_when_quality_ties():
    first = _record(0, "evaluation_fidelity", canonical_value="10.2000/shared")
    second = _record(0, "digital_twins_agents", canonical_value="10.2000/shared")
    for record in (first, second):
        record["record_id"] = "paper:shared-record"
    first["title"] = "Evaluation-fidelity description"
    second["title"] = "Digital-twins description"

    forward = _deduplicate_records([first, second])
    reverse = _deduplicate_records([second, first])

    assert forward == reverse
    assert forward[0][0]["seat"] == "digital_twins_agents"


def test_merge_refuses_to_silently_ignore_pending_seats(tmp_path):
    run_dir = tmp_path / "run"
    initialize_run(
        run_dir,
        run_id="partial-test",
        council_path=RESEARCH_ROOT / "council.yaml",
        schema_path=RESEARCH_ROOT / "paper.schema.json",
    )
    artifact = tmp_path / "psychometrics.jsonl"
    _write_seat(artifact, "psychometrics")
    record_seat_artifact(
        run_dir,
        seat_id="psychometrics",
        artifact_path=artifact,
        agent_id="agent-one",
        query_strategy=["one completed seat"],
    )

    with pytest.raises(CouncilRunError, match="9 seats are not complete"):
        merge_run(run_dir, output_path=tmp_path / "should-not-exist.jsonl")


def test_merge_refuses_to_overwrite_a_validated_seat_artifact(tmp_path):
    run_dir = tmp_path / "run"
    initialize_run(
        run_dir,
        run_id="collision-test",
        council_path=RESEARCH_ROOT / "council.yaml",
        schema_path=RESEARCH_ROOT / "paper.schema.json",
    )

    artifacts = {}
    for seat in COUNCIL_SEATS:
        artifact = tmp_path / f"{seat}.jsonl"
        _write_seat(artifact, seat, reviewer=f"reviewer-{seat}")
        artifacts[seat] = artifact
        record_seat_artifact(
            run_dir,
            seat_id=seat,
            artifact_path=artifact,
            agent_id=f"agent-{seat}",
            query_strategy=[f"primary query {seat}"],
        )

    protected_artifact = artifacts["psychometrics"]
    original_hash = hashlib.sha256(protected_artifact.read_bytes()).hexdigest()
    with pytest.raises(CouncilRunError, match="overwrite a seat artifact"):
        merge_run(run_dir, output_path=protected_artifact)

    assert hashlib.sha256(protected_artifact.read_bytes()).hexdigest() == original_hash


def test_merge_rejects_unreviewed_or_self_reviewed_evidence(tmp_path):
    run_dir = tmp_path / "run"
    initialize_run(
        run_dir,
        run_id="review-gate-test",
        council_path=RESEARCH_ROOT / "council.yaml",
        schema_path=RESEARCH_ROOT / "paper.schema.json",
    )

    for seat in COUNCIL_SEATS:
        artifact = tmp_path / f"{seat}.jsonl"
        reviewer = None if seat == "psychometrics" else f"reviewer-{seat}"
        _write_seat(artifact, seat, reviewer=reviewer)
        record_seat_artifact(
            run_dir,
            seat_id=seat,
            artifact_path=artifact,
            agent_id=f"agent-{seat}",
            query_strategy=[f"primary query {seat}"],
        )

    with pytest.raises(CorpusValidationError, match="must be reviewed"):
        merge_run(run_dir, output_path=tmp_path / "unreviewed.jsonl")

    psychometrics = tmp_path / "psychometrics.jsonl"
    records = load_jsonl(psychometrics)
    for record in records:
        record["provenance"]["reviewer"] = "agent-psychometrics"
        record["provenance"]["reviewed_at"] = datetime(
            2026,
            7,
            11,
            tzinfo=timezone.utc,
        ).isoformat()
    psychometrics.write_text(
        "\n".join(json.dumps(record, sort_keys=True) for record in records) + "\n",
        encoding="utf-8",
    )
    record_seat_artifact(
        run_dir,
        seat_id="psychometrics",
        artifact_path=psychometrics,
        agent_id="agent-psychometrics",
        query_strategy=["primary query psychometrics"],
    )

    with pytest.raises(CouncilRunError, match="review its own evidence"):
        merge_run(run_dir, output_path=tmp_path / "self-reviewed.jsonl")


def _run_cli(monkeypatch, *arguments):
    monkeypatch.setattr(sys, "argv", ["living-brain", *map(str, arguments)])
    main()


def test_research_cli_initializes_validates_and_reports_a_run(
    tmp_path,
    monkeypatch,
    capsys,
):
    run_dir = tmp_path / "run"
    artifact = tmp_path / "psychometrics.jsonl"
    _write_seat(artifact, "psychometrics")

    _run_cli(
        monkeypatch,
        "research",
        "init",
        "--run-dir",
        run_dir,
        "--run-id",
        "cli-test",
        "--council",
        RESEARCH_ROOT / "council.yaml",
        "--schema",
        RESEARCH_ROOT / "paper.schema.json",
    )
    _run_cli(
        monkeypatch,
        "research",
        "validate-seat",
        "--run-dir",
        run_dir,
        "--seat",
        "psychometrics",
        "--input",
        artifact,
        "--agent-id",
        "cli-agent",
        "--query",
        "personality inference primary paper",
    )
    _run_cli(monkeypatch, "research", "status", "--run-dir", run_dir)

    output = capsys.readouterr().out
    assert "cli-test" in output
    assert '"completed_seats": 1' in output
    assert '"pending_seats": 9' in output
    assert '"status": "complete"' in output


def test_versioned_research_prompts_preserve_extraction_and_review_boundaries():
    seat_prompt = (RESEARCH_ROOT / "prompts" / "seat.md").read_text(encoding="utf-8")
    review_prompt = (RESEARCH_ROOT / "prompts" / "cross-review.md").read_text(
        encoding="utf-8"
    )

    for placeholder in (
        "{seat_id}",
        "{owned_scope}",
        "{excluded_scope}",
        "{agent_id}",
        "{seat_output_path}",
    ):
        assert placeholder in seat_prompt
    assert "`reviewer`: `null`" in seat_prompt
    assert "Do not mark your own records reviewed" in seat_prompt
    assert "You did not extract these records" in review_prompt
    assert "approve/revise decision" in review_prompt
