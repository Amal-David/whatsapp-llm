import json
import stat
import sys
from datetime import datetime, timedelta, timezone

import pytest

from living_brain.brain.demo import run_guided_demo
from living_brain.brain.models import BrainLayer, DigitalBrain
from living_brain.identity.models import DigitalSelfProfile
from living_brain.main import _aware_datetime, main

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
EXPECTED_AXES = {
    "behavioral",
    "temporal",
    "relationship",
    "autobiographical",
    "decision",
    "calibration",
    "privacy",
    "owner_judgment",
}


def test_cli_timestamp_parser_accepts_utc_z_designator():
    assert _aware_datetime("2026-07-11T12:00:00Z") == NOW


def test_guided_demo_exercises_the_complete_local_workflow(tmp_path):
    workspace = tmp_path / "guided-demo"

    summary = run_guided_demo(workspace, as_of=NOW)
    rerun = run_guided_demo(workspace, as_of=NOW)

    assert summary.to_dict() == rerun.to_dict()
    assert summary.status == "complete"
    assert summary.steps == (
        "source_selection",
        "coverage_analysis",
        "adaptive_interview",
        "build_and_version",
        "inspect_and_correct",
        "simulate",
        "evaluate",
    )
    assert set(summary.artifacts) == {
        "source_selection",
        "brain_initial",
        "coverage_before",
        "adaptive_interview",
        "brain_interviewed",
        "inspection_before_correction",
        "correction",
        "brain_final",
        "coverage_after",
        "simulation",
        "evaluation",
        "run_summary",
    }
    for path in summary.artifacts.values():
        assert path.is_file()
        assert stat.S_IMODE(path.stat().st_mode) == 0o600

    initial = DigitalBrain.load(summary.artifacts["brain_initial"])
    final = DigitalBrain.load(summary.artifacts["brain_final"])
    interview = json.loads(
        summary.artifacts["adaptive_interview"].read_text(encoding="utf-8")
    )
    inspection = json.loads(
        summary.artifacts["inspection_before_correction"].read_text(encoding="utf-8")
    )
    simulation = json.loads(
        summary.artifacts["simulation"].read_text(encoding="utf-8")
    )
    evaluation = json.loads(
        summary.artifacts["evaluation"].read_text(encoding="utf-8")
    )

    assert final.version > initial.version
    assert interview["questions"]
    assert all(question["answer"] for question in interview["questions"])
    assert "payload" not in json.dumps(inspection)
    assert "SYNTHETIC THIRD PARTY SECRET" not in json.dumps(inspection)
    assert simulation["selected_response"]
    assert simulation["authority_granted"] is False
    assert simulation["synthetic_disclosure"]
    assert set(evaluation["axis_scores"]) == EXPECTED_AXES
    assert evaluation["passed"] is True


def test_brain_guide_cli_prints_a_text_free_summary(tmp_path, monkeypatch, capsys):
    workspace = tmp_path / "cli-demo"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "living-brain",
            "brain",
            "guide",
            "--demo",
            "--workspace",
            str(workspace),
            "--as-of",
            NOW.isoformat(),
        ],
    )

    main()

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert output["mode"] == "synthetic_demo"
    assert output["brain_version"] > 1
    assert output["evaluation_passed"] is True
    assert "SYNTHETIC THIRD PARTY SECRET" not in json.dumps(output)


def test_brain_guide_requires_an_explicit_safe_source_mode(tmp_path, monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        ["living-brain", "brain", "guide", "--workspace", str(tmp_path)],
    )

    with pytest.raises(SystemExit, match="2"):
        main()


def test_brain_cli_reopens_generated_state_for_coverage_and_safe_inspection(
    tmp_path,
    monkeypatch,
    capsys,
):
    summary = run_guided_demo(tmp_path / "inspectable", as_of=NOW)
    brain_path = summary.artifacts["brain_final"]

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "living-brain",
            "brain",
            "coverage",
            str(brain_path),
            "--as-of",
            NOW.isoformat(),
        ],
    )
    main()
    coverage = json.loads(capsys.readouterr().out)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "living-brain",
            "brain",
            "inspect",
            str(brain_path),
            "--as-of",
            NOW.isoformat(),
            "--relationship-id",
            "relationship:demo",
            "--history",
        ],
    )
    main()
    inspection = json.loads(capsys.readouterr().out)
    encoded = json.dumps(inspection)

    assert coverage["strong_item_ids"]
    assert coverage["layers"]
    assert inspection["items"]
    assert inspection["include_history"] is True
    assert "payload" not in encoded
    assert "SYNTHETIC THIRD PARTY SECRET" not in encoded
    assert all("note" not in source for item in inspection["items"] for source in item["provenance"])


def test_brain_cli_migrates_a_private_v1_profile(tmp_path, monkeypatch, capsys):
    profile_path = tmp_path / "digital-self.json"
    brain_path = tmp_path / "digital-brain.json"
    profile = DigitalSelfProfile(
        profile_id="digital-self:cli-fixture",
        owner_id="owner:cli-fixture",
        owner_name="CLI Fixture",
        created_at=NOW,
        updated_at=NOW,
        evidence=[],
        claims=[],
        relationships=[],
        communication_style={},
        source_summary={"fixture": True},
    )
    profile.save(profile_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "living-brain",
            "brain",
            "migrate",
            str(profile_path),
            "--output",
            str(brain_path),
        ],
    )

    main()

    receipt = json.loads(capsys.readouterr().out)
    brain = DigitalBrain.load(brain_path)
    assert receipt == {
        "brain_id": brain.brain_id,
        "brain_version": 1,
        "output": str(brain_path),
        "schema_version": "digital_brain.v2",
    }
    assert brain.metadata["migrated_from"] == "digital_self.v1"
    assert stat.S_IMODE(brain_path.stat().st_mode) == 0o600


def test_brain_cli_applies_an_owner_correction_from_a_private_file(
    tmp_path,
    monkeypatch,
    capsys,
):
    summary = run_guided_demo(tmp_path / "correctable", as_of=NOW)
    original = DigitalBrain.load(summary.artifacts["brain_final"])
    target = next(
        item
        for item in original.query(as_of=NOW)
        if item.layer is BrainLayer.SEMANTIC
    )
    correction_path = tmp_path / "owner-correction.json"
    output_path = tmp_path / "corrected-brain.json"
    correction_path.write_text(
        json.dumps(
            {
                "summary": "A corrected synthetic owner fact.",
                "payload": {"synthetic": True},
                "reason": "The owner replaced an outdated synthetic fact.",
                "corrected_at": (NOW + timedelta(minutes=1)).isoformat(),
            }
        ),
        encoding="utf-8",
    )
    correction_path.chmod(0o600)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "living-brain",
            "brain",
            "correct",
            str(summary.artifacts["brain_final"]),
            "--item-id",
            target.id,
            "--correction",
            str(correction_path),
            "--output",
            str(output_path),
        ],
    )

    main()

    receipt_text = capsys.readouterr().out
    receipt = json.loads(receipt_text)
    corrected = DigitalBrain.load(output_path)
    replacement = corrected.item_by_id(receipt["replacement_item_id"])
    assert receipt["brain_version"] == original.version + 1
    assert receipt["original_item_id"] == target.id
    assert replacement.supersedes == target.id
    assert "A corrected synthetic owner fact." not in receipt_text
    assert stat.S_IMODE(output_path.stat().st_mode) == 0o600
