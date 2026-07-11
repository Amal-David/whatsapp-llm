"""Deterministic, private, end-to-end demonstration of the digital brain."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from .coverage import analyze_coverage
from .evaluation import (
    CalibrationExpectation,
    EvaluationAxis,
    EvaluationCase,
    EvaluationLab,
    OutputExpectation,
    OwnerJudgment,
    PrivacyExpectation,
    StateExpectation,
)
from .inspection import inspect_brain
from .models import (
    BrainLayer,
    ContextScope,
    DigitalBrain,
    EpistemicStatus,
    Ownership,
    ProvenanceRef,
    ProvenanceRelation,
    Sensitivity,
    StateItem,
    StateStatus,
    TemporalScope,
)
from .simulation import (
    CandidateResponse,
    RequestedAuthority,
    RetrievedContext,
    SimulationEngine,
    SimulationResult,
    Situation,
    Stakes,
)

GUIDED_STEPS = (
    "source_selection",
    "coverage_analysis",
    "adaptive_interview",
    "build_and_version",
    "inspect_and_correct",
    "simulate",
    "evaluate",
)

_DEMO_ANSWERS = {
    BrainLayer.SELF_SCHEMA: (
        "The demo owner sees themself as a careful builder who learns in public."
    ),
    BrainLayer.VALUES_GOALS: (
        "The demo owner favors privacy and reversible choices over rushed certainty."
    ),
    BrainLayer.SOCIAL: (
        "The demo owner is warmer with trusted friends and precise with colleagues."
    ),
    BrainLayer.NARRATIVE: (
        "Learning to decline premature commitments changed the demo owner's direction."
    ),
    BrainLayer.COMMUNICATION: (
        "The demo owner prefers concise, warm replies that name uncertainty directly."
    ),
    BrainLayer.SEMANTIC: (
        "The demo owner works best when facts and assumptions are kept separate."
    ),
    BrainLayer.PROCEDURAL: (
        "For hard decisions, the demo owner gathers evidence and chooses a reversible step."
    ),
    BrainLayer.AFFECT: (
        "Under time pressure, the demo owner slows down before making commitments."
    ),
    BrainLayer.EPISODE: (
        "A recent project taught the demo owner to verify details before promising delivery."
    ),
    BrainLayer.UNCERTAINTY: (
        "The model should remain unsure about preferences that have not been reconfirmed."
    ),
}


@dataclass(frozen=True)
class GuidedRunSummary:
    mode: str
    status: str
    steps: tuple[str, ...]
    artifacts: dict[str, Path]
    brain_id: str
    brain_version: int
    questions_asked: int
    evaluation_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "status": self.status,
            "steps": list(self.steps),
            "artifacts": {
                name: str(path) for name, path in sorted(self.artifacts.items())
            },
            "brain_id": self.brain_id,
            "brain_version": self.brain_version,
            "questions_asked": self.questions_asked,
            "evaluation_passed": self.evaluation_passed,
        }


class SyntheticDemoProvider:
    """A deterministic provider used only by the no-network demo."""

    provider_id = "synthetic-demo-provider.v1"

    def deliberate(
        self,
        situation: Situation,
        context: RetrievedContext,
    ) -> Sequence[CandidateResponse]:
        del situation
        preferred = (
            *context.values_goals,
            *context.procedures,
            *context.communication,
            *context.self_knowledge,
            *context.episodes,
        )
        state_ids = tuple(dict.fromkeys(item.item_id for item in preferred))
        if len(state_ids) < 2:
            return ()
        return (
            CandidateResponse.create(
                text=(
                    "Thanks for asking. I cannot commit yet; I want to verify the "
                    "details, and I will follow up tomorrow."
                ),
                rationale=(
                    "Applies the owner's privacy, reversibility, and communication "
                    "preferences without making a live commitment."
                ),
                used_state_ids=state_ids[:4],
                assumptions=("The requested commitment can wait until tomorrow.",),
                confidence=0.88,
            ),
            CandidateResponse.create(
                text=(
                    "I need to check the details before saying yes. I will come back "
                    "with a clear answer tomorrow."
                ),
                rationale="A shorter, direct alternative grounded in the same owner state.",
                used_state_ids=state_ids[:2],
                assumptions=("A one-day delay is acceptable.",),
                confidence=0.78,
            ),
        )


def run_guided_demo(
    workspace: str | Path,
    *,
    as_of: datetime,
) -> GuidedRunSummary:
    """Run every guided stage locally against an explicitly synthetic fixture."""

    _require_aware(as_of)
    output_dir = Path(workspace).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_dir.chmod(0o700)
    artifacts = _artifact_paths(output_dir)

    _write_private_json(
        artifacts["source_selection"],
        {
            "schema_version": "guided_source_selection.v1",
            "mode": "synthetic_demo",
            "source_id": "fixture:digital-brain-guided-demo.v1",
            "selected_record_count": 8,
            "contains_real_private_data": False,
            "external_access": False,
        },
    )

    brain = _initial_demo_brain(as_of)
    brain.save(artifacts["brain_initial"])
    coverage_before = analyze_coverage(brain, as_of=as_of)
    _write_private_json(artifacts["coverage_before"], coverage_before.to_dict())

    interview_at = as_of - timedelta(days=1)
    interview_rows = []
    for question in coverage_before.next_questions:
        answer = _DEMO_ANSWERS[question.layer]
        interview_rows.append({**question.to_dict(), "answer": answer})
        brain.add_item(
            _interview_state(question.question_id, question.layer, answer, interview_at),
            updated_at=interview_at,
        )
    _write_private_json(
        artifacts["adaptive_interview"],
        {
            "schema_version": "adaptive_owner_interview.v1",
            "mode": "synthetic_demo",
            "generated_from_brain_version": 1,
            "questions": interview_rows,
        },
    )
    brain.save(artifacts["brain_interviewed"])

    inspection = inspect_brain(
        brain,
        as_of=as_of,
        relationship_id="relationship:demo",
    )
    _write_private_json(artifacts["inspection_before_correction"], inspection)

    correction_target = next(
        item
        for item in brain.items
        if item.layer is BrainLayer.PROCEDURAL
        and item.kind == "decision.uncertain"
        and item.status is StateStatus.ACTIVE
    )
    correction_at = as_of - timedelta(hours=12)
    corrected = brain.apply_owner_correction(
        correction_target.id,
        summary=(
            "When unsure, the demo owner asks one focused question before acting."
        ),
        payload={"demonstration": "owner-corrected procedure"},
        corrected_at=correction_at,
        reason="Synthetic owner correction of an inferred procedure.",
    )
    _write_private_json(
        artifacts["correction"],
        {
            "schema_version": "owner_correction_receipt.v1",
            "corrected_at": correction_at.isoformat(),
            "original_item_id": correction_target.id,
            "replacement_item_id": corrected.id,
            "brain_version": brain.version,
            "reason": "Synthetic owner correction of an inferred procedure.",
        },
    )
    brain.save(artifacts["brain_final"])
    coverage_after = analyze_coverage(brain, as_of=as_of)
    _write_private_json(artifacts["coverage_after"], coverage_after.to_dict())

    situation = Situation.create(
        request="Draft a brief reply declining a rushed public commitment.",
        audience="Demo colleague",
        relationship_id="relationship:demo",
        role="collaborator",
        intent="decline for now while preserving trust",
        stakes=Stakes.LOW,
        target_time=as_of,
        channel="whatsapp",
        authority=RequestedAuthority.PRIVATE_DRAFT,
        assumptions=("The colleague can wait until tomorrow.",),
    )
    simulation = SimulationEngine(
        SyntheticDemoProvider(),
        minimum_grounded_items=2,
    ).simulate(brain, situation)
    _write_private_json(artifacts["simulation"], simulation.to_dict())

    evaluation = _evaluate_demo(
        brain,
        simulation,
        correction_target_id=correction_target.id,
        corrected_item_id=corrected.id,
    )
    _write_private_json(artifacts["evaluation"], evaluation)

    summary = GuidedRunSummary(
        mode="synthetic_demo",
        status="complete",
        steps=GUIDED_STEPS,
        artifacts=artifacts,
        brain_id=brain.brain_id,
        brain_version=brain.version,
        questions_asked=len(interview_rows),
        evaluation_passed=bool(evaluation["passed"]),
    )
    _write_private_json(artifacts["run_summary"], summary.to_dict())
    return summary


def _initial_demo_brain(as_of: datetime) -> DigitalBrain:
    items = [
        _state(
            BrainLayer.EPISODE,
            "project.commitment",
            "The demo owner once delayed a commitment until the facts were checked.",
            observed_at=as_of - timedelta(days=30),
        ),
        _state(
            BrainLayer.SEMANTIC,
            "owner.fact",
            "The demo owner works on experimental software.",
            observed_at=as_of - timedelta(days=7),
        ),
        _state(
            BrainLayer.PROCEDURAL,
            "decision.uncertain",
            "The demo owner probably waits silently when uncertain.",
            observed_at=as_of - timedelta(days=45),
            confidence=0.45,
            epistemic_status=EpistemicStatus.INFERRED,
        ),
        _state(
            BrainLayer.VALUES_GOALS,
            "value.privacy",
            "The demo owner protects privacy and avoids irreversible commitments.",
            observed_at=as_of - timedelta(days=10),
        ),
        _state(
            BrainLayer.COMMUNICATION,
            "communication.style",
            "The demo owner uses concise and warm language.",
            observed_at=as_of - timedelta(days=400),
            epistemic_status=EpistemicStatus.OBSERVED,
        ),
        _state(
            BrainLayer.SOCIAL,
            "relationship.profile",
            "A demo colleague prefers a direct explanation.",
            observed_at=as_of - timedelta(days=20),
            relationship_id="relationship:demo",
            ownership=Ownership.SHARED,
            sensitivity=Sensitivity.RESTRICTED,
        ),
        _state(
            BrainLayer.UNCERTAINTY,
            "model.boundary",
            "The owner's response under urgent pressure is not well established.",
            observed_at=as_of - timedelta(days=15),
            confidence=0.5,
            epistemic_status=EpistemicStatus.GENERATED_PROPOSAL,
            status=StateStatus.PROPOSED,
        ),
        _state(
            BrainLayer.EVENT,
            "third_party.message",
            "SYNTHETIC THIRD PARTY SECRET",
            observed_at=as_of - timedelta(days=5),
            relationship_id="relationship:demo",
            ownership=Ownership.THIRD_PARTY,
            sensitivity=Sensitivity.RESTRICTED,
            payload={"raw_text": "SYNTHETIC THIRD PARTY SECRET"},
        ),
    ]
    return DigitalBrain.create(
        owner_id="owner:synthetic-demo",
        owner_name="Demo Owner",
        created_at=as_of - timedelta(days=400),
        updated_at=as_of - timedelta(days=5),
        items=items,
        metadata={
            "fixture": "digital-brain-guided-demo.v1",
            "contains_real_private_data": False,
            "external_provider": False,
        },
    )


def _state(
    layer: BrainLayer,
    kind: str,
    summary: str,
    *,
    observed_at: datetime,
    confidence: float = 0.9,
    epistemic_status: EpistemicStatus = EpistemicStatus.OWNER_DECLARED,
    status: StateStatus = StateStatus.ACTIVE,
    relationship_id: str | None = None,
    ownership: Ownership = Ownership.OWNER,
    sensitivity: Sensitivity = Sensitivity.PRIVATE,
    payload: dict[str, Any] | None = None,
) -> StateItem:
    content_hash = hashlib.sha256(
        json.dumps(
            {"summary": summary, "payload": payload or {}},
            ensure_ascii=True,
            sort_keys=True,
        ).encode()
    ).hexdigest()
    return StateItem.create(
        layer=layer,
        kind=kind,
        summary=summary,
        payload=payload or {},
        epistemic_status=epistemic_status,
        status=status,
        confidence=confidence,
        temporal=TemporalScope(
            observed_at=observed_at,
            recorded_at=observed_at,
            valid_from=observed_at,
            last_confirmed_at=(
                observed_at
                if epistemic_status is EpistemicStatus.OWNER_DECLARED
                else None
            ),
        ),
        scope=ContextScope(relationship_id=relationship_id),
        sensitivity=sensitivity,
        ownership=ownership,
        provenance=(
            ProvenanceRef(
                source_id=f"synthetic:{layer.value}:{kind}",
                source_type="synthetic_demo_fixture",
                relation=ProvenanceRelation.SUPPORTS,
                observed_at=observed_at,
                content_hash=content_hash,
            ),
        ),
        metadata={"synthetic": True},
    )


def _interview_state(
    question_id: str,
    layer: BrainLayer,
    answer: str,
    answered_at: datetime,
) -> StateItem:
    return StateItem.create(
        layer=layer,
        kind=f"owner_interview.{layer.value}",
        summary=answer,
        payload={"question_id": question_id},
        epistemic_status=EpistemicStatus.OWNER_DECLARED,
        confidence=1.0,
        temporal=TemporalScope(
            observed_at=answered_at,
            recorded_at=answered_at,
            valid_from=answered_at,
            last_confirmed_at=answered_at,
        ),
        scope=ContextScope(),
        sensitivity=Sensitivity.PRIVATE,
        ownership=Ownership.OWNER,
        provenance=(
            ProvenanceRef(
                source_id=question_id,
                source_type="synthetic_owner_interview",
                relation=ProvenanceRelation.SUPPORTS,
                observed_at=answered_at,
                content_hash=hashlib.sha256(answer.encode()).hexdigest(),
            ),
        ),
        metadata={"synthetic": True, "elicited_by_coverage": True},
    )


def _evaluate_demo(
    brain: DigitalBrain,
    simulation: SimulationResult,
    *,
    correction_target_id: str,
    corrected_item_id: str,
) -> dict[str, Any]:
    if simulation.selected_response is None:
        raise ValueError("the guided demo requires one grounded selected response")
    episode = next(item for item in brain.items if item.layer is BrainLayer.EPISODE)
    value = next(
        item
        for item in brain.items
        if item.layer is BrainLayer.VALUES_GOALS
        and item.status is StateStatus.ACTIVE
    )
    third_party = next(
        item for item in brain.items if item.ownership is Ownership.THIRD_PARTY
    )
    report = EvaluationLab().evaluate(
        (
            EvaluationCase(
                case_id="synthetic-demo:behavioral",
                axis=EvaluationAxis.BEHAVIORAL,
                result=simulation,
                output=OutputExpectation(
                    accepted_responses=(simulation.selected_response,),
                ),
            ),
            EvaluationCase(
                case_id="synthetic-demo:temporal",
                axis=EvaluationAxis.TEMPORAL,
                result=simulation,
                state=StateExpectation(
                    required_state_ids=(corrected_item_id,),
                    forbidden_state_ids=(correction_target_id,),
                    expected_target_time=simulation.situation.target_time,
                ),
            ),
            EvaluationCase(
                case_id="synthetic-demo:relationship",
                axis=EvaluationAxis.RELATIONSHIP,
                result=simulation,
                state=StateExpectation(
                    forbidden_state_ids=(third_party.id,),
                    expected_relationship_ids=("relationship:demo",),
                ),
            ),
            EvaluationCase(
                case_id="synthetic-demo:autobiographical",
                axis=EvaluationAxis.AUTOBIOGRAPHICAL,
                result=simulation,
                state=StateExpectation(
                    required_state_ids=(episode.id,),
                    required_evidence_ids=tuple(
                        provenance.source_id for provenance in episode.provenance
                    ),
                ),
            ),
            EvaluationCase(
                case_id="synthetic-demo:decision",
                axis=EvaluationAxis.DECISION,
                result=simulation,
                output=OutputExpectation(
                    accepted_responses=(simulation.selected_response,),
                    required_assumptions=(
                        "The requested commitment can wait until tomorrow.",
                    ),
                ),
                state=StateExpectation(required_state_ids=(value.id,)),
            ),
            EvaluationCase(
                case_id="synthetic-demo:calibration",
                axis=EvaluationAxis.CALIBRATION,
                result=simulation,
                calibration=CalibrationExpectation(
                    expected_correct=True,
                    maximum_brier_score=0.02,
                ),
            ),
            EvaluationCase(
                case_id="synthetic-demo:privacy",
                axis=EvaluationAxis.PRIVACY,
                result=simulation,
                privacy=PrivacyExpectation(
                    forbidden_text_fragments=("SYNTHETIC THIRD PARTY SECRET",),
                    forbidden_state_ids=(third_party.id,),
                    forbidden_evidence_ids=tuple(
                        provenance.source_id for provenance in third_party.provenance
                    ),
                ),
            ),
            EvaluationCase(
                case_id="synthetic-demo:owner-judgment",
                axis=EvaluationAxis.OWNER_JUDGMENT,
                result=simulation,
                owner_judgment=OwnerJudgment(
                    approved=True,
                    score=0.9,
                    minimum_score=0.75,
                ),
            ),
        )
    )
    payload = report.to_dict()
    payload["axis_scores"] = {
        axis.axis.value: axis.score for axis in report.axis_results
    }
    payload["external_provider"] = False
    return payload


def _artifact_paths(output_dir: Path) -> dict[str, Path]:
    filenames = {
        "source_selection": "01-source-selection.json",
        "brain_initial": "02-brain-initial.json",
        "coverage_before": "03-coverage-before.json",
        "adaptive_interview": "04-adaptive-interview.json",
        "brain_interviewed": "05-brain-interviewed.json",
        "inspection_before_correction": "06-inspection-before-correction.json",
        "correction": "07-correction.json",
        "brain_final": "08-brain-final.json",
        "coverage_after": "09-coverage-after.json",
        "simulation": "10-simulation.json",
        "evaluation": "11-evaluation.json",
        "run_summary": "12-run-summary.json",
    }
    return {name: output_dir / filename for name, filename in filenames.items()}


def _write_private_json(path: Path, value: dict[str, Any]) -> None:
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
        ) as temporary:
            temporary_path = Path(temporary.name)
            json.dump(value, temporary, ensure_ascii=True, indent=2, sort_keys=True)
            temporary.write("\n")
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, path)
        path.chmod(0o600)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("guided demo as_of must be timezone-aware")


__all__ = ["GUIDED_STEPS", "GuidedRunSummary", "run_guided_demo"]
