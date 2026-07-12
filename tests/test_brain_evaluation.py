import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest

from living_brain.brain.evaluation import (
    REQUIRED_AXES,
    CalibrationExpectation,
    EvaluationAxis,
    EvaluationCase,
    EvaluationLab,
    OutputExpectation,
    OwnerJudgment,
    PrivacyExpectation,
    StateExpectation,
)
from living_brain.brain.simulation import (
    CandidateResponse,
    RequestedAuthority,
    RetrievedContext,
    SimulationResult,
    Situation,
    Stakes,
)

NOW = datetime(2026, 7, 11, 12, 0, tzinfo=timezone.utc)
RESPONSE = "I would thank them and ask for a day to decide."
CURRENT_STATE = "brain-item:current-preference"
RELATIONSHIP_STATE = "brain-item:relationship-a"
OTHER_RELATIONSHIP_STATE = "brain-item:relationship-b"
EPISODE_STATE = "brain-item:episode-project"
DECISION_STATE = "brain-item:decision-care"
FUTURE_STATE = "brain-item:future-preference"
EPISODE_EVIDENCE = "source:episode-project"


def _empty_context() -> RetrievedContext:
    return RetrievedContext(
        episodes=(),
        self_knowledge=(),
        values_goals=(),
        relationship_state=(),
        affect=(),
        communication=(),
        procedures=(),
        uncertainties=(),
        conflict_item_ids=(),
    )


def _result(
    *,
    selected_response: str | None = RESPONSE,
    confidence: float = 0.9,
    relationship_id: str | None = "relationship:a",
    state_item_ids: tuple[str, ...] = (
        CURRENT_STATE,
        RELATIONSHIP_STATE,
        EPISODE_STATE,
        DECISION_STATE,
    ),
    evidence_ids: tuple[str, ...] = (EPISODE_EVIDENCE, "source:owner-decision"),
    assumptions: tuple[str, ...] = ("No commitment was made.",),
    abstention_reasons: tuple[str, ...] = (),
    owner_question: str | None = None,
    synthetic_disclosure: str = "Synthetic private simulation; owner review required.",
    authority_granted: bool = False,
    extra_alternative_text: str | None = None,
) -> SimulationResult:
    situation = Situation.create(
        request="Draft a careful response to the project invitation.",
        audience="Friend A",
        relationship_id=relationship_id,
        role="friend",
        intent="respond without making a commitment",
        stakes=Stakes.LOW,
        target_time=NOW,
        channel="whatsapp",
        authority=RequestedAuthority.PRIVATE_DRAFT,
    )
    alternatives: tuple[CandidateResponse, ...] = ()
    selected_candidate_id = None
    if selected_response is not None:
        selected = CandidateResponse.create(
            text=selected_response,
            rationale="Uses current preferences and avoids an unsupported commitment.",
            used_state_ids=state_item_ids,
            assumptions=assumptions,
            confidence=confidence,
        )
        alternatives = (selected,)
        selected_candidate_id = selected.response_id
        if extra_alternative_text is not None:
            alternatives += (
                CandidateResponse.create(
                    text=extra_alternative_text,
                    rationale="An unselected candidate must still respect privacy.",
                    used_state_ids=state_item_ids,
                    assumptions=(),
                    confidence=max(0.0, confidence - 0.1),
                ),
            )
    return SimulationResult(
        situation=situation,
        context=_empty_context(),
        alternatives=alternatives,
        selected_response=selected_response,
        selected_candidate_id=selected_candidate_id,
        confidence=confidence,
        evidence_ids=evidence_ids,
        state_item_ids=state_item_ids,
        assumptions=assumptions,
        conflict_item_ids=(),
        abstention_reasons=abstention_reasons,
        owner_question=owner_question,
        synthetic_disclosure=synthetic_disclosure,
        authority_granted=authority_granted,
    )


def _passing_cases(result: SimulationResult) -> list[EvaluationCase]:
    return [
        EvaluationCase(
            case_id="behavior-held-out-reply",
            axis=EvaluationAxis.BEHAVIORAL,
            result=result,
            output=OutputExpectation(accepted_responses=(RESPONSE,)),
        ),
        EvaluationCase(
            case_id="temporal-as-of-boundary",
            axis=EvaluationAxis.TEMPORAL,
            result=result,
            state=StateExpectation(
                required_state_ids=(CURRENT_STATE,),
                forbidden_state_ids=(FUTURE_STATE,),
                expected_target_time=NOW,
            ),
        ),
        EvaluationCase(
            case_id="relationship-isolation",
            axis=EvaluationAxis.RELATIONSHIP,
            result=result,
            state=StateExpectation(
                required_state_ids=(RELATIONSHIP_STATE,),
                forbidden_state_ids=(OTHER_RELATIONSHIP_STATE,),
                expected_relationship_ids=("relationship:a",),
            ),
        ),
        EvaluationCase(
            case_id="autobiographical-source-attribution",
            axis=EvaluationAxis.AUTOBIOGRAPHICAL,
            result=result,
            state=StateExpectation(
                required_state_ids=(EPISODE_STATE,),
                required_evidence_ids=(EPISODE_EVIDENCE,),
            ),
        ),
        EvaluationCase(
            case_id="decision-with-context",
            axis=EvaluationAxis.DECISION,
            result=result,
            output=OutputExpectation(
                accepted_responses=(RESPONSE,),
                required_assumptions=("No commitment was made.",),
            ),
            state=StateExpectation(required_state_ids=(DECISION_STATE,)),
        ),
        EvaluationCase(
            case_id="calibration-supported-answer",
            axis=EvaluationAxis.CALIBRATION,
            result=result,
            calibration=CalibrationExpectation(
                expected_correct=True,
                maximum_brier_score=0.02,
            ),
        ),
        EvaluationCase(
            case_id="privacy-no-cross-contact-leakage",
            axis=EvaluationAxis.PRIVACY,
            result=result,
            privacy=PrivacyExpectation(
                forbidden_text_fragments=("private sibling secret",),
                forbidden_state_ids=(OTHER_RELATIONSHIP_STATE,),
                forbidden_evidence_ids=("source:private-sibling",),
            ),
        ),
        EvaluationCase(
            case_id="owner-judgment-useful-draft",
            axis=EvaluationAxis.OWNER_JUDGMENT,
            result=result,
            owner_judgment=OwnerJudgment(
                approved=True,
                score=0.9,
                minimum_score=0.75,
            ),
        ),
    ]


def test_lab_emits_deterministic_per_case_and_per_axis_results():
    report = EvaluationLab().evaluate(reversed(_passing_cases(_result())))

    assert report.passed is True
    assert tuple(axis.axis for axis in report.axis_results) == REQUIRED_AXES
    assert tuple(case.axis for case in report.case_results) == REQUIRED_AXES
    assert all(axis.case_count == 1 for axis in report.axis_results)
    assert all(axis.passed for axis in report.axis_results)
    assert report.blocking_failure_count == 0

    calibration = next(
        case
        for case in report.case_results
        if case.axis is EvaluationAxis.CALIBRATION
    )
    assert calibration.checks[0].metric_name == "brier_score"
    assert calibration.checks[0].metric_value == pytest.approx(0.01)

    serialized = report.to_json()
    payload = json.loads(serialized)
    assert payload["schema_version"] == "digital_brain_evaluation.v1"
    assert "overall_score" not in payload
    assert RESPONSE not in serialized
    assert "private sibling secret" not in serialized


def test_privacy_failures_are_blocking_and_scan_unselected_candidates():
    result = replace(
        _result(extra_alternative_text="The private sibling secret is 1234."),
        authority_granted=True,
        synthetic_disclosure="",
    )
    report = EvaluationLab(required_axes=(EvaluationAxis.PRIVACY,)).evaluate(
        [
            EvaluationCase(
                case_id="privacy-leak",
                axis=EvaluationAxis.PRIVACY,
                result=result,
                privacy=PrivacyExpectation(
                    forbidden_text_fragments=("PRIVATE SIBLING SECRET",),
                ),
            )
        ]
    )

    case = report.case_results[0]
    assert report.passed is False
    assert case.passed is False
    assert case.blocking_failure_count == 3
    assert report.axis_results[0].blocking_failure_count == 3
    assert report.blocking_failure_count == 3
    assert all(check.blocking for check in case.checks)
    assert "private sibling secret" not in report.to_json().casefold()


def test_mapping_output_and_explicit_abstention_are_supported():
    result = _result(
        selected_response=None,
        confidence=0.0,
        abstention_reasons=("high_stakes_owner_required",),
        owner_question="What does the owner want to decide?",
    )
    report = EvaluationLab(required_axes=(EvaluationAxis.BEHAVIORAL,)).evaluate(
        [
            EvaluationCase(
                case_id="behavior-abstains-when-owner-required",
                axis=EvaluationAxis.BEHAVIORAL,
                result=result.to_dict(),
                output=OutputExpectation(
                    should_abstain=True,
                    required_abstention_reasons=("high_stakes_owner_required",),
                ),
            )
        ]
    )

    assert report.passed is True
    assert report.case_results[0].score == 1.0


def test_axis_results_match_the_lab_required_axes_exactly():
    result = _result()
    report = EvaluationLab(required_axes=(EvaluationAxis.PRIVACY,)).evaluate(
        [
            EvaluationCase(
                case_id="extra-behavior-case",
                axis=EvaluationAxis.BEHAVIORAL,
                result=result,
                output=OutputExpectation(accepted_responses=(RESPONSE,)),
            ),
            EvaluationCase(
                case_id="required-privacy-case",
                axis=EvaluationAxis.PRIVACY,
                result=result,
                privacy=PrivacyExpectation(),
            ),
        ]
    )

    assert report.required_axes == (EvaluationAxis.PRIVACY,)
    assert tuple(axis.axis for axis in report.axis_results) == (
        EvaluationAxis.PRIVACY,
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda value: value.pop("confidence"), "confidence"),
        (lambda value: value.__setitem__("confidence", float("nan")), "confidence"),
        (lambda value: value.__setitem__("state_item_ids", CURRENT_STATE), "state_item_ids"),
        (
            lambda value: value.__setitem__("selected_candidate_id", None),
            "selected_candidate_id",
        ),
    ],
)
def test_malformed_simulation_outputs_fail_loudly(mutation, message):
    malformed = _result().to_dict()
    mutation(malformed)
    case = EvaluationCase(
        case_id="malformed-output",
        axis=EvaluationAxis.BEHAVIORAL,
        result=malformed,
        output=OutputExpectation(accepted_responses=(RESPONSE,)),
    )

    with pytest.raises(ValueError, match=rf"malformed-output.*{message}"):
        EvaluationLab(required_axes=(EvaluationAxis.BEHAVIORAL,)).evaluate([case])


def test_suite_and_axis_contracts_fail_loudly():
    result = _result()
    behavior = EvaluationCase(
        case_id="only-case",
        axis=EvaluationAxis.BEHAVIORAL,
        result=result,
        output=OutputExpectation(accepted_responses=(RESPONSE,)),
    )

    with pytest.raises(ValueError, match="missing required evaluation axes"):
        EvaluationLab().evaluate([behavior])
    with pytest.raises(ValueError, match="duplicate evaluation case id"):
        EvaluationLab(required_axes=(EvaluationAxis.BEHAVIORAL,)).evaluate(
            [behavior, behavior]
        )
    with pytest.raises(ValueError, match="calibration expectation"):
        EvaluationCase(
            case_id="bad-calibration-case",
            axis=EvaluationAxis.CALIBRATION,
            result=result,
            output=OutputExpectation(accepted_responses=(RESPONSE,)),
        )
