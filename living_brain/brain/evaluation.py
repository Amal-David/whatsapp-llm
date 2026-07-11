"""Deterministic evaluation of digital-brain simulation outputs."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol


class EvaluationAxis(str, Enum):
    BEHAVIORAL = "behavioral"
    TEMPORAL = "temporal"
    RELATIONSHIP = "relationship"
    AUTOBIOGRAPHICAL = "autobiographical"
    DECISION = "decision"
    CALIBRATION = "calibration"
    PRIVACY = "privacy"
    OWNER_JUDGMENT = "owner_judgment"


REQUIRED_AXES: tuple[EvaluationAxis, ...] = (
    EvaluationAxis.BEHAVIORAL,
    EvaluationAxis.TEMPORAL,
    EvaluationAxis.RELATIONSHIP,
    EvaluationAxis.AUTOBIOGRAPHICAL,
    EvaluationAxis.DECISION,
    EvaluationAxis.CALIBRATION,
    EvaluationAxis.PRIVACY,
    EvaluationAxis.OWNER_JUDGMENT,
)

_AXIS_ORDER = {axis: index for index, axis in enumerate(REQUIRED_AXES)}
_MISSING = object()


class SituationLike(Protocol):
    @property
    def relationship_id(self) -> str | None: ...

    @property
    def target_time(self) -> datetime: ...


class SimulationResultLike(Protocol):
    @property
    def situation(self) -> SituationLike: ...

    @property
    def selected_response(self) -> str | None: ...

    @property
    def selected_candidate_id(self) -> str | None: ...

    @property
    def confidence(self) -> float: ...

    @property
    def evidence_ids(self) -> Sequence[str]: ...

    @property
    def state_item_ids(self) -> Sequence[str]: ...

    @property
    def assumptions(self) -> Sequence[str]: ...

    @property
    def conflict_item_ids(self) -> Sequence[str]: ...

    @property
    def abstention_reasons(self) -> Sequence[str]: ...

    @property
    def owner_question(self) -> str | None: ...

    @property
    def synthetic_disclosure(self) -> str: ...

    @property
    def authority_granted(self) -> bool: ...


@dataclass(frozen=True)
class OutputExpectation:
    accepted_responses: tuple[str, ...] = ()
    should_abstain: bool | None = None
    required_assumptions: tuple[str, ...] = ()
    required_conflict_ids: tuple[str, ...] = ()
    required_abstention_reasons: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "accepted_responses",
            _expectation_strings(self.accepted_responses, "accepted responses"),
        )
        object.__setattr__(
            self,
            "required_assumptions",
            _expectation_strings(self.required_assumptions, "required assumptions"),
        )
        object.__setattr__(
            self,
            "required_conflict_ids",
            _expectation_strings(self.required_conflict_ids, "required conflict ids"),
        )
        object.__setattr__(
            self,
            "required_abstention_reasons",
            _expectation_strings(
                self.required_abstention_reasons,
                "required abstention reasons",
            ),
        )
        if self.should_abstain is not None and not isinstance(
            self.should_abstain,
            bool,
        ):
            raise ValueError("should_abstain must be a boolean or None")
        if self.should_abstain is True and self.accepted_responses:
            raise ValueError("an abstention expectation cannot accept response text")
        if not self.has_checks:
            raise ValueError("output expectation must define at least one check")

    @property
    def has_checks(self) -> bool:
        return bool(
            self.accepted_responses
            or self.should_abstain is not None
            or self.required_assumptions
            or self.required_conflict_ids
            or self.required_abstention_reasons
        )


@dataclass(frozen=True)
class StateExpectation:
    required_state_ids: tuple[str, ...] = ()
    forbidden_state_ids: tuple[str, ...] = ()
    required_evidence_ids: tuple[str, ...] = ()
    forbidden_evidence_ids: tuple[str, ...] = ()
    expected_relationship_ids: tuple[str | None, ...] = ()
    expected_target_time: datetime | None = None

    def __post_init__(self) -> None:
        for field_name, label in (
            ("required_state_ids", "required state ids"),
            ("forbidden_state_ids", "forbidden state ids"),
            ("required_evidence_ids", "required evidence ids"),
            ("forbidden_evidence_ids", "forbidden evidence ids"),
        ):
            object.__setattr__(
                self,
                field_name,
                _expectation_strings(getattr(self, field_name), label),
            )
        object.__setattr__(
            self,
            "expected_relationship_ids",
            _relationship_ids(self.expected_relationship_ids),
        )
        if set(self.required_state_ids) & set(self.forbidden_state_ids):
            raise ValueError("required and forbidden state ids cannot overlap")
        if set(self.required_evidence_ids) & set(self.forbidden_evidence_ids):
            raise ValueError("required and forbidden evidence ids cannot overlap")
        if self.expected_target_time is not None:
            _require_aware(self.expected_target_time, "expected target time")
        if not self.has_checks:
            raise ValueError("state expectation must define at least one check")

    @property
    def has_checks(self) -> bool:
        return bool(
            self.required_state_ids
            or self.forbidden_state_ids
            or self.required_evidence_ids
            or self.forbidden_evidence_ids
            or self.expected_relationship_ids
            or self.expected_target_time is not None
        )


@dataclass(frozen=True)
class PrivacyExpectation:
    forbidden_text_fragments: tuple[str, ...] = ()
    forbidden_state_ids: tuple[str, ...] = ()
    forbidden_evidence_ids: tuple[str, ...] = ()
    require_synthetic_disclosure: bool = True
    require_authority_denied: bool = True

    def __post_init__(self) -> None:
        for field_name, label in (
            ("forbidden_text_fragments", "forbidden text fragments"),
            ("forbidden_state_ids", "privacy-forbidden state ids"),
            ("forbidden_evidence_ids", "privacy-forbidden evidence ids"),
        ):
            object.__setattr__(
                self,
                field_name,
                _expectation_strings(getattr(self, field_name), label),
            )
        if not isinstance(self.require_synthetic_disclosure, bool):
            raise ValueError("require_synthetic_disclosure must be a boolean")
        if not isinstance(self.require_authority_denied, bool):
            raise ValueError("require_authority_denied must be a boolean")
        if not self.has_checks:
            raise ValueError("privacy expectation must define at least one check")

    @property
    def has_checks(self) -> bool:
        return bool(
            self.forbidden_text_fragments
            or self.forbidden_state_ids
            or self.forbidden_evidence_ids
            or self.require_synthetic_disclosure
            or self.require_authority_denied
        )


@dataclass(frozen=True)
class CalibrationExpectation:
    expected_correct: bool
    maximum_brier_score: float = 0.25

    def __post_init__(self) -> None:
        if not isinstance(self.expected_correct, bool):
            raise ValueError("expected_correct must be a boolean")
        object.__setattr__(
            self,
            "maximum_brier_score",
            _unit_interval(self.maximum_brier_score, "maximum_brier_score"),
        )


@dataclass(frozen=True)
class OwnerJudgment:
    approved: bool
    score: float
    minimum_score: float = 0.5

    def __post_init__(self) -> None:
        if not isinstance(self.approved, bool):
            raise ValueError("owner approval must be a boolean")
        object.__setattr__(self, "score", _unit_interval(self.score, "owner score"))
        object.__setattr__(
            self,
            "minimum_score",
            _unit_interval(self.minimum_score, "minimum owner score"),
        )


@dataclass(frozen=True)
class EvaluationCase:
    case_id: str
    axis: EvaluationAxis
    result: SimulationResultLike | Mapping[str, Any]
    output: OutputExpectation | None = None
    state: StateExpectation | None = None
    privacy: PrivacyExpectation | None = None
    calibration: CalibrationExpectation | None = None
    owner_judgment: OwnerJudgment | None = None

    def __post_init__(self) -> None:
        normalized_case_id = self.case_id.strip()
        if not normalized_case_id:
            raise ValueError("evaluation case id is required")
        object.__setattr__(self, "case_id", normalized_case_id)
        object.__setattr__(self, "axis", _axis(self.axis))
        for value, expected_type, label in (
            (self.output, OutputExpectation, "output"),
            (self.state, StateExpectation, "state"),
            (self.privacy, PrivacyExpectation, "privacy"),
            (self.calibration, CalibrationExpectation, "calibration"),
            (self.owner_judgment, OwnerJudgment, "owner judgment"),
        ):
            if value is not None and not isinstance(value, expected_type):
                raise ValueError(f"{label} expectation has the wrong type")
        self._validate_axis_contract()

    def _validate_axis_contract(self) -> None:
        if self.axis is EvaluationAxis.BEHAVIORAL and self.output is None:
            raise ValueError("behavioral cases require an output expectation")
        if self.axis is EvaluationAxis.TEMPORAL and (
            self.state is None
            or not (
                self.state.required_state_ids
                or self.state.forbidden_state_ids
                or self.state.expected_target_time is not None
            )
        ):
            raise ValueError("temporal cases require a temporal state expectation")
        if self.axis is EvaluationAxis.RELATIONSHIP and (
            self.state is None
            or not (
                self.state.required_state_ids
                or self.state.forbidden_state_ids
                or self.state.expected_relationship_ids
            )
        ):
            raise ValueError("relationship cases require a relationship state expectation")
        if self.axis is EvaluationAxis.AUTOBIOGRAPHICAL and (
            self.state is None
            or not (
                self.state.required_state_ids or self.state.required_evidence_ids
            )
        ):
            raise ValueError(
                "autobiographical cases require state or evidence attribution"
            )
        if self.axis is EvaluationAxis.DECISION and self.output is None:
            raise ValueError("decision cases require an output expectation")
        if self.axis is EvaluationAxis.CALIBRATION and self.calibration is None:
            raise ValueError("calibration cases require a calibration expectation")
        if self.axis is EvaluationAxis.PRIVACY and self.privacy is None:
            raise ValueError("privacy cases require a privacy expectation")
        if self.axis is EvaluationAxis.OWNER_JUDGMENT and self.owner_judgment is None:
            raise ValueError("owner-judgment cases require explicit owner judgment")


@dataclass(frozen=True)
class CheckResult:
    check_id: str
    passed: bool
    score: float
    detail: str
    blocking: bool = False
    metric_name: str | None = None
    metric_value: float | None = None

    def to_dict(self) -> dict[str, Any]:
        value: dict[str, Any] = {
            "check_id": self.check_id,
            "passed": self.passed,
            "score": self.score,
            "detail": self.detail,
            "blocking": self.blocking,
        }
        if self.metric_name is not None:
            value["metric_name"] = self.metric_name
            value["metric_value"] = self.metric_value
        return value


@dataclass(frozen=True)
class CaseResult:
    case_id: str
    axis: EvaluationAxis
    passed: bool
    score: float
    checks: tuple[CheckResult, ...]

    @property
    def blocking_failure_count(self) -> int:
        return sum(not check.passed and check.blocking for check in self.checks)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "axis": self.axis.value,
            "passed": self.passed,
            "score": self.score,
            "blocking_failure_count": self.blocking_failure_count,
            "checks": [check.to_dict() for check in self.checks],
        }


@dataclass(frozen=True)
class AxisResult:
    axis: EvaluationAxis
    case_ids: tuple[str, ...]
    passed: bool
    score: float
    blocking_failure_count: int

    @property
    def case_count(self) -> int:
        return len(self.case_ids)

    def to_dict(self) -> dict[str, Any]:
        return {
            "axis": self.axis.value,
            "case_ids": list(self.case_ids),
            "case_count": self.case_count,
            "passed": self.passed,
            "score": self.score,
            "blocking_failure_count": self.blocking_failure_count,
        }


@dataclass(frozen=True)
class EvaluationReport:
    required_axes: tuple[EvaluationAxis, ...]
    case_results: tuple[CaseResult, ...]
    axis_results: tuple[AxisResult, ...]

    @property
    def passed(self) -> bool:
        return bool(self.axis_results) and all(axis.passed for axis in self.axis_results)

    @property
    def blocking_failure_count(self) -> int:
        return sum(axis.blocking_failure_count for axis in self.axis_results)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "digital_brain_evaluation.v1",
            "required_axes": [axis.value for axis in self.required_axes],
            "passed": self.passed,
            "blocking_failure_count": self.blocking_failure_count,
            "case_results": [case.to_dict() for case in self.case_results],
            "axis_results": [axis.to_dict() for axis in self.axis_results],
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"


@dataclass(frozen=True)
class _SimulationView:
    selected_response: str | None
    selected_candidate_id: str | None
    confidence: float
    evidence_ids: tuple[str, ...]
    state_item_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    conflict_item_ids: tuple[str, ...]
    abstention_reasons: tuple[str, ...]
    owner_question: str | None
    synthetic_disclosure: str
    authority_granted: bool
    relationship_id: str | None
    target_time: datetime
    generated_text: tuple[str, ...]


class EvaluationLab:
    """Evaluate an explicit, provider-free suite without collapsing its axes."""

    def __init__(
        self,
        *,
        required_axes: Iterable[EvaluationAxis | str] = REQUIRED_AXES,
    ):
        normalized = tuple(_axis(value) for value in required_axes)
        if not normalized:
            raise ValueError("the evaluation lab requires at least one axis")
        if len(set(normalized)) != len(normalized):
            raise ValueError("required evaluation axes cannot contain duplicates")
        self.required_axes = tuple(sorted(normalized, key=_AXIS_ORDER.__getitem__))

    def evaluate(self, cases: Iterable[EvaluationCase]) -> EvaluationReport:
        materialized = tuple(cases)
        if not materialized:
            raise ValueError("evaluation suite cannot be empty")
        if not all(isinstance(case, EvaluationCase) for case in materialized):
            raise ValueError("evaluation suite entries must be EvaluationCase objects")
        case_ids = [case.case_id for case in materialized]
        duplicate_ids = sorted(
            case_id for case_id in set(case_ids) if case_ids.count(case_id) > 1
        )
        if duplicate_ids:
            raise ValueError(f"duplicate evaluation case id: {duplicate_ids[0]}")
        present_axes = {case.axis for case in materialized}
        missing_axes = [axis for axis in self.required_axes if axis not in present_axes]
        if missing_axes:
            values = ", ".join(axis.value for axis in missing_axes)
            raise ValueError(f"missing required evaluation axes: {values}")

        ordered = tuple(
            sorted(
                materialized,
                key=lambda case: (_AXIS_ORDER[case.axis], case.case_id),
            )
        )
        views = tuple((case, self._view(case)) for case in ordered)
        case_results = tuple(
            self._evaluate_view(case, view) for case, view in views
        )
        axis_results = tuple(
            self._axis_result(axis, case_results)
            for axis in REQUIRED_AXES
            if axis in present_axes
        )
        return EvaluationReport(
            required_axes=self.required_axes,
            case_results=case_results,
            axis_results=axis_results,
        )

    def evaluate_case(self, case: EvaluationCase) -> CaseResult:
        if not isinstance(case, EvaluationCase):
            raise ValueError("evaluate_case requires an EvaluationCase")
        return self._evaluate_view(case, self._view(case))

    @staticmethod
    def _view(case: EvaluationCase) -> _SimulationView:
        try:
            return _simulation_view(case.result)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"evaluation case {case.case_id!r} has malformed result: {error}"
            ) from error

    @staticmethod
    def _evaluate_view(case: EvaluationCase, view: _SimulationView) -> CaseResult:
        checks: list[CheckResult] = []
        if case.output is not None:
            checks.extend(_output_checks(view, case.output))
        if case.state is not None:
            checks.extend(_state_checks(view, case.state))
        if case.calibration is not None:
            checks.append(_calibration_check(view, case.calibration))
        if case.privacy is not None:
            checks.extend(_privacy_checks(view, case.privacy))
        if case.owner_judgment is not None:
            checks.extend(_owner_checks(case.owner_judgment))
        if not checks:
            raise ValueError(f"evaluation case {case.case_id!r} defines no checks")
        score = sum(check.score for check in checks) / len(checks)
        return CaseResult(
            case_id=case.case_id,
            axis=case.axis,
            passed=all(check.passed for check in checks),
            score=score,
            checks=tuple(checks),
        )

    @staticmethod
    def _axis_result(
        axis: EvaluationAxis,
        cases: tuple[CaseResult, ...],
    ) -> AxisResult:
        matching = tuple(case for case in cases if case.axis is axis)
        return AxisResult(
            axis=axis,
            case_ids=tuple(case.case_id for case in matching),
            passed=all(case.passed for case in matching),
            score=sum(case.score for case in matching) / len(matching),
            blocking_failure_count=sum(
                case.blocking_failure_count for case in matching
            ),
        )


def _output_checks(
    view: _SimulationView,
    expected: OutputExpectation,
) -> list[CheckResult]:
    checks = []
    if expected.accepted_responses:
        passed = view.selected_response in expected.accepted_responses
        checks.append(
            _binary_check(
                "accepted_response",
                passed,
                "selected response matched an accepted response"
                if passed
                else "selected response did not match an accepted response",
            )
        )
    if expected.should_abstain is not None:
        abstained = view.selected_response is None
        passed = abstained is expected.should_abstain
        checks.append(
            _binary_check(
                "abstention",
                passed,
                "abstention behavior matched the expectation"
                if passed
                else "abstention behavior did not match the expectation",
            )
        )
    if expected.required_assumptions:
        missing = set(expected.required_assumptions) - set(view.assumptions)
        checks.append(
            _binary_check(
                "required_assumptions",
                not missing,
                _count_detail("required assumption", len(missing), missing=True),
            )
        )
    if expected.required_conflict_ids:
        missing = set(expected.required_conflict_ids) - set(view.conflict_item_ids)
        checks.append(
            _binary_check(
                "required_conflicts",
                not missing,
                _count_detail("required conflict", len(missing), missing=True),
            )
        )
    if expected.required_abstention_reasons:
        missing = set(expected.required_abstention_reasons) - set(
            view.abstention_reasons
        )
        checks.append(
            _binary_check(
                "required_abstention_reasons",
                not missing,
                _count_detail("required abstention reason", len(missing), missing=True),
            )
        )
    return checks


def _state_checks(
    view: _SimulationView,
    expected: StateExpectation,
) -> list[CheckResult]:
    checks = []
    if expected.required_state_ids:
        missing = set(expected.required_state_ids) - set(view.state_item_ids)
        checks.append(
            _binary_check(
                "required_state",
                not missing,
                _count_detail("required state item", len(missing), missing=True),
            )
        )
    if expected.forbidden_state_ids:
        present = set(expected.forbidden_state_ids) & set(view.state_item_ids)
        checks.append(
            _binary_check(
                "forbidden_state",
                not present,
                _count_detail("forbidden state item", len(present), missing=False),
            )
        )
    if expected.required_evidence_ids:
        missing = set(expected.required_evidence_ids) - set(view.evidence_ids)
        checks.append(
            _binary_check(
                "required_evidence",
                not missing,
                _count_detail("required evidence item", len(missing), missing=True),
            )
        )
    if expected.forbidden_evidence_ids:
        present = set(expected.forbidden_evidence_ids) & set(view.evidence_ids)
        checks.append(
            _binary_check(
                "forbidden_evidence",
                not present,
                _count_detail("forbidden evidence item", len(present), missing=False),
            )
        )
    if expected.expected_relationship_ids:
        passed = view.relationship_id in expected.expected_relationship_ids
        checks.append(
            _binary_check(
                "relationship_scope",
                passed,
                "relationship scope matched the expectation"
                if passed
                else "relationship scope did not match the expectation",
            )
        )
    if expected.expected_target_time is not None:
        passed = view.target_time == expected.expected_target_time
        checks.append(
            _binary_check(
                "target_time",
                passed,
                "target time matched the expectation"
                if passed
                else "target time did not match the expectation",
            )
        )
    return checks


def _calibration_check(
    view: _SimulationView,
    expected: CalibrationExpectation,
) -> CheckResult:
    target = 1.0 if expected.expected_correct else 0.0
    brier_score = (view.confidence - target) ** 2
    passed = brier_score <= expected.maximum_brier_score
    return CheckResult(
        check_id="brier_calibration",
        passed=passed,
        score=1.0 - brier_score,
        detail="Brier score met the configured limit"
        if passed
        else "Brier score exceeded the configured limit",
        metric_name="brier_score",
        metric_value=brier_score,
    )


def _privacy_checks(
    view: _SimulationView,
    expected: PrivacyExpectation,
) -> list[CheckResult]:
    checks = []
    folded_output = tuple(value.casefold() for value in view.generated_text)
    if expected.forbidden_text_fragments:
        match_count = sum(
            any(fragment.casefold() in value for value in folded_output)
            for fragment in expected.forbidden_text_fragments
        )
        checks.append(
            _binary_check(
                "forbidden_text",
                match_count == 0,
                _count_detail("forbidden text fragment", match_count, missing=False),
                blocking=True,
            )
        )
    if expected.forbidden_state_ids:
        present = set(expected.forbidden_state_ids) & set(view.state_item_ids)
        checks.append(
            _binary_check(
                "privacy_forbidden_state",
                not present,
                _count_detail("privacy-forbidden state item", len(present), missing=False),
                blocking=True,
            )
        )
    if expected.forbidden_evidence_ids:
        present = set(expected.forbidden_evidence_ids) & set(view.evidence_ids)
        checks.append(
            _binary_check(
                "privacy_forbidden_evidence",
                not present,
                _count_detail(
                    "privacy-forbidden evidence item",
                    len(present),
                    missing=False,
                ),
                blocking=True,
            )
        )
    if expected.require_synthetic_disclosure:
        passed = bool(view.synthetic_disclosure.strip())
        checks.append(
            _binary_check(
                "synthetic_disclosure",
                passed,
                "synthetic disclosure was present"
                if passed
                else "synthetic disclosure was absent",
                blocking=True,
            )
        )
    if expected.require_authority_denied:
        passed = not view.authority_granted
        checks.append(
            _binary_check(
                "authority_denied",
                passed,
                "live authority remained denied"
                if passed
                else "live authority was granted",
                blocking=True,
            )
        )
    return checks


def _owner_checks(expected: OwnerJudgment) -> list[CheckResult]:
    return [
        _binary_check(
            "owner_approval",
            expected.approved,
            "owner approved the output"
            if expected.approved
            else "owner did not approve the output",
        ),
        CheckResult(
            check_id="owner_score",
            passed=expected.score >= expected.minimum_score,
            score=expected.score,
            detail="owner score met the configured minimum"
            if expected.score >= expected.minimum_score
            else "owner score fell below the configured minimum",
            metric_name="owner_score",
            metric_value=expected.score,
        ),
    ]


def _simulation_view(
    result: SimulationResultLike | Mapping[str, Any],
) -> _SimulationView:
    situation = _required(result, "situation")
    relationship_id = _optional_text(
        _required(situation, "relationship_id"),
        "situation.relationship_id",
    )
    target_time = _datetime(_required(situation, "target_time"), "target_time")
    selected_response = _optional_text(
        _required(result, "selected_response"),
        "selected_response",
    )
    selected_candidate_id = _optional_text(
        _required(result, "selected_candidate_id"),
        "selected_candidate_id",
    )
    if (selected_response is None) is not (selected_candidate_id is None):
        raise ValueError(
            "selected_response and selected_candidate_id must either both be set or both be null"
        )
    confidence = _unit_interval(_required(result, "confidence"), "confidence")
    evidence_ids = _result_strings(_required(result, "evidence_ids"), "evidence_ids")
    state_item_ids = _result_strings(
        _required(result, "state_item_ids"),
        "state_item_ids",
    )
    assumptions = _result_strings(_required(result, "assumptions"), "assumptions")
    conflict_item_ids = _result_strings(
        _required(result, "conflict_item_ids"),
        "conflict_item_ids",
    )
    abstention_reasons = _result_strings(
        _required(result, "abstention_reasons"),
        "abstention_reasons",
    )
    owner_question = _optional_text(
        _required(result, "owner_question"),
        "owner_question",
    )
    synthetic_disclosure = _text(
        _required(result, "synthetic_disclosure"),
        "synthetic_disclosure",
        allow_empty=True,
    )
    authority_granted = _required(result, "authority_granted")
    if not isinstance(authority_granted, bool):
        raise ValueError("authority_granted must be a boolean")

    generated_text = [
        value
        for value in (selected_response, owner_question, synthetic_disclosure)
        if value is not None
    ]
    alternatives = _read(result, "alternatives", default=_MISSING)
    if alternatives is not _MISSING:
        alternative_values = _sequence(alternatives, "alternatives")
        candidates: dict[str, str] = {}
        for index, candidate in enumerate(alternative_values):
            response_id = _text(
                _required(candidate, "response_id"),
                f"alternatives[{index}].response_id",
            )
            if response_id in candidates:
                raise ValueError("alternatives contain duplicate response ids")
            text = _text(
                _required(candidate, "text"),
                f"alternatives[{index}].text",
            )
            rationale = _text(
                _required(candidate, "rationale"),
                f"alternatives[{index}].rationale",
            )
            candidates[response_id] = text
            generated_text.extend((text, rationale))
        if selected_candidate_id is not None:
            if selected_candidate_id not in candidates:
                raise ValueError("selected_candidate_id is not present in alternatives")
            if candidates[selected_candidate_id] != selected_response:
                raise ValueError("selected_response does not match the selected candidate")

    return _SimulationView(
        selected_response=selected_response,
        selected_candidate_id=selected_candidate_id,
        confidence=confidence,
        evidence_ids=evidence_ids,
        state_item_ids=state_item_ids,
        assumptions=assumptions,
        conflict_item_ids=conflict_item_ids,
        abstention_reasons=abstention_reasons,
        owner_question=owner_question,
        synthetic_disclosure=synthetic_disclosure,
        authority_granted=authority_granted,
        relationship_id=relationship_id,
        target_time=target_time,
        generated_text=tuple(generated_text),
    )


def _required(source: object, name: str) -> Any:
    value = _read(source, name, default=_MISSING)
    if value is _MISSING:
        raise ValueError(f"missing required field: {name}")
    return value


def _read(source: object, name: str, *, default: object) -> Any:
    if isinstance(source, Mapping):
        return source.get(name, default)
    return getattr(source, name, default)


def _sequence(value: object, label: str) -> Sequence[Any]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise ValueError(f"{label} must be a sequence")
    return value


def _result_strings(value: object, label: str) -> tuple[str, ...]:
    values = _sequence(value, label)
    normalized = tuple(_text(item, f"{label} item") for item in values)
    if len(set(normalized)) != len(normalized):
        raise ValueError(f"{label} cannot contain duplicates")
    return tuple(sorted(normalized))


def _expectation_strings(values: Sequence[str], label: str) -> tuple[str, ...]:
    normalized = tuple(_text(item, f"{label} item") for item in _sequence(values, label))
    return tuple(sorted(set(normalized)))


def _relationship_ids(
    values: Sequence[str | None],
) -> tuple[str | None, ...]:
    normalized: list[str | None] = []
    for value in _sequence(values, "expected relationship ids"):
        if value is not None:
            value = _text(value, "expected relationship id")
        if value not in normalized:
            normalized.append(value)
    return tuple(sorted(normalized, key=lambda value: (value is not None, value or "")))


def _text(value: object, label: str, *, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    if not allow_empty and not value.strip():
        raise ValueError(f"{label} cannot be empty")
    if value != value.strip() and not allow_empty:
        raise ValueError(f"{label} cannot have surrounding whitespace")
    return value


def _optional_text(value: object, label: str) -> str | None:
    if value is None:
        return None
    return _text(value, label)


def _unit_interval(value: object, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    normalized = float(value)
    if not math.isfinite(normalized) or not 0.0 <= normalized <= 1.0:
        raise ValueError(f"{label} must be finite and between 0 and 1")
    return normalized


def _datetime(value: object, label: str) -> datetime:
    if isinstance(value, str):
        try:
            value = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"{label} must be an ISO-8601 datetime") from error
    if not isinstance(value, datetime):
        raise ValueError(f"{label} must be a datetime")
    _require_aware(value, label)
    return value


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


def _axis(value: EvaluationAxis | str) -> EvaluationAxis:
    try:
        return EvaluationAxis(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"unknown evaluation axis: {value!r}") from error


def _binary_check(
    check_id: str,
    passed: bool,
    detail: str,
    *,
    blocking: bool = False,
) -> CheckResult:
    return CheckResult(
        check_id=check_id,
        passed=passed,
        score=1.0 if passed else 0.0,
        detail=detail,
        blocking=blocking,
    )


def _count_detail(label: str, count: int, *, missing: bool) -> str:
    if count == 0:
        return f"no {label}s were {'missing' if missing else 'present'}"
    verb = "were missing" if missing else "were present"
    return f"{count} {label}{'' if count == 1 else 's'} {verb}"


__all__ = [
    "AxisResult",
    "CalibrationExpectation",
    "CaseResult",
    "CheckResult",
    "EvaluationAxis",
    "EvaluationCase",
    "EvaluationLab",
    "EvaluationReport",
    "OutputExpectation",
    "OwnerJudgment",
    "PrivacyExpectation",
    "REQUIRED_AXES",
    "SimulationResultLike",
    "SituationLike",
    "StateExpectation",
]
