"""Relationship-scoped retrieval, deliberation, and authority-safe simulation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

from .consolidation import rank_for_retrieval
from .models import (
    BrainLayer,
    ContextScope,
    DigitalBrain,
    EpistemicStatus,
    Ownership,
    Sensitivity,
    StateItem,
)


class Stakes(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class RequestedAuthority(str, Enum):
    PRIVATE_DRAFT = "private_draft"
    SUGGESTION = "suggestion"
    SEND_MESSAGE = "send_message"
    MAKE_COMMITMENT = "make_commitment"
    TRANSACT = "transact"
    REPRESENT_OWNER = "represent_owner"


@dataclass(frozen=True)
class Situation:
    situation_id: str
    request: str
    audience: str
    relationship_id: str | None
    role: str | None
    intent: str
    stakes: Stakes
    target_time: datetime
    channel: str
    authority: RequestedAuthority
    assumptions: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "assumptions", tuple(sorted(set(self.assumptions))))
        self.validate()

    @classmethod
    def create(
        cls,
        *,
        request: str,
        audience: str,
        relationship_id: str | None,
        role: str | None,
        intent: str,
        stakes: Stakes,
        target_time: datetime,
        channel: str,
        authority: RequestedAuthority,
        assumptions: tuple[str, ...] | list[str] = (),
    ) -> Situation:
        normalized_request = request.strip()
        normalized_audience = audience.strip()
        normalized_intent = intent.strip()
        normalized_channel = channel.strip()
        normalized_assumptions = tuple(sorted(set(assumptions)))
        values: dict[str, Any] = {
            "request": normalized_request,
            "audience": normalized_audience,
            "relationship_id": relationship_id,
            "role": role,
            "intent": normalized_intent,
            "stakes": stakes,
            "target_time": target_time,
            "channel": normalized_channel,
            "authority": authority,
            "assumptions": normalized_assumptions,
        }
        situation_id = _canonical_hash("situation", cls._identity_payload(values))
        return cls(
            situation_id=situation_id,
            request=normalized_request,
            audience=normalized_audience,
            relationship_id=relationship_id,
            role=role,
            intent=normalized_intent,
            stakes=stakes,
            target_time=target_time,
            channel=normalized_channel,
            authority=authority,
            assumptions=normalized_assumptions,
        )

    def validate(self) -> None:
        if not self.situation_id.startswith("situation:"):
            raise ValueError("situation id must start with situation:")
        if not self.request or not self.audience or not self.intent or not self.channel:
            raise ValueError("situation request, audience, intent, and channel are required")
        if self.relationship_id is not None and not self.relationship_id.strip():
            raise ValueError("situation relationship_id cannot be empty")
        if self.role is not None and not self.role.strip():
            raise ValueError("situation role cannot be empty")
        _require_aware(self.target_time, "situation target_time")
        if any(not assumption.strip() for assumption in self.assumptions):
            raise ValueError("situation assumptions cannot be empty")
        expected = _canonical_hash("situation", self._identity_payload(self.__dict__))
        if expected != self.situation_id:
            raise ValueError("situation id does not match canonical content")

    @staticmethod
    def _identity_payload(values: dict[str, Any]) -> dict[str, Any]:
        stakes = values["stakes"]
        authority = values["authority"]
        target_time = values["target_time"]
        return {
            "request": values["request"],
            "audience": values["audience"],
            "relationship_id": values.get("relationship_id"),
            "role": values.get("role"),
            "intent": values["intent"],
            "stakes": stakes.value if isinstance(stakes, Stakes) else stakes,
            "target_time": (
                target_time.isoformat()
                if isinstance(target_time, datetime)
                else target_time
            ),
            "channel": values["channel"],
            "authority": (
                authority.value
                if isinstance(authority, RequestedAuthority)
                else authority
            ),
            "assumptions": sorted(values.get("assumptions", ())),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "situation_id": self.situation_id,
            **self._identity_payload(self.__dict__),
        }


@dataclass(frozen=True)
class ContextItem:
    item_id: str
    layer: BrainLayer
    kind: str
    summary: str
    confidence: float
    epistemic_status: EpistemicStatus
    sensitivity: Sensitivity
    scope: ContextScope
    evidence_ids: tuple[str, ...]

    @classmethod
    def from_state(cls, item: StateItem) -> ContextItem:
        return cls(
            item_id=item.id,
            layer=item.layer,
            kind=item.kind,
            summary=item.summary,
            confidence=item.confidence,
            epistemic_status=item.epistemic_status,
            sensitivity=item.sensitivity,
            scope=item.scope,
            evidence_ids=tuple(
                sorted({provenance.source_id for provenance in item.provenance})
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "layer": self.layer.value,
            "kind": self.kind,
            "summary": self.summary,
            "confidence": self.confidence,
            "epistemic_status": self.epistemic_status.value,
            "sensitivity": self.sensitivity.value,
            "scope": self.scope.to_dict(),
            "evidence_ids": list(self.evidence_ids),
        }


@dataclass(frozen=True)
class RetrievedContext:
    episodes: tuple[ContextItem, ...]
    self_knowledge: tuple[ContextItem, ...]
    values_goals: tuple[ContextItem, ...]
    relationship_state: tuple[ContextItem, ...]
    affect: tuple[ContextItem, ...]
    communication: tuple[ContextItem, ...]
    procedures: tuple[ContextItem, ...]
    uncertainties: tuple[ContextItem, ...]
    conflict_item_ids: tuple[str, ...]

    @property
    def all_items(self) -> tuple[ContextItem, ...]:
        return (
            *self.episodes,
            *self.self_knowledge,
            *self.values_goals,
            *self.relationship_state,
            *self.affect,
            *self.communication,
            *self.procedures,
            *self.uncertainties,
        )

    @property
    def state_item_ids(self) -> tuple[str, ...]:
        return tuple(sorted({item.item_id for item in self.all_items}))

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            sorted(
                {
                    evidence_id
                    for item in self.all_items
                    for evidence_id in item.evidence_ids
                }
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "episodes": [item.to_dict() for item in self.episodes],
            "self_knowledge": [item.to_dict() for item in self.self_knowledge],
            "values_goals": [item.to_dict() for item in self.values_goals],
            "relationship_state": [
                item.to_dict() for item in self.relationship_state
            ],
            "affect": [item.to_dict() for item in self.affect],
            "communication": [item.to_dict() for item in self.communication],
            "procedures": [item.to_dict() for item in self.procedures],
            "uncertainties": [item.to_dict() for item in self.uncertainties],
            "state_item_ids": list(self.state_item_ids),
            "evidence_ids": list(self.evidence_ids),
            "conflict_item_ids": list(self.conflict_item_ids),
        }


@dataclass(frozen=True)
class CandidateResponse:
    response_id: str
    text: str
    rationale: str
    used_state_ids: tuple[str, ...]
    assumptions: tuple[str, ...]
    confidence: float
    requires_live_authority: bool

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "used_state_ids",
            tuple(sorted(set(self.used_state_ids))),
        )
        object.__setattr__(self, "assumptions", tuple(sorted(set(self.assumptions))))
        self.validate()

    @classmethod
    def create(
        cls,
        *,
        text: str,
        rationale: str,
        used_state_ids: tuple[str, ...] | list[str],
        assumptions: tuple[str, ...] | list[str],
        confidence: float,
        requires_live_authority: bool = False,
    ) -> CandidateResponse:
        normalized_text = text.strip()
        normalized_rationale = rationale.strip()
        normalized_state_ids = tuple(sorted(set(used_state_ids)))
        normalized_assumptions = tuple(sorted(set(assumptions)))
        normalized_confidence = float(confidence)
        normalized_authority = bool(requires_live_authority)
        values: dict[str, Any] = {
            "text": normalized_text,
            "rationale": normalized_rationale,
            "used_state_ids": normalized_state_ids,
            "assumptions": normalized_assumptions,
            "confidence": normalized_confidence,
            "requires_live_authority": normalized_authority,
        }
        response_id = _canonical_hash("response", values)
        return cls(
            response_id=response_id,
            text=normalized_text,
            rationale=normalized_rationale,
            used_state_ids=normalized_state_ids,
            assumptions=normalized_assumptions,
            confidence=normalized_confidence,
            requires_live_authority=normalized_authority,
        )

    def validate(self) -> None:
        if not self.response_id.startswith("response:"):
            raise ValueError("response id must start with response:")
        if not self.text or not self.rationale:
            raise ValueError("candidate text and rationale are required")
        if not self.used_state_ids or any(
            not item_id.startswith("brain-item:") for item_id in self.used_state_ids
        ):
            raise ValueError("candidate responses require grounded state ids")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("candidate confidence must be between 0 and 1")
        expected = _canonical_hash(
            "response",
            {
                "text": self.text,
                "rationale": self.rationale,
                "used_state_ids": self.used_state_ids,
                "assumptions": self.assumptions,
                "confidence": self.confidence,
                "requires_live_authority": self.requires_live_authority,
            },
        )
        if expected != self.response_id:
            raise ValueError("response id does not match canonical content")

    def to_dict(self) -> dict[str, Any]:
        return {
            "response_id": self.response_id,
            "text": self.text,
            "rationale": self.rationale,
            "used_state_ids": list(self.used_state_ids),
            "assumptions": list(self.assumptions),
            "confidence": self.confidence,
            "requires_live_authority": self.requires_live_authority,
        }


class DeliberationProvider(Protocol):
    provider_id: str

    def deliberate(
        self,
        situation: Situation,
        context: RetrievedContext,
    ) -> Sequence[CandidateResponse]: ...


class TextCompletionProvider(Protocol):
    provider_id: str

    def complete(self, request: str) -> str: ...


class LLMDeliberationProvider:
    """Optional provider adapter using a strict JSON input and output contract."""

    def __init__(self, provider: TextCompletionProvider):
        self.provider = provider
        self.provider_id = f"llm-json:{provider.provider_id}"

    def deliberate(
        self,
        situation: Situation,
        context: RetrievedContext,
    ) -> Sequence[CandidateResponse]:
        request = json.dumps(
            {
                "schema_version": "digital_brain_deliberation.v1",
                "situation": situation.to_dict(),
                "context": context.to_dict(),
                "required_output": {
                    "alternatives": [
                        {
                            "text": "string",
                            "rationale": "string",
                            "used_state_ids": ["brain-item id"],
                            "assumptions": ["string"],
                            "confidence": "0 to 1",
                            "requires_live_authority": False,
                        }
                    ]
                },
            },
            ensure_ascii=False,
            sort_keys=True,
        )
        value = json.loads(self.provider.complete(request))
        if not isinstance(value, dict) or not isinstance(
            value.get("alternatives"),
            list,
        ):
            raise ValueError("deliberation provider must return alternatives JSON")
        alternatives = value["alternatives"]
        if not alternatives or not all(isinstance(item, dict) for item in alternatives):
            raise ValueError("deliberation provider alternatives must be objects")
        return [
            CandidateResponse.create(
                text=item["text"],
                rationale=item["rationale"],
                used_state_ids=tuple(item["used_state_ids"]),
                assumptions=tuple(item.get("assumptions", [])),
                confidence=float(item["confidence"]),
                requires_live_authority=bool(
                    item.get("requires_live_authority", False)
                ),
            )
            for item in alternatives
        ]


@dataclass(frozen=True)
class SimulationResult:
    situation: Situation
    context: RetrievedContext
    alternatives: tuple[CandidateResponse, ...]
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
    authority_granted: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": "digital_brain_simulation.v1",
            "situation": self.situation.to_dict(),
            "context": self.context.to_dict(),
            "alternatives": [item.to_dict() for item in self.alternatives],
            "selected_response": self.selected_response,
            "selected_candidate_id": self.selected_candidate_id,
            "confidence": self.confidence,
            "evidence_ids": list(self.evidence_ids),
            "state_item_ids": list(self.state_item_ids),
            "assumptions": list(self.assumptions),
            "conflict_item_ids": list(self.conflict_item_ids),
            "abstention_reasons": list(self.abstention_reasons),
            "owner_question": self.owner_question,
            "synthetic_disclosure": self.synthetic_disclosure,
            "authority_granted": self.authority_granted,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"


class SimulationEngine:
    """Retrieve, deliberate, and gate one owner-controlled private simulation."""

    def __init__(
        self,
        provider: DeliberationProvider,
        *,
        minimum_grounded_items: int = 2,
        maximum_items_per_category: int = 6,
    ):
        if minimum_grounded_items < 1 or maximum_items_per_category < 1:
            raise ValueError("simulation retrieval limits must be positive")
        self.provider = provider
        self.minimum_grounded_items = minimum_grounded_items
        self.maximum_items_per_category = maximum_items_per_category

    def simulate(self, brain: DigitalBrain, situation: Situation) -> SimulationResult:
        brain.validate()
        situation.validate()
        context = self._retrieve(brain, situation)
        reasons = []
        if situation.authority not in {
            RequestedAuthority.PRIVATE_DRAFT,
            RequestedAuthority.SUGGESTION,
        }:
            reasons.append("live_authority_not_permitted")
        if situation.stakes in {Stakes.HIGH, Stakes.CRITICAL}:
            reasons.append("high_stakes_owner_required")
        if len(context.state_item_ids) < self.minimum_grounded_items:
            reasons.append("insufficient_grounding")
        if reasons:
            return self._result(
                situation,
                context,
                alternatives=(),
                selected=None,
                confidence=0.0,
                assumptions=situation.assumptions,
                reasons=tuple(sorted(set(reasons))),
                owner_question=_owner_question(reasons),
            )

        valid_candidates = []
        known_state_ids = set(context.state_item_ids)
        for candidate in self.provider.deliberate(situation, context):
            candidate.validate()
            if not set(candidate.used_state_ids) <= known_state_ids:
                reasons.append("ungrounded_candidate_rejected")
                continue
            if candidate.requires_live_authority:
                reasons.append("live_authority_candidate_rejected")
                continue
            valid_candidates.append(candidate)
        valid_candidates.sort(
            key=lambda candidate: (-candidate.confidence, candidate.response_id)
        )
        if not valid_candidates:
            reasons.append("no_grounded_candidate")
            return self._result(
                situation,
                context,
                alternatives=(),
                selected=None,
                confidence=0.0,
                assumptions=situation.assumptions,
                reasons=tuple(sorted(set(reasons))),
                owner_question="What should the system know before drafting this?",
            )

        selected = valid_candidates[0]
        used_items = [
            item
            for item in context.all_items
            if item.item_id in set(selected.used_state_ids)
        ]
        evidence_confidence = sum(item.confidence for item in used_items) / len(
            used_items
        )
        conflict_multiplier = 0.7 if context.conflict_item_ids else 1.0
        confidence = min(selected.confidence, evidence_confidence) * conflict_multiplier
        assumptions = tuple(
            sorted({*situation.assumptions, *selected.assumptions})
        )
        return self._result(
            situation,
            context,
            alternatives=tuple(valid_candidates),
            selected=selected,
            confidence=confidence,
            assumptions=assumptions,
            reasons=tuple(sorted(set(reasons))),
            owner_question=None,
        )

    def _retrieve(self, brain: DigitalBrain, situation: Situation) -> RetrievedContext:
        current = [
            item
            for item in brain.query(
                as_of=situation.target_time,
                relationship_id=situation.relationship_id,
                role=situation.role,
                audience=situation.audience,
                channel=situation.channel,
            )
            if item.layer
            not in {BrainLayer.EVENT, BrainLayer.REFLECTION}
            and item.ownership in {Ownership.OWNER, Ownership.SYSTEM}
            and item.sensitivity
            not in {Sensitivity.SENSITIVE, Sensitivity.RESTRICTED}
        ]
        rank = {
            item.item_id: item.score
            for item in rank_for_retrieval(
                current,
                as_of=situation.target_time,
                half_life_days=180,
            )
        }
        query_terms = _terms(f"{situation.request} {situation.intent}")

        def select(layers: set[BrainLayer]) -> tuple[ContextItem, ...]:
            candidates = [item for item in current if item.layer in layers]
            candidates.sort(
                key=lambda item: (
                    -_overlap(query_terms, _terms(f"{item.kind} {item.summary}")),
                    -rank.get(item.id, 0.0),
                    item.id,
                )
            )
            return tuple(
                ContextItem.from_state(item)
                for item in candidates[: self.maximum_items_per_category]
            )

        episodes = select({BrainLayer.EPISODE})
        self_knowledge = select(
            {
                BrainLayer.SEMANTIC,
                BrainLayer.SELF_SCHEMA,
                BrainLayer.NARRATIVE,
            }
        )
        values_goals = select({BrainLayer.VALUES_GOALS})
        relationship_state = select({BrainLayer.SOCIAL})
        affect = select({BrainLayer.AFFECT})
        communication = select({BrainLayer.COMMUNICATION})
        procedures = select({BrainLayer.PROCEDURAL})
        uncertainties = select({BrainLayer.UNCERTAINTY})
        selected = (
            *episodes,
            *self_knowledge,
            *values_goals,
            *relationship_state,
            *affect,
            *communication,
            *procedures,
            *uncertainties,
        )
        selected_ids = {item.item_id for item in selected}
        conflicts = set()
        for state in current:
            if state.id not in selected_ids:
                continue
            linked_conflicts = set(state.contradiction_ids) & selected_ids
            if linked_conflicts:
                conflicts.add(state.id)
                conflicts.update(linked_conflicts)
        groups: dict[tuple[str, str, str | None], list[str]] = defaultdict(list)
        for item in selected:
            groups[
                (item.layer.value, item.kind, item.scope.relationship_id)
            ].append(item.item_id)
        for item_ids in groups.values():
            if len(item_ids) > 1:
                conflicts.update(item_ids)
        return RetrievedContext(
            episodes=episodes,
            self_knowledge=self_knowledge,
            values_goals=values_goals,
            relationship_state=relationship_state,
            affect=affect,
            communication=communication,
            procedures=procedures,
            uncertainties=uncertainties,
            conflict_item_ids=tuple(sorted(conflicts)),
        )

    def _result(
        self,
        situation: Situation,
        context: RetrievedContext,
        *,
        alternatives: tuple[CandidateResponse, ...],
        selected: CandidateResponse | None,
        confidence: float,
        assumptions: tuple[str, ...],
        reasons: tuple[str, ...],
        owner_question: str | None,
    ) -> SimulationResult:
        return SimulationResult(
            situation=situation,
            context=context,
            alternatives=alternatives,
            selected_response=selected.text if selected else None,
            selected_candidate_id=selected.response_id if selected else None,
            confidence=confidence,
            evidence_ids=context.evidence_ids,
            state_item_ids=context.state_item_ids,
            assumptions=assumptions,
            conflict_item_ids=context.conflict_item_ids,
            abstention_reasons=reasons,
            owner_question=owner_question,
            synthetic_disclosure=(
                "Synthetic private simulation; not authored, sent, or authorized "
                "by the owner."
            ),
            authority_granted=False,
        )


def _owner_question(reasons: Sequence[str]) -> str:
    if "live_authority_not_permitted" in reasons:
        return "Only the owner can authorize or perform this action."
    if "high_stakes_owner_required" in reasons:
        return "What does the owner want to decide or say in this high-stakes case?"
    return "What should the system know before drafting this?"


def _terms(value: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", value.lower()))


def _overlap(left: set[str], right: set[str]) -> float:
    if not left:
        return 0.0
    return len(left & right) / len(left)


def _canonical_hash(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


__all__ = [
    "CandidateResponse",
    "ContextItem",
    "DeliberationProvider",
    "LLMDeliberationProvider",
    "RequestedAuthority",
    "RetrievedContext",
    "SimulationEngine",
    "SimulationResult",
    "Stakes",
    "Situation",
    "TextCompletionProvider",
]
