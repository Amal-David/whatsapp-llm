"""Transactional consolidation, reflection, and non-destructive retrieval decay."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import datetime
from enum import Enum
from typing import Any, Protocol

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


class UpdateAction(str, Enum):
    CREATE = "create"
    STRENGTHEN = "strengthen"
    CONTEXTUALIZE = "contextualize"
    SUPERSEDE = "supersede"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class ConsolidationProposal:
    """A policy proposal that cannot write state until the engine validates it."""

    proposal_id: str
    action: UpdateAction
    layer: BrainLayer
    kind: str
    summary: str
    payload: dict[str, Any]
    confidence: float
    source_item_ids: tuple[str, ...]
    target_item_id: str | None
    rationale: str
    uncertainty: str
    deliberately_not_inferred: tuple[str, ...]
    epistemic_status: EpistemicStatus
    scope: ContextScope
    sensitivity: Sensitivity
    ownership: Ownership

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _json_copy(self.payload))
        object.__setattr__(
            self,
            "source_item_ids",
            tuple(sorted(set(self.source_item_ids))),
        )
        object.__setattr__(
            self,
            "deliberately_not_inferred",
            tuple(sorted(set(self.deliberately_not_inferred))),
        )
        self.validate()

    @classmethod
    def create(
        cls,
        *,
        action: UpdateAction,
        layer: BrainLayer,
        kind: str,
        summary: str,
        payload: dict[str, Any],
        confidence: float,
        source_item_ids: tuple[str, ...] | list[str],
        rationale: str,
        uncertainty: str,
        deliberately_not_inferred: tuple[str, ...] | list[str],
        target_item_id: str | None = None,
        epistemic_status: EpistemicStatus = EpistemicStatus.INFERRED,
        scope: ContextScope | None = None,
        sensitivity: Sensitivity = Sensitivity.PRIVATE,
        ownership: Ownership = Ownership.OWNER,
    ) -> ConsolidationProposal:
        values = {
            "action": action,
            "layer": layer,
            "kind": kind.strip(),
            "summary": summary.strip(),
            "payload": _json_copy(payload),
            "confidence": float(confidence),
            "source_item_ids": tuple(sorted(set(source_item_ids))),
            "target_item_id": target_item_id,
            "rationale": rationale.strip(),
            "uncertainty": uncertainty.strip(),
            "deliberately_not_inferred": tuple(
                sorted(set(deliberately_not_inferred))
            ),
            "epistemic_status": epistemic_status,
            "scope": scope or ContextScope(),
            "sensitivity": sensitivity,
            "ownership": ownership,
        }
        proposal_id = _canonical_hash("proposal", cls._identity_payload(values))
        return cls(proposal_id=proposal_id, **values)

    def validate(self) -> None:
        if not self.proposal_id.startswith("proposal:"):
            raise ValueError("proposal id must start with proposal:")
        if self.layer in {BrainLayer.EVENT, BrainLayer.REFLECTION}:
            raise ValueError("consolidation cannot create event or reflection proposals")
        if not self.kind.strip() or not self.summary.strip():
            raise ValueError("proposal kind and summary are required")
        if not self.rationale.strip() or not self.uncertainty.strip():
            raise ValueError("proposal rationale and uncertainty are required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("proposal confidence must be between 0 and 1")
        if not self.source_item_ids or any(
            not item_id.startswith("brain-item:") for item_id in self.source_item_ids
        ):
            raise ValueError("proposal source_item_ids must contain brain-item ids")
        if self.action is UpdateAction.CREATE and self.target_item_id is not None:
            raise ValueError("create proposals cannot target an existing item")
        if self.action is not UpdateAction.CREATE and (
            self.target_item_id is None
            or not self.target_item_id.startswith("brain-item:")
        ):
            raise ValueError(f"{self.action.value} proposals require a target item")
        expected = _canonical_hash("proposal", self._identity_payload(self.__dict__))
        if expected != self.proposal_id:
            raise ValueError("proposal id does not match canonical content")

    @staticmethod
    def _identity_payload(values: dict[str, Any]) -> dict[str, Any]:
        action = values["action"]
        layer = values["layer"]
        epistemic = values["epistemic_status"]
        scope = values["scope"]
        sensitivity = values["sensitivity"]
        ownership = values["ownership"]
        return {
            "action": action.value if isinstance(action, UpdateAction) else action,
            "layer": layer.value if isinstance(layer, BrainLayer) else layer,
            "kind": values["kind"],
            "summary": values["summary"],
            "payload": values["payload"],
            "confidence": values["confidence"],
            "source_item_ids": sorted(values["source_item_ids"]),
            "target_item_id": values.get("target_item_id"),
            "rationale": values["rationale"],
            "uncertainty": values["uncertainty"],
            "deliberately_not_inferred": sorted(
                values["deliberately_not_inferred"]
            ),
            "epistemic_status": (
                epistemic.value
                if isinstance(epistemic, EpistemicStatus)
                else epistemic
            ),
            "scope": scope.to_dict() if isinstance(scope, ContextScope) else scope,
            "sensitivity": (
                sensitivity.value
                if isinstance(sensitivity, Sensitivity)
                else sensitivity
            ),
            "ownership": (
                ownership.value if isinstance(ownership, Ownership) else ownership
            ),
        }


class ConsolidationPolicy(Protocol):
    policy_id: str

    def propose(
        self,
        brain: DigitalBrain,
        source_items: Sequence[StateItem],
        as_of: datetime,
    ) -> Sequence[ConsolidationProposal]: ...


class StructuredEventPolicy:
    """Deterministic policy for explicit, pre-structured event proposals."""

    policy_id = "structured_event.v1"

    def propose(
        self,
        brain: DigitalBrain,
        source_items: Sequence[StateItem],
        as_of: datetime,
    ) -> Sequence[ConsolidationProposal]:
        del brain, as_of
        proposals = []
        for source in sorted(source_items, key=lambda item: item.id):
            directives = source.payload.get("consolidation_proposals", [])
            if not isinstance(directives, list):
                raise ValueError("consolidation_proposals must be a list")
            for directive in directives:
                if not isinstance(directive, dict):
                    raise ValueError("consolidation proposal directives must be objects")
                scope_value = directive.get("scope")
                scope = (
                    ContextScope.from_dict(scope_value)
                    if isinstance(scope_value, dict)
                    else source.scope
                )
                proposals.append(
                    ConsolidationProposal.create(
                        action=UpdateAction(directive["action"]),
                        layer=BrainLayer(directive["layer"]),
                        kind=directive["kind"],
                        summary=directive["summary"],
                        payload=dict(directive.get("payload", {})),
                        confidence=float(directive["confidence"]),
                        source_item_ids=(source.id,),
                        target_item_id=directive.get("target_item_id"),
                        rationale=directive["rationale"],
                        uncertainty=directive["uncertainty"],
                        deliberately_not_inferred=tuple(
                            directive.get("deliberately_not_inferred", [])
                        ),
                        epistemic_status=EpistemicStatus(
                            directive.get("epistemic_status", "inferred")
                        ),
                        scope=scope,
                        sensitivity=Sensitivity(
                            directive.get("sensitivity", source.sensitivity.value)
                        ),
                        ownership=Ownership(
                            directive.get("ownership", Ownership.OWNER.value)
                        ),
                    )
                )
        return proposals


@dataclass(frozen=True)
class ConsolidationResult:
    run_id: str
    policy_id: str
    proposal_ids: tuple[str, ...]
    added_item_ids: tuple[str, ...]
    reflection_item_ids: tuple[str, ...]


class ConsolidationEngine:
    """Validate and apply one consolidation transaction with explicit reflection."""

    def __init__(self, policy: ConsolidationPolicy):
        self.policy = policy

    def run(
        self,
        brain: DigitalBrain,
        *,
        source_item_ids: Sequence[str],
        as_of: datetime,
    ) -> ConsolidationResult:
        _require_aware(as_of, "consolidation as_of")
        requested_ids = tuple(sorted(set(source_item_ids)))
        if not requested_ids:
            raise ValueError("consolidation requires source event ids")
        source_items = tuple(brain.item_by_id(item_id) for item_id in requested_ids)
        for source in source_items:
            if source.layer is not BrainLayer.EVENT:
                raise ValueError("consolidation sources must be event-layer items")
            if source.status in {StateStatus.REJECTED, StateStatus.DELETED}:
                raise ValueError("rejected or deleted events cannot be consolidated")

        proposals = sorted(
            self.policy.propose(brain, source_items, as_of),
            key=lambda proposal: proposal.proposal_id,
        )
        proposal_ids = tuple(proposal.proposal_id for proposal in proposals)
        run_id = _canonical_hash(
            "consolidation",
            {
                "policy_id": self.policy.policy_id,
                "as_of": as_of.isoformat(),
                "source_item_ids": list(requested_ids),
                "proposal_ids": list(proposal_ids),
            },
        )
        completed_keys = {
            (
                str(item.payload.get("policy_id")),
                str(item.payload.get("proposal_id")),
            )
            for item in brain.items
            if item.layer is BrainLayer.REFLECTION
            and item.kind == "consolidation.reflection"
        }
        pending = [
            proposal
            for proposal in proposals
            if (self.policy.policy_id, proposal.proposal_id) not in completed_keys
        ]
        if not pending:
            return ConsolidationResult(
                run_id=run_id,
                policy_id=self.policy.policy_id,
                proposal_ids=proposal_ids,
                added_item_ids=(),
                reflection_item_ids=(),
            )

        working = {item.id: item for item in brain.items}
        changed_target_ids: set[str] = set()
        added_item_ids = []
        reflection_item_ids = []
        requested_set = set(requested_ids)

        for proposal in pending:
            proposal.validate()
            if not set(proposal.source_item_ids) <= requested_set:
                raise ValueError("proposal references events outside this transaction")
            proposal_sources = [working[item_id] for item_id in proposal.source_item_ids]
            if any(source.layer is not BrainLayer.EVENT for source in proposal_sources):
                raise ValueError("proposal source references must resolve to events")
            has_third_party_source = any(
                source.ownership is Ownership.THIRD_PARTY
                for source in proposal_sources
            )
            if has_third_party_source:
                if proposal.layer is not BrainLayer.SOCIAL:
                    raise ValueError(
                        "third-party evidence cannot be promoted into owner identity"
                    )
                if proposal.ownership not in {
                    Ownership.SHARED,
                    Ownership.THIRD_PARTY,
                }:
                    raise ValueError(
                        "third-party social state must remain shared or third-party"
                    )

            target = None
            if proposal.target_item_id:
                target = working.get(proposal.target_item_id)
                if target is None:
                    raise ValueError("proposal target is missing")
                if target.id in changed_target_ids:
                    raise ValueError("multiple proposals cannot replace one target")
                if target.layer is not proposal.layer or target.kind != proposal.kind:
                    raise ValueError("proposal target layer and kind must match")
                if target.status not in {StateStatus.ACTIVE, StateStatus.PROPOSED}:
                    raise ValueError("proposal target must be active or proposed")

            sensitivity = _maximum_sensitivity(
                [proposal.sensitivity]
                + [source.sensitivity for source in proposal_sources]
                + ([target.sensitivity] if target else [])
            )
            support_ids = set(proposal.source_item_ids)
            contradiction_ids: tuple[str, ...] = ()
            supersedes = None
            if target and proposal.action is UpdateAction.CONFLICT:
                contradiction_ids = (target.id,)
            elif target:
                support_ids.add(target.id)
                supersedes = target.id
                changed_target_ids.add(target.id)

            provenance = [
                ProvenanceRef(
                    source_id=source.id,
                    source_type="digital_brain_state",
                    relation=ProvenanceRelation.DERIVED_FROM,
                    observed_at=as_of,
                )
                for source in proposal_sources
            ]
            if target:
                provenance.append(
                    ProvenanceRef(
                        source_id=target.id,
                        source_type="digital_brain_state",
                        relation=(
                            ProvenanceRelation.CONTRADICTS
                            if proposal.action is UpdateAction.CONFLICT
                            else ProvenanceRelation.DERIVED_FROM
                        ),
                        observed_at=as_of,
                    )
                )
            changed = StateItem.create(
                layer=proposal.layer,
                kind=proposal.kind,
                summary=proposal.summary,
                payload=proposal.payload,
                epistemic_status=proposal.epistemic_status,
                status=StateStatus.ACTIVE,
                confidence=proposal.confidence,
                temporal=TemporalScope(
                    observed_at=as_of,
                    recorded_at=as_of,
                    valid_from=as_of,
                    last_confirmed_at=(
                        as_of
                        if proposal.epistemic_status is EpistemicStatus.OWNER_DECLARED
                        else None
                    ),
                ),
                scope=proposal.scope,
                sensitivity=sensitivity,
                ownership=proposal.ownership,
                provenance=tuple(provenance),
                support_ids=tuple(sorted(support_ids)),
                contradiction_ids=contradiction_ids,
                supersedes=supersedes,
                metadata={
                    "consolidation_action": proposal.action.value,
                    "consolidation_policy": self.policy.policy_id,
                    "proposal_id": proposal.proposal_id,
                },
            )
            if target and proposal.action is not UpdateAction.CONFLICT:
                working[target.id] = replace(target, status=StateStatus.SUPERSEDED)
            working[changed.id] = changed
            added_item_ids.append(changed.id)

            reflection_support_ids = set(proposal.source_item_ids) | {changed.id}
            if target:
                reflection_support_ids.add(target.id)
            reflection = StateItem.create(
                layer=BrainLayer.REFLECTION,
                kind="consolidation.reflection",
                summary=(
                    f"{proposal.action.value} {proposal.layer.value}/{proposal.kind}"
                ),
                payload={
                    "run_id": run_id,
                    "policy_id": self.policy.policy_id,
                    "proposal_id": proposal.proposal_id,
                    "action": proposal.action.value,
                    "changed_item_id": changed.id,
                    "target_item_id": proposal.target_item_id,
                    "source_item_ids": list(proposal.source_item_ids),
                    "rationale": proposal.rationale,
                    "uncertainty": proposal.uncertainty,
                    "deliberately_not_inferred": list(
                        proposal.deliberately_not_inferred
                    ),
                },
                epistemic_status=EpistemicStatus.GENERATED_PROPOSAL,
                status=StateStatus.ACTIVE,
                confidence=1.0,
                temporal=TemporalScope(
                    observed_at=as_of,
                    recorded_at=as_of,
                    valid_from=as_of,
                ),
                scope=proposal.scope,
                sensitivity=sensitivity,
                ownership=Ownership.SYSTEM,
                provenance=tuple(
                    ProvenanceRef(
                        source_id=item_id,
                        source_type="digital_brain_state",
                        relation=ProvenanceRelation.DERIVED_FROM,
                        observed_at=as_of,
                    )
                    for item_id in sorted(reflection_support_ids)
                ),
                support_ids=tuple(sorted(reflection_support_ids)),
                metadata={
                    "policy_id": self.policy.policy_id,
                    "proposal_id": proposal.proposal_id,
                },
            )
            working[reflection.id] = reflection
            reflection_item_ids.append(reflection.id)

        candidate = DigitalBrain(
            brain_id=brain.brain_id,
            owner_id=brain.owner_id,
            owner_name=brain.owner_name,
            schema_version=brain.schema_version,
            version=brain.version + 1,
            created_at=brain.created_at,
            updated_at=max(brain.updated_at, as_of),
            items=sorted(working.values(), key=lambda item: item.id),
            metadata=_json_copy(brain.metadata),
        )
        candidate.validate()
        brain.items = candidate.items
        brain.version = candidate.version
        brain.updated_at = candidate.updated_at
        return ConsolidationResult(
            run_id=run_id,
            policy_id=self.policy.policy_id,
            proposal_ids=proposal_ids,
            added_item_ids=tuple(added_item_ids),
            reflection_item_ids=tuple(reflection_item_ids),
        )


@dataclass(frozen=True)
class RankedState:
    item_id: str
    score: float
    age_days: float
    protected_audit: bool


def rank_for_retrieval(
    items: Sequence[StateItem],
    *,
    as_of: datetime,
    half_life_days: float,
) -> list[RankedState]:
    """Apply reversible access decay without mutating or deleting canonical state."""
    _require_aware(as_of, "retrieval as_of")
    if half_life_days <= 0:
        raise ValueError("half_life_days must be positive")
    superseded_at_point = {
        item.supersedes
        for item in items
        if item.supersedes
        and item.status not in {StateStatus.REJECTED, StateStatus.DELETED}
        and item.temporal.contains(as_of)
    }
    epistemic_weight = {
        EpistemicStatus.OWNER_DECLARED: 1.0,
        EpistemicStatus.OBSERVED: 0.95,
        EpistemicStatus.INFERRED: 0.85,
        EpistemicStatus.GENERATED_PROPOSAL: 0.7,
        EpistemicStatus.ASSUMED: 0.5,
    }
    ranked = []
    for item in items:
        if item.id in superseded_at_point or not item.is_visible_at(
            as_of,
            include_proposed=False,
        ):
            continue
        anchor = (
            item.temporal.last_confirmed_at
            or item.temporal.valid_from
            or item.temporal.observed_at
        )
        age_days = max(0.0, (as_of - anchor).total_seconds() / 86400)
        decay = math.pow(0.5, age_days / half_life_days)
        ranked.append(
            RankedState(
                item_id=item.id,
                score=item.confidence * epistemic_weight[item.epistemic_status] * decay,
                age_days=age_days,
                protected_audit=item.layer in {
                    BrainLayer.EVENT,
                    BrainLayer.REFLECTION,
                },
            )
        )
    return sorted(ranked, key=lambda item: (-item.score, item.item_id))


def _maximum_sensitivity(values: Sequence[Sensitivity]) -> Sensitivity:
    rank = {
        Sensitivity.PUBLIC: 0,
        Sensitivity.PRIVATE: 1,
        Sensitivity.SENSITIVE: 2,
        Sensitivity.RESTRICTED: 3,
    }
    return max(values, key=lambda value: rank[value])


def _canonical_hash(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _json_copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, sort_keys=True))


def _require_aware(value: datetime, label: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")


__all__ = [
    "ConsolidationEngine",
    "ConsolidationPolicy",
    "ConsolidationProposal",
    "ConsolidationResult",
    "RankedState",
    "StructuredEventPolicy",
    "UpdateAction",
    "rank_for_retrieval",
]
