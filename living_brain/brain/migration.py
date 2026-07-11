"""Deterministic migration from the legacy digital-self profile to brain v2."""

from __future__ import annotations

from typing import Any

from ..identity.models import (
    ClaimStatus,
    DigitalSelfProfile,
    IdentityClaim,
    ProvenanceType,
)
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


def migrate_v1_profile(profile: DigitalSelfProfile) -> DigitalBrain:
    """Convert digital_self.v1 without changing the input profile."""
    profile.validate()
    if profile.schema_version != "digital_self.v1":
        raise ValueError(f"unsupported migration source: {profile.schema_version}")

    items: list[StateItem] = []
    evidence_item_ids: dict[str, str] = {}
    for evidence in sorted(profile.evidence, key=lambda item: item.id):
        ownership = (
            Ownership.THIRD_PARTY
            if evidence.metadata.get("identity_target") is False
            else Ownership.OWNER
        )
        sensitivity = _sensitivity(
            evidence.metadata.get("sensitivity"),
            default=(
                Sensitivity.RESTRICTED
                if ownership is Ownership.THIRD_PARTY
                else Sensitivity.PRIVATE
            ),
        )
        item = StateItem.create(
            layer=BrainLayer.EVENT,
            kind=evidence.source_type,
            summary=f"Source evidence: {evidence.source_type}",
            payload={
                "source_record_id": evidence.source_record_id,
                "content_hash": evidence.content_hash,
                "content": evidence.content,
                "metadata": evidence.metadata,
            },
            epistemic_status=EpistemicStatus.OBSERVED,
            status=StateStatus.ACTIVE,
            confidence=1.0,
            temporal=TemporalScope(
                observed_at=evidence.observed_at,
                recorded_at=evidence.observed_at,
                valid_from=evidence.observed_at,
            ),
            scope=ContextScope(
                relationship_id=evidence.metadata.get("relationship_id")
            ),
            sensitivity=sensitivity,
            ownership=ownership,
            provenance=(
                ProvenanceRef(
                    source_id=evidence.id,
                    source_type="digital_self.v1_evidence",
                    relation=ProvenanceRelation.MIGRATED_FROM,
                    observed_at=evidence.observed_at,
                    content_hash=evidence.content_hash,
                ),
            ),
            metadata={"legacy_evidence_id": evidence.id},
        )
        evidence_item_ids[evidence.id] = item.id
        items.append(item)

    claim_item_ids: dict[str, str] = {}
    pending = {claim.id: claim for claim in profile.claims}
    superseded_legacy_ids = {
        claim.supersedes for claim in profile.claims if claim.supersedes
    }
    while pending:
        progressed = False
        for claim_id, claim in sorted(
            pending.items(),
            key=lambda value: (value[1].created_at, value[0]),
        ):
            if claim.supersedes and claim.supersedes not in claim_item_ids:
                continue
            support_ids = tuple(
                sorted(
                    evidence_item_ids[evidence_id]
                    for evidence_id in claim.evidence_ids
                )
            )
            provenance = tuple(
                ProvenanceRef(
                    source_id=item_id,
                    source_type="digital_brain_state",
                    relation=ProvenanceRelation.SUPPORTS,
                    observed_at=claim.created_at,
                )
                for item_id in support_ids
            )
            if not provenance:
                provenance = (
                    ProvenanceRef(
                        source_id=claim.id,
                        source_type="digital_self.v1_claim",
                        relation=ProvenanceRelation.MIGRATED_FROM,
                        observed_at=claim.created_at,
                    ),
                )
            status = _claim_status(claim.status)
            if claim.id in superseded_legacy_ids:
                status = StateStatus.SUPERSEDED
            item = StateItem.create(
                layer=_claim_layer(claim.dimension),
                kind=claim.dimension,
                summary=claim.statement,
                payload={
                    "dimension": claim.dimension,
                    "legacy_metadata": claim.metadata,
                },
                epistemic_status=_epistemic_status(claim.provenance),
                status=status,
                confidence=claim.confidence,
                temporal=TemporalScope(
                    observed_at=claim.created_at,
                    recorded_at=claim.created_at,
                    valid_from=claim.valid_from or claim.created_at,
                    valid_to=claim.valid_to,
                    last_confirmed_at=(
                        claim.created_at
                        if claim.status is ClaimStatus.CONFIRMED
                        else None
                    ),
                ),
                scope=ContextScope(relationship_id=claim.relationship_id),
                sensitivity=_sensitivity(claim.sensitivity),
                ownership=Ownership.OWNER,
                provenance=provenance,
                support_ids=support_ids,
                supersedes=(
                    claim_item_ids[claim.supersedes] if claim.supersedes else None
                ),
                metadata={
                    "legacy_claim_id": claim.id,
                    "legacy_claim_status": claim.status.value,
                    "legacy_provenance": claim.provenance.value,
                },
            )
            claim_item_ids[claim.id] = item.id
            items.append(item)
            del pending[claim_id]
            progressed = True
        if not progressed:
            unresolved = sorted(pending)
            raise ValueError(
                "legacy claims contain missing or cyclic supersession: "
                f"{unresolved}"
            )

    for relationship in sorted(profile.relationships, key=lambda item: item.id):
        support_ids = tuple(
            sorted(
                {
                    *(
                        evidence_item_ids[evidence_id]
                        for evidence_id in relationship.evidence_ids
                    ),
                    *(
                        claim_item_ids[claim_id]
                        for claim_id in relationship.claim_ids
                    ),
                }
            )
        )
        provenance = tuple(
            ProvenanceRef(
                source_id=item_id,
                source_type="digital_brain_state",
                relation=ProvenanceRelation.DERIVED_FROM,
                observed_at=profile.updated_at,
            )
            for item_id in support_ids
        ) or (
            ProvenanceRef(
                source_id=relationship.id,
                source_type="digital_self.v1_relationship",
                relation=ProvenanceRelation.MIGRATED_FROM,
                observed_at=profile.updated_at,
            ),
        )
        items.append(
            StateItem.create(
                layer=BrainLayer.SOCIAL,
                kind="relationship.profile",
                summary=f"Relationship scope: {relationship.label}",
                payload={
                    "label": relationship.label,
                    "style_delta": relationship.style_delta,
                    "claim_ids": [
                        claim_item_ids[claim_id]
                        for claim_id in relationship.claim_ids
                    ],
                    "legacy_metadata": relationship.metadata,
                },
                epistemic_status=EpistemicStatus.OBSERVED,
                status=StateStatus.ACTIVE,
                confidence=1.0,
                temporal=TemporalScope(
                    observed_at=profile.created_at,
                    recorded_at=profile.updated_at,
                    valid_from=profile.created_at,
                ),
                scope=ContextScope(relationship_id=relationship.id),
                sensitivity=Sensitivity.RESTRICTED,
                ownership=Ownership.SHARED,
                provenance=provenance,
                support_ids=support_ids,
                metadata={"legacy_relationship_id": relationship.id},
            )
        )

    if profile.communication_style:
        communication_claims = tuple(
            sorted(
                item_id
                for claim_id, item_id in claim_item_ids.items()
                if _claim_layer(_claim_by_id(profile, claim_id).dimension)
                is BrainLayer.COMMUNICATION
            )
        )
        items.append(
            StateItem.create(
                layer=BrainLayer.COMMUNICATION,
                kind="communication.style",
                summary="Migrated communication-style model.",
                payload=profile.communication_style,
                epistemic_status=EpistemicStatus.INFERRED,
                status=StateStatus.ACTIVE,
                confidence=0.7,
                temporal=TemporalScope(
                    observed_at=profile.created_at,
                    recorded_at=profile.updated_at,
                    valid_from=profile.created_at,
                ),
                scope=ContextScope(),
                sensitivity=Sensitivity.PRIVATE,
                ownership=Ownership.OWNER,
                provenance=(
                    ProvenanceRef(
                        source_id=profile.profile_id,
                        source_type="digital_self.v1_profile",
                        relation=ProvenanceRelation.MIGRATED_FROM,
                        observed_at=profile.updated_at,
                    ),
                ),
                support_ids=communication_claims,
                metadata={"computed_from": "digital_self.v1.communication_style"},
            )
        )

    brain = DigitalBrain.create(
        owner_id=profile.owner_id,
        owner_name=profile.owner_name,
        created_at=profile.created_at,
        updated_at=profile.updated_at,
        items=items,
        metadata={
            "migrated_from": "digital_self.v1",
            "legacy_profile_id": profile.profile_id,
            "legacy_profile_version": profile.version,
            "source_summary": profile.source_summary,
            "legacy_metadata": profile.metadata,
        },
    )
    brain.version = max(1, profile.version)
    brain.validate()
    return brain


def _claim_by_id(profile: DigitalSelfProfile, claim_id: str) -> IdentityClaim:
    for claim in profile.claims:
        if claim.id == claim_id:
            return claim
    raise KeyError(claim_id)


def _claim_layer(dimension: str) -> BrainLayer:
    root = dimension.lower().replace("-", "_").split(".", 1)[0]
    if root in {"communication", "style", "language"}:
        return BrainLayer.COMMUNICATION
    if root in {"value", "values", "goal", "goals", "preference", "decision"}:
        return BrainLayer.VALUES_GOALS
    if root in {"affect", "emotion", "mood"}:
        return BrainLayer.AFFECT
    if root in {"relationship", "social", "audience", "role"}:
        return BrainLayer.SOCIAL
    if root in {"narrative", "life_period", "life_story", "theme"}:
        return BrainLayer.NARRATIVE
    if root in {"procedure", "procedural", "habit", "routine"}:
        return BrainLayer.PROCEDURAL
    if root in {"uncertainty", "conflict", "confidence"}:
        return BrainLayer.UNCERTAINTY
    if root in {"fact", "semantic", "knowledge"}:
        return BrainLayer.SEMANTIC
    return BrainLayer.SELF_SCHEMA


def _epistemic_status(provenance: ProvenanceType) -> EpistemicStatus:
    if provenance in {
        ProvenanceType.OWNER_INTERVIEW,
        ProvenanceType.MANUAL_CONFIRMATION,
    }:
        return EpistemicStatus.OWNER_DECLARED
    if provenance is ProvenanceType.BEHAVIORAL_INFERENCE:
        return EpistemicStatus.INFERRED
    return EpistemicStatus.OBSERVED


def _claim_status(status: ClaimStatus) -> StateStatus:
    return {
        ClaimStatus.CANDIDATE: StateStatus.PROPOSED,
        ClaimStatus.CONFIRMED: StateStatus.ACTIVE,
        ClaimStatus.REJECTED: StateStatus.REJECTED,
        ClaimStatus.SUPERSEDED: StateStatus.SUPERSEDED,
    }[status]


def _sensitivity(
    value: Any,
    *,
    default: Sensitivity = Sensitivity.PRIVATE,
) -> Sensitivity:
    if isinstance(value, Sensitivity):
        return value
    if isinstance(value, str):
        try:
            return Sensitivity(value)
        except ValueError:
            pass
    return default


__all__ = ["migrate_v1_profile"]
