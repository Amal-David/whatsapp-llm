"""Safe, explainable inspection views over digital-brain state."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from .models import (
    BrainLayer,
    DigitalBrain,
    Ownership,
    Sensitivity,
    StateItem,
    StateStatus,
)


def inspect_brain(
    brain: DigitalBrain,
    *,
    as_of: datetime,
    layer: BrainLayer | None = None,
    relationship_id: str | None = None,
    include_history: bool = False,
    include_payload: bool = False,
    include_sensitive: bool = False,
) -> dict[str, Any]:
    """Return deterministic state details with disclosure-safe defaults."""

    _require_aware(as_of)
    if relationship_id is not None and not relationship_id.strip():
        raise ValueError("relationship_id cannot be empty")
    brain.validate()

    items = _visible_items(
        brain,
        as_of=as_of,
        layer=layer,
        relationship_id=relationship_id,
        include_history=include_history,
    )
    return {
        "brain_id": brain.brain_id,
        "brain_version": brain.version,
        "as_of": as_of.isoformat(),
        "layer": layer.value if layer else None,
        "relationship_id": relationship_id,
        "include_history": include_history,
        "items": [
            _inspect_item(
                item,
                include_payload=include_payload,
                include_sensitive=include_sensitive,
            )
            for item in items
        ],
    }


def _visible_items(
    brain: DigitalBrain,
    *,
    as_of: datetime,
    layer: BrainLayer | None,
    relationship_id: str | None,
    include_history: bool,
) -> list[StateItem]:
    superseded_ids = {
        item.supersedes
        for item in brain.items
        if item.supersedes
        and item.status not in {StateStatus.REJECTED, StateStatus.DELETED}
        and item.temporal.contains(as_of)
    }

    def visible(item: StateItem) -> bool:
        if layer is not None and item.layer is not layer:
            return False
        if relationship_id is None:
            if item.scope.relationship_id is not None:
                return False
        elif item.scope.relationship_id not in {None, relationship_id}:
            return False
        if item.temporal.recorded_at > as_of:
            return False
        if include_history:
            return True
        return (
            item.id not in superseded_ids
            and item.status in {StateStatus.ACTIVE, StateStatus.PROPOSED}
            and item.temporal.contains(as_of)
        )

    return sorted(
        (item for item in brain.items if visible(item)),
        key=lambda item: (
            item.layer.value,
            item.kind,
            item.temporal.recorded_at,
            item.id,
        ),
    )


def _inspect_item(
    item: StateItem,
    *,
    include_payload: bool,
    include_sensitive: bool,
) -> dict[str, Any]:
    is_third_party = item.ownership is Ownership.THIRD_PARTY
    is_restricted = item.sensitivity in {
        Sensitivity.SENSITIVE,
        Sensitivity.RESTRICTED,
    }
    if is_third_party:
        summary = "Third-party context (redacted)"
    elif is_restricted and not include_sensitive:
        summary = "Sensitive state (redacted)"
    else:
        summary = item.summary

    result: dict[str, Any] = {
        "id": item.id,
        "layer": item.layer.value,
        "kind": item.kind,
        "summary": summary,
        "epistemic_status": item.epistemic_status.value,
        "status": item.status.value,
        "confidence": item.confidence,
        "temporal": item.temporal.to_dict(),
        "scope": item.scope.to_dict(),
        "sensitivity": item.sensitivity.value,
        "ownership": item.ownership.value,
        "provenance": [
            {
                "source_id": provenance.source_id,
                "source_type": provenance.source_type,
                "relation": provenance.relation.value,
                "observed_at": provenance.observed_at.isoformat(),
                "content_hash": provenance.content_hash,
            }
            for provenance in item.provenance
        ],
        "support_ids": list(item.support_ids),
        "contradiction_ids": list(item.contradiction_ids),
        "supersedes": item.supersedes,
    }
    if include_payload and not is_third_party and (include_sensitive or not is_restricted):
        result["payload"] = item.payload
    return result


def _require_aware(value: datetime) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("inspection as_of must be timezone-aware")
