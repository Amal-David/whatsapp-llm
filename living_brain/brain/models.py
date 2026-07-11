"""Multidimensional, provenance-bearing digital-brain state contracts."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class BrainLayer(str, Enum):
    EVENT = "event"
    EPISODE = "episode"
    SEMANTIC = "semantic"
    PROCEDURAL = "procedural"
    SELF_SCHEMA = "self_schema"
    VALUES_GOALS = "values_goals"
    AFFECT = "affect"
    SOCIAL = "social"
    NARRATIVE = "narrative"
    COMMUNICATION = "communication"
    UNCERTAINTY = "uncertainty"
    REFLECTION = "reflection"


class EpistemicStatus(str, Enum):
    OBSERVED = "observed"
    OWNER_DECLARED = "owner_declared"
    INFERRED = "inferred"
    GENERATED_PROPOSAL = "generated_proposal"
    ASSUMED = "assumed"


class StateStatus(str, Enum):
    ACTIVE = "active"
    PROPOSED = "proposed"
    SUPERSEDED = "superseded"
    REJECTED = "rejected"
    DELETED = "deleted"


class Sensitivity(str, Enum):
    PUBLIC = "public"
    PRIVATE = "private"
    SENSITIVE = "sensitive"
    RESTRICTED = "restricted"


class Ownership(str, Enum):
    OWNER = "owner"
    SHARED = "shared"
    THIRD_PARTY = "third_party"
    SYSTEM = "system"


class ProvenanceRelation(str, Enum):
    OBSERVES = "observes"
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    DERIVED_FROM = "derived_from"
    CORRECTS = "corrects"
    MIGRATED_FROM = "migrated_from"


PROTECTED_DIMENSIONS = {
    "age",
    "biometric_identity",
    "disability",
    "ethnicity",
    "gender_identity",
    "health",
    "nationality",
    "political_affiliation",
    "race",
    "religion",
    "sexual_orientation",
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


@dataclass(frozen=True)
class TemporalScope:
    """Observation, recording, validity, and confirmation time are distinct."""

    observed_at: datetime
    recorded_at: datetime
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    last_confirmed_at: datetime | None = None

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        for label, value in (
            ("observed_at", self.observed_at),
            ("recorded_at", self.recorded_at),
            ("valid_from", self.valid_from),
            ("valid_to", self.valid_to),
            ("last_confirmed_at", self.last_confirmed_at),
        ):
            if value is not None and not _is_aware(value):
                raise ValueError(f"temporal {label} must be timezone-aware")
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("temporal valid_from cannot be after valid_to")
        if self.last_confirmed_at and self.last_confirmed_at < self.observed_at:
            raise ValueError("last_confirmed_at cannot precede observed_at")

    def contains(self, point_in_time: datetime) -> bool:
        _require_aware(point_in_time, "query as_of")
        if self.valid_from and point_in_time < self.valid_from:
            return False
        return not self.valid_to or point_in_time <= self.valid_to

    def to_dict(self) -> dict[str, Any]:
        return {
            "observed_at": self.observed_at.isoformat(),
            "recorded_at": self.recorded_at.isoformat(),
            "valid_from": _datetime_json(self.valid_from),
            "valid_to": _datetime_json(self.valid_to),
            "last_confirmed_at": _datetime_json(self.last_confirmed_at),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TemporalScope:
        return cls(
            observed_at=datetime.fromisoformat(data["observed_at"]),
            recorded_at=datetime.fromisoformat(data["recorded_at"]),
            valid_from=_datetime_parse(data.get("valid_from")),
            valid_to=_datetime_parse(data.get("valid_to")),
            last_confirmed_at=_datetime_parse(data.get("last_confirmed_at")),
        )


@dataclass(frozen=True)
class ContextScope:
    """The contexts in which a state item is allowed to describe the owner."""

    relationship_id: str | None = None
    role: str | None = None
    audience: str | None = None
    channel: str | None = None
    situation_tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        normalized_tags = tuple(sorted(set(self.situation_tags)))
        object.__setattr__(self, "situation_tags", normalized_tags)
        for label, value in (
            ("relationship_id", self.relationship_id),
            ("role", self.role),
            ("audience", self.audience),
            ("channel", self.channel),
        ):
            if value is not None and not value.strip():
                raise ValueError(f"scope {label} cannot be empty")
        if any(not tag.strip() for tag in self.situation_tags):
            raise ValueError("scope situation tags cannot be empty")

    @property
    def is_global(self) -> bool:
        return not any(
            (
                self.relationship_id,
                self.role,
                self.audience,
                self.channel,
                self.situation_tags,
            )
        )

    def matches(
        self,
        *,
        relationship_id: str | None,
        role: str | None,
        audience: str | None,
        channel: str | None,
        situation_tags: tuple[str, ...],
    ) -> bool:
        if self.is_global:
            return True
        for expected, actual in (
            (self.relationship_id, relationship_id),
            (self.role, role),
            (self.audience, audience),
            (self.channel, channel),
        ):
            if expected is not None and expected != actual:
                return False
        return not self.situation_tags or set(self.situation_tags) <= set(situation_tags)

    def to_dict(self) -> dict[str, Any]:
        return {
            "relationship_id": self.relationship_id,
            "role": self.role,
            "audience": self.audience,
            "channel": self.channel,
            "situation_tags": list(self.situation_tags),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ContextScope:
        return cls(
            relationship_id=data.get("relationship_id"),
            role=data.get("role"),
            audience=data.get("audience"),
            channel=data.get("channel"),
            situation_tags=tuple(data.get("situation_tags", [])),
        )


@dataclass(frozen=True)
class ProvenanceRef:
    """One source or state reference explaining why an item exists."""

    source_id: str
    source_type: str
    relation: ProvenanceRelation
    observed_at: datetime
    content_hash: str | None = None
    note: str | None = None

    def __post_init__(self) -> None:
        if not self.source_id.strip() or not self.source_type.strip():
            raise ValueError("provenance source id and type are required")
        _require_aware(self.observed_at, "provenance observed_at")
        if self.content_hash is not None and not _is_sha256(self.content_hash):
            raise ValueError("provenance content_hash must be a lowercase SHA-256")
        if self.note is not None and not self.note.strip():
            raise ValueError("provenance note cannot be empty")

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "source_type": self.source_type,
            "relation": self.relation.value,
            "observed_at": self.observed_at.isoformat(),
            "content_hash": self.content_hash,
            "note": self.note,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ProvenanceRef:
        return cls(
            source_id=data["source_id"],
            source_type=data["source_type"],
            relation=ProvenanceRelation(data["relation"]),
            observed_at=datetime.fromisoformat(data["observed_at"]),
            content_hash=data.get("content_hash"),
            note=data.get("note"),
        )


@dataclass(frozen=True)
class StateItem:
    """One immutable version of state in a specific digital-brain layer."""

    id: str
    layer: BrainLayer
    kind: str
    summary: str
    payload: dict[str, Any]
    epistemic_status: EpistemicStatus
    status: StateStatus
    confidence: float
    temporal: TemporalScope
    scope: ContextScope
    sensitivity: Sensitivity
    ownership: Ownership
    provenance: tuple[ProvenanceRef, ...]
    support_ids: tuple[str, ...] = ()
    contradiction_ids: tuple[str, ...] = ()
    supersedes: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _json_copy(self.payload))
        object.__setattr__(self, "metadata", _json_copy(self.metadata))
        object.__setattr__(self, "provenance", tuple(self.provenance))
        object.__setattr__(self, "support_ids", tuple(sorted(set(self.support_ids))))
        object.__setattr__(
            self,
            "contradiction_ids",
            tuple(sorted(set(self.contradiction_ids))),
        )
        self.validate()

    @classmethod
    def create(
        cls,
        *,
        layer: BrainLayer,
        kind: str,
        summary: str,
        payload: dict[str, Any],
        epistemic_status: EpistemicStatus,
        confidence: float,
        temporal: TemporalScope,
        scope: ContextScope,
        sensitivity: Sensitivity,
        ownership: Ownership,
        provenance: tuple[ProvenanceRef, ...] | list[ProvenanceRef],
        status: StateStatus = StateStatus.ACTIVE,
        support_ids: tuple[str, ...] | list[str] = (),
        contradiction_ids: tuple[str, ...] | list[str] = (),
        supersedes: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> StateItem:
        values = {
            "layer": layer,
            "kind": kind.strip(),
            "summary": summary.strip(),
            "payload": _json_copy(payload),
            "epistemic_status": epistemic_status,
            "status": status,
            "confidence": float(confidence),
            "temporal": temporal,
            "scope": scope,
            "sensitivity": sensitivity,
            "ownership": ownership,
            "provenance": tuple(provenance),
            "support_ids": tuple(sorted(set(support_ids))),
            "contradiction_ids": tuple(sorted(set(contradiction_ids))),
            "supersedes": supersedes,
            "metadata": _json_copy(metadata or {}),
        }
        item_id = _canonical_hash("brain-item", cls._identity_payload(values))
        return cls(id=item_id, **values)

    def validate(self) -> None:
        if not self.id.startswith("brain-item:"):
            raise ValueError("state item id must start with brain-item:")
        if not self.kind.strip() or not self.summary.strip():
            raise ValueError("state item kind and summary are required")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("state item confidence must be between 0 and 1")
        if not self.provenance:
            raise ValueError("state item provenance cannot be empty")
        if set(self.support_ids) & set(self.contradiction_ids):
            raise ValueError("support and contradiction ids cannot overlap")
        for item_id in (*self.support_ids, *self.contradiction_ids):
            if not item_id.startswith("brain-item:"):
                raise ValueError("state links must reference brain-item ids")
        if self.supersedes is not None and not self.supersedes.startswith(
            "brain-item:"
        ):
            raise ValueError("supersedes must reference a brain-item id")
        if self.ownership is Ownership.THIRD_PARTY and self.layer not in {
            BrainLayer.EVENT,
            BrainLayer.SOCIAL,
        }:
            raise ValueError(
                "third-party state cannot be promoted into owner identity layers"
            )
        if (
            self.epistemic_status is EpistemicStatus.INFERRED
            and _contains_protected_dimension(self.kind, self.payload)
        ):
            raise ValueError("a protected trait cannot be created from inference")
        expected_id = _canonical_hash(
            "brain-item",
            self._identity_payload(self.__dict__),
        )
        if self.id != expected_id:
            raise ValueError("state item id does not match its canonical content")

    @staticmethod
    def _identity_payload(values: dict[str, Any]) -> dict[str, Any]:
        layer = values["layer"]
        epistemic_status = values["epistemic_status"]
        sensitivity = values["sensitivity"]
        ownership = values["ownership"]
        temporal = values["temporal"]
        scope = values["scope"]
        provenance = values["provenance"]
        return {
            "layer": layer.value if isinstance(layer, BrainLayer) else layer,
            "kind": values["kind"],
            "summary": values["summary"],
            "payload": values["payload"],
            "epistemic_status": (
                epistemic_status.value
                if isinstance(epistemic_status, EpistemicStatus)
                else epistemic_status
            ),
            "confidence": values["confidence"],
            "temporal": (
                temporal.to_dict() if isinstance(temporal, TemporalScope) else temporal
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
            "provenance": [
                item.to_dict() if isinstance(item, ProvenanceRef) else item
                for item in provenance
            ],
            "support_ids": sorted(values.get("support_ids", ())),
            "contradiction_ids": sorted(values.get("contradiction_ids", ())),
            "supersedes": values.get("supersedes"),
            "metadata": values.get("metadata", {}),
        }

    def is_visible_at(self, point_in_time: datetime, *, include_proposed: bool) -> bool:
        if self.status in {StateStatus.REJECTED, StateStatus.DELETED}:
            return False
        if self.status is StateStatus.PROPOSED and not include_proposed:
            return False
        return self.temporal.contains(point_in_time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "layer": self.layer.value,
            "kind": self.kind,
            "summary": self.summary,
            "payload": self.payload,
            "epistemic_status": self.epistemic_status.value,
            "status": self.status.value,
            "confidence": self.confidence,
            "temporal": self.temporal.to_dict(),
            "scope": self.scope.to_dict(),
            "sensitivity": self.sensitivity.value,
            "ownership": self.ownership.value,
            "provenance": [item.to_dict() for item in self.provenance],
            "support_ids": list(self.support_ids),
            "contradiction_ids": list(self.contradiction_ids),
            "supersedes": self.supersedes,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StateItem:
        return cls(
            id=data["id"],
            layer=BrainLayer(data["layer"]),
            kind=data["kind"],
            summary=data["summary"],
            payload=dict(data.get("payload", {})),
            epistemic_status=EpistemicStatus(data["epistemic_status"]),
            status=StateStatus(data["status"]),
            confidence=float(data["confidence"]),
            temporal=TemporalScope.from_dict(data["temporal"]),
            scope=ContextScope.from_dict(data.get("scope", {})),
            sensitivity=Sensitivity(data["sensitivity"]),
            ownership=Ownership(data["ownership"]),
            provenance=tuple(
                ProvenanceRef.from_dict(item) for item in data["provenance"]
            ),
            support_ids=tuple(data.get("support_ids", [])),
            contradiction_ids=tuple(data.get("contradiction_ids", [])),
            supersedes=data.get("supersedes"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(eq=True)
class DigitalBrain:
    """Canonical owner-governed state; model providers remain replaceable."""

    brain_id: str
    owner_id: str
    owner_name: str
    schema_version: str = "digital_brain.v2"
    version: int = 1
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    items: list[StateItem] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        owner_id: str,
        owner_name: str,
        created_at: datetime,
        items: list[StateItem] | None = None,
        updated_at: datetime | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> DigitalBrain:
        _require_aware(created_at, "brain created_at")
        resolved_updated_at = updated_at or created_at
        brain = cls(
            brain_id=_canonical_hash(
                "brain",
                {
                    "owner_id": owner_id,
                    "owner_name": owner_name.strip(),
                    "created_at": created_at.isoformat(),
                },
            ),
            owner_id=owner_id,
            owner_name=owner_name.strip(),
            created_at=created_at,
            updated_at=resolved_updated_at,
            items=sorted(list(items or []), key=lambda item: item.id),
            metadata=_json_copy(metadata or {}),
        )
        brain.validate()
        return brain

    def validate(self) -> None:
        if self.schema_version != "digital_brain.v2":
            raise ValueError(f"unsupported brain schema: {self.schema_version}")
        if self.version < 1:
            raise ValueError("brain version must be positive")
        if not self.brain_id.startswith("brain:"):
            raise ValueError("brain id must start with brain:")
        if not self.owner_id.strip() or not self.owner_name.strip():
            raise ValueError("brain owner id and name are required")
        _require_aware(self.created_at, "brain created_at")
        _require_aware(self.updated_at, "brain updated_at")
        if self.updated_at < self.created_at:
            raise ValueError("brain updated_at cannot precede created_at")

        by_id: dict[str, StateItem] = {}
        for item in self.items:
            item.validate()
            if item.id in by_id:
                raise ValueError(f"duplicate state item id: {item.id}")
            by_id[item.id] = item
        known_ids = set(by_id)
        for item in self.items:
            unknown_links = (
                set(item.support_ids)
                | set(item.contradiction_ids)
                | ({item.supersedes} if item.supersedes else set())
            ) - known_ids
            if unknown_links:
                raise ValueError(
                    f"state item {item.id} references unknown items: "
                    f"{sorted(unknown_links)}"
                )
            if item.supersedes:
                target = by_id[item.supersedes]
                if target.layer is not item.layer:
                    raise ValueError("supersession cannot cross brain layers")
                if target.status is not StateStatus.SUPERSEDED:
                    raise ValueError("a superseded target must have superseded status")
        self._validate_supersession_cycles(by_id)

    def _validate_supersession_cycles(self, by_id: dict[str, StateItem]) -> None:
        for start in by_id:
            seen = set()
            current = start
            while by_id[current].supersedes:
                if current in seen:
                    raise ValueError("supersession cycle detected")
                seen.add(current)
                current = by_id[current].supersedes or ""

    def item_by_id(self, item_id: str) -> StateItem:
        for item in self.items:
            if item.id == item_id:
                return item
        raise KeyError(item_id)

    def query(
        self,
        *,
        as_of: datetime | None = None,
        layer: BrainLayer | None = None,
        relationship_id: str | None = None,
        role: str | None = None,
        audience: str | None = None,
        channel: str | None = None,
        situation_tags: tuple[str, ...] = (),
        include_proposed: bool = False,
    ) -> list[StateItem]:
        point_in_time = as_of or utc_now()
        _require_aware(point_in_time, "query as_of")
        superseded_at_point = {
            item.supersedes
            for item in self.items
            if item.supersedes
            and item.status not in {StateStatus.REJECTED, StateStatus.DELETED}
            and item.temporal.contains(point_in_time)
        }
        return sorted(
            (
                item
                for item in self.items
                if (layer is None or item.layer is layer)
                and item.id not in superseded_at_point
                and item.is_visible_at(
                    point_in_time,
                    include_proposed=include_proposed,
                )
                and item.scope.matches(
                    relationship_id=relationship_id,
                    role=role,
                    audience=audience,
                    channel=channel,
                    situation_tags=tuple(situation_tags),
                )
            ),
            key=lambda item: (item.layer.value, item.kind, item.id),
        )

    def add_item(self, item: StateItem, *, updated_at: datetime) -> None:
        _require_aware(updated_at, "brain update time")
        if updated_at < self.updated_at:
            raise ValueError("brain updates cannot move backward in time")
        candidate = replace(
            self,
            items=sorted([*self.items, item], key=lambda value: value.id),
            version=self.version + 1,
            updated_at=updated_at,
        )
        candidate.validate()
        self.items = candidate.items
        self.version = candidate.version
        self.updated_at = candidate.updated_at

    def apply_owner_correction(
        self,
        item_id: str,
        *,
        summary: str,
        payload: dict[str, Any],
        corrected_at: datetime,
        reason: str,
    ) -> StateItem:
        _require_aware(corrected_at, "correction time")
        if not reason.strip():
            raise ValueError("correction reason is required")
        original = self.item_by_id(item_id)
        if original.status not in {StateStatus.ACTIVE, StateStatus.PROPOSED}:
            raise ValueError("only active or proposed state can be corrected")
        valid_from = original.temporal.valid_from or original.temporal.observed_at
        if corrected_at <= valid_from:
            raise ValueError("correction must follow the original valid time")
        if original.temporal.valid_to and corrected_at >= original.temporal.valid_to:
            raise ValueError("correction must occur while the original item is valid")

        superseded = replace(
            original,
            status=StateStatus.SUPERSEDED,
        )
        correction = StateItem.create(
            layer=original.layer,
            kind=original.kind,
            summary=summary,
            payload=payload,
            epistemic_status=EpistemicStatus.OWNER_DECLARED,
            status=StateStatus.ACTIVE,
            confidence=1.0,
            temporal=TemporalScope(
                observed_at=corrected_at,
                recorded_at=corrected_at,
                valid_from=corrected_at,
                last_confirmed_at=corrected_at,
            ),
            scope=original.scope,
            sensitivity=original.sensitivity,
            ownership=Ownership.OWNER,
            provenance=(
                ProvenanceRef(
                    source_id=original.id,
                    source_type="digital_brain_state",
                    relation=ProvenanceRelation.CORRECTS,
                    observed_at=corrected_at,
                    note=reason.strip(),
                ),
            ),
            support_ids=(original.id,),
            supersedes=original.id,
            metadata={
                **original.metadata,
                "correction_reason": reason.strip(),
                "corrected_item_id": original.id,
            },
        )
        candidate_items = [
            superseded if item.id == item_id else item for item in self.items
        ]
        candidate = replace(
            self,
            items=sorted([*candidate_items, correction], key=lambda item: item.id),
            version=self.version + 1,
            updated_at=max(self.updated_at, corrected_at),
        )
        candidate.validate()
        self.items = candidate.items
        self.version = candidate.version
        self.updated_at = candidate.updated_at
        return correction

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "brain_id": self.brain_id,
            "owner_id": self.owner_id,
            "owner_name": self.owner_name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "items": [item.to_dict() for item in sorted(self.items, key=lambda x: x.id)],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        self.validate()
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DigitalBrain:
        brain = cls(
            brain_id=data["brain_id"],
            owner_id=data["owner_id"],
            owner_name=data["owner_name"],
            schema_version=data.get("schema_version", ""),
            version=int(data.get("version", 1)),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            items=[StateItem.from_dict(item) for item in data.get("items", [])],
            metadata=dict(data.get("metadata", {})),
        )
        brain.items = sorted(brain.items, key=lambda item: item.id)
        brain.validate()
        return brain

    @classmethod
    def from_json(cls, content: str) -> DigitalBrain:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("digital brain must be a JSON object")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output.parent,
                prefix=f".{output.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_path = Path(temporary.name)
                temporary.write(self.to_json())
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_path, output)
            output.chmod(0o600)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> DigitalBrain:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


def _contains_protected_dimension(kind: str, payload: dict[str, Any]) -> bool:
    values = [kind, str(payload.get("dimension", ""))]
    for value in values:
        normalized = value.lower().replace("-", "_")
        path_segments = set(normalized.split("."))
        if normalized.startswith("protected.") or bool(
            path_segments & PROTECTED_DIMENSIONS
        ):
            return True
    return False


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


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _is_aware(value: datetime) -> bool:
    return value.tzinfo is not None and value.utcoffset() is not None


def _require_aware(value: datetime, label: str) -> None:
    if not _is_aware(value):
        raise ValueError(f"{label} must be timezone-aware")


def _datetime_json(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _datetime_parse(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


__all__ = [
    "BrainLayer",
    "ContextScope",
    "DigitalBrain",
    "EpistemicStatus",
    "Ownership",
    "ProvenanceRef",
    "ProvenanceRelation",
    "Sensitivity",
    "StateItem",
    "StateStatus",
    "TemporalScope",
]
