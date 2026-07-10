"""Versioned, evidence-grounded contracts for a general digital self."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


class ClaimStatus(str, Enum):
    """Lifecycle state for an identity claim."""

    CANDIDATE = "candidate"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    SUPERSEDED = "superseded"


class ProvenanceType(str, Enum):
    """How a claim entered the self model."""

    OWNER_INTERVIEW = "owner_interview"
    MANUAL_CONFIRMATION = "manual_confirmation"
    OWNER_MESSAGE = "owner_message"
    IMPORTED_PROFILE = "imported_profile"
    BEHAVIORAL_INFERENCE = "behavioral_inference"


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

INFERRED_PROVENANCE = {ProvenanceType.BEHAVIORAL_INFERENCE}

PROVENANCE_PRIORITY = {
    ProvenanceType.MANUAL_CONFIRMATION: 5,
    ProvenanceType.OWNER_INTERVIEW: 4,
    ProvenanceType.OWNER_MESSAGE: 3,
    ProvenanceType.IMPORTED_PROFILE: 2,
    ProvenanceType.BEHAVIORAL_INFERENCE: 1,
}


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _datetime_to_json(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def _datetime_from_json(value: str | None) -> datetime | None:
    return datetime.fromisoformat(value) if value else None


def _canonical_hash(prefix: str, payload: dict[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _is_protected_dimension(dimension: str) -> bool:
    normalized = dimension.lower().replace("-", "_")
    parts = normalized.split(".")
    if parts[0] == "protected":
        return True
    return any(part in PROTECTED_DIMENSIONS for part in parts)


@dataclass(eq=True)
class EvidenceRecord:
    """One immutable source record supporting zero or more identity claims."""

    id: str
    source_type: str
    source_record_id: str
    observed_at: datetime
    content_hash: str
    content: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        source_type: str,
        source_record_id: str,
        observed_at: datetime,
        content: str | None = None,
        content_hash: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> EvidenceRecord:
        resolved_content_hash = content_hash or hashlib.sha256(
            (content or "").encode("utf-8")
        ).hexdigest()
        if len(resolved_content_hash) != 64 or any(
            character not in "0123456789abcdef" for character in resolved_content_hash
        ):
            raise ValueError("evidence content_hash must be a lowercase SHA-256 digest")
        record_id = _canonical_hash(
            "evidence",
            {
                "source_type": source_type,
                "source_record_id": source_record_id,
                "observed_at": observed_at.isoformat(),
                "content_hash": resolved_content_hash,
            },
        )
        return cls(
            id=record_id,
            source_type=source_type,
            source_record_id=source_record_id,
            observed_at=observed_at,
            content_hash=resolved_content_hash,
            content=content,
            metadata=metadata or {},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "source_type": self.source_type,
            "source_record_id": self.source_record_id,
            "observed_at": self.observed_at.isoformat(),
            "content_hash": self.content_hash,
            "content": self.content,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EvidenceRecord:
        return cls(
            id=data["id"],
            source_type=data["source_type"],
            source_record_id=data["source_record_id"],
            observed_at=datetime.fromisoformat(data["observed_at"]),
            content_hash=data["content_hash"],
            content=data.get("content"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(eq=True)
class IdentityClaim:
    """A versioned statement about the owner with explicit provenance."""

    id: str
    dimension: str
    statement: str
    status: ClaimStatus
    confidence: float
    provenance: ProvenanceType
    created_at: datetime
    evidence_ids: list[str] = field(default_factory=list)
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    supersedes: str | None = None
    sensitivity: str = "private"
    relationship_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        dimension: str,
        statement: str,
        status: ClaimStatus,
        confidence: float,
        provenance: ProvenanceType,
        created_at: datetime,
        evidence_ids: list[str] | None = None,
        valid_from: datetime | None = None,
        valid_to: datetime | None = None,
        supersedes: str | None = None,
        sensitivity: str = "private",
        relationship_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> IdentityClaim:
        claim = cls(
            id="",
            dimension=dimension,
            statement=statement,
            status=status,
            confidence=confidence,
            provenance=provenance,
            created_at=created_at,
            evidence_ids=sorted(set(evidence_ids or [])),
            valid_from=valid_from,
            valid_to=valid_to,
            supersedes=supersedes,
            sensitivity=sensitivity,
            relationship_id=relationship_id,
            metadata=metadata or {},
        )
        claim.validate()
        claim.id = _canonical_hash(
            "claim",
            {
                "dimension": claim.dimension,
                "statement": claim.statement,
                "provenance": claim.provenance.value,
                "created_at": claim.created_at.isoformat(),
                "evidence_ids": claim.evidence_ids,
                "valid_from": _datetime_to_json(claim.valid_from),
                "valid_to": _datetime_to_json(claim.valid_to),
                "relationship_id": claim.relationship_id,
            },
        )
        return claim

    def validate(self) -> None:
        if not self.dimension.strip():
            raise ValueError("claim dimension cannot be empty")
        if not self.statement.strip():
            raise ValueError("claim statement cannot be empty")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("claim confidence must be between 0 and 1")
        if self.valid_from and self.valid_to and self.valid_from > self.valid_to:
            raise ValueError("claim valid_from cannot be after valid_to")
        if self.provenance in INFERRED_PROVENANCE and _is_protected_dimension(
            self.dimension
        ):
            raise ValueError("a protected trait cannot be created from inference")

    def is_current(self, as_of: datetime) -> bool:
        if self.status in {ClaimStatus.REJECTED, ClaimStatus.SUPERSEDED}:
            return False
        if self.valid_from and as_of < self.valid_from:
            return False
        return not self.valid_to or as_of <= self.valid_to

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "dimension": self.dimension,
            "statement": self.statement,
            "status": self.status.value,
            "confidence": self.confidence,
            "provenance": self.provenance.value,
            "created_at": self.created_at.isoformat(),
            "evidence_ids": self.evidence_ids,
            "valid_from": _datetime_to_json(self.valid_from),
            "valid_to": _datetime_to_json(self.valid_to),
            "supersedes": self.supersedes,
            "sensitivity": self.sensitivity,
            "relationship_id": self.relationship_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IdentityClaim:
        claim = cls(
            id=data["id"],
            dimension=data["dimension"],
            statement=data["statement"],
            status=ClaimStatus(data["status"]),
            confidence=float(data["confidence"]),
            provenance=ProvenanceType(data["provenance"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            evidence_ids=list(data.get("evidence_ids", [])),
            valid_from=_datetime_from_json(data.get("valid_from")),
            valid_to=_datetime_from_json(data.get("valid_to")),
            supersedes=data.get("supersedes"),
            sensitivity=data.get("sensitivity", "private"),
            relationship_id=data.get("relationship_id"),
            metadata=dict(data.get("metadata", {})),
        )
        claim.validate()
        return claim


@dataclass(eq=True)
class RelationshipProfile:
    """Relationship-scoped behavior without exposing the contact's identity."""

    id: str
    label: str
    evidence_ids: list[str] = field(default_factory=list)
    claim_ids: list[str] = field(default_factory=list)
    style_delta: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "evidence_ids": self.evidence_ids,
            "claim_ids": self.claim_ids,
            "style_delta": self.style_delta,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> RelationshipProfile:
        return cls(
            id=data["id"],
            label=data["label"],
            evidence_ids=list(data.get("evidence_ids", [])),
            claim_ids=list(data.get("claim_ids", [])),
            style_delta=dict(data.get("style_delta", {})),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass(eq=True)
class DigitalSelfProfile:
    """Canonical, portable state for one general digital-self simulation."""

    profile_id: str
    owner_id: str
    owner_name: str
    schema_version: str = "digital_self.v1"
    version: int = 1
    created_at: datetime = field(default_factory=utc_now)
    updated_at: datetime = field(default_factory=utc_now)
    evidence: list[EvidenceRecord] = field(default_factory=list)
    claims: list[IdentityClaim] = field(default_factory=list)
    relationships: list[RelationshipProfile] = field(default_factory=list)
    communication_style: dict[str, Any] = field(default_factory=dict)
    source_summary: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != "digital_self.v1":
            raise ValueError(f"unsupported profile schema: {self.schema_version}")
        if self.version < 1:
            raise ValueError("profile version must be positive")

        evidence_ids = [record.id for record in self.evidence]
        claim_ids = [claim.id for claim in self.claims]
        relationship_ids = [relationship.id for relationship in self.relationships]
        for label, identifiers in (
            ("evidence", evidence_ids),
            ("claim", claim_ids),
            ("relationship", relationship_ids),
        ):
            if len(identifiers) != len(set(identifiers)):
                raise ValueError(f"duplicate {label} id")

        known_evidence = set(evidence_ids)
        known_claims = set(claim_ids)
        for claim in self.claims:
            claim.validate()
            missing = set(claim.evidence_ids) - known_evidence
            if missing:
                raise ValueError(f"claim {claim.id} references unknown evidence: {sorted(missing)}")
            if claim.supersedes and claim.supersedes not in known_claims:
                raise ValueError(f"claim {claim.id} supersedes an unknown claim")

        for relationship in self.relationships:
            if set(relationship.evidence_ids) - known_evidence:
                raise ValueError(f"relationship {relationship.id} references unknown evidence")
            if set(relationship.claim_ids) - known_claims:
                raise ValueError(f"relationship {relationship.id} references unknown claims")

    def current_claims(self, as_of: datetime | None = None) -> list[IdentityClaim]:
        point_in_time = as_of or utc_now()
        active = [claim for claim in self.claims if claim.is_current(point_in_time)]
        superseded_ids = {claim.supersedes for claim in active if claim.supersedes}
        return [claim for claim in active if claim.id not in superseded_ids]

    def resolve_dimension(
        self,
        dimension: str,
        as_of: datetime | None = None,
    ) -> IdentityClaim | None:
        matches = [
            claim
            for claim in self.current_claims(as_of)
            if claim.dimension == dimension
        ]
        if not matches:
            return None
        return max(
            matches,
            key=lambda claim: (
                1 if claim.status is ClaimStatus.CONFIRMED else 0,
                PROVENANCE_PRIORITY[claim.provenance],
                claim.confidence,
                claim.created_at,
                claim.id,
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "owner_id": self.owner_id,
            "owner_name": self.owner_name,
            "version": self.version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "evidence": [record.to_dict() for record in self.evidence],
            "claims": [claim.to_dict() for claim in self.claims],
            "relationships": [relationship.to_dict() for relationship in self.relationships],
            "communication_style": self.communication_style,
            "source_summary": self.source_summary,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DigitalSelfProfile:
        profile = cls(
            profile_id=data["profile_id"],
            owner_id=data["owner_id"],
            owner_name=data["owner_name"],
            schema_version=data.get("schema_version", ""),
            version=int(data.get("version", 1)),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
            evidence=[EvidenceRecord.from_dict(item) for item in data.get("evidence", [])],
            claims=[IdentityClaim.from_dict(item) for item in data.get("claims", [])],
            relationships=[
                RelationshipProfile.from_dict(item)
                for item in data.get("relationships", [])
            ],
            communication_style=dict(data.get("communication_style", {})),
            source_summary=dict(data.get("source_summary", {})),
            metadata=dict(data.get("metadata", {})),
        )
        profile.validate()
        return profile

    def to_json(self) -> str:
        self.validate()
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    @classmethod
    def from_json(cls, content: str) -> DigitalSelfProfile:
        data = json.loads(content)
        if not isinstance(data, dict):
            raise ValueError("digital-self profile must be a JSON object")
        return cls.from_dict(data)

    def save(self, path: str | Path) -> None:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                prefix=f".{output_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary_file:
                temporary_path = Path(temporary_file.name)
                temporary_file.write(self.to_json())
                temporary_file.flush()
                os.fsync(temporary_file.fileno())
            os.replace(temporary_path, output_path)
            output_path.chmod(0o600)
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            raise

    @classmethod
    def load(cls, path: str | Path) -> DigitalSelfProfile:
        return cls.from_json(Path(path).read_text(encoding="utf-8"))


__all__ = [
    "ClaimStatus",
    "DigitalSelfProfile",
    "EvidenceRecord",
    "IdentityClaim",
    "ProvenanceType",
    "RelationshipProfile",
]
