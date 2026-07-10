"""Build a general digital-self profile from multi-chat behavioral evidence."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, cast

from ..ingest.style_analyzer import StyleAnalyzer, StyleMetrics
from ..ingest.whatsapp_parser import ChatMessage
from .models import (
    ClaimStatus,
    DigitalSelfProfile,
    EvidenceRecord,
    IdentityClaim,
    ProvenanceType,
    RelationshipProfile,
)
from .sources import NormalizedMessage


@dataclass
class DigitalSelfBuildResult:
    """Private build state plus the portable canonical profile."""

    profile: DigitalSelfProfile
    messages_by_split: dict[str, list[NormalizedMessage]]
    split_by_group: dict[str, str]
    split_by_message: dict[str, str]
    message_groups: dict[str, list[NormalizedMessage]]

    def split_for(self, message: NormalizedMessage) -> str:
        return self.split_by_message[message_key(message)]


def message_key(message: NormalizedMessage) -> str:
    """Create a source-safe key even when different chats reuse message IDs."""
    return f"{message.source}:{message.chat_id}:{message.source_message_id}"


def _profile_identifier(prefix: str, value: str) -> str:
    digest = hashlib.sha256(value.strip().lower().encode()).hexdigest()
    return f"{prefix}:{digest}"


def _parse_datetime(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _json_safe(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


class DigitalSelfBuilder:
    """Combine owner messages and explicit self-report without conflating them."""

    def __init__(self, include_third_party_context: bool = False):
        self.include_third_party_context = include_third_party_context
        self.style_analyzer = StyleAnalyzer()

    def build(
        self,
        messages: list[NormalizedMessage],
        *,
        owner_name: str,
        interview: dict[str, Any] | None = None,
    ) -> DigitalSelfBuildResult:
        if not owner_name.strip():
            raise ValueError("owner_name cannot be empty")
        ordered_messages = sorted(
            messages,
            key=lambda message: (
                message.timestamp,
                message.source,
                message.chat_id,
                message.source_message_id,
            ),
        )
        if not ordered_messages and not interview:
            raise ValueError("at least one message or interview answer is required")

        message_groups = self._group_messages(ordered_messages)
        split_by_group = self._assign_splits(message_groups)
        split_by_message = {
            message_key(message): split_by_group[group_id]
            for group_id, group in message_groups.items()
            for message in group
        }
        messages_by_split: dict[str, list[NormalizedMessage]] = {
            "train": [],
            "validation": [],
            "test": [],
        }
        for message in ordered_messages:
            messages_by_split[split_by_message[message_key(message)]].append(message)

        interview_completed_at = _parse_datetime(
            interview.get("completed_at") if interview else None
        )
        timestamps = [message.timestamp for message in ordered_messages]
        if interview_completed_at:
            timestamps.append(interview_completed_at)
        build_time = max(timestamps) if timestamps else datetime(1970, 1, 1, tzinfo=timezone.utc)

        evidence, evidence_by_message = self._message_evidence(
            ordered_messages,
            split_by_message,
        )
        interview_evidence, interview_claims = self._interview_claims(
            interview,
            build_time,
        )
        evidence.extend(interview_evidence)

        train_owner_messages = [
            message
            for message in messages_by_split["train"]
            if message.from_owner and message.text
        ]
        global_metrics = self._analyze(train_owner_messages)
        global_style = self._style_dict(global_metrics)
        training_evidence_ids = sorted(
            evidence_by_message[message_key(message)]
            for message in train_owner_messages
        )
        inferred_claims = self._behavioral_claims(
            global_metrics,
            evidence_ids=training_evidence_ids,
            created_at=build_time,
        )

        relationships, relationship_styles = self._relationship_profiles(
            ordered_messages,
            messages_by_split["train"],
            evidence_by_message,
            global_metrics,
        )
        temporal_styles = self._temporal_styles(train_owner_messages)
        communication_style = {
            "computed_from_split": "train",
            "global": global_style,
            "relationships": relationship_styles,
            "temporal": temporal_styles,
        }

        owner_id = _profile_identifier("owner", owner_name)
        profile = DigitalSelfProfile(
            profile_id=_profile_identifier("digital-self", owner_name),
            owner_id=owner_id,
            owner_name=owner_name.strip(),
            created_at=build_time,
            updated_at=build_time,
            evidence=sorted(evidence, key=lambda record: record.id),
            claims=sorted(inferred_claims + interview_claims, key=lambda claim: claim.id),
            relationships=relationships,
            communication_style=communication_style,
            source_summary=self._source_summary(ordered_messages, messages_by_split),
            metadata={
                "architecture": "profile_plus_time_aware_retrieval",
                "third_party_context_included": self.include_third_party_context,
                "behavioral_claims_are_candidates": True,
            },
        )
        profile.validate()
        return DigitalSelfBuildResult(
            profile=profile,
            messages_by_split=messages_by_split,
            split_by_group=split_by_group,
            split_by_message=split_by_message,
            message_groups=message_groups,
        )

    def _group_messages(
        self,
        messages: list[NormalizedMessage],
    ) -> dict[str, list[NormalizedMessage]]:
        groups: dict[str, list[NormalizedMessage]] = defaultdict(list)
        for message in messages:
            month = message.timestamp.astimezone(timezone.utc).strftime("%Y-%m")
            groups[f"{message.chat_id}:{month}"].append(message)
        return {
            group_id: sorted(
                group,
                key=lambda message: (message.timestamp, message.source_message_id),
            )
            for group_id, group in sorted(groups.items())
        }

    def _assign_splits(
        self,
        groups: dict[str, list[NormalizedMessage]],
    ) -> dict[str, str]:
        ordered_groups = sorted(
            groups,
            key=lambda group_id: (
                max(message.timestamp for message in groups[group_id]),
                group_id,
            ),
        )
        if len(ordered_groups) < 3:
            return {group_id: "train" for group_id in ordered_groups}

        holdout_count = min(
            len(ordered_groups) - 1,
            max(2, math.ceil(len(ordered_groups) * 0.2)),
        )
        validation_count = max(1, holdout_count // 2)
        train_count = len(ordered_groups) - holdout_count
        validation_end = train_count + validation_count
        return {
            group_id: (
                "train"
                if index < train_count
                else "validation"
                if index < validation_end
                else "test"
            )
            for index, group_id in enumerate(ordered_groups)
        }

    def _message_evidence(
        self,
        messages: list[NormalizedMessage],
        split_by_message: dict[str, str],
    ) -> tuple[list[EvidenceRecord], dict[str, str]]:
        evidence = []
        evidence_by_message = {}
        for message in messages:
            if not message.from_owner and not self.include_third_party_context:
                continue
            target = message.from_owner
            record = EvidenceRecord.create(
                source_type=(
                    f"{message.source}_owner_message"
                    if target
                    else f"{message.source}_third_party_context"
                ),
                source_record_id=message_key(message),
                observed_at=message.timestamp,
                content=message.text if not target else None,
                content_hash=message.content_hash,
                metadata={
                    "chat_id": message.chat_id,
                    "relationship_id": message.relationship_id,
                    "split": split_by_message[message_key(message)],
                    "identity_target": target,
                    "content_hash": message.content_hash,
                    "message_type": message.message_type,
                },
            )
            evidence.append(record)
            evidence_by_message[message_key(message)] = record.id
        return evidence, evidence_by_message

    def _interview_claims(
        self,
        interview: dict[str, Any] | None,
        default_time: datetime,
    ) -> tuple[list[EvidenceRecord], list[IdentityClaim]]:
        if not interview:
            return [], []
        if interview.get("schema_version") != "digital_self_interview.v1":
            raise ValueError("unsupported interview schema")

        completed_at = _parse_datetime(interview.get("completed_at")) or default_time
        records = []
        claims = []
        questions = []
        for section in interview.get("sections", []):
            for question in section.get("questions", []):
                questions.append((section.get("id", "unknown"), question))
        for section_id, question in sorted(questions, key=lambda item: item[1].get("id", "")):
            answer = question.get("answer")
            if answer is None or not str(answer).strip():
                continue
            statement = str(answer).strip()
            question_id = question.get("id")
            if not question_id:
                raise ValueError("answered interview questions require an id")
            evidence = EvidenceRecord.create(
                source_type="owner_interview",
                source_record_id=f"interview:{question_id}",
                observed_at=completed_at,
                content=statement,
                metadata={
                    "section_id": section_id,
                    "prompt": question.get("prompt"),
                    "sensitivity": question.get("sensitivity", "private"),
                },
            )
            records.append(evidence)
            claims.append(
                IdentityClaim.create(
                    dimension=question_id,
                    statement=statement,
                    status=ClaimStatus.CONFIRMED,
                    confidence=1.0,
                    provenance=ProvenanceType.OWNER_INTERVIEW,
                    created_at=completed_at,
                    evidence_ids=[evidence.id],
                    valid_from=_parse_datetime(question.get("valid_from")),
                    valid_to=_parse_datetime(question.get("valid_to")),
                    supersedes=question.get("supersedes"),
                    sensitivity=question.get("sensitivity", "private"),
                )
            )
        return records, claims

    def _analyze(self, messages: list[NormalizedMessage]) -> StyleMetrics:
        chat_messages = [
            ChatMessage(
                author="owner",
                message=message.text or "",
                timestamp=message.timestamp,
            )
            for message in messages
            if message.text
        ]
        return self.style_analyzer.analyze(chat_messages)

    def _style_dict(self, metrics: StyleMetrics) -> dict[str, Any]:
        raw = metrics.to_dict()
        raw.pop("common_phrases", None)
        return cast(dict[str, Any], _json_safe(raw))

    def _behavioral_claims(
        self,
        metrics: StyleMetrics,
        *,
        evidence_ids: list[str],
        created_at: datetime,
    ) -> list[IdentityClaim]:
        if not evidence_ids:
            return []
        confidence = min(0.95, 0.45 + math.log10(metrics.message_count + 1) * 0.2)
        length_label = (
            "brief"
            if metrics.avg_message_length < 45
            else "medium"
            if metrics.avg_message_length < 140
            else "long"
        )
        emoji_label = (
            "frequent"
            if metrics.emoji_usage_rate > 0.35
            else "occasional"
            if metrics.emoji_usage_rate > 0.08
            else "rare"
        )
        statements = {
            "communication.message_length": f"Observed messages are usually {length_label}.",
            "communication.emoji_use": f"Observed emoji use is {emoji_label}.",
            "communication.capitalization": (
                "Observed messages usually begin with capitalization."
                if metrics.capitalization_rate > 0.65
                else "Observed messages often begin without capitalization."
            ),
        }
        return [
            IdentityClaim.create(
                dimension=dimension,
                statement=statement,
                status=ClaimStatus.CANDIDATE,
                confidence=confidence,
                provenance=ProvenanceType.BEHAVIORAL_INFERENCE,
                created_at=created_at,
                evidence_ids=evidence_ids,
                metadata={"computed_from_split": "train"},
            )
            for dimension, statement in sorted(statements.items())
        ]

    def _relationship_profiles(
        self,
        all_messages: list[NormalizedMessage],
        train_messages: list[NormalizedMessage],
        evidence_by_message: dict[str, str],
        global_metrics: StyleMetrics,
    ) -> tuple[list[RelationshipProfile], dict[str, Any]]:
        relationship_ids = sorted({message.relationship_id for message in all_messages})
        profiles = []
        styles = {}
        for index, relationship_id in enumerate(relationship_ids, start=1):
            train_owner = [
                message
                for message in train_messages
                if message.from_owner
                and message.text
                and message.relationship_id == relationship_id
            ]
            all_owner = [
                message
                for message in all_messages
                if message.from_owner and message.relationship_id == relationship_id
            ]
            metrics = self._analyze(train_owner)
            evidence_ids = sorted(
                evidence_by_message[message_key(message)]
                for message in train_owner
            )
            delta = self._style_delta(metrics, global_metrics)
            profiles.append(
                RelationshipProfile(
                    id=relationship_id,
                    label=f"relationship {index}",
                    evidence_ids=evidence_ids,
                    style_delta=delta,
                    metadata={
                        "owner_message_count": len(all_owner),
                        "training_owner_message_count": len(train_owner),
                    },
                )
            )
            styles[relationship_id] = {
                "computed_from_split": "train",
                **self._style_dict(metrics),
                "delta_from_global": delta,
            }
        return profiles, styles

    def _style_delta(
        self,
        relationship: StyleMetrics,
        global_metrics: StyleMetrics,
    ) -> dict[str, float | None]:
        length_ratio = None
        if global_metrics.avg_message_length:
            length_ratio = relationship.avg_message_length / global_metrics.avg_message_length
        return {
            "avg_message_length_ratio": length_ratio,
            "emoji_usage_delta": (
                relationship.emoji_usage_rate - global_metrics.emoji_usage_rate
            ),
            "capitalization_delta": (
                relationship.capitalization_rate - global_metrics.capitalization_rate
            ),
            "avg_words_delta": (
                relationship.avg_words_per_message - global_metrics.avg_words_per_message
            ),
        }

    def _temporal_styles(
        self,
        train_owner_messages: list[NormalizedMessage],
    ) -> dict[str, Any]:
        periods: dict[str, list[NormalizedMessage]] = defaultdict(list)
        for message in train_owner_messages:
            quarter = (message.timestamp.month - 1) // 3 + 1
            periods[f"{message.timestamp.year}-Q{quarter}"].append(message)
        return {
            period: {
                "computed_from_split": "train",
                **self._style_dict(self._analyze(period_messages)),
            }
            for period, period_messages in sorted(periods.items())
        }

    def _source_summary(
        self,
        messages: list[NormalizedMessage],
        messages_by_split: dict[str, list[NormalizedMessage]],
    ) -> dict[str, Any]:
        owner_messages = [message for message in messages if message.from_owner]
        sources = Counter(message.source for message in messages)
        timestamps = [message.timestamp for message in messages]
        return {
            "chat_count": len({message.chat_id for message in messages}),
            "relationship_count": len({message.relationship_id for message in messages}),
            "message_count": len(messages),
            "owner_message_count": len(owner_messages),
            "messages_by_source": dict(sorted(sources.items())),
            "owner_messages_by_split": {
                split: sum(message.from_owner for message in split_messages)
                for split, split_messages in messages_by_split.items()
            },
            "date_range": {
                "from": min(timestamps).isoformat() if timestamps else None,
                "to": max(timestamps).isoformat() if timestamps else None,
            },
        }


__all__ = [
    "DigitalSelfBuildResult",
    "DigitalSelfBuilder",
    "message_key",
]
