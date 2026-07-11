"""Deterministic, privacy-aware evaluation artifacts for a digital self."""

from __future__ import annotations

import hashlib
import itertools
import json
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .builder import DigitalSelfBuildResult, message_key

REQUIRED_EVALUATION_TAGS = (
    "latest_preference",
    "supersession",
    "contradiction",
    "grounding",
    "abstention",
    "relationship_leakage",
)


def _stable_id(prefix: str, payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        dict(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return f"{prefix}:{hashlib.sha256(encoded).hexdigest()}"


def _parse_datetime(value: str | None, fallback: datetime) -> datetime:
    if not value:
        return fallback
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _write_private_json(path: str | Path, content: str) -> None:
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
            temporary_file.write(content)
            temporary_file.flush()
            os.fsync(temporary_file.fileno())
        os.replace(temporary_path, output_path)
        output_path.chmod(0o600)
    except Exception:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
        raise


@dataclass(frozen=True)
class EvaluationConfiguration:
    """One inference configuration evaluated against stable rows."""

    id: str
    label: str
    uses_profile: bool
    uses_retrieval: bool
    adapter_name: str | None = None

    def __post_init__(self) -> None:
        if not self.id.strip() or not self.label.strip():
            raise ValueError("evaluation configuration id and label are required")
        if self.uses_retrieval and not self.uses_profile:
            raise ValueError("retrieval configuration must also use the profile")

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "uses_profile": self.uses_profile,
            "uses_retrieval": self.uses_retrieval,
            "adapter_name": self.adapter_name,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationConfiguration:
        return cls(
            id=str(data["id"]),
            label=str(data["label"]),
            uses_profile=bool(data["uses_profile"]),
            uses_retrieval=bool(data["uses_retrieval"]),
            adapter_name=(
                str(data["adapter_name"])
                if data.get("adapter_name") is not None
                else None
            ),
        )


DEFAULT_CONFIGURATIONS = (
    EvaluationConfiguration(
        id="generic",
        label="Generic baseline",
        uses_profile=False,
        uses_retrieval=False,
    ),
    EvaluationConfiguration(
        id="profile_only",
        label="Profile only",
        uses_profile=True,
        uses_retrieval=False,
    ),
    EvaluationConfiguration(
        id="profile_retrieval",
        label="Profile plus retrieval",
        uses_profile=True,
        uses_retrieval=True,
    ),
)


@dataclass(frozen=True)
class EvaluationRow:
    """A private prompt/reference pair with configuration-independent identity."""

    id: str
    origin: str
    source_ref: str
    source_group_id: str
    prompt: str
    reference_response: str
    as_of: datetime
    tags: tuple[str, ...] = ()
    split: str | None = None
    relationship_id: str | None = None

    def to_dict(self, *, include_private: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "id": self.id,
            "origin": self.origin,
            "source_ref": self.source_ref,
            "source_group_id": self.source_group_id,
            "as_of": self.as_of.isoformat(),
            "tags": list(self.tags),
            "split": self.split,
            "relationship_id": self.relationship_id,
        }
        if include_private:
            data["prompt"] = self.prompt
            data["reference_response"] = self.reference_response
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationRow:
        return cls(
            id=str(data["id"]),
            origin=str(data["origin"]),
            source_ref=str(data["source_ref"]),
            source_group_id=str(data["source_group_id"]),
            prompt=str(data["prompt"]),
            reference_response=str(data["reference_response"]),
            as_of=datetime.fromisoformat(str(data["as_of"])),
            tags=tuple(str(tag) for tag in data.get("tags", [])),
            split=str(data["split"]) if data.get("split") is not None else None,
            relationship_id=(
                str(data["relationship_id"])
                if data.get("relationship_id") is not None
                else None
            ),
        )


@dataclass
class EvaluationSuite:
    """Private evaluation rows plus public, text-free coverage metadata."""

    profile_id: str
    created_at: datetime
    rows: list[EvaluationRow]
    configurations: tuple[EvaluationConfiguration, ...] = DEFAULT_CONFIGURATIONS
    schema_version: str = "digital_self_evaluation.v1"
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if self.schema_version != "digital_self_evaluation.v1":
            raise ValueError(f"unsupported evaluation schema: {self.schema_version}")
        row_ids = [row.id for row in self.rows]
        configuration_ids = [configuration.id for configuration in self.configurations]
        if len(row_ids) != len(set(row_ids)):
            raise ValueError("duplicate evaluation row id")
        if len(configuration_ids) != len(set(configuration_ids)):
            raise ValueError("duplicate evaluation configuration id")
        for row in self.rows:
            if row.origin == "held_out_reply" and row.split not in {
                "validation",
                "test",
            }:
                raise ValueError("chat evaluation rows must come from a held-out split")
            if row.origin not in {"held_out_reply", "interview_retest"}:
                raise ValueError(f"unsupported evaluation row origin: {row.origin}")
            if not row.prompt.strip() or not row.reference_response.strip():
                raise ValueError("evaluation prompts and references cannot be empty")

    def coverage(self) -> list[str]:
        present = {tag for row in self.rows for tag in row.tags}
        return [tag for tag in REQUIRED_EVALUATION_TAGS if tag in present]

    def summary(self) -> dict[str, Any]:
        """Return aggregate metadata that never includes prompt or response text."""
        origins = Counter(row.origin for row in self.rows)
        splits = Counter(row.split for row in self.rows if row.split)
        tags = Counter(tag for row in self.rows for tag in row.tags)
        present = set(tags)
        return {
            "schema_version": "digital_self_evaluation_summary.v1",
            "profile_id": self.profile_id,
            "row_count": len(self.rows),
            "configuration_count": len(self.configurations),
            "origin_counts": dict(sorted(origins.items())),
            "split_counts": dict(sorted(splits.items())),
            "tag_counts": dict(sorted(tags.items())),
            "missing_required_tags": [
                tag for tag in REQUIRED_EVALUATION_TAGS if tag not in present
            ],
        }

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {
            "schema_version": self.schema_version,
            "profile_id": self.profile_id,
            "created_at": self.created_at.isoformat(),
            "configurations": [
                configuration.to_dict() for configuration in self.configurations
            ],
            "rows": [row.to_dict(include_private=True) for row in self.rows],
            "metadata": self.metadata,
        }

    def to_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"

    def save(self, path: str | Path) -> None:
        """Persist the private suite with owner-only filesystem permissions."""
        _write_private_json(path, self.to_json())

    def save_summary(self, path: str | Path) -> None:
        """Persist aggregate coverage without prompt or response text."""
        self.validate()
        content = json.dumps(
            self.summary(),
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        ) + "\n"
        _write_private_json(path, content)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> EvaluationSuite:
        suite = cls(
            schema_version=str(data.get("schema_version", "")),
            profile_id=str(data["profile_id"]),
            created_at=datetime.fromisoformat(str(data["created_at"])),
            configurations=tuple(
                EvaluationConfiguration.from_dict(item)
                for item in data.get("configurations", [])
            ),
            rows=[EvaluationRow.from_dict(item) for item in data.get("rows", [])],
            metadata=dict(data.get("metadata", {})),
        )
        suite.validate()
        return suite

    @classmethod
    def load(cls, path: str | Path) -> EvaluationSuite:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("evaluation suite must be a JSON object")
        return cls.from_dict(data)

    def build_blind_pairwise_sheet(
        self,
        responses: Mapping[str, Mapping[str, str]],
        *,
        seed: str,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Create a public A/B sheet and a separate private assignment key."""
        self.validate()
        for configuration in self.configurations:
            if configuration.id not in responses:
                raise ValueError(f"missing responses for {configuration.id}")
            missing_rows = {
                row.id for row in self.rows if row.id not in responses[configuration.id]
            }
            if missing_rows:
                raise ValueError(
                    f"configuration {configuration.id} is missing "
                    f"{len(missing_rows)} rows"
                )

        comparisons = []
        assignments: dict[str, dict[str, str]] = {}
        for row in self.rows:
            for left, right in itertools.combinations(self.configurations, 2):
                comparison_id = _stable_id(
                    "comparison",
                    {"row_id": row.id, "left": left.id, "right": right.id},
                )
                ordering_digest = hashlib.sha256(
                    f"{seed}:{comparison_id}".encode()
                ).digest()
                first, second = (right, left) if ordering_digest[0] % 2 else (left, right)
                comparisons.append(
                    {
                        "comparison_id": comparison_id,
                        "row_id": row.id,
                        "prompt": row.prompt,
                        "response_a": responses[first.id][row.id],
                        "response_b": responses[second.id][row.id],
                        "more_like_me": None,
                        "notes": None,
                    }
                )
                assignments[comparison_id] = {
                    "response_a": first.id,
                    "response_b": second.id,
                }

        sheet = {
            "schema_version": "digital_self_pairwise_sheet.v1",
            "profile_id": self.profile_id,
            "instructions": [
                "For each prompt, choose A, B, tie, or neither.",
                "Judge which response sounds more like you, not which is more polished.",
                "Use notes for factual errors, stale preferences, or privacy concerns.",
            ],
            "comparisons": comparisons,
        }
        answer_key = {
            "schema_version": "digital_self_pairwise_key.v1",
            "profile_id": self.profile_id,
            "assignments": assignments,
        }
        return sheet, answer_key


class EvaluationSuiteBuilder:
    """Build held-out and explicit self-report rows without training leakage."""

    def __init__(
        self,
        configurations: Sequence[EvaluationConfiguration] = DEFAULT_CONFIGURATIONS,
    ):
        self.configurations = tuple(configurations)

    def build(
        self,
        build_result: DigitalSelfBuildResult,
        *,
        interview: Mapping[str, Any] | None = None,
    ) -> EvaluationSuite:
        rows = self._held_out_rows(build_result)
        rows.extend(self._interview_rows(build_result, interview))
        suite = EvaluationSuite(
            profile_id=build_result.profile.profile_id,
            created_at=build_result.profile.updated_at,
            rows=sorted(rows, key=lambda row: (row.origin, row.source_ref, row.id)),
            configurations=self.configurations,
            metadata={
                "private_artifact": True,
                "row_identity_excludes_configuration": True,
                "automatic_summary_contains_dialogue": False,
            },
        )
        suite.validate()
        return suite

    @staticmethod
    def _held_out_rows(
        build_result: DigitalSelfBuildResult,
    ) -> list[EvaluationRow]:
        rows = []
        for group_id, messages in sorted(build_result.message_groups.items()):
            split = build_result.split_by_group[group_id]
            if split not in {"validation", "test"}:
                continue
            for previous, response in zip(messages, messages[1:]):
                if (
                    previous.from_owner
                    or not previous.text
                    or not response.from_owner
                    or not response.text
                ):
                    continue
                source_ref = message_key(response)
                rows.append(
                    EvaluationRow(
                        id=_stable_id(
                            "evaluation-row",
                            {"origin": "held_out_reply", "source_ref": source_ref},
                        ),
                        origin="held_out_reply",
                        source_ref=source_ref,
                        source_group_id=group_id,
                        prompt=previous.text,
                        reference_response=response.text,
                        as_of=response.timestamp,
                        tags=("authenticity", "relationship_conditioning"),
                        split=split,
                        relationship_id=response.relationship_id,
                    )
                )
        return rows

    @classmethod
    def _interview_rows(
        cls,
        build_result: DigitalSelfBuildResult,
        interview: Mapping[str, Any] | None,
    ) -> list[EvaluationRow]:
        if not interview:
            return []
        if interview.get("schema_version") != "digital_self_interview.v1":
            raise ValueError("unsupported interview schema")

        completed_at = _parse_datetime(
            str(interview["completed_at"])
            if interview.get("completed_at") is not None
            else None,
            build_result.profile.updated_at,
        )
        questions = []
        for section in interview.get("sections", []):
            questions.extend(section.get("questions", []))

        rows = []
        for question in sorted(questions, key=lambda item: str(item.get("id", ""))):
            answer = question.get("answer")
            if answer is None or not str(answer).strip():
                continue
            question_id = str(question.get("id", "")).strip()
            prompt = str(question.get("retest_prompt") or question.get("prompt") or "")
            if not question_id or not prompt.strip():
                raise ValueError("answered interview questions require an id and prompt")

            tags = cls._question_tags(question_id, question.get("evaluation_tags"))
            source_ref = f"interview:{question_id}"
            rows.append(
                EvaluationRow(
                    id=_stable_id(
                        "evaluation-row",
                        {"origin": "interview_retest", "source_ref": source_ref},
                    ),
                    origin="interview_retest",
                    source_ref=source_ref,
                    source_group_id="interview",
                    prompt=prompt.strip(),
                    reference_response=str(answer).strip(),
                    as_of=completed_at,
                    tags=tags,
                    relationship_id=(
                        str(question["relationship_id"])
                        if question.get("relationship_id") is not None
                        else None
                    ),
                )
            )
        return rows

    @staticmethod
    def _question_tags(question_id: str, explicit_tags: Any) -> tuple[str, ...]:
        tags = {"grounding"}
        if isinstance(explicit_tags, (list, tuple, set)):
            tags.update(str(tag) for tag in explicit_tags if str(tag).strip())
        if question_id.startswith("preferences.") or question_id == "changes.preferences":
            tags.add("latest_preference")
        if question_id.startswith("changes."):
            tags.update({"supersession", "contradiction"})
        if question_id.startswith("boundaries."):
            tags.add("abstention")
        if question_id == "relationships.privacy":
            tags.add("relationship_leakage")
        return tuple(sorted(tags))


__all__ = [
    "DEFAULT_CONFIGURATIONS",
    "REQUIRED_EVALUATION_TAGS",
    "EvaluationConfiguration",
    "EvaluationRow",
    "EvaluationSuite",
    "EvaluationSuiteBuilder",
]
