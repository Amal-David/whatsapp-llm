"""
Persona dataset preparation for WhatsApp exports.

This module is intentionally model-free. It turns parsed WhatsApp messages into
deterministic artifacts that can be inspected before any fine-tuning run.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .style_analyzer import StyleAnalyzer, StyleMetrics
from .whatsapp_parser import ChatMessage, Conversation, WhatsAppParser

REDACTION_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("EMAIL", re.compile(r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b", re.IGNORECASE)),
    ("URL", re.compile(r"\b(?:https?://|www\.)\S+\b", re.IGNORECASE)),
    ("PHONE", re.compile(r"(?<!\d)(?:\+?\d[\d\s().-]{7,}\d)(?!\d)")),
    ("OTP_OR_CODE", re.compile(r"\b(?:otp|code|pin)\s*[:=-]?\s*\d{4,8}\b", re.IGNORECASE)),
    ("PAYMENT_ID", re.compile(r"\b[a-zA-Z0-9._-]+@[a-zA-Z]{2,}\b")),
)

FILLER_PARTICLES = {
    "acha",
    "achha",
    "ah",
    "ahh",
    "arre",
    "btw",
    "haan",
    "haaan",
    "haha",
    "hehe",
    "hmm",
    "hmmm",
    "lol",
    "lmao",
    "like",
    "na",
    "nah",
    "okay",
    "ok",
    "umm",
    "wait",
    "ya",
    "yaar",
    "yeah",
}

SCRIPT_RANGES: tuple[tuple[str, tuple[int, int]], ...] = (
    ("latin", (0x0041, 0x007A)),
    ("devanagari", (0x0900, 0x097F)),
    ("bengali", (0x0980, 0x09FF)),
    ("gurmukhi", (0x0A00, 0x0A7F)),
    ("gujarati", (0x0A80, 0x0AFF)),
    ("tamil", (0x0B80, 0x0BFF)),
    ("telugu", (0x0C00, 0x0C7F)),
    ("kannada", (0x0C80, 0x0CFF)),
    ("malayalam", (0x0D00, 0x0D7F)),
    ("arabic", (0x0600, 0x06FF)),
    ("cjk", (0x4E00, 0x9FFF)),
)


@dataclass
class PersonaBuildResult:
    """All generated artifacts for one participant."""

    participant: str
    participants: list[str]
    summary: dict[str, Any]
    canonical_examples: list[dict[str, Any]]
    sft_alpaca: list[dict[str, Any]]
    sft_messages: list[dict[str, Any]]
    dpo_trl: list[dict[str, Any]]
    dpo_openai: list[dict[str, Any]]
    eval_rows: list[dict[str, Any]]
    style_capsule: dict[str, Any]
    canonical_character: dict[str, Any]
    character_card_v2: dict[str, Any]
    persona_markdown: str
    recommendation_markdown: str


def slugify_name(name: str) -> str:
    """Create a stable, readable id from a participant name."""
    slug = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")
    return slug or "participant"


def stable_hash(*parts: object) -> str:
    """Hash stable JSON-serializable parts for deterministic ids."""
    payload = json.dumps(parts, ensure_ascii=False, sort_keys=True, default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def redact_text(text: str) -> tuple[str, list[str]]:
    """Redact common private identifiers while preserving style cues."""
    redactions: list[str] = []
    redacted = text
    for label, pattern in REDACTION_PATTERNS:
        if pattern.search(redacted):
            redacted = pattern.sub(f"[{label}]", redacted)
            redactions.append(label)
    return redacted, sorted(set(redactions))


def jsonl_dumps(rows: Iterable[dict[str, Any]]) -> str:
    """Serialize rows as JSONL."""
    return "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"


class PersonaDatasetBuilder:
    """Build persona training artifacts from a WhatsApp export."""

    def __init__(
        self,
        context_turns: int = 6,
        gap_minutes: int = 60,
        include_third_party_context: bool = False,
    ):
        self.context_turns = context_turns
        self.gap_minutes = gap_minutes
        self.include_third_party_context = include_third_party_context
        self.style_analyzer = StyleAnalyzer()

    def build_from_file(
        self,
        chat_file: str | Path,
        participant: str,
        owner_type: str = "self",
    ) -> PersonaBuildResult:
        """Parse a chat file and build every artifact for one participant."""
        parser = WhatsAppParser(your_name=participant)
        conversations = parser.parse_to_conversations(
            chat_file,
            gap_minutes=self.gap_minutes,
            skip_system=True,
        )
        source_participants = sorted(parser.get_participants(chat_file))
        participants = (
            source_participants if self.include_third_party_context else [participant]
        )
        target_messages = [
            message
            for conversation in conversations
            for message in conversation.messages
            if message.author == participant
        ]

        canonical_examples = self._canonical_examples(conversations, participant)
        split_by_group = self._assign_splits(canonical_examples)
        train_group_ids = {
            group_id for group_id, split in split_by_group.items() if split == "train"
        }
        training_messages = [
            message
            for conv_index, conversation in enumerate(conversations)
            if f"conv_{conv_index:04d}" in train_group_ids
            for message in conversation.messages
            if message.author == participant
        ]
        if not canonical_examples:
            training_messages = target_messages

        style_metrics = self.style_analyzer.analyze(training_messages)
        feature_summary = self._feature_summary(training_messages)
        training_examples = [
            row for row in canonical_examples if row["split"] == "train"
        ]

        style_capsule = self._style_capsule(participant, style_metrics, feature_summary)
        recommendation = self._recommendation(participant, len(target_messages))
        sft_alpaca = self._sft_alpaca(training_examples, style_capsule)
        sft_messages = self._sft_messages(training_examples, style_capsule)
        dpo_trl = self._dpo_trl(training_examples, style_capsule)
        dpo_openai = self._dpo_openai(training_examples)
        eval_rows = self._eval_rows(canonical_examples, style_capsule)
        canonical_character = self._canonical_character(
            participant=participant,
            participants=participants,
            style_metrics=style_metrics,
            feature_summary=feature_summary,
            canonical_examples=training_examples,
            owner_type=owner_type,
        )
        character_card_v2 = self._character_card_v2(canonical_character)
        persona_markdown = self._persona_markdown(canonical_character)
        recommendation_markdown = self._recommendation_markdown(recommendation)

        summary = {
            "participant": participant,
            "participants": participants,
            "owner_type": owner_type,
            "conversation_count": len(conversations),
            "target_message_count": len(target_messages),
            "training_target_message_count": style_metrics.message_count,
            "canonical_example_count": len(canonical_examples),
            "sft_example_count": len(sft_alpaca),
            "dpo_example_count": len(dpo_trl),
            "eval_example_count": len(eval_rows),
            "recommendation": recommendation,
            "style_features": feature_summary,
            "privacy": {
                "raw_quotes_in_shared_character_exports": False,
                "redaction_version": "basic_redaction_v1",
                "third_party_context_included": self.include_third_party_context,
            },
        }

        return PersonaBuildResult(
            participant=participant,
            participants=participants,
            summary=summary,
            canonical_examples=canonical_examples,
            sft_alpaca=sft_alpaca,
            sft_messages=sft_messages,
            dpo_trl=dpo_trl,
            dpo_openai=dpo_openai,
            eval_rows=eval_rows,
            style_capsule=style_capsule,
            canonical_character=canonical_character,
            character_card_v2=character_card_v2,
            persona_markdown=persona_markdown,
            recommendation_markdown=recommendation_markdown,
        )

    def artifact_strings(self, result: PersonaBuildResult) -> dict[str, str]:
        """Return every generated artifact as filename -> text."""
        return {
            "summary.json": json.dumps(result.summary, ensure_ascii=False, indent=2) + "\n",
            "canonical_examples.jsonl": jsonl_dumps(result.canonical_examples),
            "sft_alpaca.jsonl": jsonl_dumps(result.sft_alpaca),
            "sft_messages.jsonl": jsonl_dumps(result.sft_messages),
            "dpo_trl.jsonl": jsonl_dumps(result.dpo_trl),
            "dpo_openai.jsonl": jsonl_dumps(result.dpo_openai),
            "eval.jsonl": jsonl_dumps(result.eval_rows),
            "style_capsule.json": json.dumps(
                result.style_capsule,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            "canonical_character.json": json.dumps(
                result.canonical_character,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            "character_card_v2.json": json.dumps(
                result.character_card_v2,
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            "persona.md": result.persona_markdown,
            "recommendation.md": result.recommendation_markdown,
        }

    def write_artifacts(self, result: PersonaBuildResult, output_dir: str | Path) -> dict[str, Path]:
        """Write artifacts to an output directory."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        written: dict[str, Path] = {}
        for filename, content in self.artifact_strings(result).items():
            path = output_path / filename
            path.write_text(content, encoding="utf-8")
            written[filename] = path
        return written

    def _canonical_examples(
        self,
        conversations: list[Conversation],
        participant: str,
    ) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        persona_id = slugify_name(participant)

        for conv_index, conversation in enumerate(conversations):
            messages = conversation.messages
            for msg_index, message in enumerate(messages):
                if message.author != participant or not self._is_trainable_text(message.message):
                    continue
                if msg_index == 0:
                    continue

                context_messages = messages[max(0, msg_index - self.context_turns) : msg_index]
                context: list[dict[str, str]] = []
                redactions: list[str] = []

                for context_message in context_messages:
                    content, labels = self._context_content(context_message, participant)
                    redactions.extend(labels)
                    role = "self" if context_message.author == participant else "other"
                    speaker = context_message.author
                    if role == "other" and not self.include_third_party_context:
                        speaker = "other"
                    context.append(
                        {
                            "role": role,
                            "speaker": speaker,
                            "content": content,
                        }
                    )

                target_content, target_redactions = redact_text(message.message)
                redactions.extend(target_redactions)
                example_id = "sha256:" + stable_hash(
                    participant,
                    message.timestamp.isoformat(),
                    message.message,
                    [(item["speaker"], item["content"]) for item in context],
                )

                rows.append(
                    {
                        "schema_version": "whatsapp_persona.v1",
                        "example_id": example_id,
                        "split": "train",
                        "split_group_id": f"conv_{conv_index:04d}",
                        "persona_id": persona_id,
                        "style_capsule_id": f"{persona_id}_global_v1",
                        "source": {
                            "conversation_id": f"conv_{conv_index:04d}",
                            "target_message_index": msg_index,
                            "timestamp": message.timestamp.isoformat(),
                        },
                        "context": context,
                        "target": {
                            "role": "self",
                            "speaker": participant,
                            "content": target_content,
                        },
                        "privacy": {
                            "redaction_version": "basic_redaction_v1",
                            "redactions": sorted(set(redactions)),
                            "risk": "medium" if redactions else "low",
                        },
                        "quality": {
                            "has_context": bool(context),
                            "target_len_chars": len(target_content),
                            "synthetic": False,
                            "drop_reasons": [],
                        },
                    }
                )

        return rows

    def _assign_splits(self, rows: list[dict[str, Any]]) -> dict[str, str]:
        group_ids = list(dict.fromkeys(row["split_group_id"] for row in rows))
        if not group_ids:
            return {}

        if len(group_ids) < 3:
            split_by_group = {group_id: "train" for group_id in group_ids}
        else:
            holdout_count = min(
                len(group_ids) - 1,
                max(2, math.ceil(len(group_ids) * 0.2)),
            )
            validation_count = max(1, holdout_count // 2)
            train_count = len(group_ids) - holdout_count
            validation_end = train_count + validation_count
            split_by_group = {}
            for index, group_id in enumerate(group_ids):
                if index < train_count:
                    split = "train"
                elif index < validation_end:
                    split = "validation"
                else:
                    split = "test"
                split_by_group[group_id] = split

        for row in rows:
            row["split"] = split_by_group[row["split_group_id"]]
        return split_by_group

    def _context_content(self, message: ChatMessage, participant: str) -> tuple[str, list[str]]:
        if message.author != participant and not self.include_third_party_context:
            return "[third-party message withheld]", ["THIRD_PARTY_CONTEXT_WITHHELD"]
        return redact_text(message.message)

    def _sft_alpaca(
        self,
        rows: list[dict[str, Any]],
        style_capsule: dict[str, Any],
    ) -> list[dict[str, Any]]:
        return [
            {
                "system": style_capsule["system_prompt"],
                "instruction": "Continue this WhatsApp conversation as the target participant.",
                "input": self._context_as_text(row["context"]),
                "output": row["target"]["content"],
                "metadata": {
                    "example_id": row["example_id"],
                    "split": row["split"],
                    "style_capsule_id": row["style_capsule_id"],
                },
            }
            for row in rows
        ]

    def _sft_messages(
        self,
        rows: list[dict[str, Any]],
        style_capsule: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records = []
        for row in rows:
            records.append(
                {
                    "messages": [
                        {"role": "system", "content": style_capsule["system_prompt"]},
                        {"role": "user", "content": self._context_as_text(row["context"])},
                        {"role": "assistant", "content": row["target"]["content"]},
                    ],
                    "metadata": {
                        "example_id": row["example_id"],
                        "split": row["split"],
                    },
                }
            )
        return records

    def _dpo_trl(
        self,
        rows: list[dict[str, Any]],
        style_capsule: dict[str, Any],
    ) -> list[dict[str, Any]]:
        records = []
        for row in rows:
            rejected = self._formal_negative(row["target"]["content"])
            records.append(
                {
                    "prompt": [
                        {"role": "system", "content": style_capsule["system_prompt"]},
                        {"role": "user", "content": self._context_as_text(row["context"])},
                    ],
                    "chosen": [{"role": "assistant", "content": row["target"]["content"]}],
                    "rejected": [{"role": "assistant", "content": rejected}],
                    "metadata": {
                        "example_id": row["example_id"],
                        "split": row["split"],
                        "synthetic": True,
                        "negative_source": "formality_mutation",
                        "preference_reason": [
                            "chosen_is_observed_target_reply",
                            "rejected_is_synthetic_too_formal_variant",
                        ],
                    },
                }
            )
        return records

    def _dpo_openai(self, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records = []
        for row in rows:
            records.append(
                {
                    "input": {
                        "messages": [
                            {
                                "role": "user",
                                "content": self._context_as_text(row["context"]),
                            }
                        ],
                        "tools": [],
                        "parallel_tool_calls": False,
                    },
                    "preferred_output": [
                        {"role": "assistant", "content": row["target"]["content"]}
                    ],
                    "non_preferred_output": [
                        {"role": "assistant", "content": self._formal_negative(row["target"]["content"])}
                    ],
                    "metadata": {
                        "example_id": row["example_id"],
                        "split": row["split"],
                        "synthetic": True,
                        "negative_source": "formality_mutation",
                    },
                }
            )
        return records

    def _eval_rows(
        self,
        rows: list[dict[str, Any]],
        style_capsule: dict[str, Any],
    ) -> list[dict[str, Any]]:
        eval_source = [row for row in rows if row["split"] in {"validation", "test"}]

        return [
            {
                "eval_id": "eval_" + row["example_id"].split(":", 1)[1][:16],
                "task": "next_reply_style",
                "split": row["split"],
                "prompt_messages": [
                    {"role": "system", "content": style_capsule["system_prompt"]},
                    {"role": "user", "content": self._context_as_text(row["context"])},
                ],
                "reference": row["target"]["content"],
                "rubric": {
                    "style_match": "1-5",
                    "context_fit": "1-5",
                    "privacy_safety": "pass_fail",
                    "no_fact_invention": "pass_fail",
                    "memorization": "pass_fail",
                },
                "source_example_id": row["example_id"],
            }
            for row in eval_source
        ]

    def _style_capsule(
        self,
        participant: str,
        metrics: StyleMetrics,
        features: dict[str, Any],
    ) -> dict[str, Any]:
        persona_id = slugify_name(participant)
        tone_rules = self._style_rules(metrics, features)
        return {
            "style_capsule_id": f"{persona_id}_global_v1",
            "persona_id": persona_id,
            "participant": participant,
            "scope": "global",
            "computed_from_split": "train",
            "metrics": metrics.to_dict(),
            "features": features,
            "rules": tone_rules,
            "system_prompt": (
                f"You are drafting as {participant} using an authorized WhatsApp style profile. "
                "Match observed tone, brevity, punctuation, emoji habits, and code-switching only when "
                "the conversation context supports it. Do not invent biography, private facts, or secrets. "
                "Do not claim to be the real person outside this authorized drafting context."
            ),
        }

    def _canonical_character(
        self,
        participant: str,
        participants: list[str],
        style_metrics: StyleMetrics,
        feature_summary: dict[str, Any],
        canonical_examples: list[dict[str, Any]],
        owner_type: str,
    ) -> dict[str, Any]:
        persona_id = slugify_name(participant)
        language_profile = feature_summary["language_profile"]
        traits = self._trait_labels(style_metrics, feature_summary)
        now = datetime.now(timezone.utc).isoformat()

        return {
            "schema": "whatsapp.participant.character",
            "schema_version": "1.0",
            "participant": {
                "id_hash": "sha256:" + stable_hash(participant)[:24],
                "display_name": participant,
                "aliases": [],
                "language_profile": language_profile,
                "timezone_hint": None,
            },
            "profile": {
                "summary": self._profile_summary(participant, style_metrics, feature_summary),
                "traits": traits,
                "topics": [],
                "relationship_context": {
                    "role_in_chat": "unknown",
                    "known_relationships": [],
                    "other_participant_count": max(0, len(participants) - 1),
                },
                "boundaries": {
                    "do_not_infer": [
                        "private facts not present in the current prompt",
                        "protected traits",
                        "third-party secrets",
                    ],
                    "safety_notes": [
                        "Use only with messages written by the participant or with explicit consent.",
                        "Label outputs as AI-assisted drafts when used outside private testing.",
                    ],
                },
            },
            "communication_style": {
                "tone": traits,
                "formality": self._formality(style_metrics),
                "verbosity": self._verbosity(style_metrics),
                "emoji_use": self._emoji_use(style_metrics),
                "punctuation": feature_summary["punctuation_style"],
                "humor": "unknown",
                "typical_openings": [],
                "typical_closings": [],
                "catchphrases": [],
                "response_patterns": {
                    "latency_style": self._latency_style(style_metrics),
                    "message_chunking": feature_summary["message_chunking"],
                },
            },
            "prompting": {
                "system": (
                    f"Draft as {participant} from an authorized style profile. "
                    "Stay close to evidenced style and current context. Do not invent private facts."
                ),
                "post_history_instructions": (
                    "Preserve the participant's observed conversational mechanics. "
                    "If the prompt asks for impersonation without consent, refuse and offer a general tone."
                ),
                "first_message": "Hey",
                "scenario": "A WhatsApp-style conversation with known participants.",
            },
            "examples": {"dialogue": []},
            "knowledge": {
                "stable_facts": [],
                "uncertain_facts": [],
                "memories": [],
            },
            "provenance": {
                "source": "whatsapp_export",
                "chat_ids": [],
                "date_range": self._date_range(canonical_examples),
                "message_count": style_metrics.message_count,
                "generated_at": now,
                "generator": {"name": "whatsapp-llm", "version": "0.1.0"},
                "privacy": {
                    "owner_type": owner_type,
                    "raw_quotes_allowed": False,
                    "anonymized": False,
                    "redaction_version": "basic_redaction_v1",
                },
            },
            "extensions": {
                "style_capsule_id": f"{persona_id}_global_v1",
                "recommended_method": self._recommendation(participant, style_metrics.message_count),
            },
        }

    def _character_card_v2(self, canonical: dict[str, Any]) -> dict[str, Any]:
        name = canonical["participant"]["display_name"]
        example_blocks = []
        for dialogue in canonical["examples"]["dialogue"]:
            lines = ["<START>"]
            for msg in dialogue["messages"]:
                speaker = "{{char}}" if msg["speaker"] == name else "{{user}}"
                lines.append(f"{speaker}: {msg['text']}")
            example_blocks.append("\n".join(lines))

        return {
            "spec": "chara_card_v2",
            "spec_version": "2.0",
            "data": {
                "name": name,
                "description": canonical["profile"]["summary"],
                "personality": ", ".join(canonical["profile"]["traits"]),
                "scenario": canonical["prompting"]["scenario"],
                "first_mes": canonical["prompting"]["first_message"],
                "mes_example": "\n".join(example_blocks),
                "creator_notes": "Generated from a consented WhatsApp style profile.",
                "system_prompt": canonical["prompting"]["system"],
                "post_history_instructions": canonical["prompting"][
                    "post_history_instructions"
                ],
                "alternate_greetings": [],
                "tags": canonical["profile"]["topics"],
                "creator": "whatsapp-llm",
                "character_version": "1.0",
                "extensions": {"whatsapp_llm": canonical},
            },
        }

    def _persona_markdown(self, canonical: dict[str, Any]) -> str:
        name = canonical["participant"]["display_name"]
        style = canonical["communication_style"]
        lines = [
            f"# {name}",
            "",
            canonical["profile"]["summary"],
            "",
            "## Voice Rules",
            f"- Formality: {style['formality']}",
            f"- Verbosity: {style['verbosity']}",
            f"- Emoji use: {style['emoji_use']}",
            f"- Punctuation: {style['punctuation']}",
            f"- Language profile: {canonical['participant']['language_profile']['code_switching']}",
            "",
            "## Boundaries",
        ]
        for item in canonical["profile"]["boundaries"]["do_not_infer"]:
            lines.append(f"- Do not infer {item}.")
        lines.extend(
            [
                "",
                "## System Prompt",
                "",
                canonical["prompting"]["system"],
                "",
                "## Examples",
            ]
        )
        for dialogue in canonical["examples"]["dialogue"]:
            lines.append("")
            lines.append(f"### {dialogue['context']}")
            for msg in dialogue["messages"]:
                speaker = "{{char}}" if msg["speaker"] == name else "{{user}}"
                lines.append(f"{speaker}: {msg['text']}")
        return "\n".join(lines) + "\n"

    def _feature_summary(self, messages: list[ChatMessage]) -> dict[str, Any]:
        texts = [message.message for message in messages if message.message]
        total = len(texts) or 1
        emoji_counter: Counter[str] = Counter()
        particle_counter: Counter[str] = Counter()
        opener_counter: Counter[str] = Counter()
        closer_counter: Counter[str] = Counter()
        script_counter: Counter[str] = Counter()
        switch_messages = 0
        elongated = 0
        all_caps = 0
        repeated_punctuation = 0
        media_only = 0
        single_word = 0
        burst_sizes: list[int] = []

        previous_author: str | None = None
        burst = 0
        for message in messages:
            if message.author == previous_author:
                burst += 1
            else:
                if burst:
                    burst_sizes.append(burst)
                burst = 1
                previous_author = message.author
        if burst:
            burst_sizes.append(burst)

        for text in texts:
            words = re.findall(r"\b[\w']+\b", text.lower())
            if len(words) == 1:
                single_word += 1
            if words:
                opener_counter[words[0]] += 1
                closer_counter[words[-1]] += 1
                particle_counter.update(word for word in words if word in FILLER_PARTICLES)

            emojis = StyleAnalyzer.EMOJI_PATTERN.findall(text)
            emoji_counter.update(emojis)
            script_counts = self._script_counts(text)
            script_counter.update(script_counts)
            active_scripts = [name for name, count in script_counts.items() if count > 0]
            if len(active_scripts) > 1:
                switch_messages += 1

            if re.search(r"([A-Za-z])\1{2,}", text):
                elongated += 1
            letters = [char for char in text if char.isalpha()]
            if letters and sum(1 for char in letters if char.isupper()) / len(letters) > 0.75:
                all_caps += 1
            if re.search(r"([!?])\1{1,}|\.{3,}|…", text):
                repeated_punctuation += 1
            if re.search(r"<media omitted>|image omitted|video omitted|sticker omitted", text, re.I):
                media_only += 1

        primary_script = script_counter.most_common(1)[0][0] if script_counter else "unknown"
        secondary_scripts = [
            name for name, count in script_counter.items() if name != primary_script and count / total > 0.05
        ]
        switch_rate = switch_messages / total
        code_switching = "frequent" if switch_rate > 0.2 else "light" if switch_rate > 0.05 else "none"
        avg_burst = sum(burst_sizes) / len(burst_sizes) if burst_sizes else 1.0

        return {
            "language_profile": {
                "primary": primary_script,
                "secondary": secondary_scripts,
                "code_switching": code_switching,
                "switch_message_rate": switch_rate,
            },
            "top_particles": particle_counter.most_common(12),
            "top_openers": [word for word, _ in opener_counter.most_common(8)],
            "top_closers": [word for word, _ in closer_counter.most_common(8)],
            "top_emojis": emoji_counter.most_common(12),
            "elongation_rate": elongated / total,
            "all_caps_rate": all_caps / total,
            "repeated_punctuation_rate": repeated_punctuation / total,
            "media_only_rate": media_only / total,
            "one_word_reply_rate": single_word / total,
            "avg_burst_size": avg_burst,
            "message_chunking": "multi-message" if avg_burst > 1.6 else "mixed" if avg_burst > 1.2 else "single",
            "punctuation_style": (
                "expressive" if repeated_punctuation / total > 0.15 else "standard"
            ),
        }

    def _script_counts(self, text: str) -> Counter[str]:
        counts: Counter[str] = Counter()
        for char in text:
            code = ord(char)
            for name, (start, end) in SCRIPT_RANGES:
                if start <= code <= end and char.isalpha():
                    counts[name] += 1
                    break
        return counts

    def _recommendation(self, participant: str, message_count: int) -> dict[str, Any]:
        if message_count < 20:
            default = "prompt_card"
            reason = "Too little data for training; use a reversible style card first."
        elif message_count < 500:
            default = "prompt_card_plus_rag"
            reason = "Enough data for a useful style profile and exemplar retrieval, not enough for robust SFT."
        elif message_count < 2000:
            default = "qlora_sft_candidate"
            reason = "Enough target replies to try SFT, but evaluate memorization and overfit carefully."
        else:
            default = "qlora_sft_recommended"
            reason = "Target reply volume can support a local QLoRA SFT adapter with held-out evals."

        preference_ready = message_count >= 200
        return {
            "participant": participant,
            "default": default,
            "reason": reason,
            "preference_training": (
                "collect pairwise or unary feedback first"
                if preference_ready
                else "not enough target prompts yet"
            ),
            "ranked_methods": [
                "prompt_card",
                "rag_persona_memory",
                "qlora_sft",
                "sft_plus_dpo",
                "kto",
                "orpo",
                "persona_adapters",
                "drpo",
                "grpo",
                "full_finetune",
            ],
        }

    def _recommendation_markdown(self, recommendation: dict[str, Any]) -> str:
        lines = [
            "# Fine-Tuning Recommendation",
            "",
            f"Participant: {recommendation['participant']}",
            f"Default: `{recommendation['default']}`",
            "",
            recommendation["reason"],
            "",
            "Preference training:",
            recommendation["preference_training"],
            "",
            "Ranked methods:",
        ]
        lines.extend(f"- {method}" for method in recommendation["ranked_methods"])
        return "\n".join(lines) + "\n"

    def _style_rules(self, metrics: StyleMetrics, features: dict[str, Any]) -> list[str]:
        rules = []
        rules.append(f"Prefer {self._verbosity(metrics)} replies.")
        rules.append(f"Use {self._emoji_use(metrics)} emoji frequency.")
        rules.append(f"Keep punctuation {features['punctuation_style']}.")
        if features["language_profile"]["code_switching"] != "none":
            rules.append("Preserve light code-switching when context already uses it.")
        if features["top_particles"]:
            particles = ", ".join(word for word, _ in features["top_particles"][:5])
            rules.append(f"Observed particles: {particles}.")
        return rules

    def _trait_labels(self, metrics: StyleMetrics, features: dict[str, Any]) -> list[str]:
        traits = [self._verbosity(metrics), self._formality(metrics)]
        if metrics.emoji_usage_rate > 0.2:
            traits.append("expressive")
        if features["one_word_reply_rate"] > 0.2:
            traits.append("terse")
        if features["language_profile"]["code_switching"] != "none":
            traits.append("code-switching")
        return sorted(set(traits))

    def _profile_summary(
        self,
        participant: str,
        metrics: StyleMetrics,
        features: dict[str, Any],
    ) -> str:
        return (
            f"{participant} tends to write {self._verbosity(metrics)} WhatsApp messages with "
            f"{self._emoji_use(metrics)} emoji use, {features['punctuation_style']} punctuation, "
            f"and {features['language_profile']['code_switching']} code-switching in the observed data."
        )

    def _date_range(self, rows: list[dict[str, Any]]) -> dict[str, str | None]:
        timestamps = [row["source"]["timestamp"] for row in rows]
        if not timestamps:
            return {"from": None, "to": None}
        return {"from": min(timestamps), "to": max(timestamps)}

    def _context_as_text(self, context: list[dict[str, str]]) -> str:
        return "\n".join(f"{item['speaker']}: {item['content']}" for item in context)

    def _formal_negative(self, chosen: str) -> str:
        clean = re.sub(StyleAnalyzer.EMOJI_PATTERN, "", chosen).strip()
        clean = re.sub(r"\s+", " ", clean)
        if not clean:
            clean = "I understand."
        if clean and clean[-1] not in ".!?":
            clean += "."
        return f"I will respond in a clear and formal manner: {clean}"

    def _is_trainable_text(self, text: str) -> bool:
        stripped = text.strip()
        if len(stripped) < 2:
            return False
        return not re.search(
            r"<media omitted>|image omitted|video omitted|audio omitted|document omitted|deleted this message",
            stripped,
            re.IGNORECASE,
        )

    def _verbosity(self, metrics: StyleMetrics) -> str:
        if metrics.avg_message_length < 45:
            return "brief"
        if metrics.avg_message_length < 140:
            return "medium"
        return "long"

    def _emoji_use(self, metrics: StyleMetrics) -> str:
        if metrics.emoji_usage_rate > 0.35:
            return "frequent"
        if metrics.emoji_usage_rate > 0.08:
            return "sparse"
        return "rare"

    def _formality(self, metrics: StyleMetrics) -> str:
        if metrics.slang_usage_rate > 0.12 or metrics.capitalization_rate < 0.4:
            return "low"
        if metrics.capitalization_rate > 0.8 and metrics.slang_usage_rate < 0.03:
            return "high"
        return "medium"

    def _latency_style(self, metrics: StyleMetrics) -> str:
        minutes = metrics.response_time_avg_minutes
        if minutes is None:
            return "unknown"
        if minutes < 5:
            return "quick"
        if minutes < 60:
            return "mixed"
        return "delayed"


__all__ = [
    "PersonaBuildResult",
    "PersonaDatasetBuilder",
    "jsonl_dumps",
    "redact_text",
    "slugify_name",
    "stable_hash",
]
