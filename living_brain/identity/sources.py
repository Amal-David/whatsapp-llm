"""Normalized, read-only message-source contracts for identity evidence."""

from __future__ import annotations

import hashlib
import hmac
import json
import sqlite3
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import quote


@dataclass(frozen=True)
class ChatDescriptor:
    """Local-only chat metadata used to make an explicit source selection."""

    source_chat_id: str
    display_name: str
    kind: str
    message_count: int
    last_message_at: datetime | None = None


@dataclass(frozen=True)
class NormalizedMessage:
    """Source-independent evidence event with pseudonymous participant IDs."""

    source: str
    source_message_id: str
    chat_id: str
    relationship_id: str
    sender_id: str
    timestamp: datetime
    from_owner: bool
    text: str | None
    message_type: str
    content_hash: str
    reply_to_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "source_message_id": self.source_message_id,
            "chat_id": self.chat_id,
            "relationship_id": self.relationship_id,
            "sender_id": self.sender_id,
            "timestamp": self.timestamp.isoformat(),
            "from_owner": self.from_owner,
            "text": self.text,
            "message_type": self.message_type,
            "content_hash": self.content_hash,
            "reply_to_id": self.reply_to_id,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NormalizedMessage:
        return cls(
            source=data["source"],
            source_message_id=data["source_message_id"],
            chat_id=data["chat_id"],
            relationship_id=data["relationship_id"],
            sender_id=data["sender_id"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            from_owner=bool(data["from_owner"]),
            text=data.get("text"),
            message_type=data["message_type"],
            content_hash=data["content_hash"],
            reply_to_id=data.get("reply_to_id"),
            metadata=dict(data.get("metadata", {})),
        )


class MessageSource(Protocol):
    """The intentionally small read-only source surface."""

    def list_chats(self) -> list[ChatDescriptor]: ...

    def read_messages(
        self,
        chat_ids: Sequence[str] | None = None,
        *,
        all_chats: bool = False,
    ) -> list[NormalizedMessage]: ...


def pseudonymize(key: bytes, namespace: str, value: str) -> str:
    """Create a stable keyed identifier that cannot be reversed by dictionary lookup."""
    if not key:
        raise ValueError("pseudonym_key cannot be empty")
    digest = hmac.new(key, f"{namespace}:{value}".encode(), hashlib.sha256).hexdigest()
    return f"{namespace}:{digest}"


def ensure_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


@contextmanager
def open_sqlite_readonly(path: str | Path) -> Iterator[sqlite3.Connection]:
    """Open a SQLite database in a consistent read transaction with writes disabled."""
    database_path = Path(path).expanduser().resolve()
    if not database_path.is_file():
        raise FileNotFoundError(database_path)

    encoded_path = quote(str(database_path), safe="/")
    connection = sqlite3.connect(
        f"file:{encoded_path}?mode=ro",
        uri=True,
        timeout=10,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA query_only = ON")
    connection.execute("BEGIN")
    try:
        yield connection
    finally:
        connection.rollback()
        connection.close()


class BaseMessageSource:
    """Shared normalization and selection behavior for source adapters."""

    source_name = "unknown"

    def __init__(self, pseudonym_key: bytes):
        if not pseudonym_key:
            raise ValueError("pseudonym_key cannot be empty")
        self._pseudonym_key = pseudonym_key

    @staticmethod
    def selected_chat_ids(
        chat_ids: Sequence[str] | None,
        all_chats: bool,
    ) -> set[str] | None:
        selected = {chat_id for chat_id in chat_ids or [] if chat_id}
        if selected and all_chats:
            raise ValueError("cannot combine chat_ids with all_chats")
        if not selected and not all_chats:
            raise ValueError("explicit chat selection or all_chats=True is required")
        return None if all_chats else selected

    def normalize(
        self,
        *,
        source_message_id: str,
        source_chat_id: str,
        source_sender_id: str,
        timestamp: datetime,
        from_owner: bool,
        text: str | None,
        message_type: str,
        reply_to_id: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> NormalizedMessage:
        normalized_text = text if text and text.strip() else None
        return NormalizedMessage(
            source=self.source_name,
            source_message_id=source_message_id,
            chat_id=pseudonymize(self._pseudonym_key, "chat", source_chat_id),
            relationship_id=pseudonymize(
                self._pseudonym_key,
                "relationship",
                source_chat_id,
            ),
            sender_id=pseudonymize(
                self._pseudonym_key,
                "sender",
                "owner" if from_owner else source_sender_id,
            ),
            timestamp=ensure_utc(timestamp),
            from_owner=from_owner,
            text=normalized_text,
            message_type=message_type,
            content_hash=hashlib.sha256((normalized_text or "").encode()).hexdigest(),
            reply_to_id=reply_to_id,
            metadata=metadata or {},
        )


class JsonMessageSource(BaseMessageSource):
    """Portable deterministic source used for fixtures and offline interchange."""

    source_name = "json"

    def __init__(self, path: str | Path, pseudonym_key: bytes):
        super().__init__(pseudonym_key)
        self.path = Path(path)

    def _load(self) -> dict[str, Any]:
        data = json.loads(self.path.read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("message source JSON must be an object")
        if not isinstance(data.get("chats", []), list) or not isinstance(
            data.get("messages", []), list
        ):
            raise ValueError("message source JSON chats and messages must be lists")
        return data

    def list_chats(self) -> list[ChatDescriptor]:
        data = self._load()
        messages = data.get("messages", [])
        descriptors = []
        for chat in data.get("chats", []):
            source_chat_id = chat["source_chat_id"]
            chat_messages = [
                message
                for message in messages
                if message["source_chat_id"] == source_chat_id
            ]
            timestamps = [
                datetime.fromisoformat(message["timestamp"])
                for message in chat_messages
            ]
            descriptors.append(
                ChatDescriptor(
                    source_chat_id=source_chat_id,
                    display_name=chat.get("display_name") or source_chat_id,
                    kind=chat.get("kind", "unknown"),
                    message_count=len(chat_messages),
                    last_message_at=max(timestamps) if timestamps else None,
                )
            )
        return sorted(
            descriptors,
            key=lambda chat: (chat.last_message_at or datetime.min.replace(tzinfo=timezone.utc)),
            reverse=True,
        )

    def read_messages(
        self,
        chat_ids: Sequence[str] | None = None,
        *,
        all_chats: bool = False,
    ) -> list[NormalizedMessage]:
        selected = self.selected_chat_ids(chat_ids, all_chats)
        normalized = []
        for message in self._load().get("messages", []):
            source_chat_id = message["source_chat_id"]
            if selected is not None and source_chat_id not in selected:
                continue
            normalized.append(
                self.normalize(
                    source_message_id=message["source_message_id"],
                    source_chat_id=source_chat_id,
                    source_sender_id=message.get("sender_id", "unknown"),
                    timestamp=datetime.fromisoformat(message["timestamp"]),
                    from_owner=bool(message["from_owner"]),
                    text=message.get("text"),
                    message_type=message.get("message_type", "text"),
                    reply_to_id=message.get("reply_to_id"),
                    metadata=dict(message.get("metadata", {})),
                )
            )
        return sorted(normalized, key=lambda message: (message.timestamp, message.source_message_id))


__all__ = [
    "BaseMessageSource",
    "ChatDescriptor",
    "JsonMessageSource",
    "MessageSource",
    "NormalizedMessage",
    "open_sqlite_readonly",
    "pseudonymize",
]
