"""Read-only adapter for the local wacli SQLite mirror."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

from .sources import (
    BaseMessageSource,
    ChatDescriptor,
    NormalizedMessage,
    open_sqlite_readonly,
)


class WacliSource(BaseMessageSource):
    """Normalize messages captured by wacli without invoking its write surface."""

    source_name = "wacli"

    def __init__(self, database_path: str | Path, pseudonym_key: bytes):
        super().__init__(pseudonym_key)
        self.database_path = Path(database_path)

    def list_chats(self) -> list[ChatDescriptor]:
        query = """
            SELECT
                c.jid,
                c.kind,
                c.name,
                c.last_message_ts,
                COUNT(m.rowid) AS message_count
            FROM chats AS c
            LEFT JOIN messages AS m
              ON m.chat_jid = c.jid
             AND COALESCE(m.revoked, 0) = 0
             AND COALESCE(m.deleted_for_me, 0) = 0
            GROUP BY c.jid, c.kind, c.name, c.last_message_ts
            ORDER BY c.last_message_ts DESC, c.jid ASC
        """
        with open_sqlite_readonly(self.database_path) as connection:
            rows = connection.execute(query).fetchall()

        return [
            ChatDescriptor(
                source_chat_id=row["jid"],
                display_name=row["name"] or row["jid"],
                kind=row["kind"],
                message_count=int(row["message_count"]),
                last_message_at=(
                    datetime.fromtimestamp(row["last_message_ts"], timezone.utc)
                    if row["last_message_ts"] is not None
                    else None
                ),
            )
            for row in rows
        ]

    def read_messages(
        self,
        chat_ids: Sequence[str] | None = None,
        *,
        all_chats: bool = False,
    ) -> list[NormalizedMessage]:
        selected = self.selected_chat_ids(chat_ids, all_chats)
        parameters: list[str] = []
        selection_clause = ""
        if selected is not None:
            placeholders = ", ".join("?" for _ in selected)
            selection_clause = f"AND m.chat_jid IN ({placeholders})"
            parameters.extend(sorted(selected))

        query = f"""
            SELECT
                m.chat_jid,
                c.kind AS chat_kind,
                m.msg_id,
                m.sender_jid,
                m.sender_name,
                m.ts,
                m.from_me,
                m.text,
                m.display_text,
                m.quoted_msg_id,
                m.is_forwarded,
                m.media_type,
                m.media_caption,
                m.edited
            FROM messages AS m
            LEFT JOIN chats AS c ON c.jid = m.chat_jid
            WHERE COALESCE(m.revoked, 0) = 0
              AND COALESCE(m.deleted_for_me, 0) = 0
              {selection_clause}
            ORDER BY m.ts ASC, m.msg_id ASC
        """
        with open_sqlite_readonly(self.database_path) as connection:
            rows = connection.execute(query, parameters).fetchall()

        messages = []
        for row in rows:
            from_owner = bool(row["from_me"])
            text = row["text"] or row["display_text"] or row["media_caption"]
            messages.append(
                self.normalize(
                    source_message_id=row["msg_id"],
                    source_chat_id=row["chat_jid"],
                    source_sender_id=(
                        "owner"
                        if from_owner
                        else row["sender_jid"] or row["sender_name"] or "unknown"
                    ),
                    timestamp=datetime.fromtimestamp(row["ts"], timezone.utc),
                    from_owner=from_owner,
                    text=text,
                    message_type=row["media_type"] or "text",
                    reply_to_id=row["quoted_msg_id"],
                    metadata={
                        "chat_kind": row["chat_kind"] or "unknown",
                        "is_forwarded": bool(row["is_forwarded"]),
                        "edited": bool(row["edited"]),
                    },
                )
            )
        return messages


__all__ = ["WacliSource"]
