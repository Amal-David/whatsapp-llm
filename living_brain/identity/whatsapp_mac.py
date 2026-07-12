"""Read-only adapter for WhatsApp for Mac's local Core Data store."""

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

APPLE_REFERENCE_UNIX_SECONDS = 978_307_200


def apple_timestamp_to_datetime(value: float | int) -> datetime | None:
    """Convert Apple's 2001 reference timestamp, ignoring app sentinel values."""
    try:
        return datetime.fromtimestamp(
            float(value) + APPLE_REFERENCE_UNIX_SECONDS,
            timezone.utc,
        )
    except (OSError, OverflowError, ValueError):
        return None


def _session_kind(session_type: int | None) -> str:
    if session_type == 0:
        return "dm"
    if session_type == 1:
        return "group"
    return "unknown"


class WhatsAppMacSource(BaseMessageSource):
    """Read selected chats from ChatStorage.sqlite without touching app state."""

    source_name = "whatsapp_mac"

    def __init__(self, database_path: str | Path, pseudonym_key: bytes):
        super().__init__(pseudonym_key)
        self.database_path = Path(database_path)

    def list_chats(self) -> list[ChatDescriptor]:
        query = """
            SELECT
                COALESCE(ZCONTACTJID, CAST(Z_PK AS TEXT)) AS source_chat_id,
                ZPARTNERNAME,
                ZSESSIONTYPE,
                ZMESSAGECOUNTER,
                ZLASTMESSAGEDATE
            FROM ZWACHATSESSION
            ORDER BY ZLASTMESSAGEDATE DESC, source_chat_id ASC
        """
        with open_sqlite_readonly(self.database_path) as connection:
            rows = connection.execute(query).fetchall()

        return [
            ChatDescriptor(
                source_chat_id=row["source_chat_id"],
                display_name=row["ZPARTNERNAME"] or row["source_chat_id"],
                kind=_session_kind(row["ZSESSIONTYPE"]),
                message_count=int(row["ZMESSAGECOUNTER"] or 0),
                last_message_at=(
                    apple_timestamp_to_datetime(row["ZLASTMESSAGEDATE"])
                    if row["ZLASTMESSAGEDATE"] is not None
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
            selection_clause = (
                "AND COALESCE(c.ZCONTACTJID, CAST(c.Z_PK AS TEXT)) "
                f"IN ({placeholders})"
            )
            parameters.extend(sorted(selected))

        query = f"""
            SELECT
                COALESCE(c.ZCONTACTJID, CAST(c.Z_PK AS TEXT)) AS source_chat_id,
                c.ZSESSIONTYPE,
                m.Z_PK,
                m.ZSTANZAID,
                m.ZMESSAGEDATE,
                m.ZISFROMME,
                m.ZFROMJID,
                m.ZTOJID,
                m.ZPUSHNAME,
                m.ZTEXT,
                m.ZMESSAGETYPE
            FROM ZWAMESSAGE AS m
            JOIN ZWACHATSESSION AS c ON c.Z_PK = m.ZCHATSESSION
            WHERE m.ZMESSAGEDATE IS NOT NULL
              {selection_clause}
            ORDER BY m.ZMESSAGEDATE ASC, m.Z_PK ASC
        """
        with open_sqlite_readonly(self.database_path) as connection:
            rows = connection.execute(query, parameters).fetchall()

        messages = []
        for row in rows:
            timestamp = apple_timestamp_to_datetime(row["ZMESSAGEDATE"])
            if timestamp is None:
                continue
            from_owner = bool(row["ZISFROMME"])
            text = row["ZTEXT"]
            message_type_code = int(row["ZMESSAGETYPE"] or 0)
            messages.append(
                self.normalize(
                    source_message_id=row["ZSTANZAID"] or str(row["Z_PK"]),
                    source_chat_id=row["source_chat_id"],
                    source_sender_id=(
                        "owner"
                        if from_owner
                        else row["ZFROMJID"] or row["ZPUSHNAME"] or "unknown"
                    ),
                    timestamp=timestamp,
                    from_owner=from_owner,
                    text=text,
                    message_type=(
                        "text" if text and text.strip() else f"whatsapp_{message_type_code}"
                    ),
                    metadata={
                        "chat_kind": _session_kind(row["ZSESSIONTYPE"]),
                        "message_type_code": message_type_code,
                    },
                )
            )
        return messages


__all__ = [
    "APPLE_REFERENCE_UNIX_SECONDS",
    "WhatsAppMacSource",
    "apple_timestamp_to_datetime",
]
