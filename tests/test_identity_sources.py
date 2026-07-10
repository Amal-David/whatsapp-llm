import hashlib
import json
import sqlite3
from datetime import datetime, timezone

import pytest

from living_brain.identity.sources import JsonMessageSource, NormalizedMessage
from living_brain.identity.wacli import WacliSource
from living_brain.identity.whatsapp_mac import (
    WhatsAppMacSource,
    apple_timestamp_to_datetime,
)

KEY = b"local-test-key"


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _create_wacli_db(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE chats (
            jid TEXT PRIMARY KEY,
            kind TEXT NOT NULL,
            name TEXT,
            last_message_ts INTEGER
        );
        CREATE TABLE messages (
            rowid INTEGER PRIMARY KEY AUTOINCREMENT,
            chat_jid TEXT NOT NULL,
            chat_name TEXT,
            msg_id TEXT NOT NULL,
            sender_jid TEXT,
            sender_name TEXT,
            ts INTEGER NOT NULL,
            from_me INTEGER NOT NULL,
            text TEXT,
            display_text TEXT,
            quoted_msg_id TEXT,
            is_forwarded INTEGER NOT NULL DEFAULT 0,
            media_type TEXT,
            media_caption TEXT,
            revoked INTEGER NOT NULL DEFAULT 0,
            deleted_for_me INTEGER NOT NULL DEFAULT 0,
            edited INTEGER NOT NULL DEFAULT 0
        );
        """
    )
    connection.executemany(
        "INSERT INTO chats (jid, kind, name, last_message_ts) VALUES (?, ?, ?, ?)",
        [
            ("friend@s.whatsapp.net", "dm", "Friend", 1_700_000_020),
            ("work@g.us", "group", "Work", 1_700_000_030),
        ],
    )
    connection.executemany(
        """
        INSERT INTO messages (
            chat_jid, chat_name, msg_id, sender_jid, sender_name, ts,
            from_me, text, display_text, quoted_msg_id, is_forwarded,
            media_type, media_caption, revoked, deleted_for_me, edited
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                "friend@s.whatsapp.net",
                "Friend",
                "m2",
                "owner@s.whatsapp.net",
                "Owner",
                1_700_000_020,
                1,
                "I choose the smaller experiment first.",
                None,
                "m1",
                0,
                None,
                None,
                0,
                0,
                0,
            ),
            (
                "friend@s.whatsapp.net",
                "Friend",
                "m1",
                "friend@s.whatsapp.net",
                "Friend",
                1_700_000_010,
                0,
                "Which option do you prefer?",
                None,
                None,
                0,
                None,
                None,
                0,
                0,
                0,
            ),
            (
                "work@g.us",
                "Work",
                "m3",
                "owner@s.whatsapp.net",
                "Owner",
                1_700_000_030,
                1,
                "Let's verify it before launch.",
                None,
                None,
                0,
                None,
                None,
                0,
                0,
                0,
            ),
        ],
    )
    connection.commit()
    connection.close()


def _create_whatsapp_mac_db(path):
    connection = sqlite3.connect(path)
    connection.executescript(
        """
        CREATE TABLE ZWACHATSESSION (
            Z_PK INTEGER PRIMARY KEY,
            ZSESSIONTYPE INTEGER,
            ZMESSAGECOUNTER INTEGER,
            ZLASTMESSAGEDATE REAL,
            ZCONTACTJID TEXT,
            ZPARTNERNAME TEXT
        );
        CREATE TABLE ZWAMESSAGE (
            Z_PK INTEGER PRIMARY KEY,
            ZCHATSESSION INTEGER,
            ZMESSAGETYPE INTEGER,
            ZISFROMME INTEGER,
            ZMESSAGEDATE REAL,
            ZFROMJID TEXT,
            ZTOJID TEXT,
            ZPUSHNAME TEXT,
            ZSTANZAID TEXT,
            ZTEXT TEXT
        );
        """
    )
    connection.execute(
        """
        INSERT INTO ZWACHATSESSION (
            Z_PK, ZSESSIONTYPE, ZMESSAGECOUNTER, ZLASTMESSAGEDATE,
            ZCONTACTJID, ZPARTNERNAME
        ) VALUES (1, 0, 2, 120, 'friend@s.whatsapp.net', 'Friend')
        """
    )
    connection.executemany(
        """
        INSERT INTO ZWAMESSAGE (
            Z_PK, ZCHATSESSION, ZMESSAGETYPE, ZISFROMME, ZMESSAGEDATE,
            ZFROMJID, ZTOJID, ZPUSHNAME, ZSTANZAID, ZTEXT
        ) VALUES (?, 1, 0, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                2,
                1,
                120,
                "owner@s.whatsapp.net",
                "friend@s.whatsapp.net",
                "Owner",
                "m2",
                "I choose the smaller experiment first.",
            ),
            (
                1,
                0,
                60,
                "friend@s.whatsapp.net",
                "owner@s.whatsapp.net",
                "Friend",
                "m1",
                "Which option do you prefer?",
            ),
        ],
    )
    connection.commit()
    connection.close()


def test_wacli_source_requires_explicit_chat_selection(tmp_path):
    database = tmp_path / "wacli.db"
    _create_wacli_db(database)
    source = WacliSource(database, pseudonym_key=KEY)

    with pytest.raises(ValueError, match="explicit chat selection"):
        source.read_messages()

    with pytest.raises(ValueError, match="cannot combine"):
        source.read_messages(chat_ids=["friend@s.whatsapp.net"], all_chats=True)


def test_wacli_source_is_read_only_and_normalizes_chronologically(tmp_path):
    database = tmp_path / "wacli.db"
    _create_wacli_db(database)
    before = _sha256(database)
    source = WacliSource(database, pseudonym_key=KEY)

    chats = source.list_chats()
    messages = source.read_messages(chat_ids=["friend@s.whatsapp.net"])

    assert _sha256(database) == before
    assert [chat.source_chat_id for chat in chats] == [
        "work@g.us",
        "friend@s.whatsapp.net",
    ]
    assert [message.source_message_id for message in messages] == ["m1", "m2"]
    assert [message.from_owner for message in messages] == [False, True]
    assert messages[1].text == "I choose the smaller experiment first."
    assert messages[1].timestamp == datetime.fromtimestamp(1_700_000_020, timezone.utc)
    assert "friend@s.whatsapp.net" not in json.dumps(messages[0].to_dict())
    assert messages[0].chat_id == messages[1].chat_id
    assert not hasattr(source, "send")
    assert not hasattr(source, "delete")


def test_whatsapp_mac_source_converts_apple_timestamps_and_joins_chat(tmp_path):
    database = tmp_path / "ChatStorage.sqlite"
    _create_whatsapp_mac_db(database)
    before = _sha256(database)
    source = WhatsAppMacSource(database, pseudonym_key=KEY)

    messages = source.read_messages(chat_ids=["friend@s.whatsapp.net"])

    assert _sha256(database) == before
    assert [message.source_message_id for message in messages] == ["m1", "m2"]
    assert messages[0].timestamp == datetime(2001, 1, 1, 0, 1, tzinfo=timezone.utc)
    assert messages[1].timestamp == datetime(2001, 1, 1, 0, 2, tzinfo=timezone.utc)
    assert messages[1].from_owner is True
    assert messages[1].source == "whatsapp_mac"


def test_whatsapp_mac_source_treats_sentinel_timestamps_as_unknown():
    assert apple_timestamp_to_datetime(10**12) is None


def test_json_source_uses_the_same_normalized_contract(tmp_path):
    source_path = tmp_path / "messages.json"
    source_path.write_text(
        json.dumps(
            {
                "chats": [
                    {
                        "source_chat_id": "friend",
                        "display_name": "Friend",
                        "kind": "dm",
                    }
                ],
                "messages": [
                    {
                        "source_message_id": "message-1",
                        "source_chat_id": "friend",
                        "sender_id": "owner",
                        "timestamp": "2026-01-01T12:00:00+00:00",
                        "from_owner": True,
                        "text": "I value useful evidence.",
                        "message_type": "text",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    messages = JsonMessageSource(source_path, pseudonym_key=KEY).read_messages(
        all_chats=True
    )

    assert len(messages) == 1
    assert isinstance(messages[0], NormalizedMessage)
    assert messages[0].source == "json"
    assert messages[0].from_owner is True
    assert messages[0].content_hash == hashlib.sha256(
        b"I value useful evidence."
    ).hexdigest()


def test_all_chats_is_an_explicit_multi_chat_opt_in(tmp_path):
    database = tmp_path / "wacli.db"
    _create_wacli_db(database)

    messages = WacliSource(database, pseudonym_key=KEY).read_messages(all_chats=True)

    assert len({message.chat_id for message in messages}) == 2
    assert sum(message.from_owner for message in messages) == 2
