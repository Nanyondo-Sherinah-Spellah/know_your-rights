
import json
import sqlite3
from pathlib import Path


DB_PATH = Path("data/chat_history.db")


def get_connection():
    """Open a connection with foreign-key enforcement enabled."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def initialize_database():
    """Create user-owned conversation and message tables."""
    with get_connection() as connection:
        connection.execute("""
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                title TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users(id)
                    ON DELETE CASCADE
            )
        """)

        connection.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role TEXT NOT NULL CHECK(role IN ('user', 'assistant')),
                content TEXT NOT NULL,
                model TEXT,
                sources TEXT NOT NULL DEFAULT '[]',
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (conversation_id) REFERENCES conversations(id)
                    ON DELETE CASCADE
            )
        """)


def _owned_conversation(connection, user_id, conversation_id):
    """Return a conversation only if it belongs to the given user."""
    return connection.execute("""
        SELECT id FROM conversations
        WHERE id = ? AND user_id = ?
    """, (conversation_id, user_id)).fetchone()


def create_conversation(user_id, title="New conversation"):
    """Create a conversation owned by a user."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Conversation title cannot be empty.")

    with get_connection() as connection:
        cursor = connection.execute("""
            INSERT INTO conversations (user_id, title)
            SELECT id, ? FROM users WHERE id = ?
        """, (title.strip(), user_id))

        if cursor.rowcount == 0:
            raise ValueError("User account not found.")

        return cursor.lastrowid


def list_conversations(user_id):
    """List only conversations owned by a user."""
    with get_connection() as connection:
        rows = connection.execute("""
            SELECT id, title, created_at, updated_at
            FROM conversations
            WHERE user_id = ?
            ORDER BY updated_at DESC, id DESC
        """, (user_id,)).fetchall()

        return [dict(row) for row in rows]


def get_messages(user_id, conversation_id):
    """Read messages only from a user's own conversation."""
    with get_connection() as connection:
        if not _owned_conversation(connection, user_id, conversation_id):
            raise ValueError("Conversation not found.")

        rows = connection.execute("""
            SELECT id, role, content, model, sources, created_at
            FROM messages
            WHERE conversation_id = ?
            ORDER BY id ASC
        """, (conversation_id,)).fetchall()

        messages = []
        for row in rows:
            message = dict(row)
            try:
                message["sources"] = json.loads(message["sources"])
            except (json.JSONDecodeError, TypeError):
                message["sources"] = []
            messages.append(message)

        return messages


def save_message(
    user_id,
    conversation_id,
    role,
    content,
    model=None,
    sources=None,
):
    """Save a message only to a conversation owned by the user."""
    if role not in {"user", "assistant"}:
        raise ValueError("Role must be 'user' or 'assistant'.")

    if not isinstance(content, str):
        raise ValueError("Message content must be text.")

    serialized_sources = json.dumps(
        sources or [], ensure_ascii=False, default=str
    )

    with get_connection() as connection:
        if not _owned_conversation(connection, user_id, conversation_id):
            raise ValueError("Conversation not found.")

        connection.execute("""
            INSERT INTO messages (
                conversation_id, role, content, model, sources
            )
            VALUES (?, ?, ?, ?, ?)
        """, (conversation_id, role, content, model, serialized_sources))

        connection.execute("""
            UPDATE conversations
            SET updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
        """, (conversation_id, user_id))


def rename_conversation(user_id, conversation_id, title):
    """Rename only a conversation owned by the user."""
    if not isinstance(title, str) or not title.strip():
        raise ValueError("Conversation title cannot be empty.")

    with get_connection() as connection:
        cursor = connection.execute("""
            UPDATE conversations
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ?
        """, (title.strip(), conversation_id, user_id))

        if cursor.rowcount == 0:
            raise ValueError("Conversation not found.")


def delete_conversation(user_id, conversation_id):
    """Delete only a conversation owned by the user."""
    with get_connection() as connection:
        cursor = connection.execute("""
            DELETE FROM conversations
            WHERE id = ? AND user_id = ?
        """, (conversation_id, user_id))

        if cursor.rowcount == 0:
            raise ValueError("Conversation not found.")


def set_conversation_title_if_default(user_id, conversation_id, question):
    """Use the first question as the title of a user's new conversation."""
    if not isinstance(question, str):
        raise ValueError("Question must be text.")

    title = " ".join(question.split())
    if len(title) > 50:
        title = title[:47].rstrip() + "..."

    with get_connection() as connection:
        connection.execute("""
            UPDATE conversations
            SET title = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND user_id = ? AND title = 'New conversation'
        """, (title or "New conversation", conversation_id, user_id))
