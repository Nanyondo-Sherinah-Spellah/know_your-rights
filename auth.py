import re
import sqlite3
from pathlib import Path

from argon2 import PasswordHasher
from argon2.exceptions import (
    InvalidHashError,
    VerificationError,
    VerifyMismatchError,
)


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

DB_PATH = Path("data/chat_history.db")

password_hasher = PasswordHasher()

EMAIL_PATTERN = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


# --------------------------------------------------
# DATABASE CONNECTION
# --------------------------------------------------

def get_auth_connection():
    """Open a connection to the application database."""
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")

    return connection


def initialize_auth_database():
    """Create the users table if it does not already exist."""
    with get_auth_connection() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                email TEXT NOT NULL COLLATE NOCASE UNIQUE,
                password_hash TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )


# --------------------------------------------------
# INPUT VALIDATION
# --------------------------------------------------

def validate_email(email: str) -> str:
    """Validate an email address and return its normalised form."""
    if not isinstance(email, str):
        raise ValueError("Enter a valid email address.")

    email = email.strip().lower()

    if len(email) > 254 or not EMAIL_PATTERN.fullmatch(email):
        raise ValueError("Enter a valid email address.")

    return email


def validate_password(password: str) -> str:
    """Validate password length."""
    if not isinstance(password, str):
        raise ValueError("Password must be text.")

    if len(password) < 12:
        raise ValueError(
            "Password must contain at least 12 characters."
        )

    if len(password) > 128:
        raise ValueError("Password is too long.")

    return password


# --------------------------------------------------
# USER REGISTRATION
# --------------------------------------------------

def register_user(email: str, password: str) -> dict:
    """
    Register a new user.

    Returns:
        A dictionary containing the user's ID and email.

    Raises:
        ValueError: If the inputs are invalid or the email is registered.
    """
    email = validate_email(email)
    password = validate_password(password)

    password_hash = password_hasher.hash(password)

    try:
        with get_auth_connection() as connection:
            cursor = connection.execute(
                """
                INSERT INTO users (email, password_hash)
                VALUES (?, ?)
                """,
                (email, password_hash),
            )

            user_id = cursor.lastrowid

            if user_id is None:
                raise RuntimeError(
                    "Could not create the user account."
                )

            return {
                "id": user_id,
                "email": email,
            }

    except sqlite3.IntegrityError as error:
        if "users.email" in str(error).lower():
            raise ValueError(
                "An account with that email may already exist."
            ) from None

        raise


# --------------------------------------------------
# USER AUTHENTICATION
# --------------------------------------------------

def authenticate_user(email: str, password: str):
    """
    Verify credentials.

    Returns:
        User ID and email on success; None on failure.
    """
    if not isinstance(password, str):
        return None

    try:
        email = validate_email(email)
    except ValueError:
        return None

    with get_auth_connection() as connection:
        user = connection.execute(
            """
            SELECT id, email, password_hash
            FROM users
            WHERE email = ?
            """,
            (email,),
        ).fetchone()

    if user is None:
        return None

    stored_hash = user["password_hash"]

    try:
        password_hasher.verify(stored_hash, password)
    except (
        VerifyMismatchError,
        VerificationError,
        InvalidHashError,
    ):
        return None

    # Upgrade older hashes when Argon2 settings change.
    if password_hasher.check_needs_rehash(stored_hash):
        new_hash = password_hasher.hash(password)

        with get_auth_connection() as connection:
            connection.execute(
                """
                UPDATE users
                SET password_hash = ?
                WHERE id = ?
                """,
                (new_hash, user["id"]),
            )

    return {
        "id": user["id"],
        "email": user["email"],
    }