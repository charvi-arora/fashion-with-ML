# ============================================================
# db_handler.py  —  SQLite Database (Sessions + Feedback)
# ============================================================
# Two tables:
#
#   sessions      — every prediction made (input features + predicted style)
#   feedback      — user's rating (1-5) on the recommendation
#
# Together these form the "accumulated I/O dataset" that both
# learning approaches use to improve the model over time.
# ============================================================

import sqlite3
import os
import pandas as pd
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'fashion.db')


def get_connection():
    return sqlite3.connect(DB_PATH)


def setup_database():
    """Create all tables on first run. Safe to call multiple times."""
    conn   = get_connection()
    cursor = conn.cursor()

    # -- Table 1: every prediction session --
    # Stores the 5 input features + what style the model predicted
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sessions (
            id           INTEGER PRIMARY KEY AUTOINCREMENT,
            user_name    TEXT NOT NULL,
            body_type    TEXT NOT NULL,
            skin_tone    TEXT NOT NULL,
            occasion     TEXT NOT NULL,
            outfit_type  TEXT NOT NULL,
            season       TEXT NOT NULL,
            predicted_style TEXT NOT NULL,
            recommended  TEXT NOT NULL,
            created_at   TEXT NOT NULL
        )
    """)

    # -- Table 2: optional user feedback on the prediction --
    # rating: 1 (bad) to 5 (perfect)
    # correct_style: what the user says the style SHOULD have been
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id     INTEGER NOT NULL,
            rating         INTEGER NOT NULL,
            correct_style  TEXT,
            created_at     TEXT NOT NULL,
            FOREIGN KEY (session_id) REFERENCES sessions(id)
        )
    """)

    conn.commit()
    conn.close()
    print("[DB] Database ready (sessions + feedback tables).")


# ─────────────────────────────────────────────
# WRITE
# ─────────────────────────────────────────────

def save_session(user_name, body_type, skin_tone, occasion,
                 outfit_type, season, predicted_style, recommended_outfits) -> int:
    """
    Save one prediction session. Returns the new session ID.
    """
    conn   = get_connection()
    cursor = conn.cursor()
    outfits_str = ', '.join([o.product_name for o in recommended_outfits])
    now    = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("""
        INSERT INTO sessions
          (user_name, body_type, skin_tone, occasion, outfit_type,
           season, predicted_style, recommended, created_at)
        VALUES (?,?,?,?,?,?,?,?,?)
    """, (user_name, body_type, skin_tone, occasion, outfit_type,
          season, predicted_style, outfits_str, now))

    session_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"[DB] Session #{session_id} saved for '{user_name}'.")
    return session_id


def save_feedback(session_id: int, rating: int, correct_style: str = None):
    """
    Save user feedback for a session.
    rating        : 1–5  (5 = perfect recommendation)
    correct_style : what the correct style should have been (optional)
    """
    conn   = get_connection()
    cursor = conn.cursor()
    now    = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    cursor.execute("""
        INSERT INTO feedback (session_id, rating, correct_style, created_at)
        VALUES (?,?,?,?)
    """, (session_id, rating, correct_style, now))

    conn.commit()
    conn.close()
    print(f"[DB] Feedback saved: rating={rating}, correct='{correct_style}'.")


# ─────────────────────────────────────────────
# READ
# ─────────────────────────────────────────────

def get_sessions_as_dataframe() -> pd.DataFrame:
    """
    Load all saved sessions as a Pandas DataFrame.
    This is what learning approaches use as extra training data.
    """
    conn = get_connection()
    df   = pd.read_sql_query("SELECT * FROM sessions ORDER BY created_at", conn)
    conn.close()
    return df


def get_feedback_as_dataframe() -> pd.DataFrame:
    """
    Load all feedback joined with sessions.
    Includes the 'correct_style' column — the true label from the user.
    """
    conn = get_connection()
    df   = pd.read_sql_query("""
        SELECT s.body_type, s.skin_tone, s.occasion, s.outfit_type, s.season,
               f.correct_style AS style_tag, f.rating
        FROM   feedback f
        JOIN   sessions s ON f.session_id = s.id
        WHERE  f.correct_style IS NOT NULL
        ORDER  BY f.created_at
    """, conn)
    conn.close()
    return df


def count_sessions() -> int:
    conn    = get_connection()
    cursor  = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM sessions")
    n       = cursor.fetchone()[0]
    conn.close()
    return n


def print_session_history():
    conn = get_connection()
    df   = pd.read_sql_query(
        "SELECT id, user_name, body_type, skin_tone, occasion, predicted_style, created_at "
        "FROM sessions ORDER BY created_at DESC LIMIT 10", conn
    )
    conn.close()
    if df.empty:
        print("[DB] No sessions yet.")
        return
    print("\n--- Past Sessions (last 10) ---")
    for _, r in df.iterrows():
        print(f"  #{r['id']:>3} | {r['user_name']:<10} | "
              f"{r['body_type']}, {r['skin_tone']}, {r['occasion']} "
              f"→ {r['predicted_style']} | {r['created_at']}")
