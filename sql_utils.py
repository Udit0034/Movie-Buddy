import sqlite3
import hashlib
from datetime import datetime, UTC
import pickle
import numpy as np
from scipy.sparse import csr_matrix

DB_PATH = "user.db"

def get_connection():
    return sqlite3.connect(DB_PATH)

def create_tables():
    conn = get_connection()
    cur = conn.cursor()

    cur.execute("""
        CREATE TABLE IF NOT EXISTS users (
            user_id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            dob TEXT
        )
    """)

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_preferences (
            user_id INTEGER PRIMARY KEY,
            profile_vector BLOB,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # Check if the profile_vector column exists, add it if not
    cur.execute("PRAGMA table_info(user_preferences)")
    pref_columns = [row[1] for row in cur.fetchall()]
    if 'profile_vector' not in pref_columns:
        print("MIGRATION: Adding 'profile_vector' column to 'user_preferences'")
        cur.execute("ALTER TABLE user_preferences ADD COLUMN profile_vector BLOB")

    cur.execute("""
        CREATE TABLE IF NOT EXISTS user_movie_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            tmdb_id INTEGER,
            rating REAL,
            liked INTEGER DEFAULT 0,
            watched_on TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (user_id) REFERENCES users(user_id)
        )
    """)
    
    # --- MIGRATION LOGIC ---
    cur.execute("PRAGMA table_info(user_movie_history)")
    columns = [row[1] for row in cur.fetchall()]
    
    if 'tmdb_id' not in columns:
        print("MIGRATION: Adding 'tmdb_id' column...")
        cur.execute("ALTER TABLE user_movie_history ADD COLUMN tmdb_id INTEGER")
    if 'rating' not in columns:
        print("MIGRATION: Adding 'rating' column...")
        cur.execute("ALTER TABLE user_movie_history ADD COLUMN rating REAL")
        
    # --- THIS IS THE FIX ---
    # Add the missing migration logic for created_at
    if 'created_at' not in columns:
        print("MIGRATION: Adding 'created_at' column...")
        # Cannot add a non-constant default to an existing table
        # We just add the column. New rows will get the default from the
        # CREATE TABLE definition, and old rows will have NULL (which is fine).
        cur.execute("ALTER TABLE user_movie_history ADD COLUMN created_at TEXT")
    # --- END OF FIX ---

    conn.commit()
    conn.close()

def hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()

def signup_user(username, password, dob):
    conn = get_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            "INSERT INTO users (username, password_hash, dob) VALUES (?, ?, ?)",
            (username, hash_password(password), dob),
        )
        conn.commit()
        return True, "✅ User registered successfully!"
    except sqlite3.IntegrityError:
        return False, "⚠️ Username already exists."
    finally:
        conn.close()

def login_user(username, password):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT password_hash FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return bool(row and row[0] == hash_password(password))

# ===== Additional helpers for app =====
def get_user_id(username):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT user_id FROM users WHERE username = ?", (username,))
    row = cur.fetchone()
    conn.close()
    return row[0] if row else None

def add_user_movie_history(user_id, tmdb_id, rating=None, liked=0, watched_on=None):
    conn = get_connection()
    cur = conn.cursor()
    if watched_on is None:
        watched_on = datetime.now(UTC).isoformat()
    
    # The created_at column will be populated by its default value
    cur.execute("""
        INSERT INTO user_movie_history (user_id, tmdb_id, rating, liked, watched_on)
        VALUES (?, ?, ?, ?, ?)
    """, (user_id, tmdb_id, rating, int(bool(liked)), watched_on))
    conn.commit()
    conn.close()
    return True

def get_user_history(user_id, limit=200):
    conn = get_connection()
    cur = conn.cursor()
    # This query will now work because the migration logic in create_tables()
    # will add the 'created_at' column (as TEXT, allowing NULLs)
    cur.execute("""
        SELECT tmdb_id, rating, liked, watched_on, created_at
        FROM user_movie_history
        WHERE user_id = ?
        ORDER BY created_at DESC
        LIMIT ?
    """, (user_id, limit))
    rows = cur.fetchall()
    conn.close()
    return rows

# --- NEW FUNCTIONS FOR ML PROFILE ---

def get_user_profile_vector(user_id):
    """
    Fetches and deserializes the user's profile vector from the DB.
    """
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT profile_vector FROM user_preferences WHERE user_id = ?", (user_id,))
    row = cur.fetchone()
    conn.close()
    if row and row[0]:
        try:
            return pickle.loads(row[0])
        except Exception as e:
            print(f"Error loading profile vector, may be corrupt: {e}")
            return None
    return None

def save_user_profile_vector(user_id, vector):
    """
    Serializes and saves the user's profile vector to the DB.
    """
    conn = get_connection()
    cur = conn.cursor()
    # Serialize the sparse matrix
    serialized_vector = pickle.dumps(vector)
    # Use INSERT OR REPLACE to create or update the user's row
    cur.execute("""
        INSERT OR REPLACE INTO user_preferences (user_id, profile_vector)
        VALUES (?, ?)
    """, (user_id, serialized_vector))
    conn.commit()
    conn.close()
    return True


if __name__ == "__main__":
    print("Initializing DB (user.db) and tables if not present...")
    create_tables()
    print("Done.")

