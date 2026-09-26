import os
import psycopg2
from psycopg2.extras import RealDictCursor
from psycopg2 import pool
from dotenv import load_dotenv

load_dotenv()

# ── Configuration depuis .env ────────────────────────────────
DB_CONFIG = {
    'host'    : os.getenv('DB_HOST',     'localhost'),
    'port'    : int(os.getenv('DB_PORT', 5432)),
    'database': os.getenv('DB_NAME',     'artify_db'),
    'user'    : os.getenv('DB_USER',     'postgres'),
    'password': os.getenv('DB_PASSWORD', 'postgres'),
}

# ── Connection pool (évite d'ouvrir une connexion à chaque requête) ──
connection_pool = None

def init_pool():
    global connection_pool
    try:
        connection_pool = pool.SimpleConnectionPool(1, 10, **DB_CONFIG)
        print("✅ PostgreSQL — pool de connexions initialisé")
    except Exception as e:
        print(f"❌ PostgreSQL — impossible de se connecter : {e}")
        connection_pool = None

def get_db_connection():
    if connection_pool is None:
        raise Exception("Pool de connexions non initialisé")
    return connection_pool.getconn()

def release_connection(conn):
    if connection_pool and conn:
        connection_pool.putconn(conn)

# ── Création des tables ──────────────────────────────────────
def create_tables():
    conn = get_db_connection()
    try:
        cur = conn.cursor()

        # Table utilisateurs
        cur.execute("""
            CREATE TABLE IF NOT EXISTS users (
                id         SERIAL PRIMARY KEY,
                username   VARCHAR(100) NOT NULL,
                email      VARCHAR(100) UNIQUE NOT NULL,
                password   VARCHAR(255) NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table historique des analyses
        cur.execute("""
            CREATE TABLE IF NOT EXISTS analyses (
                id         SERIAL PRIMARY KEY,
                user_id    INTEGER REFERENCES users(id) ON DELETE CASCADE,
                type       VARCHAR(20) NOT NULL,  -- 'artify', 'insight', 'styleme'
                result     TEXT,                   -- JSON stringifié du résultat
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        conn.commit()
        cur.close()
        print("✅ Tables 'users' et 'analyses' prêtes")
    except Exception as e:
        conn.rollback()
        print(f"❌ Erreur création tables : {e}")
    finally:
        release_connection(conn)

# ── Users ────────────────────────────────────────────────────
def insert_user(username, email, password):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO users (username, email, password) VALUES (%s, %s, %s) RETURNING id",
            (username, email, password)
        )
        user_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return user_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_connection(conn)

def get_user_by_email(email):
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT * FROM users WHERE email = %s", (email,))
        user = cur.fetchone()
        cur.close()
        return user
    finally:
        release_connection(conn)

def get_user_by_id(user_id):
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        cur.execute("SELECT id, username, email, created_at FROM users WHERE id = %s", (user_id,))
        user = cur.fetchone()
        cur.close()
        return user
    finally:
        release_connection(conn)

# ── Historique analyses ──────────────────────────────────────
def save_analysis(user_id, analysis_type, result_json):
    conn = get_db_connection()
    try:
        cur = conn.cursor()
        cur.execute(
            "INSERT INTO analyses (user_id, type, result) VALUES (%s, %s, %s) RETURNING id",
            (user_id, analysis_type, result_json)
        )
        analysis_id = cur.fetchone()[0]
        conn.commit()
        cur.close()
        return analysis_id
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        release_connection(conn)

def get_user_analyses(user_id, analysis_type=None, limit=20):
    conn = get_db_connection()
    try:
        cur = conn.cursor(cursor_factory=RealDictCursor)
        if analysis_type:
            cur.execute(
                "SELECT * FROM analyses WHERE user_id = %s AND type = %s ORDER BY created_at DESC LIMIT %s",
                (user_id, analysis_type, limit)
            )
        else:
            cur.execute(
                "SELECT * FROM analyses WHERE user_id = %s ORDER BY created_at DESC LIMIT %s",
                (user_id, limit)
            )
        rows = cur.fetchall()
        cur.close()
        return rows
    finally:
        release_connection(conn)
