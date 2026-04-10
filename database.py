"""
database.py — SQLite database for customers and claims.

Tables:  customers, claims
Seeds 5 demo customers on first run.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "fraud_detection.db")


def _conn():
    """Return a connection with row_factory = sqlite3.Row."""
    con = sqlite3.connect(DB_PATH)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    return con


# ─────────────────────────────────────────────────────────────────
# Schema
# ─────────────────────────────────────────────────────────────────
_CREATE_CUSTOMERS = """
CREATE TABLE IF NOT EXISTS customers (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    full_name           TEXT    NOT NULL,
    age                 INTEGER NOT NULL,
    phone               TEXT    NOT NULL UNIQUE,
    password            TEXT    DEFAULT '1234',  -- Password for login (NEW)
    email               TEXT,
    address             TEXT,
    vehicle_type        TEXT,            -- SUV/Sedan etc
    vehicle_usage       TEXT DEFAULT 'Private', -- Private/Taxi
    vehicle_model       TEXT,
    registration_number TEXT,
    manufacturing_date  TEXT,            -- Calendar date (NEW)
    vehicle_age         INTEGER,         -- years
    policy_start_date   TEXT,
    policy_end_date     TEXT,
    license_number      TEXT,
    license_valid       TEXT DEFAULT 'Yes',
    past_claims_count   INTEGER DEFAULT 0,
    created_at          TEXT    NOT NULL
);
"""

_CREATE_CLAIMS = """
CREATE TABLE IF NOT EXISTS claims (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    customer_id       INTEGER NOT NULL REFERENCES customers(id),
    accident_date     TEXT,
    claim_date        TEXT,
    claim_delay_days  INTEGER,
    policy_age_days   INTEGER,
    fir_filed         INTEGER DEFAULT 0,
    fir_file_path     TEXT,                  -- Path to uploaded FIR pdf/img (NEW)
    damage_severity   TEXT,                  -- Minor / Major
    accident_type     TEXT,
    location_type     TEXT,                  -- Urban / Rural / Highway
    fraud_prediction  TEXT,                  -- Fraud / Non-Fraud
    fraud_confidence  REAL,
    image_fraud_count INTEGER DEFAULT 0,
    image_total_count INTEGER DEFAULT 0,
    image_descriptions TEXT,                 -- User text for each image (NEW)
    risk_score        INTEGER DEFAULT 0,     -- 0-100 logic (NEW)
    remarks           TEXT,                  -- Accept/Reject notes (NEW)
    verdict           TEXT,
    status            TEXT DEFAULT 'Submitted',
    image_paths       TEXT,
    is_analyzed       INTEGER DEFAULT 0,
    created_at        TEXT NOT NULL
);
"""

_SEED_CUSTOMERS = [
    ("Rajesh Kumar",  35, "9876543210", "rajesh.kumar@email.com",
     "12 MG Road, Bengaluru", "Private", "Maruti Swift",
     "KA-01-AB-1234", 3, "2023-06-15", "2026-06-15", "DL-KA-1234567", "Yes", 1),
    ("Priya Sharma",  28, "9123456789", "priya.sharma@email.com",
     "45 Park Street, Kolkata", "Private", "Hyundai i20",
     "WB-06-CD-5678", 2, "2024-01-10", "2027-01-10", "DL-WB-7654321", "Yes", 0),
    ("Anil Mehta",    42, "9988776655", "anil.mehta@email.com",
     "78 FC Road, Pune", "Taxi", "Toyota Innova",
     "MH-12-EF-9012", 6, "2022-03-20", "2025-03-20", "DL-MH-1122334", "Yes", 3),
    ("Sunita Rao",    31, "9012345678", "sunita.rao@email.com",
     "23 Anna Salai, Chennai", "Private", "Honda City",
     "TN-09-GH-3456", 4, "2023-11-05", "2026-11-05", "DL-TN-5566778", "Yes", 0),
    ("Vikram Singh",  50, "9876501234", "vikram.singh@email.com",
     "9 Civil Lines, Jaipur", "Private", "Tata Nexon",
     "RJ-14-IJ-7890", 1, "2024-08-01", "2027-08-01", "DL-RJ-9988776", "Yes", 2),
]


# ─────────────────────────────────────────────────────────────────
# Init
# ─────────────────────────────────────────────────────────────────
def init_db():
    """Create tables, run migrations, seed demo customers."""
    con = _conn()
    con.execute(_CREATE_CUSTOMERS)
    con.execute(_CREATE_CLAIMS)
    con.commit()

    # ── Migrate existing DB: add new columns if missing ──────────
    def _col_exists(table, column):
        cols = {row[1] for row in con.execute(f"PRAGMA table_info({table})")}
        return column in cols

    # Table: customers
    if not _col_exists("customers", "password"):
        con.execute("ALTER TABLE customers ADD COLUMN password TEXT DEFAULT '1234'")
    if not _col_exists("customers", "manufacturing_date"):
        con.execute("ALTER TABLE customers ADD COLUMN manufacturing_date TEXT")
    if not _col_exists("customers", "policy_end_date"):
        con.execute("ALTER TABLE customers ADD COLUMN policy_end_date TEXT")
    if not _col_exists("customers", "license_number"):
        con.execute("ALTER TABLE customers ADD COLUMN license_number TEXT")
    if not _col_exists("customers", "license_valid"):
        con.execute("ALTER TABLE customers ADD COLUMN license_valid TEXT DEFAULT 'Yes'")
    
    # Table: claims
    if not _col_exists("claims", "fir_file_path"):
        con.execute("ALTER TABLE claims ADD COLUMN fir_file_path TEXT")
    if not _col_exists("claims", "image_descriptions"):
        con.execute("ALTER TABLE claims ADD COLUMN image_descriptions TEXT")
    if not _col_exists("claims", "risk_score"):
        con.execute("ALTER TABLE claims ADD COLUMN risk_score INTEGER DEFAULT 0")
    if not _col_exists("claims", "remarks"):
        con.execute("ALTER TABLE claims ADD COLUMN remarks TEXT")
    if not _col_exists("claims", "status"):
        con.execute("ALTER TABLE claims ADD COLUMN status TEXT DEFAULT 'Submitted'")
    if not _col_exists("claims", "image_paths"):
        con.execute("ALTER TABLE claims ADD COLUMN image_paths TEXT")
    if not _col_exists("claims", "is_analyzed"):
        con.execute("ALTER TABLE claims ADD COLUMN is_analyzed INTEGER DEFAULT 0")
    
    con.commit()

    # ── Seed demo customers only when table is empty ─────────────
    count = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    if count == 0:
        now = datetime.now().isoformat()
        for row in _SEED_CUSTOMERS:
            con.execute(
                """INSERT INTO customers
                   (full_name, age, phone, email, address,
                    vehicle_type, vehicle_model, registration_number,
                    vehicle_age, policy_start_date, policy_end_date,
                    license_number, license_valid, past_claims_count,
                    created_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (*row, now),
            )
        con.commit()
    con.close()


# ─────────────────────────────────────────────────────────────────
# CRUD helpers
# ─────────────────────────────────────────────────────────────────
def get_customer_by_phone(phone: str):
    """Return customer dict or None."""
    con = _conn()
    row = con.execute(
        "SELECT * FROM customers WHERE phone = ?", (phone.strip(),)
    ).fetchone()
    con.close()
    return dict(row) if row else None


def get_all_customers():
    con = _conn()
    rows = con.execute("SELECT * FROM customers ORDER BY id").fetchall()
    con.close()
    return [dict(r) for r in rows]


def insert_customer(data: dict) -> int:
    """Insert a new customer and return the new id."""
    con = _conn()
    cur = con.execute(
        """INSERT INTO customers
           (full_name, age, phone, password, email, address,
            vehicle_type, vehicle_model, registration_number,
            manufacturing_date, vehicle_age, policy_start_date, policy_end_date,
            license_number, license_valid, past_claims_count,
            created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            data["full_name"], data["age"], data["phone"],
            data.get("password", "1234"),
            data.get("email", ""),
            data.get("address", ""),
            data.get("vehicle_type", "Private"),
            data.get("vehicle_model", ""),
            data.get("registration_number", ""),
            data.get("manufacturing_date", ""),
            data.get("vehicle_age", 0),
            data.get("policy_start_date", ""),
            data.get("policy_end_date", ""),
            data.get("license_number", ""),
            data.get("license_valid", "Yes"),
            data.get("past_claims_count", 0),
            datetime.now().isoformat(),
        ),
    )
    con.commit()
    cid = cur.lastrowid
    con.close()
    return cid


def insert_claim(data: dict) -> int:
    """Insert a claim record and return its id."""
    con = _conn()
    cur = con.execute(
        """INSERT INTO claims
           (customer_id, accident_date, claim_date,
            claim_delay_days, policy_age_days,
            fir_filed, fir_file_path, damage_severity,
            accident_type, location_type,
            fraud_prediction, fraud_confidence,
            image_fraud_count, image_total_count, 
            image_descriptions, risk_score, remarks,
            verdict, status, image_paths, is_analyzed,
            created_at)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (
            data["customer_id"],
            data.get("accident_date", ""),
            data.get("claim_date", ""),
            data.get("claim_delay_days", 0),
            data.get("policy_age_days", 0),
            1 if data.get("fir_filed") else 0,
            data.get("fir_file_path", ""),
            data.get("damage_severity", ""),
            data.get("accident_type", ""),
            data.get("location_type", ""),
            data.get("fraud_prediction", ""),
            data.get("fraud_confidence", 0.0),
            data.get("image_fraud_count", 0),
            data.get("image_total_count", 0),
            data.get("image_descriptions", ""),
            data.get("risk_score", 0),
            data.get("remarks", ""),
            data.get("verdict", ""),
            data.get("status", "Submitted"),
            data.get("image_paths", ""),
            1 if data.get("is_analyzed") else 0,
            datetime.now().isoformat(),
        ),
    )
    con.commit()
    cid = cur.lastrowid
    con.close()
    return cid


def get_claims_for_customer(customer_id: int):
    con = _conn()
    rows = con.execute(
        "SELECT * FROM claims WHERE customer_id = ? ORDER BY id DESC",
        (customer_id,),
    ).fetchall()
    con.close()
    return [dict(r) for r in rows]


def increment_past_claims(customer_id: int):
    """Increment past_claims_count by 1 after each claim submission."""
    con = _conn()
    con.execute(
        "UPDATE customers SET past_claims_count = past_claims_count + 1 WHERE id = ?",
        (customer_id,),
    )
    con.commit()
    con.close()

def update_claim_status(claim_id: int, new_status: str):
    """Update only the status of a specific claim, e.g. from Approved to Closed/Settled."""
    con = _conn()
    con.execute(
        "UPDATE claims SET status = ? WHERE id = ?",
        (new_status, claim_id)
    )
    con.commit()
    con.close()


def update_claim_analysis(claim_id: int, data: dict):
    """Update a claim with AI results, risk score, and final verdict."""
    con = _conn()
    con.execute(
        """UPDATE claims SET 
           fraud_prediction = ?, 
           fraud_confidence = ?, 
           image_fraud_count = ?, 
           image_total_count = ?, 
           risk_score = ?,
           remarks = ?,
           verdict = ?, 
           status = ?, 
           is_analyzed = 1 
           WHERE id = ?""",
        (
            data["fraud_prediction"],
            data["fraud_confidence"],
            data["image_fraud_count"],
            data["image_total_count"],
            data.get("risk_score", 0),
            data.get("remarks", ""),
            data["verdict"],
            data.get("status", "Analyzed"),
            claim_id
        ),
    )
    con.commit()
    con.close()


def get_all_claims():
    """Return all claims, most recent first."""
    con = _conn()
    rows = con.execute("SELECT * FROM claims ORDER BY id DESC").fetchall()
    con.close()
    return [dict(r) for r in rows]


# ─────────────────────────────────────────────────────────────────
# Standalone test
# ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    init_db()
    print(f"Database: {DB_PATH}")
    for c in get_all_customers():
        print(f"  {c['id']:2d}  {c['full_name']:<18s}  📞 {c['phone']}  "
              f"DL: {c.get('license_number','—')}  "
              f"Policy: {c.get('policy_start_date','—')} → {c.get('policy_end_date','—')}")
