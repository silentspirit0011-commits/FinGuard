"""
FinGuard - Intelligent Financial Transaction Monitoring & Anomaly Detection System
Module: Relational Database Layer (SQLite)
Manages schemas, constraints, CRUD operations, and core analytical SQL procedures.
"""

import sqlite3
import pandas as pd
import os

DB_PATH = os.path.join("data", "finguard.db")

def get_connection(db_path=DB_PATH):
    """
    Establishes connection to SQLite database and enforces foreign key constraints.
    """
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn

def init_db(db_path=DB_PATH):
    """
    Initializes the relational database schema according to specifications.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    # 1. Customers Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS customers (
        customer_id TEXT PRIMARY KEY,
        customer_name TEXT NOT NULL,
        age INTEGER,
        customer_segment TEXT NOT NULL,
        account_type TEXT NOT NULL,
        account_balance REAL NOT NULL
    );
    """)
    
    # 2. Transactions Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS transactions (
        transaction_id TEXT PRIMARY KEY,
        customer_id TEXT NOT NULL,
        transaction_date TEXT NOT NULL,
        transaction_amount REAL NOT NULL,
        transaction_type TEXT NOT NULL,
        merchant_category TEXT,
        payment_method TEXT,
        location TEXT,
        account_balance REAL,
        is_injected_anomaly INTEGER DEFAULT 0,
        FOREIGN KEY (customer_id) REFERENCES customers(customer_id) ON DELETE CASCADE
    );
    """)
    
    # 3. Anomaly Results Table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS anomaly_results (
        result_id INTEGER PRIMARY KEY AUTOINCREMENT,
        transaction_id TEXT NOT NULL UNIQUE,
        anomaly_score REAL NOT NULL,
        anomaly_flag INTEGER NOT NULL,
        risk_category TEXT NOT NULL,
        model_version TEXT NOT NULL,
        evaluated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (transaction_id) REFERENCES transactions(transaction_id) ON DELETE CASCADE
    );
    """)
    
    # Create indexes for optimal analytical query performance
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_date ON transactions(transaction_date);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_txn_cust ON transactions(customer_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_txn ON anomaly_results(transaction_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_anomaly_risk ON anomaly_results(risk_category);")
    
    conn.commit()
    conn.close()
    print(f"[Database] SQLite schema verified and initialized at {db_path}")

def insert_customers(df_customers, db_path=DB_PATH):
    """
    Insert customer records with duplicate handling.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cols = ["customer_id", "customer_name", "age", "customer_segment", "account_type", "account_balance"]
    data = df_customers[cols].to_dict(orient="records")
    
    cursor.executemany("""
    INSERT OR REPLACE INTO customers (customer_id, customer_name, age, customer_segment, account_type, account_balance)
    VALUES (:customer_id, :customer_name, :age, :customer_segment, :account_type, :account_balance);
    """, data)
    
    conn.commit()
    count = cursor.rowcount
    conn.close()
    print(f"[Database] Upserted {len(data)} customer records.")

def insert_transactions(df_transactions, db_path=DB_PATH):
    """
    Insert transaction records with duplicate handling.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cols = [
        "transaction_id", "customer_id", "transaction_date", "transaction_amount",
        "transaction_type", "merchant_category", "payment_method", "location",
        "account_balance", "is_injected_anomaly"
    ]
    # Ensure date strings
    df_copy = df_transactions.copy()
    if pd.api.types.is_datetime64_any_dtype(df_copy["transaction_date"]):
        df_copy["transaction_date"] = df_copy["transaction_date"].dt.strftime("%Y-%m-%d %H:%M:%S")
        
    data = df_copy[cols].to_dict(orient="records")
    
    cursor.executemany("""
    INSERT OR REPLACE INTO transactions (
        transaction_id, customer_id, transaction_date, transaction_amount,
        transaction_type, merchant_category, payment_method, location,
        account_balance, is_injected_anomaly
    ) VALUES (
        :transaction_id, :customer_id, :transaction_date, :transaction_amount,
        :transaction_type, :merchant_category, :payment_method, :location,
        :account_balance, :is_injected_anomaly
    );
    """, data)
    
    conn.commit()
    conn.close()
    print(f"[Database] Upserted {len(data)} transaction records.")

def insert_anomaly_results(df_results, db_path=DB_PATH):
    """
    Insert model anomaly inference results with duplicate replacement.
    """
    conn = get_connection(db_path)
    cursor = conn.cursor()
    
    cols = ["transaction_id", "anomaly_score", "anomaly_flag", "risk_category", "model_version"]
    data = df_results[cols].to_dict(orient="records")
    
    cursor.executemany("""
    INSERT OR REPLACE INTO anomaly_results (
        transaction_id, anomaly_score, anomaly_flag, risk_category, model_version
    ) VALUES (
        :transaction_id, :anomaly_score, :anomaly_flag, :risk_category, :model_version
    );
    """, data)
    
    conn.commit()
    conn.close()
    print(f"[Database] Upserted {len(data)} anomaly prediction records.")

def query_dataframe(query, params=None, db_path=DB_PATH):
    """
    Executes a SQL query and returns results as a pandas DataFrame.
    """
    conn = get_connection(db_path)
    try:
        if params:
            df = pd.read_sql_query(query, conn, params=params)
        else:
            df = pd.read_sql_query(query, conn)
        return df
    finally:
        conn.close()

def get_complete_monitoring_records(db_path=DB_PATH):
    """
    Returns full joined view of transactions, customers, and anomaly results.
    """
    query = """
    SELECT 
        t.transaction_id,
        t.customer_id,
        c.customer_name,
        c.customer_segment,
        c.account_type,
        t.transaction_date,
        t.transaction_amount,
        t.transaction_type,
        t.merchant_category,
        t.payment_method,
        t.location,
        t.account_balance,
        t.is_injected_anomaly,
        COALESCE(ar.anomaly_score, 0.0) AS anomaly_score,
        COALESCE(ar.anomaly_flag, 0) AS anomaly_flag,
        COALESCE(ar.risk_category, 'Low Risk') AS risk_category,
        COALESCE(ar.model_version, 'N/A') AS model_version
    FROM transactions t
    JOIN customers c ON t.customer_id = c.customer_id
    LEFT JOIN anomaly_results ar ON t.transaction_id = ar.transaction_id
    ORDER BY t.transaction_date DESC;
    """
    return query_dataframe(query, db_path=db_path)

