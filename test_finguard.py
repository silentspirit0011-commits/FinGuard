"""
FinGuard - Intelligent Financial Transaction Monitoring & Anomaly Detection System
Module: Automated Verification & Unit/Integration Tests
Validates database schema, foreign keys, feature pipeline, Isolation Forest, and analytical queries.
"""

import unittest
import os
import sqlite3
import pandas as pd
import numpy as np

from database import (
    get_connection,
    init_db,
    insert_customers,
    insert_transactions,
    insert_anomaly_results,
    get_complete_monitoring_records,
    DB_PATH
)
from preprocessing import DataPreprocessor, preprocess_pipeline
from anomaly_detector import AnomalyDetectionEngine, MODEL_PATH
from analytics import QUERIES, execute_analytics_query, execute_custom_sql

class TestFinGuardPlatform(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        cls.db_path = DB_PATH
        cls.assertTrue(os.path.exists(cls.db_path), "Database finguard.db should exist.")

    def test_01_database_tables_exist(self):
        """Verify customers, transactions, and anomaly_results tables exist."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        self.assertIn("customers", tables)
        self.assertIn("transactions", tables)
        self.assertIn("anomaly_results", tables)
        print("✓ Test 1 Passed: Core SQLite tables verified.")

    def test_02_foreign_key_enforcement(self):
        """Verify foreign key constraint rejects transactions with non-existent customer_id."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        # Attempt to insert a transaction with a fake customer ID
        with self.assertRaises(sqlite3.IntegrityError):
            cursor.execute("""
            INSERT INTO transactions (
                transaction_id, customer_id, transaction_date, transaction_amount, transaction_type
            ) VALUES ('TXN-INVALID-999', 'CUST-NONEXISTENT', '2026-01-01 10:00:00', 500.0, 'Debit');
            """)
        conn.close()
        print("✓ Test 2 Passed: SQLite foreign key enforcement strictly verified.")

    def test_03_data_integrity_and_records(self):
        """Verify record counts match synthetic target specifications."""
        conn = get_connection(self.db_path)
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM customers;")
        cust_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM transactions;")
        txn_count = cursor.fetchone()[0]
        cursor.execute("SELECT COUNT(*) FROM anomaly_results;")
        anom_count = cursor.fetchone()[0]
        conn.close()
        
        self.assertGreaterEqual(cust_count, 300)
        self.assertGreaterEqual(txn_count, 5000)
        self.assertEqual(txn_count, anom_count)
        print(f"✓ Test 3 Passed: Ingested {cust_count} customers, {txn_count} transactions, {anom_count} anomaly scores.")

    def test_04_preprocessing_and_feature_matrix(self):
        """Verify preprocessing pipeline extracts 11 features and excludes ground-truth label."""
        df_raw = pd.read_csv(os.path.join("data", "synthetic_transactions.csv")).head(200)
        preprocessor, df_engineered = preprocess_pipeline(df_raw)
        X, feature_cols = preprocessor.get_feature_matrix(df_engineered)
        
        self.assertEqual(len(feature_cols), 11)
        self.assertNotIn("is_injected_anomaly", feature_cols)
        self.assertNotIn("transaction_id", feature_cols)
        self.assertNotIn("customer_id", feature_cols)
        self.assertFalse(X.isna().any().any(), "Feature matrix must not contain NaNs")
        print("✓ Test 4 Passed: Feature engineering pipeline produces clean feature matrix without label leakage.")

    def test_05_model_prediction_and_scoring(self):
        """Verify Isolation Forest outputs calibrated 0-100 scores and risk tiers."""
        engine = AnomalyDetectionEngine.load(MODEL_PATH)
        self.assertIsNotNone(engine, "Trained model artifact must exist.")
        
        df_raw = pd.read_csv(os.path.join("data", "synthetic_transactions.csv")).head(100)
        preprocessor, df_eng = preprocess_pipeline(df_raw)
        X, _ = preprocessor.get_feature_matrix(df_eng)
        
        results = engine.predict_and_score(X, df_raw)
        self.assertEqual(len(results), 100)
        self.assertTrue((results["anomaly_score"] >= 0).all() and (results["anomaly_score"] <= 100).all())
        self.assertTrue(set(results["risk_category"].unique()).issubset({"Low Risk", "Medium Risk", "High Risk"}))
        print("✓ Test 5 Passed: Model outputs verified calibrated scores and risk tiers.")

    def test_06_all_sql_analytical_queries(self):
        """Verify all 10 standard analytical SQL queries execute without error."""
        for query_name in QUERIES.keys():
            df, sql = execute_analytics_query(query_name, self.db_path)
            self.assertIsInstance(df, pd.DataFrame)
            self.assertGreater(len(df), 0, f"Query '{query_name}' should return records.")
        print(f"✓ Test 6 Passed: All {len(QUERIES)} SQL analytical queries executed successfully with non-empty results.")

    def test_07_custom_sql_safety(self):
        """Verify custom SQL console rejects destructive write statements."""
        with self.assertRaises(PermissionError):
            execute_custom_sql("DROP TABLE customers;")
        with self.assertRaises(PermissionError):
            execute_custom_sql("DELETE FROM transactions;")
        print("✓ Test 7 Passed: Custom SQL engine successfully blocks destructive operations.")

    def test_08_joined_monitoring_view(self):
        """Verify complete joined monitoring dataframe loads properly."""
        df_joined = get_complete_monitoring_records(self.db_path)
        self.assertGreater(len(df_joined), 5000)
        expected_cols = [
            "transaction_id", "customer_id", "customer_name", "customer_segment",
            "transaction_amount", "anomaly_score", "anomaly_flag", "risk_category"
        ]
        for col in expected_cols:
            self.assertIn(col, df_joined.columns)
        print("✓ Test 8 Passed: Unified surveillance dataframe query validated.")

if __name__ == "__main__":
    unittest.main()

