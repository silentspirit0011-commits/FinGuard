"""
FinGuard - Intelligent Financial Transaction Monitoring & Anomaly Detection System
Module: Data Preprocessing & Feature Engineering
Performs data cleaning, validation, audit logging, and feature engineering for Isolation Forest modeling.
"""

import pandas as pd
import numpy as np
from datetime import datetime

class DataPreprocessor:
    def __init__(self):
        self.customer_profiles = {}
        self.engineered_feature_cols = [
            "transaction_amount",
            "transaction_hour",
            "transaction_day",
            "is_weekend",
            "is_night_hours",
            "customer_txn_count",
            "avg_transaction_amount_customer",
            "deviation_from_customer_avg",
            "amount_to_avg_ratio",
            "amount_to_balance_ratio",
            "time_diff_previous_txn_sec"
        ]

    def clean_and_validate(self, df_raw):
        """
        Cleans raw transaction data, validates types, handles missing values,
        removes duplicates, and flags invalid records.
        """
        df = df_raw.copy()
        initial_count = len(df)
        
        # 1. Deduplicate by transaction_id
        df = df.drop_duplicates(subset=["transaction_id"]).reset_index(drop=True)
        dedup_count = len(df)
        
        # 2. Parse timestamps
        df["transaction_date"] = pd.to_datetime(df["transaction_date"], errors="coerce")
        df = df.dropna(subset=["transaction_date"]).reset_index(drop=True)
        
        # 3. Numeric validation
        df["transaction_amount"] = pd.to_numeric(df["transaction_amount"], errors="coerce")
        df["account_balance"] = pd.to_numeric(df["account_balance"], errors="coerce")
        
        # Filter strictly invalid transaction amounts (non-positive or null)
        df = df[df["transaction_amount"] > 0].reset_index(drop=True)
        
        # 4. Fill missing non-critical strings
        df["merchant_category"] = df["merchant_category"].fillna("General Merchandise")
        df["payment_method"] = df["payment_method"].fillna("Electronic Transfer")
        df["location"] = df["location"].fillna("Domestic")
        df["transaction_type"] = df["transaction_type"].fillna("Debit")
        
        print(f"[Preprocessor] Cleaned records: {initial_count} -> {len(df)} (Duplicates/Invalid dropped: {initial_count - len(df)})")
        return df

    def engineer_features(self, df_clean):
        """
        Computes financial behavioral indicators, velocity metrics, and ratio deviations.
        Ensures the ground-truth anomaly label is never touched or used as input.
        """
        df = df_clean.copy()
        
        # Sort chronologically by customer to accurately compute delta times
        df = df.sort_values(by=["customer_id", "transaction_date"]).reset_index(drop=True)
        
        # 1. Temporal decomposition
        df["transaction_hour"] = df["transaction_date"].dt.hour
        df["transaction_day"] = df["transaction_date"].dt.dayofweek
        df["is_weekend"] = df["transaction_day"].apply(lambda d: 1 if d >= 5 else 0)
        df["is_night_hours"] = df["transaction_hour"].apply(lambda h: 1 if (h < 6 or h >= 22) else 0)
        
        # 2. Velocity: Time elapsed since customer's previous transaction (seconds)
        df["prev_txn_time"] = df.groupby("customer_id")["transaction_date"].shift(1)
        df["time_diff_previous_txn_sec"] = (df["transaction_date"] - df["prev_txn_time"]).dt.total_seconds()
        # For first transaction per customer, default to 24 hours (86,400s)
        df["time_diff_previous_txn_sec"] = df["time_diff_previous_txn_sec"].fillna(86400.0)
        df = df.drop(columns=["prev_txn_time"])
        
        # 3. Customer historical transaction aggregations
        cust_stats = df.groupby("customer_id")["transaction_amount"].agg(["count", "mean", "std"]).reset_index()
        cust_stats.columns = ["customer_id", "customer_txn_count", "avg_transaction_amount_customer", "std_transaction_amount_customer"]
        cust_stats["std_transaction_amount_customer"] = cust_stats["std_transaction_amount_customer"].fillna(1.0)
        
        # Merge customer statistics back
        df = df.merge(cust_stats, on="customer_id", how="left")
        
        # 4. Deviation & Ratio metrics
        df["deviation_from_customer_avg"] = df["transaction_amount"] - df["avg_transaction_amount_customer"]
        df["amount_to_avg_ratio"] = df["transaction_amount"] / (df["avg_transaction_amount_customer"] + 1e-5)
        df["amount_to_balance_ratio"] = df["transaction_amount"] / (df["account_balance"] + 1.0)
        
        # Sort back chronologically for timeline display
        df = df.sort_values(by="transaction_date").reset_index(drop=True)
        
        return df

    def get_feature_matrix(self, df_engineered):
        """
        Extracts only numerical features for the Isolation Forest model.
        Explicitly excludes identifiers and ground-truth labels.
        """
        feature_cols = [
            "transaction_amount",
            "transaction_hour",
            "transaction_day",
            "is_weekend",
            "is_night_hours",
            "customer_txn_count",
            "avg_transaction_amount_customer",
            "deviation_from_customer_avg",
            "amount_to_avg_ratio",
            "amount_to_balance_ratio",
            "time_diff_previous_txn_sec"
        ]
        
        X = df_engineered[feature_cols].copy()
        
        # Replace any potential infinite/NaN values with median
        for col in feature_cols:
            if np.isinf(X[col]).any():
                X[col] = X[col].replace([np.inf, -np.inf], np.nan)
            if X[col].isna().any():
                X[col] = X[col].fillna(X[col].median())
                
        return X, feature_cols

def preprocess_pipeline(df_raw):
    """
    Convenience wrapper to run full cleaning and feature engineering.
    """
    preprocessor = DataPreprocessor()
    df_clean = preprocessor.clean_and_validate(df_raw)
    df_engineered = preprocessor.engineer_features(df_clean)
    return preprocessor, df_engineered

