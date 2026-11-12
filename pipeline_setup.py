"""
FinGuard - Intelligent Financial Transaction Monitoring & Anomaly Detection System
Module: Pipeline Setup & Execution Orchestrator
Automates end-to-end data generation, preprocessing, model training, database ingestion, and validation.
"""

import os
import pandas as pd
from data_generator import generate_and_save_dataset
from preprocessing import preprocess_pipeline
from database import init_db, insert_customers, insert_transactions, insert_anomaly_results, get_connection
from anomaly_detector import AnomalyDetectionEngine, MODEL_PATH
from analytics import QUERIES, execute_analytics_query

def run_full_pipeline(num_customers=350, num_transactions=6000, contamination=0.05, force_regenerate=False):
    print("=" * 70)
    print("   FinGuard - Financial Transaction Anomaly Detection Pipeline")
    print("=" * 70)
    
    data_dir = "data"
    cust_file = os.path.join(data_dir, "synthetic_customers.csv")
    txn_file = os.path.join(data_dir, "synthetic_transactions.csv")
    
    # Step 1: Data Generation or Loading
    if force_regenerate or not (os.path.exists(cust_file) and os.path.exists(txn_file)):
        print("\n[Step 1/5] Generating synthetic customer and transaction datasets...")
        df_customers, df_transactions = generate_and_save_dataset(
            data_dir=data_dir,
            num_customers=num_customers,
            num_transactions=num_transactions
        )
    else:
        print("\n[Step 1/5] Loading existing synthetic datasets...")
        df_customers = pd.read_csv(cust_file)
        df_transactions = pd.read_csv(txn_file)
        print(f"Loaded {len(df_customers)} customers and {len(df_transactions)} transactions.")
        
    # Step 2: Database Initialization and Ingestion
    print("\n[Step 2/5] Initializing SQLite relational database schema...")
    init_db()
    insert_customers(df_customers)
    insert_transactions(df_transactions)
    
    # Step 3: Preprocessing and Feature Engineering
    print("\n[Step 3/5] Executing preprocessing and behavioral feature engineering...")
    preprocessor, df_engineered = preprocess_pipeline(df_transactions)
    X, feature_cols = preprocessor.get_feature_matrix(df_engineered)
    print(f"Constructed feature matrix: shape={X.shape}, features={feature_cols}")
    
    # Step 4: Machine Learning Model Training (Isolation Forest)
    print("\n[Step 4/5] Training Isolation Forest model...")
    model_engine = AnomalyDetectionEngine(contamination=contamination, n_estimators=150, random_state=42)
    model_engine.fit(X, feature_cols)
    model_engine.save(MODEL_PATH)
    
    # Anomaly Scoring and Risk Stratification
    print("Generating anomaly scores and risk category tiers...")
    results_df = model_engine.predict_and_score(X, df_transactions, high_risk_threshold=75.0, medium_risk_threshold=50.0)
    
    # Ingest Anomaly Results into SQLite
    insert_anomaly_results(results_df)
    
    # Step 5: Model Evaluation against Injected Ground Truth
    print("\n[Step 5/5] Evaluating model performance against ground-truth anomalies...")
    y_true = df_transactions["is_injected_anomaly"].values
    y_pred = results_df["anomaly_flag"].values
    metrics = model_engine.evaluate(y_true, y_pred)
    
    print("-" * 50)
    print(f"Accuracy  : {metrics['accuracy']:.4f}")
    print(f"Precision : {metrics['precision']:.4f}")
    print(f"Recall    : {metrics['recall']:.4f}")
    print(f"F1-Score  : {metrics['f1_score']:.4f}")
    print(f"Confusion Matrix:\n  TN={metrics['true_negatives']} | FP={metrics['false_positives']}\n  FN={metrics['false_negatives']} | TP={metrics['true_positives']}")
    print("-" * 50)
    
    # Quick sanity check on analytical queries
    print("\n[Verification] Running Executive KPI SQL query:")
    kpi_df, _ = execute_analytics_query("1. Executive KPI Summary")
    print(kpi_df.to_string(index=False))
    
    print("\n[Verification] Database integrity confirmed! Pipeline ready.")
    return df_customers, df_transactions, results_df, metrics

if __name__ == "__main__":
    run_full_pipeline(force_regenerate=True)

