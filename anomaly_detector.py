"""
FinGuard - Intelligent Financial Transaction Monitoring & Anomaly Detection System
Module: Machine Learning - Isolation Forest Anomaly Detection
Trains isolation forest, generates normalized risk scores and tiers, and evaluates metrics.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import joblib
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "isolation_forest.joblib")
MODEL_VERSION = "v1.2.0-IF-Wealth"

class AnomalyDetectionEngine:
    def __init__(self, contamination=0.05, n_estimators=150, random_state=42):
        """
        Contamination rationale:
        In institutional wealth management, compliance teams operate with strict operational
        capacities. A contamination setting of 0.05 (5%) reflects the expected baseline rate of
        suspicious activities requiring heightened due diligence while minimizing alert fatigue.
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            random_state=self.random_state,
            max_samples="auto",
            n_jobs=-1
        )
        self.feature_names = []
        self.is_fitted = False

    def fit(self, X, feature_names=None):
        """
        Trains the Isolation Forest on numerical features.
        """
        if feature_names:
            self.feature_names = list(feature_names)
        else:
            self.feature_names = list(X.columns)
            
        print(f"[Model Engine] Training IsolationForest with {len(X)} samples, {len(self.feature_names)} features, contamination={self.contamination}...")
        self.model.fit(X[self.feature_names])
        self.is_fitted = True
        return self

    def predict_and_score(self, X, df_original, high_risk_threshold=75.0, medium_risk_threshold=50.0):
        """
        Generates raw decision functions, converts them to calibrated 0-100 anomaly scores,
        and stratifies risk categories.
        """
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet. Call fit() first.")
            
        # Raw decision function: negative values indicate anomalies, positive values indicate normal
        raw_scores = self.model.decision_function(X[self.feature_names])
        
        # Scikit-learn IF: -1 for outlier, 1 for inlier
        raw_preds = self.model.predict(X[self.feature_names])
        binary_flag = np.where(raw_preds == -1, 1, 0)
        
        # Calibrate to an intuitive 0-100 Anomaly Risk Index:
        # Lower raw_score = more anomalous => higher risk index
        # We use a sigmoid-like transformation scaled by empirical distribution
        # Centered around the decision boundary (0.0)
        risk_scores = 100.0 / (1.0 + np.exp(raw_scores * 12.0))
        risk_scores = np.round(np.clip(risk_scores, 0.0, 100.0), 2)
        
        # Risk Category classification
        risk_categories = []
        for score, flag in zip(risk_scores, binary_flag):
            if score >= high_risk_threshold:
                risk_categories.append("High Risk")
            elif score >= medium_risk_threshold or flag == 1:
                risk_categories.append("Medium Risk")
            else:
                risk_categories.append("Low Risk")
                
        results_df = pd.DataFrame({
            "transaction_id": df_original["transaction_id"].values,
            "anomaly_score": risk_scores,
            "anomaly_flag": binary_flag,
            "risk_category": risk_categories,
            "model_version": MODEL_VERSION
        })
        
        return results_df

    def evaluate(self, y_true, y_pred, y_scores=None):
        """
        Calculates comprehensive statistical performance metrics against injected ground-truth.
        """
        cm = confusion_matrix(y_true, y_pred)
        # Handle 2x2 shape
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            
        acc = accuracy_score(y_true, y_pred)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        metrics = {
            "confusion_matrix": cm,
            "true_negatives": int(tn),
            "false_positives": int(fp),
            "false_negatives": int(fn),
            "true_positives": int(tp),
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "contamination_param": self.contamination,
            "n_estimators": self.n_estimators,
            "model_version": MODEL_VERSION
        }
        return metrics

    def save(self, filepath=MODEL_PATH):
        """
        Persists model artifacts to disk.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump({
            "model": self.model,
            "feature_names": self.feature_names,
            "contamination": self.contamination,
            "n_estimators": self.n_estimators,
            "version": MODEL_VERSION
        }, filepath)
        print(f"[Model Engine] Saved trained model artifact to {filepath}")

    @classmethod
    def load(cls, filepath=MODEL_PATH):
        """
        Loads saved model artifact.
        """
        if not os.path.exists(filepath):
            return None
        data = joblib.load(filepath)
        engine = cls(contamination=data["contamination"], n_estimators=data["n_estimators"])
        engine.model = data["model"]
        engine.feature_names = data["feature_names"]
        engine.is_fitted = True
        return engine

