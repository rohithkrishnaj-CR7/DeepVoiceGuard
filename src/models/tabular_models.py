"""
Tabular Acoustic Feature Classifier Ensemble (LightGBM, XGBoost, Random Forest, Extra Trees).
Provides high speed (<10ms inference), robust feature weighting, and transparent feature attribution.
"""

import os
import joblib
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
import lightgbm as lgb
import xgboost as xgb

class TabularDetector:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.feature_names: List[str] = []
        self.is_fitted = False

        # Individual base classifiers
        self.rf = RandomForestClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.et = ExtraTreesClassifier(
            n_estimators=120,
            max_depth=12,
            min_samples_split=4,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        self.lgbm = lgb.LGBMClassifier(
            n_estimators=150,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            class_weight='balanced',
            random_state=random_state,
            verbosity=-1,
            n_jobs=-1
        )
        self.xgb = xgb.XGBClassifier(
            n_estimators=150,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            eval_metric='logloss',
            n_jobs=-1
        )

        # Ensemble weights [RF, ET, LGBM, XGB]
        self.weights = np.array([0.25, 0.20, 0.30, 0.25], dtype=np.float32)

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: Optional[List[str]] = None) -> 'TabularDetector':
        """
        Fits scaler and all constituent ensemble classifiers on training set.
        """
        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        X_scaled = self.scaler.fit_transform(X)

        if feature_names is not None:
            self.feature_names = feature_names
        else:
            self.feature_names = [f'feat_{i}' for i in range(X.shape[1])]

        self.rf.fit(X_scaled, y)
        self.et.fit(X_scaled, y)
        self.lgbm.fit(X_scaled, y)
        self.xgb.fit(X_scaled, y)

        self.is_fitted = True
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Returns calibrated class probabilities [N, 2] (P(Real), P(Cloned)).
        """
        if not self.is_fitted:
            raise RuntimeError("Model is not fitted yet. Call fit() or load().")

        if X.ndim == 1:
            X = X.reshape(1, -1)

        X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=-1.0)
        X_scaled = self.scaler.transform(X)

        p_rf = self.rf.predict_proba(X_scaled)
        p_et = self.et.predict_proba(X_scaled)
        p_lgbm = self.lgbm.predict_proba(X_scaled)
        p_xgb = self.xgb.predict_proba(X_scaled)

        # Weighted soft voting
        p_ensemble = (
            self.weights[0] * p_rf +
            self.weights[1] * p_et +
            self.weights[2] * p_lgbm +
            self.weights[3] * p_xgb
        )
        return p_ensemble

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        probas = self.predict_proba(X)[:, 1]
        return (probas >= threshold).astype(int)

    def get_feature_importances(self, top_n: int = 20) -> List[Tuple[str, float]]:
        """
        Computes normalized aggregate feature importance across tree models.
        """
        if not self.is_fitted:
            return []

        imp_rf = self.rf.feature_importances_
        imp_et = self.et.feature_importances_
        imp_lgbm = self.lgbm.feature_importances_ / (np.sum(self.lgbm.feature_importances_) + 1e-10)
        imp_xgb = self.xgb.feature_importances_ / (np.sum(self.xgb.feature_importances_) + 1e-10)

        imp_avg = (imp_rf + imp_et + imp_lgbm + imp_xgb) / 4.0
        
        paired = [(name, float(score)) for name, score in zip(self.feature_names, imp_avg)]
        paired.sort(key=lambda x: x[1], reverse=True)
        return paired[:top_n]

    def save(self, filepath: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        payload = {
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'rf': self.rf,
            'et': self.et,
            'lgbm': self.lgbm,
            'xgb': self.xgb,
            'weights': self.weights,
            'is_fitted': self.is_fitted
        }
        joblib.dump(payload, filepath)

    def load(self, filepath: str) -> 'TabularDetector':
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        payload = joblib.load(filepath)
        self.scaler = payload['scaler']
        self.feature_names = payload['feature_names']
        self.rf = payload['rf']
        self.et = payload['et']
        self.lgbm = payload['lgbm']
        self.xgb = payload['xgb']
        self.weights = payload['weights']
        self.is_fitted = payload['is_fitted']
        return self
