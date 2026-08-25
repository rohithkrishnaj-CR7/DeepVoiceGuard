"""
End-to-End Model Training Pipeline.
Trains Tabular GBDT Ensembles and Deep Learning (LCNN / SpecResNet) models,
computes EER / ROC evaluation benchmarks, and exports artifacts to saved_models/.
"""

import os
import json
import numpy as np
from sklearn.model_selection import train_test_split
from typing import Dict, Any, Optional

from ..features.extractor import DEFAULT_SAMPLE_RATE
from ..models.tabular_models import TabularDetector
from ..models.deep_models import DeepClassifierWrapper
from ..data.dataset_loader import VoiceDatasetLoader
from .metrics import calculate_metrics

class ModelTrainer:
    def __init__(self, output_dir: str = "saved_models", sr: int = DEFAULT_SAMPLE_RATE):
        self.output_dir = output_dir
        self.sr = sr
        os.makedirs(output_dir, exist_ok=True)

    def train_all_models(
        self,
        X_tab: np.ndarray,
        X_mel: np.ndarray,
        X_lfcc: np.ndarray,
        y: np.ndarray,
        feature_names: list,
        epochs: int = 15,
        batch_size: int = 16
    ) -> Dict[str, Any]:
        """
        Trains Tabular, LCNN, and SpecResNet models and evaluates performance on held-out test set.
        """
        # Train / Val / Test split (70% / 15% / 15%)
        indices = np.arange(len(y))
        idx_train, idx_temp, y_train, y_temp = train_test_split(
            indices, y, test_size=0.30, stratify=y, random_state=42
        )
        idx_val, idx_test, y_val, y_test = train_test_split(
            idx_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
        )

        print(f"Dataset split: Train={len(idx_train)}, Val={len(idx_val)}, Test={len(idx_test)}")

        # 1. Train Tabular GBDT Ensemble
        print("--> Training Tabular Acoustic Classifier Ensemble (LightGBM + XGBoost + RF + ET)...")
        tab_model = TabularDetector(random_state=42)
        tab_model.fit(X_tab[idx_train], y_train, feature_names=feature_names)
        tab_path = os.path.join(self.output_dir, "tabular_model.joblib")
        tab_model.save(tab_path)
        print(f"Saved tabular model to {tab_path}")

        # 2. Train LCNN on LFCC Tensors
        print("--> Training PyTorch LFCC-LCNN Deep Network...")
        lcnn_wrapper = DeepClassifierWrapper(model_type='lcnn')
        lcnn_wrapper.fit(
            X_train=X_lfcc[idx_train],
            y_train=y_train,
            X_val=X_lfcc[idx_val],
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        lcnn_path = os.path.join(self.output_dir, "lcnn_model.pt")
        lcnn_wrapper.save(lcnn_path)
        print(f"Saved LCNN model to {lcnn_path}")

        # 3. Train SpecResNet on Log-Mel Spectrograms
        print("--> Training PyTorch SpecResNet Deep Network...")
        resnet_wrapper = DeepClassifierWrapper(model_type='specresnet')
        resnet_wrapper.fit(
            X_train=X_mel[idx_train],
            y_train=y_train,
            X_val=X_mel[idx_val],
            y_val=y_val,
            epochs=epochs,
            batch_size=batch_size
        )
        resnet_path = os.path.join(self.output_dir, "specresnet_model.pt")
        resnet_wrapper.save(resnet_path)
        print(f"Saved SpecResNet model to {resnet_path}")

        # 4. Evaluate Models on Held-out Test Set
        print("--> Evaluating models on Test Split...")
        p_tab_test = tab_model.predict_proba(X_tab[idx_test])[:, 1]
        p_lcnn_test = lcnn_wrapper.predict_proba(X_lfcc[idx_test])[:, 1]
        p_res_test = resnet_wrapper.predict_proba(X_mel[idx_test])[:, 1]

        # Fused Ensemble Prediction
        p_ensemble_test = (0.50 * p_tab_test + 0.30 * p_lcnn_test + 0.20 * p_res_test)

        metrics_tab = calculate_metrics(y_test, p_tab_test)
        metrics_lcnn = calculate_metrics(y_test, p_lcnn_test)
        metrics_resnet = calculate_metrics(y_test, p_res_test)
        metrics_ensemble = calculate_metrics(y_test, p_ensemble_test)

        feature_importances = tab_model.get_feature_importances(top_n=25)

        results = {
            'ensemble': metrics_ensemble,
            'tabular': metrics_tab,
            'lcnn': metrics_lcnn,
            'specresnet': metrics_resnet,
            'feature_importances': feature_importances,
            'test_set_size': len(idx_test)
        }

        # Save metrics json
        metrics_path = os.path.join(self.output_dir, "training_metrics.json")
        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2)
        print(f"Saved training metrics to {metrics_path}")

        print("\n=== TEST SET PERFORMANCE SUMMARY ===")
        print(f"Ensemble EER:       {metrics_ensemble['eer_percentage']}%")
        print(f"Ensemble ROC-AUC:   {metrics_ensemble['roc_auc']}")
        print(f"Ensemble Accuracy:  {metrics_ensemble['accuracy'] * 100:.2f}%")
        print(f"Ensemble F1-Score:  {metrics_ensemble['f1_score']}")
        print("=====================================\n")

        return results
