"""
Evaluation Metrics for Speech Anti-Spoofing & Deepfake Detection.
Implements Equal Error Rate (EER), ROC-AUC, Detection Error Tradeoff (DET),
and calibrated classification performance benchmarks.
"""

import numpy as np
from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from typing import Dict, Any, Tuple

def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> Tuple[float, float]:
    """
    Computes Equal Error Rate (EER) and the corresponding decision threshold.
    EER is the point on the ROC curve where False Acceptance Rate (FAR) == False Rejection Rate (FRR).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1.0 - tpr

    # Point where |fpr - fnr| is minimized
    idx_opt = np.nanargmin(np.abs(fpr - fnr))
    eer = float((fpr[idx_opt] + fnr[idx_opt]) / 2.0)
    eer_threshold = float(thresholds[idx_opt])

    return eer, eer_threshold

def calculate_metrics(y_true: np.ndarray, y_probs: np.ndarray) -> Dict[str, Any]:
    """
    Calculates comprehensive metrics suite for binary voice cloning detection.
    """
    y_true = np.asarray(y_true).astype(int)
    y_probs = np.asarray(y_probs).astype(float)

    eer, eer_thresh = compute_eer(y_true, y_probs)
    roc_auc = float(roc_auc_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 1.0
    pr_auc = float(average_precision_score(y_true, y_probs)) if len(np.unique(y_true)) > 1 else 1.0

    y_pred_eer = (y_probs >= eer_thresh).astype(int)
    y_pred_default = (y_probs >= 0.50).astype(int)

    acc = float(accuracy_score(y_true, y_pred_default))
    prec = float(precision_score(y_true, y_pred_default, zero_division=0))
    rec = float(recall_score(y_true, y_pred_default, zero_division=0))
    f1 = float(f1_score(y_true, y_pred_default, zero_division=0))

    cm = confusion_matrix(y_true, y_pred_default).tolist()

    fpr_curve, tpr_curve, _ = roc_curve(y_true, y_probs)

    return {
        'equal_error_rate_eer': round(eer, 4),
        'eer_percentage': round(eer * 100.0, 2),
        'eer_threshold': round(eer_thresh, 4),
        'roc_auc': round(roc_auc, 4),
        'pr_auc': round(pr_auc, 4),
        'accuracy': round(acc, 4),
        'precision': round(prec, 4),
        'recall': round(rec, 4),
        'f1_score': round(f1, 4),
        'confusion_matrix': cm,
        'roc_curve': {
            'fpr': [float(x) for x in fpr_curve],
            'tpr': [float(x) for x in tpr_curve]
        }
    }
