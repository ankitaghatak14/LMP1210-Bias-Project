import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score, 
    confusion_matrix, 
    brier_score_loss
)
from sklearn.calibration import calibration_curve

def calculate_overall_metrics(y_true, y_prob):
    """Overall AUROC, AUPRC, and Brier Score for risk prediction."""
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Brier_Score": brier_score_loss(y_true, y_prob)
    }

def calculate_calibration_error(y_true, y_prob, n_bins=10):
    """Evaluates model reliability by comparing predicted vs observed risk."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return np.mean(np.abs(prob_true - prob_pred))

def calculate_equalized_odds_gap(y_true, y_prob, sensitive_attr):
    """Quantifies disparities in TPR and FPR across subgroups."""
    y_pred = (y_prob >= 0.5).astype(int)
    df = pd.DataFrame({'y': y_true, 'pred': y_pred, 'group': sensitive_attr})
    groups = df['group'].unique()
    
    tprs, fprs = [], []
    for g in groups:
        subset = df[df['group'] == g]
        if len(subset) == 0: continue
        tn, fp, fn, tp = confusion_matrix(subset['y'], subset['pred'], labels=[0, 1]).ravel()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)
        
    return max(max(tprs) - min(tprs), max(fprs) - min(fprs))

def audit_subgroups(y_true, y_prob, sensitive_df):
    """Calculates AUROC for every unique subgroup (Race, Age, Gender)."""
    results = {}
    for col in sensitive_df.columns:
        for group in sensitive_df[col].unique():
            mask = sensitive_df[col] == group
            if len(np.unique(y_true[mask])) > 1:
                score = roc_auc_score(y_true[mask], y_prob[mask])
                results[f"AUROC_{col}_{group}"] = round(score, 4)
    return results