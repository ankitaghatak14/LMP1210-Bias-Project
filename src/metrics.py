import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
)


def calculate_overall_metrics(y_true, y_prob):
    """Overall AUROC, AUPRC, and Brier Score for risk prediction."""
    return {
        "AUROC": roc_auc_score(y_true, y_prob),
        "AUPRC": average_precision_score(y_true, y_prob),
        "Brier_Score": brier_score_loss(y_true, y_prob),
    }


def calculate_calibration_error(y_true, y_prob, n_bins=10):
    """Evaluates model reliability by comparing predicted vs observed risk."""
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    return np.mean(np.abs(prob_true - prob_pred))


def calculate_equalized_odds_gap(y_true, y_prob, sensitive_attr):
    """Quantifies disparities in TPR and FPR across subgroups."""
    y_pred = (y_prob >= 0.5).astype(int)
    df = pd.DataFrame({"y": y_true, "pred": y_pred, "group": sensitive_attr})
    groups = df["group"].unique()

    tprs, fprs = [], []
    for g in groups:
        subset = df[df["group"] == g]
        if len(subset) == 0:
            continue
        cm = confusion_matrix(subset["y"], subset["pred"], labels=[0, 1])
        if cm.shape != (2, 2):
            continue
        tn, fp, fn, tp = cm.ravel()
        tprs.append(tp / (tp + fn) if (tp + fn) > 0 else 0)
        fprs.append(fp / (fp + tn) if (fp + tn) > 0 else 0)

    if len(tprs) == 0 or len(fprs) == 0:
        return np.nan

    return max(max(tprs) - min(tprs), max(fprs) - min(fprs))


def audit_subgroups(y_true, y_prob, sensitive_df):
    """Calculates AUROC for every unique subgroup (Race, Age, Gender)."""
    results = {}
    for col in sensitive_df.columns:
        for group in sensitive_df[col].unique():
            mask = sensitive_df[col] == group
            if mask.sum() > 1 and len(np.unique(y_true[mask])) > 1:
                score = roc_auc_score(y_true[mask], y_prob[mask])
                results[f"AUROC_{col}_{group}"] = round(score, 4)
    return results


def compute_difficulty_scores(y_prob):
    """
    Difficulty based on closeness to 0.5.
    0 = easy/confident (raw prob near 0 or 1)
    1 = difficult/uncertain (raw prob near 0.5)
    """
    y_prob = np.asarray(y_prob)
    return 1.0 - 2.0 * np.abs(
        y_prob - 0.5
    )  # Scales to [0, 1], with 1 at raw prob of 0.5 and 0 at raw prob of 0 or 1


def assign_difficulty_bins(difficulty_scores, n_bins=3, labels=None):
    if labels is None:
        if n_bins == 3:
            labels = ["easy", "medium", "hard"]
        else:
            labels = [f"bin_{i}" for i in range(n_bins)]
    return pd.qcut(
        difficulty_scores,
        q=n_bins,
        labels=labels,
        duplicates="drop",
    )


def compute_group_auroc_deficits(y_true, y_prob, sensitive_attr):
    """
    Computes how much each subgroup's AUROC falls short of the overall AUROC, with a floor at 0.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    overall_auroc = roc_auc_score(y_true, y_prob)

    df = pd.DataFrame(
        {
            "y": y_true,
            "prob": y_prob,
            "group": pd.Series(sensitive_attr).reset_index(drop=True),
        }
    )

    deficits = {}
    for g in df["group"].dropna().unique():
        subset = df[df["group"] == g]
        if subset["y"].nunique() < 2:
            deficits[g] = 0.0
            continue
        g_auroc = roc_auc_score(subset["y"], subset["prob"])
        deficits[g] = max(0.0, overall_auroc - g_auroc)

    return deficits


def make_fair_weights(
    y_true,
    y_prob,
    sensitive_attr,
    alpha=0.5,
    beta=1.0,
):
    """
    Fair Loss:
        w_i: final weight for sample i
        difficulty_i: how difficult sample i is
        group_i: subgroup membership of sample i
        deficit(group_i): how much subgroup i's AUROC falls short of overall attribute AUROC
        alpha: controls how much we upweight difficult samples
        beta: controls how much we upweight samples from underperforming subgroups
        final formula: w_i = 1 + alpha * difficulty_i + beta * deficit(group_i) * difficulty_i
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    sensitive_attr = pd.Series(sensitive_attr).reset_index(drop=True)

    difficulty = compute_difficulty_scores(
        y_prob
    )  # Difficulty based on closeness to 0.5
    deficits = compute_group_auroc_deficits(
        y_true, y_prob, sensitive_attr
    )  # How much each subgroup's AUROC falls short of overall AUROC

    weights = []
    for i in range(len(y_true)):
        d_i = difficulty[i]
        g_i = sensitive_attr.iloc[i]
        gap_g = deficits.get(
            g_i, 0.0
        )  # Get the deficit for this sample's group (i.e., if race is "Black", get the AUROC gap for "Black" subgroup)
        w_i = 1.0 + alpha * d_i + beta * gap_g * d_i
        weights.append(w_i)

    weights = np.asarray(weights, dtype=float)
    weights = (
        weights / weights.mean()
    )  # Normalize to keep average weight at 1.0 to avoid changing overall learning rate scale
    return weights, difficulty, deficits


def audit_by_fixed_difficulty_bin(
    y_true, y_prob, sensitive_df, sensitive_col, difficulty_bin
):
    """
    Compute subgroup AUROCs within precomputed/fixed difficulty bin from XGBoost "non-fair" model.
    This allows us to see if subgroup gaps persist even when controlling for difficulty.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)

    audit_df = sensitive_df.copy().reset_index(drop=True)
    audit_df["y"] = y_true
    audit_df["prob"] = y_prob
    audit_df["difficulty_bin"] = pd.Series(difficulty_bin).reset_index(drop=True)

    results = {}

    # For each difficulty bin, compute AUROC for each subgroup and the gap between best and worst performing subgroup
    for dbin in audit_df["difficulty_bin"].dropna().unique():
        bin_df = audit_df[audit_df["difficulty_bin"] == dbin]
        aurocs = []

        for group in bin_df[sensitive_col].dropna().unique():
            group_df = bin_df[bin_df[sensitive_col] == group]
            if len(group_df) < 2 or group_df["y"].nunique() < 2:
                continue
            auc = roc_auc_score(group_df["y"], group_df["prob"])
            results[f"AUROC_{sensitive_col}_{group}_{dbin}"] = round(auc, 4)
            aurocs.append(auc)

        if len(aurocs) >= 2:
            results[f"AUROC_Gap_{sensitive_col}_{dbin}"] = round(
                max(aurocs) - min(aurocs), 4
            )

    return results
