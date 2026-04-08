import numpy as np
import pandas as pd
import torch
from src.data_loader import get_dataloaders
from src.metrics import (
    assign_difficulty_bins,
    audit_by_fixed_difficulty_bin,
    audit_subgroups,
    calculate_calibration_error,
    calculate_equalized_odds_gap,
    calculate_overall_metrics,
    compute_difficulty_scores,
)
from src.models import fit_xgboost_fair, get_baselines

DATA_PATH = "data/diabetic_data.csv"
OUTPUT_FILE = "baseline_performance_report.csv"
#INPUT_DIM = 2474


def evaluate_model(name, y_true, y_prob, sensitive_df, fixed_difficulty_bin=None):
    res = calculate_overall_metrics(y_true, y_prob)
    res["Calibration_Error"] = calculate_calibration_error(y_true, y_prob)

    for col in ["race", "gender", "age"]:
        res[f"Equalized_Odds_Gap_{col}"] = calculate_equalized_odds_gap(
            y_true, y_prob, sensitive_df[col]
        )

    subgroup_res = audit_subgroups(y_true, y_prob, sensitive_df)
    res.update(subgroup_res)

    # Difficulty-conditioned subgroup audits using FIXED bins computed from XGBoost difficulty scores (to ensure same bins across models)
    if fixed_difficulty_bin is not None:
        for col in ["race", "gender", "age"]:
            diff_res = audit_by_fixed_difficulty_bin(
                y_true=y_true,
                y_prob=y_prob,
                sensitive_df=sensitive_df,
                sensitive_col=col,
                difficulty_bin=fixed_difficulty_bin,
            )
            res.update(diff_res)

    res["Model"] = name
    return res

def main():
   
    (
        train_loader,
        test_loader,
        X_train_oh,
        X_test_oh,
        X_train_raw,
        X_test_raw,
        y_train,
        y_test,
        sens_train,
        sens_test,
        feat_names_oh,
        feat_names_raw,
    ) = get_dataloaders(DATA_PATH, return_arrays=True)

    all_results = []
    # DICTIONARY TO CAPTURE FEATURE IMPORTANCES
    feature_importances = {}

    models = get_baselines(X_train_oh.shape[1], train_loader, X_raw=X_train_raw, y_raw=y_train, performance_tuning=True)

    # CAPTURING INITIAL XGBOOST FEATURE IMPORTANCE
    if "XGBoost" in models:
        feature_importances["XGBoost_Unfair"] = pd.Series(
            models["XGBoost"].feature_importances_, index=feat_names_oh
        )

    y_true_list = []
    sens_list = []
    y_prob_dict = {name: [] for name in models}

    X_test_raw_tensor = torch.tensor(X_test_raw)

    with torch.no_grad():
        for i, (features, labels, sensitive) in enumerate(test_loader):
            y_true_list.append(labels.numpy())
            sens_list.append(pd.DataFrame(sensitive))

            start_idx = i * test_loader.batch_size
            end_idx = start_idx + len(labels)

            for name, model in models.items():
                if name == "MLP":
                    model.eval()
                    probs = model(features).squeeze().numpy()
                elif name == "TabPFN":
                    # TABPFN USES THE RAW FEATURES
                    raw_batch = X_test_raw_tensor[start_idx:end_idx].numpy()
                    probs = model.predict_proba(raw_batch)[:, 1]
                else:
                    # OTHER MODELS USE ONE-HOT ENCODED FEATURES
                    probs = model.predict_proba(features.numpy())[:, 1]
                y_prob_dict[name].append(probs)

    y_true = np.concatenate(y_true_list)
    sensitive_df = pd.concat(sens_list).reset_index(drop=True)

    xgb_test_prob = np.concatenate(y_prob_dict["XGBoost"])
    xgb_test_difficulty = compute_difficulty_scores(xgb_test_prob)
    fixed_difficulty_bin = assign_difficulty_bins(xgb_test_difficulty, n_bins=3)

    for name in models:
        y_prob = np.concatenate(y_prob_dict[name])
        res = evaluate_model(
            name=name,
            y_true=y_true,
            y_prob=y_prob,
            sensitive_df=sensitive_df,
            fixed_difficulty_bin=fixed_difficulty_bin,
        )
        all_results.append(res)

    for attr in ["race", "gender", "age", "overall"]:
        fair_model, _, _, _ = fit_xgboost_fair(
            train=train_loader,
            sensitive_train=sens_train,
            baseline_model=models["XGBoost"],
            attr=attr,
            alpha=0.5,
            beta=1.0,
        )

        # CAPTURING FAIR XGBOOST FEATURE IMPORTANCE FOR EACH ATTRIBUTE
        feature_importances[f"XGBoost_Fair_{attr}"] = pd.Series(
            fair_model.feature_importances_, index=feat_names_oh
        )

        fair_prob = fair_model.predict_proba(X_test_oh)[:, 1]
        fair_res = evaluate_model(
            f"XGBoost_Fair_{attr}",
            y_test,
            fair_prob,
            sens_test,
            fixed_difficulty_bin=fixed_difficulty_bin,
        )
        all_results.append(fair_res)

    pd.DataFrame(feature_importances).to_csv("feature_importances.csv")
    print("SUCCESS: Saved feature importance comparison to feature_importances.csv")

    pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved performance report to {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
