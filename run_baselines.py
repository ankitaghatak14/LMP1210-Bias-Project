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
INPUT_DIM = 2474


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
        X_train,
        X_test,
        y_train,
        y_test,
        sens_train,
        sens_test,
        feature_names,
    ) = get_dataloaders(DATA_PATH, return_arrays=True)

    all_results = []

    models = get_baselines(INPUT_DIM, train_loader, performance_tuning=True)

    y_true_list = []
    sens_list = []
    y_prob_dict = {name: [] for name in models}

    with torch.no_grad():
        for features, labels, sensitive in test_loader:
            y_true_list.append(labels.numpy())
            sens_list.append(pd.DataFrame(sensitive))

            for name, model in models.items():
                if name == "MLP":
                    model.eval()
                    probs = model(features).squeeze().numpy()
                else:
                    probs = model.predict_proba(features.numpy())[:, 1]
                y_prob_dict[name].append(probs)

    y_true = np.concatenate(y_true_list)
    sensitive_df = pd.concat(sens_list).reset_index(
        drop=True
    )  # assemble sensitive attributes into a single DataFrame for subgroup audits

    # create fixed difficulty bins based on XGBoost predictions on test set (to ensure same bins across models for fair comparison)
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

        fair_prob = fair_model.predict_proba(X_test)[:, 1]
        fair_res = evaluate_model(
            f"XGBoost_Fair_{attr}",
            y_test,
            fair_prob,
            sens_test,
            fixed_difficulty_bin=fixed_difficulty_bin,
        )
        all_results.append(fair_res)

    pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)
    print(f"Saved results to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
