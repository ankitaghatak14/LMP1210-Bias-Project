import numpy as np
import pandas as pd
import torch
from src.data_loader import get_dataloaders
from src.metrics import (
    audit_subgroups,
    calculate_calibration_error,
    calculate_equalized_odds_gap,
    calculate_overall_metrics,
)
from src.models import get_baselines

DATA_PATH = "data/diabetic_data.csv"
OUTPUT_FILE = "baseline_performance_report.csv"
INPUT_DIM = 2474


def main():
    # 1. Load Data
    train_loader, test_loader = get_dataloaders(DATA_PATH)

    # 2. Initialize and Train models
    models = get_baselines(INPUT_DIM, train_loader, performance_tuning=False)
    all_results = []

    # 3. Evaluate on Test Set
    y_true_list, y_prob_dict, sens_list = [], {name: [] for name in models}, []

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
    sensitive_df = pd.concat(sens_list).reset_index(drop=True)

    # 4. Compile Metrics
    for name in models:
        y_prob = np.concatenate(y_prob_dict[name])

        # Performance
        res = calculate_overall_metrics(y_true, y_prob)
        res["Calibration_Error"] = calculate_calibration_error(y_true, y_prob)
        res["Equalized_Odds_Gap_Race"] = calculate_equalized_odds_gap(
            y_true, y_prob, sensitive_df["race"]
        )

        # Subgroup Audit
        subgroup_res = audit_subgroups(y_true, y_prob, sensitive_df)
        res.update(subgroup_res)

        res["Model"] = name
        all_results.append(res)

    # 5. Export
    pd.DataFrame(all_results).to_csv(OUTPUT_FILE, index=False)


if __name__ == "__main__":
    main()
