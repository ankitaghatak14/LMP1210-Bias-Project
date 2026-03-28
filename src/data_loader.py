import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset


class DiabetesDataset(Dataset):
    def __init__(self, X, y, sensitive_attrs):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.sensitive = sensitive_attrs

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx], self.sensitive.iloc[idx].to_dict()


def get_dataloaders(data_path, batch_size=64, return_arrays=False):
    # 1. Load Data
    df = pd.read_csv(data_path)

    # 2. Simple Preprocessing (Handling '?' as NaN)
    df = df.replace("?", pd.NA).dropna(subset=["race", "gender"])

    # 3. Define target (e.g., readmitted <30 days)
    df["target"] = (df["readmitted"] == "<30").astype(int)

    # 4. Extract sensitive attributes for the audit
    sensitive_cols = ["race", "gender", "age"]
    sensitive_data = df[sensitive_cols].reset_index(drop=True)

    # 5. Feature Encoding
    X_raw = df.drop(columns=["target", "readmitted", "encounter_id", "patient_nbr"])
    num_cols = [
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "time_in_hospital",
    ]
    cat_cols = [c for c in X_raw.columns if c not in num_cols]
    X = pd.get_dummies(X_raw, columns=cat_cols, drop_first=True)
    y = df["target"].values

    feature_names = X.columns.tolist()

    # 6. Split & Scale
    X_train, X_test, y_train, y_test, sens_train, sens_test = train_test_split(
        X.values,
        y,
        sensitive_data,
        test_size=0.2,
        stratify=y,
        random_state=42,
    )

    sens_train = sens_train.reset_index(drop=True)
    sens_test = sens_test.reset_index(drop=True)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 7. Create Loaders
    train_ds = DiabetesDataset(X_train, y_train, sens_train)
    test_ds = DiabetesDataset(X_test, y_test, sens_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    if return_arrays:
        return (
            train_loader,
            test_loader,
            X_train,
            X_test,
            y_train,
            y_test,
            sens_train,
            sens_test,
            feature_names,
        )

    return train_loader, test_loader
