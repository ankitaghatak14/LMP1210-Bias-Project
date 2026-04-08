import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
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
    df = df.replace("?", pd.NA).dropna(subset=["race", "gender"])
    df["target"] = (df["readmitted"] == "<30").astype(int)

    # 4. Extract sensitive attributes 
    sensitive_cols = ["race", "gender", "age"]
    sensitive_data = df[sensitive_cols].reset_index(drop=True)

    # 5. Feature Encoding
    X_raw_df = df.drop(columns=["target", "readmitted", "encounter_id", "patient_nbr"])
    num_cols = [
        "num_lab_procedures",
        "num_procedures",
        "num_medications",
        "number_outpatient",
        "number_emergency",
        "number_inpatient",
        "time_in_hospital",
    ]
    cat_cols = [c for c in X_raw_df.columns if c not in num_cols]
    
    # ONE-HOT ENCODING 
    X_oh = pd.get_dummies(X_raw_df, columns=cat_cols, drop_first=True)
    feature_names_oh = X_oh.columns.tolist()

    # ORDINAL ENCODING PATH FOR TABPFN 
    X_raw_encoded = X_raw_df.copy()
    encoder = OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
    X_raw_encoded[cat_cols] = encoder.fit_transform(X_raw_df[cat_cols].astype(str))
    feature_names_raw = X_raw_encoded.columns.tolist()
    
    y = df["target"].values


    # 6. Split & Scale
    # SYNCHRONIZED SPLIT FOR ONE-HOT AND RAW DATA
    X_train_oh, X_test_oh, X_train_raw, X_test_raw, y_train, y_test, sens_train, sens_test = train_test_split(
        X_oh.values, X_raw_encoded.values, y, sensitive_data,
        test_size=0.2, stratify=y, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_oh = scaler.fit_transform(X_train_oh)
    X_test_oh = scaler.transform(X_test_oh)

    train_ds = DiabetesDataset(X_train_oh, y_train, sens_train.reset_index(drop=True))
    test_ds = DiabetesDataset(X_test_oh, y_test, sens_test.reset_index(drop=True))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)


    if return_arrays:
        return (train_loader, test_loader, X_train_oh, X_test_oh, X_train_raw, X_test_raw, 
                y_train, y_test, sens_train, sens_test, feature_names_oh, feature_names_raw)

    return train_loader, test_loader
