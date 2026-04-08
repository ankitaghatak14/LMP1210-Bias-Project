import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from tabpfn import TabPFNClassifier
from tqdm import tqdm
from xgboost import XGBClassifier

from src.metrics import make_fair_weights
from tabpfn import TabPFNClassifier

RS = 2032026


def extract_features_labels(trainloader):
    X_train_list, y_train_list = [], []
    for x, y, _ in trainloader:
        X_train_list.append(x.numpy())
        y_train_list.append(y.numpy())
    X_train = np.concatenate(X_train_list)
    y_train = np.concatenate(y_train_list)
    return X_train, y_train


def fit_xgboost_fair(
    train,
    sensitive_train,
    baseline_model,
    attr="race",
    alpha=0.5,
    beta=1.0,
    random_state=RS,
    allcols=["race", "age", "gender"],
):
    X_train, y_train = extract_features_labels(train)
    train_prob = baseline_model.predict_proba(X_train)[:, 1]

    # Compute cumulative weights using "non-fair model"
    if attr == "overall":
        cumulative_weights = np.zeros_like(y_train, dtype=float)
        for var in allcols:
            weights, difficulty, deficits = make_fair_weights(
                y_true=y_train,
                y_prob=train_prob,
                sensitive_attr=sensitive_train[var],
                alpha=alpha,
                beta=beta,
            )
            cumulative_weights += weights
        weights = cumulative_weights / len(
            allcols
        )  # Average weights across all attributes

    # Compute indiviuals weights using "non-fair model"
    else:
        sensitive_train = sensitive_train[attr]
        weights, difficulty, deficits = make_fair_weights(
            y_true=y_train,
            y_prob=train_prob,
            sensitive_attr=sensitive_train,
            alpha=alpha,
            beta=beta,
        )

    # retrain model using "fair weights"
    tuned_hps = baseline_model.get_params()
    fair_model = XGBClassifier(**tuned_hps)
    fair_model.fit(X_train, y_train, sample_weight=weights)

    return fair_model, weights, difficulty, deficits


def get_fold_loaders(trainloader, fold, total_folds):
    dataset = trainloader.dataset
    total_size = len(dataset)
    fold_size = total_size // total_folds
    indices = np.arange(total_size)

    val_start = fold * fold_size
    val_end = val_start + fold_size if fold < total_folds - 1 else total_size

    val_indices = indices[val_start:val_end]
    train_indices = np.concatenate((indices[:val_start], indices[val_end:]))

    train_subset = torch.utils.data.Subset(dataset, train_indices)
    val_subset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(
        train_subset, batch_size=trainloader.batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_subset, batch_size=trainloader.batch_size, shuffle=False
    )

    return train_loader, val_loader


def modeltune(modelname, input_dim, trainloader, folds=2):
    if modelname == "Logistic_Regression":
        param_dist = {"C": [0.01, 0.1, 1, 10, 100], "penalty": ["l1", "l2"]}
    elif modelname == "Random_Forest":
        param_dist = {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10],
        }
    elif modelname == "XGBoost":
        param_dist = {
            "n_estimators": [50, 100, 200],
            "learning_rate": [0.01, 0.1, 0.2],
            "max_depth": [3, 6, 9],
        }
    elif modelname == "MLP":
        param_dist = {
            "hidden_layer_sizes": [(128, 64), (256, 128)],
            "learning_rate_init": [0.001, 0.01],
        }
    else:
        raise ValueError("Unsupported model name for tuning.")

    if modelname in ["Logistic_Regression", "Random_Forest", "XGBoost"]:
        print("Tuning hyperparameters for", modelname)
        X_train, y_train = extract_features_labels(trainloader)
        if modelname == "Logistic_Regression":
            model = LogisticRegression(random_state=RS, max_iter=1000)
        elif modelname == "Random_Forest":
            model = RandomForestClassifier(random_state=RS)
        elif modelname == "XGBoost":
            model = XGBClassifier(random_state=RS, eval_metric="logloss")

        random_search = RandomizedSearchCV(
            model,
            param_distributions=param_dist,
            n_iter=5,
            cv=folds,
            random_state=RS,
            verbose=1,
        )
        random_search.fit(X_train, y_train)
        print("Best hyperparameters for", modelname, ":", random_search.best_params_)
        return random_search.best_estimator_
    elif modelname == "MLP":
        print("Tuning hyperparameters for", modelname)
        param_combinations = [
            (hls, lr)
            for hls in param_dist["hidden_layer_sizes"]
            for lr in param_dist["learning_rate_init"]
        ]
        param_scores = {comb: [] for comb in param_combinations}
        for fold in range(folds):
            trainloader, val_loader = get_fold_loaders(trainloader, fold, folds)
            for comb in tqdm(param_combinations):
                hidden_layer_sizes, lr = comb
                model = MLP(input_dim, hidden_layer_sizes)
                model.fit(trainloader, epochs=5, lr=lr)
                # Evaluate on validation set
                model.eval()
                val_loss = 0
                criterion = nn.BCELoss()
                with torch.no_grad():
                    for features, labels, _ in val_loader:
                        outputs = model(features).squeeze()
                        loss = criterion(outputs, labels)
                        val_loss += loss.item()
                avg_val_loss = val_loss / len(val_loader)
                param_scores[comb].append(avg_val_loss)
        avg_scores = {comb: np.mean(scores) for comb, scores in param_scores.items()}
        best_comb = min(avg_scores, key=avg_scores.get)
        best_model = MLP(input_dim, best_comb[0])
        best_model.fit(trainloader, epochs=10, lr=best_comb[1])
        print(
            "Best hyperparameters for MLP:",
            {"hidden_layer_sizes": best_comb[0], "learning_rate_init": best_comb[1]},
        )
        return best_model


def get_baselines(input_dim, trainloader, X_raw=None, y_raw=None, performance_tuning=True, modelchoice=None):
    if not performance_tuning:
        X, y = extract_features_labels(trainloader)
        mlp = MLP(input_dim)
        mlp.fit(trainloader)
    
    if modelchoice is None:
        models = {
            "MLP": modeltune("MLP", input_dim, trainloader) if performance_tuning else mlp,
            "Logistic_Regression": modeltune("Logistic_Regression", input_dim, trainloader) if performance_tuning else LogisticRegression(random_state=RS, max_iter=1000, penalty='l2').fit(X, y),
            "Random_Forest": modeltune("Random_Forest", input_dim, trainloader) if performance_tuning else RandomForestClassifier(random_state=RS).fit(X, y),
            "XGBoost": modeltune("XGBoost", input_dim, trainloader) if performance_tuning else XGBClassifier(random_state=RS, eval_metric="logloss").fit(X, y),
        }
    else:
        if modelchoice == "MLP":
            models = {"MLP": modeltune("MLP", input_dim, trainloader)}
        elif modelchoice == "Logistic_Regression":
            models = {"Logistic_Regression": modeltune("Logistic_Regression", input_dim, trainloader)}
        elif modelchoice == "Random_Forest":
            models = {"Random_Forest": modeltune("Random_Forest", input_dim, trainloader)}
        elif modelchoice == "XGBoost":
            models = {"XGBoost": modeltune("XGBoost", input_dim, trainloader)}

    if X_raw is not None and y_raw is not None:
        if modelchoice is None or modelchoice == "TabPFN":
            print("Fitting TabPFN on raw features (subset N=2000)...")
            # Changed N_ensemble_configurations to n_estimators for compatibility with TabPFN v2.0+
            tabpfn = TabPFNClassifier(device='cpu', n_estimators=32)
            tabpfn.fit(X_raw[:2000], y_raw[:2000])
            models["TabPFN"] = tabpfn

    return models


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_layer_sizes=(128, 64)):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_layer_sizes[0]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[0], hidden_layer_sizes[1]),
            nn.ReLU(),
            nn.Linear(hidden_layer_sizes[1], 1),
            nn.Sigmoid(),
        )

    def fit(self, trainloader, epochs=10, lr=0.001):
        criterion = nn.BCELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)

        for epoch in range(epochs):
            self.train()
            for features, labels, _ in trainloader:
                optimizer.zero_grad()
                outputs = self.network(features).squeeze()
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

    def forward(self, x):
        return self.network(x)
