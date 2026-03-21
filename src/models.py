from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
import torch.nn as nn

RS = 2032026

def get_baselines(input_dim):
    models = {
        "Logistic_Regression": LogisticRegression(penalty='l2', C=1.0, random_state=RS, max_iter=1000),
        
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=RS),
        
        "XGBoost": XGBClassifier(n_estimators=100, learning_rate=0.1, random_state=RS, use_label_encoder=False),
        
        "MLP": MLP(input_dim) # defined below
    }
    return models

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        return self.network(x)