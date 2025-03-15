import numpy as np
import pandas as pd 
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import os
COLAB = 'google.colab' in str(get_ipython())
print ( f"still using {COLAB = }")

if COLAB :
    from google.colab import drive, userdata
    COLAB = True
    print("Note: using Google CoLab")
if COLAB :
   import kagglehub
   kagglehub.login()
if COLAB :
    playground_series_s5e3_path = kagglehub.competition_download('playground-series-s5e3')
    print('Data source import complete.')
else :
    playground_series_s5e3_path = "../input/playground-series-s5e3"

import numpy as np 
import polars as pl
import os
for dirname, _, filenames in os.walk(f'{playground_series_s5e3_path}'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
import matplotlib.pyplot as plt
import seaborn as sns
print(raw_data.null_count())
print(raw_data.describe())
print(raw_data.schema)
print(raw_data.columns)
features = ['pressure', 'maxtemp', 'temparature', 'mintemp', 'dewpoint', 'humidity', 'cloud', 'sunshine', 'winddirection', 'windspeed']
for f in features :
    sns.barplot ( data = raw_data.group_by(f).len().to_pandas(), x = f, y = "len")
    plt.title(f"Unique values for {f}")
    plt.show()
plt.figure(figsize=(8, 6))
sns.histplot(data = raw_data['rainfall'], bins = 2 )
plt.title('Distribution of Target Variable')
plt.show()
numerical_cols = raw_data.select(pl.col(pl.Float64)).columns
correlation_matrix = raw_data.select(numerical_cols).corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix.to_pandas(), annot=True, cmap='coolwarm')
plt.title('Correlation Matrix of Numerical Features')
plt.show()
plt.figure(figsize=(4, 3))
for f in features :
  sns.boxplot(x=raw_data[f])
  plt.title(f'Boxplot of {f}')
  plt.show()
import itertools
if len(numerical_cols) >= 3:
    for (a,b) in itertools.permutations (features, 2) :
        sns.scatterplot(raw_data.select([a,b, "rainfall"]).to_pandas(), x = a, y = b, hue = "rainfall")
        plt.show()
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
from tqdm import tqdm
X = raw_data.select(features).to_numpy()
y = raw_data["rainfall"].to_numpy()
scaler = StandardScaler()
X = scaler.fit_transform(X)
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
class RainfallModel(nn.Module):
    def __init__(self, input_dim):
        super(RainfallModel, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.layers(x)
input_dim = len(features)
epochs = 2000
learning_rate = 0.000035
k_folds = 7

# Cross-validation and training
kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
roc_auc_scores = []
models = []

for fold, (train_index, val_index) in enumerate(kf.split(X)):
    print(f"Fold {fold + 1}")
    X_train, X_val = X[train_index], X[val_index]
    y_train, y_val = y[train_index], y[val_index]

    model = RainfallModel(input_dim)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    best_roc_auc = 0
    patience = 50
    no_improvement_count = 0
    roc_history = [] 
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            roc_auc = roc_auc_score(y_val.numpy(), val_outputs.numpy())
            roc_history.append (roc_auc)   
        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            no_improvement_count = 0
        else:
            no_improvement_count +=1
            if no_improvement_count >= patience:
                print(f"Early stopping at epoch {epoch} with {best_roc_auc = }")
                break
    roc_auc_scores.append(best_roc_auc)
    if best_roc_auc > 0.85 :
        models.append (model)
    sns.lineplot (y = roc_history, x = list (range (len(roc_history))))
plt.show ()
print(f"Average ROC AUC across folds: {np.mean(roc_auc_scores):.4f}")
