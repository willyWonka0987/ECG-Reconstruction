import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from tqdm import tqdm

from soft_dtw import SoftDTW  # custom differentiable DTW

# --- Load encoded feature names ---
encoded_feats_path = Path("encoded_feature_names.txt")
encoded_features = []
if encoded_feats_path.exists():
    with open(encoded_feats_path, 'r') as f:
        encoded_features = [line.strip() for line in f.readlines()]

# --- Config ---
leads_to_predict = ['V1', 'V3', 'V4', 'V5', 'V6']
results_dir = Path("Stacking_Ensemble_Results")
model_dir = results_dir / "models"
plot_dir = results_dir / "plots"
metrics_file = results_dir / "metrics.txt"
results_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets ---
train_data = joblib.load("PQRST_80_Datasets/pqrst_stats_train_80.pkl")
test_data = joblib.load("PQRST_80_Datasets/pqrst_stats_test_80.pkl")

def extract_features_and_targets(data, target_lead):
    X, y = [], []
    for seg in data:
        if target_lead not in seg['other_leads']:
            continue
        try:
            pqrst_I = seg['pqrst_lead_I']
            pqrst_II = seg['pqrst_lead_II']
            pqrst_V2 = seg['pqrst_lead_V2']

            (tp1, ap1), (tq1, aq1), (tr1, ar1), (ts1, as1), (tt1, at1) = pqrst_I
            (tp2, ap2), (tq2, aq2), (tr2, ar2), (ts2, as2), (tt2, at2) = pqrst_II
            (tp3, ap3), (tq3, aq3), (tr3, ar3), (ts3, as3), (tt3, at3) = pqrst_V2

            intervals = [
                tr1 - tp1, tt1 - ts1, tt1 - tq1, seg.get("rr1", 0),
                tr2 - tp2, tt2 - ts2, tt2 - tq2, seg.get("rr2", 0),
                tr3 - tp3, tt3 - ts3, tt3 - tq3, seg.get("rr3", 0)
            ]
            amplitudes = [
                aq1, ar1, as1, at1,
                aq2, ar2, as2, at2,
                aq3, ar3, as3, at3
            ]
            qrs_t_features = [
                seg.get("qrs_area_I", 0),
                seg.get("qrs_area_II", 0),
                seg.get("qrs_area_V2", 0),
                seg.get("qrs_dur_I", 0),
                seg.get("qrs_dur_II", 0),
                seg.get("qrs_dur_V2", 0),
                seg.get("t_area_I", 0),
                seg.get("t_area_II", 0),
                seg.get("t_area_V2", 0)
            ]
            slope_i = seg.get("slope_lead_I", {})
            slope_ii = seg.get("slope_lead_II", {})
            slope_v2 = seg.get("slope_lead_V2", {})
            slope_features = (
                list(slope_i.values()) +
                list(slope_ii.values()) +
                list(slope_v2.values())
            )
            age = seg.get("age", 0)
            sex = 1 if str(seg.get("sex", "M")).upper().startswith("M") else 0
            hr = seg.get("hr", 0)
            onehot_values = [seg.get(name, 0) for name in encoded_features]
            stats_features = (
                list(seg['stats_lead_I'].values()) +
                list(seg['stats_lead_II'].values()) +
                list(seg['stats_lead_V2'].values())
            )
            freq_features = (
                list(seg['freq_lead_I'].values()) +
                list(seg['freq_lead_II'].values()) +
                list(seg['freq_lead_V2'].values())
            )
            features = (
                intervals +
                amplitudes +
                qrs_t_features +
                slope_features +
                [age, sex, hr] +
                onehot_values +
                stats_features +
                freq_features
            )
            target = seg['other_leads'][target_lead]
            if len(target) == 80:
                X.append(features)
                y.append(target)
        except Exception:
            continue
    return np.array(X), np.array(y)

class MLP(nn.Module):
    def __init__(self, input_dim):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 80)
        )

    def forward(self, x):
        return self.net(x)

with open(metrics_file, 'w') as f:
    for lead in leads_to_predict:
        print(f"\nTraining ensemble for Lead {lead}...")
        X_train_full, y_train_full = extract_features_and_targets(train_data, lead)
        X_test, y_test = extract_features_and_targets(test_data, lead)

        if X_train_full.size == 0 or X_test.size == 0:
            print(f"No data for lead {lead}, skipping.")
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42
        )


        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32).to(device)
        y_train_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        X_val_tensor = torch.tensor(X_val_scaled, dtype=torch.float32).to(device)
        y_val_tensor = torch.tensor(y_val, dtype=torch.float32).to(device)
        X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32).to(device)


        train_ds = TensorDataset(X_train_tensor, y_train_tensor)
        val_ds = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(train_ds, batch_size=64, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_ds, batch_size=64)

        model = MLP(X_train_tensor.shape[1]).to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        mse_loss_fn = nn.MSELoss()
        dtw_loss_fn = SoftDTW(gamma=1.0, normalize=True)

        model.train()
        for epoch in range(1, 101):
            total_loss = 0
            loop = tqdm(train_loader, desc=f"Epoch {epoch}/100", leave=False)
            for xb, yb in train_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                print("Batch loaded")
                pred = model(xb)
                dtw = dtw_loss_fn(pred.unsqueeze(-1), yb.unsqueeze(-1)).mean()
                loss = mse_loss_fn(pred, yb) + dtw
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch}/50 - Loss: {total_loss/len(train_loader):.4f}")

        model.eval()
        with torch.no_grad():
            y_pred_test = model(X_test_tensor).cpu().numpy()

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        r2 = r2_score(y_test, y_pred_test)
        corr = pearsonr(y_test.flatten(), y_pred_test.flatten())[0]
        y_pred_tensor = torch.tensor(y_pred_test[:, :, np.newaxis], dtype=torch.float32).to(device)
        y_test_tensor = torch.tensor(y_test[:, :, np.newaxis], dtype=torch.float32).to(device)

        dtw_score_tensor = dtw_loss_fn(y_pred_tensor, y_test_tensor).mean()

        f.write(f"Lead {lead}\nRMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson Correlation: {corr:.4f}\nSoftDTW: {dtw_score_tensor.item():.4f}\n\n")
        print(f"Lead {lead}: RMSE={rmse:.4f}, RÂ²={r2:.4f}, Corr={corr:.4f}, SoftDTW={dtw_score_tensor.item():.4f}")

        plt.figure(figsize=(10, 4))
        plt.plot(y_test[0], label='Actual', linewidth=2)
        plt.plot(y_pred_test[0], label='Predicted', linestyle='--')
        plt.title(f"Lead {lead}: Actual vs Predicted")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"lead_{lead}_prediction.png")
        plt.close()

print("\nAll models trained using MSE + SoftDTW loss.")
