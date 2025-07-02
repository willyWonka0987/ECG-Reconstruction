import os
os.environ.pop("CUDA_VISIBLE_DEVICES", None)  # Enable GPU usage

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor

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
                seg.get("qrs_area_I", 0), seg.get("qrs_area_II", 0), seg.get("qrs_area_V2", 0),
                seg.get("qrs_dur_I", 0), seg.get("qrs_dur_II", 0), seg.get("qrs_dur_V2", 0),
                seg.get("t_area_I", 0), seg.get("t_area_II", 0), seg.get("t_area_V2", 0)
            ]

            slope_i = seg.get("slope_lead_I", {})
            slope_ii = seg.get("slope_lead_II", {})
            slope_v2 = seg.get("slope_lead_V2", {})
            slope_features = list(slope_i.values()) + list(slope_ii.values()) + list(slope_v2.values())

            age = seg.get("age", 0)
            sex = 1 if str(seg.get("sex", "M")).upper().startswith("M") else 0
            hr = seg.get("hr", 0)
            onehot_values = [seg.get(name, 0) for name in encoded_features]

            stats_features = list(seg['stats_lead_I'].values()) + list(seg['stats_lead_II'].values()) + list(seg['stats_lead_V2'].values())
            freq_features = list(seg['freq_lead_I'].values()) + list(seg['freq_lead_II'].values()) + list(seg['freq_lead_V2'].values())

            features = intervals + amplitudes + qrs_t_features + slope_features + [age, sex, hr] + onehot_values + stats_features + freq_features
            target = seg['other_leads'][target_lead]

            if len(target) == 80:
                X.append(features)
                y.append(target)
        except Exception:
            continue
    return np.array(X), np.array(y)

def build_mlp_model(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(256, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(80, activation='linear')(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

with open(metrics_file, 'w') as f:
    for lead in leads_to_predict:
        print(f"\n🔧 Training stacking ensemble for Lead {lead}...")
        X_train_full, y_train_full = extract_features_and_targets(train_data, lead)
        X_test, y_test = extract_features_and_targets(test_data, lead)

        if X_train_full.size == 0 or X_test.size == 0:
            print(f"⚠️ No data for lead {lead}, skipping.")
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        xgb_model = MultiOutputRegressor(
            XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, verbosity=0)
        )
        xgb_model.fit(X_train_scaled, y_train)

        mlp_model = build_mlp_model(input_dim=X_train_scaled.shape[1])
        mlp_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100, batch_size=64,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1
        )

        y_pred_mlp_test = mlp_model.predict(X_test_scaled)
        y_pred_xgb_test = xgb_model.predict(X_test_scaled)
        meta_X_test = np.hstack([y_pred_mlp_test, y_pred_xgb_test])

        meta_X_train = np.hstack([
            mlp_model.predict(X_train_scaled),
            xgb_model.predict(X_train_scaled)
        ])
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_X_train, y_train)

        y_pred_stack = meta_model.predict(meta_X_test)

        mse_per_time = np.mean((y_test - y_pred_stack) ** 2, axis=0)
        rmse_per_time = np.sqrt(mse_per_time)

        rmse = np.sqrt(np.mean((y_test - y_pred_stack) ** 2))
        r2 = r2_score(y_test, y_pred_stack)
        corr = pearsonr(y_test.flatten(), y_pred_stack.flatten())[0]

        f.write(f"Lead {lead}\nRMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson Correlation: {corr:.4f}\nRMSE per time point: {rmse_per_time.tolist()}\n\n")
        print(f"✅ Lead {lead}: RMSE={rmse:.4f}, R²={r2:.4f}, Corr={corr:.4f}")

        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        for i in range(10):
            row, col = divmod(i, 5)
            axes[row, col].plot(y_test[i], label='Actual', linewidth=2)
            axes[row, col].plot(y_pred_stack[i], label='Predicted', linestyle='--')
            axes[row, col].set_title(f'Sample {i+1}')
            axes[row, col].grid(True)
        axes[0, 0].legend()
        plt.suptitle(f"Lead {lead}: First 10 Test Predictions")
        plt.tight_layout()
        plt.savefig(plot_dir / f"lead_{lead}_test_predictions_grid.png")
        plt.close()

print("\n🎉 All stacked ensemble models trained and evaluated.")