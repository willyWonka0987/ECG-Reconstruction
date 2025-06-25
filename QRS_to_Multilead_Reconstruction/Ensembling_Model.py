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

            (_, ap1), (tq1, aq1), (tr1, ar1), (ts1, as1), (tt1, at1) = pqrst_I
            (_, ap2), (tq2, aq2), (tr2, ar2), (ts2, as2), (tt2, at2) = pqrst_II
            (_, ap3), (tq3, aq3), (tr3, ar3), (ts3, as3), (tt3, at3) = pqrst_V2

            intervals = [
                tr1 - tq1, ts1 - tq1, tt1 - tq1,
                tr2 - tq2, ts2 - tq2, tt2 - tq2,
                tr3 - tq3, ts3 - tq3, tt3 - tq3
            ]

            amplitudes = [
                aq1, ar1, as1, at1,
                aq2, ar2, as2, at2,
                aq3, ar3, as3, at3
            ]

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
                intervals + amplitudes +
                [age, sex, hr] + onehot_values +
                stats_features + freq_features
            )

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

# --- Training ---
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

        # --- Base Models ---
        xgb_model = MultiOutputRegressor(XGBRegressor(n_estimators=150, learning_rate=0.1, max_depth=6, verbosity=0))
        xgb_model.fit(X_train_scaled, y_train)

        mlp_model = build_mlp_model(input_dim=X_train_scaled.shape[1])
        mlp_model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=100, batch_size=64,
            callbacks=[EarlyStopping(patience=10, restore_best_weights=True)],
            verbose=1
        )
        y_pred_mlp_train = mlp_model.predict(X_train_scaled)
        y_pred_xgb_train = xgb_model.predict(X_train_scaled)

        # --- Meta Model ---
        meta_X_train = np.hstack([y_pred_mlp_train, y_pred_xgb_train])
        meta_model = Ridge(alpha=1.0)
        meta_model.fit(meta_X_train, y_train)

        # --- Predict ---
        y_pred_mlp_test = mlp_model.predict(X_test_scaled)
        y_pred_xgb_test = xgb_model.predict(X_test_scaled)
        meta_X_test = np.hstack([y_pred_mlp_test, y_pred_xgb_test])
        y_pred_stack = meta_model.predict(meta_X_test)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred_stack))
        r2 = r2_score(y_test, y_pred_stack)
        corr = pearsonr(y_test.flatten(), y_pred_stack.flatten())[0]

        f.write(f"Lead {lead}\nRMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson Correlation: {corr:.4f}\n\n")
        print(f"✅ Lead {lead}: RMSE={rmse:.4f}, R²={r2:.4f}, Corr={corr:.4f}")

        plt.figure(figsize=(10, 4))
        plt.plot(y_test[0], label='Actual', linewidth=2)
        plt.plot(y_pred_stack[0], label='Stacked', linestyle='--')
        plt.title(f"Lead {lead}: Actual vs Stacked Prediction")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"lead_{lead}_stacked_prediction.png")
        plt.close()

print("\n🎉 All stacked ensemble models trained and evaluated.")

