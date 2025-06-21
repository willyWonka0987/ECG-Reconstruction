import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Load encoded feature names ---
encoded_feats_path = Path("encoded_feature_names.txt")
if encoded_feats_path.exists():
    with open(encoded_feats_path, 'r') as f:
        encoded_features = [line.strip() for line in f.readlines()]
else:
    encoded_features = []

# --- Config ---
leads_to_predict = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
results_dir = Path("MLP_Model_Results")
model_dir = results_dir / "models"
plot_dir = results_dir / "plots"
metrics_file = results_dir / "metrics.txt"
results_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets (updated paths for 80 sample segments) ---
train_data = joblib.load("PQRST_Triplet_With_Stats_80/pqrst_stats_train_80.pkl")
test_data = joblib.load("PQRST_Triplet_With_Stats_80/pqrst_stats_test_80.pkl")

def extract_features_and_targets(data, target_lead):
    X, y = [], []
    for seg in data:
        if target_lead not in seg['other_leads']:
            continue
        try:
            pqrst_I = seg['pqrst_lead_I']
            pqrst_II = seg['pqrst_lead_II']
            (_, ap1), (tq1, aq1), (tr1, ar1), (ts1, as1), (tt1, at1) = pqrst_I
            (_, ap2), (tq2, aq2), (tr2, ar2), (ts2, as2), (tt2, at2) = pqrst_II

            pr_interval_I = tr1 - tq1
            qrs_duration_I = ts1 - tq1
            qt_interval_I = tt1 - tq1
            pr_interval_II = tr2 - tq2
            qrs_duration_II = ts2 - tq2
            qt_interval_II = tt2 - tq2

            age = seg.get("age", 0)
            sex = 1 if str(seg.get("sex", "M")).upper().startswith("M") else 0
            hr = seg.get("hr", 0)
            onehot_values = [seg.get(name, 0) for name in encoded_features]

            stats_i = seg['stats_lead_I']
            stats_ii = seg['stats_lead_II']
            freq_i = seg['freq_lead_I']
            freq_ii = seg['freq_lead_II']

            features = [
                pr_interval_I, qrs_duration_I, qt_interval_I,
                pr_interval_II, qrs_duration_II, qt_interval_II,
                aq1, ar1, as1, at1, aq2, ar2, as2, at2,
                age, sex, hr
            ] + onehot_values \
              + list(stats_i.values()) + list(stats_ii.values()) \
              + list(freq_i.values()) + list(freq_ii.values())

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
    x = Dropout(0.4)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(80, activation='linear')(x)
    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer='adam', loss='mse', metrics=['mae'])
    return model

def plot_prediction(y_true, y_pred, lead_name, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f"Lead {lead_name}: Actual vs Predicted")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_curve(history, lead_name, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"Loss Curve for Lead {lead_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

with open(metrics_file, 'w') as f:
    for lead in leads_to_predict:
        print(f"\n🔧 Training model for Lead {lead}...")
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

        model = build_mlp_model(input_dim=X_train_scaled.shape[1])
        history = model.fit(
            X_train_scaled, y_train,
            validation_data=(X_val_scaled, y_val),
            epochs=150,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=20, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10, min_lr=1e-6),
                ModelCheckpoint(filepath=model_dir / f"mlp_model_lead_{lead}.h5",
                                save_best_only=True,
                                monitor='val_loss')
            ],
            verbose=1
        )

        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        corr = pearsonr(y_test.flatten(), y_pred.flatten())[0]

        f.write(f"Lead {lead}\nRMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson Correlation: {corr:.4f}\n\n")
        print(f"✅ Lead {lead}: RMSE={rmse:.4f}, R²={r2:.4f}, Corr={corr:.4f}")

        plot_prediction(y_test[0], y_pred[0], lead, plot_dir / f"lead_{lead}_prediction.png")
        plot_loss_curve(history, lead, plot_dir / f"lead_{lead}_loss_curve.png")

print("\n🎉 All models trained and evaluated.")

