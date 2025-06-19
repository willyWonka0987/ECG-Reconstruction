import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

import matplotlib.pyplot as plt
import joblib
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Config ---
leads_to_predict = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
model_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/MLP_with_metadata_models")
plot_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/MLP_with_metadata_plots")
metrics_file = Path("QRS_Triplet_Input_and_FullBeat_Target/MLP_with_metadata_metrics.txt")
model_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets ---
train_data = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_train.pkl")
test_data = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_test.pkl")

# --- Helper functions ---
def extract_features_and_targets(data, target_lead):
    X, y = [] , []
    for seg in data:
        qrs_I = seg['qrs_lead_I']
        qrs_II = seg['qrs_lead_II']
        if target_lead not in seg['other_leads'] or len(qrs_I) != 3 or len(qrs_II) != 3:
            continue
        try:
            (tq1, aq1), (tr1, ar1), (ts1, as1) = qrs_I
            (tq2, aq2), (tr2, ar2), (ts2, as2) = qrs_II
            age = seg.get("age", 0)
            sex = 1 if str(seg.get("sex", "M")).upper().startswith("M") else 0
            heart_axis = seg.get("heart_axis", 0)
            if isinstance(heart_axis, str):
                heart_axis = int(heart_axis) if heart_axis.isdigit() else 0
            hr = seg.get("hr", 0)
            features = [
                tq1, aq1, tr1, ar1, ts1, as1,
                tq2, aq2, tr2, ar2, ts2, as2,
                age, sex, heart_axis, hr
            ]
            target = seg['other_leads'][target_lead]
            if len(target) == 128:
                X.append(features)
                y.append(target)
        except Exception:
            continue
    return np.array(X), np.array(y)

def build_mlp_model(input_dim):
    inp = Input(shape=(input_dim,))
    x = Dense(128, activation='relu')(inp)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu')(x)
    x = Dropout(0.3)(x)
    x = Dense(64, activation='relu')(x)
    out = Dense(128, activation='linear')(x)
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

# --- Training Pipeline ---
with open(metrics_file, 'w') as f:
    for lead in leads_to_predict:
        print(f"\n🔧 Training model for Lead {lead}...")

        X_train_full, y_train_full = extract_features_and_targets(train_data, lead)
        X_test, y_test = extract_features_and_targets(test_data, lead)

        if X_train_full.size == 0 or X_test.size == 0:
            print(f"⚠️ No data for lead {lead}, skipping.")
            continue

        # Split train into train/val
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
            epochs=100,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6),
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

        f.write(f"Lead {lead}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R^2: {r2:.4f}\n")
        f.write(f"Pearson Correlation: {corr:.4f}\n\n")

        print(f"✅ Lead {lead}: RMSE={rmse:.4f}, R²={r2:.4f}, Corr={corr:.4f}")

        plot_prediction(
            y_true=y_test[0],
            y_pred=y_pred[0],
            lead_name=lead,
            save_path=plot_dir / f"lead_{lead}_prediction.png"
        )

        plot_loss_curve(
            history,
            lead_name=lead,
            save_path=plot_dir / f"lead_{lead}_loss_curve.png"
        )

print("\n🎉 All MLP models trained, metrics saved, and plots generated.")

