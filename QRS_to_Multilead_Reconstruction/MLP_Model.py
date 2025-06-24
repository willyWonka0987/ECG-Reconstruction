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
from tensorflow.keras.metrics import MeanSquaredError
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# --- Load encoded feature names ---
encoded_feats_path = Path("encoded_feature_names.txt")
encoded_features = []
if encoded_feats_path.exists():
    with open(encoded_feats_path, 'r') as f:
        encoded_features = [line.strip() for line in f.readlines()]

# --- Config ---
leads_to_predict = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']  # Now includes V2
results_dir = Path("MLP_Model_Results_AVR")
model_dir = results_dir / "models"
plot_dir = results_dir / "plots"
metrics_file = results_dir / "metrics.txt"
results_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets ---
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
            pqrst_avr = seg['pqrst_lead_aVR']  # Changed from V2 to aVR
            
            # Extract features from I, II, aVR
            (_, ap1), (tq1, aq1), (tr1, ar1), (ts1, as1), (tt1, at1) = pqrst_I
            (_, ap2), (tq2, aq2), (tr2, ar2), (ts2, as2), (tt2, at2) = pqrst_II
            (_, ap3), (tq3, aq3), (tr3, ar3), (ts3, as3), (tt3, at3) = pqrst_avr
            
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
            
            # Removed stats/freq features for simplicity
            features = intervals + amplitudes + [age, sex, hr] + onehot_values
            target = seg['other_leads'][target_lead]
            
            if len(target) == 80:
                X.append(features)
                y.append(target)
        except Exception as e:
            # print(f"Error processing segment: {e}")
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

def plot_grid_predictions(test_data, leads_to_predict, num_samples=10):
    fig, axs = plt.subplots(num_samples, len(leads_to_predict), 
                            figsize=(25, 2.5 * num_samples))
    if num_samples == 1:
        axs = np.expand_dims(axs, 0)
    
    for lead_idx, lead in enumerate(leads_to_predict):
        X_test, y_test = extract_features_and_targets(test_data, lead)
        if len(X_test) == 0:
            continue
            
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)
        model = load_model(model_dir / f"mlp_model_lead_{lead}.h5", compile=False)
        model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
        y_pred = model.predict(X_test_scaled)
        
        for i in range(min(num_samples, len(y_test))):
            axs[i, lead_idx].plot(y_test[i], label='Actual', linewidth=1.5)
            axs[i, lead_idx].plot(y_pred[i], linestyle='--', label='Predicted', alpha=0.8)
            if i == 0:
                axs[i, lead_idx].set_title(f"Lead {lead}")
            if lead_idx == 0:
                axs[i, lead_idx].set_ylabel(f"Sample {i+1}")
            axs[i, lead_idx].set_xticks([])
            axs[i, lead_idx].set_yticks([])
    
    plt.tight_layout()
    plt.savefig(plot_dir / "grid_plot_predicted_vs_actual.png")
    plt.close()

# --- Training ---
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
                               save_best_only=True, monitor='val_loss')
            ],
            verbose=1
        )
        
        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        corr = pearsonr(y_test.flatten(), y_pred.flatten())[0]
        
        f.write(f"Lead {lead}\nRMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson Correlation: {corr:.4f}\n\n")
        print(f"✅ Lead {lead}: RMSE={rmse:.4f}, R²={r2:.4f}, Corr={corr:.4f}")
        
        # Plot sample prediction
        plt.figure(figsize=(10, 4))
        plt.plot(y_test[0], label='Actual', linewidth=2)
        plt.plot(y_pred[0], label='Predicted', linestyle='--')
        plt.title(f"Lead {lead}: Actual vs Predicted")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"lead_{lead}_prediction.png")
        plt.close()
        
        # Plot loss curve
        plt.figure(figsize=(8, 4))
        plt.plot(history.history['loss'], label='Train Loss')
        plt.plot(history.history['val_loss'], label='Val Loss')
        plt.title(f"Loss Curve for Lead {lead}")
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"lead_{lead}_loss_curve.png")
        plt.close()

# Plot grid predictions for all leads
plot_grid_predictions(test_data, leads_to_predict, num_samples=10)
print("\n🎉 All models trained and evaluated.")

