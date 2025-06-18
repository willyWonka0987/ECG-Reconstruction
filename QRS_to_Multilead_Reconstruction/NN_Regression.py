import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend (for headless systems)

import matplotlib.pyplot as plt

import joblib
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# --- Config ---
leads_to_predict = ['II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
model_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/NN_models")
plot_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/NN_first_segment_plots")
metrics_file = Path("QRS_Triplet_Input_and_FullBeat_Target/individual_model_metrics.txt")
model_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets ---
train_data = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_train.pkl")
test_data = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_test.pkl")

# --- Helper functions ---
def extract_features_and_targets(data, target_lead):
    X, y = [], []
    for seg in data:
        qrs = seg['qrs_lead_I']  # This should be a list of 3 tuples: [(tq, aq), (tr, ar), (ts, as)]
        
        # Skip if target lead is missing
        if target_lead not in seg['other_leads']:
            continue
        
        # Skip if qrs doesn't have exactly 3 points
        if len(qrs) != 3:
            continue
            
        try:
            # Safer unpacking
            q_point = qrs[0]  # (tq, aq)
            r_point = qrs[1]  # (tr, ar)
            s_point = qrs[2]  # (ts, as)
            
            tq, aq = q_point
            tr, ar = r_point
            ts, as_ = s_point  # Note: 'as' is a Python keyword, so we use as_
            
            # Compute timing deltas (in seconds)
            rq_interval = tr - tq
            sr_interval = ts - tr
            
            # Final feature vector
            features = [tq, aq, tr, ar, ts, as_, rq_interval, sr_interval]
            
            target = seg['other_leads'][target_lead]
            if len(target) == 128:
                X.append(features)
                y.append(target)
        except (ValueError, TypeError) as e:
            print(f"Skipping malformed segment: {e}")
            continue
    
    return np.array(X), np.array(y)


def build_model(input_dim):
    model = Sequential([
        Input(shape=(input_dim,)),
        Dense(64, activation='relu'),
        Dense(128, activation='linear')  # Output: 128 samples
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

def plot_prediction(y_true, y_pred, lead_name, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f"Lead {lead_name}: Actual vs Predicted (First Test Segment)")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Train + Evaluate ---
with open(metrics_file, 'w') as f:
    for lead in leads_to_predict:
        print(f"\n🔧 Training model for Lead {lead}...")
        X_train, y_train = extract_features_and_targets(train_data, lead)
        X_test, y_test = extract_features_and_targets(test_data, lead)

        model = build_model(input_dim=X_train.shape[1])
        model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=50,
            batch_size=64,
            callbacks=[EarlyStopping(patience=5, restore_best_weights=True)],
            verbose=0
        )

        # Save model
        model_path = model_dir / f"model_lead_{lead}.keras"
        model.save(model_path)

        # Predict on test set
        y_pred = model.predict(X_test)

        # Metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        corr = pearsonr(y_test.flatten(), y_pred.flatten())[0]

        # Save metrics
        f.write(f"Lead {lead}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"Pearson Correlation: {corr:.4f}\n\n")

        print(f"✅ Lead {lead}: RMSE={rmse:.4f}, R²={r2:.4f}, Corr={corr:.4f}")

        # Plot for the first test sample
        plot_prediction(
            y_true=y_test[0],
            y_pred=y_pred[0],
            lead_name=lead,
            save_path=plot_dir / f"lead_{lead}_prediction.png"
        )

print("\nAll models trained, metrics saved, and plots generated.")

