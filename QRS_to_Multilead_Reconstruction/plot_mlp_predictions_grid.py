import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

import matplotlib.pyplot as plt
import joblib
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.losses import MeanSquaredError
from sklearn.preprocessing import StandardScaler

# --- Config ---
leads_to_predict = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6']
model_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/MLP_with_metadata_models")
plot_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/MLP_with_metadata_plots")
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets ---
test_data = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_test.pkl")

# --- Extract features ---
def extract_features_and_targets(data, target_lead):
    X, y = [], []
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

# --- Plotting multiple predictions ---
def plot_multiple_predictions(test_data, leads_to_predict, num_samples=10):
    fig, axes = plt.subplots(num_samples, len(leads_to_predict), figsize=(20, 2.5 * num_samples))
    if num_samples == 1:
        axes = np.expand_dims(axes, 0)  # ensure 2D shape

    for col_idx, lead in enumerate(leads_to_predict):
        X_test, y_test = extract_features_and_targets(test_data, lead)
        scaler = StandardScaler()
        X_test_scaled = scaler.fit_transform(X_test)  # note: may differ from training
        model_path = model_dir / f"mlp_model_lead_{lead}.h5"
        model = load_model(model_path, compile=False)
        model.compile(optimizer='adam', loss=MeanSquaredError(), metrics=['mae'])
        y_pred = model.predict(X_test_scaled)

        for row_idx in range(min(num_samples, len(y_test))):
            ax = axes[row_idx][col_idx]
            ax.plot(y_test[row_idx], label='Actual', linewidth=2)
            ax.plot(y_pred[row_idx], label='Predicted', linestyle='--')
            if row_idx == 0:
                ax.set_title(f"Lead {lead}")
            if col_idx == 0:
                ax.set_ylabel(f"Sample {row_idx + 1}")
            ax.grid(True)
            if row_idx == 0 and col_idx == len(leads_to_predict) - 1:
                ax.legend(loc='upper right')

    plt.tight_layout()
    plt.savefig(plot_dir / "first_10_predictions_all_leads.png")
    plt.close()

# --- Call plotting function ---
plot_multiple_predictions(test_data, leads_to_predict, num_samples=10)
