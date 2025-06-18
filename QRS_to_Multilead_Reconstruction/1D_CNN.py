import matplotlib
matplotlib.use('Agg')  # Use a non-interactive backend (for headless systems)

import matplotlib.pyplot as plt
import joblib
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# --- Config ---
leads_to_predict = ['II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
model_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/CNN_models")
plot_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/CNN_first_segment_plots")
metrics_file = Path("QRS_Triplet_Input_and_FullBeat_Target/CNN_individual_model_metrics.txt")
model_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets ---
train_data = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_train.pkl")
test_data = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_test.pkl")

# --- Helper functions ---
def extract_features_and_targets(data, target_lead):
    X, y = [], []
    for seg in data:
        qrs = seg['qrs_lead_I']  # List of 3 tuples: [(tq, aq), (tr, ar), (ts, as)]
        
        # Skip if target lead is missing or malformed
        if target_lead not in seg['other_leads'] or len(qrs) != 3:
            continue
            
        try:
            # Unpack Q, R, S points
            (tq, aq), (tr, ar), (ts, as_) = qrs[0], qrs[1], qrs[2]
            
            # Compute timing intervals
            rq_interval = tr - tq
            sr_interval = ts - tr
            
            # Create feature vector
            features = [tq, aq, tr, ar, ts, as_, rq_interval, sr_interval]
            
            # Get target waveform
            target = seg['other_leads'][target_lead]
            if len(target) == 128:
                X.append(features)
                y.append(target)
        except (ValueError, TypeError) as e:
            print(f"Skipping malformed segment: {e}")
            continue
    
    return np.array(X), np.array(y)

def build_cnn_model():
    model = Sequential([
        Input(shape=(8, 1)),  # 8 input features reshaped to (8, 1)
        
        # Feature extraction blocks
        Conv1D(64, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        Conv1D(128, kernel_size=3, activation='relu', padding='same'),
        BatchNormalization(),
        MaxPooling1D(2),
        
        # Dense layers for regression
        Flatten(),
        Dense(256, activation='relu'),
        Dropout(0.3),
        Dense(128, activation='linear')  # Output 128 samples
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(0.001),
                loss='mse',
                metrics=['mae'])
    return model

def plot_prediction(y_true, y_pred, lead_name, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f"CNN Lead {lead_name}: Actual vs Predicted")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Training Pipeline ---
with open(metrics_file, 'w') as f:
    for lead in leads_to_predict:
        print(f"\n🔧 Training CNN model for Lead {lead}...")
        
        # Prepare data
        X_train, y_train = extract_features_and_targets(train_data, lead)
        X_test, y_test = extract_features_and_targets(test_data, lead)
        
        # Reshape for CNN (samples, timesteps, features)
        X_train = np.expand_dims(X_train, axis=-1)
        X_test = np.expand_dims(X_test, axis=-1)
        
        # Build and train model
        model = build_cnn_model()
        
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=100,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=model_dir / f"CNN_model_lead_{lead}.h5",
                    save_best_only=True,
                    monitor='val_loss'
                )
            ],
            verbose=1
        )
        
        # Evaluate
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        corr = pearsonr(y_test.flatten(), y_pred.flatten())[0]
        
        # Save metrics
        f.write(f"CNN Lead {lead}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R² Score: {r2:.4f}\n")
        f.write(f"Pearson Correlation: {corr:.4f}\n\n")
        
        print(f"✅ CNN Lead {lead}: RMSE={rmse:.4f}, R²={r2:.4f}, Corr={corr:.4f}")
        
        # Plot first test prediction
        plot_prediction(
            y_true=y_test[0],
            y_pred=y_pred[0],
            lead_name=lead,
            save_path=plot_dir / f"CNN_lead_{lead}_prediction.png"
        )

print("\n🎉 All CNN models trained, metrics saved, and plots generated.")
