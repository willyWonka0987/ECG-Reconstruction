import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import EarlyStopping

# Load QRS segments
train_segments = joblib.load("qrs_train_segments.pkl")
test_segments = joblib.load("qrs_test_segments.pkl")

lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
input_lead_idx = 0  # Lead I
target_leads_idx = [1, 6, 7, 8, 9, 10, 11]  # II, V1–V6

os.makedirs("regression_models", exist_ok=True)

def extract_features(lead_data):
    return np.array(lead_data).flatten()  # shape: (6,)

with open("regression_results.txt", "w") as f:
    for target_idx in target_leads_idx:
        X_train, Y_train, X_test, Y_test = [], [], [], []

        # Prepare train
        for beat in train_segments:
            if len(beat) <= max(input_lead_idx, target_idx):
                continue
            x = extract_features(beat[input_lead_idx])
            y = extract_features(beat[target_idx])
            X_train.append(x)
            Y_train.append(y)

        # Prepare test
        for beat in test_segments:
            if len(beat) <= max(input_lead_idx, target_idx):
                continue
            x = extract_features(beat[input_lead_idx])
            y = extract_features(beat[target_idx])
            X_test.append(x)
            Y_test.append(y)

        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)

        # Train/val split
        X_train_split, X_val, Y_train_split, Y_val = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)

        # Build model
        model = Sequential([
            Input(shape=(6,)),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(256, activation='relu'),
            Dense(6)
        ])
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Train
        es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        model.fit(X_train_split, Y_train_split, validation_data=(X_val, Y_val),
                  epochs=100, batch_size=32, verbose=1, callbacks=[es])

        # Validation Evaluation
        Y_val_pred = model.predict(X_val)
        mse_val = mean_squared_error(Y_val, Y_val_pred)
        rmse_val = np.sqrt(mse_val)
        r2_val = r2_score(Y_val, Y_val_pred)
        pearsons_val = [pearsonr(Y_val[:, i], Y_val_pred[:, i])[0] for i in range(6)]
        pearson_avg_val = np.mean(pearsons_val)

        # Test Evaluation
        Y_test_pred = model.predict(X_test)
        mse_test = mean_squared_error(Y_test, Y_test_pred)
        rmse_test = np.sqrt(mse_test)
        r2_test = r2_score(Y_test, Y_test_pred)
        pearsons_test = [pearsonr(Y_test[:, i], Y_test_pred[:, i])[0] for i in range(6)]
        pearson_avg_test = np.mean(pearsons_test)

        # Save model
        model_path = f"regression_models/model_predict_{lead_names[target_idx]}.keras"
        model.save(model_path)

        # Write results
        f.write(f"Model for Lead {lead_names[target_idx]}:\n")
        f.write(f"  [VAL]  RMSE: {rmse_val:.4f}, R2: {r2_val:.4f}, Pearson Corr: {pearson_avg_val:.4f}\n")
        f.write(f"  [TEST] RMSE: {rmse_test:.4f}, R2: {r2_test:.4f}, Pearson Corr: {pearson_avg_test:.4f}\n")
        f.write(f"  Model saved to: {model_path}\n\n")

print("✅ All models trained and evaluated.")
