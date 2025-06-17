import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import joblib  # if you're not using utils.py

# === Load test data ===
ecg_test = joblib.load(open("/home/youssef/Documents/ecg_reconstruction-main_v1/data/ecg_test_clean.pkl", "rb"))

# Input to model: Lead I (channel 0)
X_test = ecg_test[:, :, 0]

# True target: Lead II (channel 1)
y_test = ecg_test[:, :, 1]

# === Load trained model for Lead II ===
model_path = "model_II.keras"
model_ii = load_model(model_path)

# === Predict for the first sample ===
predicted_lead2 = model_ii.predict(X_test[0:1])[0]  # shape: (128,)
true_lead2 = y_test[0]                              # shape: (128,)

# === Plot prediction vs ground truth ===
plt.figure(figsize=(10, 4))
plt.plot(true_lead2, label="True Lead II", linewidth=2)
plt.plot(predicted_lead2, label="Predicted Lead II", linestyle='--')
plt.title("Reconstruction of Lead II from Lead I (First Test Sample)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

