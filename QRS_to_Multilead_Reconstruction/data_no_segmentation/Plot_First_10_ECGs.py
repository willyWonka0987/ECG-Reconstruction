import os
import pickle
import numpy as np
import matplotlib.pyplot as plt
import joblib


# Path setup
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PKL_PATH = os.path.join(BASE_DIR, 'ecg_train_clean.pkl')
SAVE_DIR = os.path.join(BASE_DIR, 'images')
os.makedirs(SAVE_DIR, exist_ok=True)

# Lead names
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Load ECG data
with open(PKL_PATH, 'rb') as f:
    ecgs = joblib.load("ecg_train_clean.pkl") # Shape: (N, 1000, 12) if 10s at 100 Hz
    print(ecgs.shape)  # Should be (N, 1000, 12)

print(f"✅ Loaded ECG data with shape: {ecgs.shape}")

# Plot first 10 ECGs
for i in range(min(10, len(ecgs))):
    fig, axes = plt.subplots(6, 2, figsize=(12, 10))
    fig.suptitle(f"ECG Sample #{i}", fontsize=16)
    axes = axes.ravel()
    
    for j in range(12):
        signal = ecgs[i, :, j]
        axes[j].plot(np.arange(len(signal)), signal, color='blue')
        axes[j].set_title(f"Lead {lead_names[j]}")
        axes[j].set_xlabel("Time (samples)")
        axes[j].set_ylabel("Amplitude (mV)")
        axes[j].grid(True)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    save_path = os.path.join(SAVE_DIR, f"ecg_sample_{i}.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📷 Saved: {save_path}")

print("✅ Done. All ECG images saved.")

