import pickle
import matplotlib.pyplot as plt
import numpy as np

# Load the ECG data
with open('ecg_train_clean.pkl', 'rb') as f:
    ecg_train = pickle.load(f)

print("ECG train shape:", ecg_train.shape)  # Should be (N, 128, 12)

# Select the first ECG sample (shape: [128, 12])
first_ecg = ecg_train[0]

plt.figure(figsize=(12, 8))

# Plot each lead
for i in range(12):
    plt.plot(first_ecg[:, i], label=f'Lead {i+1}')

plt.title("First ECG Sample (128 points for 12 leads)")
plt.xlabel("Time Step (sample index)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
