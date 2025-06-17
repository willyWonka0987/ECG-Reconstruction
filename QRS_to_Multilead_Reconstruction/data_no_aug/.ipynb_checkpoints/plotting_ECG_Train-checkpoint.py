import joblib

ecg_train = joblib.load('ecg_train_clean.pkl')

print(type(ecg_train))     # Should show <class 'numpy.ndarray'>
print(ecg_train.shape)     # Should be (N, 12, 128) or similar

import matplotlib.pyplot as plt

sample = ecg_train[0]  # shape: (12, 128) → 12 leads, 128 points

plt.figure(figsize=(12, 6))
for i in range(12):
    plt.plot(sample[i] + i * 2, label=f'Lead {i+1}')  # Offset each lead for clarity

plt.title("First ECG Sample - 12 Leads")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude + Offset")
plt.legend()
plt.tight_layout()
plt.show()
