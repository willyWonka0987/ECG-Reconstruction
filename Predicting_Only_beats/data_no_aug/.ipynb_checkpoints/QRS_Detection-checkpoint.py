import joblib
import matplotlib.pyplot as plt
import neurokit2 as nk

# Load the first ECG
ecg_data = joblib.load("data_no_segmentation/ecg_train_clean.pkl")
first_ecg = ecg_data[0]  # Shape: (1000, 12)

# Sampling rate is 100 Hz for PTB-XL
sampling_rate = 100
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Plot each lead with detected R-peaks
plt.figure(figsize=(15, 10))
for lead in range(12):
    signal = first_ecg[:, lead]

    # Process ECG using NeuroKit2
    try:
        processed = nk.ecg_process(signal, sampling_rate=sampling_rate)
        rpeaks = processed[1]['ECG_R_Peaks']  # Indexes of R-peaks
    except Exception as e:
        print(f"Failed to process Lead {lead_names[lead]}: {e}")
        rpeaks = []

    plt.subplot(6, 2, lead + 1)
    plt.plot(signal, label='ECG', linewidth=1)
    if len(rpeaks) > 0:
        plt.plot(rpeaks, signal[rpeaks], 'ro', markersize=3, label='R-peaks')
    plt.title(f"Lead {lead_names[lead]}")
    plt.xticks([])
    plt.yticks([])

plt.suptitle("First Cleaned ECG with Detected R-peaks (Pan-Tompkins)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()
