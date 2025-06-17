import joblib
import matplotlib.pyplot as plt

# Load the cleaned ECG training data
with open("ecg_train_clean.pkl", "rb") as f:
    ecg_train = joblib.load(f)

print("Data shape:", ecg_train.shape)  # (samples, 128, 12)

# Plot the first 5 ECG segments
for seg_idx in range(30):
    segment = ecg_train[seg_idx]  # Shape: (128, 12)
    
    plt.figure(figsize=(15, 10))
    for lead in range(12):
        plt.subplot(6, 2, lead + 1)
        plt.plot(segment[:, lead])
        plt.title(f"Lead {lead + 1}")
        plt.xticks([])
        plt.yticks([])
    
    plt.tight_layout()
    plt.suptitle(f"ECG Segment #{seg_idx + 1} (All 12 Leads)", fontsize=16, y=1.02)
    plt.show()
