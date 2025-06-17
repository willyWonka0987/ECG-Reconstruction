import pickle
import matplotlib.pyplot as plt

# Load the cleaned ECG training data
with open("ecg_train_clean.pkl", "rb") as f:
    ecg_train = pickle.load(f)

# Get the first ECG sample
first_ecg = ecg_train[0]  # Shape should be (7, 12, N) where N is samples per segment

# Plotting
fig, axs = plt.subplots(7, 12, figsize=(20, 10), sharex=True, sharey=True)
fig.suptitle("First ECG: 7 Segments × 12 Leads", fontsize=16)

for seg_idx in range(7):
    for lead_idx in range(12):
        axs[seg_idx, lead_idx].plot(first_ecg[seg_idx][lead_idx], linewidth=0.8)
        axs[seg_idx, lead_idx].set_xticks([])
        axs[seg_idx, lead_idx].set_yticks([])
        if seg_idx == 0:
            axs[seg_idx, lead_idx].set_title(f"Lead {lead_idx+1}", fontsize=8)
        if lead_idx == 0:
            axs[seg_idx, lead_idx].set_ylabel(f"Seg {seg_idx+1}", fontsize=8)

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.show()
