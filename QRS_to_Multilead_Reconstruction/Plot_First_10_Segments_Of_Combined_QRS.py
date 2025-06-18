import os
import joblib
import matplotlib.pyplot as plt

# --- Parameters ---
output_folder = "Plot_First_10_Segments_Of_Dataset"
os.makedirs(output_folder, exist_ok=True)

# --- Load combined QRS segments ---
segments = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_train.pkl")
print(f"Number of segments: {len(segments)}")

# --- Lead names (excluding I) ---
lead_names = ['II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# --- Plot and save first N segments ---
N = 10
for idx, segment in enumerate(segments[:N]):
    fig, axs = plt.subplots(4, 3, figsize=(12, 8))
    axs = axs.flatten()

    # --- Plot QRS points from Lead I ---
    qrs_points = segment["qrs_lead_I"]
    q_colors = ['blue', 'red', 'green']
    q_labels = ['Q', 'R', 'S']

    times = [t for (t, _) in qrs_points]
    amps = [a for (_, a) in qrs_points]

    axs[0].scatter(times, amps, c=q_colors, label=q_labels, s=50)
    axs[0].set_title("Lead I QRS points")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid(True)

    # --- Plot other leads (128-sample beats) ---
    for i, lead in enumerate(lead_names):
        signal = segment["other_leads"].get(lead)
        if signal is not None:
            axs[i + 1].plot(signal)
            axs[i + 1].set_title(f"Lead {lead}")
            axs[i + 1].grid(True)
        else:
            axs[i + 1].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"segment_{idx:04d}.png"))
    plt.close()

print(f"✅ Plots saved in '{output_folder}'")

