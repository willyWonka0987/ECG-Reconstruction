import os
import joblib
import matplotlib.pyplot as plt

# --- Parameters ---
output_folder = "Plot_First_10_Segments_Of_Dataset"
os.makedirs(output_folder, exist_ok=True)

# --- Load combined QRS segments ---
segments = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_train.pkl")
print(f"Number of segments: {len(segments)}")

# --- Lead names (excluding I and II) ---
lead_names = ['III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# --- Plot and save first N segments ---
N = 10
for idx, segment in enumerate(segments[:N]):
    fig, axs = plt.subplots(4, 3, figsize=(14, 10))
    axs = axs.flatten()

    # --- Plot QRS points from Lead I ---
    qrs_points_I = segment["qrs_lead_I"]
    q_colors = ['blue', 'red', 'green']
    q_labels = ['Q', 'R', 'S']
    times_I = [t for (t, _) in qrs_points_I]
    amps_I = [a for (_, a) in qrs_points_I]
    axs[0].scatter(times_I, amps_I, c=q_colors, label=[f"Lead I {l}" for l in q_labels], s=60)
    axs[0].set_title("Lead I QRS")
    axs[0].set_xlabel("Time (s)")
    axs[0].set_ylabel("Amplitude")
    axs[0].legend()
    axs[0].grid(True)

    # --- Plot QRS points from Lead II ---
    qrs_points_II = segment.get("qrs_lead_II")
    if qrs_points_II is not None:
        times_II = [t for (t, _) in qrs_points_II]
        amps_II = [a for (_, a) in qrs_points_II]
        axs[1].scatter(times_II, amps_II, c=q_colors, label=[f"Lead II {l}" for l in q_labels], marker='x', s=60)
        axs[1].set_title("Lead II QRS")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Amplitude")
        axs[1].legend()
        axs[1].grid(True)
    else:
        axs[1].set_visible(False)

    # --- Plot other leads (heartbeat segments) ---
    for i, lead in enumerate(lead_names):
        ax_idx = i + 2  # Start plotting from axs[2]
        signal = segment["other_leads"].get(lead)
        if signal is not None:
            axs[ax_idx].plot(signal)
            axs[ax_idx].set_title(f"Lead {lead}")
            axs[ax_idx].grid(True)
        else:
            axs[ax_idx].set_visible(False)

    plt.tight_layout()
    plt.savefig(os.path.join(output_folder, f"segment_{idx:04d}.png"))
    plt.close()

print(f"✅ Plots saved in '{output_folder}'")

