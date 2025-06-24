import os
import joblib
import matplotlib.pyplot as plt
from pathlib import Path

# --- Parameters ---
output_dir = Path("PQRST_Triplet_With_Stats_80")
plot_dir = output_dir / "segment_plots"
plot_dir.mkdir(parents=True, exist_ok=True)

lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# --- Load the processed training data ---
train_set = joblib.load(output_dir / "pqrst_stats_train_80.pkl")

# Sampling rate (from original extraction code)
sampling_rate = 100

for idx, entry in enumerate(train_set[:100]):
    fig, axs = plt.subplots(4, 1, figsize=(12, 12), sharex=True)
    fig.suptitle(f"Sample {idx} | Source Index: {entry['source_index']}", fontsize=16)
    
    # Get R-peak time for segment alignment
    r_time = entry['pqrst_lead_I'][2][0]  # R-peak time in seconds
    segment_start_time = r_time - 0.4  # Segment starts 0.4s before R-peak
    
    # Plot Lead I with PQRST markers
    times_i = [t for t, _ in entry['pqrst_lead_I']]
    positions_i = [(t - segment_start_time) * sampling_rate for t in times_i]
    axs[0].scatter(positions_i, [v for _, v in entry['pqrst_lead_I']], 
                   marker='o', color='red', label='PQRST')
    axs[0].set_ylabel('Lead I')
    axs[0].legend()
    
    # Plot Lead II with PQRST markers
    times_ii = [t for t, _ in entry['pqrst_lead_II']]
    positions_ii = [(t - segment_start_time) * sampling_rate for t in times_ii]
    axs[1].scatter(positions_ii, [v for _, v in entry['pqrst_lead_II']], 
                   marker='o', color='red', label='PQRST')
    axs[1].set_ylabel('Lead II')
    axs[1].legend()
    
    # Plot Lead aVR with PQRST markers
    times_avr = [t for t, _ in entry['pqrst_lead_aVR']]
    positions_avr = [(t - segment_start_time) * sampling_rate for t in times_avr]
    axs[2].scatter(positions_avr, [v for _, v in entry['pqrst_lead_aVR']], 
                   marker='o', color='red', label='PQRST')
    axs[2].set_ylabel('Lead aVR')
    axs[2].legend()
    
    # Plot other leads (full waveforms)
    x = range(80)  # 80-sample segment
    for lead in lead_names:
        if lead not in ['I', 'II', 'aVR'] and lead in entry['other_leads']:
            axs[3].plot(x, entry['other_leads'][lead], label=lead)
    axs[3].set_ylabel('Other Leads')
    axs[3].legend(loc='upper right', ncol=4, fontsize=8)
    axs[3].set_xlabel('Sample Index')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(plot_dir / f"segment_{idx:02d}.png")
    plt.close(fig)

