import os
import joblib
import matplotlib.pyplot as plt
import neurokit2 as nk
import numpy as np

# --- Parameters ---
sampling_rate = 100  # Hz
pre_window = 0.08    # Q window (s)
post_window = 0.15   # S/T window (s)
p_window = 0.25      # Larger window before Q for P (s)
t_window = 0.4       # Larger window after S for T (s)
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Create output directory
os.makedirs("PQRST_plots", exist_ok=True)

# Load ECG dataset
ecg_data = joblib.load("../data_no_segmentation/ecg_test_clean.pkl")

def find_qspt(signal, r_peaks, sampling_rate, lead_name):
    pre_samples = int(pre_window * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    p_samples = int(p_window * sampling_rate)
    t_samples = int(t_window * sampling_rate)
    q_points, s_points, p_points, t_points = [], [], [], []

    for r in r_peaks:
        # --- Q ---
        q_start = max(0, r - pre_samples)
        q_seg = signal[q_start:r]
        if lead_name == 'aVR':
            q = q_start + np.argmax(q_seg) if len(q_seg) > 0 else r
        else:
            q = q_start + np.argmin(q_seg) if len(q_seg) > 0 else r
        q_points.append(q)

        # --- S ---
        s_end = min(len(signal), r + post_samples)
        s_seg = signal[r:s_end]
        if lead_name == 'aVR':
            s = r + np.argmax(s_seg) if len(s_seg) > 0 else r
        else:
            s = r + np.argmin(s_seg) if len(s_seg) > 0 else r
        s_points.append(s)

        # --- P ---
        if lead_name == 'aVR':
            p_start = max(0, q - p_samples)
            p_seg = signal[p_start:q]
            p = p_start + np.argmin(p_seg) if len(p_seg) > 0 else q
        else:
            p_start = max(0, q - p_samples)
            p_seg = signal[p_start:q]
            p = p_start + np.argmax(p_seg) if len(p_seg) > 0 else q
        p_points.append(p)

        # --- T ---
        if lead_name == 'aVR':
            t_end = min(len(signal), s + t_samples)
            t_seg = signal[s:t_end]
            t = s + np.argmin(t_seg) if len(t_seg) > 0 else s
        else:
            t_end = min(len(signal), s + t_samples)
            t_seg = signal[s:t_end]
            t = s + np.argmax(t_seg) if len(t_seg) > 0 else s
        t_points.append(t)

    return np.array(p_points), np.array(q_points), np.array(r_peaks), np.array(s_points), np.array(t_points)

# --- Loop through first 10 ECGs ---
for idx in range(10):
    ecg_12lead = ecg_data[idx]  # Shape: (1000, 12)

    plt.figure(figsize=(22, 30))
    for lead_idx in range(12):
        signal = ecg_12lead[:, lead_idx]
        lead_name = lead_names[lead_idx]

        # Flip signal for aVR to detect negative R peaks correctly
        flip = lead_name == 'aVR'
        raw_signal = -signal if flip else signal
        cleaned = nk.ecg_clean(raw_signal, sampling_rate=sampling_rate)
        r_peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
        cleaned = -cleaned if flip else cleaned  # Restore original polarity for plotting

        p_peaks, q_peaks, r_peaks, s_peaks, t_peaks = find_qspt(cleaned, r_peaks, sampling_rate, lead_name)

        # Plot
        plt.subplot(12, 1, lead_idx + 1)
        plt.plot(cleaned, label='ECG Signal', linewidth=1)
        if len(r_peaks) > 0:
            plt.plot(r_peaks, cleaned[r_peaks], 'ro', markersize=3, label='R-peaks')
        if len(q_peaks) > 0:
            plt.plot(q_peaks, cleaned[q_peaks], 'go', markersize=3, label='Q-points')
        if len(s_peaks) > 0:
            plt.plot(s_peaks, cleaned[s_peaks], 'bo', markersize=3, label='S-points')
        if len(p_peaks) > 0 and p_peaks[0] is not None:
            plt.plot(p_peaks, cleaned[p_peaks], 'mo', markersize=3, label='P-points')
        if len(t_peaks) > 0 and t_peaks[0] is not None:
            plt.plot(t_peaks, cleaned[t_peaks], 'co', markersize=3, label='T-points')

        plt.title(f"Lead {lead_name}")
        plt.ylabel("Amplitude")
        plt.grid(True)
        if lead_idx == 0:
            plt.legend(loc='upper right', ncol=5)

    plt.xlabel("Sample Index (0 to 999)")
    plt.tight_layout()
    plt.savefig(f"PQRST_plots/ecg_{idx:02d}_qrsp_t.png")
    plt.close()

print("✅ Saved QRS + P/T plots for the first 10 ECGs to 'PQRST_plots' folder.")

