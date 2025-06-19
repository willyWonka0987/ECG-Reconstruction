import os
import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt

# --- Parameters ---
sampling_rate = 100
padding = 64
segment_length = 2 * padding
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# --- Input/Output Paths ---
input_train_path = Path("data_no_segmentation/ecg_train_clean.pkl")
input_test_path = Path("data_no_segmentation/ecg_test_clean.pkl")
meta_train_path = Path("train_split.csv")
meta_test_path = Path("test_split.csv")
output_dir = Path("PQRST_Triplet_Input_and_FullBeat_Target")
output_dir.mkdir(parents=True, exist_ok=True)

def find_qspt(signal, r_peaks, sampling_rate, pre_window=0.06, post_window=0.12):
    """
    Extract P, Q, R, S, T points from ECG signal based on R-peak locations.
    Args:
        signal: 1D ECG signal array
        r_peaks: Array of R-peak indices
        sampling_rate: Sampling rate in Hz
        pre_window: Window before R-peak for Q search (seconds)
        post_window: Window after R-peak for S search (seconds)
    Returns:
        Tuple of arrays: (p_points, q_points, r_peaks, s_points, t_points)
    """
    pre_samples = int(pre_window * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    q_points, s_points, p_points, t_points = [], [], [], []
    for r in r_peaks:
        # --- Q Point (minimum before R) ---
        q_start = max(0, r - pre_samples)
        q_seg = signal[q_start:r]
        q = q_start + np.argmin(q_seg) if len(q_seg) > 0 else r
        q_points.append(q)
        # --- S Point (minimum after R, wider window) ---
        s_end = min(len(signal), r + post_samples)
        s_seg = signal[r:s_end]
        s = r + np.argmin(s_seg) if len(s_seg) > 0 else r
        s_points.append(s)
        # --- P Point (maximum before Q) ---
        p_start = max(0, q - int(0.15 * sampling_rate))
        p_seg = signal[p_start:q]
        p = p_start + np.argmax(p_seg) if len(p_seg) > 0 else q
        p_points.append(p)
        # --- T Point (maximum after S) ---
        t_end = min(len(signal), s + int(0.4 * sampling_rate))
        t_seg = signal[s:t_end]
        t = s + np.argmax(t_seg) if len(t_seg) > 0 else s
        t_points.append(t)
    return np.array(p_points), np.array(q_points), np.array(r_peaks), np.array(s_points), np.array(t_points)

def build_combined_dataset(ecg_dataset, meta_df):
    """
    Build dataset with PQRST points from leads I & II plus full waveforms from other leads.
    Args:
        ecg_dataset: Array of ECG samples (shape: [n_samples, 1000, 12])
        meta_df: DataFrame with metadata (age, sex, heart_axis, hr)
    Returns:
        List of dictionaries containing PQRST data and metadata
    """
    final_dataset = []
    for i, sample in enumerate(tqdm(ecg_dataset, desc="Building PQRST + metadata dataset")):
        row_meta = meta_df.iloc[i]
        age = row_meta["age"]
        sex = row_meta["sex"]
        heart_axis = row_meta.get("heart_axis", 0)
        hr = row_meta.get("hr", 0)

        # --- Lead I Processing ---
        lead_i = sample[:, 0]
        cleaned_i = nk.ecg_clean(lead_i, sampling_rate=sampling_rate)
        r_peaks_i = nk.ecg_peaks(cleaned_i, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
        # Extract PQRST points from Lead I
        p_peaks_i, q_peaks_i, r_peaks_i, s_peaks_i, t_peaks_i = find_qspt(
            cleaned_i, r_peaks_i, sampling_rate
        )

        # --- Lead II Processing ---
        lead_ii = sample[:, 1]
        cleaned_ii = nk.ecg_clean(lead_ii, sampling_rate=sampling_rate)
        r_peaks_ii_data = nk.ecg_peaks(cleaned_ii, sampling_rate=sampling_rate)
        # Handle case where no R-peaks are found in lead II
        if 'ECG_R_Peaks' not in r_peaks_ii_data[1] or len(r_peaks_ii_data[1]['ECG_R_Peaks']) == 0:
            continue  # Skip this sample if no R-peaks in lead II
        r_peaks_ii = r_peaks_ii_data[1]['ECG_R_Peaks']
        # Extract PQRST points from Lead II
        p_peaks_ii, q_peaks_ii, r_peaks_ii, s_peaks_ii, t_peaks_ii = find_qspt(
            cleaned_ii, r_peaks_ii, sampling_rate
        )

        # --- Process Each R-peak ---
        for j, r in enumerate(r_peaks_i):
            if j >= len(p_peaks_i) or j >= len(q_peaks_i) or j >= len(s_peaks_i) or j >= len(t_peaks_i):
                continue
            p_i, q_i, s_i, t_i = p_peaks_i[j], q_peaks_i[j], s_peaks_i[j], t_peaks_i[j]
            # Find closest R-peak in lead II (with safety check)
            if len(r_peaks_ii) > 0:
                r_ii_idx = np.argmin(np.abs(r_peaks_ii - r))
                if r_ii_idx >= len(p_peaks_ii) or r_ii_idx >= len(q_peaks_ii) or r_ii_idx >= len(s_peaks_ii) or r_ii_idx >= len(t_peaks_ii):
                    continue
                r_ii = r_peaks_ii[r_ii_idx]
                p_ii, q_ii, s_ii, t_ii = p_peaks_ii[r_ii_idx], q_peaks_ii[r_ii_idx], s_peaks_ii[r_ii_idx], t_peaks_ii[r_ii_idx]
            else:
                continue  # Skip if no R-peaks in lead II

            # Ensure valid segment boundaries
            start, end = r - padding, r + padding
            if (start >= 0 and end <= sample.shape[0]) and all([
                p_i != r, q_i != r, s_i != r, t_i != r,  # Lead I points are distinct
                p_ii != r_ii, q_ii != r_ii, s_ii != r_ii, t_ii != r_ii  # Lead II points are distinct
            ]):
                # PQRST points for Lead I (time, amplitude)
                pqrst_lead_i = [
                    (p_i / sampling_rate, lead_i[p_i]),  # P point
                    (q_i / sampling_rate, lead_i[q_i]),  # Q point
                    (r / sampling_rate, lead_i[r]),      # R point
                    (s_i / sampling_rate, lead_i[s_i]),  # S point
                    (t_i / sampling_rate, lead_i[t_i])   # T point
                ]
                # PQRST points for Lead II (time, amplitude)
                pqrst_lead_ii = [
                    (p_ii / sampling_rate, lead_ii[p_ii]),  # P point
                    (q_ii / sampling_rate, lead_ii[q_ii]),  # Q point
                    (r_ii / sampling_rate, lead_ii[r_ii]),  # R point
                    (s_ii / sampling_rate, lead_ii[s_ii]),  # S point
                    (t_ii / sampling_rate, lead_ii[t_ii])   # T point
                ]
                # Extract full waveforms from other leads (unchanged)
                other_leads_waveforms = {
                    lead_names[k]: sample[start:end, k] for k in range(12) if k not in [0, 1]
                }
                final_dataset.append({
                    "pqrst_lead_I": pqrst_lead_i,
                    "pqrst_lead_II": pqrst_lead_ii,
                    "other_leads": other_leads_waveforms,
                    "age": age,
                    "sex": sex,
                    "heart_axis": heart_axis,
                    "hr": hr,
                    "source_index": i
                })
    return final_dataset

# --- Main Execution ---
if __name__ == "__main__":
    print("Loading ECG datasets...")
    train_data = joblib.load(input_train_path)
    test_data = joblib.load(input_test_path)
    print("Loading metadata...")
    train_meta = pd.read_csv(meta_train_path)
    test_meta = pd.read_csv(meta_test_path)
    print("Building combined datasets...")
    train_combined = build_combined_dataset(train_data, train_meta)
    test_combined = build_combined_dataset(test_data, test_meta)
    print("Saving datasets...")
    joblib.dump(train_combined, output_dir / "combined_pqrst_train.pkl")
    joblib.dump(test_combined, output_dir / "combined_pqrst_test.pkl")
    print(f"✅ Saved PQRST datasets:")
    print(f"   - Train: {len(train_combined)} samples")
    print(f"   - Test: {len(test_combined)} samples")
    print(f"   - Output directory: {output_dir}")

    # --- PLOTTING FIRST 10 SEGMENTS ---
    print("Plotting first 10 segments with PQRST points...")
    plot_output_folder = output_dir / "plots"
    plot_output_folder.mkdir(exist_ok=True)
    segments = train_combined
    lead_names_other = ['III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
    pqrst_colors = ['magenta', 'blue', 'red', 'green', 'cyan']
    pqrst_labels = ['P', 'Q', 'R', 'S', 'T']
    N = 10

    for idx, segment in enumerate(segments[:N]):
        fig, axs = plt.subplots(4, 3, figsize=(14, 10))
        axs = axs.flatten()

        # --- Plot PQRST points from Lead I ---
        pqrst_points_I = segment["pqrst_lead_I"]
        times_I = [t for (t, _) in pqrst_points_I]
        amps_I = [a for (_, a) in pqrst_points_I]
        axs[0].scatter(times_I, amps_I, c=pqrst_colors, label=[f"Lead I {l}" for l in pqrst_labels], s=60)
        for i, label in enumerate(pqrst_labels):
            axs[0].annotate(label, (times_I[i], amps_I[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
        axs[0].set_title("Lead I PQRST")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Amplitude")
        axs[0].legend()
        axs[0].grid(True)

        # --- Plot PQRST points from Lead II ---
        pqrst_points_II = segment.get("pqrst_lead_II")
        if pqrst_points_II is not None:
            times_II = [t for (t, _) in pqrst_points_II]
            amps_II = [a for (_, a) in pqrst_points_II]
            axs[1].scatter(times_II, amps_II, c=pqrst_colors, marker='x', label=[f"Lead II {l}" for l in pqrst_labels], s=60)
            for i, label in enumerate(pqrst_labels):
                axs[1].annotate(label, (times_II[i], amps_II[i]), textcoords="offset points", xytext=(0,8), ha='center', fontsize=9)
            axs[1].set_title("Lead II PQRST")
            axs[1].set_xlabel("Time (s)")
            axs[1].set_ylabel("Amplitude")
            axs[1].legend()
            axs[1].grid(True)
        else:
            axs[1].set_visible(False)

        # --- Plot other leads (heartbeat segments) ---
        for i, lead in enumerate(lead_names_other):
            ax_idx = i + 2  # Start plotting from axs[2]
            signal = segment["other_leads"].get(lead)
            if signal is not None:
                axs[ax_idx].plot(signal)
                axs[ax_idx].set_title(f"Lead {lead}")
                axs[ax_idx].grid(True)
            else:
                axs[ax_idx].set_visible(False)

        plt.tight_layout()
        plt.savefig(plot_output_folder / f"segment_{idx:04d}.png")
        plt.close()

    print(f"✅ Plots saved in '{plot_output_folder}'")

