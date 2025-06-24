import os
import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from scipy.stats import skew, kurtosis, entropy
from scipy.signal import welch
import pywt
import matplotlib.pyplot as plt

# --- Parameters ---
sampling_rate = 100
padding = 40  # 80 samples total, centered on R-peak
segment_length = 2 * padding
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Detection parameters (from Test_PQRST.py)
pre_window = 0.08   # Q window (s)
post_window = 0.15   # S/T window (s)
p_window = 0.25      # P window (s)
t_window = 0.4       # T window (s)

# --- Input/Output Paths ---
input_train_path = Path("data_no_segmentation/ecg_train_clean.pkl")
input_test_path = Path("data_no_segmentation/ecg_test_clean.pkl")
meta_train_path = Path("train_split.csv")
meta_test_path = Path("test_split.csv")
output_dir = Path("PQRST_Triplet_With_Stats_80")
output_dir.mkdir(parents=True, exist_ok=True)
plot_dir = output_dir / "segment_plots"
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load encoded feature names ---
encoded_feats_path = Path("encoded_feature_names.txt")
if encoded_feats_path.exists():
    with open(encoded_feats_path, 'r') as f:
        encoded_features = [line.strip() for line in f.readlines()]
else:
    encoded_features = []

def detect_pqrst(signal, r_peaks, sampling_rate, lead_name):
    """
    Robust PQRST detector based on Test_PQRST.py implementation
    Handles aVR's inverted physiology with specialized logic
    """
    pre_samples = int(pre_window * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    p_samples = int(p_window * sampling_rate)
    t_samples = int(t_window * sampling_rate)

    p_points, q_points, s_points, t_points = [], [], [], []

    for r in r_peaks:
        # --- Q point detection ---
        q_start = max(0, r - pre_samples)
        q_seg = signal[q_start:r]
        
        # aVR requires inverted logic (positive deflection)
        if lead_name == 'aVR':
            q = q_start + np.argmax(q_seg) if len(q_seg) > 0 else r
        else:
            q = q_start + np.argmin(q_seg) if len(q_seg) > 0 else r
        q_points.append(q)

        # --- S point detection ---
        s_end = min(len(signal), r + post_samples)
        s_seg = signal[r:s_end]
        
        # aVR requires inverted logic
        if lead_name == 'aVR':
            s = r + np.argmax(s_seg) if len(s_seg) > 0 else r
        else:
            s = r + np.argmin(s_seg) if len(s_seg) > 0 else r
        s_points.append(s)

        # --- P point detection ---
        p_start = max(0, q - p_samples)
        p_seg = signal[p_start:q]
        
        # aVR requires inverted logic
        if lead_name == 'aVR':
            p = p_start + np.argmin(p_seg) if len(p_seg) > 0 else q
        else:
            p = p_start + np.argmax(p_seg) if len(p_seg) > 0 else q
        p_points.append(p)

        # --- T point detection ---
        t_end = min(len(signal), s + t_samples)
        t_seg = signal[s:t_end]
        
        # aVR requires inverted logic
        if lead_name == 'aVR':
            t = s + np.argmin(t_seg) if len(t_seg) > 0 else s
        else:
            t = s + np.argmax(t_seg) if len(t_seg) > 0 else s
        t_points.append(t)

    return (
        np.array(p_points), 
        np.array(q_points), 
        np.array(r_peaks), 
        np.array(s_points), 
        np.array(t_points)
    )

def build_dataset_with_stats(ecg_dataset, meta_df):
    dataset = []
    for i, sample in enumerate(tqdm(ecg_dataset, desc="Building dataset with stats (80 samples)")):
        row_meta = meta_df.iloc[i]
        age = row_meta.get("age", 0)
        sex = row_meta.get("sex", 0)
        hr = row_meta.get("hr", 0)
        onehot_features = {key: row_meta.get(key, 0) for key in encoded_features}
        
        # Process Lead I
        lead_i = sample[:, 0]
        cleaned_i = nk.ecg_clean(lead_i, sampling_rate)
        r_peaks_i = nk.ecg_peaks(cleaned_i, sampling_rate)[1]['ECG_R_Peaks']
        p_i, q_i, r_i, s_i, t_i = detect_pqrst(cleaned_i, r_peaks_i, sampling_rate, 'I')
        
        # Process Lead II
        lead_ii = sample[:, 1]
        cleaned_ii = nk.ecg_clean(lead_ii, sampling_rate)
        r_peaks_ii = nk.ecg_peaks(cleaned_ii, sampling_rate)[1]['ECG_R_Peaks']
        p_ii, q_ii, r_ii, s_ii, t_ii = detect_pqrst(cleaned_ii, r_peaks_ii, sampling_rate, 'II')
        
        # Process Lead aVR with special handling
        lead_avr = sample[:, 3]
        # Flip signal for proper R-peak detection in aVR
        flipped_avr = -lead_avr
        cleaned_flipped_avr = nk.ecg_clean(flipped_avr, sampling_rate)
        r_peaks_avr = nk.ecg_peaks(cleaned_flipped_avr, sampling_rate)[1]['ECG_R_Peaks']
        # Use original signal for PQRST detection
        p_avr, q_avr, r_avr, s_avr, t_avr = detect_pqrst(lead_avr, r_peaks_avr, sampling_rate, 'aVR')
        
        # Find common valid segments across all leads
        min_len = min(len(p_i), len(q_i), len(r_i), len(s_i), len(t_i),
                      len(p_ii), len(q_ii), len(r_ii), len(s_ii), len(t_ii),
                      len(p_avr), len(q_avr), len(r_avr), len(s_avr), len(t_avr))
        
        for j in range(min_len):
            r = r_i[j]
            start, end = r - padding, r + padding
            
            # Skip segments near boundaries
            if start < 0 or end > sample.shape[0]:
                continue
                
            segment_i = lead_i[start:end]
            segment_ii = lead_ii[start:end]
            segment_avr = lead_avr[start:end]
            
            # Extract other leads (excluding I, II, aVR)
            other_leads = {
                lead_names[k]: sample[start:end, k] 
                for k in range(12) 
                if k not in [0, 1, 3]
            }
            
            # Build dataset entry with full PQRST data
            dataset.append({
                # Lead I features
                "pqrst_lead_I": [
                    (p_i[j]/sampling_rate, lead_i[p_i[j]]),
                    (q_i[j]/sampling_rate, lead_i[q_i[j]]),
                    (r_i[j]/sampling_rate, lead_i[r_i[j]]),
                    (s_i[j]/sampling_rate, lead_i[s_i[j]]),
                    (t_i[j]/sampling_rate, lead_i[t_i[j]])
                ],
                # Lead II features
                "pqrst_lead_II": [
                    (p_ii[j]/sampling_rate, lead_ii[p_ii[j]]),
                    (q_ii[j]/sampling_rate, lead_ii[q_ii[j]]),
                    (r_ii[j]/sampling_rate, lead_ii[r_ii[j]]),
                    (s_ii[j]/sampling_rate, lead_ii[s_ii[j]]),
                    (t_ii[j]/sampling_rate, lead_ii[t_ii[j]])
                ],
                # Lead aVR features
                "pqrst_lead_aVR": [
                    (p_avr[j]/sampling_rate, lead_avr[p_avr[j]]),
                    (q_avr[j]/sampling_rate, lead_avr[q_avr[j]]),
                    (r_avr[j]/sampling_rate, lead_avr[r_avr[j]]),
                    (s_avr[j]/sampling_rate, lead_avr[s_avr[j]]),
                    (t_avr[j]/sampling_rate, lead_avr[t_avr[j]])
                ],
                "other_leads": other_leads,
                "age": age,
                "sex": sex,
                "hr": hr,
                **onehot_features,
                "source_index": i
            })
            
    return dataset

if __name__ == "__main__":
    print("Loading data...")
    train_data = joblib.load(input_train_path)
    test_data = joblib.load(input_test_path)
    train_meta = pd.read_csv(meta_train_path)
    test_meta = pd.read_csv(meta_test_path)

    print("Building datasets with stats (80 samples)...")
    train_set = build_dataset_with_stats(train_data, train_meta)
    test_set = build_dataset_with_stats(test_data, test_meta)

    print("Saving...")
    joblib.dump(train_set, output_dir / "pqrst_stats_train_80.pkl")
    joblib.dump(test_set, output_dir / "pqrst_stats_test_80.pkl")

    print("✅ Done with statistical + frequency PQRST dataset (80 samples)!")

