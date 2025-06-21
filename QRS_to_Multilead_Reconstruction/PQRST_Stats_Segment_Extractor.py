import os
import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.stats import skew, kurtosis, entropy

# --- Parameters ---
sampling_rate = 100
padding = 40  # 80 samples total, centered on R-peak
segment_length = 2 * padding
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# --- Input/Output Paths ---
input_train_path = Path("data_no_segmentation/ecg_train_clean.pkl")
input_test_path = Path("data_no_segmentation/ecg_test_clean.pkl")
meta_train_path = Path("train_split.csv")
meta_test_path = Path("test_split.csv")
output_dir = Path("PQRST_Triplet_With_Stats_80")
output_dir.mkdir(parents=True, exist_ok=True)

def extract_stat_features(segment):
    hist, _ = np.histogram(segment, bins=10, density=True)
    return {
        'mean': np.mean(segment),
        'std': np.std(segment),
        'min': np.min(segment),
        'max': np.max(segment),
        'skewness': skew(segment),
        'kurtosis': kurtosis(segment),
        'rms': np.sqrt(np.mean(np.square(segment))),
        'entropy': entropy(hist)
    }

def find_qspt(signal, r_peaks, sampling_rate, pre_window=0.06, post_window=0.12):
    pre_samples = int(pre_window * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    q_points, s_points, p_points, t_points = [], [], [], []
    for r in r_peaks:
        q_start = max(0, r - pre_samples)
        q_seg = signal[q_start:r]
        q = q_start + np.argmin(q_seg) if len(q_seg) > 0 else r
        q_points.append(q)
        s_end = min(len(signal), r + post_samples)
        s_seg = signal[r:s_end]
        s = r + np.argmin(s_seg) if len(s_seg) > 0 else r
        s_points.append(s)
        p_start = max(0, q - int(0.15 * sampling_rate))
        p_seg = signal[p_start:q]
        p = p_start + np.argmax(p_seg) if len(p_seg) > 0 else q
        p_points.append(p)
        t_end = min(len(signal), s + int(0.4 * sampling_rate))
        t_seg = signal[s:t_end]
        t = s + np.argmax(t_seg) if len(t_seg) > 0 else s
        t_points.append(t)
    return np.array(p_points), np.array(q_points), np.array(r_peaks), np.array(s_points), np.array(t_points)

def build_dataset_with_stats(ecg_dataset, meta_df):
    dataset = []
    for i, sample in enumerate(tqdm(ecg_dataset, desc="Building dataset with stats (80 samples)")):
        row_meta = meta_df.iloc[i]
        age, sex = row_meta["age"], row_meta["sex"]
        heart_axis = row_meta.get("heart_axis", 0)
        hr = row_meta.get("hr", 0)
        lead_i = sample[:, 0]
        cleaned_i = nk.ecg_clean(lead_i, sampling_rate=sampling_rate)
        r_peaks_i = nk.ecg_peaks(cleaned_i, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
        p_i, q_i, r_i, s_i, t_i = find_qspt(cleaned_i, r_peaks_i, sampling_rate)
        lead_ii = sample[:, 1]
        cleaned_ii = nk.ecg_clean(lead_ii, sampling_rate=sampling_rate)
        r_peaks_ii_data = nk.ecg_peaks(cleaned_ii, sampling_rate=sampling_rate)
        if 'ECG_R_Peaks' not in r_peaks_ii_data[1] or len(r_peaks_ii_data[1]['ECG_R_Peaks']) == 0:
            continue
        r_peaks_ii = r_peaks_ii_data[1]['ECG_R_Peaks']
        p_ii, q_ii, r_ii, s_ii, t_ii = find_qspt(cleaned_ii, r_peaks_ii, sampling_rate)
        for j, r in enumerate(r_i):
            if j >= len(p_i) or j >= len(q_i) or j >= len(s_i) or j >= len(t_i):
                continue
            if len(r_peaks_ii) > 0:
                r_ii_idx = np.argmin(np.abs(r_peaks_ii - r))
                if r_ii_idx >= len(p_ii) or r_ii_idx >= len(q_ii) or r_ii_idx >= len(s_ii) or r_ii_idx >= len(t_ii):
                    continue
                r_ii_val = r_peaks_ii[r_ii_idx]
            else:
                continue
            # --- MODIFIED SEGMENTATION ---
            start, end = r - 40, r + 40  # 80 samples centered on R-peak
            if start >= 0 and end <= sample.shape[0]:
                segment_i = lead_i[start:end]
                segment_ii = lead_ii[start:end]
                stats_i = extract_stat_features(segment_i)
                stats_ii = extract_stat_features(segment_ii)
                other_leads_waveforms = {
                    lead_names[k]: sample[start:end, k] for k in range(12) if k not in [0, 1]
                }
                dataset.append({
                    "pqrst_lead_I": [(p_i[j]/sampling_rate, lead_i[p_i[j]]),
                                     (q_i[j]/sampling_rate, lead_i[q_i[j]]),
                                     (r/sampling_rate, lead_i[r]),
                                     (s_i[j]/sampling_rate, lead_i[s_i[j]]),
                                     (t_i[j]/sampling_rate, lead_i[t_i[j]])],
                    "pqrst_lead_II": [(p_ii[r_ii_idx]/sampling_rate, lead_ii[p_ii[r_ii_idx]]),
                                      (q_ii[r_ii_idx]/sampling_rate, lead_ii[q_ii[r_ii_idx]]),
                                      (r_ii_val/sampling_rate, lead_ii[r_ii_val]),
                                      (s_ii[r_ii_idx]/sampling_rate, lead_ii[s_ii[r_ii_idx]]),
                                      (t_ii[r_ii_idx]/sampling_rate, lead_ii[t_ii[r_ii_idx]])],
                    "stats_lead_I": stats_i,
                    "stats_lead_II": stats_ii,
                    "other_leads": other_leads_waveforms,
                    "age": age,
                    "sex": sex,
                    "heart_axis": heart_axis,
                    "hr": hr,
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
    print("✅ Done with statistical PQRST dataset (80 samples)!")

