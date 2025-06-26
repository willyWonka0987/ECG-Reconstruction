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

# --- Parameters ---
sampling_rate = 100
padding = 40
segment_length = 2 * padding
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# --- Input/Output Paths ---
input_train_path = Path("data_no_segmentation/ecg_train_clean.pkl")
input_test_path = Path("data_no_segmentation/ecg_test_clean.pkl")
meta_train_path = Path("train_split.csv")
meta_test_path = Path("test_split.csv")
output_dir = Path("PQRST_80_Datasets")
output_dir.mkdir(parents=True, exist_ok=True)

# --- Load encoded feature names ---
encoded_feats_path = Path("encoded_feature_names.txt")
encoded_features = []
if encoded_feats_path.exists():
    with open(encoded_feats_path, 'r') as f:
        encoded_features = [line.strip() for line in f.readlines()]

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

def extract_freq_features(segment, fs=100):
    freqs, psd = welch(segment, fs=fs, nperseg=min(64, len(segment)))
    dom_freq = freqs[np.argmax(psd)] if len(psd) > 0 else 0
    psd_sum = np.sum(psd)
    wavelet_coeffs = pywt.wavedec(segment, 'db6', level=2)
    wavelet_energy = sum(np.sum(c**2) for c in wavelet_coeffs)
    return {
        'psd_total': psd_sum,
        'dominant_freq': dom_freq,
        'wavelet_energy': wavelet_energy
    }

def find_qspt(signal, r_peaks, sampling_rate):
    pre_samples = int(0.06 * sampling_rate)
    post_samples = int(0.12 * sampling_rate)
    q_points, s_points, p_points, t_points = [], [], [], []
    for r in r_peaks:
        # Q
        q_start = max(0, r - pre_samples)
        q_seg = signal[q_start:r]
        q = q_start + np.argmin(q_seg) if len(q_seg) > 0 else r
        q_points.append(q)
        # S
        s_end = min(len(signal), r + post_samples)
        s_seg = signal[r:s_end]
        s = r + np.argmin(s_seg) if len(s_seg) > 0 else r
        s_points.append(s)
        # P
        p_start = max(0, q - int(0.15 * sampling_rate))
        p_seg = signal[p_start:q]
        p = p_start + np.argmax(p_seg) if len(p_seg) > 0 else q
        p_points.append(p)
        # T
        t_end = min(len(signal), s + int(0.4 * sampling_rate))
        t_seg = signal[s:t_end]
        t = s + np.argmax(t_seg) if len(t_seg) > 0 else s
        t_points.append(t)
    return np.array(p_points), np.array(q_points), np.array(r_peaks), np.array(s_points), np.array(t_points)

def compute_area(signal, start, end):
    if start is None or end is None or end <= start:
        return np.nan
    return np.trapz(signal[start:end], dx=1.0/sampling_rate)

def compute_duration(start, end):
    if start is None or end is None or end <= start:
        return np.nan
    return (end - start) / sampling_rate

def build_dataset_with_stats(ecg_dataset, meta_df):
    dataset = []
    for i, sample in enumerate(tqdm(ecg_dataset, desc="Building dataset with stats (80 samples)")):
        if i >= len(meta_df):
            break
        row_meta = meta_df.iloc[i]
        age = row_meta.get("age", 0)
        sex = row_meta.get("sex", 0)
        hr = row_meta.get("hr", 0)
        onehot_features = {key: row_meta.get(key, 0) for key in encoded_features}

        lead_i = sample[:, 0]
        lead_ii = sample[:, 1]
        lead_v2 = sample[:, lead_names.index('V2')]
        lead_v3 = sample[:, lead_names.index('V3')]

        r_i = nk.ecg_peaks(nk.ecg_clean(lead_i, sampling_rate), sampling_rate)[1]['ECG_R_Peaks']
        r_ii = nk.ecg_peaks(nk.ecg_clean(lead_ii, sampling_rate), sampling_rate)[1]['ECG_R_Peaks']
        r_v2 = nk.ecg_peaks(nk.ecg_clean(lead_v2, sampling_rate), sampling_rate)[1]['ECG_R_Peaks']
        r_v3 = nk.ecg_peaks(nk.ecg_clean(lead_v3, sampling_rate), sampling_rate)[1]['ECG_R_Peaks']

        if len(r_i) < 2 or len(r_ii) < 2 or len(r_v2) < 2 or len(r_v3) == 0:
            continue

        p_i, q_i, r_i, s_i, t_i = find_qspt(lead_i, r_i, sampling_rate)
        p_ii, q_ii, r_ii, s_ii, t_ii = find_qspt(lead_ii, r_ii, sampling_rate)
        p_v2, q_v2, r_v2, s_v2, t_v2 = find_qspt(lead_v2, r_v2, sampling_rate)
        p_v3, q_v3, r_v3, s_v3, t_v3 = find_qspt(lead_v3, r_v3, sampling_rate)

        for j in range(1, len(r_i)):
            r = r_i[j]
            if j >= len(p_i) or j >= len(q_i) or j >= len(s_i) or j >= len(t_i):
                continue
            r_ii_idx = np.argmin(np.abs(r_ii - r))
            r_v2_idx = np.argmin(np.abs(r_v2 - r))
            r_v3_idx = np.argmin(np.abs(r_v3 - r))
            if r_ii_idx >= len(p_ii) or r_v2_idx >= len(p_v2) or r_v3_idx >= len(p_v3):
                continue
            start, end = r - padding, r + padding
            if start < 0 or end > sample.shape[0]:
                continue

            segment_i = lead_i[start:end]
            segment_ii = lead_ii[start:end]
            segment_v2 = lead_v2[start:end]

            stats_i = extract_stat_features(segment_i)
            stats_ii = extract_stat_features(segment_ii)
            stats_v2 = extract_stat_features(segment_v2)
            freq_i = extract_freq_features(segment_i, fs=sampling_rate)
            freq_ii = extract_freq_features(segment_ii, fs=sampling_rate)
            freq_v2 = extract_freq_features(segment_v2, fs=sampling_rate)

            rr1 = (r_i[j] - r_i[j - 1]) / sampling_rate
            rr2 = (r_ii[r_ii_idx] - r_ii[r_ii_idx - 1]) / sampling_rate if r_ii_idx > 0 else 0
            rr3 = (r_v2[r_v2_idx] - r_v2[r_v2_idx - 1]) / sampling_rate if r_v2_idx > 0 else 0

            # QRS area and duration
            qrs_area_i = compute_area(lead_i, q_i[j], s_i[j])
            qrs_area_ii = compute_area(lead_ii, q_ii[r_ii_idx], s_ii[r_ii_idx])
            qrs_area_v2 = compute_area(lead_v2, q_v2[r_v2_idx], s_v2[r_v2_idx])
            qrs_dur_i = compute_duration(q_i[j], s_i[j])
            qrs_dur_ii = compute_duration(q_ii[r_ii_idx], s_ii[r_ii_idx])
            qrs_dur_v2 = compute_duration(q_v2[r_v2_idx], s_v2[r_v2_idx])
            t_area_i = compute_area(lead_i, s_i[j], t_i[j])
            t_area_ii = compute_area(lead_ii, s_ii[r_ii_idx], t_ii[r_ii_idx])
            t_area_v2 = compute_area(lead_v2, s_v2[r_v2_idx], t_v2[r_v2_idx])

            # --- NaN/Inf check: skip if any feature is NaN or Inf ---
            features_to_check = [
                qrs_area_i, qrs_area_ii, qrs_area_v2,
                qrs_dur_i, qrs_dur_ii, qrs_dur_v2,
                t_area_i, t_area_ii, t_area_v2,
                rr1, rr2, rr3
            ]
            stats_list = list(stats_i.values()) + list(stats_ii.values()) + list(stats_v2.values())
            freq_list = list(freq_i.values()) + list(freq_ii.values()) + list(freq_v2.values())
            features_to_check += stats_list + freq_list

            if any(np.isnan(f) or np.isinf(f) for f in features_to_check):
                continue

            other_leads_waveforms = {
                lead_names[k]: sample[start:end, k] for k in range(12)
                if k not in [0, 1, lead_names.index('V2')]
            }

            dataset.append({
                "pqrst_lead_I": [(p_i[j]/sampling_rate, lead_i[p_i[j]]), (q_i[j]/sampling_rate, lead_i[q_i[j]]),
                                 (r/sampling_rate, lead_i[r]), (s_i[j]/sampling_rate, lead_i[s_i[j]]), (t_i[j]/sampling_rate, lead_i[t_i[j]])],
                "pqrst_lead_II": [(p_ii[r_ii_idx]/sampling_rate, lead_ii[p_ii[r_ii_idx]]),
                                  (q_ii[r_ii_idx]/sampling_rate, lead_ii[q_ii[r_ii_idx]]),
                                  (r_ii[r_ii_idx]/sampling_rate, lead_ii[r_ii[r_ii_idx]]),
                                  (s_ii[r_ii_idx]/sampling_rate, lead_ii[s_ii[r_ii_idx]]),
                                  (t_ii[r_ii_idx]/sampling_rate, lead_ii[t_ii[r_ii_idx]])],
                "pqrst_lead_V2": [(p_v2[r_v2_idx]/sampling_rate, lead_v2[p_v2[r_v2_idx]]),
                                  (q_v2[r_v2_idx]/sampling_rate, lead_v2[q_v2[r_v2_idx]]),
                                  (r_v2[r_v2_idx]/sampling_rate, lead_v2[r_v2[r_v2_idx]]),
                                  (s_v2[r_v2_idx]/sampling_rate, lead_v2[s_v2[r_v2_idx]]),
                                  (t_v2[r_v2_idx]/sampling_rate, lead_v2[t_v2[r_v2_idx]])],
                "rr1": rr1,
                "rr2": rr2,
                "rr3": rr3,
                "qrs_area_I": qrs_area_i,
                "qrs_area_II": qrs_area_ii,
                "qrs_area_V2": qrs_area_v2,
                "qrs_dur_I": qrs_dur_i,
                "qrs_dur_II": qrs_dur_ii,
                "qrs_dur_V2": qrs_dur_v2,
                "t_area_I": t_area_i,
                "t_area_II": t_area_ii,
                "t_area_V2": t_area_v2,
                "stats_lead_I": stats_i,
                "stats_lead_II": stats_ii,
                "stats_lead_V2": stats_v2,
                "freq_lead_I": freq_i,
                "freq_lead_II": freq_ii,
                "freq_lead_V2": freq_v2,
                "other_leads": other_leads_waveforms,
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
    print("✅ Done with V2-based PQRST dataset including QRS/T area and duration features!")
