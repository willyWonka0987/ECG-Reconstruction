import os
import joblib
import neurokit2 as nk
import numpy as np
from tqdm import tqdm

# --- Parameters ---
sampling_rate = 100  # Hz
padding = 64  # 64 samples before and after R-peak
segment_length = 2 * padding  # Total 128 samples (centered around R)
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

# Output directory
output_dir = "QRS_waves_dataset_128samples"
os.makedirs(output_dir, exist_ok=True)

# --- Add the missing function ---
def find_qs_minima(signal, r_peaks, pre_window, post_window, sampling_rate):
    pre_samples = int(pre_window * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    q_points, s_points = [], []

    for r in r_peaks:
        # Find Q point (before R)
        start = max(0, r - pre_samples)
        seg_q = signal[start:r]
        q = start + np.argmin(seg_q) if len(seg_q) > 0 else r
        q_points.append(q)

        # Find S point (after R)
        end = min(len(signal), r + post_samples)
        seg_s = signal[r:end]
        s = r + np.argmin(seg_s) if len(seg_s) > 0 else r
        s_points.append(s)

    return np.array(q_points), np.array(s_points)

def extract_qrs_wave_segments(ecg_dataset):
    all_segments = []

    for sample in tqdm(ecg_dataset, desc="Extracting QRS wave-centered beats"):
        # Detect R-peaks on Lead II
        lead_ii_signal = sample[:, 1]
        cleaned = nk.ecg_clean(lead_ii_signal, sampling_rate=sampling_rate)
        r_peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
        q_peaks, s_peaks = find_qs_minima(cleaned, r_peaks, pre_window=0.06, post_window=0.06, sampling_rate=sampling_rate)

        for i, r in enumerate(r_peaks):
            q, s = q_peaks[i], s_peaks[i]
            start, end = r - padding, r + padding
            
            # Only keep if Q/S are found AND window fits
            if (start >= 0 and end <= sample.shape[0]) and (q != r and s != r):
                beat_matrix = sample[start:end, :]
                if beat_matrix.shape[0] == segment_length:
                    all_segments.append(beat_matrix)

    return all_segments

def remove_outliers_qrs(all_segments, quantile=0.99):
    lead_amplitudes = {lead_idx: [] for lead_idx in range(12)}
    
    # Collect amplitudes per lead
    for segment in all_segments:
        for lead_idx in range(12):
            amplitude = np.max(segment[:, lead_idx]) - np.min(segment[:, lead_idx])
            lead_amplitudes[lead_idx].append(amplitude)
    
    # Compute thresholds (99th percentile per lead)
    thresholds = {
        lead_idx: np.quantile(lead_amplitudes[lead_idx], quantile)
        for lead_idx in range(12)
    }
    
    # Filter segments
    cleaned_segments = []
    for segment in all_segments:
        keep = True
        for lead_idx in range(12):
            amplitude = np.max(segment[:, lead_idx]) - np.min(segment[:, lead_idx])
            if amplitude > thresholds[lead_idx]:
                keep = False
                break
        if keep:
            cleaned_segments.append(segment)
    
    return cleaned_segments

# --- Load ECG data ---
ecg_train_data = joblib.load("data_no_segmentation/ecg_train_clean.pkl")
ecg_test_data = joblib.load("data_no_segmentation/ecg_test_clean.pkl")

# --- Extract and clean segments ---
train_segments = extract_qrs_wave_segments(ecg_train_data)
test_segments = extract_qrs_wave_segments(ecg_test_data)

print(f"Before cleaning: Train = {len(train_segments)}, Test = {len(test_segments)}")

train_segments_clean = remove_outliers_qrs(train_segments)
test_segments_clean = remove_outliers_qrs(test_segments)

print(f"✅ After cleaning: Train = {len(train_segments_clean)}, Test = {len(test_segments_clean)}")

# --- Save cleaned segments ---
joblib.dump(train_segments_clean, os.path.join(output_dir, "qrs_wave_train_segments.pkl"))
joblib.dump(test_segments_clean, os.path.join(output_dir, "qrs_wave_test_segments.pkl"))