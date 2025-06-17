import joblib
import neurokit2 as nk
import numpy as np
from tqdm import tqdm

# --- Parameters ---
sampling_rate = 100  # Hz
pre_window = 0.06  # seconds before R to search for Q
post_window = 0.06  # seconds after R to search for S
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

def find_qs_minima(signal, r_peaks, pre_window, post_window, sampling_rate):
    pre_samples = int(pre_window * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    q_points, s_points = [], []

    for r in r_peaks:
        start = max(0, r - pre_samples)
        seg_q = signal[start:r]
        q = start + np.argmin(seg_q) if len(seg_q) > 0 else r
        q_points.append(q)

        end = min(len(signal), r + post_samples)
        seg_s = signal[r:end]
        s = r + np.argmin(seg_s) if len(seg_s) > 0 else r
        s_points.append(s)

    return np.array(q_points), np.array(s_points)

def extract_qrs_segments(ecg_dataset):
    all_segments = []
    for sample in tqdm(ecg_dataset, desc="Processing ECG samples"):
        lead_ii_signal = sample[:, 1]
        cleaned = nk.ecg_clean(lead_ii_signal, sampling_rate=sampling_rate)
        r_peaks = nk.ecg_peaks(cleaned, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']

        for r in r_peaks:  # No Q/S detection
            beat_segment = []
            for lead_idx in range(12):
                signal = sample[:, lead_idx]
                beat_segment.append([
                    (r / sampling_rate, signal[r]),  # Only R-peak
                ])
            all_segments.append(beat_segment)
    return all_segments

def remove_outliers_qrs(all_segments, quantile=0.99):
    """
    Unified outlier removal for both QRS point-based and waveform-based segments.
    For each lead, computes max-min amplitude and removes segments exceeding the quantile threshold.
    Works for:
      - First code: Segments as lists of (Q, R, S) tuples per lead.
      - Second code: Segments as (128, 12) numpy arrays.
    """
    lead_amplitudes = {lead_idx: [] for lead_idx in range(12)}
    
    # Collect amplitudes per lead
    for segment in all_segments:
        for lead_idx in range(12):
            if isinstance(segment, np.ndarray):  # Second code (128-sample segments)
                amplitude = np.max(segment[:, lead_idx]) - np.min(segment[:, lead_idx])
            else:  # First code (Q/R/S tuples)
                qrs_points = segment[lead_idx]  # [(t_q, amp_q), (t_r, amp_r), (t_s, amp_s)]
                amps = [amp for (_, amp) in qrs_points]
                amplitude = max(amps) - min(amps)
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
            if isinstance(segment, np.ndarray):  # Second code
                amplitude = np.max(segment[:, lead_idx]) - np.min(segment[:, lead_idx])
            else:  # First code
                amps = [amp for (_, amp) in segment[lead_idx]]
                amplitude = max(amps) - min(amps)
            if amplitude > thresholds[lead_idx]:
                keep = False
                break
        if keep:
            cleaned_segments.append(segment)
    
    return cleaned_segments

# --- Load ECG data ---
ecg_train_data = joblib.load("data_no_segmentation/ecg_train_clean.pkl")
ecg_test_data = joblib.load("data_no_segmentation/ecg_test_clean.pkl")

# --- Extract raw QRS segments ---
train_segments = extract_qrs_segments(ecg_train_data)
test_segments = extract_qrs_segments(ecg_test_data)

# --- Apply quantile-based amplitude filtering ---
train_segments_clean = remove_outliers_qrs(train_segments, quantile=0.99)
test_segments_clean = remove_outliers_qrs(test_segments, quantile=0.99)

# --- Save cleaned QRS segments ---
joblib.dump(train_segments_clean, "qrs_train_segments.pkl")
joblib.dump(test_segments_clean, "qrs_test_segments.pkl")

print(f"Cleaned and saved: {len(train_segments_clean)} train segments, {len(test_segments_clean)} test segments")
print(f"Original train: {len(train_segments)}, Cleaned: {len(train_segments_clean)}")
print(f"Original test:  {len(test_segments)}, Cleaned: {len(test_segments_clean)}")
