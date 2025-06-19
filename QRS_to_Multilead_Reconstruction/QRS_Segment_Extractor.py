import joblib
import neurokit2 as nk
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

sampling_rate = 100
padding = 64
segment_length = 2 * padding
lead_names = ['I', 'II', 'III', 'aVR', 'aVL', 'aVF',
              'V1', 'V2', 'V3', 'V4', 'V5', 'V6']

input_train_path = Path("data_no_segmentation/ecg_train_clean.pkl")
input_test_path = Path("data_no_segmentation/ecg_test_clean.pkl")
meta_train_path = Path("train_split.csv")
meta_test_path = Path("test_split.csv")
output_dir = Path("QRS_Triplet_Input_and_FullBeat_Target")
output_dir.mkdir(parents=True, exist_ok=True)

def find_qs_minima(signal, r_peaks, pre_window=0.06, post_window=0.06):
    pre_samples = int(pre_window * sampling_rate)
    post_samples = int(post_window * sampling_rate)
    q_points, s_points = [], []
    for r in r_peaks:
        start_q = max(0, r - pre_samples)
        seg_q = signal[start_q:r]
        q = start_q + np.argmin(seg_q) if len(seg_q) > 0 else r
        q_points.append(q)
        end_s = min(len(signal), r + post_samples)
        seg_s = signal[r:end_s]
        s = r + np.argmin(seg_s) if len(seg_s) > 0 else r
        s_points.append(s)
    return np.array(q_points), np.array(s_points)

def build_combined_dataset(ecg_dataset, meta_df):
    final_dataset = []
    for i, sample in enumerate(tqdm(ecg_dataset, desc="Building QRS + metadata dataset")):
        row_meta = meta_df.iloc[i]
        age = row_meta["age"]
        sex = row_meta["sex"]
        heart_axis = row_meta.get("heart_axis", 0)
        hr = row_meta.get("hr", 0)

        # Lead I processing
        lead_i = sample[:, 0]
        cleaned_i = nk.ecg_clean(lead_i, sampling_rate=sampling_rate)
        r_peaks_i = nk.ecg_peaks(cleaned_i, sampling_rate=sampling_rate)[1]['ECG_R_Peaks']
        q_peaks_i, s_peaks_i = find_qs_minima(cleaned_i, r_peaks_i)

        # Lead II processing
        lead_ii = sample[:, 1]
        cleaned_ii = nk.ecg_clean(lead_ii, sampling_rate=sampling_rate)
        r_peaks_ii_data = nk.ecg_peaks(cleaned_ii, sampling_rate=sampling_rate)
        
        # Handle case where no R-peaks are found in lead II
        if 'ECG_R_Peaks' not in r_peaks_ii_data[1] or len(r_peaks_ii_data[1]['ECG_R_Peaks']) == 0:
            continue  # Skip this sample if no R-peaks in lead II
            
        r_peaks_ii = r_peaks_ii_data[1]['ECG_R_Peaks']
        q_peaks_ii, s_peaks_ii = find_qs_minima(cleaned_ii, r_peaks_ii)

        # Process each R-peak
        for j, r in enumerate(r_peaks_i):
            q_i, s_i = q_peaks_i[j], s_peaks_i[j]
            
            # Find closest R-peak in lead II (with safety check)
            if len(r_peaks_ii) > 0:
                r_ii_idx = np.argmin(np.abs(r_peaks_ii - r))
                r_ii = r_peaks_ii[r_ii_idx]
                q_ii, s_ii = q_peaks_ii[r_ii_idx], s_peaks_ii[r_ii_idx]
            else:
                continue  # Skip if no R-peaks in lead II

            start, end = r - padding, r + padding
            if (start >= 0 and end <= sample.shape[0]) and (q_i != r and s_i != r and q_ii != r_ii and s_ii != r_ii):
                qrs_lead_i = [
                    (q_i / sampling_rate, lead_i[q_i]),
                    (r / sampling_rate, lead_i[r]),
                    (s_i / sampling_rate, lead_i[s_i])
                ]
                qrs_lead_ii = [
                    (q_ii / sampling_rate, lead_ii[q_ii]),
                    (r_ii / sampling_rate, lead_ii[r_ii]),
                    (s_ii / sampling_rate, lead_ii[s_ii])
                ]
                other_leads_waveforms = {
                    lead_names[k]: sample[start:end, k] for k in range(12) if k not in [0, 1]
                }
                final_dataset.append({
                    "qrs_lead_I": qrs_lead_i,
                    "qrs_lead_II": qrs_lead_ii,
                    "other_leads": other_leads_waveforms,
                    "age": age,
                    "sex": sex,
                    "heart_axis": heart_axis,
                    "hr": hr,
                    "source_index": i
                })
    return final_dataset




train_data = joblib.load(input_train_path)
test_data = joblib.load(input_test_path)
train_meta = pd.read_csv(meta_train_path)
test_meta = pd.read_csv(meta_test_path)

train_combined = build_combined_dataset(train_data, train_meta)
test_combined = build_combined_dataset(test_data, test_meta)

joblib.dump(train_combined, output_dir / "combined_qrs_train.pkl")
joblib.dump(test_combined, output_dir / "combined_qrs_test.pkl")
