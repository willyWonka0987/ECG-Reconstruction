import numpy as np
import matplotlib.pyplot as plt
import joblib
import pandas as pd  # ✅ THIS is the missing import

from tensorflow.keras.models import load_model


# === Load the data ===
ecg_test = joblib.load(open("/home/youssef/Documents/ecg_reconstruction-main_v1/data/ecg_test_clean.pkl", "rb"))
features_test = joblib.load(open("/home/youssef/Documents/ecg_reconstruction-main_v1/data/features_test_clean.pkl", "rb"))

# === Extract input (Lead I) and target (Lead II) ===
X_test = ecg_test[:, :, 0]   # Lead I
y_test = ecg_test[:, :, 1]   # Lead II

# === One-hot encode test features ===
def create_one_hot(features_test):
    f_lst = ['ecg_id', 'superclasses', 'heart_axis']
    df_test = pd.DataFrame(features_test, columns=f_lst)

    # Ensure heart_axis includes all categories
    categories = list(range(9))
    df_test['heart_axis'] = pd.Categorical(df_test['heart_axis'], categories=categories)

    # Ensure same one-hot structure as training
    heart_axis = pd.get_dummies(df_test, columns=['heart_axis'], drop_first=True)

    ha_cols = [f'heart_axis_{i}' for i in range(1, 9)]  # 1 to 8
    for col in ha_cols:
        if col not in heart_axis.columns:
            heart_axis[col] = 0  # Add missing columns

    return heart_axis[ha_cols]


one_hot_test = create_one_hot(features_test)

# === Prepare single sample for prediction ===
X_sample = X_test[0:1]  # shape (1, 128)
one_hot_sample = one_hot_test.iloc[0:1].to_numpy()  # (1, 8)
one_hot_sample = np.expand_dims(one_hot_sample, axis=-1)  # (1, 8, 1)


# === Load model ===
model_path = "/home/youssef/Documents/ecg_reconstruction-main_v1/ann136_models/model_II.keras"
model = load_model(model_path)

# === Predict ===
predicted_lead2 = model.predict([X_sample, one_hot_sample])[0].squeeze()  # shape (128,)
true_lead2 = y_test[0]  # shape (128,)

# === Plot ===
plt.figure(figsize=(10, 4))
plt.plot(true_lead2, label="True Lead II", linewidth=2)
plt.plot(predicted_lead2, label="Predicted Lead II", linestyle='--')
plt.title("Predicted vs Actual Lead II (from Lead I)")
plt.xlabel("Time (samples)")
plt.ylabel("Amplitude (mV)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
