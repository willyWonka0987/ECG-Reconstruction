import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for saving plots

import matplotlib.pyplot as plt
import joblib
import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Add, Activation

# --- Config ---
leads_to_predict = ['II', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6']
model_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/CNN_with_metadata_models")
plot_dir = Path("QRS_Triplet_Input_and_FullBeat_Target/CNN_with_metadata_plots")
metrics_file = Path("QRS_Triplet_Input_and_FullBeat_Target/CNN_with_metadata_metrics.txt")
model_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets ---
train_data = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_train.pkl")
test_data = joblib.load("QRS_Triplet_Input_and_FullBeat_Target/combined_qrs_test.pkl")

# --- Helper functions ---
def extract_features_and_targets(data, target_lead):
    X, y = [], []
    for seg in data:
        qrs = seg['qrs_lead_I']
        if target_lead not in seg['other_leads'] or len(qrs) != 3:
            continue
        try:
            (tq, aq), (tr, ar), (ts, as_) = qrs
            rq_interval = tr - tq
            sr_interval = ts - tr

            age = seg.get("age", 0)
            sex = 1 if str(seg.get("sex", "M")).upper().startswith("M") else 0
            heart_axis = seg.get("heart_axis", 0)
            if isinstance(heart_axis, str):
                heart_axis = int(heart_axis) if heart_axis.isdigit() else 0

            features = [tq, aq, tr, ar, ts, as_, rq_interval, sr_interval, age, sex, heart_axis]
            target = seg['other_leads'][target_lead]
            if len(target) == 128:
                X.append(features)
                y.append(target)
        except Exception as e:
            continue
    X = np.array(X)
    y = np.array(y)
    return X[..., np.newaxis], y



def conv_block(x, filters, kernel_size=3, dropout_rate=0.3):
    shortcut = x
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dropout(dropout_rate)(x)
    
    x = Conv1D(filters, kernel_size, padding='same')(x)
    x = BatchNormalization()(x)
    x = Add()([x, shortcut])  # Residual connection
    x = Activation('relu')(x)
    return x

def build_deeper_cnn_model(input_shape):
    inp = Input(shape=input_shape)  # shape = (11, 1)

    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(inp)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    # Add 2 residual conv blocks
    x = conv_block(x, 64)
    x = conv_block(x, 64)

    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)

    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)

    x = GlobalAveragePooling1D()(x)

    # Dense layers
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.4)(x)
    x = Dense(256, activation='relu')(x)
    x = Dropout(0.3)(x)
    out = Dense(128, activation='linear')(x)

    model = Model(inputs=inp, outputs=out)
    model.compile(optimizer=tf.keras.optimizers.Adam(0.0003), loss='mse', metrics=['mae'])
    return model



def plot_prediction(y_true, y_pred, lead_name, save_path):
    plt.figure(figsize=(10, 4))
    plt.plot(y_true, label='Actual', linewidth=2)
    plt.plot(y_pred, label='Predicted', linestyle='--')
    plt.title(f"Lead {lead_name}: Actual vs Predicted")
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_loss_curve(history, lead_name, save_path):
    plt.figure(figsize=(8, 4))
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Val Loss')
    plt.title(f"Loss Curve for Lead {lead_name}")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# --- Training Pipeline ---
with open(metrics_file, 'w') as f:
    for lead in leads_to_predict:
        print(f"\n🔧 Training model for Lead {lead}...")

        X_train, y_train = extract_features_and_targets(train_data, lead)
        X_test, y_test = extract_features_and_targets(test_data, lead)

        scaler = StandardScaler()
        X_train_2d = scaler.fit_transform(X_train.squeeze()).reshape(X_train.shape)
        X_test_2d = scaler.transform(X_test.squeeze()).reshape(X_test.shape)

        model = build_deeper_cnn_model(input_shape=(X_train.shape[1], 1))

        history = model.fit(
            X_train_2d, y_train,
            validation_split=0.1,
            epochs=100,
            batch_size=64,
            callbacks=[
                EarlyStopping(patience=15, restore_best_weights=True),
                ModelCheckpoint(
                    filepath=model_dir / f"cnn_1d_model_lead_{lead}.h5",
                    save_best_only=True,
                    monitor='val_loss'
                )
            ],
            verbose=1
        )

        y_pred = model.predict(X_test_2d)

        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        corr = pearsonr(y_test.flatten(), y_pred.flatten())[0]

        f.write(f"Lead {lead}\n")
        f.write(f"RMSE: {rmse:.4f}\n")
        f.write(f"R^2: {r2:.4f}\n")
        f.write(f"Pearson Correlation: {corr:.4f}\n\n")

        print(f"✅ Lead {lead}: RMSE={rmse:.4f}, R²={r2:.4f}, Corr={corr:.4f}")

        plot_prediction(
            y_true=y_test[0],
            y_pred=y_pred[0],
            lead_name=lead,
            save_path=plot_dir / f"lead_{lead}_prediction.png"
        )

        plot_loss_curve(
            history,
            lead_name=lead,
            save_path=plot_dir / f"lead_{lead}_loss_curve.png"
        )

print("\n🎉 All models trained, metrics saved, and plots generated.")

