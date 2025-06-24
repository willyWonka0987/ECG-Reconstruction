import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import joblib
import numpy as np
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from scipy.stats import pearsonr

# --- Load encoded feature names ---
encoded_feats_path = Path("encoded_feature_names.txt")
encoded_features = []
if encoded_feats_path.exists():
    with open(encoded_feats_path, 'r') as f:
        encoded_features = [line.strip() for line in f.readlines()]

# --- Config ---
leads_to_predict = ['V1', 'V3', 'V4', 'V5', 'V6']
results_dir = Path("XGBoost_Model_Results")
model_dir = results_dir / "models"
plot_dir = results_dir / "plots"
metrics_file = results_dir / "metrics.txt"
results_dir.mkdir(parents=True, exist_ok=True)
model_dir.mkdir(parents=True, exist_ok=True)
plot_dir.mkdir(parents=True, exist_ok=True)

# --- Load datasets ---
train_data = joblib.load("PQRST_Triplet_With_Stats_80/pqrst_stats_train_80.pkl")
test_data = joblib.load("PQRST_Triplet_With_Stats_80/pqrst_stats_test_80.pkl")

def extract_features_and_targets(data, target_lead):
    X, y = [], []
    for seg in data:
        if target_lead not in seg['other_leads']:
            continue
        try:
            pqrst_I = seg['pqrst_lead_I']
            pqrst_II = seg['pqrst_lead_II']
            pqrst_V2 = seg['pqrst_lead_V2']

            (_, ap1), (tq1, aq1), (tr1, ar1), (ts1, as1), (tt1, at1) = pqrst_I
            (_, ap2), (tq2, aq2), (tr2, ar2), (ts2, as2), (tt2, at2) = pqrst_II
            (_, ap3), (tq3, aq3), (tr3, ar3), (ts3, as3), (tt3, at3) = pqrst_V2

            intervals = [
                tr1 - tq1, ts1 - tq1, tt1 - tq1,
                tr2 - tq2, ts2 - tq2, tt2 - tq2,
                tr3 - tq3, ts3 - tq3, tt3 - tq3
            ]

            amplitudes = [
                aq1, ar1, as1, at1,
                aq2, ar2, as2, at2,
                aq3, ar3, as3, at3
            ]

            age = seg.get("age", 0)
            sex = 1 if str(seg.get("sex", "M")).upper().startswith("M") else 0
            hr = seg.get("hr", 0)
            onehot_values = [seg.get(name, 0) for name in encoded_features]

            stats_features = (
                list(seg['stats_lead_I'].values()) +
                list(seg['stats_lead_II'].values()) +
                list(seg['stats_lead_V2'].values())
            )

            freq_features = (
                list(seg['freq_lead_I'].values()) +
                list(seg['freq_lead_II'].values()) +
                list(seg['freq_lead_V2'].values())
            )

            features = (
                intervals + amplitudes +
                [age, sex, hr] + onehot_values +
                stats_features + freq_features
            )

            target = seg['other_leads'][target_lead]
            if len(target) == 80:
                X.append(features)
                y.append(target)
        except Exception:
            continue
    return np.array(X), np.array(y)

# --- Training ---
with open(metrics_file, 'w') as f:
    for lead in tqdm(leads_to_predict, desc="Training leads"):
        tqdm.write(f"\n🔧 Training XGBoost model for Lead {lead}...")
        X_train_full, y_train_full = extract_features_and_targets(train_data, lead)
        X_test, y_test = extract_features_and_targets(test_data, lead)

        if X_train_full.size == 0 or X_test.size == 0:
            tqdm.write(f"⚠️ No data for lead {lead}, skipping.")
            continue

        X_train, X_val, y_train, y_val = train_test_split(
            X_train_full, y_train_full, test_size=0.1, random_state=42
        )

        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        X_test_scaled = scaler.transform(X_test)

        param_grid = {
        'estimator__n_estimators': [100, 150],
        'estimator__max_depth': [4, 6],
        'estimator__learning_rate': [0.05, 0.1],
        'estimator__subsample': [0.8, 1.0]
        }

        # Step 1: Define base model
        xgb_base = XGBRegressor(verbosity=0, n_jobs=-1)

        # Step 2: Wrap with MultiOutput
        multi_model = MultiOutputRegressor(xgb_base)

        # Step 3: Use GridSearchCV
        grid_search = GridSearchCV(
        estimator=multi_model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        verbose=1,
        n_jobs=-1
        )


        # Step 4: Fit Grid Search
        grid_search.fit(X_train_scaled, y_train)

        # Step 5: Use best model
        model = grid_search.best_estimator_


        joblib.dump(model, model_dir / f"xgb_model_lead_{lead}.pkl")

        y_pred = model.predict(X_test_scaled)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        corr = pearsonr(y_test.flatten(), y_pred.flatten())[0]

        f.write(f"Lead {lead}\nRMSE: {rmse:.4f}\nR^2: {r2:.4f}\nPearson Correlation: {corr:.4f}\n\n")
        tqdm.write(f"✅ Lead {lead}: RMSE={rmse:.4f}, R²={r2:.4f}, Corr={corr:.4f}")

        # Plot sample prediction
        plt.figure(figsize=(10, 4))
        plt.plot(y_test[0], label='Actual', linewidth=2)
        plt.plot(y_pred[0], label='Predicted', linestyle='--')
        plt.title(f"Lead {lead}: Actual vs Predicted")
        plt.xlabel("Time (samples)")
        plt.ylabel("Amplitude")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(plot_dir / f"lead_{lead}_prediction.png")
        plt.close()

print("\n🎉 All XGBoost models trained and evaluated.")

