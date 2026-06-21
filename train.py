import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import f1_score, r2_score
from sklearn.base import clone
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/FeNdB_ML_dataset_long_constrained.csv')

PHASES   = sorted(df['phase'].unique().tolist())
FEATURES = ['x_Nd', 'x_B', 'temperature_C']

# ── Lookup table ──────────────────────────────────────────────────────────────
LOOKUP_PHASES = [p for p in PHASES if p != 'LIQUID']
lookup_table  = (df[df['phase'].isin(LOOKUP_PHASES)]
                 .groupby('phase')[['X_Nd','X_B','X_Fe']]
                 .mean().round(6))
joblib.dump(lookup_table, 'models/lookup_table_constrained.pkl')
print("Lookup table saved.\n")



# build classification table
ct_df = (df.groupby(FEATURES)['phase']
           .apply(lambda x: set(x)).reset_index())
ct_df.columns = FEATURES + ['phases_present']
for phase in PHASES:
    ct_df[phase] = ct_df['phases_present'].apply(lambda s: int(phase in s))
ct_df = ct_df.drop(columns=['phases_present'])

X_cls = ct_df[FEATURES].values
y_cls = ct_df[PHASES].values

def make_lgbm(y_train_binary):
    neg = (y_train_binary == 0).sum()
    pos = (y_train_binary == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    return lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                               scale_pos_weight=spw, random_state=42, verbose=-1)

# ── 5-fold CV for classification ──────────────────────────────────────────────
print("=== Classification CV (5-fold) ===")
print(f"{'Phase':<15} {'Mean F1':>10} {'Std F1':>10}")
print("-" * 37)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
cls_cv_results = {}

for i, phase in enumerate(PHASES):
    fold_f1s = []
    for train_idx, val_idx in kf.split(X_cls):
        X_tr, X_val = X_cls[train_idx], X_cls[val_idx]
        y_tr, y_val = y_cls[train_idx, i], y_cls[val_idx, i]
        clf = make_lgbm(y_tr)
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_val)
        fold_f1s.append(f1_score(y_val, y_pred, zero_division=0))
    mean_f1 = np.mean(fold_f1s)
    std_f1  = np.std(fold_f1s)
    cls_cv_results[phase] = (mean_f1, std_f1)
    print(f"{phase:<15} {mean_f1:>10.4f} {std_f1:>10.4f}")

macro_mean = np.mean([v[0] for v in cls_cv_results.values()])
macro_std  = np.mean([v[1] for v in cls_cv_results.values()])
print("-" * 37)
print(f"{'MACRO':<15} {macro_mean:>10.4f} {macro_std:>10.4f}")

# ── Train final classifiers on full data ─────────────────────────────────────
print("\nTraining final classifiers on constrained data...")
classifiers = {}
for i, phase in enumerate(PHASES):
    clf = make_lgbm(y_cls[:, i])
    clf.fit(X_cls, y_cls[:, i])
    classifiers[phase] = clf

joblib.dump(classifiers, 'models/lgbm_classifiers_constrained.pkl')
print("Classifiers saved → models/lgbm_classifiers_constrained.pkl\n")



# PART 2 — NP Regression (RandomForest, one per phase)

print("=== NP Regression CV (5-fold) ===")
print(f"{'Phase':<15} {'Mean R²':>10} {'Std R²':>10} {'Rows':>6}")
print("-" * 44)

kf_reg = KFold(n_splits=5, shuffle=True, random_state=42)
np_regressors = {}

for phase in PHASES:
    phase_df = df[df['phase'] == phase]
    if len(phase_df) < 50:
        print(f"{phase:<15} {'skipped (< 50 rows)':>28}")
        continue

    X_reg = phase_df[FEATURES].values
    y_reg = phase_df['NP'].values

    fold_r2s = []
    for train_idx, val_idx in kf_reg.split(X_reg):
        X_tr, X_val = X_reg[train_idx], X_reg[val_idx]
        y_tr, y_val = y_reg[train_idx], y_reg[val_idx]
        rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
        rf.fit(X_tr, y_tr)
        fold_r2s.append(r2_score(y_val, rf.predict(X_val)))

    mean_r2 = np.mean(fold_r2s)
    std_r2  = np.std(fold_r2s)
    print(f"{phase:<15} {mean_r2:>10.4f} {std_r2:>10.4f} {len(phase_df):>6}")

    # train final on full phase data
    rf_final = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf_final.fit(X_reg, y_reg)
    np_regressors[phase] = rf_final

joblib.dump(np_regressors, 'models/rf_np_regressors_constrained.pkl')
print("\nNP regressors saved → models/rf_np_regressors_constrained.pkl\n")



# PART 3 — LIQUID composition regressor

print("Training LIQUID composition regressor...")
liquid_df = df[df['phase'] == 'LIQUID']
X_liq = liquid_df[FEATURES].values
y_liq = liquid_df[['X_Nd','X_B','X_Fe']].values

liquid_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
liquid_regressor.fit(X_liq, y_liq)

joblib.dump(liquid_regressor, 'models/rf_liquid_regressor_constrained.pkl')
print("LIQUID regressor saved → models/rf_liquid_regressor_constrained.pkl\n")

print("=== Training complete. All models saved to models/ ===")