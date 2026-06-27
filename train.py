import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.dummy import DummyClassifier
from sklearn.base import clone
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

os.makedirs('models', exist_ok=True)
os.makedirs('reports', exist_ok=True)

NP_TRACE_THRESHOLD = 0.005

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/FeNdB_dataset_long.csv')
PHASES   = sorted(df['phase'].unique().tolist())
FEATURES = ['x_Nd', 'x_B', 'temperature_C']

print(f"Dataset: {len(df)} rows | {df['x_Nd'].nunique()} compositions | {len(PHASES)} phases")
print(f"Phases: {PHASES}\n")

# ── Composition-level train/test split (80/20) ─────────────────────────────────
unique_comps = df[['x_Nd', 'x_B', 'x_Fe']].drop_duplicates().reset_index(drop=True)
train_comps, test_comps = train_test_split(unique_comps, test_size=0.2, random_state=42)
train_mask = df.set_index(['x_Nd', 'x_B', 'x_Fe']).index.isin(
    train_comps.set_index(['x_Nd', 'x_B', 'x_Fe']).index)
df_train = df[train_mask].reset_index(drop=True)
df_test  = df[~train_mask].reset_index(drop=True)
print(f"Train: {df_train['x_Nd'].nunique()} compositions ({len(df_train)} rows)")
print(f"Test:  {df_test['x_Nd'].nunique()} compositions ({len(df_test)} rows)\n")

# ── Lookup table (stoichiometric phases) ───────────────────────────────────────
LOOKUP_PHASES = [p for p in PHASES if p != 'LIQUID']
lookup_table  = (df_train[df_train['phase'].isin(LOOKUP_PHASES)]
                 .groupby('phase')[['X_Nd', 'X_B', 'X_Fe']].mean().round(6))
joblib.dump(lookup_table, 'models/lookup_table_constrained.pkl')
print("Lookup table saved.")

# ── Build classification table ─────────────────────────────────────────────────
def build_cls_table(data, threshold=NP_TRACE_THRESHOLD):
    ct = data.groupby(FEATURES)['phase'].apply(lambda x: set(x)).reset_index()
    ct.columns = FEATURES + ['phases_present']
    np_pivot = (data[data['NP'] >= threshold]
                .groupby(FEATURES)['phase'].apply(lambda x: set(x)).reset_index())
    np_pivot.columns = FEATURES + ['phases_above_threshold']
    ct = ct.merge(np_pivot, on=FEATURES, how='left')
    ct['phases_above_threshold'] = ct['phases_above_threshold'].apply(
        lambda x: x if isinstance(x, set) else set())
    for phase in PHASES:
        ct[phase] = ct['phases_above_threshold'].apply(lambda s: int(phase in s))
    return ct.drop(columns=['phases_present', 'phases_above_threshold'])

ct_train = build_cls_table(df_train)
X_cls_train = ct_train[FEATURES].values
y_cls_train = ct_train[PHASES].values

# ── Train LightGBM classifiers (one per phase) ────────────────────────────────
print("Training LightGBM classifiers...")
classifiers = {}
for i, phase in enumerate(PHASES):
    y_tr = y_cls_train[:, i]
    neg, pos = (y_tr == 0).sum(), (y_tr == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    clf = lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                              scale_pos_weight=spw, random_state=42, verbose=-1)
    if len(np.unique(y_tr)) < 2:
        clf = DummyClassifier(strategy='constant', constant=y_tr[0])
    clf.fit(X_cls_train, y_tr)
    classifiers[phase] = clf
joblib.dump(classifiers, 'models/lgbm_classifiers_constrained.pkl')
print("LightGBM classifiers saved.")

# ── Train NP regressors (RandomForest, one per phase) ─────────────────────────
print("Training NP regressors...")
np_regressors = {}
for phase in PHASES:
    phase_train = df_train[df_train['phase'] == phase]
    if len(phase_train) < 50:
        continue
    X_tr = phase_train[FEATURES].values
    y_tr = phase_train['NP'].values
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_tr, y_tr)
    np_regressors[phase] = rf
joblib.dump(np_regressors, 'models/rf_np_regressors_constrained.pkl')
print("NP regressors saved.")

# ── Train LIQUID internal composition regressor ────────────────────────────────
print("Training LIQUID composition regressor...")
liq_train = df_train[df_train['phase'] == 'LIQUID']
X_liq_tr  = liq_train[FEATURES].values
y_liq_tr  = liq_train[['X_Nd', 'X_B', 'X_Fe']].values
liquid_regressor = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
liquid_regressor.fit(X_liq_tr, y_liq_tr)
joblib.dump(liquid_regressor, 'models/rf_liquid_regressor_constrained.pkl')
print("LIQUID regressor saved.")

# ── Correlation matrix ─────────────────────────────────────────────────────────
print("\nBuilding correlation matrix...")
corr_rows = []
for (x_nd, x_b, temp), grp in df_train.groupby(['x_Nd', 'x_B', 'temperature_C']):
    row = {'x_Nd': x_nd, 'x_B': x_b, 'x_Fe': grp['x_Fe'].iloc[0], 'temperature_C': temp}
    for phase in PHASES:
        ph = grp[grp['phase'] == phase]
        row[f'NP_{phase}'] = ph['NP'].values[0] if len(ph) > 0 else 0.0
        if phase == 'LIQUID' and len(ph) > 0:
            row['LIQUID_X_Nd'] = ph['X_Nd'].values[0]
            row['LIQUID_X_B']  = ph['X_B'].values[0]
            row['LIQUID_X_Fe'] = ph['X_Fe'].values[0]
    corr_rows.append(row)
pd.DataFrame(corr_rows).fillna(0).corr().round(4).to_csv('reports/correlation_matrix.csv')
print("Correlation matrix saved → reports/correlation_matrix.csv")

print("\n=== Training complete. Run train_compare.py for evaluation. ===")