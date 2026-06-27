import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestRegressor, RandomForestClassifier,
                               GradientBoostingClassifier, GradientBoostingRegressor)
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.metrics import f1_score, r2_score
from sklearn.multioutput import MultiOutputRegressor
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

os.makedirs('reports', exist_ok=True)

NP_TRACE_THRESHOLD = 0.005

# ── Load ───────────────────────────────────────────────────────────────────────
df = pd.read_csv('data/FeNdB_dataset_long.csv')
PHASES   = sorted(df['phase'].unique().tolist())
FEATURES = ['x_Nd', 'x_B', 'temperature_C']

print(f"Dataset: {len(df)} rows | {df['x_Nd'].nunique()} compositions | {len(PHASES)} phases\n")

# ── Composition-level train/test split ────────────────────────────────────────
unique_comps = df[['x_Nd', 'x_B', 'x_Fe']].drop_duplicates().reset_index(drop=True)
train_comps, test_comps = train_test_split(unique_comps, test_size=0.2, random_state=42)
train_mask = df.set_index(['x_Nd', 'x_B', 'x_Fe']).index.isin(
    train_comps.set_index(['x_Nd', 'x_B', 'x_Fe']).index)
df_train = df[train_mask].reset_index(drop=True)
df_test  = df[~train_mask].reset_index(drop=True)
print(f"Train: {df_train['x_Nd'].nunique()} comps | Test: {df_test['x_Nd'].nunique()} comps\n")

# ── Lookup table ───────────────────────────────────────────────────────────────
LOOKUP_PHASES = [p for p in PHASES if p != 'LIQUID']
lookup_table  = (df_train[df_train['phase'].isin(LOOKUP_PHASES)]
                 .groupby('phase')[['X_Nd', 'X_B', 'X_Fe']].mean().round(6))

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
ct_test  = build_cls_table(df_test)
X_cls_train = ct_train[FEATURES].values
y_cls_train = ct_train[PHASES].values
X_cls_test  = ct_test[FEATURES].values
y_cls_test  = ct_test[PHASES].values

# ── Model factories ────────────────────────────────────────────────────────────
def make_classifiers(y_binary):
    neg = (y_binary == 0).sum()
    pos = (y_binary == 1).sum()
    spw = neg / pos if pos > 0 else 1.0
    return {
        'LightGBM':        lgb.LGBMClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                               scale_pos_weight=spw, random_state=42, verbose=-1),
        'RandomForest':    RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1,
                                                  class_weight='balanced'),
        'GradientBoosting':GradientBoostingClassifier(n_estimators=200, max_depth=4,
                                                       learning_rate=0.1, random_state=42),
        'SVM':             make_pipeline(StandardScaler(),
                               SVC(kernel='rbf', C=1.0, gamma='scale',
                                   class_weight='balanced', random_state=42)),
        'KNN':             make_pipeline(StandardScaler(),
                               KNeighborsClassifier(n_neighbors=5, weights='distance')),
    }

def make_regressors_single():
    return {
        'RandomForest':    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting':GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                                      learning_rate=0.1, random_state=42),
        'LightGBM':        lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                              random_state=42, verbose=-1),
        'LinearRegression':LinearRegression(),
    }

def make_regressors_multi():
    return {
        'RandomForest':    RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
        'GradientBoosting':MultiOutputRegressor(
                               GradientBoostingRegressor(n_estimators=100, max_depth=4,
                                                         learning_rate=0.1, random_state=42)),
        'LightGBM':        MultiOutputRegressor(
                               lgb.LGBMRegressor(n_estimators=200, max_depth=6, learning_rate=0.1,
                                                 random_state=42, verbose=-1)),
        'LinearRegression':LinearRegression(),
    }

def fit_classifier(clf, X_train, y_train):
    if len(np.unique(y_train)) < 2:
        clf = DummyClassifier(strategy='constant', constant=y_train[0])
    clf.fit(X_train, y_train)
    return clf

# ══════════════════════════════════════════════════════════════════════════════
# 1. CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
MODEL_NAMES = ['LightGBM', 'RandomForest', 'GradientBoosting', 'SVM', 'KNN']

# Store train/test F1 per model per phase
cls_train_f1 = {m: {} for m in MODEL_NAMES}
cls_test_f1  = {m: {} for m in MODEL_NAMES}

for i, phase in enumerate(PHASES):
    y_tr = y_cls_train[:, i]
    y_te = y_cls_test[:, i]
    for name, clf in make_classifiers(y_tr).items():
        clf = fit_classifier(clf, X_cls_train, y_tr)
        cls_train_f1[name][phase] = f1_score(y_tr, clf.predict(X_cls_train), zero_division=0)
        cls_test_f1[name][phase]  = f1_score(y_te, clf.predict(X_cls_test),  zero_division=0)

# ── Print: one table per model, Train F1 | Test F1 per phase ──────────────────
sep = "=" * 52
print(sep)
print("CLASSIFICATION — Train F1 vs Test F1")
print(sep)

for name in MODEL_NAMES:
    print(f"\n  {name}")
    print(f"  {'Phase':<15} {'Train F1':>10} {'Test F1':>10}")
    print(f"  {'-'*37}")
    for phase in PHASES:
        tr = cls_train_f1[name][phase]
        te = cls_test_f1[name][phase]
        print(f"  {phase:<15} {tr:>10.4f} {te:>10.4f}")
    macro_tr = np.mean(list(cls_train_f1[name].values()))
    macro_te = np.mean(list(cls_test_f1[name].values()))
    print(f"  {'MACRO':<15} {macro_tr:>10.4f} {macro_te:>10.4f}")

# ── Summary: all models macro F1 side by side ─────────────────────────────────
print(f"\n{sep}")
print("CLASSIFICATION SUMMARY — Macro F1")
print(sep)
print(f"{'Model':<20} {'Train F1':>10} {'Test F1':>10}")
print("-" * 42)
for name in MODEL_NAMES:
    macro_tr = np.mean(list(cls_train_f1[name].values()))
    macro_te = np.mean(list(cls_test_f1[name].values()))
    print(f"{name:<20} {macro_tr:>10.4f} {macro_te:>10.4f}")



# ══════════════════════════════════════════════════════════════════════════════
# 2. NP REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
REG_NAMES = ['RandomForest', 'GradientBoosting', 'LightGBM', 'LinearRegression']

reg_train_r2 = {m: {} for m in REG_NAMES}
reg_test_r2  = {m: {} for m in REG_NAMES}

for phase in PHASES:
    phase_train = df_train[df_train['phase'] == phase]
    phase_test  = df_test[df_test['phase'] == phase]
    if len(phase_train) < 50:
        continue
    X_tr = phase_train[FEATURES].values
    y_tr = phase_train['NP'].values
    X_te = phase_test[FEATURES].values
    y_te = phase_test['NP'].values
    for name, reg in make_regressors_single().items():
        reg.fit(X_tr, y_tr)
        reg_train_r2[name][phase] = r2_score(y_tr, reg.predict(X_tr))
        reg_test_r2[name][phase]  = r2_score(y_te, reg.predict(X_te)) if len(y_te) > 1 else float('nan')

# ── Print: one table per model ─────────────────────────────────────────────────
print(sep)
print("NP REGRESSION — Train R² vs Test R²")
print(sep)

for name in REG_NAMES:
    print(f"\n  {name}")
    print(f"  {'Phase':<15} {'Train R²':>10} {'Test R²':>10}")
    print(f"  {'-'*37}")
    for phase in PHASES:
        if phase not in reg_train_r2[name]:
            continue
        tr = reg_train_r2[name][phase]
        te = reg_test_r2[name][phase]
        te_str = f"{te:>10.4f}" if not np.isnan(te) else f"{'nan':>10}"
        print(f"  {phase:<15} {tr:>10.4f} {te_str}")
    valid_tr = [v for v in reg_train_r2[name].values()]
    valid_te = [v for v in reg_test_r2[name].values() if not np.isnan(v)]
    print(f"  {'MACRO':<15} {np.mean(valid_tr):>10.4f} {np.mean(valid_te):>10.4f}")

# ── Summary ────────────────────────────────────────────────────────────────────
print(f"\n{sep}")
print("NP REGRESSION SUMMARY — Macro R²")
print(sep)
print(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10}")
print("-" * 42)
for name in REG_NAMES:
    valid_tr = [v for v in reg_train_r2[name].values()]
    valid_te = [v for v in reg_test_r2[name].values() if not np.isnan(v)]
    print(f"{name:<20} {np.mean(valid_tr):>10.4f} {np.mean(valid_te):>10.4f}")



# ══════════════════════════════════════════════════════════════════════════════
# 3. LIQUID INTERNAL COMPOSITION
# ══════════════════════════════════════════════════════════════════════════════
liq_train = df_train[df_train['phase'] == 'LIQUID']
liq_test  = df_test[df_test['phase'] == 'LIQUID']
X_liq_tr  = liq_train[FEATURES].values
y_liq_tr  = liq_train[['X_Nd', 'X_B', 'X_Fe']].values
X_liq_te  = liq_test[FEATURES].values
y_liq_te  = liq_test[['X_Nd', 'X_B', 'X_Fe']].values
TARGETS   = ['X_Nd', 'X_B', 'X_Fe']

liq_train_r2 = {}
liq_test_r2  = {}

for name, reg in make_regressors_multi().items():
    reg.fit(X_liq_tr, y_liq_tr)
    liq_train_r2[name] = [r2_score(y_liq_tr[:, j], reg.predict(X_liq_tr)[:, j]) for j in range(3)]
    liq_test_r2[name]  = [r2_score(y_liq_te[:, j], reg.predict(X_liq_te)[:, j]) for j in range(3)]

print(sep)
print("LIQUID COMPOSITION — Train R² vs Test R²")
print(sep)
for name in REG_NAMES:
    print(f"\n  {name}")
    print(f"  {'Target':<10} {'Train R²':>10} {'Test R²':>10}")
    print(f"  {'-'*32}")
    for j, t in enumerate(TARGETS):
        print(f"  {t:<10} {liq_train_r2[name][j]:>10.4f} {liq_test_r2[name][j]:>10.4f}")
    print(f"  {'MEAN':<10} {np.mean(liq_train_r2[name]):>10.4f} {np.mean(liq_test_r2[name]):>10.4f}")

print(f"\n{sep}")
print("LIQUID COMPOSITION SUMMARY — Mean R²")
print(sep)
print(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10}")
print("-" * 42)
for name in REG_NAMES:
    print(f"{name:<20} {np.mean(liq_train_r2[name]):>10.4f} {np.mean(liq_test_r2[name]):>10.4f}")

# ══════════════════════════════════════════════════════════════════════════════
# 4. Correlation matrix — input vs output, saved as PNG
# ══════════════════════════════════════════════════════════════════════════════
corr_rows = []
for (x_nd, x_b, temp), grp in df_train.groupby(['x_Nd', 'x_B', 'temperature_C']):
    row = {'x_Nd': x_nd, 'x_B': x_b, 'temperature_C': temp}
    for phase in PHASES:
        ph = grp[grp['phase'] == phase]
        row[f'NP_{phase}'] = ph['NP'].values[0] if len(ph) > 0 else 0.0
        if phase == 'LIQUID' and len(ph) > 0:
            row['LIQUID_X_Nd'] = ph['X_Nd'].values[0]
            row['LIQUID_X_B']  = ph['X_B'].values[0]
            row['LIQUID_X_Fe'] = ph['X_Fe'].values[0]
    corr_rows.append(row)

corr_df  = pd.DataFrame(corr_rows).fillna(0)
input_cols  = ['x_Nd', 'x_B', 'temperature_C']
output_cols = [c for c in corr_df.columns if c not in input_cols]
corr_matrix = corr_df[input_cols + output_cols].corr().loc[output_cols, input_cols]

fig, ax = plt.subplots(figsize=(6, max(6, len(output_cols) * 0.45)))
fig.patch.set_facecolor('#0f0f0f')
ax.set_facecolor('#1a1a1a')
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm',
            vmin=-1, vmax=1, center=0, linewidths=0.4,
            cbar_kws={'label': 'Pearson r', 'shrink': 0.6}, ax=ax)
ax.set_title('Input vs Output Correlation', color='#e0e0e0', fontsize=11, fontweight='bold')
ax.tick_params(colors='#e0e0e0', labelsize=8)
ax.xaxis.label.set_color('#e0e0e0')
ax.yaxis.label.set_color('#e0e0e0')
plt.tight_layout()
plt.savefig('reports/correlation_matrix.png', dpi=150, bbox_inches='tight', facecolor='#0f0f0f')
plt.close(fig)
print("Correlation matrix saved → reports/correlation_matrix.png")

# ══════════════════════════════════════════════════════════════════════════════
# 5. Dump all results to txt
# ══════════════════════════════════════════════════════════════════════════════
with open('reports/comparison_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 52 + "\n")
    f.write("CLASSIFICATION — Train F1 vs Test F1\n")
    f.write("=" * 52 + "\n")
    for name in MODEL_NAMES:
        f.write(f"\n  {name}\n")
        f.write(f"  {'Phase':<15} {'Train F1':>10} {'Test F1':>10}\n")
        f.write(f"  {'-'*37}\n")
        for phase in PHASES:
            f.write(f"  {phase:<15} {cls_train_f1[name][phase]:>10.4f} {cls_test_f1[name][phase]:>10.4f}\n")
        f.write(f"  {'MACRO':<15} {np.mean(list(cls_train_f1[name].values())):>10.4f} "
                f"{np.mean(list(cls_test_f1[name].values())):>10.4f}\n")

    f.write("\n" + "=" * 52 + "\n")
    f.write("CLASSIFICATION SUMMARY — Macro F1\n")
    f.write("=" * 52 + "\n")
    f.write(f"{'Model':<20} {'Train F1':>10} {'Test F1':>10}\n")
    f.write("-" * 42 + "\n")
    for name in MODEL_NAMES:
        f.write(f"{name:<20} {np.mean(list(cls_train_f1[name].values())):>10.4f} "
                f"{np.mean(list(cls_test_f1[name].values())):>10.4f}\n")

    f.write("\n" + "=" * 52 + "\n")
    f.write("NP REGRESSION — Train R² vs Test R²\n")
    f.write("=" * 52 + "\n")
    for name in REG_NAMES:
        f.write(f"\n  {name}\n")
        f.write(f"  {'Phase':<15} {'Train R²':>10} {'Test R²':>10}\n")
        f.write(f"  {'-'*37}\n")
        for phase in PHASES:
            if phase not in reg_train_r2[name]:
                continue
            te = reg_test_r2[name][phase]
            te_str = f"{te:>10.4f}" if not np.isnan(te) else f"{'nan':>10}"
            f.write(f"  {phase:<15} {reg_train_r2[name][phase]:>10.4f} {te_str}\n")
        valid_tr = list(reg_train_r2[name].values())
        valid_te = [v for v in reg_test_r2[name].values() if not np.isnan(v)]
        f.write(f"  {'MACRO':<15} {np.mean(valid_tr):>10.4f} {np.mean(valid_te):>10.4f}\n")

    f.write("\n" + "=" * 52 + "\n")
    f.write("NP REGRESSION SUMMARY — Macro R²\n")
    f.write("=" * 52 + "\n")
    f.write(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10}\n")
    f.write("-" * 42 + "\n")
    for name in REG_NAMES:
        valid_tr = list(reg_train_r2[name].values())
        valid_te = [v for v in reg_test_r2[name].values() if not np.isnan(v)]
        f.write(f"{name:<20} {np.mean(valid_tr):>10.4f} {np.mean(valid_te):>10.4f}\n")

    f.write("\n" + "=" * 52 + "\n")
    f.write("LIQUID COMPOSITION — Train R² vs Test R²\n")
    f.write("=" * 52 + "\n")
    for name in REG_NAMES:
        f.write(f"\n  {name}\n")
        f.write(f"  {'Target':<10} {'Train R²':>10} {'Test R²':>10}\n")
        f.write(f"  {'-'*32}\n")
        for j, t in enumerate(TARGETS):
            f.write(f"  {t:<10} {liq_train_r2[name][j]:>10.4f} {liq_test_r2[name][j]:>10.4f}\n")
        f.write(f"  {'MEAN':<10} {np.mean(liq_train_r2[name]):>10.4f} {np.mean(liq_test_r2[name]):>10.4f}\n")

    f.write("\n" + "=" * 52 + "\n")
    f.write("LIQUID COMPOSITION SUMMARY — Mean R²\n")
    f.write("=" * 52 + "\n")
    f.write(f"{'Model':<20} {'Train R²':>10} {'Test R²':>10}\n")
    f.write("-" * 42 + "\n")
    for name in REG_NAMES:
        f.write(f"{name:<20} {np.mean(liq_train_r2[name]):>10.4f} {np.mean(liq_test_r2[name]):>10.4f}\n")

print("Results saved → reports/comparison_results.txt")
print("\n=== Done. Reports → reports/ ===")