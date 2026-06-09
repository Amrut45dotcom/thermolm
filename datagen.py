from pycalphad import Database, equilibrium
import pycalphad.variables as v
import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.interpolate import interp1d
import warnings
warnings.filterwarnings('ignore')


# ── Database & System Setup ──────────────────────────────────────────────────
db = Database('bfend_hal.tdb.txt')
components = ['ND', 'FE', 'B', 'VA']
phases = list(db.phases.keys())
print('Phases in DB:', phases)

T_celsius = np.arange(300, 1550, 50)   # 300°C … 1500°C, step 50
T_kelvin  = T_celsius + 273.15         # 573K … 1823K — within ND function limit of 1800K
                                        # NOTE: ND Gibbs functions only parametrized to 1800K


# ── Composition Validity ─────────────────────────────────────────────────────
def valid_mask(nd, b, tol=1e-9):
    """Keep only physically valid compositions in the Fe-Nd-B ternary."""
    return (nd >= 0) & (b >= 0) & (nd + b <= 1.0 - tol)


# ── 1) LHS Sampling ──────────────────────────────────────────────────────────
# Two independent variables: x_Nd, x_B  (x_Fe = 1 - x_Nd - x_B)
n_lhs = 300
sampler = qmc.LatinHypercube(d=2, seed=42)
lhs_raw = sampler.random(n=n_lhs)

nd_lhs = lhs_raw[:, 0]
b_lhs  = lhs_raw[:, 1] * (1.0 - lhs_raw[:, 0])   # uniform fill of composition triangle
mask = valid_mask(nd_lhs, b_lhs)
nd_lhs, b_lhs = nd_lhs[mask], b_lhs[mask]


# ── 2) Random Sampling ───────────────────────────────────────────────────────
rng = np.random.default_rng(seed=0)
n_random = 200
nd_rnd = rng.uniform(0, 1, n_random)
b_rnd  = rng.uniform(0, 1, n_random)
mask = valid_mask(nd_rnd, b_rnd)
nd_rnd, b_rnd = nd_rnd[mask], b_rnd[mask]


# ── 3) Extreme / Boundary Sampling ──────────────────────────────────────────
# Corners: near-pure Fe, near-pure Nd, near-pure B
corners = np.array([
    [0.0005, 0.0005],   # near-pure Fe  (Nd≈0, B≈0)
    [0.999,  0.0005],   # near-pure Nd  (Nd≈1, B≈0)
    [0.0005, 0.999 ],   # near-pure B   (Nd≈0, B≈1)
])

# Binary edges: Fe-Nd (B≈0), Fe-B (Nd≈0), Nd-B (Fe≈0)
edges = []
for frac in np.arange(0.05, 1.0, 0.05):
    edges.append([frac,        1e-4         ])   # Fe-Nd edge  (B≈0)
    edges.append([1e-4,        frac         ])   # Fe-B  edge  (Nd≈0)
    edges.append([frac,        1.0-frac-1e-4])   # Nd-B  edge  (Fe≈0)
edges = np.array(edges)

# Dilute compositions: one element at a time at low concentration
# Physically important: Nd-lean (Fe-rich) and B-lean compositions near Nd2Fe14B
dilute = []
for frac in [0.01, 0.02, 0.05]:
    for b_f in np.arange(0.1, 0.9, 0.1):        # dilute Nd
        if frac + b_f < 1.0:
            dilute.append([frac, b_f])
    for nd_f in np.arange(0.1, 0.9, 0.1):       # dilute B
        if nd_f + frac < 1.0:
            dilute.append([nd_f, frac])
dilute = np.array(dilute)

nd_ext = np.concatenate([corners[:, 0], edges[:, 0], dilute[:, 0]])
b_ext  = np.concatenate([corners[:, 1], edges[:, 1], dilute[:, 1]])
mask = valid_mask(nd_ext, b_ext)
nd_ext, b_ext = nd_ext[mask], b_ext[mask]


# ── Combine & Deduplicate ────────────────────────────────────────────────────
nd_all = np.concatenate([nd_lhs, nd_rnd, nd_ext])
b_all  = np.concatenate([b_lhs,  b_rnd,  b_ext ])
fe_all = 1.0 - nd_all - b_all

coords = np.round(np.stack([nd_all, b_all], axis=1), 4)
_, unique_idx = np.unique(coords, axis=0, return_index=True)
nd_all = nd_all[unique_idx]
b_all  = b_all[unique_idx]
fe_all = fe_all[unique_idx]

print(f'Total unique composition points: {len(nd_all)}')


# ── Equilibrium Calculations ─────────────────────────────────────────────────
# Calculate on a finer T grid, then interpolate to target grid
T_calc_K = np.arange(T_kelvin[0] - 50, T_kelvin[-1] + 100, 25)
all_eq_results = []

for i in range(len(nd_all)):
    nd = float(nd_all[i])
    b  = float(b_all[i])
    cond = {
        v.T:       T_calc_K,
        v.P:       101325,
        v.X('ND'): nd,
        v.X('B'):  b,
    }
    try:
        eq = equilibrium(db, components, phases, cond)
        all_eq_results.append((i, eq))
    except Exception as e:
        print(f'  [WARN] Composition {i} (Nd={nd:.3f}, B={b:.3f}) failed: {e}')
        all_eq_results.append((i, None))

    if (i + 1) % 50 == 0:
        print(f'Done {i+1}/{len(nd_all)}')

print('Equilibrium calculations complete.')


# ── Interpolation Helper ─────────────────────────────────────────────────────
def interp_to_grid(values, src_T, tgt_T, fill=np.nan):
    """Linear interpolation from calculation grid to target temperature grid."""
    nan_mask = np.isnan(values)
    if nan_mask.all():
        return np.full(len(tgt_T), fill)
    src_T_v = src_T[~nan_mask]
    vals_v  = values[~nan_mask]
    if len(src_T_v) < 2:
        return np.full(len(tgt_T), fill)
    f = interp1d(src_T_v, vals_v, kind='linear', bounds_error=False, fill_value=np.nan)
    return f(tgt_T)


# ── Extract Phase Data ───────────────────────────────────────────────────────
rows = []

for idx, eq in all_eq_results:
    row = {
        'x_Nd': nd_all[idx],
        'x_B':  b_all[idx],
        'x_Fe': fe_all[idx],
    }

    if eq is None:
        rows.append(row)
        continue

    eq_sq = eq.squeeze()

    phase_names_arr = eq_sq.Phase.values
    np_arr          = eq_sq.NP.values
    X_arr           = eq_sq.X.values

    comp_names = [str(c) for c in eq_sq.coords['component'].values]
    nd_idx_c = comp_names.index('ND') if 'ND' in comp_names else None
    b_idx_c  = comp_names.index('B')  if 'B'  in comp_names else None
    fe_idx_c = comp_names.index('FE') if 'FE' in comp_names else None

    nT_calc, nV = phase_names_arr.shape

    unique_phases = sorted(set(
        p for row_phases in phase_names_arr
        for p in row_phases
        if p not in ('', None)
    ))

    for pname in unique_phases:
        np_series  = np.full(nT_calc, np.nan)
        xnd_series = np.full(nT_calc, np.nan)
        xb_series  = np.full(nT_calc, np.nan)
        xfe_series = np.full(nT_calc, np.nan)

        for t_i in range(nT_calc):
            for v_i in range(nV):
                if phase_names_arr[t_i, v_i] == pname:
                    val = np_arr[t_i, v_i]
                    if not np.isnan(val):
                        np_series[t_i]  = val
                        if nd_idx_c is not None: xnd_series[t_i] = X_arr[t_i, v_i, nd_idx_c]
                        if b_idx_c  is not None: xb_series[t_i]  = X_arr[t_i, v_i, b_idx_c]
                        if fe_idx_c is not None: xfe_series[t_i] = X_arr[t_i, v_i, fe_idx_c]
                    break

        np_interp  = interp_to_grid(np_series,  T_calc_K, T_kelvin)
        xnd_interp = interp_to_grid(xnd_series, T_calc_K, T_kelvin)
        xb_interp  = interp_to_grid(xb_series,  T_calc_K, T_kelvin)
        xfe_interp = interp_to_grid(xfe_series, T_calc_K, T_kelvin)

        for t_j, Tc in enumerate(T_celsius):
            T_label = f'{int(Tc)}C'
            np_val  = np_interp[t_j] if not np.isnan(np_interp[t_j]) else np.nan
            if np_val == 0.0:
                np_val = np.nan

            row[f'NP_{pname}_{T_label}']   = np_val
            row[f'X_Nd_{pname}_{T_label}'] = xnd_interp[t_j] if not np.isnan(np_val) else np.nan
            row[f'X_B_{pname}_{T_label}']  = xb_interp[t_j]  if not np.isnan(np_val) else np.nan
            row[f'X_Fe_{pname}_{T_label}'] = xfe_interp[t_j] if not np.isnan(np_val) else np.nan

    rows.append(row)

print(f'Extracted {len(rows)} rows.')


# ── Build DataFrame & Reorder Columns ───────────────────────────────────────
df = pd.DataFrame(rows)

comp_cols = ['x_Nd', 'x_B', 'x_Fe']

all_phases_found = sorted(set(
    col.split('_')[1]
    for col in df.columns
    if col.startswith('NP_')
))

ordered_cols = comp_cols.copy()
for Tc in T_celsius:
    T_label = f'{int(Tc)}C'
    for pname in all_phases_found:
        for prefix in ['NP', 'X_Nd', 'X_B', 'X_Fe']:
            col = f'{prefix}_{pname}_{T_label}'
            if col in df.columns:
                ordered_cols.append(col)

ordered_cols = [c for c in ordered_cols if c in df.columns]
df = df[ordered_cols]

print(f'DataFrame shape: {df.shape}')
print(f'Phases found: {all_phases_found}')

# NaN summary per phase
for pname in all_phases_found:
    np_cols = [c for c in df.columns if c.startswith(f'NP_{pname}_')]
    if np_cols:
        missing = df[np_cols].isna().mean().mean()
        print(f'  {pname}: {missing*100:.1f}% NaN')


# ── Save Wide Format ─────────────────────────────────────────────────────────
df.to_excel('FeNdB_ML_dataset.xlsx', index=False)
df.to_csv('FeNdB_ML_dataset.csv', index=False)
print('Saved wide format: FeNdB_ML_dataset.xlsx / .csv')


# ── Melt to Long Format ──────────────────────────────────────────────────────
# Column naming: NP_<phase>_<T>C  |  X_Nd_<phase>_<T>C  |  X_B_<phase>_<T>C  |  X_Fe_<phase>_<T>C
# Phase names in this TDB have NO underscores (e.g. FE14ND2B, LIQUID, BCC_A2)
# Exception: BCC_A2 and FCC_A1 contain one underscore — handle explicitly

def parse_col(col):
    parts = col.split('_')
    if parts[0] == 'NP':
        # NP_FE14ND2B_300C or NP_BCC_A2_300C
        temp = int(parts[-1].replace('C', ''))
        phase = '_'.join(parts[1:-1])   # handles BCC_A2, FCC_A1 correctly
        return 'NP', phase, temp
    else:
        # X_Nd_FE14ND2B_300C or X_Nd_BCC_A2_300C
        prop  = f'{parts[0]}_{parts[1]}'        # X_Nd / X_B / X_Fe
        temp  = int(parts[-1].replace('C', ''))
        phase = '_'.join(parts[2:-1])            # handles BCC_A2, FCC_A1 correctly
        return prop, phase, temp


df_long = df.melt(
    id_vars=['x_Nd', 'x_B', 'x_Fe'],
    var_name='column',
    value_name='value'
)

df_long[['property', 'phase', 'temperature_C']] = pd.DataFrame(
    df_long['column'].apply(parse_col).tolist(),
    index=df_long.index
)

df_long = df_long.drop(columns='column')

df_long = df_long.pivot_table(
    index=['x_Nd', 'x_B', 'x_Fe', 'phase', 'temperature_C'],
    columns='property',
    values='value'
).reset_index()

df_long.columns.name = None
df_long = df_long.dropna(subset=['NP'])

print(f'Long format shape: {df_long.shape}')
df_long.to_csv('FeNdB_ML_dataset_long.csv', index=False)
print('Saved long format: FeNdB_ML_dataset_long.csv')