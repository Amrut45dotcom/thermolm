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


# ── Constrained Composition Sampling: Fe 60-70%, Nd 20-35%, B 0.1-2% ─────────
# Three bounds, two free variables (Nd, B) since Fe = 1 - Nd - B.
# Strategy: oversample (Nd, B) via LHS inside their individual boxes,
# then reject any point whose IMPLIED Fe falls outside [0.60, 0.70].
# Repeat until the target number of valid points is reached.

TARGET_N = 300

nd_lo, nd_hi = 0.0, 0.35
b_lo,  b_hi  = 0.001, 0.05
fe_lo, fe_hi = 0.60, 0.70


def valid_mask(nd, b, fe_lo=fe_lo, fe_hi=fe_hi):
    """Keep only points where Nd, B, AND the implied Fe all satisfy their bounds."""
    fe = 1.0 - nd - b
    return (nd >= nd_lo) & (nd <= nd_hi) & \
           (b  >= b_lo)  & (b  <= b_hi)  & \
           (fe >= fe_lo) & (fe <= fe_hi)


def sample_constrained_box(target_n, seed=42, batch_size=None, max_attempts=20):
    """
    Draw LHS points from the Nd x B box, reject points whose implied Fe
    falls outside [fe_lo, fe_hi], and keep drawing until target_n valid
    points are collected (or max_attempts batches are exhausted).
    """
    if batch_size is None:
        # oversample heavily — at B<=0.02, Fe>=0.60 effectively requires
        # Nd >= ~0.28-0.30, so roughly half the Nd range gets rejected
        batch_size = target_n * 4

    nd_valid, b_valid = [], []
    attempt = 0
    rng_seed = seed

    while sum(len(x) for x in [nd_valid]) < target_n and attempt < max_attempts:
        sampler = qmc.LatinHypercube(d=2, seed=rng_seed)
        raw = sampler.random(n=batch_size)

        nd_batch = nd_lo + raw[:, 0] * (nd_hi - nd_lo)
        b_batch  = b_lo  + raw[:, 1] * (b_hi  - b_lo)

        mask = valid_mask(nd_batch, b_batch)
        nd_valid.append(nd_batch[mask])
        b_valid.append(b_batch[mask])

        attempt += 1
        rng_seed += 1  # vary seed each retry so we're not redrawing the same batch

    nd_all = np.concatenate(nd_valid)
    b_all  = np.concatenate(b_valid)

    if len(nd_all) < target_n:
        print(f'[WARN] Only found {len(nd_all)} valid points after {attempt} '
              f'attempts (target was {target_n}). Consider widening fe_lo/fe_hi '
              f'or nd_lo/nd_hi — the constraint band is tight relative to the box.')
    else:
        nd_all = nd_all[:target_n]
        b_all  = b_all[:target_n]

    return nd_all, b_all


nd_all, b_all = sample_constrained_box(TARGET_N, seed=42)
fe_all = 1.0 - nd_all - b_all

# sanity check — confirm every kept point actually satisfies all three bounds
assert (nd_all >= nd_lo).all() and (nd_all <= nd_hi).all()
assert (b_all  >= b_lo).all()  and (b_all  <= b_hi).all()
assert (fe_all >= fe_lo).all() and (fe_all <= fe_hi).all()

print(f'Total valid composition points: {len(nd_all)}')
print(f'  Nd range achieved: [{nd_all.min():.4f}, {nd_all.max():.4f}]')
print(f'  B  range achieved: [{b_all.min():.4f}, {b_all.max():.4f}]')
print(f'  Fe range achieved: [{fe_all.min():.4f}, {fe_all.max():.4f}]')
if nd_all.min() > nd_lo + 1e-6:
    print(f'  [NOTE] Requested Nd range was [{nd_lo}, {nd_hi}], but Fe>=0.60 '
          f'with B<=0.02 makes Nd < {nd_all.min():.3f} infeasible. '
          f'Flag this to the professor — the stated bounds overconstrain.')

# de-duplicate, same as original script
coords = np.round(np.stack([nd_all, b_all], axis=1), 4)
_, unique_idx = np.unique(coords, axis=0, return_index=True)
nd_all = nd_all[unique_idx]
b_all  = b_all[unique_idx]
fe_all = fe_all[unique_idx]

print(f'After dedup: {len(nd_all)} unique composition points')


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
df.to_excel('FeNdB_dataset.xlsx', index=False)
df.to_csv('FeNdB_dataset.csv', index=False)
print('Saved wide format: FeNdB_dataset.xlsx / .csv')



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
df_long.to_csv('FeNdB_dataset_long.csv', index=False)
print('Saved long format: FeNdB_dataset_long.csv')