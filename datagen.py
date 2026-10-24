import argparse
import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from scipy.stats import qmc
from scipy.interpolate import interp1d
from tqdm import tqdm
from pycalphad import Database, Model, equilibrium
import pycalphad.variables as v

warnings.filterwarnings('ignore')


# ── EDIT THESE — everything except the TDB file itself ──────────────────
ELEMENTS = None          # e.g. 'DY,FE,ND,B' — or None to auto-use all elements in the TDB
EXCLUDE_PHASES = ''      # comma-separated phase names to drop, e.g. 'FCC_A1#2,BCC_A2#2'
N_LHS = 600
N_RANDOM = 400
T_MIN_C = 300
T_MAX_C = 1500
T_STEP_C = 100
WORKERS = 15
OUT_PREFIX = None        # None = auto-include the TDB filename, so output always shows which TDB made it
FOCUS_POINTS = [
    # -- T1 (FE14ND2B): fixed Fe:B:RE ratio, RE sublattice is a (Nd,Dy) solid
    # solution -- these six points sample evenly along that Nd<->Dy
    # substitution line (x_Dy = 0, 0.0236, 0.0472, 0.0708, 0.0944, 0.118)
    # instead of just 3 sparse endpoints, so there's no gap in coverage
    # across the line. Each center keeps a tight spread (0.01) so Fe/B stay
    # pinned to the real T1 ratio -- a single wide-spread center was
    # considered and rejected: the spread is applied independently to all
    # four elements, so widening it enough to span the full Dy range would
    # also drag Fe/B off their fixed ratio, and a Gaussian's density peaks
    # at the center and thins toward the tails anyway, under-sampling
    # exactly the pure-Dy end where T1 actually starts to collapse.
    ('T1_Dy00', {'FE': 0.824, 'B': 0.059, 'ND': 0.118, 'DY': 0.000}, 200, 0.01),
    ('T1_Dy02', {'FE': 0.824, 'B': 0.059, 'ND': 0.0944, 'DY': 0.0236}, 200, 0.01),
    ('T1_Dy05', {'FE': 0.824, 'B': 0.059, 'ND': 0.0708, 'DY': 0.0472}, 200, 0.01),
    ('T1_Dy07', {'FE': 0.824, 'B': 0.059, 'ND': 0.0472, 'DY': 0.0708}, 200, 0.01),
    ('T1_Dy09', {'FE': 0.824, 'B': 0.059, 'ND': 0.0236, 'DY': 0.0944}, 200, 0.01),
    ('T1_Dy12', {'FE': 0.824, 'B': 0.059, 'ND': 0.000, 'DY': 0.118}, 200, 0.01),

    # -- T2: two separate stoichiometric phases in this TDB (not one solution)
    # -- FE4NDB4 (Nd-side) and DYFE4B4 (Dy-side, added via Yang et al. 2025).
    # Centers are the exact TDB PHASE/CONSTITUENT site fractions, not the
    # naive formula ratio (FE4NDB4's site fractions are 0.4396/0.4396/0.1209,
    # not the nominal 4:4:1 = 0.4444/0.4444/0.1111 that DYFE4B4 actually uses).
    ('T2_Nd', {'FE': 0.43956044, 'B': 0.43956044, 'ND': 0.12087912, 'DY': 0.0}, 150, 0.01),
    ('T2_Dy', {'FE': 0.4444, 'B': 0.4444, 'ND': 0.0, 'DY': 0.1111}, 150, 0.01),

    # -- Fe17Re2: FE17ND2 (Nd-side) and DY2FE17 (Dy-side), exact TDB site
    # fractions.
    ('Fe17Re2_Nd', {'FE': 0.8947, 'B': 0.0, 'ND': 0.1053, 'DY': 0.0}, 150, 0.01),
    ('Fe17Re2_Dy', {'FE': 0.894737, 'B': 0.0, 'ND': 0.0, 'DY': 0.105263}, 150, 0.01),

    # -- Fe17Re5: only FE17ND5 exists in this TDB, no Dy version.
    ('Fe17Re5_Nd', {'FE': 0.7727, 'B': 0.0, 'ND': 0.2273, 'DY': 0.0}, 150, 0.01),

    # -- DHCP: true (Nd,Dy) solution phase (CONSTITUENT DHCP :ND,DY: -- no
    # Fe/B on its own sublattice), present across a wide overall-alloy range
    # (dataset: B 0-0.63, DY 0-0.9999, FE 0-0.84, ND 0-0.95). One blob can't
    # cover that, so these four centers spread across the Nd/Dy ratio and
    # RE-richness instead of clustering on a single point. Wider spread than
    # the stoichiometric points above since this is a diffuse region, not a
    # pinned composition.
    ('DHCP_NdRich', {'ND': 0.85, 'DY': 0.0, 'FE': 0.10, 'B': 0.05}, 100, 0.03),
    ('DHCP_DyRich', {'DY': 0.85, 'ND': 0.0, 'FE': 0.10, 'B': 0.05}, 100, 0.03),
    ('DHCP_Mixed', {'ND': 0.35, 'DY': 0.35, 'FE': 0.20, 'B': 0.10}, 100, 0.03),
    ('DHCP_LowRE', {'FE': 0.55, 'ND': 0.20, 'DY': 0.15, 'B': 0.10}, 100, 0.03),

    # -- LIQUID: genuinely spans nearly the whole simplex (dataset: B 0-0.65,
    # DY 0-0.9999, FE 0-0.95, ND 0-0.9999). Same spread-centers approach as
    # DHCP, covering the Fe-rich, RE-rich, and B-rich corners of the region
    # where LIQUID actually appears.
    ('LIQUID_FeRich', {'FE': 0.65, 'ND': 0.15, 'DY': 0.10, 'B': 0.10}, 100, 0.03),
    ('LIQUID_RErich', {'ND': 0.40, 'DY': 0.40, 'FE': 0.15, 'B': 0.05}, 100, 0.03),
    ('LIQUID_Brich', {'B': 0.45, 'FE': 0.35, 'ND': 0.15, 'DY': 0.05}, 100, 0.03),
]


# ── CLI ───────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('tdb_path')
    return p.parse_args()


# ── Composition sampling ─────────────────────────────────────────────────
def stick_break_to_simplex(u):
    """(n, k) uniforms in [0,1] -> points UNIFORMLY distributed on the k-dim
    simplex (sum <= 1).

    FIX (bug #3): naive sequential stick-breaking (comp[:,i] = remaining *
    u[:,i]) is NOT uniform on the simplex -- it biases earlier coordinates
    toward larger values (E[comp[i]] decays as 0.5**i), which silently
    skewed composition density by alphabetical element order (B > DY > FE
    > ND) in the LHS/random portion of the dataset. The correct construction
    transforms each u[:,i] through the inverse CDF of Beta(1, k-1-i) before
    stick-breaking, which yields a uniform distribution on the simplex.
    """
    n, k = u.shape
    comp = np.zeros((n, k))
    remaining = np.ones(n)
    for i in range(k - 1):
        # Inverse-CDF sample from Beta(1, k-1-i) applied to the existing
        # uniform draw -- drop-in replacement, doesn't touch the sampler.
        beta_sample = 1 - (1 - u[:, i]) ** (1.0 / (k - 1 - i))
        comp[:, i] = remaining * beta_sample
        remaining *= (1 - beta_sample)
    comp[:, k - 1] = remaining
    return comp


def valid_mask(comp, tol=1e-4):
    # FIX: the old bound (sum <= 1.0 - tol) only passed because the previous
    # (buggy) stick-breaking never fully consumed the remainder, so points
    # always landed strictly under 1.0. Correct stick-breaking assigns the
    # full remainder to the last coordinate, so sum lands at ~1.0 (up to fp
    # noise) for essentially every valid point -- the old bound rejected
    # nearly all of them. A proper composition should sum to 1, so allow a
    # symmetric tolerance around 1.0 (loose enough to absorb rounding from
    # add_focus_points' normalization too) instead of requiring strictly less.
    return (comp >= 0).all(axis=1) & (np.abs(comp.sum(axis=1) - 1.0) <= tol)


def sample_compositions(n_elements, n_lhs, n_random):
    sampler = qmc.LatinHypercube(d=n_elements, seed=42)
    comp_lhs = stick_break_to_simplex(sampler.random(n=n_lhs))

    rng = np.random.default_rng(seed=0)
    comp_rnd = stick_break_to_simplex(rng.uniform(0, 1, size=(n_random, n_elements)))

    extreme = []
    # near-pure corners
    for i in range(n_elements):
        pt = np.full(n_elements, 1e-4 / max(n_elements - 1, 1))
        pt[i] = 1 - 1e-4
        extreme.append(pt)
    # binary edges
    for i in range(n_elements):
        for j in range(i + 1, n_elements):
            for frac in np.arange(0.05, 1.0, 0.1):
                pt = np.full(n_elements, 1e-5)
                pt[i] = frac
                pt[j] = 1 - frac - (n_elements - 2) * 1e-5
                extreme.append(pt)
    # equimolar + dilute-in-one
    extreme.append(np.full(n_elements, 1.0 / n_elements))
    for i in range(n_elements):
        for dilute in (0.02, 0.05, 0.10):
            pt = np.full(n_elements, (1 - dilute) / max(n_elements - 1, 1))
            pt[i] = dilute
            extreme.append(pt)
    comp_ext = np.array(extreme)

    comp = np.concatenate([comp_lhs, comp_rnd, comp_ext], axis=0)
    comp = comp[valid_mask(comp)]
    comp = np.round(comp, 4)
    comp = np.unique(comp, axis=0)
    return comp


def add_focus_points(comp_all, elements):
    """Adds extra points clustered around each entry in FOCUS_POINTS (see top
    of file), so phases with a small composition range get denser sampling."""
    if not FOCUS_POINTS:
        return comp_all

    rng = np.random.default_rng(123)
    extra = []
    for name, fractions, n_points, spread in FOCUS_POINTS:
        center = np.array([fractions.get(el, 0.0) for el in elements])
        pts = rng.normal(loc=center, scale=spread, size=(n_points, len(elements)))
        pts = np.clip(pts, 1e-6, None)
        pts = pts / pts.sum(axis=1, keepdims=True)  # normalize to sum exactly to 1, no artificial shrink
        extra.append(pts)
        print(f'Added {n_points} extra points around "{name}"')

    comp_all = np.concatenate([comp_all] + extra, axis=0)
    comp_all = comp_all[valid_mask(comp_all)]
    comp_all = np.round(comp_all, 4)
    comp_all = np.unique(comp_all, axis=0)
    return comp_all


# ── Phase selection ──────────────────────────────────────────────────────
def usable_phases(db, components, exclude):
    active = set(components)
    phases = [
        name for name, obj in db.phases.items()
        if all(set(str(c) for c in sub) & active for sub in obj.constituents)
    ]
    return [p for p in phases if p not in exclude]


# ── Interpolation ────────────────────────────────────────────────────────
def interp_to_grid(values, src_T, tgt_T):
    mask = ~np.isnan(values)
    if mask.sum() < 2:
        return np.full(len(tgt_T), np.nan)
    f = interp1d(src_T[mask], values[mask], kind='linear', bounds_error=False, fill_value=np.nan)
    return f(tgt_T)


# ── Worker (one process per core, reused across points) ─────────────────
_db = None
_models = None


def _init_worker(tdb_path, components, phases):
    global _db, _models
    _db = Database(tdb_path)
    _models = {p: Model(_db, components, p) for p in phases}


def _solve_point(args):
    """
    FIX (bug #4): previously, absence of a phase was silently deleted rather
    than recorded. This function now iterates over the FULL `phases` list
    (every phase the solver could have produced), not just the phases that
    happened to appear (`found_phases`):

      - If the solver call itself fails: every phase is genuinely UNKNOWN at
        this composition (NaN across the board) -- we don't know what would
        have happened.
      - If the solver succeeds but a given phase never appears at any T for
        this composition: that's a real, legitimate NEGATIVE EXAMPLE
        (NP = 0.0), not a missing value. Internal composition (X_*) for an
        absent phase is left NaN since it's undefined when the phase doesn't
        exist.
      - If the phase appears at some T's but not this specific T within its
        stability window, interpolation may still yield NaN for a gap --
        that stays NaN as a true unknown at that specific point.

    This preserves phase-boundary information (where a phase's fraction
    goes to zero) instead of deleting it before it reaches to_long_format.
    """
    i, comp_row, elements, components, phases, T_calc_K, T_out_K = args
    row = {f'x_{el}': comp_row[k] for k, el in enumerate(elements)}

    cond = {v.T: T_calc_K, v.P: 101325}
    for el, frac in zip(elements[:-1], comp_row[:-1]):
        cond[v.X(el)] = float(frac)

    try:
        eq = equilibrium(_db, components, phases, cond, model=_models).squeeze()
    except Exception as e:
        # Genuine solver failure: every phase is truly unknown (NaN) at this
        # composition, not "absent" -- we don't know what would have happened.
        for pname in phases:
            for T_K in T_out_K:
                label = f'{int(round(T_K - 273.15))}C'
                row[f'NP_{pname}_{label}'] = np.nan
                for el in elements:
                    row[f'X_{el}_{pname}_{label}'] = np.nan
        return i, row, f'Composition {i} failed: {e}'

    phase_arr = eq.Phase.values
    np_arr = eq.NP.values
    x_arr = eq.X.values
    comp_names = [str(c) for c in eq.coords['component'].values]
    el_idx = {el: (comp_names.index(el) if el in comp_names else None) for el in elements}
    nT, nV = phase_arr.shape

    found_phases = set(p for r in phase_arr for p in r if p)

    # Iterate over ALL phases the solver had available, not just found ones,
    # so absent phases become explicit negative examples rather than gaps.
    for pname in phases:
        np_series = np.full(nT, np.nan)
        x_series = {el: np.full(nT, np.nan) for el in elements}

        if pname in found_phases:
            for t in range(nT):
                matched = False
                for vi in range(nV):
                    if phase_arr[t, vi] != pname:
                        continue
                    val = np_arr[t, vi]
                    if np.isnan(val):
                        continue
                    np_series[t] = val
                    for el in elements:
                        ci = el_idx[el]
                        if ci is not None:
                            x_series[el][t] = x_arr[t, vi, ci]
                    matched = True
                    break
                if not matched:
                    # FIX: the solver ran successfully at this specific
                    # T_calc point and simply did not produce this phase
                    # here -- that's a real, per-temperature negative
                    # example, not an unknown. Previously this was left as
                    # NaN, which then got smeared by interpolation instead
                    # of being a clean 0.0 (this is what caused LIQUID's
                    # NaN%% to be high at low T even for compositions where
                    # LIQUID legitimately just hadn't formed yet).
                    np_series[t] = 0.0
                    # x_series stays NaN at this point: composition is
                    # undefined for a phase that isn't present.
        # else: pname genuinely absent at every T for this composition --
        # np_series stays all-NaN pre-interpolation; we write explicit 0.0
        # below instead of NaN, since the solver DID run and DID conclude
        # this phase isn't stable here.

        np_interp = interp_to_grid(np_series, T_calc_K, T_out_K)
        x_interp = {el: interp_to_grid(x_series[el], T_calc_K, T_out_K) for el in elements}

        for j, T_K in enumerate(T_out_K):
            label = f'{int(round(T_K - 273.15))}C'
            np_val = np_interp[j]
            if pname not in found_phases:
                # Solver ran successfully; phase was never stable at this
                # composition -> legitimate negative example.
                row[f'NP_{pname}_{label}'] = 0.0
                for el in elements:
                    row[f'X_{el}_{pname}_{label}'] = np.nan  # undefined if phase doesn't exist
            else:
                # Phase appeared at some T's; keep NaN only where
                # interpolation itself couldn't resolve a value.
                row[f'NP_{pname}_{label}'] = np_val
                for el in elements:
                    row[f'X_{el}_{pname}_{label}'] = x_interp[el][j] if not np.isnan(np_val) else np.nan

    return i, row, None


# ── Column reordering ────────────────────────────────────────────────────
def extract_phase_name(col):
    parts = col.split('_')
    return '_'.join(parts[1:-1])


def reorder_columns(df, elements, T_celsius):
    comp_cols = [f'x_{el}' for el in elements]
    phases_found = sorted(set(extract_phase_name(c) for c in df.columns if c.startswith('NP_')))

    ordered = comp_cols.copy()
    for Tc in T_celsius:
        label = f'{int(Tc)}C'
        for pname in phases_found:
            for prefix in ['NP'] + [f'X_{el}' for el in elements]:
                col = f'{prefix}_{pname}_{label}'
                if col in df.columns:
                    ordered.append(col)
    return df[[c for c in ordered if c in df.columns]], phases_found


# ── Long format ───────────────────────────────────────────────────────────
def parse_col(col):
    parts = col.split('_')
    temp = int(parts[-1].replace('C', ''))
    if parts[0] == 'NP':
        return 'NP', '_'.join(parts[1:-1]), temp
    return f'{parts[0]}_{parts[1]}', '_'.join(parts[2:-1]), temp


def to_long_format(df, comp_cols):
    """
    FIX (bug #4): previously `dropna(subset=['NP'])` deleted every row where
    a phase's fraction was 0 (collapsed to NaN in the wide format), which
    threw away all negative examples needed to learn phase boundaries. Now
    that `_solve_point` writes explicit 0.0 for legitimate absence and NaN
    only for genuine solver failure / unresolved interpolation gaps, this
    dropna only removes true unknowns, not valid zero-fraction rows.
    """
    long_df = df.melt(id_vars=comp_cols, var_name='column', value_name='value')
    long_df[['property', 'phase', 'temperature_C']] = pd.DataFrame(
        long_df['column'].apply(parse_col).tolist(), index=long_df.index
    )
    long_df = long_df.drop(columns='column')
    # FIX: pivot_table builds a full cartesian product of every index level
    # before filling values -- with near-continuous comp_cols (thousands of
    # unique floats each) x phase x temperature_C, that blows up to TiB-scale
    # memory even though the real data is only ~n_compositions x n_phases x
    # n_temps rows. set_index + unstack only ever materializes combinations
    # that actually exist in the data.
    long_df = long_df.set_index(
        comp_cols + ['phase', 'temperature_C', 'property']
    )['value'].unstack('property').reset_index()
    long_df.columns.name = None
    # Only drop rows where NP is NaN (genuine solver failure / unknown),
    # NOT rows where NP == 0.0 (legitimate "phase absent" negative example).
    return long_df.dropna(subset=['NP'])


# ── Main ──────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    db = Database(args.tdb_path)

    if ELEMENTS:
        elements = [e.strip().upper() for e in ELEMENTS.split(',')]
    else:
        elements = sorted(e for e in db.elements if e not in ('VA', '/-', 'ELECTRON_GAS'))
    print(f'Elements: {elements}')

    components = elements + ['VA']
    exclude = set(x.strip() for x in EXCLUDE_PHASES.split(',') if x.strip())
    phases = usable_phases(db, components, exclude)
    print(f'Using {len(phases)} phases (of {len(db.phases)} in DB): {phases}')

    tdb_name = os.path.splitext(os.path.basename(args.tdb_path))[0]
    prefix = OUT_PREFIX or tdb_name
    print(f'TDB used: {args.tdb_path}  ->  output prefix: {prefix}')

    # Bracket points around the literature T1 (Nd2Fe14B) peritectic decomposition
    # (~1180C) to pin down where the solver's T1 -> 0% transition actually falls,
    # since the base 100C grid only bracketed it between 1100C and 1200C.
    T1_PERITECTIC_BRACKET_C = [1160, 1170, 1180, 1190]

    T_celsius = np.arange(T_MIN_C, T_MAX_C, T_STEP_C)
    T_celsius = np.unique(np.concatenate([T_celsius, T1_PERITECTIC_BRACKET_C]))
    T_out_K = T_celsius + 273.15

    # Base solver grid stays at its normal 50K spacing, but the bracket
    # temperatures are forced in explicitly so pycalphad solves equilibrium
    # directly at 1160/1170/1180/1190C instead of those rows being
    # interpolated from the nearest 50K-spaced solved points.
    T_calc_K_base = np.arange(T_out_K[0] - 50, T_out_K[-1] + 100, 50)
    T_bracket_K = np.array(T1_PERITECTIC_BRACKET_C) + 273.15
    T_calc_K = np.unique(np.concatenate([T_calc_K_base, T_bracket_K]))

    comp_all = sample_compositions(len(elements), N_LHS, N_RANDOM)
    comp_all = add_focus_points(comp_all, elements)
    print(f'Total unique composition points: {len(comp_all)}')

    print(f'Using {WORKERS} worker processes.')
    rows = [None] * len(comp_all)
    with ProcessPoolExecutor(max_workers=WORKERS, initializer=_init_worker,
                              initargs=(args.tdb_path, components, phases)) as executor:
        futures = {
            executor.submit(_solve_point, (i, comp_all[i], elements, components, phases, T_calc_K, T_out_K)): i
            for i in range(len(comp_all))
        }
        with tqdm(total=len(futures), desc='Equilibrium calc', unit='pt') as pbar:
            for fut in as_completed(futures):
                i, row, warn_msg = fut.result()
                if warn_msg:
                    pbar.write(f'  [WARN] {warn_msg}')
                rows[i] = row
                pbar.update(1)

    df = pd.DataFrame(rows)
    df, phases_found = reorder_columns(df, elements, T_celsius)
    print(f'DataFrame shape: {df.shape}')
    print(f'Phases found: {phases_found}')

    for pname in phases_found:
        np_cols = [c for c in df.columns if c.startswith(f'NP_{pname}_')]
        if np_cols:
            print(f'  {pname}: {df[np_cols].isna().mean().mean() * 100:.1f}% NaN')

    df.to_csv(f'{prefix}_dataset.csv', index=False)
    df.to_excel(f'{prefix}_dataset.xlsx', index=False)
    print(f'Saved: {prefix}_dataset.csv / .xlsx')

    comp_cols = [f'x_{el}' for el in elements]
    long_df = to_long_format(df, comp_cols)
    long_df.to_csv(f'{prefix}_dataset_long.csv', index=False)
    print(f'Saved: {prefix}_dataset_long.csv ({long_df.shape})')


if __name__ == '__main__':
    main()
