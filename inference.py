import numpy as np
import joblib
import pandas as pd

classifiers      = joblib.load('models/lgbm_classifiers_constrained.pkl')
np_regressors    = joblib.load('models/rf_np_regressors_constrained.pkl')
liquid_regressor = joblib.load('models/rf_liquid_regressor_constrained.pkl')
lookup_table     = joblib.load('models/lookup_table_constrained.pkl')

PHASES = sorted(classifiers.keys())


def predict_equilibrium(x_Nd, x_B, temperature_C, verbose=True):
    point_df  = pd.DataFrame([[x_Nd, x_B, temperature_C]], columns=['x_Nd', 'x_B', 'temperature_C'])
    point_np  = np.array([[x_Nd, x_B, temperature_C]])
    result = {}

    for phase in PHASES:
        if not classifiers[phase].predict(point_df)[0]:  
            continue

        np_val = float(np_regressors[phase].predict(point_np)[0]) \
                 if phase in np_regressors else None  

        if phase == 'LIQUID':
            comp = liquid_regressor.predict(point_np)[0]  
            x_nd_p, x_b_p, x_fe_p = comp[0], comp[1], comp[2]
        else:
            row    = lookup_table.loc[phase]
            x_nd_p = row['X_Nd']
            x_b_p  = row['X_B']
            x_fe_p = row['X_Fe']
            
        result[phase] = {
            'NP':   np_val,
            'X_Nd': round(x_nd_p, 4),
            'X_B':  round(x_b_p,  4),
            'X_Fe': round(x_fe_p, 4),
        }

    
    total_np = sum(v['NP'] for v in result.values() if v['NP'] is not None)
    if total_np > 0:
        for phase in result:
            if result[phase]['NP'] is not None:
                result[phase]['NP'] = round(result[phase]['NP'] / total_np, 4)

    if verbose:
        x_Fe = round(1 - x_Nd - x_B, 4)
        print(f"\nInput: x_Nd={x_Nd}, x_B={x_B}, x_Fe={x_Fe}, T={temperature_C}°C")
        print(f"  {'Phase':<13} {'NP':>8} {'X_Nd':>8} {'X_B':>8} {'X_Fe':>8}")
        print("  " + "-" * 43)
        for phase, vals in result.items():
            print(f"  {phase:<13} {str(vals['NP']):>8} {vals['X_Nd']:>8} "
                  f"{vals['X_B']:>8} {vals['X_Fe']:>8}")

    return result



if __name__ == '__main__':
    test_points = [
        (0.118, 0.059, 300),   # near Nd2Fe14B stoichiometry, low T
        (0.25,  0.05,  700),   # Nd-rich, mid T
        (0.05,  0.02,  1400),  # Fe-rich, high T
        (0.12,  0.06,  1300),  # sintering T
    ]
    for x_Nd, x_B, T in test_points:
        predict_equilibrium(x_Nd, x_B, T)