import numpy as np

import fns_sim_general as fns_sim_gen
import fns_eval
from scipy.optimize import differential_evolution


def thresholds_opt(fate_map,X_vec_all,Z,g_max):

    N_cells = X_vec_all.shape[1]

    def objective(v):
        params_thresh = np.sort(v)
        z_vec_all = fns_sim_gen.func_thresh_ensemble(X_vec_all,fate_map,params_thresh)
        S_rep = fns_eval.entropy_reproducibility(z_vec_all)/N_cells
        S_pat = fns_eval.entropy_patterning(z_vec_all)
        utility = -S_rep+S_pat
        return -utility
    
    bounds = [[0,g_max]]*(Z-1)
    
    result = differential_evolution(objective, bounds)
    params_thresh = result['x']
    
    z_vec_all = fns_sim_gen.func_thresh_ensemble(X_vec_all,fate_map,params_thresh)
    
    S_rep = fns_eval.entropy_reproducibility(z_vec_all)
    S_pat = fns_eval.entropy_patterning(z_vec_all)
    S_cf = fns_eval.entropy_corrfree(z_vec_all)
    
    return S_rep,S_pat,S_cf,z_vec_all,params_thresh

