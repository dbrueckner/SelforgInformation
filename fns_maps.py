import numpy as np
from numba import jit

@jit(nopython=True)
def func_fate_map_Zfates(X_vec,params_thresh):    
    
    params_thresh = np.sort(params_thresh)
    
    N_cells = X_vec.shape[0]
    z_vec = np.zeros(N_cells)
    variable = X_vec[:,-1,0]

    for j in range(0,N_cells):
        m = np.argmin(np.abs(variable[j]-params_thresh))
        if variable[j] <= params_thresh[m]:
            z_vec[j] = m
        else:
            z_vec[j] = m+1
    return z_vec
