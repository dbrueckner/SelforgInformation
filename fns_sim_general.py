import numpy as np
from numba import jit


@jit(nopython=True)
def func_sim_ensemble(func,params,params_var,modes,N_sys,N_var=1):
    
    N_cells = modes[0]
    N_t = modes[1]

    X_vec_all = np.zeros((N_sys,N_cells,N_t,N_var))
    for it in range(0,N_sys):
        X_vec_all[it,:,:,:] = func(params,params_var,modes)
        
    return X_vec_all

@jit(nopython=True)
def func_thresh_ensemble(X_vec_all,fate_map,params_thresh):
    
    N_sys = X_vec_all.shape[0]
    N_cells = X_vec_all.shape[1]

    z_vec_all = np.zeros((N_sys,N_cells), dtype=np.int8)
    for it in range(0,N_sys):
        z_vec_all[it,:] = fate_map(X_vec_all[it,:,:,:],params_thresh)
        
    return z_vec_all
        
@jit(nopython=True)
def func_align_ensemble_max(X_vec_all,var_align=0):

    N_sys = X_vec_all.shape[0]
    X_vec_all_aligned = np.zeros(X_vec_all.shape)
    
    for it in range(0,N_sys):

        X_here = X_vec_all[it,:,:,:]
        
        #align maxima to be on the left
        j_max = np.argmax(X_here[:,-1,var_align]) #cell with maximum expression of flip1-component at final time point
        X_here = np.concatenate((X_here[j_max:,:,:],X_here[:j_max,:,:]))
        
        X_vec_all_aligned[it,:,:,:] = X_here
         
    return X_vec_all_aligned

@jit(nopython=True)
def func_align_ensemble_flip(X_vec_all,var_align=0,N_t_points=1):

    N_sys = X_vec_all.shape[0]
    X_vec_all_aligned = np.zeros(X_vec_all.shape)
    
    for it in range(0,N_sys):

        X_here = X_vec_all[it,:,:,:]
        
        #align maxima to be on the left
        if X_here[-1,-N_t_points,var_align] > X_here[0,-N_t_points,var_align]:
            X_here = np.flipud(X_here)
        
        X_vec_all_aligned[it,:,:,:] = X_here
         
    return X_vec_all_aligned
