import numpy as np
from numba import jit

@jit(nopython=True)
def inhibition(s,g,h_s,h_g): return 1/(1+np.exp(-2*(h_g*g-h_s*s)))

@jit(nopython=True)
def em_integrate(x_prev,c,params,sigma,dt):
    (h_s,h_g) = params
    
    x_next = np.zeros(x_prev.shape)
    
    s_ext = np.dot(c,x_prev[:,0])
    
    x_next[:,0] = ( inhibition(s_ext,x_prev[:,0],h_s,h_g) - x_prev[:,0])*dt+sigma*np.random.normal(0,1,x_prev.shape[0])*np.sqrt(dt)
    
    return x_next,s_ext

@jit(nopython=True)
def func_sim(params,params_var,modes):
    
    (sigma,sigma_var,sigma_IC) = params_var
    (N_cells,N_t,delta_t,oversampling,mode_periodic,mode_connectivity) = modes
    
    dt = delta_t/oversampling
    N_t_oversampling = int(N_t*oversampling)
    
    if mode_connectivity == 'NN':
        onediag = np.tri(N_cells,N_cells)-np.tri(N_cells,N_cells,-2)-np.eye(N_cells)
        if mode_periodic:
            onediag[-1,0]=1 #this creates cyclic boundaries
        c = onediag+onediag.T
    elif mode_connectivity == 'NNN':
        onediag = np.tri(N_cells,N_cells)-np.tri(N_cells,N_cells,-3)-np.eye(N_cells)
        if mode_periodic:
            onediag += np.tri(N_cells,N_cells,-6)
        c = onediag+onediag.T
    elif mode_connectivity == 'NNNN':
        onediag = np.tri(N_cells,N_cells)-np.tri(N_cells,N_cells,-4)-np.eye(N_cells)
        if mode_periodic:
            onediag += np.tri(N_cells,N_cells,-5)
        c = onediag+onediag.T
    elif mode_connectivity == 'global':
        c = np.ones((N_cells,N_cells))-np.eye(N_cells)
    elif mode_connectivity == 'global_self':
        c = np.ones((N_cells,N_cells))
    
    N_neighbours = np.sum(c,axis=0)
    c /= N_neighbours
    
    N_var = 2
    X_all = np.zeros((N_cells,N_t,N_var))
    
    if sigma_IC == 0:
        X_0 = np.zeros((N_cells,N_var))
    else:
        X_0 = np.zeros((N_cells,N_var))
        X_0[:,0] = np.abs(sigma_IC*np.random.normal(0,1,(N_cells)))
        
    X = np.zeros((N_cells,N_t_oversampling,N_var))

    X[:,0,:] = X_0
    
    for t in np.arange(N_t_oversampling-1):
        x_next,s_ext = em_integrate(X[:,t,:],c,params,sigma,dt)
        X[:,t+1,0] = np.abs(X[:,t,0] + x_next[:,0])
        X[:,t+1,1] = s_ext

    X_all[:,:,:] = X[:,::oversampling,:]
           
    return X_all

