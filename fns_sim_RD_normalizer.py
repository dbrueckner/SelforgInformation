import numpy as np
from numba import jit

@jit(nopython=True)
def P_func(A,B,h):
    denominator = (A**h + B**h)
    if denominator > 0:
        result = A**h/denominator
    else:
        result = 0
    return result

@jit(nopython=True)
def func_sim(X_in,params,modes):
    
    (D_N,alpha_C,kappa_C,alpha_N,kappa_N) = params
    (N_cells,N_t,delta_t,oversampling,mode_periodic,oversampling_x) = modes
    
    if mode_periodic:
        def func_diff(c,j,N_cells,dx):
            if j==0:
                c_2nd = ( c[j+1]-2*c[j]+c[N_cells-1] )/dx**2
            elif j==N_cells-1:
                c_2nd = ( c[0]-2*c[j]+c[j-1] )/dx**2
            else:
                c_2nd = ( c[j+1]-2*c[j]+c[j-1] )/dx**2
            return c_2nd
    else:
        #Neumann BC implementation:
        def func_diff(c,j,N_cells,dx):
            if j==0:
                c_2nd = ( 2*c[j+1]-2*c[j] )/dx**2
            elif j==N_cells-1:
                c_2nd = ( -2*c[j]+2*c[j-1] )/dx**2
            else:
                c_2nd = ( c[j+1]-2*c[j]+c[j-1] )/dx**2
            return c_2nd
    
        
    dt = delta_t/oversampling
    N_t_oversampling = int(N_t*oversampling)
    
    dx = 1/oversampling_x
    N_cells_oversampling = int(N_cells*oversampling_x)
    
    X_all = np.zeros((N_cells,N_t,2))
    
    #INITIAL CONDITIONS
    C_prev = np.zeros(N_cells_oversampling)
    N_prev = np.zeros(N_cells_oversampling)
            
    C_next = np.zeros(N_cells_oversampling)
    N_next = np.zeros(N_cells_oversampling)

    A_prev = X_in

    
    count_t = 0
    for t in range(0,N_t_oversampling):
        
        for j in range(0,N_cells_oversampling):
            
            N_2nd = func_diff(N_prev,j,N_cells_oversampling,dx)

            C_next[j] = np.abs(C_prev[j] + ( alpha_C*A_prev[j] - kappa_C*N_prev[j]*C_prev[j] )*dt )

            N_next[j] = np.abs(N_prev[j] + ( D_N*N_2nd + alpha_N*A_prev[j] - kappa_N*N_prev[j] )*dt )
            
        C_prev = C_next
        N_prev = N_next

        if(np.mod(t,oversampling)==0):
            X_all[:,count_t,0] = C_next[::oversampling_x]
            X_all[:,count_t,1] = N_next[::oversampling_x]
            count_t += 1
                
    return X_all
