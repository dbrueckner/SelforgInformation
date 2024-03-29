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
def func_sim(params,params_var,modes):
    
    (D_A,D_B,alpha_A,alpha_B,kappa_A,kappa_B,h) = params
    (sigma,sigma_var_A,sigma_var_B,sigma_IC,mode_multiplicative) = params_var
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
    sqrtdt = np.sqrt(dt)
    N_t_oversampling = int(N_t*oversampling)
    
    dx = 1/oversampling_x
    N_cells_oversampling = int(N_cells*oversampling_x)
    
    X_all = np.zeros((N_cells,N_t,2))
    
    #INITIAL CONDITIONS
    A_prev = np.zeros(N_cells_oversampling)
    B_prev = np.zeros(N_cells_oversampling)
    for j in range(0,N_cells_oversampling):
        if sigma_IC == 0:
            A_prev[j] = np.exp(-j/(2*N_cells))
            B_prev[j] = np.exp(-j/N_cells)
        else:
            A_prev[j] = sigma_IC*np.random.rand()
            B_prev[j] = sigma_IC*np.random.rand()
    
    A_next = np.zeros(N_cells_oversampling)
    B_next = np.zeros(N_cells_oversampling)

    #embryo-level heterogeneity
    if sigma_var_A>0:
        A_var=(1+sigma_var_A*np.random.normal())
    else:
        A_var = 1      
    if sigma_var_B>0:
        B_var=(1+sigma_var_B*np.random.normal())
    else:
        B_var = 1

    
    count_t = 0
    for t in range(0,N_t_oversampling):
        
        for j in range(0,N_cells_oversampling):
            
            A_2nd = func_diff(A_prev,j,N_cells_oversampling,dx)
            B_2nd = func_diff(B_prev,j,N_cells_oversampling,dx)
            
            production_A = A_var*alpha_A*P_func(A_prev[j],B_prev[j],h)
            production_B = B_var*alpha_B*P_func(A_prev[j],B_prev[j],h)
            
            if sigma > 0:
                if mode_multiplicative:
                    noise_term_A = sigma*np.sqrt(production_A)*np.random.normal()*sqrtdt
                    noise_term_B = sigma*np.sqrt(production_B)*np.random.normal()*sqrtdt
                else:
                    noise_term_A = sigma*np.random.normal()*sqrtdt
                    noise_term_B = sigma*np.random.normal()*sqrtdt
            else:
                noise_term_A = 0
                noise_term_B = 0

            A_next[j] = np.abs(A_prev[j] + ( D_A*A_2nd + production_A - kappa_A*A_prev[j] )*dt + noise_term_A)
            
            B_next[j] = np.abs(B_prev[j] + ( D_B*B_2nd + production_B - kappa_B*B_prev[j] )*dt + noise_term_B)
            
        A_prev = A_next
        B_prev = B_next

        if(np.mod(t,oversampling)==0):
            X_all[:,count_t,0] = A_next[::oversampling_x]
            X_all[:,count_t,1] = B_next[::oversampling_x]
            count_t += 1
                
    return X_all

def return_modes(ell_A,ell_B,kappa_B,delta_t,delta_x):
    dx_min = 0.2*min(ell_A,ell_B)
    oversampling_x = max(delta_x,int(delta_x/dx_min)) #oversampling_x should not be smaller than delta_x=1
    dx = delta_x/oversampling_x
    
    D_B = kappa_B*ell_B**2
    criterion = 2*dx**2/(4*D_B+kappa_B*dx**2)
    dt_ideal = 0.2*criterion
    oversampling = int(delta_t/dt_ideal)
    
    return oversampling,oversampling_x,dx