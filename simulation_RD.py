import numpy as np
import fns_sim_general as fns_sim_gen
import fns_maps
import fns_eval
import fns_opt_thresholds

import fns_sim_RD as fns_sim_model
model = fns_sim_model.func_sim
fate_map = fns_maps.func_fate_map_Zfates

N_sys = 100 #number of replicates
N_cells = 8 #number of cells
N_var = 2 #number of recorded variables
Z = 3 #number of fates

mode_periodic = None #periodic or closed boundary conditions
mode_connectivity = 'global' #connectivity of cells
    
sigma = 0.05 #intrinsic noise
sigma_IC = 0.1 #initial condition noise
sigma_var = 0 #extrinsic noise

#define model parameters
alpha_A = 1
alpha_B = 4*alpha_A
kappa_A = 1
kappa_B = 2*kappa_A
h = 5
ratio_A = 4 #ratio L/ell_A
ratio_B = 2 #ratio L/ell_B
mode_multiplicative = 1

g_max = 0.05 #max concentration (for threshold optimization)

delta_x = 1
delta_t = 0.01 #time-step
N_t = 5000 #number of time-steps

ell_A = (N_cells/ratio_A)
ell_B = (N_cells/ratio_B)
D_A = kappa_A*ell_A**2
D_B = kappa_B*ell_B**2

#identify dx and dt according to CFL criterion
oversampling,oversampling_x,dx = fns_sim_model.return_modes(ell_A,ell_B,kappa_B,delta_t,delta_x)

modes = (N_cells,N_t,delta_t,oversampling,mode_periodic,oversampling_x)
params = (D_A,D_B,alpha_A,alpha_B,kappa_A,kappa_B,h)
params_var = (sigma,sigma_var,sigma_var,sigma_IC,mode_multiplicative)

#run simulation for concentration profiles
X_vec_all = fns_sim_gen.func_sim_ensemble(model,params,params_var,modes,N_sys,N_var=N_var)
X_vec_all_align = fns_sim_gen.func_align_ensemble_flip(X_vec_all) #align profiles

#run optimization of fate thresholds and generate fate patterns
S_rep,S_pat,S_cf,z_vec_all,params_thresh = fns_opt_thresholds.thresholds_opt(fate_map,X_vec_all_align,Z,g_max)

#evaluate information terms
utility,PI,CI = fns_eval.infos(S_rep,S_pat,S_cf)

#plotting
import matplotlib.pyplot as plt
plt.close('all')
file_suffix = '.png'
fig_size = [5,5]     
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
H,W = 2,2

chrt=0
plt.figure()

chrt+=1
plt.subplot(H,W,chrt)
for j in range(0,N_cells):
    plt.plot(X_vec_all_align[0,j,:,0])
   
plt.xlabel('time')
plt.ylabel('concentration')
    
chrt+=1
plt.subplot(H,W,chrt)
for it in range(0,N_sys):
    plt.plot(X_vec_all_align[it,:,-1,0])
      
plt.xlabel('cells')
plt.ylabel('concentration')
    
chrt+=1  
plt.subplot(H,W,chrt)
N_sys_plot = 100
plt.imshow( np.rot90(z_vec_all[:N_sys_plot,:],k=0),cmap='Set1',vmin=0,vmax=9,aspect='auto')

plt.xlabel('cells')
plt.ylabel('replicates')

plt.xticks([])
plt.yticks([])

chrt+=1  
plt.subplot(H,W,chrt)
xx = np.arange(6)
plt.bar(xx,np.array([S_rep,S_pat,S_cf,PI,CI,utility])/np.log2(Z),width=0.5)
plt.xticks(xx,labels=[r'$S_\mathrm{rep}$',r'$S_\mathrm{pat}$',r'$S_\mathrm{cf}$',r'$\mathrm{PI}$',r'$\mathrm{CI}$',r'$U$'],rotation = 45,fontsize=10)
plt.ylabel('information (bits)')
plt.ylim([0,1])
plt.yticks([0,1])

plt.tight_layout()
