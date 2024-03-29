import numpy as np
import fns_sim_general as fns_sim_gen
import fns_maps
import fns_eval
import fns_opt_thresholds

import fns_sim_LIS as fns_sim_model
model = fns_sim_model.func_sim
fate_map = fns_maps.func_fate_map_Zfates

N_sys = 100 #number of replicates
N_cells = 8 #number of cells
N_var = 2 #number of recorded variables
Z = 2 #number of fates

mode_periodic = 0 #periodic or closed boundary conditions
mode_connectivity = 'NN' #connectivity of cells
    
sigma = 0.1 #intrinsic noise
sigma_IC = 0.1 #initial condition noise
sigma_var = 0 #extrinsic noise

#define model parameters
h_s = 4
h_g = 2

g_max = 1 #max concentration (for threshold optimization)

N_t = 200 #number of time-steps
delta_t = 0.1 #time-step
oversampling = 10 #subsampling of time-step
dt = delta_t/oversampling

modes = (N_cells,N_t,delta_t,oversampling,mode_periodic,mode_connectivity)
params = (h_s,h_g)
params_var = (sigma,sigma_var,sigma_IC)

#run simulation for concentration profiles
X_vec_all = fns_sim_gen.func_sim_ensemble(model,params,params_var,modes,N_sys,N_var=N_var)

#run optimization of fate thresholds and generate fate patterns
S_rep,S_pat,S_cf,z_vec_all,params_thresh = fns_opt_thresholds.thresholds_opt(fate_map,X_vec_all,Z,g_max)

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
    plt.plot(X_vec_all[0,j,:,0])
   
plt.xlabel('time')
plt.ylabel('concentration')
    
chrt+=1
plt.subplot(H,W,chrt)
for it in range(0,N_sys):
    plt.plot(X_vec_all[it,:,-1,0])
      
plt.xlabel('cells')
plt.ylabel('concentration')
    
chrt+=1  
plt.subplot(H,W,chrt)
N_sys_plot = 100
plt.imshow( np.rot90(z_vec_all[:N_sys_plot,:],k=0),cmap='binary',vmin=0,vmax=1,aspect='auto')

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
