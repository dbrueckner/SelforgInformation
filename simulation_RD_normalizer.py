import numpy as np

import fns_sim_RD_normalizer as fns_sim_model
model = fns_sim_model.func_sim

N_sys = 100 #number of replicates
N_cells = 8 #number of cells
N_var = 3 #number of recorded variables
    
sigma_var = 0.1 #extrinsic noise amplitude
decay_length = 0.5
N_sys = 10
N_cells = 20

## generate profiles with extrinsic noise
X_profiles = np.zeros((N_sys,N_cells))
x_cells = np.linspace(0,1,N_cells)
for it in range(0,N_sys):
    X_profiles[it,:] = (1+sigma_var*np.random.randn())*np.exp(-x_cells/decay_length)

# normalizer parameters
alpha_A2 = 1
kappa_A2 = 1

D_E = 1e3
alpha_E = 1
kappa_E = 1

delta_x = 1
delta_t = 0.01 #time-step
N_t = 1000 #number of time-steps

mode_periodic = 0
oversampling_x = 1
oversampling = 1000

modes = (N_cells,N_t,delta_t,oversampling,mode_periodic,oversampling_x)
params = (D_E,alpha_A2,kappa_A2,alpha_E,kappa_E)

## simulation of normalizer circuit
X_profiles_corrected = np.zeros((N_sys,N_cells,N_t,2))
for it in range(0,N_sys):
    X_profiles_corrected[it,:,:,:] = model(X_profiles[it,:],params,modes)
    

colors = ['b','green','r']
indices = [3,1,2]

import matplotlib.pyplot as plt
plt.close('all')
file_suffix = '.pdf'
fig_size = [12,6]     
params = {
          'figure.figsize': fig_size,
          }
plt.rcParams.update(params)
H,W = 2,4
chrt=0
plt.figure()
lims = True

param_ratio = kappa_E*alpha_A2/(kappa_A2*alpha_E)

ymax_C = 2.3
ymax_N = 0.7
m=0


def fn_plot(profile):
    for it in range(0,N_sys):
        plt.plot(x_cells,profile[it,:],lw=1,color='lightgrey')       
    cnt=0
    for it in indices:
        plt.plot(x_cells,profile[it,:],lw=2,color=colors[cnt])
        cnt+=1
    return 0

chrt+=1
plt.subplot(H,W,chrt)

fn_plot(X_profiles)
  
if lims:
    plt.ylim([0,ymax_C])
    plt.yticks([0,ymax_C])
plt.xlim([0,1])
plt.xticks([0,1])
plt.xlabel(r'cell index $i$')
plt.ylabel(r'activator $A$')


chrt+=1
plt.subplot(H,W,chrt)

fn_plot(X_profiles_corrected[:,:,-1,1])
 
if lims:
    plt.ylim([0,ymax_N])
    plt.yticks([0,ymax_N])
plt.xlim([0,1])
plt.xticks([0,1])
if lims:
    plt.ylim([0,ymax_N])
    plt.yticks([0,ymax_N])
plt.xlabel(r'cell index $i$')
plt.ylabel(r'normalizer $N$')


chrt+=1
plt.subplot(H,W,chrt)
for j in range(0,N_cells):
    plt.plot(X_profiles_corrected[it,j,:,1])
 
plt.xlabel(r'time')
plt.ylabel(r'normalizer $N$')


chrt+=1
plt.subplot(H,W,chrt)

plt.plot([0,ymax_N],[0,ymax_N],'-k',zorder=-1)
for it in range(0,N_sys):
    profile_Eth = (alpha_E/kappa_E)*np.mean(X_profiles[it,:])*np.ones(x_cells.shape)

    plt.scatter(profile_Eth,X_profiles_corrected[it,:,-1,1],color='lightgrey',zorder=1)    

cnt=0
for it in indices:
    profile_Eth = (alpha_E/kappa_E)*np.mean(X_profiles[it,:])*np.ones(x_cells.shape)

    plt.scatter(profile_Eth,X_profiles_corrected[it,:,-1,1],color=colors[cnt])    
    cnt+=1

if lims:
    plt.ylim([0,ymax_N])
    plt.yticks([0,ymax_N])
    plt.xlim([0,ymax_N])
    plt.xticks([0,ymax_N])
plt.ylabel(r'normalizer $N$')
plt.xlabel(r'theoretical normalizer $N_\mathrm{theory}$')

    
chrt+=1
plt.subplot(H,W,chrt)

fn_plot(X_profiles_corrected[:,:,-1,m])
 
if lims:
    plt.ylim([0,ymax_C])
    plt.yticks([0,ymax_C])
plt.xlim([0,1])
plt.xticks([0,1])
plt.xlabel(r'cell index $i$')
plt.ylabel(r'$\tilde{A}$')



chrt+=1
plt.subplot(H,W,chrt)

for it in range(0,N_sys):
    profile_Cth = param_ratio*X_profiles[it,:]/np.mean(X_profiles[it,:])
    plt.plot(x_cells,profile_Cth,lw=1,color='lightgrey')       
cnt=0
for it in indices:
    profile_Cth = param_ratio*X_profiles[it,:]/np.mean(X_profiles[it,:])
    
    plt.plot(x_cells,profile_Cth,lw=2,color=colors[cnt])
    cnt+=1

if lims:
    plt.ylim([0,ymax_C])
    plt.yticks([0,ymax_C])
plt.xlim([0,1])
plt.xticks([0,1])
plt.xlabel(r'cell index $i$')
plt.ylabel(r'theoretical $\tilde{A}_\mathrm{theory}$')


chrt+=1
plt.subplot(H,W,chrt)

plt.plot([0,ymax_C],[0,ymax_C],'-k',zorder=-1)
for it in range(0,N_sys):
    profile_Cth = param_ratio*X_profiles[it,:]/np.mean(X_profiles[it,:])
    plt.scatter(profile_Cth,X_profiles_corrected[it,:,-1,m],color='lightgrey')    

cnt=0
for it in indices:
    profile_Cth = param_ratio*X_profiles[it,:]/np.mean(X_profiles[it,:])
    
    plt.scatter(profile_Cth,X_profiles_corrected[it,:,-1,m],color=colors[cnt])    
    cnt+=1
    
if lims:
    plt.ylim([0,ymax_C])
    plt.yticks([0,ymax_C])
    plt.xlim([0,ymax_C])
    plt.xticks([0,ymax_C])
plt.ylabel(r'$\tilde{A}$')
plt.xlabel(r'theoretical $\tilde{A}_\mathrm{theory}$')
       
plt.tight_layout()