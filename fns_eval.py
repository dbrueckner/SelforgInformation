import numpy as np
from scipy.stats import entropy

def entropy_reproducibility(x):
    N_sys,N_cells = x.shape
    unique_array,counts = np.unique(x,axis=0,return_counts=True)
    p = counts / N_sys
    return entropy(p,base=2)/N_cells

def entropy_patterning(x):
    N_cells = x.shape[1]
    nbins = x.max() + 1 
    counts = np.bincount(x.ravel(), minlength=nbins)
    p = counts / N_cells
    entropy_pat = entropy(p,base=2)
    return entropy_pat

def entropy_corrfree(x): 
    N_sys,N_cells = x.shape
    nbins = x.max() + 1 
    entropy_all = np.zeros(N_cells)
    for j in range(0,N_cells):
        counts = np.bincount(x[:,j].ravel(), minlength=nbins)
        p = counts / N_cells
        entropy_all[j] = entropy(p,base=2)
    return np.sum(entropy_all)/N_cells

def entropies_max_CI(N_cells,Z):
    S_cf = np.log2(Z)
    
    N_cells_per_fate = int(N_cells/Z)
    N_combs = np.math.factorial(N_cells)/(np.math.factorial(N_cells_per_fate)**Z)
    S_rep = np.log2(N_combs)/N_cells
    
    CI_max = S_cf - S_rep
    
    return CI_max

def infos(S_rep,S_pat,S_cf):
    utility = -S_rep + S_pat
    PI = S_pat - S_cf
    CI = S_cf - S_rep
    return utility,PI,CI

