# Simulation code for information content analysis and optimization of self-organized developmental systems


**Reference**: 
    David Brückner and Gašper Tkačik, 
    "*Information content and optimization of self-organized developmental systems*"
    PNAS 121 (23) e2322326121
    https://www.pnas.org/doi/10.1073/pnas.2322326121

**Contact**: david.brueckner@ist.ac.at

**Website**: www.davidbrueckner.de


-----------------------------------------------------------------------

Developed in Python 3.6. Dependencies:

- NumPy, SciPy, Numba

- Optional: Matplotlib

-----------------------------------------------------------------------

Contents:

**simulation_XXX.py**: front-end file for the three example models:

XXX = LIS: lateral inhibition signaling

XXX = proportions: cell-type proportioning

XXX = RD: reaction-diffusion dynamics

XXX = RD_normalizer: normalizer circuit initialized with numerically generated exponential profiles with extrinsic amplitude fluctuations

Each examples demonstrates how simulations are setup, run, optimized, evaluated using information terms, and plotted.

**fns_sim_XXX.py**: simulation code for each of the three example models.

**fns_sim_general.py**: helper functions to run the simulation code of all the models.

**fns_maps.py**: functions for mapping concentrations to cell fates.

**fns_opt_thresholds.py**: functions to optimize cell fate thresholds of simulated concentration profiles.

**fns_eval.py**: functions to evaluate entropy and information quantities for input developmental ensembles.

-----------------------------------------------------------------------


Enjoy, and please send feedback to david.brueckner@ist.ac.at!

       	   	       				    
						
-----------------------------------------------------------------------
