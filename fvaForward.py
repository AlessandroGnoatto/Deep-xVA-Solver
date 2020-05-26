import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from solver import BSDESolver
from XvaSolver import XvaSolver
import xvaEquation as eqn
import RecursiveEquation as receqn
import munch
from scipy.stats import norm

 

if __name__ == "__main__":
    dim = 1 #dimension of brownian motion
    P = 2048 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval=100
    strike = 100
    r = 0.02
    sigma=0.25
    x_init=100
    config = {
                "eqn_config": {
                    "_comment": "a forward contract",
                    "eqn_name": "ForwardContract",
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval,
                    "strike":strike,
                    "r":r,
                    "sigma":sigma,
                    "x_init":x_init

                },
                "net_config": {
                    "y_init_range": [-5, 5],
                    "num_hiddens": [dim+20, dim+20],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 4000,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
                }
    config = munch.munchify(config) 
    bsde = getattr(eqn, config.eqn_config.eqn_name)(config.eqn_config)
    tf.keras.backend.set_floatx(config.net_config.dtype)
    
    #apply algorithm 1
    bsde_solver = BSDESolver(config, bsde)
    training_history = bsde_solver.train()  

    
    #apply trained model to evaluate value of the forward contract via Monte Carlo
    simulations = bsde_solver.model.simulate_path(bsde.sample(P))    
     
    #estimated epected positive and negative exposure
    time_stamp = np.linspace(0,1,num_time_interval+1)
    epe = np.mean(np.exp(-r*time_stamp)*np.maximum(simulations,0),axis=0)    
    ene = np.mean(np.exp(-r*time_stamp)*np.minimum(simulations,0),axis=0)

    #exact solution
    rv = norm()    
    
    d1 = np.array([(-r * s + np.log(x_init/strike) + (r+sigma**2/2)*s)/sigma/np.sqrt(s) 
                for s in time_stamp[1:]])
    d2 = np.array([d1[i]-sigma*np.sqrt(s) for i,s in enumerate(time_stamp[1:])])

    epe_exact = x_init*rv.cdf(d1) - strike*np.exp(-r)*rv.cdf(d2)
    ene_exact = x_init*rv.cdf(-d1) - strike*np.exp(-r)*rv.cdf(-d2)

    plt.figure()
    plt.plot(time_stamp,[epe_exact[0]]+list(epe_exact),'b--',label='DEPE = exact solution')
    plt.plot(time_stamp,epe[0],'b',label='DEPE = deep solver approximation')

    plt.plot(time_stamp,[ene_exact[0]]+list(ene_exact),'r--',label='DNPE = exact solution')
    plt.plot(time_stamp,ene[0],'r',label='DNPE = deep solver approximation')

    plt.xlabel('t')
    plt.legend()

    plt.show()

   # bsde_solver.model.save('testmodel.tf',save_format='tf')
   
   # XVA computation step. 
    r_f = 0.04
    configFVA = {
                "eqn_config": {
                    "_comment": "XVA on a forward",
                    "eqn_name": "FVA",
                    "total_time": total_time,
                    "num_time_interval": num_time_interval,
                    "r":r,
                    "r_fl": r_f,
                    "r_fb": r_f,
                    "r_cl": 0.00,
                    "r_cl": 0.00,
                    "clean_value": bsde,
                    "clean_value_model": bsde_solver.model
                },
                "net_config": {
                    "y_init_range": [-5, 5],
                    "num_hiddens": [dim+20, dim+20],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 4000,
                    "batch_size": batch_size,
                    "valid_size": 256,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
                }
    configFVA = munch.munchify(configFVA) 
    fvabsde = getattr(receqn, configFVA.eqn_config.eqn_name)(configFVA.eqn_config) 
    tf.keras.backend.set_floatx(configFVA.net_config.dtype)
    
    #apply algorithm 3
    xva_solver = XvaSolver(config, fvabsde)
    xva_training_history = xva_solver.train()  
    
    fva_simulations = xva_solver.model.simulate_path(fvabsde.sample(P))    
    
    print("Exact Values from analytic formulas")
    exactVhat = x_init - strike*np.exp(-r * total_time)
    exactV = np.exp(-(r_f - r) * total_time)*x_init - strike*np.exp(-r_f * total_time)
    exactFVA = exactVhat - exactV
    print("exactV = " + str(exactV))
    print("exactVhat = " + str(exactVhat))
    print("exactFVA = " + str(exactFVA))
    
    
    print("FVA from Algorithm 3")
    fvaFromSolver = fva_simulations[0,0,0]
    print("fvaFromSolver = " +str(fvaFromSolver) )
    fvaError = fva_simulations[0,0,0] - exactFVA
    print("error = "+ str(fvaError))
    
    

