import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xvaEquation as eqn
from solver import BSDESolver
from XvaSolver import XvaSolver
import RecursiveEquation as receqn
import munch
import pandas as pd

 

if __name__ == "__main__":
    dim = 100 #dimension of brownian motion
    P = 2048 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval = 100
    r = 0.01
    sigma = 0.25
    x_init = 100
    strike =  x_init*dim
    exact = 157.99
    
    config = {

                "eqn_config": {
                    "_comment": "a basket call option",
                    "eqn_name": "BasketOption",
                    "total_time": total_time,
                    "dim": dim,
                    "num_time_interval": num_time_interval,
                    "strike":strike,
                    "r":r,
                    "sigma":sigma,
                    "x_init":x_init
                },

                "net_config": {
                    "y_init_range": [150, 170],#[154.37,165.41], #set to None when not sure
                    "num_hiddens": [dim+10, dim+10],
                    "lr_values": [5e-2, 5e-3],#[5e-1,5e-2, 5e-3],
                    "lr_boundaries": [2000],#[1000,2000],
                    "num_iterations": 4000,
                    "batch_size": batch_size,
                    "valid_size": 128,
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

    #Simulate the BSDE after training - MtM scenarios
    simulations = bsde_solver.model.simulate_path(bsde.sample(P))
    
    #estimated epected positive and negative exposure
    time_stamp = np.linspace(0,1,num_time_interval+1)
    epe = np.mean(np.exp(-r*time_stamp)*np.maximum(simulations,0),axis=0)    
    ene = np.mean(np.exp(-r*time_stamp)*np.minimum(simulations,0),axis=0)
    
    epe_exact = np.array([exact for s in time_stamp[1:]])
    ene_exact = np.array([0.0 for s in time_stamp[1:]])

    fig = plt.figure()
    plt.plot(time_stamp,[exact] + list(epe_exact),'b--',label='DEPE = exact solution',)
    plt.plot(time_stamp,np.transpose(epe),'b',label='DEPE = deep solver approximation')

    plt.plot(time_stamp,[0.0]+ list(ene_exact),'r--',label='DNPE = exact solution',)
    plt.plot(time_stamp,np.transpose(ene),'r',label='DNPE = deep solver approximation')

    plt.xlabel('t')
    plt.legend()

    plt.show()
    fig.savefig(config.eqn_config.eqn_name + '.pdf',format = 'pdf')
    
    df = pd.DataFrame(simulations[:,0,:])
    filepath = 'exposure' + config.eqn_config.eqn_name + '.xlsx'
    df.to_excel(filepath, index=False)

    configBCVA = {
                "eqn_config": {
                    "_comment": "BCVA on a basket call",
                    "eqn_name": "BCVA",
                    "dim": dim,
                    "total_time": total_time,
                    "num_time_interval": num_time_interval,
                    "r":r,
                    "recoveryC" : 0.3,
                    "lambdaC" : 0.1,
                    "recoveryB" : 0.4,
                    "lambdaB" : 0.01,
                    "clean_value": bsde,
                    "clean_value_model": bsde_solver.model
                },
                "net_config": {
                    "y_init_range": [0, 20],
                    "num_hiddens": [dim+10, dim+10],
                    "lr_values": [5e-2, 5e-3],
                    "lr_boundaries": [2000],
                    "num_iterations": 4000,
                    "batch_size": batch_size,
                    "valid_size": 128,
                    "logging_frequency": 100,
                    "dtype": "float64",
                    "verbose": True
                }
                }
    configBCVA = munch.munchify(configBCVA) 
    bcvabsde = getattr(receqn, configBCVA.eqn_config.eqn_name)(configBCVA.eqn_config) 
    tf.keras.backend.set_floatx(configBCVA.net_config.dtype)
    
    #apply algorithm 3
    bcva_solver = XvaSolver(configBCVA, bcvabsde)
    #loss: 1.7611e-01, Y0: 6.9664e-01,
    bcva_training_history = bcva_solver.train()  
    
    bcva_simulations = bcva_solver.model.simulate_path(bcvabsde.sample(P)) 
    #(0.699395244753698, [0.6903630282972714, 0.7084274612101246])
    print(bcvabsde.monte_carlo(100000))