import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from solver import BSDESolver
import xvaEquation as eqn
import munch
import pandas as pd

 

if __name__ == "__main__":
    dim = 100 #dimension of brownian motion
    P = 2048 #number of outer Monte Carlo Loops
    batch_size = 64
    total_time = 1.0
    num_time_interval = 200
    strike = 100
    r = 0.01
    sigma = 0.25
    x_init = 100
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
                    "x_init":x_init,
                    "strike":strike

                },
                "net_config": {
                    "y_init_range": [150, 170],
                    "num_hiddens": [dim+10, dim+10],
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

    #Simulate the BSDE after training - MtM scenarios
    simulations = bsde_solver.model.simulate_path(bsde.sample(P))
    
    #estimated epected positive and negative exposure
    time_stamp = np.linspace(0,1,num_time_interval+1)
    epe = np.mean(np.exp(-r*time_stamp)*np.maximum(simulations,0),axis=0)    
    ene = np.mean(np.exp(-r*time_stamp)*np.minimum(simulations,0),axis=0)
    exact = 157.9914 #101.0721 #398.08
    
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


    #Check Monte Carlo price
    dw, x = bsde.sample(10000)
    portfolio = tf.reduce_sum(x, 1,keepdims=True)
    payoff=tf.math.maximum(portfolio-dim * strike,0)
    average = tf.reduce_mean(payoff,0,keepdims=True)
    mcprice = np.exp(- r * total_time) * average[:,:,num_time_interval]
    print(mcprice)