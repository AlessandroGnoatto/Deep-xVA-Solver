import numpy as np
import tensorflow as tf
from tqdm import tqdm

class RecursiveEquation(object):
    """Base class for defining PDE related function."""

    def __init__(self, eqn_config):
        self.dim = eqn_config.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None
        self.clean_bsde = eqn_config.clean_bsde
        self.clean_bsde_model = eqn_config.clean_bsde_model

    def sample(self, num_sample):
        """Sample forward SDE."""
        """Sample clean BSDE and the collateral account."""
        raise NotImplementedError

    def f_tf(self, t, x, y, z, vhat, collateral):
        """Generator function in the PDE."""
        raise NotImplementedError

    def g_tf(self, t, x, vhat,collateral):
        """Terminal condition of the PDE."""
        raise NotImplementedError

class FVA(RecursiveEquation):
    def __init__(self,eqn_config):
        self.dim = eqn_config.clean_value.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None       
        self.rate = eqn_config.r # risk-free return
        self.r_fl = eqn_config.r_fl # unsecured funding lending rate
        self.r_fb = eqn_config.r_fb # unsecured funding borrowing rate
        self.clean_value = eqn_config.clean_value  # Class Equation, to simulate dw and x
        self.clean_value_model = eqn_config.clean_value_model
# Class NonsharedModel to simulate vhat
      

    def sample(self, num_sample):
        forward_sde = self.clean_value.sample(num_sample)
        dw, x = forward_sde
        clean_value =  self.clean_value_model.simulate_path(forward_sde)
        
        #this is where we model the collateral account. 
        collateral = clean_value * 0.0
        return dw, x, clean_value, collateral

    def f_tf(self, t, x, y, z,v_clean, collateral):
        fva = (self.r_fl-self.rate)*tf.maximum(v_clean-y-collateral,0) - (self.r_fb-self.rate)*tf.maximum(collateral+y-v_clean,0)
        discount = -(self.rate)*y

        return  fva + discount

    def g_tf(self, t, x,v_clean, collateral):
        return 0.

class BCVA(RecursiveEquation):
    def __init__(self,eqn_config):
        self.dim = eqn_config.clean_value.dim
        self.total_time = eqn_config.total_time
        self.num_time_interval = eqn_config.num_time_interval
        self.delta_t = self.total_time / self.num_time_interval
        self.sqrt_delta_t = np.sqrt(self.delta_t)
        self.y_init = None       
        self._r = eqn_config.r      # interest rate
        self._recoveryC = eqn_config.recoveryC
        self._lambdaC = eqn_config.lambdaC
        self._recoveryB = eqn_config.recoveryB
        self._lambdaB = eqn_config.lambdaB
        self.clean_value = eqn_config.clean_value  # Class Equation, to simulate dw and x
        self.clean_value_model = eqn_config.clean_value_model
        

    def sample(self, num_sample):
        forward_sde = self.clean_value.sample(num_sample)
        dw, x = forward_sde
        clean_value =  self.clean_value_model.simulate_path(forward_sde)
        
        #this is where we model the collateral account. 
        collateral = clean_value * 0.0
        return dw, x, clean_value, collateral

    
    def f_tf(self, t, x, y, z,v_clean, collateral):
        cva = (1.0-self._recoveryC)*tf.maximum(-v_clean+collateral,0.0)*self._lambdaC
        dva = (1.0-self._recoveryB)*tf.maximum(v_clean-collateral,0.0)*self._lambdaB
        discount = (self._r + self._lambdaC + self._lambdaB)* y
        return -cva + dva - discount
                

    
    def g_tf(self, t, x,v_clean, collateral):
        return 0.
    
    def monte_carlo(self,num_sample=1024): #monte carlo estimation of CVA and DVA
        
        print('CVA and DVA Monte Carlo estimation started')
        discount = np.exp(-(self._r+self._lambdaB+self._lambdaC)*np.linspace(0,self.total_time,self.num_time_interval+1))
       
        estimate = []           
        for i in tqdm(range(num_sample//1024)): #split into batches to estimate
            dw, x, clean_value, collateral= self.sample(1024) #shape (num_sample, dim=1, num_time_interval+1) 
            phi_cva = (1-self._recoveryC)*discount*np.maximum(collateral-clean_value,0)*self._lambdaC
            phi_dva = (1-self._recoveryB)*discount*np.maximum(clean_value-collateral,0)*self._lambdaB

            #trapeziodal rule
            cva = np.sum(phi_cva,axis=-1)-(phi_cva[:,:,-1]+phi_cva[:,:,0])/2
            dva = np.sum(phi_dva,axis=-1)-(phi_dva[:,:,-1]+phi_dva[:,:,0])/2

            estimate += list(dva[:,0]-cva[:,0])


        if num_sample%1024!= 0: #calculate the remaining number of smaples
            dw, x, clean_value, collateral = self.sample(num_sample%1024) #shape (num_sample, dim=1, num_time_interval+1) 
            phi_cva = (1-self._recoveryC)*discount*np.maximum(collateral-clean_value,0)*self._lambdaC
            phi_dva = (1-self._recoveryB)*discount*np.maximum(clean_value-collateral,0)*self._lambdaB

            #trapeziodal rule
            cva = np.sum(phi_cva,axis=-1)-(phi_cva[:,:,-1]+phi_cva[:,:,0])/2
            dva = np.sum(phi_dva,axis=-1)-(phi_dva[:,:,-1]+phi_dva[:,:,0])/2

            estimate += list(dva[:,0]-cva[:,0])
        estimate = np.array(estimate)*self.delta_t #times time-interval (height of a trapezium)
        mean = np.mean(estimate)
        std = np.std(estimate)/np.sqrt(num_sample)

        return mean, [mean-3*std,mean+3*std]
