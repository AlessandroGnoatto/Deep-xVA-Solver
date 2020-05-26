import numpy as np
import tensorflow as tf

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
