from equation import Equation
import numpy as np
import tensorflow as tf
from scipy.stats import multivariate_normal as normal

class PricingForward(Equation):
    def __init__(self,eqn_config):
        super(PricingForward, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init  # initial value of x, the underlying
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r        
        self.useExplict = False #whether to use explict formula to evaluate dyanamics of x

    
    def sample(self, num_sample):      
        
        dw_sample = normal.rvs(size=[num_sample,     
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: #use analytic solution of linear SDE
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t)
            for i in range(self.num_time_interval):   
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        else:   #use Euler-Maruyama scheme
            for i in range(self.num_time_interval):
         	    x_sample[:, :, i + 1] = (1 + self.r * self.delta_t) * x_sample[:, :, i] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])           
      
        
        return dw_sample, x_sample   

  
    def f_tf(self, t, x, y, z):
        return -self.r * y
   
    def g_tf(self, t, x):
        return x - self.strike
    
class BasketOption(Equation):
    def __init__(self, eqn_config):
        super(BasketOption, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.useExplict = False #whether to use explict formula to evaluate dyanamics of x

    def sample(self, num_sample):      
        
        dw_sample = normal.rvs(size=[num_sample,     
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: #use analytic solution of linear SDE
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t)
            for i in range(self.num_time_interval):   
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        else:   #use Euler-Maruyama scheme
            for i in range(self.num_time_interval):
         	    x_sample[:, :, i + 1] = (1 + self.r * self.delta_t) * x_sample[:, :, i] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])           
      
        
        return dw_sample, x_sample   

    def f_tf(self, t, x, y, z):
        return -self.r * y

    def g_tf(self, t, x):
        temp = tf.reduce_sum(x, 1,keepdims=True)
        return tf.maximum(temp - self.dim * self.strike, 0)

class CallOption(Equation):
    def __init__(self, eqn_config):
        super(CallOption, self).__init__(eqn_config)
        self.strike = eqn_config.strike
        self.x_init = np.ones(self.dim) * eqn_config.x_init
        self.sigma = eqn_config.sigma
        self.r = eqn_config.r
        self.useExplict = False #whether to use explict formula to evaluate dyanamics of x

    def sample(self, num_sample):      
        
        dw_sample = normal.rvs(size=[num_sample,     
                                     self.dim,
                                     self.num_time_interval]) * self.sqrt_delta_t
        if self.dim==1:
            dw_sample = np.expand_dims(dw_sample,axis=0)
            dw_sample = np.swapaxes(dw_sample,0,1)

        x_sample = np.zeros([num_sample, self.dim, self.num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self.dim]) * self.x_init  

        if self.useExplict: #use analytic solution of linear SDE
            factor = np.exp((self.r-(self.sigma**2)/2)*self.delta_t)
            for i in range(self.num_time_interval):   
                x_sample[:, :, i + 1] = (factor * np.exp(self.sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        else:   #use Euler-Maruyama scheme
            for i in range(self.num_time_interval):
         	    x_sample[:, :, i + 1] = (1 + self.r * self.delta_t) * x_sample[:, :, i] + (self.sigma * x_sample[:, :, i] * dw_sample[:, :, i])           
      
        
        return dw_sample, x_sample  

    
    def f_tf(self, t, x, y, z):
        return -self.r * y

    
    def g_tf(self, t, x):
        return tf.maximum( x - self.strike, 0)

    def SolExact(self, t, x):
        d1 = (np.log(x / self.strike) + ( self.r + 0.5 * self.sigma ** 2) * (self.total_time - t) ) / (self.sigma * np.sqrt(self.total_time - t))
        d2 = (np.log(x / self.strike) + ( self.r - 0.5 * self.sigma ** 2) * (self.total_time - t) ) / (self.sigma * np.sqrt(self.total_time - t))

        call = (x * normal.cdf(d1, 0.0, 1.0) - self.strike * np.exp(-self.r * (self.total_time - t) ) * normal.cdf(d2, 0.0, 1.0))
        return call
    
class CVA(Equation):
    def __init__(self, dim, total_time, num_time_interval):
        super(CVA, self).__init__(dim, total_time, num_time_interval)
        self._x0 = 100
        self._x_init = np.ones(self._dim) * self._x0  # x0 dinamica forward, vettore x0(i) = 100 per i=1...PricingCall.dim
        self._sigma = 0.25    # volatility dinamica forward
        self._r = 0.0        # interest rate
        self._recoveryC = 0.3
        self._lambdaC = 0.10
        self._recoveryB = 0.4
        self._lambdaB = 0.01
        

    def sample(self, num_sample):

        dw_sample = normal.rvs(size=[num_sample,     
                                     self._dim,
                                     self._num_time_interval]) * self._sqrt_delta_t

        x_sample = np.zeros([num_sample, self._dim, self._num_time_interval + 1]) 
        x_sample[:, :, 0] = np.ones([num_sample, self._dim]) * self._x_init       

        factor = np.exp((self._r-(self._sigma**2)/2)*self._delta_t)
        for i in range(self._num_time_interval):    
            x_sample[:, :, i + 1] = (factor * np.exp(self._sigma * dw_sample[:, :, i])) * x_sample[:, :, i]
        return dw_sample, x_sample   
   
    
    def f_tf(self, t, x, y, z, vHat):
        return (-(1.0-self._recoveryC)*tf.maximum(-vHat,0.0)*self._lambdaC-(self._r + self._lambdaC + self._lambdaB)* y 
                + (1.0-self._recoveryB)*tf.maximum(vHat,0.0)*self._lambdaB-(self._r + self._lambdaC + self._lambdaB)* y)

    
    def g_tf(self, t, x):
        return 0.0