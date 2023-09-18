from DataProcuring import Data
from parameter import *
from PyAstronomy.pyasl import fastRotBroad
import astropy.constants as const
import astropy.units as u
from generate import param_set

class ProcessSpec():
    d = Data()
    data_wavelengths = d.data_wavelengths
    model_wavelengths = d.model_wavelengths
    flux_scaling = d.flux_scaling
    data_wavelengths_norm = d.data_wavelengths_norm
    
    def __call__(self, theta, x):
        self.theta = theta
        self.x = x
        self.param_set_ext, self.theta_ext = self.params_ext()  #external param set, one batch of theta ext
        self.x_new = self.process_x()
        self.theta_new = self.params_combine()
        
        return self.theta_new, self.x_new
    
    def params_ext(self):
        batch_size = self.theta.shape[0]
        # Define additional parameters
        radius = Parameter('radius', uniform_prior(0.8, 2.0))
        rv = Parameter('rv', uniform_prior(10, 30))
        limb_dark = Parameter('limb_dark', uniform_prior(0,1))
        vsini = Parameter('vsini', uniform_prior(0, 50))
        param_set_ext = ParameterSet([radius, rv, vsini, limb_dark])
        # Generate theta_ext
        theta_ext = param_set_ext.sample(batch_size)
        return param_set_ext, theta_ext
        
    def process_x(self):
        batch_size = self.theta.shape[0]
        x_obs = np.zeros((batch_size, 2, self.data_wavelengths.size))
        for i, xi, theta_ext_i in zip(range(self.x.shape[0]), self.x, self.theta_ext):
            param_dict = self.param_set_ext.param_dict(theta_ext_i)
            #Apply radius scaling
            xi = xi * param_dict['radius']**2
            # Apply line spread function and radial velocity
            xi = fastRotBroad(self.model_wavelengths,xi, param_dict['limb_dark'], param_dict['vsini'])
            shifted_wavelengths = (1+param_dict['rv']/const.c.to(u.km/u.s).value) * self.model_wavelengths
            # Convolve to instrument resolution
            x_obs[i, 0, :] = np.interp(self.data_wavelengths, shifted_wavelengths, xi) #flux at data_wavelengths
        # Scaling
        x_obs[:,0] = x_obs[:,0] * self.flux_scaling
        x_obs[:, 1, :] = self.data_wavelengths_norm
#       if np.any(np.isnan(x_obs)):
#       print('NaNs in x_obs') 
        return x_obs

        
    def params_combine(self):
        # Add theta_ext to theta's
        ## add to param_set here
        theta = self.theta.numpy()
        theta_norm = (self.theta-param_set.lower)/(param_set.upper - param_set.lower)
        theta_ext_norm = (self.theta_ext - self.param_set_ext.lower)/(self.param_set_ext.upper - self.param_set_ext.lower)
        theta_new = np.concatenate([theta_norm, theta_ext_norm], axis=-1)
        if np.any(np.isnan(theta)):
             print('NaNs in theta')

        return theta_new