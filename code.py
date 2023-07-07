class Data():
    def __init__(self, path= 'data_to_fit.dat'):
        self.path = Path(path)
        self.data = np.loadtxt(self.path)
        self.wl, f, er, _, trans = self.data.T
        self.flux, self.err = self.FluxandError_processing(f, er)
        self.model_wavelengths = self.get_modelW()
        self.data_wavelengths = self.wl/1000
        self.data_wavelengths_norm = self.norm_data_wavelengths()
                
    def unit_conversion(self, flux, distance=4.866*u.pc):
        flux_units = flux * u.erg/u.s/u.cm**2/u.nm
        flux_dens_emit = (flux_units * distance**2/const.R_jup**2).to(u.W/u.m**2/u.micron)
        return flux_dens_emit.value
        
    def FluxandError_processing(self, flux, err):
        nans = np.isnan(flux)
        flux[nans] = np.interp(self.wl[nans], self.wl[~nans], flux[~nans])
        flux = self.unit_conversion(flux)
        flux_scaling = 1./np.nanmean(flux)
        
        err[nans] = np.interp(self.wl[nans], self.wl[~nans], err[~nans])
        err = self.unit_conversion(err)
                
        return flux_scaling, err 
        
    def get_modelW(self):
        sim_res = 2e5
        dlam = 2.350/sim_res
        return np.arange(2.320, 2.371, dlam)
        
    def norm_data_wavelengths(self):
        return (self.data_wavelengths - np.nanmean(self.data_wavelengths))/\
                    (np.nanmax(self.data_wavelengths)-np.nanmin(self.data_wavelengths))
        
#     def main(self):

class Processing():
    d = Data()
    data_wavelengths = d.data_wavelengths
    model_wavelengths = d.model_wavelengths
    flux_scaling = d.flux
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
            x_obs[i, 0, :] = np.interp(self.data_wavelengths, shifted_wavelengths, xi)
        # Scaling
        x_obs[:,0] = x_obs[:,0] * self.flux_scaling
        x_obs[:,1, :] = self.data_wavelengths_norm
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