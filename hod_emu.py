import scipy, scipy.stats
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cholesky, cho_solve, solve_triangular

from _hod_emu_sklearn_gpr_serialized import emu_sklearn_dump_mcri as emu_data_mcri, emu_sklearn_dump_mvir as emu_data_mvir, Mp

__version__='0.1'
__author__='Antonio Ragagnin <antonio.ragagnin@inaf.it>'

def non_neg_normal_sample(loc, scale,  max_iters=1000):
    "Given a numpy-array of loc and scale, return data from only-positive normal distribution."
    vals = scipy.stats.norm.rvs(loc = loc, scale=scale)
    mask_negative = vals<0.
    if(np.any(vals[mask_negative])):
        non_neg_normal_sample(loc[mask_negative], scale[mask_negative],  max_iters=1000)
    # after the recursion, we should have all positive numbers
    mask_negative = vals<0.
    if(np.any(vals[mask_negative])):
        raise Exception("non_neg_normal_sample function failed to provide  positive-normal")    
    return vals
    
def constant_times_RBF(constant, length, X1, X2):
    """ Apply constant times gaussian RBF kernel as in Eq. 7 in Ragagnin et al. 2021 """ 
    return constant*np.exp(-.5 *  cdist(X1 / length, X2 / length, metric='sqeuclidean'))

def emu_predict_mean_and_std(X, constant_value, length_scale, x_train, alpha, y_train_mean, y_train_std, L):
    """ Predict Emulator as minimised according to Eq. 2.30 in Rasmussen and Williams 2005,
        see sklearn implementation for more information:
        https://github.com/scikit-learn/scikit-learn/blob/95119c13af77c76e150b753485c662b7c52a41a2/sklearn/gaussian_process/_gpr.py#L283
    """ 
    K_trans =  constant_times_RBF(constant_value,length_scale,X,x_train)
    y_mean = K_trans.dot(alpha)
    y_mean_norm = np.array(y_train_std).T * np.array(y_mean) + y_train_mean
    L_inv = solve_triangular(L.T,  np.eye(L.shape[0]))
    K_inv = L_inv.dot(L_inv.T)
    y_var =  np.diag(constant_times_RBF(constant_value,length_scale,X,X))
    y_var_m = y_var - np.einsum("ij,ij->i",   np.dot(K_trans, K_inv), K_trans)
    #print(y_var_m.T.shape,  (y_train_std**2). shape, )
    y_var_n = np.matmul(np.array([y_var_m]).T , np.array([y_train_std])**2)
    return y_mean_norm, np.sqrt(y_var_n  )   



def log_power_law(Mp, A, beta, M):
    "Log power low of the number of satellites as Ns = A * (M/Mp)**beta, as in Eq. 4 of Ragagnin et al. 2020"
    return A + np.log(M/Mp)*beta

class GPEmulator():
    """
     Gaussian process emulator for a constant times RBF kernel as serialized from a sklearn GP emulator.
     Use this class with 
     ```
     emulator = GPEmulator().set_parameters(constant_value, length_scale, x_train, alpha, y_train_mean, y_train_std, L)
     y = emulator.predict(X)
     ```
     
     The parameters `constant_value, length_scale, x_train, alpha, y_train_mean, y_train_std, L` must be taken from the sklearn object 
     `GaussianProcessRegressor` (https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html)
     and its kernels https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.RBF.html#sklearn.gaussian_process.kernels.RBF and
     https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.kernels.ConstantKernel.html#sklearn.gaussian_process.kernels.ConstantKernel
    """
    def set_parameters(self, constant_value, length_scale, x_train, alpha, y_train_mean, y_train_std, L):
        """
        Set GPR emulation paramters. For their meaning see the code where I got inspiration: 
        https://github.com/scikit-learn/scikit-learn/blob/95119c13a/sklearn/gaussian_process/_gpr.py#L23
        """
        self.constant_value = np.array(constant_value)
        self.length_scale = np.array(length_scale)
        self.x_train = np.array(x_train)
        self.alpha = np.array(alpha)
        self.y_train_mean = np.array(y_train_mean)
        self.y_train_std = np.array(y_train_std)
        self.L = np.array(L)
        return self
    def predict(self, X):
        "Predict value from GP emulator"
        return emu_predict_mean_and_std(X, self.constant_value, self.length_scale, self.x_train,
                        self.alpha, self.y_train_mean, self.y_train_std,
                        self.L)
    
class GPEmulatorNs(GPEmulator):
    "Emulator as in Sec. 4 of Ragagnin et al. 2021, predicts Ns and mock Ns based on GPR emulator of residual of Eq. 6."
    def set_parameters(self, Mp, power_law_pivots, power_law_exponents, power_law_norms, constant_value, length_scale,
                 x_train, alpha, y_train_mean, y_train_std,
                 L, median):
        self.Mp = Mp
        self.power_law_pivots =  np.array(power_law_pivots)
        self.power_law_exponents =  np.array(power_law_exponents)
        self.power_law_norms = np.array(power_law_norms)
        self.median = median
        return super(GPEmulatorNs, self).set_parameters(constant_value, length_scale,
                 x_train, alpha, y_train_mean, y_train_std,
                 L)
    def inp_to_x(self, inp):
        median = self.median
        X = (np.log(inp/median))
        return X
    def predict_ns(self, inp):
            inp = np.atleast_2d(inp)
            _inp =  self.inp_to_x(inp)
            ln, p = self.predict(_inp)
            power_laws = [
                self.power_law_norms  + np.sum( (np.log(_inp/self.power_law_pivots))*self.power_law_exponents, axis=1)
                for _inp in inp
            ]  
            return np.exp(ln + power_laws),  p

    def Ns(self, inp, emulator_std=False):
        """
        provided an input in the form of [[Omega_m, Omega_b, sigma8, h0, scale factor, mass], ....]
        returns an array of [[<Ns>, logscatter sigma, emulator error on log A, emulator error on log B, emulator error on log sigma]...],
        where `<Ns> = A * (mass/mass_pivor)^B`, the logscatter sigma is the gaussian error on the satellite HOD fit the `log scatter sigma` is $\sigma$
        """
        inp = np.atleast_2d(inp)
        _inp = inp[:,:-1]
        M = inp[:,-1]
        r = self.predict_ns(_inp)
        A, beta, sigma = r[0].T
        p = r[1]
        _Ns = np.exp(log_power_law(self.Mp, A, beta, M))
        #print(_Ns.shape)
        #print(sigma.shape)
        #error = print(p.shape)
        errorlogA = r[1][:,0]
        errorlogB = r[1][:,1]
        errorlogsigma = r[1][:,2]
        if emulator_std:
            return np.array([_Ns, sigma, errorlogA, errorlogB, errorlogsigma]).T
        else:
            return np.array([_Ns, sigma]).T
        
    def mock_Ns(self, inp):
        inp = np.atleast_2d(inp)
        _inp = inp[:,:-1]
        M = inp[:,-1]
        r = self.predict_ns(_inp)[0]      
        A, beta, sigma = r.T
        Ns  = np.exp(log_power_law(self.Mp, A, beta, M))
        modelmu = non_neg_normal_sample(loc = Ns, scale=sigma*Ns)
        modelpois = scipy.stats.poisson.rvs(modelmu)
        modelmock = modelpois
        return modelmock

    
def get_emulator_mcri():
    return GPEmulatorNs().set_parameters(**emu_data_mcri)
def get_emulator_mvir():
    return GPEmulatorNs().set_parameters(**emu_data_mvir)

def main(*argv):
    try:
        overdensity = argv[1]
        omega_m, omega_b, sigma8, h0, z, min_mass, max_mass = map(float, argv[2:-2])
        bins = int(argv[-2]) 
        command = argv[-1]
        if overdensity=='200c':
            emu = get_emulator_mcri()
        if overdensity=='vir':
            emu = get_emulator_mvir()
        else: 
            raise Exception('overdensity must be vir or 200c')
        if command!='mock' and command!='average':
            raise Exception('command must be mock or average')
        
    except Exception as e:
        
        print('      ', file=sys.stderr)
        print('HODEmu v',__version__, 'by',__author__, file=sys.stderr)
        print('      ', file=sys.stderr)
        print('Usage: python hod_emu.py overdensity omega_m omega_b sigma8 h0 z min_mass[Msun] max_mass[Msun] bins command ', file=sys.stderr)
        print('       where overdensity can be `200c` or `vir`, ', file=sys.stderr)
        print('       omega_m, omega_b, sigma8m h0 are the cosmological parameter', file=sys.stderr)
        print('       z is the redshift', file=sys.stderr)
        print('       min_mass, max_mass and bins defines the mass range of the emulation', file=sys.stderr)
        print('       and command can be `average` to get <Ns> and the respective logscatter or `mock`, file=sys.stderr)
        print('       to get a list of Ns distributed according a Poisson+Gaussian distribution', file=sys.stderr)
        print('      ', file=sys.stderr)
        print('Example 1: python hod_emu.py 200c 0.3 0.04 0.7 0.7 0.8 1e14 1e15 average 20', file=sys.stderr)
        print('Example 2: python hod_emu.py vir 0.27 0.05 0.7 0.7 0.8 1e14 1.2e14 mock 1000', file=sys.stderr)
        print('      ', file=sys.stderr)
        print(str(e), file=sys.stderr)
        print('      ', file=sys.stderr)
    massrange = np.logspace(np.log10(min_mass), np.log10(max_mass), bins)
    input = [
        [omega_m, omega_b, sigma8, h0, 1./(1.+z), mass]
        for mass in massrange
    ]
    if command='average':
         Ns, logscatter, emuerror_logA, emuerror_logbeta, emuerror_logsigma = emu.Ns(input).T
         print('#mass[Msun] <Ns> log-scatter')
         for mass, n, s in zip(massrange, Ns, logscatter):
              print(mass, n, s)
    if command='mock':
         Ns_mock = emu.mock_Ns(input)
         print('#mass[Msun] Ns')
         for mass, n, s in zip(Ns, massrange):
              print(n, s)
if __name__ == "__main__":
    import sys
    main(sys.argv)
