#!/usr/bin/env python3

import scipy, scipy.stats
from argparse import RawTextHelpFormatter

import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from scipy.linalg import cholesky, cho_solve, solve_triangular

from _hod_emu_sklearn_gpr_serialized import emu_sklearn_dump_mcri as emu_data_mcri, emu_sklearn_dump_mvir as emu_data_mvir

__version__ = '0.1'
__author__ = 'Antonio Ragagnin <antonio.ragagnin@inaf.it>'
__url__ = 'https://github.com/aragagnin/HODEmu'
__description__ = 'HODEmu from Ragagnin et al. 2021, v'+__version__+' by '+__author__ +' see: '+__url__
__doc__="""
        The following will return 6 floats: A, beta, sigma, emulator error of logA, emulator error of log beta,
        and emulator error of log sigma, where A,beta and sigma comes from Eq. 4-5:
        
        Usage: ./hod_emu.py delta  omegam omegab sigma8  h0  redshift 
        Example: ./hod_emu.py 200c   .27    .04    0.8     0.7 0.8 
        
        Remember that <Ns> = A*(M/5e14)**B and sigma is its log scatter
        
        Note that the emulator has been trained to provide the abundance of halo for satellites with Mstar > 2e11 Msun.
        To rescale this value to a lower cut use Eq. 3 in the paper.
        
        See https://github.com/aragagnin/HODEmu for a complete guide on how to use this source as a python library
        """


    
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
                 L, median, factor1e10, factor2e11):
        self.Mp = Mp
        self.power_law_pivots =  np.array(power_law_pivots)
        self.power_law_exponents =  np.array(power_law_exponents)
        self.power_law_norms = np.array(power_law_norms)
        self.factor1e10 = np.array(factor1e10)
        self.factor2e11 = np.array(factor2e11)
        self.median = median
        return super(GPEmulatorNs, self).set_parameters(constant_value, length_scale,
                 x_train, alpha, y_train_mean, y_train_std,
                 L)
    def inp_to_x(self, inp):
        median = self.median
        X = (np.log(inp/median))
        return X
    def predict_A_beta_sigma(self, inp, emulator_std=False):
            inp = np.atleast_2d(inp)
            _inp =  self.inp_to_x(inp)
            ln, p = self.predict(_inp)
            power_laws = [
                self.power_law_norms  + np.sum( (np.log(_inp/self.power_law_pivots))*self.power_law_exponents, axis=1)
                for _inp in inp
            ]
            if emulator_std:
                return np.exp(ln + power_laws),  p
            else:
                return np.exp(ln + power_laws)
            
          

    
def get_emulator_m200c():
    return GPEmulatorNs().set_parameters(**emu_data_mcri)
def get_emulator_mvir():
    return GPEmulatorNs().set_parameters(**emu_data_mvir)

def main():
    import sys
    argv = sys.argv
    try:
        if(len(argv)!=7):
            raise Exception('you must provide 7 arguments')
               
        overdensity = argv[1]
        if overdensity=='200c':
            emu = get_emulator_m200c()
        elif overdensity=='vir':
            emu = get_emulator_mvir()
        else: 
            raise Exception('overdensity must be vir or 200c. Found "%s"'%overdensity)         

        omega_m, omega_b, sigma8, h0, z = map(float, argv[2:])
    except Exception as e:
        
        print('', file=sys.stderr)
        print(__description__, file=sys.stderr)
        print(__doc__, file=sys.stderr)
        print('', file=sys.stderr)
        print('Error: '+str(e), file=sys.stderr)
        print('', file=sys.stderr)
        sys.exit(1)
        
    input = [ [omega_m, omega_b, sigma8, h0, 1./(1.+z)] ]
    r = emu.predict_A_beta_sigma(input, emulator_std=True)
    A, beta, sigma = r[0][0].T
    p = r[1]
    errorlogA, errorlogB, errorlogsigma = r[1][0].T
    print('#A,    beta,  sigma,     Emu error logA, Emu error logB, Emu error log-sigma')
    print('%.4f'%A, '%.4f'%beta, '%.4f'%sigma, '    %.4e'%errorlogA, '     %.4e'%errorlogB, '     %.4e'%errorlogsigma)
        
if __name__ == "__main__":
    main()

    
    
