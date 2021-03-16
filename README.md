# HODEmu

HODEmu, is both an executable and a python library that is based on [Ragagnin 2021 in prep.](https://aragagnin.github.io) and emulates satellite abundance as a function of cosmological parameters `Omega_m, Omega_b, sigma_8, h_0` and redshift.

The Emulator is trained on satellite abundance of [Magneticum simulations](https://www.magneticum.org/simulations.html) Box1a/mr spanning 15 cosmologies (see Table 1 of the paper) and on all satellites with a stellar mass cut of M<sub>*</sub> > 2 10<sup>11</sup> M<sub>&odot;</sub>. Use Eq. 3 to rescale it to a stelalr mass cut of 10<sup>10</sup>M<sub>&odot;</sub>.

The Emulator has been trained with [sklearn GPR](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html), however the class implemented in `hod_emu.py` is a stand-alone porting and does not need sklearn to be installed.

![satellite average abundance for two Magneticum Box1a/mr simulations, from Ragagnin et al. 2021](https://imgur.com/vGyhJC3.png)

**TOC**:

- [Install](#install)
- [Example 1: Obtain normalisation, logslope and gaussian scatter of Ns-M relation](#example-1-obtain-normalisation-logslope-and-gaussian-scatter-of-ns-m-relation)
- [Example 2: Produce mock catalog of galaxies](#example-2-produce-mock-catalog-of-galaxies)

## Install

You can either )1) download the file `hod_emu.py` and `_hod_emu_sklearn_gpr_serialized.py`  or (2) install it with `python -mpip install  git+https://github.com/aragagnin/HODEmu`. The package depends only on [scipy](https://www.scipy.org).
The file `hod_emu.py` can be executed from your command line interface by running `./hod_emu.py` in the installation folder.

Check this ipython-notebook for a guided usage on a python code: https://github.com/aragagnin/HODEmu/blob/main/examples.ipynb

## Example 1: Obtain normalisation, logslope and gaussian scatter of Ns-M relation

The following command will output, respectively, normalisation `A`, log-slope `\beta`, log-scatter `\sigma`, and the respective standard deviation from the emulator.
Since the emulator has been trained on the residual of the power-law dependency in Eq. 6, the errors are respectively, the standard deviation on log-A, on log-beta, and on log-sigma. Note that `--delta` can be only `200c` or `vir` as the paper only emulates these two overdensities. 

     ./hod_emu.py  200c  .27  .04   0.8  0.7   0.0 #overdensity omega_m omega_b sigma8 h0 redshift


Here below we will use `hod_emyu` as python library to plot the `Ns-M` relation.
First we use `hod_emu.get_emulator_m200c()` to obtain an instance of the Emulator class trianed on `Delta_200c`, and the function `emu.predict_A_beta_sigma(input)` to retrieve `A`,`\beta` and `\sigma`.

Note that **input** can be evaluated on a number `N` of data points (in this example only one), thus being is a N x 5 numpy array and the **return value** is  a N x 3 numpy array. 
The parameter `emulator_std=True` will also return  a  N x 3 numpy array with the corresponding emulator standard deviations.

```python
import hod_emu
Om0, Ob0, s8, h0, z = 0.3, 0.04, 0.8, 0.7, 0.9

input = [[Om0, Ob0, s8, h0, 1./(1.+z)]] #the input must be a 2d array because you can feed an array of data points

emu = hod_emu.get_emulator_m200c() # use get_emulator_mvir to obtain the emulator within Delta_vir

A, beta, sigma  =  emu.predict_A_beta_sigma(input).T #the function outputs a 1x3 matrix 

masses = np.logspace(14.5,15.5,20)
Ns = A*(masses/5e14)**beta 

plt.plot(masses,Ns)
plt.fill_between(masses, Ns*(1.-sigma), Ns*(1.+sigma),alpha=0.2)
plt.xlabel(r'$M_{\rm{halo}}$')
plt.ylabel(r'$N_s$')
plt.title(r'$M_\bigstar>2\cdot10^{11}M_\odot \ \ \ \tt{ and }  \ \ \ \ \  r<R_{\tt{200c}}$')
plt.xscale('log')
plt.yscale('log')

params_tuple, stds_tuple  =  emu.predict_A_beta_sigma(input, emulator_std=True) #here we also asks for Emulator std deviation

A, beta, sigma = params_tuple.T
error_logA, error_logbeta, error_logsigma = stds_tuple.T

print('A: %.3e, log-std A: %.3e'%(A[0], error_logA[0]))
print('B: %.3e, log-std beta: %.3e'%(beta[0], error_logbeta[0]))
print('sigma: %.3e, log-std sigma: %.3e'%(sigma[0], error_logsigma[0]))
``` 

Will show the following figure:

![Ns-M relation produced by HODEmu](https://imgur.com/2fp5Flw.png)

And print the following output:

```bash
A: 1.933e+00, log-std A: 1.242e-01
B: 1.002e+00, log-std beta: 8.275e-02
sigma: 6.723e-02, log-std sigma: 2.128e-01
```

## Example 2: Produce mock catalog of galaxies

In this example we use package [hmf](https://hmf.readthedocs.io/en/latest/) to produce a mock catalog of haloe masses.
Note that the mock number of satellite is based on a gaussian distribution with a cut on negative value (see Eq. 5 of the paper), hence the function `non_neg_normal_sample`.


```python
import hmf.helpers.sample
import scipy.stats

masses = hmf.helpers.sample.sample_mf(400,14.0,hmf_model="PS",Mmax=17,sort=True)[0]    
    
def non_neg_normal_sample(loc, scale,  max_iters=1000):
    "Given a numpy-array of loc and scale, return data from only-positive normal distribution."
    vals = scipy.stats.norm.rvs(loc = loc, scale=scale)
    mask_negative = vals<0.
    if(np.any(vals[mask_negative])):
        non_neg_normal_sample(loc[mask_negative], scale[mask_negative],  max_iters=1000)
    # after the recursion, we should have all positive numbers
    
    if(np.any(vals<0.)):
        raise Exception("non_neg_normal_sample function failed to provide  positive-normal")    
    return vals

A, beta, logscatter = emu.predict_A_beta_sigma( [Om0, Ob0, s8, h0, 1./(1.+z)])[0].T

Ns = A*(masses/5e14)**beta

modelmu = non_neg_normal_sample(loc = Ns, scale=logscatter*Ns)
modelpois = scipy.stats.poisson.rvs(modelmu)
modelmock = modelpois

plt.fill_between(masses, Ns *(1.-logscatter), Ns *(1.+logscatter), label='Ns +/- log scatter from Emu', color='black',alpha=0.5)
plt.scatter(masses, modelmock , label='Ns mock', color='orange')
plt.plot(masses, Ns , label='<Ns> from Emu', color='black')
plt.ylim([0.1,100.])
plt.xscale('log')
plt.yscale('log')
plt.xlabel(r'$M_{\rm {halo}} [M_\odot]$')
plt.ylabel(r'$N_s$')
plt.title(r'$M_\bigstar>2\cdot10^{11}M_\odot \ \ \ \tt{ and }  \ \ \ \ \  r<R_{\tt{200c}}$')

plt.legend();
```

Will show the following figure:

![Mock catalog of halos and satellite abundance produced by HODEmu](https://imgur.com/6pg3LSk.png)
