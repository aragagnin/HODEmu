# HODEmu

HODEmu, is both an executable and a python library that is based on [Ragagnin 2021 in prep.](https://aragagnin.github.io) and emulates satellite abundance as a function of cosmological parameters `Omega_m, Omega_b, sigma_8, h_0` and redshift. The Emulator is trained on satellite abundance of [Magneticum simulations](https://www.magneticum.org/simulations.html) Box1a/mr spanning 15 cosmologies (see Table 1 of the paper) and on all satellites with a stellar mass cut of M<sub>*</sub> > 2 10<sup>11</sup> M<sub>&odot;</sub>.

The Emulator has been trained with [sklearn GPR](https://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html), however the class implemented in `hod_emu.py` is a stand-alone porting and does not need sklearn to be installed.

**TOC**:

- [Install](#install)
- [Execute HODEmu](#execute-hodemu)
- [Example 1: Obtain normalisation, logslope and gaussian scatter of Ns-M relation](#example-1-obtain-normalisation-logslope-and-gaussian-scatter-of-ns-m-relation)
- [Example 2: Produce mock catalog of galaxies](#example-2-produce-mock-catalog-of-galaxies)

## Install

You can either download the file `hod_emu.py` and `_hod_emu_sklearn_gpr_serialized.py`  or install it with `python -mpip install  git+https://github.com/aragagnin/HODEmu`.
The package depends only on [scipy](https://www.scipy.org).

## Execute HODEmu

The file `hod_emu.py` can be executed from your command line interface by running `./hod_emu.py` in the installation folder.
If you installed it through `pip` you can execute it by running `python -mhod_emu`.
Finally, you can integrate `hod_emu` in your python code by adding `import hod_emu`.

Check this ipython-notebook for a guided usage:

## Example 1: Obtain normalisation, logslope and gaussian scatter of Ns-M relation

The command following command will output, respectively, normalisation A, log-slope \beta, log-scatter \sigma, and the respective standard deviation from the emulator.
Note that `--delta` can be only `200c` or `vir` as the paper only emulates these two overdensities.
Since the emulator has been trained on the residual of the power-law dependency in Eq. 6, the errors are respectively, the standard deviation on log-A, on log-beta, and on log-sigma.

        ./hod_emu.py --delta 200c --omegam .27 --omegab .04 --sigma8 0.8 --h0 0.7 --z 0.8

The following python code will plot the Ns-M relation for a given cosmology and output the corresponding error caming from the Emulator.
Here we use `hod_emu.get_emulator_m200c()` to obtain an instance of the Emulator class trianed on Delta_200c, and the function `emu.predict_A_beta_sigma(input)` to retrieve A,\beta and \sigma..
Note that input can be evaluated on a number `N` of data points (in this example only one) and is a N x 5 numpy array and the result is  a N x 3 numpy array. 

To obtain Emulator error one should call `emu.predict_A_beta_sigma(input, emulator_std=True)`. The function will then return a tuple with two elements: (1) thre first element is the same as before (i.e. a N x 3 numpy array with A, \beta and \sigma) and (2) another N x 3 numpy array with the corresponding emulator standard deviations.


```python
Om0 = 0.3
Ob0 = 0.04
s8 = 0.8
h0 = 0.7
z = 0.8

input = [[Om0, Ob0, s8, h0, 1./(1.+z)]] #the input must be a 2d array because you can feed an array of data points

emu = hod_emu.get_emulator_m200c() # use get_emulator_mvir to obtain the emulator within Delta_vir

A, beta, sigma  =  emu.predict_A_beta_sigma(input).T

masses = np.logspace(14.5,15.5,20)

Ns = A*(masses/5e14)**beta


plt.plot(masses,Ns)
plt.fill_between(masses, Ns*(1.-sigma), Ns*(1.+sigma),alpha=0.2)
plt.xlabel(r'$M_{\rm{halo}}$')
plt.ylabel(r'$N_s$')
plt.title(r'$M_\bigstar>2\cdot10^{11}M_\odot \ \ \ \tt{ and }  \ \ \ \ \  r<R_{\tt{200c}}$')
plt.xscale('log')
plt.yscale('log')

params_tuple, stds_tuple  =  emu.predict_A_beta_sigma(input, emulator_std=True)

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
    mask_negative = vals<0.
    if(np.any(vals[mask_negative])):
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
