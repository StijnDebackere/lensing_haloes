"""This module contains routines to analyze a given cluster sample,
i.e. the cluster likelihood, finding the MAP of the cluster sample,
...
"""
from datetime import datetime
import traceback
from multiprocessing import Process, Manager
from pathlib import Path
import os
import time

import asdf
import emcee
from george import kernels
import h5py
import numpy as np
from numpy.random import default_rng
from pyccl.halos.hmfunc import MassFuncTinker08
import scipy.optimize as opt
from scipy.special import factorial
from tqdm import tqdm

import lensing_haloes.cosmo.generate_mock_cluster_sample as mock_sample
from lensing_haloes.cosmo.cosmo import cosmology
import lensing_haloes.halo.abundance as abundance
from lensing_haloes.util.tools import chunks, within_bounds, datetime_in_range


# factor needed in likelihood calculation
LN_SQRT_2PI = np.log(np.sqrt(2 * np.pi))


def log_prior(theta, bounds):
    if within_bounds(theta, bounds):
        return 0.0
    return -np.inf


def log_prob(theta, **kwargs):
    bounds = kwargs.pop("bounds")
    lnlike = kwargs.pop("lnlike")
    lp = log_prior(theta, bounds)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, **kwargs)


def lnlike_poisson_mizi(
    theta,
    cosmo_fixed,
    prms_varied,
    prms_fixed,
    m200m_sample,
    z_sample,
    z_min,
    z_max,
    m200m_min,
    A_survey=2500,
    MassFunc=MassFuncTinker08,
):
    """
    Poisson likelihood calculation

    Parameters
    ----------
    theta : list
        sampling parameters
    cosmo_fixed : list
        fixed cosmological parameters
    prms_varied : list matching theta
        cosmological parameters that are varied and possible mass uncertainty
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
            - sigma_log10_mobs
    prms_fixed : list matching cosmo_fixed
        cosmological parameters that are kept fixed
    m200m_sample : array [M_sun / h]
        extrapolated observed halo masses
    z_sample : array
        redshifts of observed haloes
    z_min : float
        minimum redshift of sample
    z_max : float
        maximum redshift of sample
    m200m_min : float [M_sun / h]
        minimum halo mass in sample
    A_survey : float [deg^2]
        area of survey in square degrees
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use

    Returns
    -------
    lnlike : float
        likelihood of the given parameters
    """
    prms = {
        **{prm: val for prm, val in zip(prms_varied, theta)},
        **{prm: val for prm, val in zip(prms_fixed, cosmo_fixed)},
    }
    omega_m = prms["omega_m"]
    sigma_8 = prms["sigma_8"]
    w0 = prms["w0"]
    omega_b = prms["omega_b"]
    h = prms["h"]
    n_s = prms["n_s"]
    # # sigma_log10_mobs can either be varied, or it is given as a kwarg
    # sigma_log10_mobs = prms.get('sigma_log10_mobs', sigma_log10_mobs)

    if z_max > z_sample.max():
        z_max = z_sample.max()
    if z_min < z_sample.min():
        z_min = z_sample.min()
    if m200m_min < m200m_sample.min():
        m200m_min = m200m_sample.min()

    # sigma_8 and n are passed as arguments
    # other parameters should be updated in the astropy FlatwCDM object
    # through cosmo_params
    cosmo = cosmology(
        omega_m=omega_m, sigma_8=sigma_8, w0=w0, omega_b=omega_b, h=h, n_s=n_s
    )

    # calculate E from Cash (1979)
    E = abundance.dNdlog10mdz_integral(
        z_min=z_min,
        z_max=z_max,
        log10_m200m_min=np.log10(m200m_min),
        cosmo=cosmo,
        A_survey=A_survey,
        MassFunc=MassFunc,
    )

    selection = (z_sample <= z_max) & (m200m_sample >= m200m_min) & (z_sample >= z_min)
    # now calculate the term for each observed cluster
    dNdlog10mdz_i = abundance.dNdlog10mdz_mizi(
        z=z_sample[selection],
        log10_m200m=np.log10(m200m_sample[selection]),
        cosmo=cosmo,
        A_survey=A_survey,
        MassFunc=MassFunc,
    )
    sum_obs = np.sum(np.log(dNdlog10mdz_i))

    # need to return a float since otherwise log_prob is assumed to have blobs
    return float(2 * (sum_obs - E))


def sample_poisson_likelihood(
    fnames,
    method,
    z_min,
    z_max,
    m200m_min,
    theta_init,
    prms_varied,
    prms_fixed,
    bounds,
    mcmc_name=None,
    nwalkers=32,
    nsamples=5000,
    discard=100,
    out_q=None,
    pool=None,
):
    """Sample the likelihood

    Parameters
    ----------
    fnames : list
        list of filenames
    method : str
        mass fitting method
    lnlike : str
        likelihood to use: gaussian or mixed
    z_min : float
        minimum redshift in the sample
    z_max : float
        maximum redshift in the sample
    m200m_min : float
        minimum halo mass in the sample
    theta_init : list of lists
        values of initial cosmology guess for each fname in order
        [omega_m, sigma_8, w0, omega_b, h, n_s]
    prms_varied : list matching theta
        cosmological parameters that are varied and possible mass uncertainty
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
            - sigma_log10_mobs
    prms_fixed : list
        cosmological parameters that are kept fixed
    bounds : (ndim, 2) array-like
        array containing lower and upper bounds for each dimension
    mcmc_name : str or None
        group to save MCMC samples to
    nwalkers : int
        number of walkers to use in MCMC
    nsamples : int
        number of samples for each walker
    discard : int
        burn-in of the chains
    out_q : Queue() instance or None
        queue to put results in
        [Default: None]
    pool : Pool object
        optional pool to use for EnsembleSampler

    Returns
    -------
    - (samples, log_probs) to out_q if not None
    """
    bounds = np.atleast_2d(bounds)
    ndim = bounds.shape[0]
    maps = np.empty((0, ndim), dtype=float)

    for fname in fnames:
        with asdf.open(fname, copy_arrays=True) as af:
            # load possibly referenced values
            af.resolve_references()
            m200m_sample = af[method]["m200m_sample"][:]
            selection = af[method]["selection"][:]
            z_sample = af["z_sample"][selection]
            A_survey = af["A_survey"]

            selection = (
                (z_sample > z_min) & (z_sample < z_max) & (m200m_sample > m200m_min)
            )
            kwargs = {
                "m200m_min": m200m_min,
                "z_min": z_min,
                "z_max": z_max,
                "m200m_sample": m200m_sample[selection],
                "z_sample": z_sample[selection],
                "A_survey": A_survey,
                "cosmo_fixed": [af[prm] for prm in prms_fixed],
                "prms_varied": prms_varied,
                "prms_fixed": prms_fixed,
                "bounds": bounds,
            }
        if mcmc_name is None:
            name = (
                f'{method}/{np.round(np.log10(kwargs["m200m_min"]), 2)}/'
                f"/poisson/mcmc/{datetime.now().strftime('%Y%m%d_%H:%M')}"
            )
            pos = theta_init + 1e-3 * np.random.randn(nwalkers, ndim)
        else:
            name = (
                f'{method}/{np.round(np.log10(kwargs["m200m_min"]), 2)}/'
                f"/poisson/mcmc/{mcmc_name}"
            )
            pos = None

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnlike_poisson_mizi,
            kwargs=kwargs,
            pool=pool,
            backend=emcee.backends.HDFBackend(
                filename=str(Path(fname).with_suffix(".chains.hdf5")),
                name=name,
            ),
        )
        t1 = time.time()
        sampler.run_mcmc(pos, nsamples, progress=True)
        t2 = time.time()

        samples = sampler.get_chain()[discard:].reshape(-1, ndim)
        log_probs = sampler.get_log_prob()[discard:].reshape(-1)

        print(f"{os.getpid()} (took {t2 - t1:.2f}s)")

    if out_q is not None:
        out_q.put([os.getpid(), [samples, log_probs]])

    else:
        return (samples, log_probs)


def fit_maps_poisson(
    theta_init, kwargs, bounds, fnames=None, method="true", out_q=None
):
    """Fit the maximum a posteriori probability for the halo samples
    saved in fnames.

    Parameters
    ----------
    theta_init : list of arrays
        initial guesses (randomize these to avoid biases)
    kwargs : list of dicts
        keyword arguments to pass to lnlike function
    bounds : (ndim, 2) array-like
        array containing lower and upper bounds for each dimension
    fnames : list
        filename corresponding to each set of kwargs
    method : str
        mass fitting method
    out_q : Queue() instance or None
        queue to put results in
        [Default: None]

    Returns
    -------
    - to out_q if not None
    - otherwise, (n, ndim) array with MAP for each fname
    """
    bounds = np.atleast_2d(bounds)
    ndim = bounds.shape[0]
    maps = np.empty((0, ndim), dtype=float)

    for idx, kws in enumerate(kwargs):

        def nll(theta, kwargs):
            return -lnlike_poisson_mizi(theta=theta, **kwargs)

        t1 = time.time()
        res = opt.minimize(
            nll, theta_init[idx], args=(kws,), bounds=bounds, method="L-BFGS-B"
        )
        t2 = time.time()

        print(
            f"{os.getpid()} (took {t2 - t1:.2f}s): succes = {res.success} x = {res.x} nfev = {res.nfev}"
        )
        maps = np.concatenate([maps, res.x.reshape(-1, ndim)], axis=0)
        if fnames is not None:
            # append results to asdf file
            with asdf.open(fnames[idx], mode="rw") as af:
                if (
                    np.round(np.log10(kws["m200m_min"]), 2)
                    not in af.tree[method].keys()
                ):
                    af.tree[method][np.round(np.log10(kws["m200m_min"]), 2)] = {}

                af.tree[method][np.round(np.log10(kws["m200m_min"]), 2)][
                    "res_poisson"
                ] = {
                    "theta_init": theta_init[idx],
                    "fun": res.fun,
                    "jac": res.jac,
                    "message": res.message,
                    "nfev": res.nfev,
                    "nit": res.nit,
                    "njev": res.njev,
                    "status": res.status,
                    "success": res.success,
                    "x": res.x,
                    "m200m_min": kws["m200m_min"],
                    "z_min": kws["z_min"],
                    "z_max": kws["z_max"],
                    "A_survey": kws["A_survey"],
                    "cosmo_fixed": kws["cosmo_fixed"],
                    "prms_varied": kws["prms_varied"],
                    "prms_fixed": kws["prms_fixed"],
                    "bounds": bounds,
                }
                af.update()

    if out_q is not None:
        out_q.put([os.getpid(), maps])

    else:
        return maps


def fit_maps_poisson_mp(
    fnames,
    method,
    z_min,
    z_max,
    m200m_min,
    bounds,
    theta_init=[
        0.315,
        0.811,
        -1.0,
    ],
    prms_varied=["omega_m", "sigma_8", "w0"],
    prms_fixed=["omega_b", "h", "n_s"],
    rng=default_rng(0),
    n_cpus=None,
):
    """Fit the maximum a posteriori probability for the halo samples
    saved in fnames.

    Parameters
    ----------
    fnames : list
        list of filenames
    method : str
        mass fitting method
    z_min : float
        minimum redshift in the sample
    z_max : float
        maximum redshift in the sample
    m200m_min : float
        minimum halo mass in the sample
    bounds : (ndim, 2) array-like
        array containing lower and upper bounds for each dimension
    theta_init : list
        values of initial cosmology guess in order
        [sigma_8, omega_m, w0, omega_b, h, n_s]
    prms_varied : list matching theta
        cosmological parameters that are varied and possible mass uncertainty
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
            - sigma_log10_mobs
    prms_fixed : list
        cosmological parameters that are kept fixed
    rng : np.random.Generator instance
        Generator for random numbers
    n_cpus : int
        number of cpus to use
    Returns
    -------
    (n, ndim) array with MAP for each fname
    """
    method_options = ["WL", "WL_c", "true"]
    if method not in method_options:
        raise ValueError(f"{method} not in {method_options}")

    if bounds.shape[0] != len(theta_init):
        raise ValueError(f"bounds and theta_init need to have same dimension.")

    theta_init = np.atleast_1d(theta_init)
    # add random error to theta_init to avoid exact result
    theta_init = theta_init[None, :] + 1e-3 * rng.normal(
        size=(len(fnames), len(theta_init))
    )

    # set up process management
    manager = Manager()
    out_q = manager.Queue()
    procs = []

    kwargs_lst = []
    for fname in tqdm(fnames, desc="Loading kwargs", position=0):
        with asdf.open(fname, copy_arrays=True) as af:
            # load possibly referenced values
            af.resolve_references()
            m200m_sample = af[method]["m200m_sample"][:]
            selection = af[method]["selection"][:]
            z_sample = af["z_sample"][selection]
            A_survey = af["A_survey"]

            selection = (
                (z_sample > z_min) & (z_sample < z_max) & (m200m_sample > m200m_min)
            )
            kwargs = {
                "m200m_min": m200m_min,
                "z_min": z_min,
                "z_max": z_max,
                "m200m_sample": m200m_sample[selection],
                "z_sample": z_sample[selection],
                "A_survey": A_survey,
                "prms_varied": prms_varied,
                "prms_fixed": prms_fixed,
                "cosmo_fixed": [af[prm] for prm in prms_fixed],
            }
            kwargs_lst.append(kwargs)

    kwargs_split = list(chunks(kwargs_lst, n_cpus))
    fnames_split = list(chunks(fnames, n_cpus))
    theta_split = list(chunks(theta_init, n_cpus))

    t1 = time.time()
    for kwargs, fns, theta in zip(kwargs_split, fnames_split, theta_split):
        process = Process(
            target=fit_maps_poisson,
            args=(
                # initial guess which has been randomized
                theta,
                kwargs,
                bounds,
                fns,
                method,
                out_q,
            ),
        )

        procs.append(process)
        process.start()

    results = []
    for _ in range(n_cpus):
        results.append(out_q.get())

    # make sure processes are ended nicely
    for proc in procs:
        proc.join()

    # need to sort results by pid
    results.sort()
    maps = np.concatenate([item[1] for item in results], axis=0)
    t2 = time.time()
    print(f"Poisson likelihood took {(t2 - t1)/3600:.2f}h")

    return maps


def lnlike_gaussian_poisson_mizi(
    theta,
    cosmo_fixed,
    prms_varied,
    prms_fixed,
    Nobs_mizi,
    log10_m200m_bin_edges,
    z_bin_edges,
    A_survey=2500,
    MassFunc=MassFuncTinker08,
    pool=None,
    sigma_log10_mobs=None,
    sigma_log10_mobs_dist=None,
    **sigma_log10_mobs_dist_kwargs,
):
    """
    Mixed likelihood with Gaussian expectation value

    Parameters
    ----------
    theta : list
        sampling parameters
    cosmo_fixed : list
        fixed cosmological parameters
    prms_varied : list matching theta
        cosmological parameters that are varied and possible mass uncertainty
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
            - sigma_log10_mobs
    prms_fixed : list matching cosmo_fixed
        cosmological parameters that are kept fixed
    Nobs_mizi : (n_z, n_m) array
        number of clusters in each (z, m) bin
    log10_m200m_bin_edges : (n_z + 1) array [M_sun / h]
        bin edges in m200m
    z_bin_edges : (n_z + 1) array
        bin edges in z
    A_survey : float [deg^2]
        area of survey in square degrees
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use
    pool : multiprocessing pool or None
    sigma_log10_mobs : array-like, float or None
        uncertainty range on the mass
    sigma_log10_mobs_dist : callable
        scipy.stats distribution for sigma_log10_mobs
        [Default : None]

    Returns
    -------
    lnlike : float
        likelihood of the given parameters
    """
    # omega_m, sigma_8, w0, sigma_log10_mobs, omega_b, h, n_s = [*theta, *cosmo_fixed]
    prms = {
        **{prm: val for prm, val in zip(prms_varied, theta)},
        **{prm: val for prm, val in zip(prms_fixed, cosmo_fixed)},
    }
    omega_m = prms["omega_m"]
    sigma_8 = prms["sigma_8"]
    w0 = prms["w0"]
    omega_b = prms["omega_b"]
    h = prms["h"]
    n_s = prms["n_s"]
    # sigma_log10_mobs can either be varied, or it is given as a kwarg
    sigma_log10_mobs = prms.get("sigma_log10_mobs", sigma_log10_mobs)

    # sigma_8 and n are passed as arguments
    # other parameters should be updated in the astropy FlatwCDM object
    # through cosmo_params
    cosmo = cosmology(
        omega_m=omega_m, sigma_8=sigma_8, w0=w0, omega_b=omega_b, h=h, n_s=n_s
    )

    N_mizi = abundance.N_in_bins(
        z_bin_edges=z_bin_edges,
        m200m_bin_edges=10 ** log10_m200m_bin_edges,
        n_z=50,
        n_m=100,
        cosmo=cosmo,
        A_survey=A_survey,
        MassFunc=MassFunc,
        pool=pool,
        sigma_log10_mobs=sigma_log10_mobs,
        sigma_log10_mobs_dist=sigma_log10_mobs_dist,
        **sigma_log10_mobs_dist_kwargs,
    )
    lnlike_mixed_mizi = np.where(
        Nobs_mizi > 10,
        # Gaussian likelihood if N in bin is > 10
        (
            -((Nobs_mizi - N_mizi) ** 2) / (2 * N_mizi)
            - 0.5 * np.log(N_mizi)
            - LN_SQRT_2PI
        ),
        (Nobs_mizi * np.log(N_mizi) - N_mizi - np.log(factorial(Nobs_mizi))),
    )

    # need to return a float since otherwise log_prob is assumed to have blobs
    return float(np.sum(lnlike_mixed_mizi))


def lnlike_gaussian_mizi(
    theta,
    cosmo_fixed,
    prms_varied,
    prms_fixed,
    Nobs_mizi,
    log10_m200m_bin_edges,
    z_bin_edges,
    A_survey=2500,
    MassFunc=MassFuncTinker08,
    pool=None,
    sigma_log10_mobs=None,
    sigma_log10_mobs_dist=None,
    **sigma_log10_mobs_dist_kwargs,
):
    """
    Gaussian likelihood calculation

    Parameters
    ----------
    theta : list
        sampling parameters
    cosmo_fixed : list
        fixed cosmological parameters
    prms_varied : list matching theta
        cosmological parameters that are varied and possible mass uncertainty
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
            - sigma_log10_mobs
    prms_fixed : list matching cosmo_fixed
        cosmological parameters that are kept fixed
    Nobs_mizi : (n_z, n_m) array
        number of clusters in each (z, m) bin
    log10_m200m_bin_edges : (n_z + 1) array [M_sun / h]
        bin edges in m200m
    z_bin_edges : (n_z + 1) array
        bin edges in z
    cosmo_fixed : list
        fixed cosmological parameters to be passed to cosmo
    A_survey : float [deg^2]
        area of survey in square degrees
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use
    pool : multiprocessing pool or None
    sigma_log10_mobs : array-like or float
        uncertainty range on the mass
    sigma_log10_mobs_dist : callable
        scipy.stats distribution for sigma_log10_mobs
        [Default : None]

    Returns
    -------
    lnlike : float
        likelihood of the given parameters
    """
    # omega_m, sigma_8, w0, omega_b, h, n_s = [*theta, *cosmo_fixed]
    prms = {
        **{prm: val for prm, val in zip(prms_varied, theta)},
        **{prm: val for prm, val in zip(prms_fixed, cosmo_fixed)},
    }
    omega_m = prms["omega_m"]
    sigma_8 = prms["sigma_8"]
    w0 = prms["w0"]
    omega_b = prms["omega_b"]
    h = prms["h"]
    n_s = prms["n_s"]
    # sigma_log10_mobs can either be varied, or it is given as a kwarg
    sigma_log10_mobs = prms.get("sigma_log10_mobs", sigma_log10_mobs)

    # sigma_8 and n are passed as arguments
    # other parameters should be updated in the astropy FlatwCDM object
    # through cosmo_params
    cosmo = cosmology(
        omega_m=omega_m, sigma_8=sigma_8, w0=w0, omega_b=omega_b, h=h, n_s=n_s
    )

    N_mizi = abundance.N_in_bins(
        z_bin_edges=z_bin_edges,
        m200m_bin_edges=10 ** log10_m200m_bin_edges,
        n_z=50,
        n_m=100,
        cosmo=cosmo,
        A_survey=A_survey,
        MassFunc=MassFunc,
        pool=pool,
        sigma_log10_mobs=sigma_log10_mobs,
        sigma_log10_mobs_dist=sigma_log10_mobs_dist,
        **sigma_log10_mobs_dist_kwargs,
    )

    # need to return a float since otherwise log_prob is assumed to have blobs
    return -float(
        np.sum((Nobs_mizi - N_mizi) ** 2 / (2 * N_mizi) + 0.5 * np.log(N_mizi))
    )


def sample_gaussian_likelihood(
    fnames,
    method,
    lnlike,
    z_min,
    z_max,
    m200m_min,
    theta_init,
    prms_varied,
    prms_fixed,
    bounds,
    z_bins,
    log10_m200m_bins,
    mcmc_name=None,
    nwalkers=32,
    nsamples=5000,
    discard=100,
    out_q=None,
    pool=None,
    sigma_log10_mobs=None,
    sigma_log10_mobs_dist=None,
    **sigma_log10_mobs_dist_kwargs,
):
    """Sample the likelihood

    Parameters
    ----------
    fnames : list
        list of filenames
    method : str
        mass fitting method
    lnlike : str
        likelihood to use: gaussian or mixed
    z_min : float
        minimum redshift in the sample
    z_max : float
        maximum redshift in the sample
    m200m_min : float
        minimum halo mass in the sample
    theta_init : list
        sampling parameters
    prms_varied : list matching theta
        cosmological parameters that are varied and possible mass uncertainty
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
            - sigma_log10_mobs
    prms_fixed : list
        cosmological parameters that are kept fixed
    bounds : (ndim, 2) array-like
        array containing lower and upper bounds for each dimension
    z_bins : int
        number of bins for z
    log10_m200m_bins : int
        number bins for log10_m200m
    sigma_log10_mobs : array-like or float
        uncertainty range on the mass
    sigma_log10_mobs_dist : callable
        scipy.stats distribution for sigma_log10_mobs
        [Default : None]
    mcmc_name : str or None
        group to save MCMC samples to
    nwalkers : int
        number of walkers to use in MCMC
    nsamples : int
        number of samples for each walker
    discard : int
        burn-in of the chains
    out_q : Queue() instance or None
        queue to put results in
        [Default: None]
    pool : Pool object
        optional pool to use for EnsembleSampler
    sigma_log10_mobs : array-like or float
        uncertainty range on the mass
    sigma_log10_mobs_dist : callable
        scipy.stats distribution for sigma_log10_mobs
        [Default : None]

    Returns
    -------
    - (samples, log_probs) to out_q if not None
    """
    bounds = np.atleast_2d(bounds)
    ndim = bounds.shape[0]
    maps = np.empty((0, ndim), dtype=float)

    lnlike_options = {
        "gaussian": lnlike_gaussian_mizi,
        "mixed": lnlike_gaussian_poisson_mizi,
    }
    res_options = {"gaussian": "gaussian", "mixed": "gaussian_poisson"}

    for idx, fname in enumerate(fnames):
        with asdf.open(fname, copy_arrays=True) as af:
            # load possibly referenced values
            af.resolve_references()
            m200m_sample = af[method]["m200m_sample"][:]
            selection = af[method]["selection"][:]
            z_sample = af["z_sample"][selection]
            A_survey = af["A_survey"]

            if z_min < z_sample.min():
                z_min_sample = z_sample.min()
            else:
                z_min_sample = z_min
            if z_max > z_sample.max():
                z_max_sample = z_sample.max()
            else:
                z_max_sample = z_max
            if m200m_min < m200m_sample.min():
                m200m_min_sample = m200m_sample.min()
            else:
                m200m_min_sample = m200m_min

            selection = (
                (z_sample > z_min_sample)
                & (z_sample < z_max_sample)
                & (m200m_sample > m200m_min_sample)
            )
            Nobs_mizi, z_edges, log10_m200m_edges = np.histogram2d(
                x=z_sample[selection],
                y=np.log10(m200m_sample[selection]),
                bins=[z_bins, log10_m200m_bins],
            )

            kwargs = {
                "lnlike": lnlike_options[lnlike],
                "cosmo_fixed": [af[prm] for prm in prms_fixed],
                "prms_varied": prms_varied,
                "prms_fixed": prms_fixed,
                "bounds": bounds,
                "Nobs_mizi": Nobs_mizi,
                "z_bin_edges": z_edges,
                "log10_m200m_bin_edges": log10_m200m_edges,
                "A_survey": A_survey,
                "sigma_log10_mobs": sigma_log10_mobs,
                "sigma_log10_mobs_dist": sigma_log10_mobs_dist,
                **sigma_log10_mobs_dist_kwargs,
            }
            # # added for loading previous result if stopped by error
            # if np.round(np.log10(kwargs['m200m_min']), 2) in af[method].keys():
            #     fname_map = af[method][np.round(np.log10(kwargs['m200m_min']), 2)]['res_gaussian']['x']
        if mcmc_name is None:
            name = (
                f"{method}/{np.round(np.log10(m200m_min_sample), 2)}/"
                f"/{res_options[lnlike]}/mcmc/{datetime.now().strftime('%Y%m%d_%H:%M')}"
            )
            pos = theta_init + 1e-3 * np.random.randn(nwalkers, ndim)
        else:
            name = (
                f"{method}/{np.round(np.log10(m200m_min_sample), 2)}/"
                f"/{res_options[lnlike]}/mcmc/{mcmc_name}"
            )
            with h5py.File(str(Path(fname).with_suffix(".chains.hdf5")), "r") as f:
                items = []
                f.visit(items.append)
                if name in items:
                    pos = None
                else:
                    pos = theta_init + 1e-3 * np.random.randn(nwalkers, ndim)

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            log_prob,
            kwargs=kwargs,
            pool=pool,
            backend=emcee.backends.HDFBackend(
                filename=str(Path(fname).with_suffix(".chains.hdf5")),
                name=name,
            ),
        )
        t1 = time.time()
        sampler.run_mcmc(pos, nsamples, progress=True)
        t2 = time.time()

        samples = sampler.get_chain(flat=True, discard=discard).reshape(-1, ndim)
        log_probs = sampler.get_log_prob(flat=True, discard=discard).reshape(-1)

        print(f"{os.getpid()} (took {t2 - t1:.2f}s)")

    if out_q is not None:
        out_q.put([os.getpid(), [samples, log_probs]])

    else:
        return (samples, log_probs)


def fit_maps_gaussian(
    fnames,
    methods,
    lnlike,
    z_min,
    z_max,
    m200m_min,
    theta_init,
    prms_varied,
    prms_fixed,
    bounds,
    z_bins,
    log10_m200m_bins,
    maps_name=None,
    out_q=None,
    sigma_log10_mobs=None,
    sigma_log10_mobs_dist=None,
    **sigma_log10_mobs_dist_kwargs,
):
    """Fit the maximum a posteriori probability for the halo samples
    saved in fnames.

    Parameters
    ----------
    fnames : list
        list of filenames
    methods : list
        mass fitting method for each fname
    lnlike : str
        likelihood to use: gaussian or mixed
    z_min : float
        minimum redshift in the sample
    z_max : float
        maximum redshift in the sample
    m200m_min : float
        minimum halo mass in the sample
    theta_init : list
        sampling parameters
    prms_varied : list matching theta
        cosmological parameters that are varied and possible mass uncertainty
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
            - sigma_log10_mobs
    prms_fixed : list
        cosmological parameters that are kept fixed
    bounds : (ndim, 2) array-like
        array containing lower and upper bounds for each dimension
    z_bins : int
        number of bins for z
    log10_m200m_bins : int
        number bins for log10_m200m
    maps_name : str or None
        name to save maps under in asdf file
    out_q : Queue() instance or None
        queue to put results in
        [Default: None]
    sigma_log10_mobs : array-like or float
        uncertainty range on the mass
    sigma_log10_mobs_dist : callable
        scipy.stats distribution for sigma_log10_mobs
        [Default : None]

    Returns
    -------
    - to out_q if not None
    - otherwise, (n, ndim) array with MAP for each fname
    """
    bounds = np.atleast_2d(bounds)
    ndim = bounds.shape[0]
    maps = np.empty((0, ndim), dtype=float)

    lnlike_options = {
        "gaussian": lnlike_gaussian_mizi,
        "mixed": lnlike_gaussian_poisson_mizi,
    }
    res_options = {"gaussian": "res_gaussian", "mixed": "res_gaussian_poisson"}

    for idx, (fname, method) in enumerate(zip(fnames, methods)):
        with asdf.open(fname, copy_arrays=True) as af:
            # load possibly referenced values
            af.resolve_references()
            m200m_sample = af[method]["m200m_sample"][:]
            selection = af[method]["selection"][:]
            z_sample = af["z_sample"][selection]
            A_survey = af["A_survey"]

            if z_min < z_sample.min():
                z_min_sample = z_sample.min()
            else:
                z_min_sample = z_min
            if z_max > z_sample.max():
                z_max_sample = z_sample.max()
            else:
                z_max_sample = z_max
            if m200m_min < m200m_sample.min():
                m200m_min_sample = m200m_sample.min()
            else:
                m200m_min_sample = m200m_min

            selection = (
                (z_sample > z_min_sample)
                & (z_sample < z_max_sample)
                & (m200m_sample > m200m_min_sample)
            )
            Nobs_mizi, z_edges, log10_m200m_edges = np.histogram2d(
                x=z_sample[selection],
                y=np.log10(m200m_sample[selection]),
                bins=[z_bins, log10_m200m_bins],
            )

            kwargs = {
                "Nobs_mizi": Nobs_mizi,
                "log10_m200m_bin_edges": log10_m200m_edges,
                "z_bin_edges": z_edges,
                "prms_varied": prms_varied,
                "prms_fixed": prms_fixed,
                "cosmo_fixed": [af[prm] for prm in prms_fixed],
                "A_survey": A_survey,
                "sigma_log10_mobs": sigma_log10_mobs,
                "sigma_log10_mobs_dist": sigma_log10_mobs_dist,
                **sigma_log10_mobs_dist_kwargs,
            }
            # # added for loading previous result if stopped by error
            # if np.round(np.log10(kwargs['m200m_min']), 2) in af[method].keys():
            #     fname_map = af[method][np.round(np.log10(kwargs['m200m_min']), 2)]['res_gaussian']['x']

        def nll(theta, kwargs):
            return -lnlike_options[lnlike](theta=theta, **kwargs)

        t1 = time.time()
        res = opt.minimize(
            nll, theta_init[idx], args=(kwargs,), bounds=bounds, method="L-BFGS-B"
        )
        t2 = time.time()

        print(
            f"{os.getpid()} (took {t2 - t1:.2f}s): succes = {res.success} x = {res.x} nfev = {res.nfev}"
        )
        maps = np.concatenate([maps, res.x.reshape(-1, ndim)], axis=0)

        # append results to asdf file
        with asdf.open(fname, mode="rw") as af:
            m_key = np.round(np.log10(m200m_min_sample), 2)
            # check whether this is the first time this mass cut has been fit
            if m_key not in af[method].keys():
                af.tree[method][m_key] = {}
                af.tree[method][m_key][res_options[lnlike]] = {}

            to_save = {
                "theta_init": theta_init[idx],
                "fun": res.fun,
                "jac": res.jac,
                "message": res.message,
                "nfev": res.nfev,
                "nit": res.nit,
                "njev": res.njev,
                "status": res.status,
                "success": res.success,
                "x": res.x,
                # setup information
                "prms_varied": prms_varied,
                "m200m_min": m200m_min_sample,
                "z_min": z_min_sample,
                "z_max": z_max_sample,
                "A_survey": A_survey,
                "prms_fixed": prms_fixed,
                "cosmo_fixed": kwargs["cosmo_fixed"],
                "bounds": bounds,
                "z_bin_edges": z_edges,
                "log10_m200m_bin_edges": log10_m200m_edges,
                # "sigma_log10_mobs": sigma_log10_mobs,
                # "sigma_log10_mobs_dist_kwargs": sigma_log10_mobs_dist_kwargs,
            }

            if maps_name is None:
                if sigma_log10_mobs_dist is not None:
                    maps_name = f"sigma_log10_mobs_dist_{getattr(sigma_log10_mobs_dist, 'name')}"
                    kwargs_name = "_".join(
                        [
                            f"{key}_{val:.3f}"
                            for key, val in sigma_log10_mobs_dist_kwargs.items()
                        ]
                    )
                    # try to save to unique location matching method, mass, lnlike and sigma_log10_mobs
                    af.tree[method][m_key][res_options[lnlike]][maps_name] = {}

                    # no kwargs
                    if kwargs_name == "":
                        af.tree[method][m_key][res_options[lnlike]][maps_name] = {
                            **to_save
                        }
                    else:
                        af.tree[method][m_key][res_options[lnlike]][name][
                            kwargs_name
                        ] = {**to_save}
                else:
                    maps_name = datetime.now().strftime("%Y%m%d_%H:%M")

            else:
                af.tree[method][m_key][res_options[lnlike]][maps_name] = {**to_save}

            af.update()

    if out_q is not None:
        out_q.put([os.getpid(), maps])

    else:
        return maps


def fit_maps_gaussian_mp(
    fnames,
    methods,
    z_min,
    z_max,
    m200m_min,
    bounds,
    lnlike="mixed",
    theta_init=[
        0.315,
        0.811,
        -1.0,
        # sigma_log10_mobs included in theta
        np.log10(1.1),
    ],
    prms_varied=["omega_m", "sigma_8", "w0", "sigma_log10_mobs"],
    prms_fixed=["omega_b", "h", "n_s"],
    z_bins=8,
    log10_m200m_bins=40,
    maps_name=None,
    rng=default_rng(0),
    n_cpus=None,
    sigma_log10_mobs=None,
    sigma_log10_mobs_dist=None,
    **sigma_log10_mobs_dist_kwargs,
):
    """Fit the maximum a posteriori probability for the halo samples
    saved in fnames.

    Parameters
    ----------
    fnames : list
        list of filenames
    methods : list
        mass fitting method for each fname
    lnlike : str
        likelihood to use: 'gaussian' or 'mixed'
    z_min : float
        minimum redshift in the sample
    z_max : float
        maximum redshift in the sample
    m200m_min : float
        minimum halo mass in the sample
    bounds : (ndim, 2) array-like
        array containing lower and upper bounds for each dimension
    theta_init : list
        sampling parameters
    prms_varied : list matching theta
        cosmological parameters that are varied and possible mass uncertainty
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
            - sigma_log10_mobs
    prms_fixed : list
        cosmological parameters that are kept fixed
    z_bins : int
        number of bins for z
    log10_m200m_bins : int
        number of bins for log10_m200m
    maps_name : str or None
        name to save maps under in asdf file
    rng : np.random.Generator instance
        Generator for random numbers
    n_cpus : int
        number of cpus to use
    sigma_log10_mobs : array-like or float
        uncertainty range on the mass
        [Default : None]
    sigma_log10_mobs_dist : callable
        scipy.stats distribution for sigma_log10_mobs
        [Default : None]

    Returns
    -------
    (n, ndim) array with MAP for each fname
    """
    # method_options = [
    #     "WL", "WL_min", "WL_max", "WL_c", "WL_c_min", "WL_c_max", "true",
    # ]
    # if method not in method_options:
    #     raise ValueError(f"{method} not in {method_options}")

    if type(methods) == str:
        methods = [methods] * len(fnames)

    lnlike_options = ["gaussian", "mixed"]
    if lnlike not in lnlike_options:
        raise ValueError(f"lnlike should be one of {lnlike_options}")

    if bounds.shape[0] != len(theta_init):
        raise ValueError(f"bounds and theta_init need to have same dimension.")

    theta_init = np.atleast_1d(theta_init)
    # add random error to theta_init to avoid exact result
    theta_init = theta_init[None, :] + 1e-3 * rng.normal(
        size=(len(fnames), len(theta_init))
    )

    # set up process management
    manager = Manager()
    out_q = manager.Queue()
    procs = []

    fnames_split = list(chunks(fnames, n_cpus))
    methods_split = list(chunks(methods, n_cpus))
    theta_split = list(chunks(theta_init, n_cpus))
    # add random error to theta_init to avoid exact result

    t1 = time.time()
    for fns, meths, theta in zip(fnames_split, methods_split, theta_split):
        process = Process(
            target=fit_maps_gaussian,
            kwargs={
                "fnames": fns,
                "methods": meths,
                "lnlike": lnlike,
                "z_min": z_min,
                "z_max": z_max,
                "m200m_min": m200m_min,
                "theta_init": theta,
                "prms_varied": prms_varied,
                "prms_fixed": prms_fixed,
                "bounds": bounds,
                "z_bins": z_bins,
                "log10_m200m_bins": log10_m200m_bins,
                "maps_name": maps_name,
                "out_q": out_q,
                "sigma_log10_mobs": sigma_log10_mobs,
                "sigma_log10_mobs_dist": sigma_log10_mobs_dist,
                **sigma_log10_mobs_dist_kwargs,
            },
        )

        procs.append(process)
        process.start()

    results = []
    for _ in range(n_cpus):
        results.append(out_q.get())

    # make sure processes are ended nicely
    for proc in procs:
        proc.join()

    # need to sort results by pid
    results.sort()
    maps = np.concatenate([item[1] for item in results], axis=0)
    t2 = time.time()
    print(f"Gaussian likelihood took {(t2 - t1)/3600:.2f}h")

    return maps


def generate_maps_file(
    fnames=None,
    res=["res_gaussian_poisson"],
    methods=["WL", "WL_c", "true"],
    mcuts=[14.0, 14.25, 14.5],
    maps_names=None,
    date_range=None,
    fname_append="",
    cosmo_name=None,
):
    """Consolidate all the MAPs from fnames into a single file."""

    # maps are save under method/mcut/res/maps_name
    maps = {}
    # load general info
    with asdf.open(fnames[0], copy_arrays=True, lazy_load=True) as af:
        maps["A_survey"] = af["A_survey"]
        maps["log10_m_min"] = af["log10_m_min"]
        maps["log10_m_max"] = af["log10_m_max"]
        maps["z_min"] = af["z_min"]
        maps["z_max"] = af["z_max"]
        maps["h"] = af["h"]
        maps["n_s"] = af["n_s"]
        maps["w0"] = af["w0"]
        maps["omega_m"] = af["omega_m"]
        maps["omega_b"] = af["omega_b"]
        maps["sigma_8"] = af["sigma_8"]

        # only load methods that are in fnames
        methods_in_files = [m for m in methods if m in af.keys()]
        for method in methods_in_files:
            maps[method] = {}
            for mcut in mcuts:
                # only load mcuts that are in fnames
                if mcut not in af[method].keys():
                    continue
                maps[method][mcut] = {}

                for r in res:
                    if r not in af[method][mcut].keys():
                        continue

                    maps[method][mcut][r] = {}

                    if maps_names is not None:
                        for maps_name in maps_names:
                            try:
                                maps[method][mcut][r][maps_name] = {
                                    "z_min": af[method][mcut][r][maps_name]["z_min"],
                                    "z_max": af[method][mcut][r][maps_name]["z_max"],
                                    "m200m_min": af[method][mcut][r][maps_name]["m200m_min"],
                                    "cosmo_fixed": af[method][mcut][r][maps_name]["cosmo_fixed"],
                                    "maps": [],
                                    "fun": [],
                                    "success": [],
                                }
                            except KeyError:
                                continue

                    else:
                        maps[method][mcut][r] = {}

                        if date_range is not None:
                            # find dates
                            for key in af[method][mcut][r].keys():
                                try:
                                    date = datetime.strptime(key, "%Y%m%d_%H:%M")
                                    if datetime_in_range(date, *date_range):
                                        maps[method][mcut][r][date] = {
                                            "z_min": af[method][mcut][r][date]["z_min"],
                                            "z_max": af[method][mcut][r][date]["z_max"],
                                            "m200m_min": af[method][mcut][r][date]["m200m_min"],
                                            "cosmo_fixed": af[method][mcut][r][date]["cosmo_fixed"],
                                            "maps": [],
                                            "fun": [],
                                            "success": [],
                                        }
                                        continue
                                except ValueError:
                                    continue
                        else:
                            try:
                                maps[method][mcut][r] = {
                                    "z_min": af[method][mcut][r]["z_min"],
                                    "z_max": af[method][mcut][r]["z_max"],
                                    "m200m_min": af[method][mcut][r]["m200m_min"],
                                    "cosmo_fixed": af[method][mcut][r]["cosmo_fixed"],
                                    "maps": [],
                                    "fun": [],
                                    "success": [],
                                }
                            except KeyError:
                                continue

    # now go through the different mass cuts for each method
    for fname in tqdm(fnames, desc="Loading files"):
        with asdf.open(fname, copy_arrays=True, lazy_load=True) as af:
            for method in methods_in_files:
                for mcut in maps[method].keys():
                    for res in maps[method][mcut].keys():
                        if "maps" in maps[method][mcut][res].keys():
                            maps[method][mcut][res]["maps"].append(
                                af[method][mcut][res]["x"][:]
                            )
                            maps[method][mcut][res]["fun"].append(
                                af[method][mcut][res]["fun"]
                            )
                            maps[method][mcut][res]["success"].append(
                                af[method][mcut][res]["success"]
                            )

                        if maps_names is not None:
                            for maps_name in maps_names:
                                try:
                                    maps[method][mcut][res][maps_name]["maps"].append(
                                        af[method][mcut][res][maps_name]["x"][:]
                                    )
                                    maps[method][mcut][res][maps_name]["fun"].append(
                                        af[method][mcut][res][maps_name]["fun"]
                                    )
                                    maps[method][mcut][res][maps_name]["success"].append(
                                        af[method][mcut][res][maps_name]["success"]
                                    )
                                except KeyError:
                                    continue

                        if date_range is not None:
                            # find dates
                            for key in maps[method][mcut][res].keys():
                                try:
                                    date = datetime.strptime(key, "%Y%m%d_%H:%M")
                                    if datetime_in_range(date, *date_range):
                                        maps[method][mcut][res][date]["maps"].append(
                                            af[method][mcut][res][date]["x"][:]
                                        )
                                        maps[method][mcut][res][date]["fun"].append(
                                            af[method][mcut][res][date]["fun"]
                                        )
                                        maps[method][mcut][res][date]["success"].append(
                                            af[method][mcut][res][date]["success"]
                                        )
                                except:
                                    continue

    for method in methods_in_files:
        for mcut in maps[method].keys():
            for res in maps[method][mcut].keys():
                for key in maps[method][mcut][res].keys():
                    if key == "maps":
                        maps[method][mcut][res]["maps"] = np.atleast_2d(
                            maps[method][mcut][res]["maps"]
                        )
                        maps[method][mcut][res]["fun"] = np.atleast_1d(
                            maps[method][mcut][res]["fun"]
                        )
                        maps[method][mcut][res]["success"] = np.atleast_1d(
                            maps[method][mcut][res]["success"]
                        )

                    if key in maps_names:
                        maps[method][mcut][res][key]["maps"] = np.atleast_2d(
                            maps[method][mcut][res][key]["maps"]
                        )
                        maps[method][mcut][res][key]["fun"] = np.atleast_1d(
                            maps[method][mcut][res][key]["fun"]
                        )
                        maps[method][mcut][res][key]["success"] = np.atleast_1d(
                            maps[method][mcut][res][key]["success"]
                        )

                    try:
                        if datetime_in_range(key, *date_range):
                            maps[method][mcut][res][key]["maps"] = np.atleast_2d(
                                maps[method][mcut][res][key]["maps"]
                            )
                            maps[method][mcut][res][key]["fun"] = np.atleast_1d(
                                maps[method][mcut][res][key]["fun"]
                            )
                            maps[method][mcut][res][key]["success"] = np.atleast_1d(
                                maps[method][mcut][res][key]["success"]
                            )
                    except TypeError:
                        continue

    # maps_fname_base = mock_sample.gen_fname(
    #     A_survey=A_survey, z_min=z_min, z_max=z_max, m200m_min=m200m_min, cosmo=cosmo
    # )

    if cosmo_name is None:
        cosmo_name = ""
    fname_base = mock_sample.gen_fname(
        A_survey=maps['A_survey'], z_min=maps['z_min'], z_max=maps['z_max'],
        m200m_min=maps['log10_m_min'], cosmo=cosmo_name
    )
    fname = f'{fname_base}_{fname_append}.asdf'
    try:
        with asdf.AsdfFile(maps) as af:
            af.write_to(fname)
    except Exception as e:
        print(f"{fname} failed with Exception")
        traceback.print_exc()

    return maps
