"""This module contains routines to analyze a given cluster sample,
i.e. the cluster likelihood, finding the MAP of the cluster sample,
...
"""
from datetime import datetime
from multiprocessing import Process, Manager
from pathlib import Path
import os
import time

import asdf
import emcee
from george import kernels
import numpy as np
from numpy.random import default_rng
from pyccl.halos.hmfunc import MassFuncTinker08
import scipy.optimize as opt
from scipy.special import factorial
from tqdm import tqdm

import lensing_haloes.cosmo.generate_mock_cluster_sample as mock_sample
from lensing_haloes.cosmo.cosmo import cosmology
import lensing_haloes.halo.abundance as abundance
from lensing_haloes.util.tools import chunks

from pdb import set_trace


# factor needed in likelihood calculation
LN_SQRT_2PI = np.log(np.sqrt(2 * np.pi))


def lnlike_poisson_mizi(
    theta,
    m200m_sample,
    z_sample,
    z_min,
    z_max,
    m200m_min,
    cosmo_fixed,
    A_survey=2500,
    MassFunc=MassFuncTinker08,
):
    """
    Poisson likelihood calculation

    Parameters
    ----------
    theta : list
        sampling parameters
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
    cosmo_fixed : list
        fixed parameters
            - any parameters not in theta, [*theta, *theta_fixed] should have
              same order as sampling parameters
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
    omega_m, sigma_8, w0, omega_b, h, n_s = [*theta, *cosmo_fixed]

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
    bounds,
    cosmo_fixed,
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
    bounds : (ndim, 2) array-like
        array containing lower and upper bounds for each dimension
    cosmo_fixed : list of dict keys
        cosmological parameters that are kept fixed
        ['omega_m', 'sigma_8', 'w0', 'omega_b', 'h', 'n_s']
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
                "cosmo_fixed": [af[prm] for prm in cosmo_fixed],
            }

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnlike_poisson_mizi,
            kwargs=kwargs,
            pool=pool,
            backend=emcee.backends.HDFBackend(
                filename=str(Path(fname).with_suffix(".chains.hdf5")),
                name=(
                    f'{method}/{np.round(np.log10(kwargs["m200m_min"]), 2)}/'
                    f"/poisson/mcmc/{datetime.now().strftime('%Y%m%d_%H:%M')}"
                ),
            ),
        )
        pos = theta_init + 1e-3 * np.random.randn(nwalkers, ndim)
        t1 = time.time()
        sampler.run_mcmc(pos, nsamples, progress=True)
        t2 = time.time()

        samples = sampler.get_chain()[discard:].reshape(-1, ndim)
        log_probs = sampler.get_log_probs()[discard:].reshape(-1)

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
    cosmo_fixed=["omega_b", "h", "n_s"],
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
    cosmo_fixed : list of dict keys
        cosmological parameters that are kept fixed
        [sigma_8, 'omega_m', 'w0', 'omega_b', 'h', 'n_s']
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
                "cosmo_fixed": [af[prm] for prm in cosmo_fixed],
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
    Nobs_mizi,
    log10_m200m_bin_edges,
    z_bin_edges,
    cosmo_fixed,
    A_survey=2500,
    MassFunc=MassFuncTinker08,
    sigma_log10_mobs=None,
    pool=None,
    **kwargs,
):
    """
    Mixed likelihood with Gaussian expectation value

    Parameters
    ----------
    theta : list
        sampling parameters
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
    cosmo_fixed : list
        fixed parameters
            - any parameters not in theta, [*theta, *theta_fixed] should have
              same order as sampling parameters
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
    sigma_log10_mobs : float
        uncertainty on the mass
    pool : multiprocessing pool or None

    Returns
    -------
    lnlike : float
        likelihood of the given parameters
    """
    omega_m, sigma_8, w0, omega_b, h, n_s = [*theta, *cosmo_fixed]

    # sigma_8 and n are passed as arguments
    # other parameters should be updated in the astropy FlatwCDM object
    # through cosmo_params
    cosmo = cosmology(
        omega_m=omega_m, sigma_8=sigma_8, w0=w0, omega_b=omega_b, h=h, n_s=n_s
    )

    N_mizi = abundance.N_in_bins(
        z_bin_edges=z_bin_edges,
        m200m_bin_edges=10 ** log10_m200m_bin_edges,
        sigma_log10_mobs=sigma_log10_mobs,
        n_z=50,
        n_m=100,
        cosmo=cosmo,
        A_survey=A_survey,
        MassFunc=MassFunc,
        pool=pool,
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
    Nobs_mizi,
    log10_m200m_bin_edges,
    z_bin_edges,
    cosmo_fixed,
    A_survey=2500,
    MassFunc=MassFuncTinker08,
    sigma_log10_mobs=None,
    pool=None,
    **kwargs,
):
    """
    Gaussian likelihood calculation

    Parameters
    ----------
    theta : list
        sampling parameters
            - omega_m
            - sigma_8
            - w0
            - omega_b
            - h
            - n_s
    cosmo_fixed : list
        fixed parameters
            - any parameters not in theta, [*theta, *theta_fixed] should have
              same order as sampling parameters
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
    sigma_log10_mobs : float
        uncertainty on the mass
    pool : multiprocessing pool or None

    Returns
    -------
    lnlike : float
        likelihood of the given parameters
    """
    omega_m, sigma_8, w0, omega_b, h, n_s = [*theta, *cosmo_fixed]

    # sigma_8 and n are passed as arguments
    # other parameters should be updated in the astropy FlatwCDM object
    # through cosmo_params
    cosmo = cosmology(
        omega_m=omega_m, sigma_8=sigma_8, w0=w0, omega_b=omega_b, h=h, n_s=n_s
    )

    N_mizi = abundance.N_in_bins(
        z_bin_edges=z_bin_edges,
        m200m_bin_edges=10 ** log10_m200m_bin_edges,
        sigma_log10_mobs=sigma_log10_mobs,
        n_z=50,
        n_m=100,
        cosmo=cosmo,
        A_survey=A_survey,
        MassFunc=MassFunc,
        pool=pool,
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
    bounds,
    cosmo_fixed,
    z_bins,
    log10_m200m_bins,
    sigma_log10_mobs,
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
    bounds : (ndim, 2) array-like
        array containing lower and upper bounds for each dimension
    cosmo_fixed : list of dict keys
        cosmological parameters that are kept fixed
        ['omega_m', 'sigma_8', 'w0', 'omega_b', 'h', 'n_s']
    z_bins : int
        number of bins for z
    log10_m200m_bins : int
        number bins for log10_m200m
    sigma_log10_mobs : float
        uncertainty on the mass
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
                "Nobs_mizi": Nobs_mizi,
                "z_bin_edges": z_edges,
                "log10_m200m_bin_edges": log10_m200m_edges,
                "A_survey": A_survey,
                "theta_init": theta_init,
                "cosmo_fixed": [af[prm] for prm in cosmo_fixed],
                "m200m_min": m200m_min_sample,
                "z_min": z_min_sample,
                "z_max": z_max_sample,
                "sigma_log10_mobs": sigma_log10_mobs,
            }
            # # added for loading previous result if stopped by error
            # if np.round(np.log10(kwargs['m200m_min']), 2) in af[method].keys():
            #     fname_map = af[method][np.round(np.log10(kwargs['m200m_min']), 2)]['res_gaussian']['x']

        sampler = emcee.EnsembleSampler(
            nwalkers,
            ndim,
            lnlike_options[lnlike],
            kwargs=kwargs,
            pool=pool,
            backend=emcee.backends.HDFBackend(
                filename=str(Path(fname).with_suffix(".chains.hdf5")),
                name=(
                    f'{method}/{np.round(np.log10(kwargs["m200m_min"]), 2)}/'
                    f"/{res_options[lnlike]}/mcmc/{datetime.now().strftime('%Y%m%d_%H:%M')}"
                ),
            ),
        )
        pos = theta_init + 1e-3 * np.random.randn(nwalkers, ndim)
        t1 = time.time()
        sampler.run_mcmc(pos, nsamples, progress=True)
        t2 = time.time()

        samples = sampler.get_chain(flatten=True, discard=discard).reshape(-1, ndim)
        log_probs = sampler.get_log_probs(flatten=True, discard=discard).reshape(-1)

        print(f"{os.getpid()} (took {t2 - t1:.2f}s)")

    if out_q is not None:
        out_q.put([os.getpid(), [samples, log_probs]])

    else:
        return (samples, log_probs)


def fit_maps_gaussian(
    fnames,
    method,
    lnlike,
    z_min,
    z_max,
    m200m_min,
    theta_init,
    bounds,
    cosmo_fixed,
    z_bins,
    log10_m200m_bins,
    out_q=None,
):
    """Fit the maximum a posteriori probability for the halo samples
    saved in fnames.

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
    bounds : (ndim, 2) array-like
        array containing lower and upper bounds for each dimension
    cosmo_fixed : list of dict keys
        cosmological parameters that are kept fixed
        ['omega_m', 'sigma_8', 'w0', 'omega_b', 'h', 'n_s']
    z_bins : int
        number of bins for z
    log10_m200m_bins : int
        number bins for log10_m200m
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

    lnlike_options = {
        "gaussian": lnlike_gaussian_mizi,
        "mixed": lnlike_gaussian_poisson_mizi,
    }
    res_options = {"gaussian": "res_gaussian", "mixed": "res_gaussian_poisson"}

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
                "Nobs_mizi": Nobs_mizi,
                "z_bin_edges": z_edges,
                "log10_m200m_bin_edges": log10_m200m_edges,
                "A_survey": A_survey,
                "theta_init": theta_init,
                "cosmo_fixed": [af[prm] for prm in cosmo_fixed],
                "m200m_min": m200m_min_sample,
                "z_min": z_min_sample,
                "z_max": z_max_sample,
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
            # check whether this is the first time this mass cut has been fit
            if np.round(np.log10(kwargs["m200m_min"]), 2) not in af[method].keys():
                af.tree[method][np.round(np.log10(kwargs["m200m_min"]), 2)] = {}
            af.tree[method][np.round(np.log10(kwargs["m200m_min"]), 2)][
                res_options[lnlike]
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
                "m200m_min": kwargs["m200m_min"],
                "z_min": kwargs["z_min"],
                "z_max": kwargs["z_max"],
                "A_survey": kwargs["A_survey"],
                "cosmo_fixed": kwargs["cosmo_fixed"],
                "z_bin_edges": z_edges,
                "log10_m200m_bin_edges": log10_m200m_edges,
            }
            af.update()

    if out_q is not None:
        out_q.put([os.getpid(), maps])

    else:
        return maps


def fit_maps_gaussian_mp(
    fnames,
    method,
    z_min,
    z_max,
    m200m_min,
    bounds,
    lnlike="gaussian",
    theta_init=[
        0.315,
        0.811,
        -1.0,
    ],
    cosmo_fixed=["omega_b", "h", "n_s"],
    z_bins=8,
    log10_m200m_bins=20,
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
        values of initial cosmology guess in order
        [sigma_8, omega_m, w0, omega_b, h, n_s]
    cosmo_fixed : list of dict keys
        cosmological parameters that are kept fixed
        [sigma_8, 'omega_m', 'w0', 'omega_b', 'h', 'n_s']
    z_bins : int
        number of bins for z
    log10_m200m_bins : int
        number of bins for log10_m200m
    rng : np.random.Generator instance
        Generator for random numbers
    n_cpus : int
        number of cpus to use
    Returns
    -------
    (n, ndim) array with MAP for each fname
    """
    method_options = ["WL", "WL_min", "WL_max", "WL_c", "WL_c_min", "WL_c_max", "true"]
    if method not in method_options:
        raise ValueError(f"{method} not in {method_options}")

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
    theta_split = list(chunks(theta_init, n_cpus))
    # add random error to theta_init to avoid exact result

    t1 = time.time()
    for fns, theta in zip(fnames_split, theta_split):
        process = Process(
            target=fit_maps_gaussian,
            args=(
                fns,
                method,
                lnlike,
                z_min,
                z_max,
                m200m_min,
                theta,
                bounds,
                cosmo_fixed,
                z_bins,
                log10_m200m_bins,
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
    print(f"Gaussian likelihood took {(t2 - t1)/3600:.2f}h")

    return maps


def generate_maps_file(
    A_survey="15k",
    z_min="0p1",
    z_max="4",
    m200m_min="13p76",
    cosmo="planck2019",
    res=["res_gaussian"],
    methods=["WL", "WL_c", "true"],
    mcuts=[14.0, 14.25, 14.5],
    fname_append="",
):
    """Consolidate all the MAPs from fnames into a single file."""
    fnames = mock_sample.get_fnames(
        A_survey=A_survey,
        z_min=z_min,
        z_max=z_max,
        m200m_min=m200m_min,
        method=None,
        cosmo=cosmo,
    )

    maps = {}
    # load general info
    with asdf.open(fnames[0], copy_arrays=True, lazy_load=False) as af:
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
                maps[method][mcut]["z_min"] = af[method][mcut]["z_min"]
                maps[method][mcut]["z_max"] = af[method][mcut]["z_max"]
                maps[method][mcut]["m200m_min"] = af[method][mcut]["m200m_min"]
                maps[method][mcut]["cosmo_fixed"] = af[method][mcut]["cosmo_fixed"]

                for r in res:
                    if r not in af[method][mcut].keys():
                        continue
                    maps[method][mcut][r] = {}
                    maps[method][mcut][r]["maps"] = []
                    maps[method][mcut][r]["fun"] = []
                    maps[method][mcut][r]["success"] = []

    # now go through the different mass cuts for each method
    for fname in tqdm(fnames, desc="Loading files"):
        with asdf.open(fname, copy_arrays=True, lazy_load=False) as af:
            for method in methods_in_files:
                for mcut in maps[method].keys():
                    for r in res:
                        if r not in maps[method][mcut].keys():
                            continue
                        try:
                            maps[method][mcut][r]["maps"].append(
                                af[method][mcut][r]["x"]
                            )
                            maps[method][mcut][r]["fun"].append(
                                af[method][mcut][r]["fun"]
                            )
                            maps[method][mcut][r]["success"].append(
                                af[method][mcut][r]["success"]
                            )
                        except Exception as e:
                            print(f"{fname} failed with Exception {e}")

    for method in methods_in_files:
        for mcut in maps[method].keys():
            for r in res:
                if r not in maps[method][mcut].keys():
                    continue
                maps[method][mcut][r]["maps"] = np.atleast_2d(
                    maps[method][mcut][r]["maps"]
                )
                maps[method][mcut][r]["fun"] = np.atleast_1d(
                    maps[method][mcut][r]["fun"]
                )
                maps[method][mcut][r]["success"] = np.atleast_1d(
                    maps[method][mcut][r]["success"]
                )

    maps_fname_base = mock_sample.gen_fname(
        A_survey=A_survey, z_min=z_min, z_max=z_max, m200m_min=m200m_min, cosmo=cosmo
    )

    with asdf.AsdfFile(maps) as af:
        if fname_append != "":
            fname_append = f"_{fname_append}"
        af.write_to(f"{maps_fname_base}_maps{fname_append}.asdf")

    return maps
