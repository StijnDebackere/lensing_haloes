from multiprocessing import Process, Manager
import os
from time import time

import numpy as np
import scipy.optimize as opt
from tremulator import Interpolator

import lensing_haloes.lensing.generate_mock_lensing as mock_lensing
import lensing_haloes.halo.profiles as profs
import lensing_haloes.util.tools as tools

import pdb


# global value of critical density
RHO_CRIT = 2.7763458 * (10.0**11.0)  # [h^2 M_sun / Mpc^3]


def shear_red_nfw_rs_fixed_minimize(
        prms, z, R_bins, shear_red_obs, shear_err,
        sigma_crit, omega_m, c200m_interp, upsample=20,
        **kwargs):
    """Fit an NFW profile with given prms = [log10_m200m] to
    shear_red_obs.

    Parameters
    ----------
    prms : iterable
        - log10_m200m : log of halo mass [M_sun / h]
    z : float
        redshift of the halo
    R_bins : (n+1) array
        observed bin edges
    shear_red_obs : (n) array
        observed reduced shear
    shear_err : (n) array
        shape noise
    sigma_crit : float
        critical surface mass density of observations
    omega_m : float
        mean matter density of the Universe
    c200m_interp : tremulator.Interpolator object
        interpolated concentration-mass relation
    upsample : int
        oversampling to compute NFW average in R_bins

    Returns
    -------
    chi_squared : the chi-squared value of the fit parameters
    """
    log10_m200m = prms
    m200m = 10**log10_m200m
    r200m = tools.mass_to_radius(m200m, 200 * omega_m * RHO_CRIT * (1 + z)**3)
    c200m = c200m_interp.predict([log10_m200m, z]).reshape(-1)[0]

    # get NFW halo to small scales to prevent numerical integration error
    R_temp = np.logspace(np.log10(R_bins.min()) - 3, np.log10(R_bins.max()), 200)
    sigma_nfw = profs.sigma_nfw(R=R_temp, m_x=m200m, r_x=r200m, c_x=c200m)
    shear_red_obs_nfw = mock_lensing.observed_reduced_shear(
        R_bins=R_bins, R=R_temp, sigma_tot=sigma_nfw,
        sigma_crit=sigma_crit, upsample=upsample

    )

    return np.sum(((shear_red_obs - shear_red_obs_nfw) / shear_err)**2)


def shear_red_nfw_rs_free_minimize(
        prms, z, R_bins, shear_red_obs, shear_err,
        sigma_crit, omega_m, upsample=20,
        **kwargs):
    """Fit an NFW profile with given prms = [log10_m200m, c200m] to
    shear_red_obs.

    Parameters
    ----------
    prms : iterable
        - log10_m200m : log of halo mass [M_sun / h]
        - c200m : concentration
    z : float
        redshift of the halo
    R_bins : (n+1) array
        observed bin edges
    shear_red_obs : (n) array
        observed reduced shear
    shear_err : (n) array
        shape noise
    sigma_crit : float
        critical surface mass density of observations
    omega_m : float
        mean matter density of the Universe
    upsample : int
        oversampling to compute NFW average in R_bins

    Returns
    -------
    chi_squared : the chi-squared value of the fit parameters
    """
    log10_m200m, c200m = prms
    m200m = 10**log10_m200m
    r200m = tools.mass_to_radius(m200m, 200 * omega_m * RHO_CRIT * (1 + z)**3)

    # get NFW halo to small scales to prevent numerical integration error
    R_temp = np.logspace(np.log10(R_bins.min()) - 3, np.log10(R_bins.max()), 200)
    sigma_nfw = profs.sigma_nfw(R=R_temp, m_x=m200m, r_x=r200m, c_x=c200m)
    shear_red_obs_nfw = mock_lensing.observed_reduced_shear(
        R_bins=R_bins, R=R_temp, sigma_tot=sigma_nfw,
        sigma_crit=sigma_crit, upsample=upsample
    )

    return np.sum(((shear_red_obs - shear_red_obs_nfw) / shear_err)**2)


def fit_nfw_rs_fixed(
        z, R_bins, shear_red_obs, shear_err,
        sigma_crit, omega_m, c200m_interp,
        **kwargs):
    """Get the best-fitting parameters for the observed shear profile
    assuming an NFW profile with fixed scale radius following the
    c200m_interp relation out to infinity.

    Parameters
    ----------
    z : float
        redshift of the halo
    R_bins : (n+1) array
        observed bin edges
    shear_red_obs : (n) array
        observed reduced shear
    shear_err : (n) array
        shape noise
    sigma_crit : float
        critical surface mass density of observations
    omega_m : float
        mean matter density of the Universe
    c200m_interp : tremulator.Interpolator
        interpolator for the c200m(m200m) relation

    Returns
    -------
    scipy.optimize.OptimizeResult : best-fitting parameters

    """
    def f_min(prms):
        return shear_red_nfw_rs_fixed_minimize(
            prms=prms, z=z, R_bins=R_bins, shear_red_obs=shear_red_obs,
            shear_err=shear_err, sigma_crit=sigma_crit, omega_m=omega_m,
            c200m_interp=c200m_interp,
        )
    res = opt.minimize(
        f_min, [14.5],
        bounds=opt.Bounds(10, 16)
    )

    m200m = 10**res.x[0]
    r200m = tools.mass_to_radius(m200m, 200 * omega_m * RHO_CRIT * (1 + z)**3)
    c200m = c200m_interp.predict([res.x[0], z]).reshape(-1)[0]
    results = {
        "m_x": m200m,
        "r_x": r200m,
        "c_x": c200m,
        "r_s": r200m / c200m,
    }
    return results, res


def fit_nfw_rs_free(
        z, R_bins, shear_red_obs, shear_err,
        sigma_crit, omega_m,
        **kwargs):
    """Get the best-fitting parameters for the observed shear profile
    assuming an NFW profile with free scale radius out to infinity.

    Parameters
    ----------
    z : float
        redshift of the halo
    R_bins : (n+1) array
        observed bin edges
    shear_red_obs : (n) array
        observed reduced shear
    shear_err : (n) array
        shape noise
    sigma_crit : float
        critical surface mass density of observations
    omega_m : float
        mean matter density of the Universe

    Returns
    -------
    scipy.optimize.OptimizeResult : best-fitting parameters

    """
    def f_min(prms):
        return shear_red_nfw_rs_free_minimize(
            prms=prms, z=z, R_bins=R_bins, shear_red_obs=shear_red_obs,
            shear_err=shear_err, sigma_crit=sigma_crit, omega_m=omega_m
        )
    res = opt.minimize(
        f_min, [14.5, 5],
        bounds=opt.Bounds([10, 1], [16, 20])
    )

    m200m = 10**res.x[0]
    c200m = res.x[1]
    r200m = tools.mass_to_radius(m200m, 200 * omega_m * RHO_CRIT * (1 + z)**3)
    results = {
        "m_x": m200m,
        "r_x": r200m,
        "c_x": c200m,
    }
    return results, res
