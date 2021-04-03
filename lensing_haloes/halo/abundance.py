"""This module contains the halo abundance calculations.
"""
import numpy as np
from pyccl.halos.hmfunc import MassFuncTinker08
import scipy.integrate as intg
import scipy.interpolate as interp
from scipy.special import erfc

from lensing_haloes.cosmo.cosmo import cosmology, dVdz

from pdb import set_trace


def z2a(z):
    """Convert z to a"""
    return 1.0 / (1 + z)


def a2z(a):
    """Convert a to z"""
    return 1.0 / a - 1


def dndlog10mdz(z, log10_m200m, cosmo=cosmology(), MassFunc=MassFuncTinker08):
    """Return the differential number density of haloes at redshifts z for
    masses m200m for the given cosmology, cosmo, and survey area,
    A_survey.

    Parameters
    ----------
    z : array-like
        redshifts
    log10_m200m : float or array
        log10 of the mass [M_sun / h]
    cosmo : pyccl.Cosmology object
        cosmology
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use

    Returns
    -------
    dNdlog10mdz : array-like, with (z, m) along axes [h^3 / Mpc^3]

    """
    z = np.atleast_1d(z)
    m200m = np.atleast_1d(10 ** log10_m200m)

    # initialize MassFunc
    hmf = MassFunc(cosmo)

    # pyccl works without h units
    # -> take out mass scaling
    # -> add back in scaling in final result
    dndlg10mdz = (
        np.array(
            [
                hmf.get_mass_function(cosmo=cosmo, M=m200m / cosmo._params.h, a=a)
                for a in z2a(z)
            ]
        )
        * (1.0 / cosmo._params.h) ** 3
    )

    return dndlg10mdz


def dndlog10mdz_mizi(z, log10_m200m, cosmo=cosmology(), MassFunc=MassFuncTinker08):
    """Return the differential number density of haloes for each (z, m200m) pair
    for the given cosmology, cosmo, and survey area, A_survey.

    Parameters
    ----------
    z : array-like
        redshifts
    log10_m200m : float or array
        log10 of the mass [M_sun / h]
    cosmo : pyccl.Cosmology object
        cosmology
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use

    Returns
    -------
    dndlog10mdz : array-like [h^3 / Mpc^3]

    """
    z = np.atleast_1d(z)
    m200m = np.atleast_1d(10 ** log10_m200m)
    if z.shape != m200m.shape:
        raise ValueError("z and m200m need to have the same shape.")

    # initialize MassFunc
    hmf = MassFunc(cosmo)

    # pyccl works without h units
    # -> take out mass scaling
    # -> add back in scaling in final result
    dndlg10mdz = (
        np.array(
            [
                hmf.get_mass_function(cosmo=cosmo, M=m / cosmo._params.h, a=a)
                for (m, a) in zip(m200m, z2a(z))
            ]
        )
        * (1 / cosmo._params.h) ** 3
    )

    return dndlg10mdz


def dNdlog10mdz(
    z, log10_m200m, cosmo=cosmology(), A_survey=2500, MassFunc=MassFuncTinker08
):
    """Return the differential number of haloes at redshifts z for masses
    m200m for the given cosmology, cosmo, and survey area, A_survey.

    Parameters
    ----------
    z : array-like
        redshifts
    log10_m200m : float or array
        log10 of the mass [M_sun / h]
    cosmo : pyccl.Cosmology object
        cosmology
    A_survey : float
        survey area [deg^2]
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use

    Returns
    -------
    dNdlog10mdz : array-like, with (z, m) along axes [h^3 / Mpc^3]
    """
    z = np.atleast_1d(z)

    dndlg10mdz = dndlog10mdz(
        z=z, log10_m200m=log10_m200m, cosmo=cosmo, MassFunc=MassFunc
    )

    volume = A_survey * dVdz(
        z=z, omega_m=cosmo._params.Omega_m, h=cosmo._params.h, w0=cosmo._params.w0
    )

    return dndlg10mdz * volume.reshape(-1, 1)


def dNdlog10mdz_mizi(
    z, log10_m200m, cosmo=cosmology(), A_survey=2500, MassFunc=MassFuncTinker08
):
    """Return the differential number of haloes for each (z, m200m) pair
    for the given cosmology, cosmo, and survey area, A_survey.

    Parameters
    ----------
    z : array-like
        redshifts
    log10_m200m : float or array
        log10 of the mass [M_sun / h]
    cosmo : pyccl.Cosmology object
        cosmology
    A_survey : float
        survey area [deg^2]
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use

    Returns
    -------
    dNdlog10mdz : array-like [h^3 / Mpc^3]

    """
    z = np.atleast_1d(z)
    m200m = np.atleast_1d(10 ** log10_m200m)
    if z.shape != m200m.shape:
        raise ValueError("z and m200m need to have the same shape.")

    dndlg10mdz = dndlog10mdz_mizi(
        z=z, log10_m200m=log10_m200m, cosmo=cosmo, MassFunc=MassFunc
    )

    volume = A_survey * dVdz(
        z=z, omega_m=cosmo._params.Omega_m, h=cosmo._params.h, w0=cosmo._params.w0
    )

    return dndlg10mdz * volume.reshape(-1)


def dNdlog10mdz_integral(
    z_min=0.25,
    z_max=10,
    n_z=100,
    log10_m200m_min=np.log10(3e14),
    log10_m200m_max=18,
    log10_mobs_min=None,
    log10_mobs_max=None,
    n_m=400,
    cosmo=cosmology(),
    A_survey=2500,
    MassFunc=MassFuncTinker08,
    sigma_log10_mobs=None,
    sigma_log10_mobs_dist=None,
    **sigma_log10_mobs_dist_kwargs
):
    """Return the integral of the total number of objects expected in a
    survey of area A_survey [deg^2]

    Parameters
    ----------
    z_min : float
        lowest redshift in sample
    z_max : float
        maximum redshift in integration
        [Default: 10]
    n_z : int
        number of redshifts to sample
    log10_m200m_min : float
        log10 of the minimum mass in the survey [M_sun / h]
    log10_m200m_max : float
        log10 of the maximum  mass in the integration [M_sun / h]
        [Default: 18]
    log10_mobs_min : float
        log10 of the observed mass bin [M_sun / h]
        [Default : None]
    log10_mobs_max : float
        log10 of the observed mass bin [M_sun / h]
        [Default : None]
    sigma_log10_mobs : array-like or float
        range of sigma_log10_m_obs
        [Default : None]
    sigma_log10_mobs_dist : callable
        distribution for sigma_log10_mobs
        [Default : None]
    n_m : int
        number of masses to sample
    cosmo : pyccl.Cosmology object
        cosmology
    A_survey : float
        survey area [deg^2]
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use

    Returns
    -------
    N : float
        number of expected objects in survey
    """
    z_range = np.linspace(z_min, z_max, n_z)

    # create initial range to calculate hmf that will not crash
    log10_m = np.linspace(log10_m200m_min, 16, n_m)
    dNdlg10mdz_range = dNdlog10mdz(
        z=z_range,
        log10_m200m=log10_m,
        A_survey=A_survey,
        cosmo=cosmo,
        MassFunc=MassFunc,
    )

    # now create interpolator that will linearly extrapolate the result
    interp_func = interp.interp1d(
        log10_m,
        np.log10(dNdlg10mdz_range),
        kind="linear",
        fill_value="extrapolate",
        axis=1,
    )

    log10_m_full = np.linspace(log10_m200m_min, log10_m200m_max, n_m)

    if sigma_log10_mobs is None or log10_mobs_max is None or log10_mobs_min is None:
        conv_obs = 1.

    else:
        sigma_log10_mobs = np.atleast_2d(sigma_log10_mobs)

        xi = (
            (log10_mobs_min - log10_m_full)[..., None]
            / (2 * sigma_log10_mobs ** 2)**0.5
        )
        xiplusone = (
            (log10_mobs_max - log10_m_full)[..., None]
            / (2 * sigma_log10_mobs ** 2)**0.5
        )
        erfc_term = 0.5 * (erfc(xi) - erfc(xiplusone))
        if sigma_log10_mobs.shape[-1] == 1:
            conv_obs = erfc_term.reshape(-1)
        else:
            conv_obs = intg.simps(
                y=erfc_term * sigma_log10_mobs_dist(
                    sigma_log10_mobs, **sigma_log10_mobs_dist_kwargs
                ),
                x=sigma_log10_mobs, axis=-1
            )

    dNdlg10mdz_full = 10 ** interp_func(log10_m_full) * conv_obs


    # the interpolator returns nan for np.log10(0), these values should be 0
    dNdlg10mdz_full[np.isnan(dNdlg10mdz_full)] = 0.0

    # now integrate the m and z dimensions
    Nz = intg.simps(y=dNdlg10mdz_full, x=log10_m_full, axis=1)
    N = intg.simps(y=Nz, x=z_range)
    return N


def N_in_bins(
    z_bin_edges,
    m200m_bin_edges,
    n_z=50,
    n_m=1000,
    cosmo=cosmology(),
    A_survey=2500,
    MassFunc=MassFuncTinker08,
    pool=None,
    sigma_log10_mobs=None,
    sigma_log10_mobs_dist=None,
    **kwargs
):
    """Return the integral of the total number of objects expected in a
    survey of area A_survey [deg^2]

    Parameters
    ----------
    z_bin_edges : (z,) array
        redshift bins
    m200m_bin_edges : (m,) array
        mass bins
    n_z : int
        number of redshifts to sample
    n_m : int
        number of masses to sample
    cosmo : pyccl.Cosmology object
        cosmology
    A_survey : float
        survey area [deg^2]
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use
    pool : multiprocessing pool or None
    sigma_log10_mobs : array-like or float
        uncertainty on the mass
    sigma_log10_mobs_dist : callable
        distribution for sigma_log10_mobs
        [Default : None]

    Returns
    -------
    N : (z, m) array
        number of expected objects in survey
    """
    z_mins, m_mins = np.meshgrid(z_bin_edges[:-1], m200m_bin_edges[:-1])
    z_maxs, m_maxs = np.meshgrid(z_bin_edges[1:], m200m_bin_edges[1:])

    # prepare coordinates to be passed to N
    coords = np.concatenate(
        [
            z_mins.ravel().reshape(-1, 1),
            z_maxs.ravel().reshape(-1, 1),
            m_mins.ravel().reshape(-1, 1),
            m_maxs.ravel().reshape(-1, 1),
        ],
        axis=-1,
    )

    def N(edges):
        z_min, z_max, m_min, m_max = edges
        if sigma_log10_mobs is None:
            log10_m200m_min = np.log10(m_min)
            log10_m200m_max = np.log10(m_max)
            log10_mobs_min = None
            log10_mobs_max = None
        else:
            log10_m200m_min = np.log10(m_min) - 2.5 * np.max(sigma_log10_mobs)
            log10_m200m_max = np.log10(m_max) + 2.5 * np.max(sigma_log10_mobs)
            log10_mobs_min = np.log10(m_min)
            log10_mobs_max = np.log10(m_max)

        return dNdlog10mdz_integral(
            z_min=z_min,
            z_max=z_max,
            n_z=n_z,
            log10_m200m_min=log10_m200m_min,
            log10_m200m_max=log10_m200m_max,
            log10_mobs_min=log10_mobs_min,
            log10_mobs_max=log10_mobs_max,
            sigma_log10_mobs=sigma_log10_mobs,
            sigma_log10_mobs_dist=sigma_log10_mobs_dist,
            n_m=n_m,
            cosmo=cosmo,
            A_survey=A_survey,
            MassFunc=MassFunc,
            **kwargs
        )

    if pool is not None:
        map_fn = pool.map
    else:
        map_fn = map

    return np.asarray(list(map_fn(N, coords))).reshape(z_mins.shape).T
