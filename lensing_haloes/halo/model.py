from pathlib import Path

import numpy as np
import scipy.optimize as opt
from tremulator import Interpolator
from tqdm import tqdm

import lensing_haloes.settings as settings
import lensing_haloes.cosmo.cosmo as cosmo
import lensing_haloes.data.observational_data as obs_data
import lensing_haloes.halo.profiles as profs
import lensing_haloes.util.tools as tools
import lensing_haloes.util.plot as plot

import pdb

TABLE_DIR = settings.TABLE_DIR

# global value of critical density
RHO_CRIT = 2.7763458 * (10.0**11.0)  # [h^2 M_sun / Mpc^3]


def fbar_rx(rx, log10_rt, alpha, fbar, fbar0=0):
    """Fitting relation for the baryon fraction.

    Parameters
    ----------
    rx : array-like
        radial range in units of r_x
    log10_rt : float
        turnover radius
    alpha : float
        sharpness of the turnover
    fbar : float
        cosmic baryon fraction
    fbar0 : float [Default: 0]
        baryon fraction for rx -> 0

    Returns
    -------
    fbar_rx : baryon fraction at rx
    """
    return (
        fbar0 + 0.5 * (fbar - fbar0)
        * (1 + np.tanh((np.log10(rx) - log10_rt) / alpha))
    )


def dfbar_drx(rx, log10_rt, alpha, fbar, fbar0=0):
    """Derivative of the baryon fraction relation with respect to rx.

    Parameters
    ----------
    rx : array-like
        radial range in units of r_x
    log10_rt : float
        turnover radius
    alpha : float
        sharpness of the turnover
    fbar : float
        cosmic baryon fraction
    fbar0 : float [Default: 0]
        baryon fraction for rx -> 0

    Returns
    -------
    dfbar_drx : derivative of fbar_rx at rx
    """
    return (
        0.5 * (fbar - fbar0) / (np.log(10) * alpha * rx)
        * np.cosh((np.log10(rx) - log10_rt) / alpha)**-2
    )


def fit_fbar_rx(
        r, fbar_r, r_x, sigma=None,
        omega_b=0.0493, omega_m=0.315,
        log10_rt=None, alpha=None):
    """Fit the given fbar_r(r) with fbar_rx."""
    fbar0 = 0
    fbar = omega_b / omega_m

    # fbar_r could be nan at first r if derived from mr_from_rho
    nan_mask = ~np.isnan(fbar_r)
    if sigma is not None:
        sigma = sigma[nan_mask]

    try:
        if log10_rt is None:
            if alpha is None:
                def fbar_fit(rx, log10_rt, alpha):
                    return fbar_rx(
                        rx=rx, log10_rt=log10_rt, alpha=alpha,
                        fbar=fbar, fbar0=fbar0
                    )
                opt_prms, opt_cov = opt.curve_fit(
                    f=fbar_fit,
                    xdata=r[nan_mask]/r_x,
                    ydata=fbar_r[nan_mask],
                    sigma=sigma,
                    bounds=(np.array([-np.inf, 0]), np.array([np.inf, np.inf]))
                )
                prms = {
                    'log10_rt': opt_prms[0],
                    'alpha': opt_prms[1],
                    'fbar': fbar,
                    'fbar0': fbar0,
                }
            else:
                def fbar_fit(rx, log10_rt):
                    return fbar_rx(
                        rx=rx, log10_rt=log10_rt, alpha=alpha,
                        fbar=fbar, fbar0=fbar0
                    )
                opt_prms, opt_cov = opt.curve_fit(
                    f=fbar_fit,
                    xdata=r[nan_mask]/r_x,
                    ydata=fbar_r[nan_mask],
                    sigma=sigma,
                    bounds=(np.array([-np.inf]), np.array([np.inf]))
                )
                prms = {
                    'log10_rt': opt_prms[0],
                    'alpha': alpha,
                    'fbar': fbar,
                    'fbar0': fbar0,
                }
        else:
            if alpha is None:
                def fbar_fit(rx, alpha):
                    return fbar_rx(
                        rx=rx, log10_rt=log10_rt, alpha=alpha,
                        fbar=fbar, fbar0=fbar0
                    )
                opt_prms, opt_cov = opt.curve_fit(
                    f=fbar_fit,
                    xdata=r[nan_mask]/r_x,
                    ydata=fbar_r[nan_mask],
                    sigma=sigma,
                    bounds=(np.array([0]), np.array([np.inf]))
                )
                prms = {
                    'log10_rt': log10_rt,
                    'alpha': opt_prms[0],
                    'fbar': fbar,
                    'fbar0': fbar0,
                }
            else:
                prms = {
                    'log10_rt': log10_rt,
                    'alpha': alpha,
                    'fbar': fbar,
                    'fbar0': fbar0,
                }

    except RuntimeError:
        pdb.set_trace()

    return prms


def dm_kwargs_from_fbar(z, r_x, m_dm_x, omega_b, omega_m, return_dmo=False):
    """Return the kwargs for the NFW profiles that match m_dm_x at r_x.

    Parameters
    ----------
    z : float
        redshift
    r_x : float
        overdensity radius [Mpc / h]
    m_dm_x : float
        dark matter mass at r_x [M_sun / h]
    omega_b : float
        baryon density of the Universe
    omega_m : float
        matter density of the Universe
    return_dmo : bool
        also return equivalent DMO halo kwargs

    Returns
    -------
    dm_kwargs : dict with NFW kwargs
    dmo_kwargs, dm_kwargs if return_dmo
    """
    m_dmo_x = m_dm_x / (1 - omega_b / omega_m)
    rho_mz = omega_m * RHO_CRIT * (1 + z)**3

    # load the concentration mass-relation interpolator
    c200m_interp = Interpolator()
    c200m_interp.load(f'{TABLE_DIR}/c200m_correa_planck_2019.asdf')

    # get the equivalent DMO halo properties
    # the interpolator has h=1 units
    m200m_dmo = tools.m200m_dmo_from_mx_dmo(
        mx=m_dmo_x, rx=r_x, z=z, rho_mz=rho_mz,
        c200m_interp=c200m_interp,
    )
    r200m_dmo = tools.mass_to_radius(m200m_dmo, 200 * rho_mz)
    c200m_dmo = c200m_interp.predict(
        tools.arrays_to_coords(np.log10(m200m_dmo), z)
    ).reshape(-1)

    dmo_kwargs = {
        'm_x': m200m_dmo,
        'r_x': r200m_dmo,
        'c_x': c200m_dmo,
    }
    dm_kwargs = {
        'm_x': m_dm_x,
        'r_x': r_x,
        'c_x': c200m_dmo / r200m_dmo * r_x,
    }
    if not return_dmo:
        return dm_kwargs
    else:
        return dmo_kwargs, dm_kwargs


def get_equivalent_dmo_halo(
        z, m500c, r, rho_gas,
        omega_b=0.0493, omega_m=0.315,
        dlog10r=2, n_int=1000,):
    """Get the equivalent DMO halo for the given gas density profile.

    Parameters
    ----------
    z : float
        redshift
    m500c : float
        halo mass
    r : array-like
        radial range for rho_gas
    rho_gas : array-like
        gas density at r
    omega_b : float [Default: 0.0493]
        baryon density of the Universe
    omega_m : float [Default: 0.315]
        matter density of the Universe
    dlog10r : float
        logarithmic ratio to extend rx down to
    n_int : int
        number of steps to perform interpolation over

    Returns
    -------
    dmo_kwargs : dict with kwargs for profs.profile_nfw
        - 'm_x': mass at r_x of equivalent DMO halo
        - 'r_x': radius at which m_x and c_x are defined
        - 'c_x': concentration
    dm_kwargs : dict with kwargs for profs.profile_nfw
        - 'm_x': mass at r_x of the dark matter
        - 'r_x': radius at which m_x and c_x are defined
        - 'c_x': concentration
    """
    r500c = tools.mass_to_radius(
        m500c, 500 * RHO_CRIT * cosmo.E2z(z=z, omega_m=omega_m)
    )

    # get the enclosed mass profile from the given gas profile
    mr_gas_500c = profs.mr_from_rho(
        r=r500c, rs=r, rho=rho_gas, dlog10r=dlog10r, n_int=n_int
    )
    mr_dm_500c = (m500c - mr_gas_500c)
    dmo_kwargs, dm_kwargs = dm_kwargs_from_fbar(
        z=z, r_x=r500c, m_dm_x=mr_dm_500c, omega_b=omega_b, omega_m=omega_m,
        return_dmo=True
    )

    return dmo_kwargs, dm_kwargs


def mr_gas_from_rho_gas(r, rs, rho_gas, dlog10r=2, n_int=1000):
    """Infer the gas mass from the given gas density profiles
    rho_gas(rs|z, m500c).

    Parameters
    ----------
    r : array-like
        radial range to compute fbar_r for
    rs : array-like
        radial range for rho_gas
    rho_gas : array-like
        gas density at rs
    dlog10r : float
        logarithmic ratio to extend rx down to
    n_int : int
        number of steps to perform interpolation over

    Returns
    -------
    mr_gas : array-like
        inferred gas mass m_gas(<r) for the given rho_gas(rs|z, m500c)
    """
    mr_gas = profs.mr_from_rho(
        r=r, rs=rs, rho=rho_gas, n_int=n_int, dlog10r=dlog10r,
    )
    return mr_gas


def mr_dm_from_rho_gas(
        z, r, rs, rho_gas, m500c,
        omega_b=0.0493, omega_m=0.315,
        dlog10r=2, n_int=1000, **dm_kwargs):
    """Infer the dark matter mass from the given gas density profiles
    rho_gas(rs|z, m500c).

    We assume that the gas captures all the baryons and the inferred
    halo mass m500c is correct. The equivalent DMO halo is inferred
    from correcting the dark matter mass for the presence of baryons.
    The dark matter follows the DMO halo.

    Parameters
    ----------
    z : float
        redshift
    r : array-like
        radial range to compute fbar_r for
    rs : array-like
        radial range for rho_gas
    rho_gas : array-like
        gas density at rs
    m500c : float
        halo mass
    dlog10r : float
        logarithmic ratio to extend rx down to
    n_int : int
        number of steps to perform interpolation over

    Returns
    -------
    mr_gas : array-like
        inferred gas mass m_gas(<r) for the given rho_gas(rs|z, m500c)
    """
    if 'm_x' in dm_kwargs and 'r_x' in dm_kwargs and 'r_x' in dm_kwargs:
        pass
    else:
        dmo_kwargs, dm_kwargs = get_equivalent_dmo_halo(
            z=z, m500c=m500c, r=rs, rho_gas=rho_gas,
            omega_b=omega_b, omega_m=omega_m,
            dlog10r=dlog10r, n_int=n_int,
    )
    mr_dm = profs.m_nfw(r=r, **dm_kwargs)
    return mr_dm


def fbar_r_from_rho_gas(
        z, r, rs, rho_gas, m500c,
        omega_b=0.0493, omega_m=0.315,
        dlog10r=2, n_int=1000, **dm_kwargs):
    """Infer the baryon fraction from the given gas density profiles
    rho_gas(rs|z, m500c).

    We assume that the gas captures all the baryons and the inferred
    halo mass m500c is correct. The equivalent DMO halo is inferred
    from correcting the dark matter mass for the presence of baryons.
    The dark matter follows the DMO halo.

    Parameters
    ----------
    z : float
        redshift
    r : array-like
        radial range to compute fbar_r for
    rs : array-like
        radial range for rho_gas
    rho_gas : array-like
        gas density at rs
    m500c : float
        halo mass
    omega_b : float [Default: 0.0493]
        baryon density of the Universe
    omega_m : float [Default: 0.315]
        matter density of the Universe
    dlog10r : float
        logarithmic ratio to extend rx down to
    n_int : int
        number of steps to perform interpolation over
    dm_kwargs : dict with kwargs for profs.profile_nfw
        - 'm_x': mass at r_x of the dark matter
        - 'r_x': radius at which m_x and c_x are defined
        - 'c_x': concentration

    Returns
    -------
    fbar_r : array-like
        inferred baryon fraction for the given rho_gas(rs|z, m500c)
    """
    mr_dm = mr_dm_from_rho_gas(
        z=z, r=r, rs=rs, rho_gas=rho_gas, m500c=m500c,
        omega_b=omega_b, omega_m=omega_m,
        dlog10r=dlog10r, n_int=n_int, **dm_kwargs
    )
    mr_gas = mr_gas_from_rho_gas(
        r=r, rs=rs, rho_gas=rho_gas, n_int=n_int, dlog10r=dlog10r,
    )
    mr_tot = mr_dm + mr_gas

    return mr_gas / mr_tot


def rho_gas_from_fbar(r, r_y, log10_rt, alpha, fbar, fbar0, return_dm=False, **dm_kwargs):
    """
    Get the gas profile for the given fbar(ry) parameters.

    Parameters
    ----------
    r : array-like
        radial range to compute rho_gas for
    r_y : float
        radius to normalize r to for fbar_rx
    log10_rt : float
        turnover radius for fbar_rx
    alpha : float
        sharpness of turnover
    fbar : float
        cosmic baryon fraction
    fbar0 : float
        baryon fraction for rx -> 0
    dm_kwargs : dict with kwargs for profs.profile_nfw
        - 'm_x': mass at r_x
        - 'r_x': radius at which m_x and c_x are defined
        - 'c_x': concentration

    Returns
    -------
    rho_gas(r) : array-like
        gas density derived from fbar_rx
    """
    ry = r / r_y
    # compensate for derivative wrt rx instead of r
    fb_prime = 1 / r_y * dfbar_drx(
        rx=ry, log10_rt=log10_rt, alpha=alpha, fbar=fbar, fbar0=fbar0
    )
    fb = fbar_rx(rx=ry, log10_rt=log10_rt, alpha=alpha, fbar=fbar, fbar0=fbar0)
    m_dm = profs.m_nfw(r, **dm_kwargs)
    rho_dm = profs.profile_nfw(r, **dm_kwargs)

    rho_gas = (
        fb_prime * m_dm / (4 * np.pi * r**2 * (1 - fb)**2)
        + fb * rho_dm / (1 - fb)
    )

    if return_dm:
        return rho_gas, rho_dm
    else:
        return rho_gas


def mr_gas_from_fbar(
        r, r_y, log10_rt, alpha, fbar, fbar0,
        return_dm=False, **dm_kwargs):
    """
    Get the gas profile for the given fbar(ry) parameters.

    Parameters
    ----------
    r : array-like
        radial range to compute mr_gas for
    r_y : float
        radius to normalize r to for fbar_rx
    log10_rt : float
        turnover radius for fbar_rx
    alpha : float
        sharpness of turnover
    fbar : float
        cosmic baryon fraction
    fbar0 : float
        baryon fraction for rx -> 0
    dm_kwargs : dict with kwargs for profs.profile_nfw
        - 'm_x': mass at r_x
        - 'r_x': radius at which m_x and c_x are defined
        - 'c_x': concentration

    Returns
    -------
    mr_gas(r) : array-like
        gas density derived from fbar_rx
    """
    ry = r / r_y
    fb = fbar_rx(rx=ry, log10_rt=log10_rt, alpha=alpha, fbar=fbar, fbar0=fbar0)
    m_dm = profs.m_nfw(r, **dm_kwargs)

    m_gas = fb * m_dm / (1 - fb)
    if return_dm:
        return m_gas, m_dm
    else:
        return m_gas


def fit_rho_gas(
        rx, rho_gas, z, m500c, r500c,
        rho_gas_err=None, z_l=None, rx_fit=None, log=False,
        omega_b=0.0493, omega_m=0.315,
        n_r=30, dlog10r=1, n_int=1000,
        outer_norm=None,):
    """Fit the given density profile rho_gas(rx|m500c) by reconstructing and
    fitting the radial baryon fraction

    Parameters
    ----------
    rx : array-like
        radial range normalized to r500c
    rho_gas : array-like
        gas density at rx
    z : float
        redshift
    m500c : float
        halo mass
    r500c : float
        halo radius
    rho_gas_err : array-like, optional [Default: None]
        error in rho_gas
    z_l : float, optional [Default: None]
        redshift to evolve to self-similarly
    omega_b : float [Default: 0.0493]
        baryon density of the Universe
    omega_m : float [Default: 0.315]
        matter density of the Universe
    n_r : int
        number of bins to rebin rho_gas to
    dlog10r : float
        logarithmic ratio to extend rx down to
    n_int : int
        number of steps to perform interpolation over

    Returns
    -------
    results : dict
        - 'opt_prms' : best-fitting parameters for fbar_rx
        - 'dm_kwargs': corresponding dark matter profile kwargs
        - 'dmo_kwargs': equivalent DMO halo profile kwargs
        - 'rx': rebinned radial range for rho_gas
        - 'rho_gas': rebinned gas density profile
        - 'fbar_rx': derived baryon fraction for the data at rx
    """
    # rescale the measured radius to our chosen one
    # assuming self-similar evolution
    if z_l is None:
        z_l = z
    r500c_z = r500c * cosmo.h2z_ratio(z_1=z_l, z_2=z, omega_m=omega_m)**(-1/3)

    # ditto for density
    # also rebin to fixed grid, set by data
    if rx_fit is None:
        log = True
        rx_new_bins = np.logspace(np.log10(rx.min()), np.log10(rx.max()), n_r + 1)
        rx_new = tools.bin_centers(rx_new_bins, log=log)
    else:
        rx_new_bins = rx_fit
        rx_new = tools.bin_centers(rx_new_bins, log=log)

    rho_gas_z = tools.resample_to_bins(
        bins=rx_new_bins, x=rx, f=rho_gas, log=log
    ) * cosmo.h2z_ratio(z_1=z_l, z_2=z, omega_m=omega_m)

    # get the DM and DMO halo kwargs
    dmo_kwargs, dm_kwargs = get_equivalent_dmo_halo(
        z=z_l, m500c=m500c, r=rx_new*r500c_z, rho_gas=rho_gas_z,
        omega_b=omega_b, omega_m=omega_m, dlog10r=dlog10r, n_int=n_int,
    )


    fbar_data = fbar_r_from_rho_gas(
        z=z_l, r=rx_new*r500c_z, rs=rx_new*r500c_z, rho_gas=rho_gas_z,
        m500c=m500c, omega_b=omega_b, omega_m=omega_m,
        dlog10r=dlog10r, n_int=n_int, **dm_kwargs
    )

    # also resample the error
    if rho_gas_err is not None:
        rho_gas_err_z = tools.resample_to_bins(
            bins=rx_new_bins, x=rx, f=rho_gas_err, log=log
        ) * cosmo.h2z_ratio(z_1=z_l, z_2=z, omega_m=omega_m)

        fbar_plus = fbar_r_from_rho_gas(
            z=z_l, r=rx_new*r500c_z, rs=rx_new*r500c_z,
            rho_gas=rho_gas_z + rho_gas_err_z,
            m500c=m500c, omega_b=omega_b, omega_m=omega_m,
            dlog10r=dlog10r, n_int=n_int,
        )
        fbar_min = fbar_r_from_rho_gas(
            z=z_l, r=rx_new*r500c_z, rs=rx_new*r500c_z,
            rho_gas=rho_gas_z - rho_gas_err_z,
            m500c=m500c, omega_b=omega_b, omega_m=omega_m,
            dlog10r=dlog10r, n_int=n_int,
        )
        fbar_err = np.mean([
            fbar_plus - fbar_data,
            fbar_data - fbar_min
        ], axis=0)
    else:
        fbar_err = None

    if outer_norm is not None:
        fbar = omega_b / omega_m
        rx_new = np.concatenate([rx_new, np.atleast_1d(outer_norm['rx'])])
        fbar_data = np.concatenate([fbar_data, np.atleast_1d(outer_norm['fbar'] * fbar)])
        fbar_err = np.concatenate([fbar_err, np.atleast_1d(outer_norm['fbar_err'])])

    opt_prms = fit_fbar_rx(
        r=rx_new*r500c_z, fbar_r=fbar_data, r_x=r500c_z, sigma=fbar_err,
        omega_b=omega_b, omega_m=omega_m,
    )

    results = {
        'outer_norm': outer_norm,
        'opt_prms': opt_prms,
        'dm_kwargs': dm_kwargs,
        'dmo_kwargs': dmo_kwargs,
        'rx': rx_new,
        'rho_gas': rho_gas_z,
        'rho_gas_err': rho_gas_err_z,
        'r_x': r500c_z,
        'fbar_rx': fbar_data,
        'fbar_min': fbar_min,
        'fbar_plus': fbar_plus,
        'fbar_rx_err': fbar_err,
    }

    if rho_gas_err is not None:
        opt_prms_plus = fit_fbar_rx(
            r=rx_new*r500c_z, fbar_r=fbar_plus, r_x=r500c_z, sigma=fbar_err,
            omega_b=omega_b, omega_m=omega_m, alpha=opt_prms['alpha'],
        )
        opt_prms_min = fit_fbar_rx(
            r=rx_new*r500c_z, fbar_r=fbar_min, r_x=r500c_z, sigma=fbar_err,
            omega_b=omega_b, omega_m=omega_m, alpha=opt_prms['alpha'],
        )
        results['opt_prms_plus'] = opt_prms_plus
        results['opt_prms_min'] = opt_prms_min

    return results


def get_rho_gas_fits_all(
        r_range=None, rx_range=None, log=None, z_l=None,
        omega_b=0.0493, omega_m=0.315,
        datasets=['croston+08'], outer_norm=None,
        n_r=30, dlog10r=1, n_int=1000):
    """Get the best-fitting gas density profiles to the observed enclosed
    gas fractions.

    Parameters
    ----------
    r_range : array-like
        radial range to compute rho_gas_fit for
    z_l : float, optional [Default: None]
        redshift to evolve to self-similarly
    omega_b : float [Default: 0.0493]
        baryon density of the Universe
    omega_m : float [Default: 0.315]
        matter density of the Universe
    datasets : iterable
        names of observational datasets to load
    n_r : int
        number of bins to rebin rho_gas to
    dlog10r : float
        logarithmic ratio to extend rx down to
    n_int : int
        number of steps to perform interpolation over

    Returns
    -------
    results : dict with keys for each dataset
        - dataset : list of dicts for each profile
            - 'rho_gas_fit': best-fitting gas profile
            - 'opt_prms' : best-fitting parameters for fbar_rx
            - 'dm_kwargs': corresponding dark matter profile kwargs
            - 'dmo_kwargs': equivalent DMO halo profile kwargs
            - 'rx': rebinned radial range for rho_gas
            - 'rho_gas': rebinned gas density profile
            - 'fbar_rx': derived baryon fraction for the data at rx
    """
    data = obs_data.load_datasets(datasets=datasets, h_units=True)
    results = {
        'r_range': r_range,
        'rx_range': rx_range,
        'z_l': z_l,
        'omega_m': omega_m,
        'omega_b': omega_b,
    }

    for dataset in datasets:
        results[dataset] = {
            'fit_results': [],
            'z': np.asarray(data[dataset]['z'][:], dtype=float),
            'm500c': np.asarray(data[dataset]['m500c'][:], dtype=float),
            'r500c': np.asarray(data[dataset]['r500c'][:], dtype=float),
            'mgas_500c': np.asarray(data[dataset]['mgas_500c'][:], dtype=float),
            'rx': data[dataset]['rx'][:],
            'rho_gas': data[dataset]['rho'][:],
            'rho_gas_err': data[dataset]['rho_err'][:],
        }
        for idx, (z, rx, rho, rho_err, m500c, r500c) in enumerate(tqdm(
                zip(
                    results[dataset]['z'][:],
                    results[dataset]['rx'][:],
                    results[dataset]['rho_gas'][:],
                    results[dataset]['rho_gas_err'][:],
                    results[dataset]['m500c'][:],
                    results[dataset]['r500c'][:]
                ), desc=f'Fitting profiles {dataset}')):
            fit_results = fit_rho_gas(
                rx=rx, rx_fit=rx_range, log=log, rho_gas=rho, rho_gas_err=rho_err,
                z=z, m500c=m500c, r500c=r500c,
                z_l=z_l, omega_b=omega_b, omega_m=omega_m,
                n_r=n_r, dlog10r=dlog10r, n_int=n_int,
                outer_norm=outer_norm,
            )

            # r500c will be changed to match z_l!
            if rx_range is None and r_range is None:
                r_range_fit = fit_results['rx'] * fit_results['r_x']
            elif rx_range is not None:
                r_range_fit = rx_range * fit_results['r_x']
            elif r_range is not None:
                r_range_fit = r_range

            rho_gas_fit = rho_gas_from_fbar(
                r=r_range_fit, r_y=fit_results['r_x'],
                **fit_results['opt_prms'], **fit_results['dm_kwargs']
            )

            # add the fit to the results
            results[dataset]['fit_results'].append(
                {
                    'rho_gas_fit': rho_gas_fit,
                    'rx_fit': r_range_fit / fit_results['r_x'],
                    'r500c': fit_results['r_x'],
                    **fit_results
                }
            )

    return results


def get_rho_gas_fits_bins(
        n_bins=3, r_range=None, z_l=None,
        omega_b=0.0493, omega_m=0.315,
        datasets=['croston+08'], outer_norm=None,
        n_r=30, dlog10r=2, n_int=1000):
    """Get the best-fitting gas density profiles to the observed enclosed
    gas fractions. All results assume h=1 units.

    Parameters
    ----------
    r_range : array-like
        radial range to compute rho_gas_fit for
    z_l : float, optional [Default: None]
        redshift to evolve to self-similarly
    omega_b : float [Default: 0.0493]
        baryon density of the Universe
    omega_m : float [Default: 0.315]
        matter density of the Universe
    datasets : iterable
        names of observational datasets to load
    n_r : int
        number of bins to rebin rho_gas to
    dlog10r : float
        logarithmic ratio to extend rx down to
    n_int : int
        number of steps to perform interpolation over

    Returns
    -------
    results : dict with keys for each dataset
        - dataset : list of dicts for each profile
            - 'rho_gas_fit': best-fitting gas profile
            - 'opt_prms' : best-fitting parameters for fbar_rx
            - 'dm_kwargs': corresponding dark matter profile kwargs
            - 'dmo_kwargs': equivalent DMO halo profile kwargs
            - 'rx': rebinned radial range for rho_gas
            - 'rho_gas': rebinned gas density profile
            - 'fbar_rx': derived baryon fraction for the data at rx
    """
    data = obs_data.load_datasets(datasets=datasets, h_units=True)
    results = {
        'r_range': r_range,
        'z_l': z_l,
        'omega_m': omega_m,
        'omega_b': omega_b,
    }

    for dataset in datasets:
        # load sample info
        z = np.asarray(data[dataset]['z'][:], dtype=float)
        m500c = np.asarray(data[dataset]['m500c'][:], dtype=float)
        mgas_500c = np.asarray(data[dataset]['mgas_500c'][:], dtype=float)
        r500c = np.asarray(data[dataset]['r500c'][:], dtype=float)

        rx = np.asarray(data[dataset]['rx'][:], dtype=object)
        rho_gas = np.asarray(data[dataset]['rho'][:], dtype=object)
        rho_gas_err = np.asarray(data[dataset]['rho_err'][:], dtype=object)

        # sort by halo mass and split in n_bins ~equal bins
        ids_sorted = np.argsort(m500c)
        m_bins = np.array_split(ids_sorted, n_bins)

        # save results
        results[dataset] = {
            'fit_results': [],
            'm500c': m500c,
            'mgas_500c': np.asarray(data[dataset]['mgas_500c'], dtype=float),
            'r500c': r500c,
            'z': z,
            'm500c_bins': np.empty(len(m_bins), dtype=float),
            'r500c_bins': np.empty(len(m_bins), dtype=float),
            'z_bins': np.empty(len(m_bins), dtype=float),
            'm_bins': m_bins,
            'rx': rx,
            'rho_gas': rho_gas,
            'rho_gas_err': rho_gas_err,
        }

        for idx_m, m_bin in enumerate(m_bins):
            # get the maximum overlapping range for all profiles in the bin
            rx_min = np.max([np.min(r) for r in rx[m_bin]])
            rx_max = np.min([np.max(r) for r in rx[m_bin]])
            rx_bins = np.logspace(np.log10(rx_min), np.log10(rx_max), n_r + 1)
            rx_new = tools.bin_centers(rx_bins, log=True)

            # use the redshifts of the clusters if no common redshift provided
            if z_l is None:
                z_bin = np.median(z[m_bin])
            else:
                z_bin = z_l

            # get the median properties of the gas in the bin
            m500c_bin = 10**np.median(np.log10(m500c[m_bin]))
            r500c_bin = np.median(
                r500c[m_bin]
                * cosmo.h2z_ratio(z_1=z_bin, z_2=z[m_bin], omega_m=omega_m)**(-1/3)
            )
            mgas_500c_bin = 10**np.median(np.log10(mgas_500c[m_bin]))
            rho_gas_bin = np.array([
                tools.resample_to_bins(bins=rx_bins, x=r, f=rho, log=True)
                for (r, rho)  in zip(rx[m_bin], rho_gas[m_bin])
            ]) * cosmo.h2z_ratio(z_1=z_bin, z_2=z[m_bin], omega_m=omega_m).reshape(-1, 1)

            rho_gas_med = np.percentile(rho_gas_bin, 50, axis=0)
            rho_gas_q16 = np.percentile(rho_gas_bin, 16, axis=0)
            rho_gas_q84 = np.percentile(rho_gas_bin, 84, axis=0)
            rho_gas_err = np.mean([
                rho_gas_q84 - rho_gas_med,
                rho_gas_med - rho_gas_q16
            ], axis=0)

            fit_results_med = fit_rho_gas(
                rx=rx_new, rho_gas=rho_gas_med, rho_gas_err=rho_gas_err,
                z=z_bin, m500c=m500c_bin, r500c=r500c_bin,
                z_l=z_bin, omega_b=omega_b, omega_m=omega_m,
                n_r=n_r, dlog10r=dlog10r, n_int=n_int,
                outer_norm=outer_norm,
            )

            if r_range is None:
                r_range_fit = rx_new * r500c_bin
            else:
                r_range_fit = r_range

            rho_gas_med_fit = rho_gas_from_fbar(
                r=r_range_fit, r_y=r500c_bin,
                **fit_results_med['opt_prms'], **fit_results_med['dm_kwargs']
            )

            results[dataset]['m500c_bins'][idx_m] = m500c_bin
            results[dataset]['r500c_bins'][idx_m] = r500c_bin
            results[dataset]['z_bins'][idx_m] = z_bin
            results[dataset]['rx_range'] = r_range_fit / r500c_bin

            # add the fit to the results
            results[dataset]['fit_results'].append(
                {
                    "rho_gas_fit": rho_gas_med_fit,
                    "rho_gas_med": rho_gas_med,
                    "rho_gas_q16": rho_gas_q16,
                    "rho_gas_q84": rho_gas_q84,
                    **fit_results_med
                }
            )

    return results
