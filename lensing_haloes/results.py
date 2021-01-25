from copy import deepcopy
from pathlib import Path

import asdf
import numpy as np
import matplotlib.pyplot as plt
import scipy.interpolate as interp
import scipy.optimize as opt
from tremulator import Interpolator

import lensing_haloes.settings as settings
from lensing_haloes.cosmo.cosmo import cosmology, cosmo_dict, E2z
import lensing_haloes.halo.model as halo_model
import lensing_haloes.halo.profiles as profs
import lensing_haloes.lensing.fit_mock_lensing as fit_lensing
import lensing_haloes.lensing.generate_mock_lensing as mock_lensing
import lensing_haloes.util.tools as tools

import pdb


TABLE_DIR = settings.TABLE_DIR
RHO_CRIT = 2.7763458 * (10.0**11.0)  # [h^2 M_sun / Mpc^3]


def linear_fit(x, a, b):
    log10_m = x
    return a * (log10_m - b)


def fit_observational_dataset(
        dataset, z, omega_b, omega_m, outer_norm=None,
        dlog10r=2, n_int=1000, n_bins=4, err=True, bins=True,
        diagnostic=False):
    """Fit linear relations to the alpha(m500c, z) and log10_rt(m500c, z)
    relations for the given gas density profiles dataset."""
    if bins:
        results = halo_model.get_rho_gas_fits_bins(
            datasets=[dataset], z_l=z, omega_b=omega_b, omega_m=omega_m,
            dlog10r=dlog10r, n_int=n_int, outer_norm=outer_norm, n_bins=n_bins
        )
        results = results[dataset]
        m500c_data = results['m500c_bins'][:]
    else:
        results = halo_model.get_rho_gas_fits_all(
            datasets=[dataset], z_l=z, omega_b=omega_b, omega_m=omega_m,
            dlog10r=dlog10r, n_int=n_int, outer_norm=outer_norm
        )
        results = results[dataset]
        m500c_data = results['m500c'][:]


    if bins:
        # simply take the best-fitting values from the median, q16 and q84 profiles
        log10_rt = np.array([r['opt_prms']['log10_rt'] for r in results['fit_results']])
        alpha = np.array([r['opt_prms']['alpha'] for r in results['fit_results']])
        log10_rt_plus = np.array([r['opt_prms_plus']['log10_rt'] for r in results['fit_results']])
        log10_rt_min = np.array([r['opt_prms_min']['log10_rt'] for r in results['fit_results']])
    else:
        # bin the best-fitting parameters to all profiles
        bin_ids = np.array_split(np.argsort(m500c_data), n_bins)
        log10_rt = np.array([r['opt_prms']['log10_rt'] for r in results['fit_results']])
        alpha = np.array([r['opt_prms']['alpha'] for r in results['fit_results']])
        log10_rt_med = np.array([
            np.percentile(log10_rt[bin_ids[i]], 50) for i in range(len(bin_ids))
        ])
        alpha_med = np.array([
            np.percentile(alpha[bin_ids[i]], 50) for i in range(len(bin_ids))
        ])
        log10_rt_plus = np.array([
            np.percentile(log10_rt[bin_ids[i]], 16) for i in range(len(bin_ids))
        ])
        log10_rt_min = np.array([
            np.percentile(log10_rt[bin_ids[i]], 84) for i in range(len(bin_ids))
        ])
        log10_rt = log10_rt_med
        alpha = alpha_med
        m500c_data = np.array([
            np.percentile(m500c_data[bin_ids[i]], 50) for i in range(len(bin_ids))
        ])


    log10_rt_prms, cov = opt.curve_fit(
        linear_fit, xdata=np.log10(m500c_data), ydata=log10_rt,
        p0=[-0.5, 13.5], maxfev=5000
    )
    # alpha is the same for all, by construction
    alpha_prms, cov = opt.curve_fit(
        linear_fit, xdata=np.log10(m500c_data), ydata=alpha,
        p0=[-0.5, 15], maxfev=5000)

    # force slope to equal median value
    log10_rt_prms_plus, cov = opt.curve_fit(
        lambda m, b: linear_fit(m, a=log10_rt_prms[0], b=b),
        xdata=np.log10(m500c_data), ydata=log10_rt_plus,
        p0=[13.5], maxfev=5000
    )
    log10_rt_prms_plus = [log10_rt_prms[0], log10_rt_prms_plus[0]]
    log10_rt_prms_min, cov = opt.curve_fit(
        lambda m, b: linear_fit(m, a=log10_rt_prms[0], b=b),
        xdata=np.log10(m500c_data), ydata=log10_rt_min,
        p0=[13.5], maxfev=5000
    )
    log10_rt_prms_min = [log10_rt_prms[0], log10_rt_prms_min[0]]

    # alpha is the same for all, by construction
    alpha_prms, cov = opt.curve_fit(
        linear_fit, xdata=np.log10(m500c_data), ydata=alpha,
        p0=[-0.5, 15], maxfev=5000)

    if diagnostic:
        plt.clf()
        lr, = plt.plot(m500c_data, log10_rt, lw=0, marker='o', label='med')
        la, = plt.plot(m500c_data, alpha, lw=0, marker='x')
        lrp, = plt.plot(m500c_data, log10_rt_plus, lw=0, marker='o', label='plus')
        lrm, = plt.plot(m500c_data, log10_rt_min, lw=0, marker='o', label='min')

        plt.plot(m500c_data, linear_fit(np.log10(m500c_data), *log10_rt_prms), c=lr.get_color(), ls="--")
        plt.plot(m500c_data, linear_fit(np.log10(m500c_data), *alpha_prms), c=la.get_color(), ls="--")
        plt.plot(
            m500c_data, linear_fit(np.log10(m500c_data), *log10_rt_prms_plus),
            c=lrp.get_color(), ls="--"
        )
        plt.plot(
            m500c_data, linear_fit(np.log10(m500c_data), *log10_rt_prms_min),
            c=lrm.get_color(), ls="--"
        )
        plt.xlabel(r'$m_\mathrm{500c}$')
        plt.xlim(7e13, 6e14)
        plt.ylim(-1.5, 2.5)
        plt.xscale('log')
        plt.legend()
        plt.show()

    if err:
        return {
            'med': {
                'log10_rt': log10_rt_prms,
                'alpha': alpha_prms,
            },
            'min': {
                'log10_rt': log10_rt_prms_min,
                'alpha': alpha_prms,
            },
            'plus': {
                'log10_rt': log10_rt_prms_plus,
                'alpha': alpha_prms,
            }
        }
    else:
        return {
            'med': {
                'log10_rt': log10_rt_prms,
                'alpha': alpha_prms,
            }
        }


def save_halo_model(
        z, m500c,
        cosmo={
            "omega_m": 0.315,
            "sigma_8": 0.811,
            "n_s": 0.965,
            "h": 0.674,
            "omega_b": 0.0493,
            "w0": -1,
        },
        dataset='croston+08',
        n_bins=3,
        r_range=np.logspace(-2, 1, 100),
        dlog10r=3, n_int=1000,
        outer_norm=None,
        profile=None,
        save_dir=TABLE_DIR, model_fname_append='planck2019'):
    """Save the halo model for the given dataset, redshift range, halo
    mass range and cosmology.

    Parameters
    ----------
    z : array-like
        redshift range
    m500c : array-like
        halo mass range [M_sun / h]
    cosmo : dict
        cosmology
    dataset : str
        dataset to load in
        (make sure that it is available in lensing_haloes.data.observational_data)
    n_bins : int
        number of mass bins to use
    r_range : array-like
        radial range to generate density profiles for [Mpc / h]
    dlog10r : float
        number of logarithmic steps that will be extrapolated down in r_range
        integration
    n_int : int
        number of steps to take in the integration
    outer_norm : float
        fraction of cosmic baryon fraction to force outer normalization to
    profile : ['median', 'plus', 'min']
        take median, q84 or q16 of observational data
    save_dir : str
        directory to save resulting asdf file to
    model_fname_append : str
        identifier to append to the filename

    Returns
    -------
    results : dict
        dictionary containing resulting density profiles.
    """
    z = np.atleast_1d(z)
    omega_m = cosmo['omega_m']
    omega_b = cosmo['omega_b']
    fbar = omega_b / omega_m

    results = {
        'z': z,
        'm500c': m500c,
        'cosmo': cosmo,
        'omega_b': omega_b,
        'omega_m': omega_m,
        'r_range': r_range,
        'r500c': np.empty(z.shape + m500c.shape, dtype=float),
        'fbar_500c': np.empty(z.shape + m500c.shape, dtype=float),
        'fbar_200m': np.empty(z.shape + m500c.shape, dtype=float),
        'm200m_obs': np.empty(z.shape + m500c.shape, dtype=float),
        'r200m_obs': np.empty(z.shape + m500c.shape, dtype=float),
        'fbar_200c': np.empty(z.shape + m500c.shape, dtype=float),
        'm200c_obs': np.empty(z.shape + m500c.shape, dtype=float),
        'r200c_obs': np.empty(z.shape + m500c.shape, dtype=float),
        'm200m_dmo': np.empty(z.shape + m500c.shape, dtype=float),
        'r200m_dmo': np.empty(z.shape + m500c.shape, dtype=float),
        'c200m_dmo': np.empty(z.shape + m500c.shape, dtype=float),
        'alpha': np.empty(z.shape + m500c.shape, dtype=float),
        'log10_rt': np.empty(z.shape + m500c.shape, dtype=float),
        'm_gas': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'm_dm': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'm_tot': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'm_dmo': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'rho_gas': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'rho_dm': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'rho_tot': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'rho_dmo': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'sigma_gas': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'sigma_dm': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'sigma_tot': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'sigma_dmo': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
    }
    for idx_z, zz in enumerate(z):
        fit_prms = fit_observational_dataset(
            dataset=dataset, z=zz, omega_b=omega_b, omega_m=omega_m,
            dlog10r=dlog10r, n_int=n_int, n_bins=n_bins, bins=True,
            outer_norm=outer_norm, err=True, diagnostic=False
        )
        if profile == 'median' or profile is None:
            profile_append = ''
            log10_rt_prms = fit_prms['med']['log10_rt']
            alpha_prms = fit_prms['med']['alpha']
        elif profile == 'plus':
            profile_append = '_plus'
            log10_rt_prms = fit_prms['plus']['log10_rt']
            alpha_prms = fit_prms['med']['alpha']
        elif profile == 'min':
            profile_append = '_min'
            log10_rt_prms = fit_prms['min']['log10_rt']
            alpha_prms = fit_prms['med']['alpha']

        log10_rt = linear_fit(np.log10(m500c), *log10_rt_prms)
        alpha = linear_fit(np.log10(m500c), *alpha_prms)

        results['log10_rt'][idx_z] = log10_rt
        results['alpha'][idx_z] = alpha

        # get the halo baryon fraction at r500c
        fbar_500c = halo_model.fbar_rx(
            1, log10_rt=log10_rt, alpha=alpha, fbar=fbar, fbar0=0
        )
        r500c = tools.mass_to_radius(
            m500c, 500 * RHO_CRIT * E2z(z=zz, omega_m=omega_m)
        )
        results['r500c'][idx_z] = r500c
        results['fbar_500c'][idx_z] = fbar_500c

        for idx_m in range(len(m500c)):
            # get the dm halo kwargs
            dmo_kwargs, dm_kwargs = halo_model.dm_kwargs_from_fbar(
                z=zz, r_x=r500c[idx_m], m_dm_x=m500c[idx_m]*(1-fbar_500c[idx_m]),
                omega_b=omega_b, omega_m=omega_m, return_dmo=True
            )

            def rho_gas(r_range):
                return halo_model.rho_gas_from_fbar(
                    r=r_range, r_y=r500c[idx_m], log10_rt=log10_rt[idx_m],
                    alpha=alpha[idx_m], fbar=fbar, fbar0=0, return_dm=False, **dm_kwargs
                )
            rho_dm = profs.profile_nfw(r_range, **dm_kwargs)
            rho_dmo = profs.profile_nfw(r_range, **dmo_kwargs)

            def m_gas(r_range):
                return halo_model.mr_gas_from_fbar(
                    r=r_range, r_y=r500c[idx_m], log10_rt=log10_rt[idx_m],
                    alpha=alpha[idx_m], fbar=fbar, fbar0=0, return_dm=False, **dm_kwargs
                )
            m_dm = profs.m_nfw(r_range, **dm_kwargs)
            m_dmo = profs.m_nfw(r_range, **dmo_kwargs)
            def m_tot(r_range):
                return m_gas(r_range) + profs.m_nfw(r_range, **dm_kwargs)

            m200m_obs = tools.mr_to_mx(
                m_tot, rho_x=200*omega_m*RHO_CRIT*(1+zz)**3)
            r200m_obs = tools.mass_to_radius(
                m200m_obs, rho_mean=200*omega_m*RHO_CRIT*(1+zz)**3)
            fbar_200m = m_gas(r200m_obs) / m_tot(r200m_obs)

            m200c_obs = tools.mr_to_mx(
                m_tot, rho_x=200 * RHO_CRIT * E2z(z=zz, omega_m=omega_m))
            r200c_obs = tools.mass_to_radius(
                m200c_obs, rho_mean=200 * RHO_CRIT * E2z(z=zz, omega_m=omega_m))
            fbar_200c = m_gas(r200c_obs) / m_tot(r200c_obs)

            m200m_dmo = dmo_kwargs['m_x']
            r200m_dmo = dmo_kwargs['r_x']
            c200m_dmo = dmo_kwargs['c_x']

            # get weak lensing relevant properties
            sigma_gas = profs.sigma_from_rho(R=r_range, rho_func=rho_gas)
            sigma_dm = profs.sigma_nfw(R=r_range, **dm_kwargs)
            sigma_tot = sigma_gas + sigma_dm
            sigma_dmo = profs.sigma_nfw(R=r_range, **dmo_kwargs)

            sigma_mean, sigma = profs.sigma_mean_from_sigma(
                R=r_range, Rs=r_range, sigma=sigma_tot, n_int=n_int, return_sigma=True
            )

            results['fbar_200m'][idx_z, idx_m] = fbar_200m
            results['m200m_obs'][idx_z, idx_m] = m200m_obs
            results['r200m_obs'][idx_z, idx_m] = r200m_obs
            results['fbar_200c'][idx_z, idx_m] = fbar_200c
            results['m200c_obs'][idx_z, idx_m] = m200c_obs
            results['r200c_obs'][idx_z, idx_m] = r200c_obs
            results['m200m_dmo'][idx_z, idx_m] = m200m_dmo
            results['r200m_dmo'][idx_z, idx_m] = r200m_dmo
            results['c200m_dmo'][idx_z, idx_m] = c200m_dmo
            results['m_gas'][idx_z, idx_m] = m_gas(r_range)
            results['m_dm'][idx_z, idx_m] = m_dm
            results['m_tot'][idx_z, idx_m] = m_dm + m_gas(r_range)
            results['m_dmo'][idx_z, idx_m] = m_dmo
            results['rho_gas'][idx_z, idx_m] = rho_gas(r_range)
            results['rho_dm'][idx_z, idx_m] = rho_dm
            results['rho_tot'][idx_z, idx_m] = rho_dm + rho_gas(r_range)
            results['rho_dmo'][idx_z, idx_m] = rho_dmo
            results['sigma_gas'][idx_z, idx_m] = sigma_gas
            results['sigma_dm'][idx_z, idx_m] = sigma_dm
            results['sigma_tot'][idx_z, idx_m] = sigma_dm + sigma_gas
            results['sigma_dmo'][idx_z, idx_m] = sigma_dmo

    with asdf.AsdfFile(results) as af:
        fname_base = f'{save_dir}/model_fgas_r_{model_fname_append}'
        if z.shape[0] == 1:
            fname = (
                f'{fname_base}'
                f'_z_{str(np.round(z[0], 2)).replace(".", "p")}'
                )
        else:
            fname = (
                f'{fname_base}'
                f'_z_{str(np.round(z.min(), 2)).replace(".", "p")}'
                f'-{str(np.round(z.max(), 2)).replace(".", "p")}'
            )
        fname = (
            f'{fname}_m500c_'
            f'{str(np.round(np.log10(m500c.min()), 2)).replace(".", "p")}'
            f'-{str(np.round(np.log10(m500c.max()), 2)).replace(".", "p")}'
            f'_nbins_{n_bins:d}{profile_append}.asdf'
        )
        print(f'Saving to {fname}')
        af.write_to(fname)

    return results


def save_halo_model_lensing(
        R_bins, log, z_ref=0.43, beta_mean=0.5, n_arcmin2=10,
        zscale=True, dlog10r=3, n_int=1000,
        save_dir=TABLE_DIR, model_fname_append=''):
    """Get the lensing information for the specified halo model.

    Parameters
    ----------
    R_bins : array-like
        projected radii to compute the lensing profiles for [Mpc / h]
    log : bool
        are the bins log-spaced or not
    z_ref : float
        reference redshift for which we scale the R_bins range
    beta_mean : float
        lensing efficiency
    n_arcmin2 : float
        mean background galaxy number density [# / arcmin^2]
    zscale : bool
        scale fitting range with respect to z_ref
    dlog10r : float
        number of logarithmic steps that will be extrapolated down in r_range
        integration
    n_int : int
        number of steps to take in the integration
    save_dir : str
        directory to save resulting asdf file to
    model_fname_append : str
        identifier to append to the filename

    Returns
    -------
    results : dict
        dictionary containing resulting reduced shear profiles and
        best-fitting NFW parameters.

    """
    with asdf.open(
            f'{save_dir}/model_fgas_r_{model_fname_append}.asdf',
            copy_arrays=True, lazy_load=False) as af:
        results = af.tree

    # load model parameters
    cosmo = results['cosmo']
    z = results['z']
    m500c = results['m500c']
    r_range = results['r_range']
    c200m_interp = Interpolator()
    c200m_interp.load(f'{TABLE_DIR}/c200m_correa_planck_2019.asdf')

    # load lensing properties
    # (z, R) array
    n_bins = R_bins.shape[-1] - 1
    if zscale:
        R_bins = R_bins.reshape(1, -1) * (1 + z_ref) / (1 + z).reshape(-1, 1)
    R_obs = tools.bin_centers(R_bins, log=log)
    sigma_crit = mock_lensing.sigma_critical(
        z_l=z, beta_mean=beta_mean, cosmo=cosmo,
    )
    shape_noise = mock_lensing.shape_noise(
        R_bins=R_bins, z_l=z, cosmo=cosmo, n_arcmin2=n_arcmin2, log=log
    )
    sigma_tot = results['sigma_tot']
    sigma_dmo = results['sigma_dmo']

    # cosmological parameters
    omega_m = results['omega_m']
    omega_b = results['omega_b']
    fbar = omega_b / omega_m

    results_new = {
        'R_bins': R_bins,
        'R_obs': R_obs,
        'shape_noise': shape_noise,
        'sigma_crit': sigma_crit,
        'm200m_WL': np.empty(z.shape + m500c.shape, dtype=float),
        'r200m_WL': np.empty(z.shape + m500c.shape, dtype=float),
        'c200m_WL': np.empty(z.shape + m500c.shape, dtype=float),
        'm200m_WL_rs': np.empty(z.shape + m500c.shape, dtype=float),
        'r200m_WL_rs': np.empty(z.shape + m500c.shape, dtype=float),
        'c200m_WL_rs': np.empty(z.shape + m500c.shape, dtype=float),
        'shear_red_obs': np.empty(z.shape + m500c.shape + (n_bins, ), dtype=float),
        'shear_red_dmo': np.empty(z.shape + m500c.shape + (n_bins, ), dtype=float),
        'shear_red_WL': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
        'shear_red_WL_rs': np.empty(z.shape + m500c.shape + r_range.shape, dtype=float),
    }
    results['shear_red_tot'] = profs.shear_red_from_sigma(
        R=r_range, Rs=r_range, sigma=results['sigma_tot'], n_int=n_int,
        sigma_crit=sigma_crit[:, None, None]
    )

    for idx_z, zz in enumerate(z):
        for idx_m in range(len(m500c)):
            if zscale:
                Rbins = R_bins[idx_z]
                Robs = R_obs[idx_z]
            else:
                Rbins = R_bins
                Robs = R_obs
            shear_red_obs = mock_lensing.observed_reduced_shear(
                R_bins=Rbins, R=r_range, sigma_tot=sigma_tot[idx_z, idx_m],
                sigma_crit=sigma_crit[idx_z], n_int=n_int,
            )
            results_WL, res_WL = fit_lensing.fit_nfw_rs_fixed(
                z=zz, R_bins=Rbins, shear_red_obs=shear_red_obs,
                shear_err=shape_noise[idx_z], sigma_crit=sigma_crit[idx_z],
                omega_m=omega_m, c200m_interp=c200m_interp
            )
            results_WL_rs, res_WL_rs = fit_lensing.fit_nfw_rs_free(
                z=zz, R_bins=Rbins, shear_red_obs=shear_red_obs,
                shear_err=shape_noise[idx_z], sigma_crit=sigma_crit[idx_z],
                omega_m=omega_m,
            )
            shear_red_tot_WL = profs.shear_red_nfw(
                r_range, sigma_crit=sigma_crit[idx_z], **results_WL)
            shear_red_tot_WL_rs = profs.shear_red_nfw(
                r_range, sigma_crit=sigma_crit[idx_z], **results_WL_rs)
            shear_red_dmo = profs.shear_red_nfw(
                Robs, sigma_crit=sigma_crit[idx_z],
                m_x=results['m200m_dmo'][idx_z, idx_m],
                r_x=results['r200m_dmo'][idx_z, idx_m],
                c_x=results['c200m_dmo'][idx_z, idx_m],
            )

            results_new['m200m_WL'][idx_z, idx_m] = results_WL['m_x']
            results_new['r200m_WL'][idx_z, idx_m] = results_WL['r_x']
            results_new['c200m_WL'][idx_z, idx_m] = results_WL['c_x']
            results_new['m200m_WL_rs'][idx_z, idx_m] = results_WL_rs['m_x']
            results_new['r200m_WL_rs'][idx_z, idx_m] = results_WL_rs['r_x']
            results_new['c200m_WL_rs'][idx_z, idx_m] = results_WL_rs['c_x']
            results_new['shear_red_obs'][idx_z, idx_m] = shear_red_obs
            results_new['shear_red_dmo'][idx_z, idx_m] = shear_red_dmo
            results_new['shear_red_WL'][idx_z, idx_m] = shear_red_tot_WL
            results_new['shear_red_WL_rs'][idx_z, idx_m] = shear_red_tot_WL_rs

    results = {**results, **results_new}
    with asdf.AsdfFile(results) as af:
        fname_base = (
            f'{save_dir}/observational_results_fgas_r_{model_fname_append}'
            f'_R_{str(np.round(R_bins.min(), 2)).replace(".", "p")}'
            f'-{str(np.round(R_bins.max(), 2)).replace(".", "p")}'
        )
        if zscale:
            fname = f'{fname_base}_z_scaled.asdf'
        else:
            fname = f'{fname_base}.asdf'
        af.write_to(fname)

    return {**results, **results_new}
