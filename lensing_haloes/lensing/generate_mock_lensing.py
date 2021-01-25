import astropy.constants as constants
from astropy.cosmology import FlatwCDM
import astropy.units as u
import numpy as np
import scipy.integrate as intg
import scipy.interpolate as interp

import lensing_haloes.halo.profiles as profs
import lensing_haloes.util.tools as tools

import pdb


def n_mpch2(n_arcmin2, z_l, cosmo):
    """
    Convert a mean background galaxy density per arcmin^2 for a lens
    at redshift z_l to a density per (Mpc/h)^2 assuming cosmo

    Parameters
    ----------
    n_arcmin2 : float
        background galaxy density per arcmin^2
    z_l : array-like
        redshift of the lens
    cosmo : dictionary
        cosmological information, needs keywords
        - omega_m
        - h
        - w0

    Returns
    -------
    n_mpch2 : array-like
        background galaxy density per (Mpc/h)^2 for each z_l
    """
    c = FlatwCDM(Om0=cosmo["omega_m"], H0=100*cosmo["h"], w0=cosmo["w0"])
    # arcminute to Mpc/h conversion factor
    mpch_per_arcmin = (
        np.tan((1 * u.arcmin).to(u.rad)) *
        c.angular_diameter_distance(z_l) * c.h
    )

    # galaxy density is arcmin^-2
    n_mpch2 = (n_arcmin2 * mpch_per_arcmin**(-2)).value
    return n_mpch2


def shape_noise(R_bins, z_l, cosmo, sigma_e=0.25, n_arcmin2=10, log=False):
    """
    Calculate the uncertainty due to intrinsic shape noise for bin r_i
    due to the number of background galaxies

    Parameters
    ----------
    R_bins : (r,) array
        bin edges
    z_l : (z,) array
        redshift of the lens
    cosmo : dictionary
        cosmological information, needs keywords
        - omega_m
        - h
        - w0
    sigma_e : float
        ellipticity noise per galaxy
    n_arcmin2 : float
        background galaxy density [arcmin^-2]
    log : bool
        logarithmic bins

    Returns
    -------
    shape_noise : (z, r) array
        average shape noise for each R_bin
    """
    # (1, R) array or (z, R) array
    R_bins = np.atleast_2d(R_bins)
    R_centers = tools.bin_centers(R_bins, log=log)
    # (z, 1) array
    nmpch2 = n_mpch2(n_arcmin2=n_arcmin2, z_l=z_l, cosmo=cosmo).reshape(-1, 1)

    # (z, R) array
    N_bins = 2 * np.pi * nmpch2 * (np.diff(R_bins) * R_centers)
    sigma_bins = sigma_e / (N_bins)**0.5
    return sigma_bins


def sigma_critical(z_l, beta_mean, cosmo):
    """Return the critical surface mass density for a lens as z_l with
    mean lensing efficiency beta_mean.

    Parameters
    ----------
    z_l : array-like
        redshifts
    beta_mean : float
        mean lensing efficiency
    cosmo : dictionary
        cosmological information, needs keywords
        - omega_m
        - h
        - w0

    Returns
    -------
    sigma_crit : array-like
        critical surface mass density for each lens
    """
    c = FlatwCDM(Om0=cosmo["omega_m"], H0=100*cosmo["h"], w0=cosmo["w0"])
    # [(M_odot / h) / (Mpc / h)]
    alpha = (
        constants.c**2 / (4 * np.pi * constants.G)).to(u.solMass / u.Mpc)
    sigma_crit = (
        alpha * 1 / (beta_mean * c.angular_diameter_distance(z_l).to(u.Mpc) * c.h)
    ).value
    return sigma_crit


def R_to_theta(R, z, cosmo):
    """Convert the given projected radius in Mpc/h to the angular size
    for given cosmo.

    Parameters
    ----------
    z : array-like
        redshifts
    R : array-like
        projected radii [Mpc/h]
    cosmo : dictionary
        cosmological information, needs keywords
        - omega_m
        - h
        - w0

    Returns
    -------
    theta : array-like
        angular sizes [arcmin]
    """
    c = FlatwCDM(Om0=cosmo["omega_m"], H0=100*cosmo["h"], w0=cosmo["w0"])
    theta = (R * (u.Mpc / c.h) / c.angular_diameter_distance(z)) * u.rad

    return theta.to(u.arcmin).value


def theta_to_R(theta, z, cosmo, unit=u.arcmin):
    """Convert the given angular size with to a projected radius in Mpc/h
     for given cosmo.

    Parameters
    ----------
    theta : array-like
        angular sizes [Default: arcmin]
    z : array-like
        redshifts
    cosmo : dictionary
        cosmological information, needs keywords
        - omega_m
        - h
        - w0

    Returns
    -------
    R : array-like
        projected radii [Mpc / h]"""


    if not hasattr(theta, "unit"):
        theta = theta * unit

    c = FlatwCDM(Om0=cosmo["omega_m"], H0=100*cosmo["h"], w0=cosmo["w0"])
    R = (theta.to(u.rad).value * c.angular_diameter_distance(z) * c.h).value

    return R


def zeta_clowe_data(R_bins, R_t, gamma_t, R1, R2, Rmax, n_int=1000):
    """
    Compute the Clowe+1998 aperture mass statistic from the tangential shear
    gamma_t in bins R_t purely from the data.

        zeta_c(R1) = \int_R1^R2 \mean{gamma_t} dlnR + 1 / (1 - R2^2/Rmax^2)
                     \int_R2^Rmax \mean{gamma_t} dlnR

    Parameters
    ----------
    R_t : (R, ) array
        radial bins between R1 and Rmax [Mpc/h]
    gamma_t : (m, R) array
        shear in the bin
    R1 : float
        inner radius [Mpc/h]
    R2 : float
        annulus inner radius [Mpc/h]
    Rmax : float
        annulus outer radius [Mpc/h]
    n_int : int
        sampling points for integration range

    Returns
    -------
    zeta_c : (m, R) array
        Clowe+1998 aperture mass statistic
    """
    log10_gamma_int = interp.interp1d(
        R_t, np.log10(gamma_t),
        fill_value="extrapolate", axis=-1
    )

    if R1 == R2:
        zeta_R2 = 0
    else:
        # in case we have (m, ) arrays, we need to transpose to get (m, R) shape
        R1R2_range = np.linspace(R1, R2, n_int).T
        new_shape = tuple(np.ones(len(gamma_t.shape[:-1]), dtype=int)) + (n_int,)
        zeta_R2 = 2 * intg.simps(
            10**log10_gamma_int(R1R2_range) / R1R2_range.reshape(new_shape),
            x=R1R2_range.reshape(new_shape), axis=-1
        )

    if R2 == Rmax:
        zeta_Rmax = 0
    else:
        # in case we have (m, ) arrays, we need to transpose to get (m, R) shape
        R2Rm_range = np.linspace(R2, Rmax, n_int).T
        new_shape = tuple(np.ones(len(gamma_t.shape[:-1]), dtype=int)) + (n_int,)
        zeta_Rmax = 2 * intg.simps(
            10**log10_gamma_int(R2Rm_range) / R2Rm_range.reshape(new_shape),
            x=R2Rm_range.reshape(new_shape), axis=-1
        )

    zeta_Rmax *= 1. / (1 - R2**2 / Rmax**2)

    return zeta_R2 + zeta_Rmax


def sigma_mean_within_R2_Rmax_nfw(R2, Rmax, m_x, r_x, c_x, **kwargs):
    """
    Compute the boundary term of the Clowe+1998 aperture mass statistic assuming
    the best-fitting NFW profile parameters m_x, r_x and c_x.

    <Sigma>(R2 < R < Rmax) = 1 / (1 - (R2/Rmax)^2) <Sigma>(<Rmax) - 1 / ((Rmax/R2)**2 - 1) <Sigma>(<R2)

    Parameters
    ----------
    R2 : float
        inner radius of annulus [Mpc/h]
    Rmax : float
        outer radius of annulus [Mpc/h]
    m_x : (m, ) array
        best-fitting mass at overdensity radius r_x
    r_x : (m, ) array
        overdensity radius
    c_x : (m, ) array
        best-fitting concentration at r_x

    Returns
    -------
    error_R2 : (m, ) array
        boundary term of zeta statistic
    """
    sigma_mean_R2 = profs.sigma_mean_nfw(
        R2, m_x=m_x, r_x=r_x, c_x=c_x
    )
    sigma_mean_Rmax = profs.sigma_mean_nfw(
        Rmax, m_x=m_x, r_x=r_x, c_x=c_x
    )

    sigma_mean_error = (
        1. / (1 - (R2/Rmax)**2) * sigma_mean_Rmax
        - 1. / ((Rmax/R2)**2 - 1) * sigma_mean_R2)
    return sigma_mean_error


def M_ap_clowe_nfw(R_bins, R_obs, g_obs, R1, R2, Rmax, sigma_crit=1, n_int=1000, **NFW_kwargs):
    """
    Get the aperture mass M(<R) using the Clowe+1998 statistic.

    M(<R_1) = zeta_c(R_1) + kappa_NFW(R_2 < R < R_max)

    Compute the Clowe+1998 aperture mass statistic from the measured shear
    gamma_t in bins R_t purely from the data.

        zeta_c(R1) = \int_R1^R2 \mean{gamma_t} dlnR + 1 / (1 - R2^2/Rmax^2)
                     \int_R2^Rmax \mean{gamma_t} dlnR

    Parameters
    ----------
    R_bins : (R+1,) array
        radial bin edges between R1 and Rmax
    R_obs : (R, ) array
        radial bin centers
    g_obs : (m, R) array
        reduced shear in the bin
    R1 : float
        inner radius
    R2 : float
        annulus inner radius
    Rmax : float
        annulus outer radius
    n_int : int
        sampling points for integration range

    Returns
    -------
    M_R : (m, R) array
        Clowe+1998 aperture mass statistic
    """
    if R1 < R_bins.min():
        raise ValueError(f'R1 cannot be smaller than {R_bins.min()}')
    if R2 > R_bins.max():
        raise ValueError(f'R2 cannot be larger than {R_bins.max()}')
    if Rmax > R_bins.max():
        raise ValueError(f'Rmax cannot be larger than {R_bins.max()}')
    if R2 >= Rmax:
        raise ValueError('R2 cannot be larger than Rmax')

    NFW_kwargs_R = {
        'm_x': NFW_kwargs['m_x'][..., None],
        'r_x': NFW_kwargs['r_x'][..., None],
        'c_x': NFW_kwargs['c_x'][..., None],
    }

    gamma_obs = g_obs * (1 - profs.sigma_nfw(R=R_obs, **NFW_kwargs_R) / sigma_crit)
    zeta_c = zeta_clowe_data(
        R_bins=R_bins, R_t=R_obs, gamma_t=gamma_obs,
        R1=R1, R2=R2, Rmax=Rmax, n_int=n_int
    )

    sigma_R2_Rmax = sigma_mean_within_R2_Rmax_nfw(
        R2=R2, Rmax=Rmax, **NFW_kwargs
    )

    return (zeta_c * sigma_crit + sigma_R2_Rmax) * np.pi * R1**2


def sigma_mean_within_R2_Rmax(R2, Rmax, Rs, sigma, n_int=1000):
    """Compute the boundary term of the Clowe+1998 aperture mass
    statistic for the given surface mass density profile Sigma(Rs)

    <Sigma>(R2 < R < Rmax) = 1 / (1 - (R2/Rmax)^2) <Sigma>(<Rmax) - 1 / ((Rmax/R2)**2 - 1) <Sigma>(<R2)

    Parameters
    ----------
    R2 : float
        inner radius of annulus [Mpc/h]
    Rmax : float
        outer radius of annulus [Mpc/h]
    Rs : array-like
        radial range for sigma
    sigma : array-like
        enclosed surface mass density
    n_int : int
        sampling points for integration range

    Returns
    -------
    error_R2 : (m, ) array
        boundary term of zeta statistic

    """
    sigma_mean = profs.sigma_mean_from_sigma(
        R=np.array([R2, Rmax]), Rs=Rs, sigma=sigma, n_int=n_int
    )

    sigma_mean_R2 = sigma_mean[..., 0]
    sigma_mean_Rmax = sigma_mean[..., 1]

    sigma_mean_error = (
        1. / (1 - (R2/Rmax)**2) * sigma_mean_Rmax
        - 1. / ((Rmax/R2)**2 - 1) * sigma_mean_R2)

    return sigma_mean_error


def M_ap_clowe_sigma(
        R_bins, R_obs, g_obs, R1, R2, Rmax,
        Rs, sigma, sigma_crit=1, n_int=1000, **kwargs):
    """
    Get the aperture mass M(<R) using the Clowe+1998 statistic.

    M(<R_1) = zeta_c(R_1) + kappa_NFW(R_2 < R < R_max)

    Compute the Clowe+1998 aperture mass statistic from the measured shear
    gamma_t in bins R_t purely from the data.

        zeta_c(R1) = \int_R1^R2 \mean{gamma_t} dlnR + 1 / (1 - R2^2/Rmax^2)
                     \int_R2^Rmax \mean{gamma_t} dlnR

    Parameters
    ----------
    R_bins : (R+1,) array
        radial bin edges between R1 and Rmax
    g_obs : (m, R) array
        reduced shear in the bin
    R1 : float
        inner radius
    R2 : float
        annulus inner radius
    Rmax : float
        annulus outer radius
    Rs : array-like
        radial range for sigma
    sigma : array-like
        enclosed surface mass density
    sigma_crit : float
        critical surface mass density for the lens and source distribution
    n_int : int
        sampling points for integration range

    Returns
    -------
    M_R : (m, R) array
        Clowe+1998 aperture mass statistic
    """
    if R1 < R_bins.min():
        raise ValueError(f'R1 cannot be smaller than {R_bins.min()}')
    if R2 > R_bins.max():
        raise ValueError(f'R2 cannot be larger than {R_bins.max()}')
    if Rmax > R_bins.max():
        raise ValueError(f'Rmax cannot be larger than {R_bins.max()}')
    if R2 >= Rmax:
        raise ValueError('R2 cannot be larger than Rmax')

    kappa_obs = profs.kappa_from_sigma(
        R=R_obs, Rs=Rs, sigma=sigma, sigma_crit=sigma_crit
    )
    gamma_obs = g_obs * (1 - kappa_obs)
    zeta_c = zeta_clowe_data(
        R_bins=R_bins, R_t=R_obs, gamma_t=gamma_obs,
        R1=R1, R2=R2, Rmax=Rmax, n_int=n_int
    )

    sigma_R2_Rmax = sigma_mean_within_R2_Rmax(
        R2=R2, Rmax=Rmax, Rs=Rs, sigma=sigma, n_int=n_int
    )

    return (zeta_c * sigma_crit + sigma_R2_Rmax) * np.pi * R1**2


def observed_reduced_shear(
        R_bins, R, sigma_tot, sigma_crit,
        n_int=1000, upsample=20):
    """
    Return the observed reduced shear at R for sigma_tot .

    Parameters
    ----------
    R_bins : (n + 1) array-like
        observed bin edges
    R : (r) array
        radial range for sigma_tot
    sigma_tot : (r) array
        total model surface mass density
    n_int : int
        number of interpolating steps in integration
    upsample : int
        number of extra samples per bin for average

    Returns
    -------
    shear_red_obs : (n) array
        average reduced shear in each R_bin
    """
    R_obs = tools.bin_centers(R_bins)
    shear_red_obs = tools.avg_in_bins(
        function=profs.shear_red_from_sigma,
        bins=R_bins, upsample=upsample, log=False,
        **{
            "Rs": R,
            "sigma": sigma_tot,
            "sigma_crit": sigma_crit,
            "n_int": n_int
        }
    )

    return shear_red_obs
