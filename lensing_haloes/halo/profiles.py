import numpy as np
import mpmath as mp
import scipy.special as spec
from scipy import integrate as intg
from scipy import interpolate as interp

import lensing_haloes.util.tools as tools

from pdb import set_trace


def profile_nfw(r_range, m_x, c_x, r_x, **kwargs):
    """
    Returns an NFW profile with mass m_x and concentration c_x at r_x.

    Shapes of all inputs must match.

    Parameters
    ----------
    r_range : array
        array containing r_range for each m
    m_x : array
        array containing mass inside r_x
    c_x : array
        array containing mass-concentration relation
    r_x : array
        array containing r_x to evaluate r_s from r_s = r_x/c_x

    Returns
    -------
    profile : array
        array containing NFW profile
    """
    r_s = r_x / c_x
    rho_s = (m_x / (4. * np.pi * r_x**3) * c_x**3 /
             (np.log(1+c_x) - c_x/(1+c_x)))

    x = r_range / r_s

    profile = rho_s / (x * (1+x)**2)
    return profile


@np.vectorize
def profile_nfw_rmax(r_range, r_max, m_x, c_x, r_x, **kwargs):
    """
    Returns an NFW profile with mass m_x and concentration c_x at r_x.

    Shapes of all inputs must match.

    Parameters
    ----------
    r_range : array
        radial range for each m
    r_max : array
        maximum radius for each m
    m_x : array
        mass inside r_x
    c_x : array
        concentration
    r_x : array
        r_x to evaluate r_s from r_s = r_x/c_x

    Returns
    -------
    profile : array
        array containing NFW profile
    """
    if r_range > r_max:
        return 0
    r_s = r_x / c_x
    rho_s = (m_x / (4. * np.pi * r_x**3) * c_x**3 /
             (np.log(1+c_x) - c_x/(1+c_x)))

    x = r_range / r_s

    profile = rho_s / (x * (1+x)**2)
    return profile


@np.vectorize
def profile_nfw_rmin_rmax(r_range, r_min, r_max, m_x, c_x, r_x, rho_scale, **kwargs):
    """
    Returns an NFW profile with mass m_x and concentration c_x at r_x.

    Shapes of all inputs must match.

    Parameters
    ----------
    r_range : array
        radial range for each m
    r_min : array
        minimum radius for each m
    r_max : array
        maximum radius for each m
    m_x : array
        mass inside r_x
    c_x : array
        concentration
    r_x : array
        r_x to evaluate r_s from r_s = r_x/c_x
    rho_scale : array
        scale factor for the density profile
    Returns
    -------
    profile : array
        array containing NFW profile
    """
    if r_range < r_min or r_range > r_max:
        return 0
    r_s = r_x / c_x
    rho_s = (
        rho_scale * m_x / (4. * np.pi * r_x**3) * c_x**3 /
        (np.log(1+c_x) - c_x/(1+c_x))
    )

    x = r_range / r_s

    profile = rho_s / (x * (1+x)**2)
    return profile


@np.vectorize
def sigma_nfw_fit(R, log10_mx, r_s, rho_x, **kwargs):
    """Return the surface mass density profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    log10_mx : float
        log_10 halo mass inside r_x [h^-1 M_sun]
    r_s : float
        physical scale radius [h^-1 Mpc]
    rho_x : float
        mean overdensity at r_x [h^2 M_sun/Mpc^3]

    Returns
    -------
    sigma_nfw : array
        surface mass density of NFW profile at projected radius R
    """
    m_x = 10**log10_mx
    r_x = tools.mass_to_radius(m_x, rho_x)
    c_x = r_x / r_s
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = 2 * r_s * rho_s

    if R == r_s:
        sigma = 1. / 3 * prefactor

    elif R < r_s:
        x = R / r_s
        prefactor *= 1. / (x**2 - 1)
        sigma = prefactor * (1 - 2 / np.sqrt(1 - x**2) *
                             np.arctanh(np.sqrt((1 - x) / (1 + x))))

    else:
        x = R / r_s
        prefactor *= 1. / (x**2 - 1)
        sigma = prefactor * (1 - 2 / np.sqrt(x**2 - 1) *
                             np.arctan(np.sqrt((x - 1) / (x + 1))))

    return sigma


@np.vectorize
def sigma_nfw(R, m_x, r_x, c_x, **kwargs):
    """Return the surface mass density profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : (m,) array
        array containing mass inside r_x
    c_x : (m,) or (m, z) array
        array containing mass-concentration relation
    r_x : (m,) or (m, z) array
        array containing r_x to evaluate r_s from r_s = r_x/c_x

    Returns
    -------
    sigma_nfw : array
        surface mass density of NFW profile at projected radius R
    """
    r_s = r_x / c_x
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = 2 * r_s * rho_s

    if R == r_s:
        sigma = 1. / 3 * prefactor

    elif R < r_s:
        x = R / r_s
        prefactor *= 1. / (x**2 - 1)
        sigma = prefactor * (1 - 2 / np.sqrt(1 - x**2) *
                             np.arctanh(np.sqrt((1 - x) / (1 + x))))

    else:
        x = R / r_s
        prefactor *= 1. / (x**2 - 1)
        sigma = prefactor * (1 - 2 / np.sqrt(x**2 - 1) *
                             np.arctan(np.sqrt((x - 1) / (x + 1))))

    return sigma


@np.vectorize
def sigma_nfw_rmax(R, r_max, m_x, r_x, c_x, **kwargs):
    """Return the surface mass density profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x that terminates at r_max.

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    r_max : (m,) array
        maximum radius for the NFW profile
    m_x : (m,) array
        array containing mass inside r_x
    c_x : (m,) or (m, z) array
        array containing mass-concentration relation
    r_x : (m,) or (m, z) array
        array containing r_x to evaluate r_s from r_s = r_x/c_x

    Returns
    -------
    sigma_nfw : array
        surface mass density of NFW profile at projected radius R
    """
    r_s = r_x / c_x
    rho_s = (
        m_x / (4 * np.pi * r_x**3) * c_x**3
        / (np.log(1 + c_x) - c_x / (1 + c_x))
    )

    prefactor = 2 * r_s * rho_s

    if R == r_s:
        sigma = (
            prefactor * (r_max - r_s) * (2 * r_s + r_max)
            / (3 * (r_s + r_max) * (r_max**2 - r_s**2)**0.5))

    elif R < r_s:
        x = R / r_s
        prefactor *= 1. / (1 - x**2)**1.5
        sigma = (
            prefactor * (
                np.log(
                    x * (r_s + r_max) / (
                        r_max + r_s * x**2 - ((1 - x**2) * (r_max**2 - r_s**2 * x**2))**0.5
                    )
                ) - ((1 - x**2) * (r_max**2 - r_s**2 * x**2))**0.5 / (r_s + r_max)
            )
        )

    elif R > r_s and R < r_max:
        x = R / r_s
        prefactor *= 1. / (x**2 - 1)**1.5
        sigma = (
            prefactor * (
                -np.arctan(
                    ((r_max**2 - r_s**2 * x**2) * (x**2 - 1))**0.5
                    / (r_max + r_s * x**2)
                )
                + ((x**2 - 1) * (r_max**2 - r_s**2 * x**2))**0.5
                / (r_s + r_max)
            )
        )

    else:
        sigma = 0.

    return sigma


@np.vectorize
def sigma_nfw_rmin_rmax(R, r_min, r_max, m_x, r_x, c_x, rho_scale, **kwargs):
    """Return the surface mass density profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x that starts at r_min and
    terminates at r_max.

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    r_min : (m,) array
        minimum radius for the NFW profile
    r_max : (m,) array
        maximum radius for the NFW profile
    m_x : (m,) array
        array containing mass inside r_x
    c_x : (m,) or (m, z) array
        array containing mass-concentration relation
    r_x : (m,) or (m, z) array
        array containing r_x to evaluate r_s from r_s = r_x/c_x
    rho_scale : (m,) or (m, z) array
        factor to rescale density profile by

    Returns
    -------
    sigma_nfw : array
        surface mass density of NFW profile at projected radius R

    """
    if r_min < R:
        return sigma_nfw_rmax(R=R, r_max=r_max, m_x=rho_scale*m_x, r_x=r_x, c_x=c_x)

    r_s = r_x / c_x
    rho_s = (
        rho_scale * m_x / (4 * np.pi * r_x**3) * c_x**3
        / (np.log(1 + c_x) - c_x / (1 + c_x))
    )

    prefactor = 2 * r_s * rho_s
    a = r_min / r_s
    b = r_max / r_s
    c = R / r_s

    sigma = prefactor * (
        - 1. / (2 * (1 + a) * (1 + b) * (1 - c**2)**1.5) * (
            + (2 + 2 * a) * ((c**2 - b**2) * (c**2 - 1))**0.5
            - (2 + 2 * b) * ((c**2 - a**2) * (c**2 - 1))**0.5
            + (1 + a) * (1 + b) * np.log(
                (((c**2 - a**2) * (c**2 - 1))**0.5 + (a + c**2))
                / (((c**2 - a**2) * (c**2 - 1))**0.5 - (a + c**2))
                * (((c**2 - b**2) * (c**2 - 1))**0.5 - (b + c**2))
                / (((c**2 - b**2) * (c**2 - 1))**0.5 + (b + c**2))
            )
        )
    )

    return sigma


@np.vectorize
def shear_nfw_fit(R, log10_mx, r_s, rho_x, sigma_crit=1, **kwargs):
    """Return the surface mass density profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    log10_mx : float
        log_10 halo mass inside r_x [h^-1 M_sun]
    r_s : float
        physical scale radius [h^-1 Mpc]
    rho_x : float
        mean overdensity at r_x [h^2 M_sun/Mpc^3]

    Returns
    -------
    shear_nfw : array
        surface mass density of NFW profile at projected radius R
    """
    m_x = 10**log10_mx
    r_x = tools.mass_to_radius(m_x, rho_x)
    c_x = r_x / r_s
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = r_s * rho_s / sigma_crit

    if R == r_s:
        shear = prefactor * (10. / 3 + 4 * np.log(0.5))

    elif R < r_s:
        x = R / r_s
        f_atanh = np.arctanh(np.sqrt((1 - x) / (1 + x)))
        g = ((8 * f_atanh / (x**2 * np.sqrt(1 - x**2)))
             + 4. / x**2 * np.log(0.5 * x)
             - 2. / (x**2 - 1)
             + 4 * f_atanh / ((x**2 - 1) * np.sqrt(1 - x**2)))
        shear = prefactor * g

    else:
        x = R / r_s
        f_atan = np.arctan(np.sqrt((x - 1) / (x + 1)))
        g = ((8 * f_atan / (x**2 * np.sqrt(x**2 - 1)))
             + 4. / x**2 * np.log(0.5 * x)
             - 2. / (x**2 - 1)
             + 4 * f_atan / ((x**2 - 1)**(1.5)))
        shear = prefactor * g

    return shear


@np.vectorize
def shear_nfw(R, m_x, r_x, c_x, sigma_crit=1, **kwargs):
    """Return the shear profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : (m,) array
        array containing mass inside r_x
    c_x : (m,) or (m, z) array
        array containing mass-concentration relation
    r_x : (m,) or (m, z) array
        array containing r_x to evaluate r_s from r_s = r_x/c_x
    sigma_crit : (m,) array or (m, z) array or float
        critical surface mass density of the observed systems

    Returns
    -------
    shear_nfw : array
        shear of NFW profile at projected radius R
    """
    r_s = r_x / c_x
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = r_s * rho_s / sigma_crit

    if R == r_s:
        shear = prefactor * (10. / 3 + 4 * np.log(0.5))

    elif R < r_s:
        x = R / r_s
        f_atanh = np.arctanh(np.sqrt((1 - x) / (1 + x)))
        g = ((8 * f_atanh / (x**2 * np.sqrt(1 - x**2)))
             + 4. / x**2 * np.log(0.5 * x)
             - 2. / (x**2 - 1)
             + 4 * f_atanh / ((x**2 - 1) * np.sqrt(1 - x**2)))
        shear = prefactor * g

    else:
        x = R / r_s
        f_atan = np.arctan(np.sqrt((x - 1) / (x + 1)))
        g = ((8 * f_atan / (x**2 * np.sqrt(x**2 - 1)))
             + 4. / x**2 * np.log(0.5 * x)
             - 2. / (x**2 - 1)
             + 4 * f_atan / ((x**2 - 1)**(1.5)))
        shear = prefactor * g

    return shear


@np.vectorize
def shear_red_nfw(R, m_x, r_x, c_x, sigma_crit=1, **kwargs):
    """Return the reduced shear profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : (m,) array
        array containing mass inside r_x
    c_x : (m,) or (m, z) array
        array containing mass-concentration relation
    r_x : (m,) or (m, z) array
        array containing r_x to evaluate r_s from r_s = r_x/c_x
    sigma_crit : (m,) array or (m, z) array or float
        critical surface mass density of the observed systems

    Returns
    -------
    shear_red_nfw : array
        reduced shear of NFW profile at projected radius R
    """
    if "sigma" not in kwargs:
        sigma = sigma_nfw(R=R, m_x=m_x, r_x=r_x, c_x=c_x, **kwargs)
        kappa = sigma / sigma_crit
    else:
        sigma = kwargs["sigma"]
        kappa = sigma / sigma_crit

    shear = shear_nfw(R=R, m_x=m_x, r_x=r_x, c_x=c_x, sigma_crit=sigma_crit)
    return shear / (1 - kappa)


@np.vectorize
def sigma_mean_nfw(R, m_x, r_x, c_x, **kwargs):
    """Return the mean surface mass density  profile of an NFW halo with mass
    m_x, radius r_x and concentration c_x within R

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : (m,) array
        array containing mass inside r_x
    c_x : (m,) or (m, z) array
        array containing mass-concentration relation
    r_x : (m,) or (m, z) array
        array containing r_x to evaluate r_s from r_s = r_x/c_x

    Returns
    -------
    sigma_mean : array
        mean enclosed surface mass NFW profile at projected radius R
    """
    r_s = r_x / c_x
    rho_s = m_x / (4 * np.pi * r_x**3) * c_x**3 / (np.log(1 + c_x) -
                                                   c_x / (1 + c_x))

    prefactor = 4 * r_s * rho_s

    if R == r_s:
        sigma_mean = prefactor * (1. + np.log(0.5))

    elif R < r_s:
        x = R / r_s
        prefactor *= 1. / x**2

        f_atanh = np.arctanh(np.sqrt((1 - x) / (1 + x)))
        s_m = ((2 * f_atanh / np.sqrt(1 - x**2))
               + np.log(0.5 * x))
        sigma_mean = prefactor * s_m

    else:
        x = R / r_s
        prefactor *= 1. / x**2

        f_atan = np.arctan(np.sqrt((x - 1) / (x + 1)))
        s_m = ((2 * f_atan / np.sqrt(x**2 - 1))
               + np.log(0.5 * x))
        sigma_mean = prefactor * s_m

    return sigma_mean


@np.vectorize
def m_nfw(r, m_x, r_x, c_x, **kwargs):
    """
    Calculate the mass of the NFW profile with c_x and r_x and m_x at r_x

    Parameters
    ----------
    r : float
        radius to compute mass for
    m_x : float
        mass inside r_x
    r_x : float
        r_x to evaluate r_s from r_s = r_x/c_x
    c_x : float
        concentration of halo

    Returns
    -------
    m_h : float
        mass
    """
    rho_s = m_x / (4. * np.pi * r_x**3) * c_x**3/(np.log(1+c_x) - c_x/(1+c_x))
    r_s = (r_x / c_x)

    prefactor = 4 * np.pi * rho_s * r_s**3
    c_factor = np.log((r_s + r) / r_s) - r / (r + r_s)

    mass = prefactor * c_factor

    return mass


@np.vectorize
def m_nfw_rmax(r, r_max, m_x, r_x, c_x, **kwargs):
    """
    Calculate the mass of the NFW profile with c_x and r_x and m_x at r_x

    Parameters
    ----------
    r : float
        radius to compute mass for
    m_x : float
        mass inside r_x
    r_x : float
        r_x to evaluate r_s from r_s = r_x/c_x
    c_x : float
        concentration of halo

    Returns
    -------
    m_h : float
        mass
    """
    if r > r_max:
        return 0.
    rho_s = m_x / (4. * np.pi * r_x**3) * c_x**3/(np.log(1+c_x) - c_x/(1+c_x))
    r_s = (r_x / c_x)

    prefactor = 4 * np.pi * rho_s * r_s**3
    c_factor = np.log((r_s + r) / r_s) - r / (r + r_s)

    mass = prefactor * c_factor

    return mass


@np.vectorize
def m_nfw_rmin_rmax(
        r, r_min, r_max, m_x, r_x, c_x, rho_scale,
        include_m_rmin=True,
        **kwargs):
    """
    Calculate the mass of the NFW profile with c_x and r_x and m_x at r_x

    Parameters
    ----------
    r : float
        radius to compute mass for
    r_min : (m,) array
        minimum radius for the NFW profile
    r_max : (m,) array
        maximum radius for the NFW profile
    m_x : float
        mass inside r_x
    r_x : float
        r_x to evaluate r_s from r_s = r_x/c_x
    c_x : float
        concentration of halo
    rho_scale : (m,) or (m, z) array
        factor to rescale density profile by

    Returns
    -------
    m_h : float
        mass
    """
    if r <= r_min:
        return 0.
    if r > r_max:
        return np.nan

    r_s = (r_x / c_x)
    rho_s = (
        rho_scale * m_x / (4. * np.pi * r_x**3) * c_x**3
        / (np.log(1+c_x) - c_x/(1+c_x))
    )
    prefactor = 4 * np.pi * rho_s * r_s**3

    m_r = prefactor * (np.log((r_s + r) / r_s) - r / (r + r_s))

    if include_m_rmin:
        return m_r
    else:
        m_rmin = prefactor * (np.log(1 + r_min / r_s) - r_min / (r_s + r_min)**-1.)
        return m_r - m_rmin


def profile_beta(r_range, m_x, r_x, r_c, beta, **kwargs):
    """
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : array
        array containing r_range for each m
    m_x : array
        array containing masses to match at r_x
    r_x : array
        x overdensity radius to match m_x at, in units of r_range
    beta : array
        power law slope of profile
    r_c : array
        physical core radius of beta profile in as a fraction

    Returns
    -------
    profile : array
        array containing beta profile
    """
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(
        3./2, 3. * beta / 2, 5./2, -(r_x / r_c)**2))

    profile = rho_0 / (1 + (r_range / r_c)**2)**(3*beta/2)

    return profile


def sigma_beta(R, m_x, r_x, r_c, beta, **kwargs):
    """
    Return a beta profile with mass m_x inside r_range <= r_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : array
        array containing masses to match at r_x
    r_x : array
        x overdensity radius to match m_x at, in units of r_range
    beta : array
        power law slope of profile
    r_c : array
        physical core radius of beta profile in as a fraction

    Returns
    -------
    sigma : array
        array containing beta surface mass density
    """
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(
        3./2, 3. * beta / 2, 5./2, -(r_x / r_c)**2))

    prefactor = np.pi**0.5 * r_c * rho_0
    sigma = prefactor * (
        (((R/r_c)**2 + 1)**(0.5 - 3 * beta / 2) *
         spec.gamma(3 * beta / 2 - 0.5)) / spec.gamma(3 * beta / 2))

    return sigma


def sigma_beta_rmax(R, r_max, m_x, r_x, r_c, beta, **kwargs):
    """Return a beta profile up to r_max with mass m_x inside r_range <=
    r_x.

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    r_max : array
        maximum radius of profile
    m_x : array
        array containing masses to match at r_x
    r_x : array
        x overdensity radius to match m_x at, in units of r_range
    beta : array
        power law slope of profile
    r_c : array
        physical core radius of beta profile in as a fraction

    Returns
    -------
    sigma : array
        array containing beta surface mass density
    """
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (
        4./3 * np.pi * r_x**3 * spec.hyp2f1(
            1.5, 1.5 * beta, 2.5, -(r_x / r_c)**2
        )
    )

    prefactor = 2 * r_c * rho_0
    sigma = prefactor * (
        ((r_max / r_c)**2 - (R / r_c)**2)**0.5
        / (1 + (R / r_c)**2)**(1.5 * beta)
        * spec.hyp2f1(
            0.5, 1.5 * beta, 1.5,
            -(((r_max / r_c)**2 - (R/r_c)**2) / (1 + (R / r_c)**2))
        )
    ).real

    return sigma.astype(float)


@np.vectorize
def shear_beta(R, m_x, r_x, r_c, beta, sigma_crit=1, **kwargs):
    """
    Return a beta profile with mass m_x inside r_range <= r_x

    Parameters
    ----------
    r_range : array
        array containing r_range for each m
    m_x : array
        array containing masses to match at r_x
    r_x : array
        x overdensity radius to match m_x at, in units of r_range
    beta : array
        power law slope of profile
    r_c : array
        physical core radius of beta profile in as a fraction
    sigma_crit : array
        critical surface mass density of the observed systems

    Returns
    -------
    shear : array
        shear for a beta profile
    """
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(
        1.5, 1.5 * beta, 2.5, -(r_x / r_c)**2))

    prefactor = (np.pi**0.5 * r_c * rho_0 / sigma_crit)
    x2 = (R / r_c)**2

    if beta != 1:
        prefactor *= spec.gamma(1.5 * beta - 0.5) / spec.gamma(1.5 * beta)
        f = (
            1. / (1.5 - 1.5 * beta) * 1. / x2 *
             ((1 + x2)**(1.5 - 1.5 * beta) - 1) -
             (1 + x2)**(0.5 - 1.5 * beta)
        )

        shear = prefactor * f

    else:
        prefactor *= 2. / np.pi**0.5
        f = 1. / x2 * np.log(1 + x2) - 1. / (1 + x2)

        shear = prefactor * f

    return shear


@np.vectorize
def sigma_mean_beta(R, m_x, r_x, r_c, beta, **kwargs):
    """Return a mean enclosed surface mass density of a beta profile with
    mass m_x inside r_range <= r_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : array
        array containing masses to match at r_x
    r_x : array
        x overdensity radius to match m_x at, in units of r_range
    beta : array
        power law slope of profile
    r_c : array
        physical core radius of beta profile in as a fraction

    Returns
    -------
    sigma_mean : array
        mean enclosed surface mass density for a beta profile

    """
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(
        1.5, 1.5 * beta, 2.5, -(r_x / r_c)**2))

    x2 = (R / r_c)**2

    if beta != 1:
        prefactor = (
            np.pi**0.5 * r_c * rho_0 * 1. / x2 *
            spec.gamma(1.5 * beta - 0.5) /
            spec.gamma(1.5 * beta))
        f = 1. / (1.5 - 1.5 * beta) * ((1 + x2)**(1.5 - 1.5 * beta) - 1)

        sigma_mean = prefactor * f

    else:
        prefactor = (2 * r_c * rho_0 * 1. / x2)
        f = np.log(1 + x2)

        sigma_mean = prefactor * f

    return sigma_mean


@np.vectorize
def m_beta(r, m_x, r_x, r_c, beta, **kwargs):
    """
    Return the analytic enclosed mass for the beta profile normalized to
    m_x at r_x

    Parameters
    ----------
    r : float
        radius to compute for
    m_x : float
        gas mass at r500c
    r_x : float
        physical radius corresponding to halo mass m500c
    r_c : float
        physical core radius r_c of the profile
    beta : float
        beta slope of the profile
    """
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(
        3./2, 3 * beta / 2, 5./2, -(r_x / r_c)**2))

    m = 4./3 * np.pi * rho_0 * r**3 * spec.hyp2f1(
        3./2, 3 * beta / 2, 5./2, -(r/r_c)**2)

    return m


def profile_plaw(r_range, rho_x, r_x, gamma, **kwargs):
    """
    Return a power law profile with density rho_x at r_x that decays with a
    power law slope of gamma

        rho(r|m) = rho_x(m) (r_range / r_x)**(-gamma)

    Parameters
    ----------
    r_range : array
        array containing r_range for each m
    rho_x : array
        array containing density to match at r_x
    r_x : array
        radius to match rho_x at, in units of r_range
    gamma : float
        power law slope of profile

    Returns
    -------
    profile : array
        array containing beta profile
    """
    profile = rho_x * (r_range / r_x)**(-gamma)
    profile[(r_range < r_x)] = 0.

    return profile


@np.vectorize
def m_plaw(r, rho_x, r_x, gamma, **kwargs):
    """
    Return the analytic enclosed mass for the power law profile

    Parameters
    ----------
    r : float
        radius to compute for
    rho_x : float
        density to match at r_x
    r_x : float
        radius to match rho_x at, in units of r_range
    gamma : float
        power law slope of profile
    """
    if r < r_x:
        m = 0.

    else:
        prefactor = 4 * np.pi * rho_x * r_x**3
        if gamma == 3:
            m = prefactor * np.log(r / r_x)
        else:
            m = prefactor * ((r/r_x)**(3 - gamma) - 1) / (3 - gamma)

    return m


@np.vectorize
def profile_beta_plaw(r_range, m_x, r_x, r_c, beta, gamma, rho_x=None, **kwargs):
    """
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : array
        array containing r_range for each m
    m_x : array
        array containing masses to match at r_x
    r_x : array
        x overdensity radius to match m_x at, in units of r_range
    beta : array
        power law slope of profile
    r_c : array
        physical core radius of beta profile
    gamma : array
        power law index

    Returns
    -------
    profile : array
        array containing beta profile
    """
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(
        3./2, 3 * beta / 2, 5./2, -(r_x / r_c)**2))

    if rho_x is None:
        rho_x = rho_0 / (1 + (r_x / r_c)**2)**(3 * beta / 2)

    if r_range <= r_x:
        profile = (rho_0 / (1 + (r_range / r_c)**2)**(3 * beta / 2))
    else:
        profile = (rho_x * (r_range / r_x)**(-gamma))

    return profile


@np.vectorize
def profile_beta_plaw_rmax(r_range, r_max, m_x, r_x, r_c, beta, gamma, rho_x=None, **kwargs):
    """
    Return a beta profile with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r_range : array
        array containing r_range for each m
    r_max : array
        maximum radius for each m
    m_x : array
        array containing masses to match at r_x
    r_x : array
        x overdensity radius to match m_x at, in units of r_range
    beta : array
        power law slope of profile
    r_c : array
        physical core radius of beta profile
    gamma : array
        power law index

    Returns
    -------
    profile : array
        array containing beta profile
    """
    if r_range > r_max:
        return 0
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(
        3./2, 3 * beta / 2, 5./2, -(r_x / r_c)**2))

    if rho_x is None:
        rho_x = rho_0 / (1 + (r_x / r_c)**2)**(3 * beta / 2)

    if r_range <= r_x:
        profile = (rho_0 / (1 + (r_range / r_c)**2)**(3 * beta / 2))
    else:
        profile = (rho_x * (r_range / r_x)**(-gamma))

    return profile


@np.vectorize
def sigma_beta_plaw(R, m_x, r_x, r_c, beta, gamma, **kwargs):
    """Return the surface mass density of a beta profile with mass m_x
    inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    R : (m, r) array
        array containing R for each m
    m_x : float
        mass inside r_x
    r_x : float
        radius to match rho_x at, in units of r_range
    r_c : float
        physical core radius r_c of the profile
    beta : float
        beta slope of the profile
    gamma : float
        power law slope of profile

    Returns
    -------
    sigma : array
        surface mass density of beta plaw profile

    """
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(
        1.5, 1.5 * beta, 2.5, -(r_x / r_c)**2))
    rho_x = rho_0 / (1 + (r_x / r_c)**2)**(1.5 * beta)

    if (gamma <= 1):
        raise ValueError("for gamma <= 1 the profile diverges.")

    elif gamma == 2:
        if R <= r_x:
            a = R / r_x
            sigma_beta_rx = sigma_beta_rmax(
                R=R, r_max=r_x, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta
            )
            sigma_gamma_rx = np.arcsin(a) / a
            sigma = sigma_beta_rx + sigma_gamma_rx
        else:
            a = R / r_x
            sigma = 0.5 * np.pi / a

    else:
        if R < r_x:
            a = R / r_x
            sigma_beta_rx = sigma_beta_rmax(
                R=R, r_max=r_x, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta
            )
            sigma_gamma_rx = 2 * rho_x * r_x * (
                0.5 / np.pi**0.5 * (-1j * a)**(1 - gamma)
                * mp.gamma(1 - 0.5 * gamma) * mp.gamma(0.5 * (gamma - 1))
                + 1j / (a * (gamma - 2)) * mp.hyp2f1(0.5, 1 - 0.5 * gamma, 2 - 0.5 * gamma, a**(-2))
            ).real
            sigma = sigma_beta_rx + sigma_gamma_rx

        elif R == r_x:
            sigma = 2 * rho_x * r_x * (
                0.5 * np.pi**0.5 * mp.gamma(0.5 * (gamma - 1)) / mp.gamma(0.5 * gamma)
            ).real
        else:
            a = R / r_x
            sigma = np.pi**0.5 * rho_x * r_x * a**(1 - gamma) * (
                mp.gamma(0.5 * (gamma - 1)) / mp.gamma(0.5 * gamma)
            ).real

    return float(sigma)


@np.vectorize
def sigma_beta_plaw_rmax(R, r_max, m_x, r_x, r_c, beta, gamma, **kwargs):
    """Return the surface mass density of a beta profile with mass m_x
    inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    R : (m, r) array
        array containing R for each m
    m_x : float
        mass inside r_x
    r_x : float
        radius to match rho_x at, in units of r_range
    r_c : float
        physical core radius r_c of the profile
    beta : float
        beta slope of the profile
    gamma : float
        power law slope of profile

    Returns
    -------
    sigma : array
        surface mass density of beta plaw profile

    """
    # analytic enclosed mass inside r_x gives normalization rho_0
    rho_0 = m_x / (4./3 * np.pi * r_x**3 * spec.hyp2f1(
        1.5, 1.5 * beta, 2.5, -(r_x / r_c)**2))
    rho_x = rho_0 / (1 + (r_x / r_c)**2)**(1.5 * beta)

    if (gamma <= 1):
        raise ValueError("for gamma <= 1 the profile diverges.")

    if R <= r_x:
        a = R / r_x
        b = r_max / r_x
        sigma_beta_rx = sigma_beta_rmax(
            R=R, r_max=r_x, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta
        )
        # needs minus sign
        sigma_gamma_rx = 2 * rho_x * r_x * (
            1. / (a * (2 - gamma)) * (
                1j * b**(2 - gamma) * mp.hyp2f1(0.5, 1 - 0.5 * gamma, 2 - 0.5 * gamma, (b / a)**2 + 0j)
                - 1j * mp.hyp2f1(0.5, 1 - 0.5 * gamma, 2 - 0.5 * gamma, a**(-2) + 0j)
            ).real
        )
        sigma = sigma_beta_rx + sigma_gamma_rx

    elif R > r_x and R < r_max:
        a = R / r_x
        b = r_max / r_x
        if gamma == 3:
            sigma = 2 * rho_x * r_x * (b**2 - a**2)**0.5 / (b * a**2)
        else:
            sigma = rho_x * r_x * (
                np.pi**0.5 * a**(1 - gamma) * spec.gamma(0.5 * (gamma - 1)) / spec.gamma(0.5 * gamma)
                + b**(1 - gamma) * spec.gamma(0.5 * (1 - gamma)) / spec.gamma(1.5 - 0.5 * gamma)
                * spec.hyp2f1(0.5, 0.5 * (gamma - 1), 0.5 * (gamma + 1), (a / b)**2)
            )
    else:
        sigma = 0


    return float(sigma)


def mr_from_rho(r, rs, rho, n_int=1000, dlog10r=0):
    """Return the mean enclosed mass within r from rho.

    Parameters
    ----------
    r : (r,) array
        radial range
    rs : (..., r) array
        radial range for each rho
    rho : (..., r) array
        density profiles
    n_int : int
        number of interpolation steps in integration
    dlog10r : float
        logarithmic decrement with respect to rs.min() to start integration
        from

    Returns
    -------
    sigma_mean : array
        mean surface mass density of sigma

    """
    log10_rho_int = interp.interp1d(
        np.log10(rs), np.log10(rho),
        axis=-1, fill_value="extrapolate"
    )

    # array that runs from 10^-5 to R for each dimension, need to move
    # iteration axis to the final one
    r_int = np.moveaxis(
        np.logspace(np.log10(rs.min()) - dlog10r, np.log10(r), n_int),
        0, -1
    )
    log10_r_int = np.moveaxis(
        np.linspace(np.log10(rs.min()) - dlog10r, np.log10(r), n_int),
        0, -1
    )

    rho_int = 10**log10_rho_int(log10_r_int)
    rho_int[np.isnan(rho_int)] = 0.

    # need to reshape R_int to match sigma_int
    if len(rho_int.shape) >= 2:
        shape = tuple(np.ones(len(rho_int.shape) - 2, dtype=int)) + (-1, n_int)
        r_int = r_int.reshape(shape)

    mr = 4 * np.pi * intg.simps(rho_int * r_int**2, x=r_int, axis=-1)

    return mr


def sigma_from_rho(R, rho_func, R_max=None):
    """Return the surface mass within R from rho_func.

    Gauss-Jacobi quadrature is used to integrate the singularity at r=R.

    Parameters
    ----------
    R : array-like
        radial range
    rho_func : callable
        density function that takes r as argument

    Returns
    -------
    sigma : array-like
        surface mass density

    """
    if R_max is None:
        R_max = 10**(np.log10(R.max()) + 5)
    # define the integrand for the quadrature
    def integrand(r, R): return 2 * rho_func(r) * r / np.sqrt(r + R)

    sigma = np.empty_like(R, dtype=float)
    for idx, rr in enumerate(R):
        # use Gauss-Jacobi integration to get around singularity at r=R
        s, sigma_err = intg.quad(
            integrand, rr, R_max, args=(rr,),
            weight='alg', wvar=(-0.5, 0)
        )
        sigma[idx] = s

    return sigma


def kappa_from_sigma(R, Rs, sigma, sigma_crit, n_int=1000):
    """Return the convergence for sigma(Rs) at R.

    Parameters
    ----------
    Rx : (r,) array
        array containing R for each m normalized to r_wrt
    Rs : (..., R) array
        array containing R for each sigma
    sigma : (..., R) array
        array containing sigma
    n_int : int
        number of interpolation steps in integration

    Returns
    -------
    kappa : (..., r) array
        convergence

    """
    log10_sigma_int = interp.interp1d(
        np.log10(Rs), np.log10(sigma),
        axis=-1, fill_value="extrapolate"
    )

    log10_R = np.log10(R)
    kappa = 10**log10_sigma_int(log10_R) / sigma_crit
    return kappa


def sigma_mean_from_sigma(R, Rs, sigma, n_int=1000, return_sigma=False):
    """Return the mean enclosed surface mass density for sigma(Rs) at R.

    Parameters
    ----------
    Rx : (r,) array
        array containing R for each m normalized to r_wrt
    Rs : (..., R) array
        array containing R for each sigma
    sigma : (..., R) array
        array containing sigma
    n_int : int
        number of interpolation steps in integration
    return_sigma : bool
        return interpolated sigma

    Returns
    -------
    sigma_mean : (..., r) array
        mean surface mass density of sigma

    """
    log10_sigma_int = interp.interp1d(
        np.log10(Rs), np.log10(sigma),
        axis=-1, fill_value="extrapolate"
    )

    # array that runs from 10^-5 to R for each dimension, need to move
    # iteration axis to the final one
    R_int = np.moveaxis(np.logspace(-5, np.log10(R), n_int), 0, -1)
    log10_R_int = np.moveaxis(np.linspace(-5, np.log10(R), n_int), 0, -1)

    sigma_int = 10**log10_sigma_int(log10_R_int)
    # apparently only first -inf value returns 0, rest is nan
    sigma_int[np.isnan(sigma_int)] = 0.

    # need to reshape R_int to match sigma_int
    shape = tuple(np.ones(len(sigma_int.shape) - 2, dtype=int)) + (-1, n_int)

    R_int = R_int.reshape(shape)
    sigma_mean = 2. / R**2 * intg.simps(sigma_int * R_int, x=R_int, axis=-1)

    if return_sigma:
        sigma_int = 10**log10_sigma_int(np.log10(R))
        # apparently only first -inf value returns 0, rest is nan
        sigma_int[np.isnan(sigma_int)] = 0.
        return sigma_mean, sigma_int
    return sigma_mean


def shear_from_sigma(R, Rs, sigma, sigma_crit=1, n_int=1000):
    """Return the mean enclosed surface mass density for sigma(Rs) at R.

    Parameters
    ----------
    R : (..., r) array
        array containing R for each m, should equal Rx*r_x
    Rs : (..., r) array
        array containing R for each sigma
    sigma : (..., r) array
        array containing sigma
    n_int : int
        number of interpolation steps in integration

    Returns
    -------
    shear : array
        shear of sigma

    """
    sigma_mean, sigma = sigma_mean_from_sigma(
        R=R, Rs=Rs, sigma=sigma, n_int=n_int, return_sigma=True
    )

    shear = (sigma_mean - sigma) / sigma_crit
    return shear


def shear_red_from_sigma(R, Rs, sigma, sigma_crit=1, n_int=1000):
    """Return the mean enclosed surface mass density for sigma(Rs) at R.

    Parameters
    ----------
    R : (..., r) array
        array containing R for each m, should equal Rx*r_x
    Rs : (..., r) array
        array containing R for each sigma
    sigma : (..., r) array
        array containing sigma
    n_int : int
        number of interpolation steps in integration

    Returns
    -------
    shear : array
        shear of sigma

    """
    sigma_mean, sigma = sigma_mean_from_sigma(
        R=R, Rs=Rs, sigma=sigma, n_int=n_int, return_sigma=True
    )

    shear_red = (sigma_mean - sigma) / (sigma_crit - sigma)
    return shear_red


@np.vectorize
def m_beta_plaw(r, m_x, r_x, r_c, beta, gamma, rho_x=None, **kwargs):
    """Return the mean enclosed mass of a beta profile
    with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r : float
        radius to compute for
    m_x : float
        mass inside r_x
    r_x : float
        radius to match rho_x at, in units of r_range
    r_c : float
        physical core radius r_c of the profile
    beta : float
        beta slope of the profile
    gamma : float
        power law slope of profile
    rho_x : density at r_x

    Returns
    -------
    m_beta_plaw : array
        the mass enclosed by the beta profile with power law extension

    """
    if r <= r_x:
        m = m_beta(r=r, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta)
    else:
        if rho_x is None:
            rho_x = profile_beta(np.array([r_x]).reshape(-1, 1),
                                 m_x=np.array([m_x]).reshape(-1, 1),
                                 r_x=np.array([r_x]).reshape(-1, 1),
                                 r_c=np.array([r_c]).reshape(-1, 1),
                                 beta=np.array([beta]).reshape(-1, 1))
            rho_x = rho_x.reshape(-1)
        m = (m_x + m_plaw(r=r, rho_x=rho_x, r_x=r_x, gamma=gamma))

    return m


@np.vectorize
def m_beta_plaw_rmax(r, r_max, m_x, r_x, r_c, beta, gamma, rho_x=None, **kwargs):
    """Return the mean enclosed mass of a beta profile
    with mass m_x inside r_range <= r_x

        rho[r] =  rho_c[m_range, r_c, r_x] / (1 + ((r/r_x)/r_c)^2)^(beta / 2)

    and a power law outside

        rho[r] = rho_x (r/r_x)^(-gamma)

    rho_c is determined by the mass of the profile.

    Parameters
    ----------
    r : float
        radius to compute for
    m_x : float
        mass inside r_x
    r_x : float
        radius to match rho_x at, in units of r_range
    r_c : float
        physical core radius r_c of the profile
    beta : float
        beta slope of the profile
    gamma : float
        power law slope of profile
    rho_x : density at r_x

    Returns
    -------
    m_beta_plaw : array
        the mass enclosed by the beta profile with power law extension

    """
    if r <= r_x:
        m = m_beta(r=r, m_x=m_x, r_x=r_x, r_c=r_c, beta=beta)
    else:
        if r > r_max:
            return 0.
        if rho_x is None:
            rho_x = profile_beta(np.array([r_x]).reshape(-1, 1),
                                 m_x=np.array([m_x]).reshape(-1, 1),
                                 r_x=np.array([r_x]).reshape(-1, 1),
                                 r_c=np.array([r_c]).reshape(-1, 1),
                                 beta=np.array([beta]).reshape(-1, 1))
            rho_x = rho_x.reshape(-1)
        m = (m_x + m_plaw(r=r, rho_x=rho_x, r_x=r_x, gamma=gamma))

    return m


@np.vectorize
def sigma_delta(R, m_x, **kwargs):
    """Return the surface mass density profile of a delta profile with mass
    m_x

    Parameters
    ----------
    R : array-like
        projected radius [h^-1 Mpc]
    m_x : (m,) array
        array containing mass

    Returns
    -------
    sigma_delta : array
        surface mass density of delta profile
    """
    if R == 0:
        sigma = m_x
    else:
        sigma = np.zeros_like(m_x)

    return sigma


def sigma_mean_delta(R, m_x, **kwargs):
    """Return the mean surface mass density profile of a delta profile with mass
    m_x

    Parameters
    ----------
    R : (R, 1) array
        projected radius [h^-1 Mpc]
    m_x : (1, m) array
        array containing mass

    Returns
    -------
    sigma_mean : array
        mean surface mass density of delta profile
    """
    return m_x / (np.pi * R**2)


def shear_delta(R, m_x, sigma_crit=1, **kwargs):
    """Return the shear profile of a delta profile with mass m_x

    Parameters
    ----------
    R : (R, 1) array
        projected radius [h^-1 Mpc]
    m_x : (1, m) array
        array containing mass
    sigma_crit : (m,) array or (m, z) array or float
        critical surface mass density of the observed systems

    Returns
    -------
    shear_delta : array
        shear of delta profile at projected radius R
    """
    if (R == 0).any():
        raise ValueError("mean surface mass density at R=0 is undefined")

    shear = sigma_mean_delta(R=R, m_x=m_x) / sigma_crit
    return shear


@np.vectorize
def m_delta(r, m_x, **kwargs):
    """
    Returns a delta function mass
    """
    return m_x
