from pathlib import Path

import commah
from george import kernels
import numpy as np
import os
import scipy.optimize as opt
from tremulator import Interpolator, Emulator

import lensing_haloes.settings as settings

TABLE_DIR = settings.TABLE_DIR


def c200c_correa(m200c, z, sigma_8, omega_m, n_s, h):
    """Return the Correa+2015 concentration for the given masses,
    redshifts and cosmology.

    Parameters
    ----------
    m200c : (m, ) array [M_sun / h]
        halo mass at overdensity 200 rho_crit
    z : array
        redshifts
    sigma_8 : float
        value of sigma_8
    omega_m : float
        value of omega_m
    n_s : float
        value of n_s
    h : float
        value of h

    Returns
    ------
    c200c : (z, m) array
        concentration at given mass and redshift

    """
    cosmo = {}
    cosmo["omega_M_0"] = omega_m
    cosmo["omega_lambda_0"] = 1 - omega_m
    cosmo["sigma_8"] = sigma_8
    cosmo["n"] = n_s
    cosmo["h"] = h

    c200c = commah.run(cosmology=cosmo,
                       Mi=m200c / cosmo["h"],
                       z=z,
                       mah=False)["c"].T
    return np.squeeze(c200c)


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


def rho_crit(z, omega_m):
    rho_crit0 = 2.7763458 * (10.0**11.0)  # [h^2 M_sun / Mpc^3]
    Ez2 = omega_m * (1 + z)**3 + 1. - omega_m
    return rho_crit0 * Ez2


def rho_mean(z, omega_m):
    rho_crit0 = 2.7763458 * (10.0**11.0)  # [h^2 M_sun / Mpc^3]
    return omega_m * rho_crit0 * (1. + z)**3


@np.vectorize
def log10_m200c_correa(log10_m200m, z, sigma_8, omega_m, n_s, h):
    """Return the halo mass m200c for the given m200m redshifts and
    cosmology assuming the Correa+2015 c(m) relation

    Parameters
    ----------
    log10_m200m : (m, ) array [M_sun / h]
        halo mass at overdensity 200 rho_c
    z : array
        redshifts
    sigma_8 : float
        value of sigma_8
    omega_m : float
        value of omega_m
    n_s : float
        value of n_s
    h : float
        value of h

    Returns
    ------
    m200c : (z, m) array

    """
    def m_diff(r200c):
        m200c = 4 / 3. * np.pi * 200 * rho_c * r200c**3

        # get the concentration for the halo mass
        c200c = c200c_correa(m200c=m200c,
                             z=z,
                             omega_m=omega_m,
                             sigma_8=sigma_8,
                             n_s=n_s, h=h)

        m = m_nfw(r=r200m, m_x=m200c, r_x=r200c, c_x=c200c)
        return m - 10**log10_m200m

    # get densities at z
    rho_c = rho_crit(z=z, omega_m=omega_m)
    rho_m = rho_mean(z=z, omega_m=omega_m)

    r200m = (10**log10_m200m / (4 / 3. * np.pi * 200 * rho_m))**(1. / 3)
    r200c = opt.brentq(m_diff, 0.1 * r200m, r200m)
    m200c = 4 / 3. * np.pi * 200 * rho_c * r200c**3

    return np.log10(m200c)


def c200m_correa(theta, *args, **kwargs):
    """Return the concentration for the given m200m, redshifts and
    cosmology assuming the Correa+2015 c(m) relation

    Parameters
    ----------
    theta : parameter vector containing
        log10_m200m : (m, ) array [M_sun / h]
            halo mass at overdensity 200 rho_m
        z : array
            redshifts
        sigma_8 : float
            value of sigma_8
        omega_m : float
            value of omega_m
        n_s : float
            value of n_s
        h : float
            value of h

    Returns
    ------
    c200m : (z, m) array

    """
    log10_m200m, z, sigma_8, omega_m, n_s, h = list(theta) + list(args)
    # get densities at z
    rho_c = rho_crit(z=z, omega_m=omega_m)
    rho_m = rho_mean(z=z, omega_m=omega_m)

    # get mean matter density values
    m200m = 10**log10_m200m
    r200m = (m200m / (4. / 3 * np.pi * 200 * rho_m))**(1. / 3)

    m200c = 10**log10_m200c_correa(log10_m200m=log10_m200m, z=z,
                                   omega_m=omega_m,
                                   sigma_8=sigma_8,
                                   n_s=n_s, h=h)
    c200c = c200c_correa(m200c=m200c,
                         z=z,
                         omega_m=omega_m,
                         sigma_8=sigma_8,
                         n_s=n_s, h=h)
    r200c = (m200c / (4. / 3 * np.pi * 200 * rho_c))**(1. / 3)

    # convert concentration
    c200m = c200c * r200m / r200c

    return np.squeeze(c200m)


def c200m_correa_emu(
        n_steps=100,
        n_add=10,
        a=0,
        b=1,
        epsilon=0.1,
        # bound are on log10_m200m, z, sigma_8, omega_m
        bounds=np.array([[8, 16],
                         [0, 6],
                         [0.7, 0.9],
                         [0.25, 0.35]]),
        # hyper parameters are the normalization and
        # correlation lengths in each dimension
        hyper_bounds=np.array([[np.log(1), np.log(1000)],
                               [np.log(0.001), np.log(5)],
                               [np.log(0.001), np.log(1)],
                               [np.log(0.001), np.log(0.1)],
                               [np.log(0.001), np.log(0.1)]]),
        # n_s, h
        args=[0.965, 0.674],
        kwargs=None,
        pool=None):
    """Use a GP to interpolate the c200m relation from Correa+2015 in a
    high-dimensional parameter space.

    Parameters
    ----------
    n_steps : int


    """
    # kernel parameters should be changed to fit data
    kernel = (kernels.ConstantKernel(np.log(5), ndim=len(bounds)) *
              kernels.ExpSquaredKernel(np.ones(len(bounds)),
                                       ndim=len(bounds)))

    # bounds should not exceed prior range in lnprob_mizi
    c200m_emu = Emulator(
        f=c200m_correa,
        kernel=kernel,
        bounds=bounds,
        args=args, kwargs=kwargs,
        hyper_bounds=hyper_bounds,
        pool=pool,
    )

    try:
        c200m_emu.train(n_steps=n_steps, n_add=n_add, a=a, b=b, epsilon=epsilon)
        return c200m_emu

    except (KeyboardInterrupt, ValueError) as e:
        print(
            f'{e}\n'
            '======================================================================\n'
            'Returning emulator as is')
        return c200m_emu
