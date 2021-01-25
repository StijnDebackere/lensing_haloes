"""This module contains functionality to convert different generally
used cosmology descriptions to a unified dictionary representation.

"""
from astropy.cosmology import FlatwCDM
import astropy.units as u
import numpy as np
import pyccl


def cosmology(
        omega_m=0.315,
        sigma_8=0.811,
        # A_s=2.1,
        omega_b=0.0493,
        h=0.674,
        n_s=0.965,
        w0=-1.,
        transfer_function='eisenstein_hu',
        mass_function='tinker',
        **kwargs
):
    """
    Return a pyccl.Cosmology object with the given cosmological parameters.

    Parameters
    ----------
    omega_m : float
        mean matter density at z=0
    sigma_8: float
        normalization of the matter power spectrum
    omega_b : float
        mean baryon density at z=0
    h : float
        dimensionless Hubble parameter
    n_s: float
        spectral index of the linear power spectrum
    w0 : float
        equation of state parameter of Dark Energy
    **kwargs : other arguments to pass to pyccl.Cosmology

    Returns
    -------
    pyccl.Cosmology object for given cosmology
    """
    return pyccl.Cosmology(
        Omega_c=omega_m - omega_b, Omega_b=omega_b,
        sigma8=sigma_8, h=h, n_s=n_s, w0=w0,
        transfer_function=transfer_function,
        mass_function=mass_function,
        **kwargs)


def cosmo_dict(cosmo):
    """
    Return a dictionary for cosmo.

    Parameters
    ----------
    cosmo : pyccl.Cosmology object
        cosmology

    Returns
    -------
    cosmo_dict : dict
        - omega_m : float
            mean matter density at z=0
        - sigma_8: float
            normalization of the matter power spectrum
        - omega_b : float
            mean baryon density at z=0
        - h : float
            dimensionless Hubble parameter
        - n_s: float
            spectral index of the linear power spectrum
        - w0 : float
            equation of state parameter of Dark Energy

    """
    return {
        "omega_m": cosmo._params.Omega_m,
        "omega_b": cosmo._params.Omega_b,
        "sigma_8": cosmo._params.sigma8,
        "h": cosmo._params.h,
        "n_s": cosmo._params.n_s,
        "w0": cosmo._params.w0
    }


def dVdz(z, omega_m=0.315, h=0.674, w0=-1, **kwargs):
    """Return the differential comoving volume at z for cosmo

    Parameters
    ----------
    z : array-like
        redshifts
    omega_m : float
        mean matter density at z=0
    h : float
        dimensionless Hubble parameter
    w0 : float
        equation of state parameter of Dark Energy

    Returns
    -------
    dVdz : array like [h^-3 Mpc^3/deg^2]
        comoving volume
    """
    c = FlatwCDM(**{"Om0": omega_m, "w0": w0, "H0": h * 100})
    # convert differential comoving volume to deg^-2 instead of sr^-1
    Vz = c.differential_comoving_volume(z=z).to(u.Mpc**3 / u.deg**2) * h**3

    return Vz.value


def E2z(z, omega_m):
    """
    Return the self-similar evolution factor for the critical density.

    Parameters
    ----------
    z : array
        redshifts
    omega_m : float
        cosmological matter density

    Returns
    -------
    E2z : (z,) array
        omega_m * (1 + z)^3 + 1 - omega_m
    """
    return np.atleast_1d(omega_m * (1 + z)**3 + 1 - omega_m)


def h2z_ratio(z_1, z_2, omega_m):
    """Return the ratio self-similar evolution factor for the critical
    density between z_1 and z_2.

    Parameters
    ----------
    z_1 : array
        numerator redshifts
    z_2 : array
        denominator redshifts
    omega_m : float
        cosmological matter density

    Returns
    -------
    E2z : (z,) array
        omega_m * (1 + z)^3 + 1 - omega_m

    """
    z_1 = np.atleast_1d(z_1)
    z_2 = np.atleast_1d(z_2)
    if np.all(z_1 == z_2):
        return np.ones_like(z_1)
    else:
        return E2z(z=z_1, omega_m=omega_m) / E2z(z=z_2, omega_m=omega_m)
