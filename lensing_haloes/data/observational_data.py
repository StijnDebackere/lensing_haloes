"""This module parses the observational data and creates the data
products that are required by the halo.model module.
"""
import glob
from pathlib import Path
import re

import asdf
from astropy import units as u
from astropy import constants as const
import astropy.io.fits as fits
import numpy as np

import lensing_haloes.settings as settings
import lensing_haloes.cosmo.cosmo as cosmo
import lensing_haloes.halo.profiles as profs
import lensing_haloes.util.tools as tools

import pdb

OBS_DIR = settings.OBS_DIR
# global value of critical density
RHO_CRIT = 2.7763458 * (10.0**11.0)  # [h^2 M_sun / Mpc^3]


def load_datasets(datasets=['croston+08'], h_units=False):
    """Load the given datasets a dictionary.

    Parameters
    ----------
    datasets : iterable
        names of datasets to load
    h_units : bool
        convert to h=1 units

    Returns
    -------
    data : dict
        dictionary with information of each dataset
    """
    if h_units:
        h_append = '_h_1'
    else:
        h_append = '_h_0p7'

    dataset_options = {
        'croston+08': {
            'fname':  f'{OBS_DIR}/croston+08_parameters{h_append}.asdf',
            'marker': 'o',
        },
        'eckert+16': {
            'fname': f'{OBS_DIR}/eckert+16_parameters{h_append}.asdf',
            'marker': 'x'
        },
        'eckert+12': {
            'fname': f'{OBS_DIR}/eckert+12_parameters{h_append}.asdf',
            'marker': 'D'
        },
    }
    dataset_functions = {
        'croston+08': read_croston_2008,
        'eckert+16': read_eckert_2016,
        'eckert+12': read_eckert_2012,
    }

    data = {}
    for dataset in datasets:
        if dataset in dataset_options.keys():
            try:
                with asdf.open(
                        dataset_options[dataset]['fname'],
                        lazy_load=False, copy_arrays=True) as af:
                    data[dataset] = {**af.tree, **dataset_options[dataset]}
            except FileNotFoundError:
                data[dataset] = {
                    **dataset_functions[dataset](h_units=h_units),
                    **dataset_options[dataset]
                }

    return data


def read_croston_2008(h_units=False):
    # ---------------------------------------- #
    # We leave all the original h_70 scalings, #
    # we scale our model when comparing        #
    # ---------------------------------------- #

    data = np.loadtxt(f'{OBS_DIR}/croston+2008/Pratt09.dat')
    # r500 = data[:,1] * 1e-3 * 0.7 # [Mpc/h]
    # mgas500 = np.power(10, data[:,2]) * (0.7)**(5./2) # [Msun/h^(5/2)]
    # mgas500_err = np.power(10, data[:,3]) * (0.7)**(5./2) # [Msun/h^(5/2)]
    z = data[:, 0]
    r500 = data[:, 1] * 1e-3  # [Mpc/h_70]
    mgas500 = np.power(10, data[:, 2])  # [Msun/h_70^(5/2)]
    mgas500_err = np.power(10, data[:, 3])  # [Msun/h_70^(5/2)]
    T1 = data[:, 7]  # [keV] spectroscopic temperature in [0-1] r500c
    T2 = data[:, 12]  # [keV] spectroscopic temperature in [0.15-1] r500c
    T3 = data[:, 17]  # [keV] spectroscopic temperature in [0.15-0.75] r500c
    Y_X = data[:, 20] * 1e13  # [1e13 Msun/h_70^{5/2} keV]

    m500c = tools.radius_to_mass(r500, 500 * RHO_CRIT * 0.7**2 * cosmo.E2z(z=z, omega_m=0.315))
    logsigma_mt = 0.064

    fnames = [
        f"{OBS_DIR}/croston+2008/Croston08_system{i+1}.dat"
        for i in range(len(z))
    ]
    # Load in croston data -> n = n_e
    rx = [np.loadtxt(f)[:, 1] for f in fnames]

    # from Flux ~ EM ~ 1/d_A^2 * int n^2 dV ~ h^2 * h^-3 * [n^2]
    # where n is theoretical model and Flux is measurement, so n^2 ~ F * h
    # n ~ h^(1/2)
    n = [np.loadtxt(f)[:, 2] for f in fnames]  # [cm^-3 h_70^(1/2)]
    n_err = [np.loadtxt(f)[:, 3] for f in fnames]  # [cm^-3 h_70^(1/2)]

    # Convert electron densities to gas density
    # Interpolate (X_0, Y_0, Z_0) = (0.75, 0.25, 0)
    #          to (X_s, Y_s, Z_s) = (0.7133, 0.2735, 0.0132)
    #         for (X, Y, Z) = (0.73899, 0.25705, 0.00396) for Z = 0.3Z_s
    # mu = (1 + Y/X) / (2 + 3Y / (4X))
    # rho = mu * m_p * n_gas
    # -> fully ionised: mu=0.60 (X=0.73899, Y=0.25705, Z=0.00396)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    #       = (2 + 3Y/(4X))n_H
    # n_He = Y/(4X) n_H
    # n_e = n_H + 2n_He = (1 + Y/(2X)) n_H
    # n_gas = (2 + 3Y/(4X)) / (1 + Y/(2X)) n_e
    # => n_gas = 1.93 n_e
    # calculate correct mu factor for Z metallicity gas
    n2rho = (1.93 * 0.6 * const.m_p.cgs * 1/u.cm**3).value  # [g/cm^3]
    # change to ``cosmological'' coordinates
    cgs2cos = ((1e6 * const.pc.cgs)**3 / const.M_sun.cgs).value
    rho = [(ne * n2rho * cgs2cos) for ne in n]
    rho_err = [(ne * n2rho * cgs2cos) for ne in n_err]

    m500gas = np.empty((0,), dtype=float)
    mgas = np.empty((0,), dtype=float)
    mgas_err = np.empty((0, 2), dtype=float)
    m500c_err = np.empty((0, 2), dtype=float)
    fgas = np.empty((0,), dtype=float)
    fgas_err = np.empty((0, 2), dtype=float)
    for idx, prof in enumerate(rho):
        m500c_dist = np.random.lognormal(
            np.log(m500c[idx]), logsigma_mt, 100
        )
        r500c_dist = tools.mass_to_radius(
            m500c_dist, 500 * RHO_CRIT * 0.7**2 * cosmo.E2z(z=z[idx], omega_m=0.315)
        )
        mgas_dist = profs.mr_from_rho(
            r=r500c_dist,
            rs=rx[idx] * r500[idx],
            rho=prof
        )

        # get the covariance between mgas and m500c, derive the logarithmic fgas error
        cov_log = np.cov(np.log(mgas_dist), np.log(m500c_dist))

        # get the gas errors
        mgas_median = np.median(mgas_dist)
        mgas_errs = np.array([
            mgas_median - np.exp(np.log(mgas_median) - 0.5 * cov_log[0, 0]),
            np.exp(np.log(mgas_median) + 0.5 * cov_log[0, 0]) - mgas_median,
        ])

        # get the total mass errors
        m500c_median = np.median(m500c_dist)
        m500c_errs = np.array([
            m500c_median - np.exp(np.log(m500c_median) - 0.5 * cov_log[1, 1]),
            np.exp(np.log(m500c_median) + 0.5 * cov_log[1, 1]) - m500c_median,
        ])

        # F = ln(f_gas) = Mgas - M500c = ln(mgas) - ln(m500c)
        # sigma_F^2 = cov(0,0) + 2 cov(0, 1) + cov(1, 1)^2
        sigma_flog = np.einsum('i,ij,j', [1, 1], cov_log, [1, 1])**0.5

        # sigma_f = mean(exp(F - 0.5 sigma_F), exp(F + 0.5 sigma_F))
        fgas_median = np.median(mgas_dist / m500c_dist)
        fgas_errs = np.array([
            fgas_median - np.exp(np.log(fgas_median) - 0.5 * sigma_flog),
            np.exp(np.log(fgas_median) + 0.5 * sigma_flog) - fgas_median,
        ])

        m500gas = np.append(m500gas, mgas500[idx])
        mgas = np.append(mgas, np.median(mgas_dist))
        fgas = np.append(fgas, fgas_median)
        mgas_err = np.concatenate([mgas_err, mgas_errs.reshape(1, 2)], axis=0)
        m500c_err = np.concatenate([m500c_err, m500c_errs.reshape(1, 2)], axis=0)
        fgas_err = np.concatenate([fgas_err, fgas_errs.reshape(1, 2)], axis=0)
        # mgas = np.append(
        #     mgas, profs.mr_from_rho(
        #         r=r500[idx],
        #         rs=rx[idx] * r500[idx],
        #         rho=prof)
        # )
        # print(
        #     f'{idx}: rx_max = {rx[idx][-1]},'
        #     f' mgas_ratio={mgas[idx] / mgas500[idx]}'
        # )
        # print('-----------')

    # we save our gas mass determinations!
    if h_units:
        h = 0.7
        h_append = '_h_1'
    else:
        h = 1
        h_append = '_h_0p7'

    # we are rescaling our theoretical model, where rho ~ h^2, r ~ h^-1, m ~ h^-1
    data = {
        'mgas_500c': mgas * h,
        'mgas_500c_err': mgas_err * h,
        'mgas_500c_original': mgas500 * h,
        'fgas_500c': fgas,
        'fgas_500c_err': fgas_err,
        'r500c': r500 * h,
        'm500c': m500c * h,
        'm500c_err': m500c_err * h,
        'T1': T1,
        'T2': T2,
        'T3': T3,
        'T': T3,
        'Y_X': Y_X * h,
        'z': z,
        'rx': rx,
        'rho': [x / h**2 for x in rho],
        'rho_err': [x / h**2 for x in rho_err]
    }

    with asdf.AsdfFile(data) as af:
        af.write_to(f"{OBS_DIR}/croston+08_parameters{h_append}.asdf")

    return data


def read_eckert_2012():
    pass


def read_eckert_2016(h_units=False):
    # metadata
    mdata = fits.open(f'{OBS_DIR}/eckert+2016/XXL100GC.fits')
    fnames = glob.glob(f'{OBS_DIR}/eckert+2016/XLSSC*_nh.fits')
    nums = np.array([int(f.split('/')[-1][5:8]) for f in fnames])

    # ---------------------------------------- #
    # We leave all the original h_70 scalings, #
    # we scale our model when comparing        #
    # ---------------------------------------- #

    # number of cluster
    num = mdata[1].data['xlssc']
    with_profiles = np.array([n in nums for n in num], dtype=bool)

    # load observables
    m500mt = mdata[1].data['M500MT'][with_profiles]  # [Msun/h_70]
    m500mt_err = mdata[1].data['M500MT_err'][with_profiles]  # [Msun/h_70]
    mgas500 = mdata[1].data['Mgas500'][with_profiles]  # [Msun/h_70^(5/2)]
    mgas500_err = mdata[1].data['Mgas500_err'][with_profiles]  # [Msun/h_70^(5/2)]
    T300 = mdata[1].data['T300kpc'][with_profiles]  # [keV]
    mdata.close()

    # some values are invalid
    mask = ((m500mt > 0) & (mgas500 > 0))

    # mask the results
    num = np.asarray(num[with_profiles][mask])
    m500mt = np.asarray(m500mt[mask])
    m500mt_err = np.asarray(m500mt_err[mask])
    mgas500 = np.asarray(mgas500[mask])
    mgas500_err = np.asarray(mgas500_err[mask])
    T300 = np.asarray(T300[mask])

    m500mt *= 1e13
    m500mt_err *= 1e13
    mgas500 *= 1e13
    mgas500_err *= 1e13

    # now load the good files
    fnames = [f'{OBS_DIR}/eckert+2016/XLSSC{n:03d}_nh.fits' for n in num]
    # Interpolate (X_0, Y_0, Z_0) = (0.75, 0.25, 0)
    #          to (X_s, Y_s, Z_s) = (0.7133, 0.2735, 0.0132)
    #         for (X, Y, Z) = (0.73899, 0.25705, 0.00396) for Z = 0.3Z_s
    # mu = (1 + Y/X) / (2 + 3Y / (4X))
    # rho = mu * m_p * n_gas
    # -> fully ionised: mu=0.6 (X=0.73899, Y=0.25705, Z=0.00396)
    # n_gas = n_e + n_H + n_He = 2n_H + 3n_He (fully ionized)
    # n_He = Y/(4X) n_H
    # n_gas = 2 + 3Y/(4X) n_H
    # => n_gas = 2.26 n_H
    n2rho = (2.26 * 0.6 * const.m_p.cgs * 1./u.cm**3).value  # [cm^-3]
    cgs2cos = ((1e6 * const.pc.cgs)**3 / const.M_sun.cgs).value

    r500 = np.empty((0,), dtype=float)
    z = np.empty((0,), dtype=float)
    rx = []
    rho = []
    rho_err = []
    for f in fnames:
        # actual data
        data = fits.open(f)
        z = np.append(z, data[1].header['REDSHIFT'])

        # !! check whether this still needs correction factor from weak lensing
        # Currently included it, but biases the masses, since this wasn't done
        # in measurement by Eckert, of course
        r500 = np.append(r500, data[1].header['R500'] * 1e-3 * (1.3)**(-1./3))  # [Mpc/h_70]
        # r500 = np.append(r500, data[1].header['R500'] * 1e-3) # [Mpc/h_70]

        rx.append(data[1].data['RADIUS'] * (1.3)**(1./3))
        # rx.append(data[1].data['RADIUS'])

        rho.append(data[1].data['NH'] * n2rho * cgs2cos)  # [Msun / Mpc^3 h_70^(1/2)]
        rho_err.append(data[1].data['ENH'] * n2rho * cgs2cos)
        data.close()

    # also convert m500c from M-T relation
    m500mt = m500mt / 1.3
    m500c = tools.radius_to_mass(
        r500, 500 * RHO_CRIT * 0.7**2 * cosmo.E2z(z=z, omega_m=0.3)
    )
    r500mt = tools.mass_to_radius(
        m500mt, 500 * RHO_CRIT * 0.7**2 * cosmo.E2z(z=z, omega_m=0.3)
    )
    logsigma_mt = 0.53

    mgas = np.empty((0,), dtype=float)
    mgas_err = np.empty((0, 2), dtype=float)
    m500c_err = np.empty((0, 2), dtype=float)
    fgas = np.empty((0,), dtype=float)
    fgas_err = np.empty((0, 2), dtype=float)
    for idx, prof in enumerate(rho):
        # we take into account the distribution of m500c_MT values
        # from Lieu+2016 -> sigma_int,ln(M|T) = 0.53
        m500c_dist = np.random.lognormal(
            np.log(m500c[idx]), logsigma_mt, 100
        )
        r500c_dist = tools.mass_to_radius(
            m500c_dist, 500 * RHO_CRIT * 0.7**2 * cosmo.E2z(z=z[idx], omega_m=0.315)
        )
        mgas_dist = profs.mr_from_rho(
            r=r500c_dist,
            rs=rx[idx] * r500[idx],
            rho=prof
        )

        # get the covariance between mgas and m500c, derive the logarithmic fgas error
        cov_log = np.cov(np.log(mgas_dist), np.log(m500c_dist))

        # get the gas errors
        mgas_median = np.median(mgas_dist)
        mgas_errs = np.array([
            mgas_median - np.exp(np.log(mgas_median) - 0.5 * cov_log[0, 0]),
            np.exp(np.log(mgas_median) + 0.5 * cov_log[0, 0]) - mgas_median,
        ])

        # get the total mass errors
        m500c_median = np.median(m500c_dist)
        m500c_errs = np.array([
            m500c_median - np.exp(np.log(m500c_median) - 0.5 * cov_log[1, 1]),
            np.exp(np.log(m500c_median) + 0.5 * cov_log[1, 1]) - m500c_median,
        ])

        # F = ln(f_gas) = Mgas - M500c = ln(mgas) - ln(m500c)
        # sigma_F^2 = cov(0,0) + 2 cov(0, 1) + cov(1, 1)^2
        sigma_flog = np.einsum('i,ij,j', [1, 1], cov_log, [1, 1])**0.5

        # sigma_f = mean(exp(F - 0.5 sigma_F), exp(F + 0.5 sigma_F))
        fgas_median = np.median(mgas_dist / m500c_dist)
        fgas_errs = np.array([
            fgas_median - np.exp(np.log(fgas_median) - 0.5 * sigma_flog),
            np.exp(np.log(fgas_median) + 0.5 * sigma_flog) - fgas_median,
        ])

        mgas = np.append(mgas, np.median(mgas_dist))
        fgas = np.append(fgas, fgas_median)
        mgas_err = np.concatenate([mgas_err, mgas_errs.reshape(1, 2)], axis=0)
        m500c_err = np.concatenate([m500c_err, m500c_errs.reshape(1, 2)], axis=0)
        fgas_err = np.concatenate([fgas_err, fgas_errs.reshape(1, 2)], axis=0)
        # print(
        #     f'{idx}: rx_max = {rx[idx][-1]},'
        #     f' mgas_ratio={mgas[idx] / mgas500[idx]}'
        # )
        # print('-----------')

    if h_units:
        h = 0.7
        h_append = '_h_1'
    else:
        h = 1
        h_append = '_h_0p7'
    # we are rescaling our theoretical model, where rho ~ h^2, r ~ h^-1, m ~ h^-1
    data = {
        'XLSSC': num,
        'mgas_500c': mgas * h,
        'mgas_500c_err': mgas_err * h,
        'mgas_500c_original': mgas500 * h,
        'fgas_500c': fgas,
        'fgas_500c_err': fgas_err,
        'r500c': r500 * h,
        'm500c': m500c * h,
        'm500c_err': m500c_err * h,
        'm500c_MT': m500mt * h,
        'T': T300,
        'Y_X': T300 * mgas * h,
        'z': z,
        'rx': rx,
        'rho': [x / h**2 for x in rho],
        'rho_err': [x / h**2 for x in rho_err]
    }

    with asdf.AsdfFile(data) as af:
        af.write_to(f"{OBS_DIR}/eckert+16_parameters{h_append}.asdf")

    return data
