"""This module generates mock cluster samples."""
from copy import deepcopy
from multiprocessing import Process, Manager
from pathlib import Path
import os
import re

import asdf
import numpy as np
from numpy.random import SeedSequence, default_rng
from pyccl.halos.hmfunc import MassFuncTinker08
import scipy.interpolate as interp
import scipy.optimize as opt
from tqdm import tqdm

import lensing_haloes.settings as settings
from lensing_haloes.cosmo.cosmo import cosmology
import lensing_haloes.halo.abundance as abundance
from lensing_haloes.util.tools import chunks, matched_arrays_to_coords, despike

import pdb


MOCK_DIR = settings.MOCK_DIR
TABLE_DIR = settings.TABLE_DIR


def homogeneous_N_in_bin(
        z, dz,
        log10_m200m, dlog10_m200m,
        cosmo=cosmology(), A_survey=2500,
        MassFunc=MassFuncTinker08
):
    """Return the homogeneous expectation value of the number of haloes at
    (z, log10_m200m) in a bin of size (dz, dlog10_m200m) for cosmo and
    A_survey. We do not assume variation in the expectation value
    inside each bin.

    Parameters
    ----------
    z : float or array
        redshifts
    dz : float or array
        bin size
    log10_m200m : float or array
        log10 of the mass [M_sun / h]
    dlog10_m200m : float or array
        bin size
    cosmo : pyccl.Cosmology object
        cosmology
    A_survey : float
        survey area [deg^2]
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use

    Returns
    -------
    N : float or array
        number of haloes in each bin
    """
    z = np.atleast_1d(z)
    dz = np.atleast_1d(dz)
    log10_m200m = np.atleast_1d(log10_m200m)
    dlog10_m200m = np.atleast_1d(dlog10_m200m)

    dNdlog10mdz_edges = abundance.dNdlog10mdz(
        z=z, log10_m200m=log10_m200m,
        cosmo=cosmo, A_survey=A_survey,
        MassFunc=MassFunc)

    return dNdlog10mdz_edges * dz.reshape(-1, 1) * dlog10_m200m.reshape(1, -1)


def constant_N_m200m_bins(
        log10_m200m, z, dz,
        N_in_bin=1000,
        cosmo=cosmology(), A_survey=2500,
        MassFunc=MassFuncTinker08
):
    """Optimize the sampling of log10_m200m such that each bin contains
    approximately N_in_bin haloes.

    Parameters
    ----------
    log10_m200m : float or array
        log10 of the mass [M_sun / h]
    z : float or array
        redshifts
    dz : float or array
        bin size
    cosmo : pyccl.Cosmology object
        cosmology
    A_survey : float
        survey area [deg^2]
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use

    Returns
    -------
    log10_m200m_bins : array
        bins for log10_m200m such that each bin contains ~N_in_bin haloes
    """
    log10_m200m = np.atleast_1d(log10_m200m)
    z = np.atleast_1d(z)
    dz = np.atleast_1d(dz)

    if len(dz) == 1:
        dz = np.ones_like(z) * dz
    elif dz.shape != z.shape:
        raise ValueError('if dz is not a float, needs same shape as z')

    dlog10_m200m = []
    log10_m200m_bins = []

    # iterate over each z bin to find the log10_m200m bins
    for idx_z, zz in enumerate(z):
        dlog10_m200m.append(np.zeros((0, ), dtype=float))
        log10_m200m_bins.append(np.atleast_1d(log10_m200m[0]))

        while log10_m200m_bins[idx_z][-1] < log10_m200m[-1]:
            # number scales linearly with bin size due to homogeneous
            # assumption => set bin size to 1 and N_in_bin / result
            # gives required bin size
            dm = N_in_bin / homogeneous_N_in_bin(
                z=zz, dz=dz[idx_z],
                log10_m200m=log10_m200m_bins[idx_z][-1],
                dlog10_m200m=1,
                cosmo=cosmo,
                A_survey=A_survey,
                MassFunc=MassFunc).reshape(-1)

            dlog10_m200m[idx_z] = np.concatenate(
                [dlog10_m200m[idx_z], dm], axis=-1)

            log10_m200m_bins[idx_z] = np.concatenate(
                [log10_m200m_bins[idx_z], log10_m200m_bins[idx_z][-1] + dm],
                axis=-1)

    # set max(log10_m200m) as final bin edge
    for idx_z, (m_bins, dms) in enumerate(zip(log10_m200m_bins, dlog10_m200m)):
        m_bins[-1] = log10_m200m[-1]
        dms[-1] = m_bins[-1] - m_bins[-2]

    return log10_m200m_bins


def thin_samples(
        bins_lower,
        bins_upper,
        p_hmg,
        f_htg, htg_args=None,
        rng=default_rng(0),
        out_q=None):
    """Draw and keep the samples that survive the homogeneous thinning.

    Parameters
    ----------
    bins_lower : array-like
        lower bin limit, ordered to match f_htg call
    bins_upper : array-like
        upper bin limit
    p_hmg : array-like
        homogeneous expectation value for each bin
    f_htg : callable
        function to calculate heterogeneous expectation value in each bin.
        First arguments need to coincide with *bins_lower
    htg_args : iterable
        arguments for p_htg
    rng : np.random.Generator instance
        Generator for random numbers
    out_q : multiprocessing.Queue object or None
        queue to output to if running as a process
        [Default: None]

    Returns
    -------
    samples drawn according to f_htg within bins defined by bins_lower
    and bins_upper
    """
    bin_volume = np.product(bins_upper - bins_lower, axis=-1)
    N_hmg = rng.poisson(p_hmg * bin_volume)

    to_sample = (N_hmg > 0)

    sampled = []
    for (bl, bu, N, p) in zip(
            bins_lower[to_sample],
            bins_upper[to_sample],
            N_hmg[to_sample],
            p_hmg[to_sample]
    ):
        # draw the number of samples expected from the homogeneous value
        samples = rng.uniform(low=bl, high=bu, size=(N, len(bl)))

        # pass these along to the heterogeneous calculation
        p_htg = f_htg(*samples.T, *htg_args)

        U_i = rng.uniform(size=N)
        keep = (U_i <= (p_htg / p))
        if keep.sum() > 0:
            sampled.append(samples[keep])

    if out_q is not None:
        out_q.put([os.getpid(), sampled])

    else:
        return sampled


def sample_haloes(
        log10_m0=13.5, log10_m1=16,
        z0=0.25, z1=2, n_z=10,
        cosmo=cosmology(), A_survey=2500,
        MassFunc=MassFuncTinker08,
        n_cpus=1,
        save=True, fname=f'{MOCK_DIR}/halo_sample_planck2019.asdf',
        seedsequence=SeedSequence(12345)
):
    """Generate a halo sample from the 2D Poisson distribution with
    expectation value

        N(mi, zi) = int_mi^mi+dm int_zi^zi+dz dNdlnmdz dlnm dz

    We sample the haloes by thinning a homogeneous Poisson process
    with expectation N_star > N(mi, zi) in the bin (mi, zi) to give
    the non-homogeneous process with expectation N(m, z).

    Parameters
    ----------
    log10_m0 : float
        log10 m of minimum halo mass in the sample [M_sun / h]
        [Default: 13.5]
    log10_m1 : float
        log10 m of maximum halo mass in the sample [M_sun / h]
        [Default: 15.5]
    z0 : float
        minimum redshift in the sample
        [Default: 0.25]
    z1 : float
        maximum redshift in the sample
        [Default: 2]
    n_z : int
        number of bins in z
        [Default: 10]
    cosmo : pyccl.Cosmology object
        cosmology
    A_survey : float [deg^2]
        survey area
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use
    n_cpus : int
        number of cpus to use
    save : bool
        save results to asdf file with fname
    fname : string
        filename to save to
    seedsequence : numpy.random.SeedSequence
        seed for random number generation

    Returns
    -------
    m_sample, z_sample : arrays
        masses and redshifts of the haloes in the sample
    """
    z_bins = np.linspace(z0, z1, n_z + 1)
    dz = (z1 - z0) / n_z

    # we calculate the m200m_bins such that the expected number of haloes
    # per bin is approximately constant. This results in an unstructured grid
    log10_m200m_bins = constant_N_m200m_bins(
        log10_m200m=np.array([log10_m0, log10_m1]),
        z=z_bins, dz=dz,
        N_in_bin=2000,
        cosmo=cosmo, A_survey=A_survey,
        MassFunc=MassFunc
    )

    # Now we convert bins to ((z - 1) * (m - 1), 2) arrays with (zi, mi) for
    # multiprocessing
    # => get lower and upper bin edges and their maximum abundances
    bins_lower = np.concatenate(
        [
            matched_arrays_to_coords(
                z_bins[idx_z].reshape(-1, 1),
                log10_m200m_bins[idx_z][:-1].reshape(1, -1))
            for idx_z in range(z_bins.shape[0] - 1)
        ], axis=0)
    bins_upper = np.concatenate(
        [
            matched_arrays_to_coords(
                z_bins[idx_z + 1].reshape(-1, 1),
                log10_m200m_bins[idx_z][1:].reshape(1, -1))
            for idx_z in range(z_bins.shape[0] - 1)
        ], axis=0)

    def neg_dNdlog10mdz(coords): return -abundance.dNdlog10mdz(
                z=coords[0],
                log10_m200m=coords[1],
                cosmo=cosmo, A_survey=A_survey,
                MassFunc=MassFunc).reshape(-1)

    # get the maximum value of dNdlog10mdz for each bin
    dNdlog10mdz_max = np.concatenate(
        [
            -opt.minimize(
                neg_dNdlog10mdz,
                x0=bin_lo,
                bounds=[[bin_lo[0], bin_up[0]], [bin_lo[1], bin_up[1]]]
            ).fun
            for bin_lo, bin_up in zip(bins_lower, bins_upper)
        ])

    # homogeneous abundance for each bin
    dNdlog10mdz_hmg = dNdlog10mdz_max.flatten()

    # set up process management
    manager = Manager()
    out_q = manager.Queue()
    procs = []

    # split arrays over cpus
    bins_lower_split = np.array_split(bins_lower, n_cpus)
    bins_upper_split = np.array_split(bins_upper, n_cpus)
    dNdlog10mdz_hmg_split = np.array_split(dNdlog10mdz_hmg, n_cpus)
    streams = [default_rng(s) for s in seedsequence.spawn(n_cpus)]

    for idx, (bl, bu, dNdl10mdz) in enumerate(
            zip(bins_lower_split, bins_upper_split, dNdlog10mdz_hmg_split)):
        process = Process(
            target=thin_samples,
            args=(
                bl, bu, dNdl10mdz,
                abundance.dNdlog10mdz_mizi,
                (cosmo, A_survey, MassFunc),
                streams[idx],
                out_q)
        )

        procs.append(process)
        process.start()

    # get results
    results = []
    for _ in range(n_cpus):
        results.append(out_q.get())

    # make sure processes are ended nicely
    for proc in procs:
        proc.join()

    # need to sort results
    results.sort()
    sample = np.concatenate(
        [
            np.concatenate(item[1], axis=0) for item in results
        ], axis=0)

    sample_dict = {
        "z_sample": sample[:, 0],
        "true": {
            "selection": np.ones_like(sample[:, 0], dtype=bool),
            "m200m_sample": 10**sample[:, 1],
            "z_min": z0,
            "z_max": z1,
            "m200m_min": 10**log10_m0,
        },
        "A_survey": A_survey,
        "omega_m": cosmo._params.Omega_m,
        "sigma_8": cosmo._params.sigma8,
        "omega_b": cosmo._params.Omega_b,
        "h": cosmo._params.h,
        "n_s": cosmo._params.n_s,
        "w0": cosmo._params.w0,
        "log10_m_min": log10_m0,
        "log10_m_max": log10_m1,
        "z_min": z0,
        "z_max": z1,
    }

    if save:
        af = asdf.AsdfFile(sample_dict)
        af.write_to(fname)

    return sample_dict


def generate_many_samples(
        n0=0, n_steps=1000,
        log10_m0=14, log10_m1=16,
        z0=0.25, z1=2, n_z=10,
        cosmo=cosmology(),
        A_survey=2500,
        MassFunc=MassFuncTinker08,
        n_cpus=5,
        fname_base=f"{MOCK_DIR}/halo_sample_Asurvey_2p5kdeg2_z_0p25-2_m200m_min_14_planck2019",
        seedsequence=SeedSequence(12345)
):
    """Generate n_samples halo samples with parameters to be passed to
    sample_haloes().

    Parameters
    ----------
    n0 : int
        initial halo sample
    n_steps : int
        total number of haloes to samples, starting from n0
    log10_m0 : float
        log10 m of minimum halo mass in the sample [M_sun / h]
        [Default: 13.5]
    log10_m1 : float
        log10 m of maximum halo mass in the sample [M_sun / h]
        [Default: 15.5]
    z0 : float
        minimum redshift in the sample
        [Default: 0.25]
    z1 : float
        maximum redshift in the sample
        [Default: 2]
    n_z : int
        number of bins in z
        [Default: 10]
    cosmo : pyccl.Cosmology object
        cosmology
    A_survey : float [deg^2]
        survey area
    MassFunc : pyccl.halos.hmfunc.MassFunc object
        mass function to use
    n_cpus : int
        number of cpus to use
    fname_base : string
        base filename to save to, will be appended with _i.asdf
    seedsequence : numpy.random.SeedSequence
        seed for random number generation

    Returns
    -------
    saves samples to fname_base
    """
    child_seeds = seedsequence.spawn(n0 + n_steps)

    for i in tqdm(
            range(n0, n0 + n_steps),
            position=0,
            desc="Running samples"):
        sample = sample_haloes(
            log10_m0=log10_m0, log10_m1=log10_m1,
            z0=z0, z1=z1, n_z=n_z,
            A_survey=A_survey,
            cosmo=cosmo, MassFunc=MassFunc,
            n_cpus=n_cpus,
            save=True, fname=f"{fname_base}_{i}.asdf",
            seedsequence=child_seeds[i],
        )


def gen_fname(A_survey, z_min, z_max, m200m_min, cosmo):
    if z_max is None:
        z_str = f'zmin_{z_min}'
    else:
        if z_min is None:
            z_str = f'zmax_{z_max}'
        else:
            z_str = f'z_{z_min}-{z_max}'

    str_base = (
        MOCK_DIR +
        f"halo_sample_Asurvey_{A_survey}deg2_"
        f"{z_str}_m200m_min_{m200m_min}"
        f"_{cosmo}")
    return str_base


def get_fnames(
        A_survey="15k",
        z_min="0p25",
        z_max=None,
        m200m_min="14",
        method="WL",
        cosmo="planck2019",
):
    """Get the filenames of the mock surveys matching the given
    properties.

    """
    method_options = ['WL', 'WL_c', 'X', 'ap', None]
    if method not in method_options:
        raise ValueError(f'{method} not in {method_options}')

    method_str = {
        'WL': 'WL',
        'WL_c': 'WL_c_free',
        'X': 'WL\+X-ray'
    }
    if z_max is None:
        z_str = f'zmin_{z_min}'
    else:
        if z_min is None:
            z_str = f'zmax_{z_max}'
        else:
            z_str = f'z_{z_min}-{z_max}'

    re_str_base = (
        MOCK_DIR +
        f"halo_sample_Asurvey_{A_survey}deg2_"
        f"{z_str}_m200m_min_{m200m_min}"
        f"_{cosmo}"
        "_[0-9]{1,3}"
    )

    if method is None:
        regexp = re.compile(re_str_base + "\.asdf")
    else:
        regexp = re.compile(
            re_str_base
            + f"_{method_str[method]}_gamma0_.*\.asdf"
        )

    mock_files = [
        os.path.join(f'{MOCK_DIR}', fname)
        for fname in os.listdir(f'{MOCK_DIR}')
    ]

    fnames = [
        regexp.match(fname).group() for fname in mock_files
        if regexp.match(fname)
    ]
    return fnames


def bias_interp(m, m_ref, b_ref):
    """Return the reference bias relation interpolated to m."""
    b_interp = interp.interp1d(np.log10(m_ref), b_ref)
    return b_interp(np.log10(m))


def bias_samples_fbar(
        fnames,
        method,
        m200m_method,
        m200m_min, z_min, z_max,
        m200m_ratio_interp
):
    """Bias the halo samples with fnames according to method with
    m200m_method.

    Parameters
    ----------
    fnames : list
        list of halo sample filenames
    method : one of ['WL', 'WL_c', 'X', 'ap']
        mass determination method
    m200m_method : (z, m) array
        m200m determined for m500c from method
    m200m_min : float
        minimum halo mass in the sample
    z_min : float
        minimum redshift in the sample
    z_max : float
        maximum redshift in the sample
    m200m_ratio_interp : scipy.interpolate.Rbf object
        interpolator for the m200m ratio

    Returns
    -------
    saves biased samples to fnames_[WL, WL_c_free, WL+X-ray, WL_ap]_gamma_0_[g0].asdf
    """
    method_options = ['WL', 'WL_min', 'WL_max', 'WL_c', 'WL_c_min', 'WL_c_max']
    if method not in method_options:
        raise ValueError(f"{method} not in {method_options}")

    for fname in fnames:
        with asdf.open(fname, copy_arrays=True, mode='rw') as af:
            if z_min < af['true']['z_min']:
                raise ValueError(f'z_min should be larger than {af["true"]["z_min"]}')
            if z_max > af['true']['z_max']:
                raise ValueError(f'z_max should be smaller than {af["true"]["z_max"]}')

            m200m_sample = af['true']['m200m_sample'][:]
            z_sample = af['z_sample'][:]

            selection = (
                (m200m_sample > m200m_min) & (z_sample < z_max)
                & (z_sample > z_min)
            )
            z_sample = z_sample[selection]
            m200m_sample = m200m_sample[selection]

            coords_interp = np.array([z_sample, np.log10(m200m_sample)])
            m200m_sample_method = m200m_ratio_interp(*coords_interp) * m200m_sample

            print(f"{os.getpid()}: adding {method} to {fname}")
            # create or open file for gamma_0
            af.tree = {
                **af,
                **{
                    method: {
                        'selection': selection,
                        'm200m_sample': m200m_sample_method,
                        'm200m_min': m200m_min,
                        'z_min': z_min,
                        'z_max': z_max,
                    }
                }
            }
            af.update()


def bias_samples_fbar_mp(
        fnames,
        model_file_zrange=f"{TABLE_DIR}/observational_results_fgas_r_planck2019_z_0p1-2p0_m500c_13p5-15p5_nbins_4_R_0p75-2p5.asdf",
        prof=None,
        methods=['WL', 'WL_c'],
        m200m_min=10**14, z_min=0.25, z_max=2,
        n_cpus=8,
):
    """Generate biased halo samples for the list of fnames. Only haloes
    >m200m_min and z_min < z < z_max are included. An interpolated
    relation for the ratio between (z, log10(m200m_dmo)) and m500c needs
    to be passed along

    """
    prof_options = [None, 'median', 'min', 'max']
    if prof not in prof_options:
        raise ValueError(f'prof should be in {prof_options}')

    if prof is None or prof == 'median':
        method_options = {
            'WL': 'm200m_WL',
            'WL_c': 'm200m_WL_rs',
        }
    elif prof == 'min':
        method_options = {
            'WL_min': 'm200m_WL',
            'WL_c_min': 'm200m_WL_rs',
        }

    elif prof == 'max':
        method_options = {
            'WL_max': 'm200m_WL',
            'WL_c_max': 'm200m_WL_rs',
        }

    methods_in_options = [m for m in methods if m in method_options.keys()]
    if len(methods_in_options) == 0:
        raise ValueError(f"{methods} need to be in {method_options.keys()}")

    with asdf.open(model_file_zrange, copy_arrays=True) as af:
        model_z = af.tree
        coords = matched_arrays_to_coords(
            model_z['z'].reshape(-1, 1), np.log10(model_z['m200m_dmo'])
        )

        m200m_methods = {}
        m200m_dmo_ratio_interp = {}
        for method in methods_in_options:
            # remove spikes from the ratios due to fitting errors
            # spikes are most clear in m200m_obs ratio
            pdb.set_trace()
            m200m_methods[method] = np.asarray([
                despike(m, fill='mean') for m in (
                    af[method_options[method]][:] / model_z['m200m_obs'])
            ]) * model_z['m200m_obs']

        for method in methods_in_options:
            m_ratio = m200m_methods[method] / model_z['m200m_dmo']
            m200m_dmo_ratio_interp[method] = interp.Rbf(
                *coords.T, m_ratio.flatten()
            )

    # set up process management
    manager = Manager()
    out_q = manager.Queue()
    procs = []

    fnames_split = list(chunks(fnames, n_cpus))
    # need to copy interpolator for independent use in processes
    interp_split = [deepcopy(m200m_dmo_ratio_interp) for i in range(n_cpus)]
    for method in methods_in_options:
        for intrp, fns in zip(interp_split, fnames_split):
            process = Process(
                target=bias_samples_fbar,
                args=(
                    fns,
                    method,
                    m200m_methods[method],
                    m200m_min,
                    z_min, z_max,
                    intrp[method]
                )
            )
            procs.append(process)
            process.start()

        for proc in procs:
            proc.join()
