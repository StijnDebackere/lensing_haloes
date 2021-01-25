"""Utility functions for object manipulation that is used throughout
different modules.
"""
import asdf
import numpy as np
import scipy.integrate as intg
import scipy.interpolate as interp
import scipy.optimize as opt

import pdb


RHO_CRIT = 2.7763458 * (10.0**11.0)  # [h^2 M_sun / Mpc^3]


def _check_iterable(prms):
    """
    Go through prms and make them iterable. Useful if not sure if input
    is scalar or array

    Parameters
    ----------
    prms : list
        list of parameters to check

    Returns
    -------
    prms_iterable : list
        list of all parameters as lists
    """
    prms_iterable = []
    for prm in prms:
        prm = np.asarray(prm)
        if prm.shape == ():
            prm = prm.reshape(-1)
        prms_iterable.append(prm)

    return prms_iterable


def chunks(lst, n):
    """Yield n successive chunks from lst."""
    newn = int(1.0 * len(lst) / n + 0.5)
    for i in range(0, n-1):
        yield lst[i * newn:i * newn + newn]

    yield lst[n * newn - newn:]


def vectorize(*args, **kwargs):
    """Vectorize a function while passing args and kwargs"""
    def vectorize_wrapper(func):
        return np.vectorize(func, *args, **kwargs)
    return vectorize_wrapper


def open_or_create_asdf(fname):
    """Open AsdfFile with fname or create it."""
    try:
        af = asdf.open(fname, mode='rw')
    except FileNotFoundError:
        af = asdf.AsdfFile()
        af.tree = {}
        af.write_to(fname)
        af.close()
        af = asdf.open(fname, mode='rw')

    return af


def calc_significance_offset(samples1, samples2, axis=0):
    diff = np.abs(
        np.percentile(samples1, q=50, axis=axis)
        - np.percentile(samples2, q=50, axis=axis)
    )
    norm = np.median([
        np.abs(
            np.percentile(samples1, q=84, axis=axis)
            - np.percentile(samples1, q=50, axis=axis)
        ),
        np.abs(
            np.percentile(samples1, q=16, axis=axis)
            - np.percentile(samples1, q=50, axis=axis)
        )
    ], axis=axis)
    return diff / norm

def num_to_str(num, unit=None, log=False, precision=3):
    """Convert num to a formatted string with precision, converted to
    unit and with all '.' replaced by 'p'."""
    units = {
        None: 1,
        'd': 10,
        'c': 100,
        'k': 1000,
        'M': 1e6,
        'G': 1e9,
        'T': 1e12,
        'P': 1e15
    }
    if unit not in units.keys():
        raise ValueError(f'unit should be in {units.keys()}')
    if log:
        n = np.log10(num) / units[unit]
    else:
        n = num / units[unit]

    if n % 1 == 0:
        significand = ''
    else:
        significand = f'p{format(n % 1, f".{precision}f").replace("0.", "")}'

    res = f'{format(n // 1, ".0f")}{significand}{unit}'.replace('None', '')

    return res


def data_to_median_bins(data, medians, n_max=20):
    """Bin data to obtain medians."""
    ids_sorted = np.argsort(data)
    data_sorted = data[ids_sorted]

    # get the values in data_sorted that are closest to medians
    ids_medians = np.array([
        np.argmin(np.abs(data_sorted - med)) for med in medians
    ])
    # lowest and highest value need to be compared to start and end of array
    ids_lower = np.append([0], ids_medians)
    ids_higher = np.append(ids_medians, [len(data_sorted) - 1])

    # get the maximum allowed distance so there is no overlap between bins
    dist = np.array([np.min([
        (ids_medians[i] - ids_lower[i]) // 2,
        (ids_higher[i + 1] - ids_medians[i]) // 2,
        n_max]) for i in range(len(ids_medians))
    ])

    bin_ids = np.array([
        np.arange(ids_med - d, ids_med + d + 1, 1)
        for ids_med, d in zip(ids_medians, dist)
    ])

    # convert to the unsorted array
    bin_ids = np.array([ids_sorted[bin_id] for bin_id in bin_ids], dtype=object)
    medians_actual = np.array([
        np.median(data[bin_ids[i]]) for i in range(len(bin_ids))
    ])
    return bin_ids, medians_actual


def bin_centers(bins, log=False):
    """Return the center position of bins, with bins along axis -1."""
    if log:
        centers = (
            (bins[..., 1:] - bins[..., :-1])
            / (np.log(bins[..., 1:]) - np.log(bins[..., :-1]))
        )
    else:
        centers = 0.5 * (bins[..., :-1] + bins[..., 1:])

    return centers


def resample_to_bins(bins, f, x, upsample=20, log=False):
    """Resample data values f at x to bins.

    Interpolate f(x) and average the function value in bins, if bins
    outside x, extrapolate.

    Parameters
    ----------
    bins : array-like
        bins to calculate f(x) for
    f : array-like
        data values
    x : array-like
        data coordinates
    upsample : int
        factor by which we will upsample function over bins
    log : bool
        logarithmic or linear binning

    Returns
    -------
    f(bins)

    """
    f_interp = interp.interp1d(
        x, f, axis=-1, bounds_error=False, fill_value='extrapolate'
    )
    f_bins = avg_in_bins(
        function=f_interp, bins=bins, upsample=upsample, log=log
    )
    return f_bins


def avg_in_bins(function, bins, upsample=20, log=False, **kwargs):
    """Return the average of function in bins upsampled by factor upsample.

    Parameters
    ----------
    function : callable
        function to average, takes bins as first argument and kwargs as others
    bins : array-like
        bins over which function will be averaged
    upsample : int
        factor by which we will upsample function over bins
    log : bool
        logarithmic or linear binning
    kwargs : dict
        extra arguments to pass to function

    Returns
    -------
    function_avg : array-like
        average of function over bins
    """
    # divide each bin in upsample bins
    if log:
        bins_up = np.logspace(
            np.log10(bins[:-1]), np.log10(bins[1:]),
            upsample + 1
        ).T
    else:
        bins_up = np.linspace(
            bins[:-1], bins[1:],
            upsample + 1
        ).T

    centers = bin_centers(bins_up, log=log)
    function_up = function(centers.flatten(), **kwargs).reshape(centers.shape)
    # average the upsampled function to bins
    function_avg = (function_up * np.diff(bins_up)).sum(axis=-1) / np.diff(bins)

    return function_avg


def rolling_window(data, block):
    """Return the data rolled from 0 to block along the final axis.

    Parameters
    ----------
    data : array-like
        data to roll
    block : int
        number of times to roll data

    Returns
    -------
    data_rolled : (..., block) array
        data rolled by 0, block along final axis
    """
    if not type(block) is int:
        raise TypeError('block should be int.')
    shape = data.shape[:-1] + (data.shape[-1] - block + 1, block)
    strides = data.strides + (data.strides[-1],)
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)


def despike(arr, n1=2, n2=20, block=10, fill='mean'):
    """Remove spikes from arr by sigma_clipping in 2 passes with n1 sigma
    and n2 sigma.

    See http://ocefpaf.github.io/python4oceanographers/blog/2013/05/20/spikes/

    Parameters
    ----------
    arr : (n,) array
        data to despike
    n1 : float
        number of sigma to clip initial pass
    n2 : float
        number of sigma to clip on second pass
    block : int
        size of window to include
    fill : ['mean', None]
        fill spikes with mean or NaN

    Returns
    -------
    arr : (n,) array
        data with spikes filled
    """
    if len(arr.shape) > 1:
        raise ValueError('arr should be 1D array.')
    fill_options = ['mean', None]
    if fill not in fill_options:
        raise ValueError(f'fill should be in {fill_options}')

    offset = arr.min()
    arr -= offset
    data = arr.copy()
    roll = rolling_window(data, block)
    roll = np.ma.masked_invalid(roll)
    std = n1 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    mask = (np.abs(data - mean.filled(fill_value=np.NaN)) >
            std.filled(fill_value=np.NaN))
    data[mask] = np.NaN
    # Pass two: recompute the mean and std without the flagged values from pass
    # one now removing the flagged data.
    roll = rolling_window(data, block)
    roll = np.ma.masked_invalid(roll)
    std = n2 * roll.std(axis=1)
    mean = roll.mean(axis=1)
    # Use the last value to fill-up.
    std = np.r_[std, np.tile(std[-1], block - 1)]
    mean = np.r_[mean, np.tile(mean[-1], block - 1)]
    mask = (np.abs(arr - mean.filled(fill_value=np.NaN)) >
            std.filled(fill_value=np.NaN))
    if fill is None:
        arr[mask] = np.NaN
    elif fill == 'mean':
        arr[mask] = np.mean(
            [
                arr[np.where(mask)[0] - 1], arr[np.where(mask)[0] + 1]
            ], axis=0)

    return arr + offset


def arrays_to_coords(*xi):
    """
    Convert a set of N 1-D coordinate arrays to a regular coordinate grid of
    dimension (npoints, N) for the interpolator
    """
    # the meshgrid matches each of the *xi to all the other *xj
    Xi = np.meshgrid(*xi, indexing='ij')

    # now we create a column vector of all the coordinates
    coords = np.concatenate([X.reshape(X.shape + (1,)) for X in Xi], axis=-1)

    return coords.reshape(-1, len(xi))


def matched_arrays_to_coords(*xi):
    """
    Convert a set of n arrays of shape (i_0, ..., i_n) to a list
    of (i_0*...*i_n, n) coordinates
    """
    # get single matching shape between all arrays
    # we expect sane inputs, i.e. arrays are already reshaped
    # to have empty dimension axes set to 1
    b = np.broadcast(*xi)
    shape = np.array(b.shape)
    arrays = []
    for x in xi:
        arrays.append(
            np.tile(x, (shape / np.array(x.shape)).astype(int)).flatten()
        )

    return np.array(arrays).T


def Integrate(y, x, axis=-1):
    """
    Integrate array at sample points x over axis using Simpson
    integration

    Parameters
    ----------
    y : array

    """
    y_new = np.nan_to_num(y)
    # the last interval is computed with trapz
    result = intg.simps(y=y_new, x=x, axis=axis, even='first')
    # result = np.trapz(y=y_new, x=x, axis=axis)

    return result


def m_h(rho, r_range, r_0=None, r_1=None, axis=-1):
    """
    Calculate the mass of the density profile between over r_range, or between
    r_0 and r_1.

    Parameters
    ----------
    rho : array
        density profile
    r_range : array
        radius
    r_0 : float
        start radius of integration interval
    r_1 : float
        end radius of integration interval

    Returns
    -------
    m_h : float
        mass
    """
    int_range = r_range
    int_rho = rho

    if r_0 is not None:
        idx_0 = np.argmin(np.abs(r_0 - int_range))
        int_range = int_range[..., idx_0:]
        int_rho = int_rho[..., idx_0:]

    if r_1 is not None:
        idx_1 = np.argmin(np.abs(r_1 - int_range))
        int_range = int_range[..., :idx_1]
        int_rho = int_rho[..., :idx_1]

    return 4 * np.pi * Integrate(int_rho * int_range**2, int_range, axis=axis)


def cum_m(rho_r, r_range):
    """
    Returns the cumulative mass profile

    Parameters
    ----------
    rho_r : (...,r) array
        density profile
    r_range : (...,r) array
        radial range

    Returns
    -------
    cum_m : (...,r-1) array
        cumulative mass profile
    """
    r = r_range.shape[-1]

    result = np.array([
        m_h(rho_r[..., :idx], r_range[..., :idx], axis=-1)
        for idx in np.arange(1, r+1)
    ])

    return result


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
def mx_from_my(my, ry, cy, rho_x):
    """
    Convert a halo with mass m_y and concentration c_y at r_y
    to the mass m_x corresponding to mean overdensity rho_x

    Parameters
    ----------
    my : array
        halo masses
    ry : array
        halo radii
    cy : array
        halo concentrations at r_y
    rho_x : array
        mean overdensity to solve for

    Returns
    -------
    mx : array
        masses of haloes my at mean overdensity rho_x
    """
    def solve_mass(rx):
        mx = radius_to_mass(rx, rho_x)
        m_enc = m_nfw(r=rx, m_x=my, r_x=ry, c_x=cy)

        return m_enc - mx

    rx = opt.brentq(solve_mass, 0.1 * ry, 10 * ry)
    mx = radius_to_mass(rx, rho_x)
    return mx


def mass_to_radius(m, rho_mean):
    """
    Calculate radius of a region of space from its mass.

    Parameters
    ----------
    m : float or array of floats
        Masses
    rho_mean : float
        The mean density of the universe

    Returns
    -------
    r : float or array of floats
        The corresponding radii to m

    Notes
    -----
    The units of r don't matter as long as they are consistent with
    rho_mean.
    """
    return (3. * m / (4 * np.pi * rho_mean))**(1. / 3)


def radius_to_mass(r, rho_mean):
    """
    Calculates mass of a region of space from its radius.

    Parameters
    ----------
    r : float or array of floats
        Radii
    rho_mean : float
        The mean density of the universe

    Returns
    -------
    m : float or array of floats
        The corresponding masses in r

    Notes
    -----
    The units of r don't matter as long as they are consistent with
    rho_mean.
    """
    return 4 * np.pi * r ** 3 * rho_mean / 3.


@np.vectorize
def m200m_dmo_from_mx_dmo(mx, rx, z, rho_mz, c200m_interp):
    """Convert the mass mx at rx to the overdensity mass m200m for an NFW
    halo with concentration-mass relation interpolator c200m_interp.

    Parameters
    ----------
    mx : array-like
        halo masses
    rx : array-like
        radii where mass is computed
    z : array-like
        redshifts
    rho_mz : array-like
        mean matter density at z
    c200m_interp : tremulator.Interpolator object
        interpolator for c200m(m200m) relation

    Returns
    -------
    m200m_dmo : array-like
        halo masses m200m
    """
    def solve_mass(m200m_dmo):
        c200m_dmo = c200m_interp.predict((np.log10(m200m_dmo), z)).reshape(-1)
        r200m_dmo = mass_to_radius(m200m_dmo, 200 * rho_mz)
        m_enc = m_nfw(r=rx, m_x=m200m_dmo, r_x=r200m_dmo, c_x=c200m_dmo)
        return m_enc - mx

    m200m_dmo = opt.brentq(solve_mass, 0.5 * mx, 20 * mx)
    return m200m_dmo


@np.vectorize
def mx_dmo_to_m200m_dmo(mx, rx, rs, z, rho_m):
    """Convert the mass mx at rx with concentration cx to the overdensity
    mass m200m for an NFW halo.

    Parameters
    ----------
    mx : array-like
        halo masses
    rx : array-like
        radii where mass is computed
    rs : array-like
        scale radius
    z : array-like
        redshifts
    rho_m : array-like
        mean matter density at z

    Returns
    -------
    m200m_dmo : array-like
        halo masses m200m

    """
    def solve_mass(m200m_dmo):
        r200m_dmo = mass_to_radius(m200m_dmo, 200 * rho_m)
        c200m_dmo = r200m_dmo / rs
        m_enc = m_nfw(r=rx, m_x=m200m_dmo, r_x=r200m_dmo, c_x=c200m_dmo)
        return m_enc - mx

    m200m_dmo = opt.brentq(solve_mass, 1e-10 * mx, 1e10 *  mx)
    return m200m_dmo


@np.vectorize
def mr_to_mx(m_func, rho_x):
    """Convert the mass mx at rx with concentration cx to the overdensity
    mass m200m for an NFW halo.

    Parameters
    ----------
    m_func : callable
        enclosed mass
    rho_x : float
        mean enclosed overdensity

    Returns
    -------
    m_x : array-like
        halo mass within radius that enclosed rho_x

    """
    def solve_mass(log10_m_x):
        r_x = mass_to_radius(10**log10_m_x, rho_x)
        log10_m_enc = np.log10(m_func(r_x))
        return log10_m_enc - log10_m_x

    log10_m_x = opt.brentq(solve_mass, 1, 20)
    return 10**log10_m_x
