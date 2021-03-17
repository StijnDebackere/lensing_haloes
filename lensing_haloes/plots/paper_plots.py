import pdb
from pathlib import Path

import asdf
import getdist
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import plotting_tools.saving as saving
import scipy.interpolate as interp
import scipy.optimize as opt
from getdist import plots
from matplotlib.legend_handler import HandlerTuple

import lensing_haloes.cosmo.cosmo as cosmo
import lensing_haloes.data.observational_data as obs_data
import lensing_haloes.halo.abundance as abundance
import lensing_haloes.halo.model as halo_model
import lensing_haloes.halo.profiles as profs
import lensing_haloes.lensing.generate_mock_lensing as mock_lensing
import lensing_haloes.results as results
import lensing_haloes.settings as settings
import lensing_haloes.util.plot as plot
import lensing_haloes.util.tools as tools

TABLE_DIR = settings.TABLE_DIR
FIGURE_DIR = settings.FIGURE_DIR
PAPER_DIR = settings.PAPER_DIR
OBS_DIR = settings.OBS_DIR
LIGHT = True
if LIGHT:
    bw_color = 'black'
else:
    bw_color = 'white'
saving.paper_style(light=LIGHT)

dashes_WL = (15, 10)
dashes_WL_rs = (2, 2, 2, 2)
dashes_X = (5, 5)
dashes_ap = (10, 10)


def plot_fit_parameters_fit(
        dataset='croston+08', n_bins=4, z_ls=[0.1, 0.43, 1.0, 2.0], dlog10r=2,
        outer_norm=None):
    """Plot the comparison between the linear fit and the true
    best-fitting model parameters.

    """
    z_ls = np.atleast_1d(z_ls)
    # prepare figure
    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(10, 9, forward=True)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax_cb = fig.add_axes([0.6, 0.425, 0.3, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.viridis_r, N=len(z_ls))

    ls_rt = []
    ls_a = []
    ls_rt_o = []
    ls_a_o = []
    ls_rt_f = []
    ls_a_f = []
    for idx_z, z_l in enumerate(z_ls):
        results_all = halo_model.get_rho_gas_fits_all(
            r_range=None, z_l=z_l, dlog10r=dlog10r, datasets=[dataset],
            outer_norm=outer_norm)
        results_bins = halo_model.get_rho_gas_fits_bins(
            r_range=None, z_l=z_l, dlog10r=dlog10r, datasets=[dataset],
            n_bins=n_bins, outer_norm=outer_norm)

        z = results_all[dataset]['z'][:]
        m500c_all = results_all[dataset]['m500c'][:]
        log10_rt_all = np.array([r['opt_prms']['log10_rt'] for r in results_all[dataset]['fit_results']])
        alpha_all = np.array([r['opt_prms']['alpha'] for r in results_all[dataset]['fit_results']])

        m500c_bins = results_bins[dataset]['m500c_bins'][:]
        log10_rt_bins = np.array([r['opt_prms']['log10_rt'] for r in results_bins[dataset]['fit_results']])
        alpha_bins = np.array([r['opt_prms']['alpha'] for r in results_bins[dataset]['fit_results']])

        def linear_fit(log10_m500c, a, b):
            return a * (log10_m500c - b)

        # opt_rt, pcov = opt.curve_fit(linear_fit, xdata=np.log10(m500c), ydata=log10_rt_all, maxfev=1000)
        # opt_a, pcov = opt.curve_fit(linear_fit, xdata=np.log10(m500c), ydata=alpha_all, maxfev=5000)
        opt_rt, pcov = opt.curve_fit(linear_fit, xdata=np.log10(m500c_bins), ydata=log10_rt_bins, maxfev=1000)
        opt_a, pcov = opt.curve_fit(
            linear_fit, xdata=np.log10(m500c_bins), ydata=alpha_bins,
            p0=[-0.5, 15.5], maxfev=5000)

        m = np.linspace(13.5, 15.5, 50)
        l_rt_f, = ax.plot(10**m, linear_fit(m, *opt_rt), c=cmap(idx_z))
        l_a_f, = ax.plot(10**m, linear_fit(m, *opt_a), c=cmap(idx_z))

        l_f, = ax.plot(m500c_bins, log10_rt_bins, c=bw_color, lw=3)
        ax.plot(m500c_bins, alpha_bins, c=bw_color, lw=3)

        rt_rel_diff = np.abs(log10_rt_all / linear_fit(np.log10(m500c_all), *opt_rt) - 1)
        a_rel_diff = np.abs(alpha_all / linear_fit(np.log10(m500c_all), *opt_a) - 1)

        outliers_rt = rt_rel_diff > 1.5
        outliers_a = a_rel_diff > 1.5

        l_rt, = ax.plot(
            m500c_all[~outliers_rt], log10_rt_all[~outliers_rt],
            lw=0, marker='o', color=cmap(idx_z),
        )
        l_a, = ax.plot(
            m500c_all[~outliers_a], alpha_all[~outliers_a],
            lw=0, marker='^', color=cmap(idx_z),
        )
        l_rt_o, = ax.plot(
            m500c_all[outliers_rt], log10_rt_all[outliers_rt],
            marker='o', color=cmap(idx_z), lw=0, mec='r', mew=1,
        )
        l_a_o, = ax.plot(
            m500c_all[outliers_a], alpha_all[outliers_a],
            marker='^', color=cmap(idx_z), lw=0, mec='r', mew=1,
        )

        ls_rt.append(l_rt)
        ls_a.append(l_a)
        ls_rt_o.append(l_rt_o)
        ls_a_o.append(l_a_o)
        ls_rt_f.append(l_rt_f)
        ls_a_f.append(l_a_f)

    print(f'log10_rt relative differences: {rt_rel_diff}')
    print(f'alpha relative differences   : {a_rel_diff}')
    print(f'log10_rt difference sorted   : {np.argsort(rt_rel_diff)[::-1]}')
    print(f'alpha difference sorted      : {np.argsort(a_rel_diff)[::-1]}')
    print(f'masses log10_rt : {np.log10(m500c_all)[rt_rel_diff > 1.5]}')
    print(f'masses alpha : {np.log10(m500c_all)[a_rel_diff > 1.5]}')

    cb = plot.add_colorbar_indexed(
        cmap, z_ls, ticks='center',
        fig=fig, ax_cb=ax_cb, orientation='horizontal'
    )
    cb.set_label(r'$z$')

    ax.set_xscale('log')
    ax.set_ylim(-2, 3)
    ax.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())

    ax.set_xlabel(r'$m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax.set_ylabel(r'$\log_{10}r_\mathrm{t}/r_\mathrm{500c}, \alpha$')
    leg = ax.legend(
        [
            tuple(ls_rt),
            tuple(ls_a),
            tuple([ls_rt_o[-1], ls_a_o[-1]]),
            tuple(ls_rt_f),
            l_f,
        ],
        [
            r'$\log_{10}r_\mathrm{t}/r_\mathrm{500c}$',
            r'$\alpha$',
            r'outliers',
            r'best-fit',
            r'median',
        ],
        markerfirst=False, loc=1,
        handler_map={tuple: plot.HandlerTupleVertical()},
    )

    fname = (
        'fbar_r_fit_parameters_fit_'
        f'z_{tools.num_to_str(z_ls.min(), precision=2)}'
        f'-{tools.num_to_str(z_ls.max(), precision=2)}'
    )
    plt.savefig(f'{FIGURE_DIR}/{fname}.pdf', bbox_inches='tight')
    plt.show()

    mismatches = np.unique(np.concatenate([
        np.where(rt_rel_diff > 1.5)[0], np.where(a_rel_diff > 1.5)[0]]))
    return mismatches


def plot_N_mz(fname, z_bin_edges, log10_m200m_bin_edges):
    """Plot the cluster number counts as a function of mass and redshift."""
    with asdf.open(fname, lazy_load=False, copy_arrays=True) as af:
        results = af.tree

    cosmology = {
        'omega_m': results['omega_m'],
        'sigma_8': results['sigma_8'],
        'omega_b': results['omega_b'],
        'w0': results['w0'],
        'h': results['h'],
        'n_s': results['n_s'],
    }

    z = results['z_sample']
    m200m = results['true']['m200m_sample']

    N_obs_bin, _, _ = np.histogram2d(
        z, np.log10(m200m), bins=[
            z_bin_edges, log10_m200m_bin_edges
        ], density=False,
    )
    z_bins = tools.bin_centers(z_bin_edges)
    log10_m200m_bins = tools.bin_centers(log10_m200m_bin_edges)

    N_theory_bin = abundance.N_in_bins(
        z_bin_edges=z_bin_edges, m200m_bin_edges=10**log10_m200m_bin_edges,
        n_z=20, n_m=100, cosmo=cosmo.cosmology(**cosmology),
        A_survey=results['A_survey']
    )

    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax_cb = fig.add_axes([0.5, 0.825, 0.3, 0.05])

    cmap = plot.get_partial_cmap(mpl.cm.plasma_r, a=0.25, b=0.75)
    norm = mpl.colors.Normalize(vmin=z_bins.min(), vmax=z_bins.max())
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap = sm.to_rgba

    for idx_z, z_bin in enumerate(z_bins):
        c = cmap(z_bin)
        ax.errorbar(
            log10_m200m_bins, N_obs_bin[idx_z],
            yerr=np.sqrt(N_theory_bin[idx_z]),
            lw=0, marker='o', elinewidth=1, c=c
        )
        ax.plot(log10_m200m_bins, N_theory_bin[idx_z], c=c)

    cb = plt.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_label(r'$z$')

    ax.set_title(
        f'$\Omega_\mathrm{{survey}}={results["A_survey"]:d} \, \mathrm{{deg^2}}$',
        color=bw_color,
    )


    ax.set_ylim([0.5, 5e3])
    ax.set_xlabel(r'$\log_{10}m_\mathrm{200m} \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax.set_ylabel(r'$N(z, m_\mathrm{200m})$')
    # ax.set_xscale('log')
    ax.set_yscale('log')
    plt.savefig(f'{PAPER_DIR}/N_obs_vs_theory.pdf', bbox_inches='tight')
    plt.show()


def plot_rho_gas_fits_bins(
        dataset='croston+08', n_bins=3, n_r=15,
        z_l=0.43, dlog10r=2, outer_norm=None):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile for the binned density profiles.

    """
    results = halo_model.get_rho_gas_fits_bins(
        datasets=[dataset], n_bins=n_bins, z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm, n_r=n_r,
    )

    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(10, 12, forward=True)
    # fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.15, 0.525, 0.8, 0.4])
    ax_r = fig.add_axes([0.15, 0.125, 0.8, 0.4])
    ax_cb = fig.add_axes([0.175, 0.65, 0.45, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=n_bins, b=0.75)

    ax_r.text(
        0.95, 0.95, f'$z={z_l:.2f}$',
        va='top', ha='right',
        transform=ax_r.transAxes,
        color=bw_color, fontsize=30)
    # ax.text(
    #     0.95, 0.9, 'Croston+2008',
    #     va='center', ha='right',
    #     transform=ax.transAxes,
    #     color=bw_color, fontsize=30)
    ax_r.axhline(y=0, ls="--", c="k")
    ax_r.axhspan(-0.01, .01, facecolor=bw_color, alpha=0.3)
    ax_r.axhspan(-0.05, .05, facecolor=bw_color, alpha=0.1)

    m500c = results[dataset]['m500c_bins'][:]
    r500c = results[dataset]['r500c_bins'][:]
    ls_d = []
    for idx, res in enumerate(results[dataset]['fit_results']):
        fbar = results['omega_b'] / results['omega_m']
        rx = res['rx']
        rho_gas_fit = res['rho_gas_fit']
        rho_gas_med = res['rho_gas_med']
        rho_gas_q16 = res['rho_gas_q16']
        rho_gas_q84 = res['rho_gas_q84']

        ax.plot(rx * r500c[idx], rho_gas_fit, c=cmap(idx), lw=3)
        # lf = ax.fill_between(
        #     rx * r500c[idx], rho_gas_q16, rho_gas_q84,
        #     color=cmap(idx), alpha=0.2
        # )
        # ls_d.append((ld, lf))
        ld = ax.errorbar(
            rx * r500c[idx],
            rho_gas_med,
            yerr=(rho_gas_med - rho_gas_q16, rho_gas_q84 - rho_gas_med),
            c=cmap(idx),
            marker="o",
            elinewidth=1,
            lw=0,
        )
        ls_d.append(ld)
        # ld, = ax.plot(rx * r500c[idx], rho_gas_med, c=cmap(idx), lw=2)

        ax_r.plot(
            # rx * r500c[idx], (rho_gas_med - rho_gas_fit) / ((rho_gas_q84 - rho_gas_q16)),
            rx * r500c[idx], (rho_gas_med / rho_gas_fit - 1),
            c=cmap(idx), marker='o', lw=3
        )
        # print(np.median((rho_gas_med / rho_gas_fit - 1) / (rho_gas_q84 / rho_gas_fit - rho_gas_q16 / rho_gas_fit)))
        # ax_r.plot(rx * r500c[idx], rho_gas_med / rho_gas_fit - 1, c=cmap(idx), lw=3)
        # ax_r.plot(rx * r500c[idx], rho_gas_med / rho_gas_fit - 1, c=cmap(idx), lw=2)
        # ax_r.fill_between(
        #     rx * r500c[idx], rho_gas_q16 / rho_gas_fit - 1, rho_gas_q84 / rho_gas_fit -1,
        #     color=cmap(idx), alpha=0.2
        # )

    cb = plot.add_colorbar_indexed(
        cmap_indexed=cmap, fig=fig, ax_cb=ax_cb,
        items=np.log10(m500c),
        orientation='horizontal',
    )
    cb.set_label(r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')

    ax.legend(
        [tuple(ls_d)],
        ['Croston+2008'],
        handler_map={tuple: plot.HandlerTuple(n_bins)},
        markerfirst=False, loc=1,
        labelcolor=bw_color,
    )

    ax.set_xlim(left=0.01)
    ax.set_ylim(bottom=10**13.01, top=10**15.5)
    ax_r.set_xlim(left=0.01)
    # ax_r.set_ylim(-1, 1)
    ax_r.set_ylim(-0.1, 0.1)
    ax.set_xscale('log')
    ax.set_xticklabels([])
    ax.set_yscale('log')
    ax_r.set_xscale('log')
    ax.set_ylabel(
        r'$\rho_\mathrm{gas}(r) \, [h^{2} \, \mathrm{M_\odot/Mpc^3}]$',
        labelpad=-0.5,
    )
    ax_r.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')
    ax_r.set_ylabel(
        r'$\rho_\mathrm{gas}(r) / \rho_\mathrm{gas,fit}(r) - 1$',
        labelpad=-0.5,
    )
    # ax_r.set_ylabel(
    #     r'$(\rho_\mathrm{gas}(r) - \rho_\mathrm{gas,fit}(r)) / (P_{84\%} - P_{16\%}) $',
    #     labelpad=-0.5,
    # )
    if outer_norm is not None:
        fname_append = f"_rx_{outer_norm['rx']:d}_fbar_{outer_norm['fbar']:.2f}"
    else:
        fname_append = ''
    plt.savefig(f'{PAPER_DIR}/fbar_r_fit_rho_gas_vs_true_bins{fname_append}.pdf')
    plt.show()


def plot_rho_gas_fits_all(
        datasets=['croston+08'], z_l=None, dlog10r=0,
        outer_norm=None, ids=None, plot_fit=True):
    """Plot the fractional difference between the best-fitting gas
    profile and the true gas profile for all density profiles.

    """
    results = halo_model.get_rho_gas_fits_all(
        r_range=None, z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm)

    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax_cb = fig.add_axes([0.5, 0.825, 0.375, 0.05])

    cmap = plot.get_partial_cmap(mpl.cm.plasma_r)
    norm = mpl.colors.Normalize(
        vmin=np.min([
            np.min(np.log10(results[d]['m500c'])) for d in datasets
        ]),
        vmax=np.max([
            np.max(np.log10(results[d]['m500c'])) for d in datasets
        ])
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap = sm.to_rgba


    for dataset in datasets:
        if ids is None:
            ids = np.ones(results[dataset]['m500c'].shape, dtype=int)
        for idx, (m500c, res) in enumerate(zip(
                np.asarray(results[dataset]['m500c'])[ids],
                np.asarray(results[dataset]['fit_results'])[ids])):
            rx = res['rx']
            rho_gas_fit = res['rho_gas_fit']
            rho_gas = res['rho_gas']
            rho_gas_err = res['rho_gas_err']

            c = cmap(np.log10(m500c))

            ax.plot(rx, rho_gas, c=c, lw=3, alpha=0.5)
            if plot_fit is True:
                ax.plot(rx, rho_gas_fit, c=c, lw=2)
            ax.fill_between(
                rx,
                rho_gas + rho_gas_err,
                rho_gas - rho_gas_err,
                color=c, alpha=0.2
            )

            ax_r.plot(rx, rho_gas / rho_gas_fit - 1, c=c, lw=2)
            ax_r.fill_between(
                rx,
                (rho_gas - rho_gas_err) / rho_gas_fit - 1,
                (rho_gas + rho_gas_err) / rho_gas_fit - 1,
                color=c, alpha=0.2
            )

    ax_r.axhline(y=0, ls="--", c="k")

    cb = plt.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_label(r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')

    ax.set_xlim(left=0.01)
    ax_r.set_xlim(left=0.01)
    ax_r.set_ylim(-0.5, 0.5)
    ax.set_xscale('log')
    ax.set_xticklabels([])
    ax.set_yscale('log')
    ax_r.set_xscale('log')
    ax.set_ylabel(r'$\rho_\mathrm{gas}(r) \, [h^{2} \, \mathrm{M_\odot/Mpc^3}]$')
    # ax_r.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')
    ax_r.set_xlabel(r'$r / r_\mathrm{500c}$')
    ax_r.set_ylabel(r'$\rho_\mathrm{gas}(r) / \rho_\mathrm{gas,fit}(r) - 1$')
    ax_r.set_ylim(-0.5, 0.5)
    plt.savefig(f'{FIGURE_DIR}/fbar_r_fit_rho_gas_vs_true_all.pdf', bbox_inches='tight')
    plt.show()


def plot_fbar_fits_all(
        datasets=['croston+08'], z_l=None,
        outer_norm=None,
        dlog10r=2, n_int=1000):
    """Plot the fractional difference between the best-fitting fbar
    profile and the true gas profile for all density profiles.

    """
    results = halo_model.get_rho_gas_fits_all(
        z_l=z_l, dlog10r=0, outer_norm=outer_norm)

    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(8, 10, forward=True)
    ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax_cb = fig.add_axes([0.45, 0.65, 0.425, 0.05])

    cmap = plot.get_partial_cmap(mpl.cm.plasma_r, b=0.75)
    norm = mpl.colors.Normalize(
        vmin=np.min([
            np.min(np.log10(results[d]['m500c'])) for d in datasets
        ]),
        vmax=np.max([
            np.max(np.log10(results[d]['m500c'])) for d in datasets
        ])
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap = sm.to_rgba

    ax_r.axhline(y=0, ls="--", c="k")

    for dataset in datasets:
        m500c = results[dataset]['m500c']
        z = results[dataset]['z']
        fbar = results['omega_b'] / results['omega_m']

        for idx, res in enumerate(results[dataset]['fit_results']):
            rx = res['rx']
            fbar_data = res['fbar_rx']
            fbar_err = res['fbar_rx_err']
            r500c = res['r_x']

            fbar_fit = halo_model.fbar_rx(rx=rx, **res['opt_prms'])
            c = cmap(np.log10(m500c[idx]))
            ax.plot(rx * r500c, fbar_data / fbar, c=c, lw=3, alpha=1)
            # ax.plot(rx * r500c, fbar_fit, c=c, lw=1, ls="--", alpha=0.5)
            ax.fill_between(
                rx * r500c,
                (fbar_data - fbar_err) / fbar, (fbar_data + fbar_err) / fbar,
                color=c, alpha=0.2
            )

            ax_r.plot(rx * r500c, fbar_data / fbar_fit - 1, c=c, lw=2)
            ax_r.fill_between(
                rx * r500c,
                (fbar_data - fbar_err) / fbar_fit - 1,
                (fbar_data + fbar_err) / fbar_fit -1,
                color=c, alpha=0.2
            )

    cb = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_label(r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')

    ax.set_xlim(left=1e-2)
    ax.set_ylim(1.01e-3, 1.1)
    ax_r.set_xlim(left=1e-2)
    ax_r.set_ylim(-0.5, 0.5)
    ax.set_xscale('log')
    ax.set_xticklabels([])
    ax.set_yscale('log')
    ax_r.set_xscale('log')
    ax.set_ylabel(r'$f_\mathrm{bar}(<r) / (\Omega_\mathrm{b} / \Omega_\mathrm{m})$')
    ax_r.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')
    ax_r.set_ylabel(r'$f_\mathrm{bar}(<r) / f_\mathrm{bar,fit}(<r) - 1$')
    plt.savefig(f'{FIGURE_DIR}/fbar_r_fit_fbar_vs_true_all.pdf', bbox_inches='tight')
    plt.show()


def plot_fbar_fits_bins(
        dataset='croston+08',
        n_bins=3, z_l=0.43, n_r=15, dlog10r=2, n_int=1000,
        outer_norm=None):
    """Plot the fractional difference between the best-fitting fbar
    profile and the true gas profile for the binned density profiles.

    """
    results = halo_model.get_rho_gas_fits_bins(
        datasets=[dataset], n_bins=n_bins, z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm, n_r=n_r,
    )

    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(10, 12, forward=True)
    # fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.15, 0.525, 0.8, 0.4])
    ax_r = fig.add_axes([0.15, 0.125, 0.8, 0.4])
    ax_cb = fig.add_axes([0.475, 0.65, 0.45, 0.05])

    cmap = plot.get_partial_cmap_indexed(
        mpl.cm.plasma_r, b=0.75, N=n_bins)

    ax_r.text(
        0.95, 0.95, f'$z={z_l:.2f}$',
        va='top', ha='right',
        transform=ax_r.transAxes,
        color=bw_color, fontsize=30)
    ax_r.axhline(y=0, ls="--", c="k")
    ax_r.axhspan(-0.01, .01, facecolor=bw_color, alpha=0.3)
    ax_r.axhspan(-0.05, .05, facecolor=bw_color, alpha=0.1)

    m500c = results[dataset]['m500c_bins'][:]
    r500c = results[dataset]['r500c_bins'][:]
    ls_d = []
    for idx, res in enumerate(results[dataset]['fit_results']):
        rx = res['rx']
        fbar_data = res['fbar_rx']
        fbar_err = res['fbar_rx_err']
        fbar = results['omega_b'] / results['omega_m']

        fbar_fit = halo_model.fbar_rx(rx=rx, **res['opt_prms'])

        ld = ax.errorbar(
            rx * r500c[idx], fbar_data / fbar, yerr=fbar_err / fbar,
            c=cmap(idx), marker='o', elinewidth=1, lw=0
        )
        ls_d.append(ld)
        ax.plot(rx * r500c[idx], fbar_fit / fbar, c=cmap(idx), lw=3)

        ax_r.plot(
            rx * r500c[idx], fbar_data / fbar_fit - 1,
            c=cmap(idx), marker='o', lw=3
        )
        if outer_norm is not None:
            ax.plot(outer_norm['rx'] * r500c[idx], outer_norm['fbar'], c="k", marker="o")
            ax.plot(outer_norm['rx'] * r500c[idx], outer_norm['fbar'], c="k", marker="o")


    cb = plot.add_colorbar_indexed(
        cmap_indexed=cmap, fig=fig, ax_cb=ax_cb,
        items=np.log10(m500c),
        orientation='horizontal',
    )
    cb.set_label(r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')

    ax.legend(
        [tuple(ls_d)],
        ['Croston+2008'],
        handler_map={tuple: plot.HandlerTuple(n_bins)},
        markerfirst=True, loc=2,
        labelcolor=bw_color,
    )

    ax.set_xlim(left=1e-2)
    ax.set_ylim(1.01e-2, 1.1)
    ax_r.set_xlim(left=1e-2)
    ax_r.set_ylim(-0.1, 0.1)
    ax.set_xscale('log')
    ax.set_xticklabels([])
    ax.set_yscale('log')
    ax_r.set_xscale('log')
    ax.set_ylabel(r'$f_\mathrm{bar}(<r) / (\Omega_\mathrm{b} / \Omega_\mathrm{m})$')
    ax_r.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')
    ax_r.set_ylabel(r'$f_\mathrm{bar}(<r) / f_\mathrm{bar,fit}(<r) - 1$', labelpad=-5)
    if outer_norm is not None:
        fname_append = f"_rx_{outer_norm['rx']:d}_fbar_{outer_norm['fbar']:.2f}"
    else:
        fname_append = ''
    plt.savefig(f'{PAPER_DIR}/fbar_r_fit_fbar_vs_true_bins{fname_append}.pdf')
    plt.show()


def plot_fbar_fits_rx(
        dataset='croston+08',
        n_bins=3,
        m500c=np.logspace(13.5, 15.5, 50),
        r500cs=np.array([1, 2, 4]),
        z_l=0.43, n_r=15, dlog10r=2, n_int=1000,
        outer_norm=None):
    """Plot the baryon fractions at different r500cs for the best-fitting
    model."""
    results_data = obs_data.load_datasets(datasets=[dataset], h_units=True)
    results_data = results_data[dataset]

    # get the best-fitting linear relation for log10_rt and alpha
    omega_b = 0.0493
    omega_m = 0.315
    fbar = omega_b / omega_m

    fit_prms = results.fit_observational_dataset(
        dataset=dataset, z=z_l, omega_b=omega_b, omega_m=omega_m,
        dlog10r=dlog10r, n_int=n_int, n_bins=n_bins, outer_norm=outer_norm,
        err=True, diagnostic=True, bins=True
    )
    log10_rt_prms = fit_prms['med']['log10_rt']
    alpha_prms = fit_prms['med']['alpha']
    log10_rt_prms_min = fit_prms['min']['log10_rt']
    log10_rt_prms_plus = fit_prms['plus']['log10_rt']

    log10_rt = results.linear_fit(np.log10(m500c), *log10_rt_prms)
    alpha = results.linear_fit(np.log10(m500c), *alpha_prms)
    log10_rt_plus = results.linear_fit(np.log10(m500c), *log10_rt_prms_plus)
    log10_rt_min = results.linear_fit(np.log10(m500c), *log10_rt_prms_min)

    # set up the figure
    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(10, 10, forward=True)
    # fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax_cb = fig.add_axes([0.6, 0.3, 0.2, 0.05])

    cmap = plot.get_partial_cmap_indexed(
        mpl.cm.plasma_r, N=len(r500cs), b=0.75)

    # ax.text(
    #     0.95, 0.95, f'$z={z_l:.2f}$',
    #     va='top', ha='right',
    #     transform=ax.transAxes,
    #     color=bw_color, fontsize=30)

    ld = ax.errorbar(
        results_data['m500c'][:],
        results_data['fgas_500c'] / fbar,
        xerr=results_data['m500c_err'].T,
        yerr=results_data['fgas_500c_err'].T / fbar,
        lw=0, color=cmap(0), marker='o', elinewidth=2,
    )


    for idx, rx in enumerate(r500cs):
        fbar_fit = halo_model.fbar_rx(
            rx=rx, log10_rt=log10_rt, alpha=alpha,
            fbar=fbar, fbar0=0)
        fbar_fit_plus = halo_model.fbar_rx(
            rx=rx, log10_rt=log10_rt_plus, alpha=alpha,
            fbar=fbar, fbar0=0)
        fbar_fit_min = halo_model.fbar_rx(
            rx=rx, log10_rt=log10_rt_min, alpha=alpha,
            fbar=fbar, fbar0=0)

        l, = ax.plot(m500c, fbar_fit / fbar, c=cmap(idx), lw=3, alpha=0.5)
        ax.fill_between(
            m500c, fbar_fit_min / fbar, fbar_fit_plus / fbar,
            color=cmap(idx), alpha=0.2
        )

    cb = plot.add_colorbar_indexed(
        cmap_indexed=cmap, fig=fig, ax_cb=ax_cb,
        items=r500cs, orientation='horizontal',
    )
    cb.set_label(r'$x \, r_\mathrm{500c}$')

    ax.legend(
        [ld],
        ['Croston+2008'],
        markerfirst=True, loc=2,
        labelcolor=bw_color,
    )

    ax.set_ylim(top=1.1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax.set_ylabel(r'$f_\mathrm{bar}(<x \, r_\mathrm{500c}) / (\Omega_\mathrm{b} / \Omega_\mathrm{m})$')
    plt.savefig(f'{PAPER_DIR}/fbar_r_fit_fbar_vs_true_r_bins.pdf')
    plt.show()


def plot_shear_red_profile_bins(
        z_ref=0.43,
        m500c_refs=[10**13.97, 10**14.27, 10**14.52, 10**15],
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_nbins_4_R_0p75-2p5'):
    """Plot reduced shear profiles and their best-fits for the mass-binned
    clusters.

    """
    with asdf.open(
            f'{TABLE_DIR}/observational_results_fgas_r_{model_fname_append}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree

    z = results['z']
    m500c = results['m500c']
    r500c = results['r500c']
    r200m = results['r200m_obs']

    R_bins = results['R_bins']
    R_obs = results['R_obs']
    r_range = results['r_range']
    shear_red_tot = results['shear_red_tot']
    shear_red_obs = results['shear_red_obs']
    shear_red_err = results['shape_noise']
    shear_red_WL = results['shear_red_WL']
    shear_red_WL_rs = results['shear_red_WL_rs']

    idx_z_ref = np.argmin(np.abs(results['z'] - z_ref))
    idx_m_refs = np.array([
        np.argmin(np.abs(results['m500c'] - m500c_ref))
        for m500c_ref in m500c_refs
    ])

    # set up figure style and axes
    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(10, 12, forward=True)
    ax = fig.add_axes([0.15, 0.675, 0.8, 0.25])
    ax_r = fig.add_axes([0.15, 0.125, 0.8, 0.55])
    # ax_cb = fig.add_axes([0.45, 0.65, 0.475, 0.05])
    ax_cb = fig.add_axes([0.475, 0.6, 0.4, 0.05])

    cmap = plot.get_partial_cmap_indexed(
        mpl.cm.plasma_r, N=len(m500c_refs), b=0.75)

    ls_d = []
    ls_t = []
    ls_w = []
    ls_wc = []

    for idx, (idx_m_ref, m500c_ref) in enumerate(zip(idx_m_refs, m500c_refs)):
        # plot the observed shear
        lt, = ax.plot(
            r_range, shear_red_tot[idx_z_ref, idx_m_ref], c=cmap(idx)
        )
        ld = ax.errorbar(
            R_obs, shear_red_obs[idx_z_ref, idx_m_ref],
            yerr=shear_red_err[idx_z_ref],
            marker="o", lw=0, elinewidth=1, c=cmap(idx)
        )
        lw, = ax.plot(
            r_range, shear_red_WL[idx_z_ref, idx_m_ref],
            dashes=dashes_WL, c=cmap(idx))
        lwc, = ax.plot(
            r_range, shear_red_WL_rs[idx_z_ref, idx_m_ref],
            dashes=dashes_WL_rs, c=cmap(idx))

        ls_d.append(ld)
        ls_t.append(lt)
        ls_w.append(lw)
        ls_wc.append(lwc)

        ax_r.plot(
            r_range,
            shear_red_WL[idx_z_ref, idx_m_ref] / shear_red_tot[idx_z_ref, idx_m_ref],
            dashes=dashes_WL, c=lw.get_color()
        )
        ax_r.plot(
            r_range,
            shear_red_WL_rs[idx_z_ref, idx_m_ref] / shear_red_tot[idx_z_ref, idx_m_ref],
            dashes=dashes_WL_rs, c=lwc.get_color()
        )
        ax_r.annotate(
            '',
            xy=(r500c[idx_z_ref, idx_m_ref], 0.95),
            xytext=(r500c[idx_z_ref, idx_m_ref], 0.96),
            arrowprops=dict(
                facecolor=cmap(idx), shrink=0.,
                edgecolor=bw_color,
            ),
        )

    ax_r.text(
        r500c[idx_z_ref, idx_m_ref], 0.96,
        '$r_\mathrm{500c}$', va='bottom', ha='center',
        color=bw_color, fontsize=30
    )

    ax.axvspan(
        R_bins.min(),
        R_bins.max(),
        color="g", alpha=0.2
    )
    ax_r.axvspan(
        R_bins.min(),
        R_bins.max(),
        color="g", alpha=0.2
    )

    ax_r.axhline(y=1, c="k", ls="--")
    ax_r.axhspan(0.99, 1.01, facecolor=bw_color, alpha=0.3)
    ax_r.axhspan(0.98, 1.02, facecolor=bw_color, alpha=0.1)

    cb = plot.add_colorbar_indexed(
        cmap_indexed=cmap, fig=fig, ax_cb=ax_cb,
        items=np.log10(m500c_refs),
        orientation='horizontal',
    )
    cb.set_label(
        r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$',
        labelpad=-8)

    # add the legend with data points, need horizontal spacing
    leg = ax.legend(
        [tuple(ls_d)],
        [r'"observed"'], loc=1,
        handler_map={tuple: plot.HandlerTuple(len(m500c_refs))},
        frameon=False, markerfirst=False,
        labelcolor=bw_color,
    )
    ax.add_artist(leg)

    # need empty handle to overlay with already present legend
    r = mpl.patches.Rectangle(
        (0,0), 1, 1, fill=False, edgecolor='none', visible=False
    )
    leg = ax.legend(
        [r, tuple(ls_t), tuple(ls_w), tuple(ls_wc)],
        [r"", r"true", r"NFW", r"NFW $r_\mathrm{s}$ free"],
        handler_map={tuple: plot.HandlerTupleVertical()},
        frameon=False, markerfirst=False,
        labelcolor=bw_color,
    )

    ax.set_xlim(
        left=R_bins.min() / 2,
        right=R_bins.max() + 2
    )
    ax.set_ylim(bottom=0, top=0.16)

    ax_r.set_xlim(
        left=R_bins.min() / 2,
        right=R_bins.max() + 2
    )
    ax_r.set_ylim(0.95, 1.049)

    ax_r.text(
        0.95, 0.05, f"$z={z[idx_z_ref]:.2f}$",
        va='bottom', ha='right',
        transform=ax_r.transAxes,
        color=bw_color, fontsize=30
    )

    ax.set_xticklabels([])
    ax_r.set_xlabel(r'$R \, [h^{-1} \, \mathrm{Mpc}]$')
    ax.set_ylabel(r'$g_\mathrm{T}(R)$')
    ax_r.set_ylabel(r'$g_\mathrm{T,fit}(R) / g_\mathrm{T,true}(R)$')

    fname=f'g_T_fit_{model_fname_append}_bins'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def plot_deprojected_masses_bins(
        z_ref=0.43,
        data_dir=TABLE_DIR,
        m500c_refs=[10 ** 14, 10 ** 14.5, 10 ** 15, 10 ** 15.5],
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_nbins_4_R_0p75-2p5'):
    """Plot the ratio between the best-fitting and true deprojected masses."""
    with asdf.open(
            f'{TABLE_DIR}/observational_results_fgas_r_{model_fname_append}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree

    # load the best-fitting predictions
    z = results['z']
    m500c = results['m500c']
    r500c = results['r500c']
    r200m = results['r200m_obs']
    R_bins = results['R_bins']

    r_range = results['r_range']
    mr_tot = results['m_tot']
    mr_WL = profs.m_nfw(
        r_range[None, None, :],
        m_x=results['m200m_WL'][..., None],
        r_x=results['r200m_WL'][..., None],
        c_x=results['c200m_WL'][..., None],
    )
    mr_WL_rs = profs.m_nfw(
        r_range[None, None, :],
        m_x=results['m200m_WL_rs'][..., None],
        r_x=results['r200m_WL_rs'][..., None],
        c_x=results['c200m_WL_rs'][..., None]
    )

    # prepare the axes
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax_cb = fig.add_axes([0.525, 0.875, 0.4, 0.05])

    cmap = plot.get_partial_cmap_indexed(
        mpl.cm.plasma_r, N=len(m500c_refs), b=0.75)

    idx_z_ref = np.argmin(np.abs(z - z_ref))
    idx_m_refs = np.array([
        np.argmin(np.abs(results['m500c'] - m500c_ref))
        for m500c_ref in m500c_refs
    ])

    ls_wl = []
    ls_wl_c = []
    for idx, (idx_m_ref, m500c_ref) in enumerate(zip(idx_m_refs, m500c_refs)):
        l_wl, = ax.plot(
            r_range, mr_WL[idx_z_ref, idx_m_ref] / mr_tot[idx_z_ref, idx_m_ref],
            dashes=dashes_WL, c=cmap(idx)
        )
        l_wl_c, = ax.plot(
            r_range,
            mr_WL_rs[idx_z_ref, idx_m_ref] / mr_tot[idx_z_ref, idx_m_ref],
            dashes=dashes_WL_rs, c=cmap(idx)
        )

        ls_wl.append(l_wl)
        ls_wl_c.append(l_wl_c)

        ax.annotate(
            '',
            xy=(r500c[idx_z_ref, idx_m_ref], 0.9),
            xytext=(r500c[idx_z_ref, idx_m_ref], 0.91),
            arrowprops=dict(
                facecolor=cmap(idx), shrink=0.,
                edgecolor=bw_color,
            ),
        )

    ax.axvspan(R_bins.min(), R_bins.max(), color="g", alpha=0.2)
    ax.axhline(y=1, c="k", ls="--")
    ax.axhspan(0.99, 1.01, facecolor=bw_color, alpha=0.3)
    ax.axhspan(0.95, 1.05, facecolor=bw_color, alpha=0.1)

    ax.text(
        r500c[idx_z_ref, idx_m_refs[0]], 0.915,
        '$r_\mathrm{500c}$', va='bottom', ha='center',
        color=bw_color, fontsize=30,
    )

    ax.set_xlim(0.1, 6)
    ax.set_ylim(0.9, 1.1)
    ax.set_xscale("log")

    ax.text(
        0.05, 0.05, f"$z={z[idx_z_ref]}$",
        va='center', ha='left',
        transform=ax.transAxes,
        color=bw_color, fontsize=30
    )

    cb = plot.add_colorbar_indexed(
        cmap_indexed=cmap, fig=fig, ax_cb=ax_cb,
        items=np.log10(m500c_refs),
        orientation='horizontal',
    )
    cb.set_label(
        r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$',
        labelpad=-8)

    leg = ax.legend(
        [tuple(ls_wl), tuple(ls_wl_c)],
        [r"NFW", r"NFW $r_\mathrm{s}$ free"],
        handler_map={tuple: plot.HandlerTupleVertical()},
        loc=4, frameon=False, markerfirst=False,
        labelcolor=bw_color, fontsize=30
    )
    leg.get_frame().set_linewidth(0)

    ax.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')

    ax.set_ylabel(r'$m_\mathrm{NFW}(<r)/m_\mathrm{true}(<r)$')

    fname = f'mr_enc_obs_vs_WL_reconstruction_{model_fname_append}_bins'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True)
    plt.show()


def plot_mass_ratio_m200m_obs_dmo(
        z_ref=0.43,
        data_dir=TABLE_DIR,
        model_fname_base='planck2019_z_0p43_m500c_13p5-15p5_nbins_4',
        model_fname_range='R_0p75-2p5'):
    """Plot the ratio between the best-fitting and the true m200m_obs,
    m200m_dmo and r_s.

    """
    results = {}
    fname_base = f'{TABLE_DIR}/observational_results_fgas_r_{model_fname_base}'
    model_fname_append = f'{model_fname_base}_{model_fname_range}'
    with asdf.open(
            f'{fname_base}_{model_fname_range}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree
    with asdf.open(
            f'{fname_base}_min_{model_fname_range}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results_min = af.tree
    with asdf.open(
            f'{fname_base}_plus_{model_fname_range}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results_plus = af.tree



    z = results['z']
    m500c = results['m500c']
    fbar = results['omega_b'] / results['omega_m']

    idx_z_ref = np.argmin(np.abs(z - 0.43))

    m200m_obs = results['m200m_obs']
    m200m_dmo = results['m200m_dmo']
    c200m_dmo = results['c200m_dmo']
    rs_dmo = results['r200m_dmo'] / results['c200m_dmo']
    m200m_WL = results['m200m_WL']
    rs_WL = results['r200m_WL'] / results['c200m_WL']
    m200m_WL_rs = results['m200m_WL_rs']
    c200m_WL_rs = results['c200m_WL_rs']
    rs_WL_rs = results['r200m_WL_rs'] / results['c200m_WL_rs']

    m200m_obs_min = results_min['m200m_obs']
    m200m_dmo_min = results_min['m200m_dmo']
    m200m_WL_min = results_min['m200m_WL']
    m200m_WL_rs_min = results_min['m200m_WL_rs']
    m200m_obs_plus = results_plus['m200m_obs']
    m200m_dmo_plus = results_plus['m200m_dmo']
    m200m_WL_plus = results_plus['m200m_WL']
    m200m_WL_rs_plus = results_plus['m200m_WL_rs']

    # set up style and axes
    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(10, 9, forward=True)
    dy = 0.8
    ax_obs = fig.add_axes([0.15, 0.15 + 0.2 + 0.3, 0.8, 0.3])
    ax_dmo = fig.add_axes([0.15, 0.15 + 0.2, 0.8, 0.3])
    ax_c = fig.add_axes([0.15, 0.15, 0.8, 0.2])

    # plot concentration ratio
    ax_c.axhline(y=1, c=bw_color, ls='--')
    ax_c.axhspan(0.99, 1.01, facecolor=bw_color, alpha=0.3)
    ax_c.axhspan(0.95, 1.05, facecolor=bw_color, alpha=0.1)
    ax_c.plot(
        m500c, rs_WL[idx_z_ref] / rs_dmo[idx_z_ref],
        dashes=dashes_WL
    )
    ax_c.plot(
        m500c, rs_WL_rs[idx_z_ref] / rs_dmo[idx_z_ref],
        dashes=dashes_WL_rs
    )

    ax_dmo.axhline(y=1, c="k", ls="--")
    ax_dmo.axhspan(0.99, 1.01, facecolor=bw_color, alpha=0.3)
    ax_dmo.axhspan(0.98, 1.02, facecolor=bw_color, alpha=0.1)

    ax_obs.axhline(y=1, c="k", ls="--")
    ax_obs.axhspan(0.99, 1.01, facecolor=bw_color, alpha=0.3)
    ax_obs.axhspan(0.98, 1.02, facecolor=bw_color, alpha=0.1)

    # plot obs lines
    l_WL_obs, = ax_obs.plot(
        m500c, m200m_WL[idx_z_ref] / m200m_obs[idx_z_ref],
        dashes=dashes_WL
    )
    ax_obs.fill_between(
        m500c, m200m_WL_min[idx_z_ref] / m200m_obs_min[idx_z_ref],
        m200m_WL_plus[idx_z_ref] / m200m_obs_plus[idx_z_ref],
        color=l_WL_obs.get_color(), alpha=0.3
    )
    l_WL_c_obs, = ax_obs.plot(
        m500c, m200m_WL_rs[idx_z_ref] / m200m_obs[idx_z_ref],
        dashes=dashes_WL_rs
    )
    ax_obs.fill_between(
        m500c, m200m_WL_rs_min[idx_z_ref] / m200m_obs_min[idx_z_ref],
        m200m_WL_rs_plus[idx_z_ref] / m200m_obs_plus[idx_z_ref],
        color=l_WL_c_obs.get_color(), alpha=0.3
    )

    # plot dmo lines
    l_WL_dmo, = ax_dmo.plot(
        m500c, m200m_WL[idx_z_ref] / m200m_dmo[idx_z_ref],
        dashes=dashes_WL, c=l_WL_obs.get_color(),
    )
    ax_dmo.fill_between(
        m500c, m200m_WL_min[idx_z_ref] / m200m_dmo_min[idx_z_ref],
        m200m_WL_plus[idx_z_ref] / m200m_dmo_plus[idx_z_ref],
        color=l_WL_dmo.get_color(), alpha=0.3
    )
    l_WL_c_dmo, = ax_dmo.plot(
        m500c, m200m_WL_rs[idx_z_ref] / m200m_dmo[idx_z_ref],
        dashes=dashes_WL_rs, c=l_WL_c_obs.get_color()
    )
    ax_dmo.fill_between(
        m500c, m200m_WL_rs_min[idx_z_ref] / m200m_dmo_min[idx_z_ref],
        m200m_WL_rs_plus[idx_z_ref] / m200m_dmo_plus[idx_z_ref],
        color=l_WL_c_dmo.get_color(), alpha=0.3
    )

    ax_obs.text(
        0.05, 0.1, f'$z={z[idx_z_ref]:.2f}$',
        va='center', ha='left',
        transform=ax_obs.transAxes,
        color=bw_color, fontsize=30
    )

    leg = ax_dmo.legend(
        [
            l_WL_dmo,
            l_WL_c_dmo,
         ],
        [
            r"$i=$NFW",
            r"$i=$NFW $r_\mathrm{s}$ free",
        ],
        handler_map={tuple: plot.HandlerTupleVertical()},
        loc=4, ncol=1,
        frameon=False, markerfirst=False,
        labelcolor=bw_color
    )
    leg.get_frame().set_linewidth(0)

    ax_dmo.set_ylim(0.87, 1.05)
    ax_dmo.set_xscale("log")

    ax_obs.set_ylim(0.951, 1.05)
    ax_obs.set_xscale("log")

    ax_c.set_xscale('log')
    ax_c.set_xlabel(r'$m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax_c.set_ylabel(
        r'$r_{\mathrm{s},i} / r_\mathrm{s,dmo}$',
        fontsize=25, labelpad=15)

    ax_dmo.set_xticklabels([])
    ax_obs.set_xticklabels([])

    ax_dmo.set_ylabel(
        r'$m_{\mathrm{200m},i}/m_\mathrm{200m,dmo}$',
        fontsize=25, labelpad=15)
    ax_obs.set_ylabel(
        r'$m_{\mathrm{200m},i}/m_\mathrm{200m,true}$',
        fontsize=25, labelpad=0)

    fname = f'm200m_dmo+obs_vs_WL_fgas_r_{model_fname_append}'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True)
    plt.show()


def plot_mass_ratio_m200m_dmo(
        z_ref=0.43,
        data_dir=TABLE_DIR,
        model_fname_base='planck2019_z_0p43_m500c_13p5-15p5_nbins_4',
        model_fname_range='R_0p75-2p5'):
    """Plot the ratio between the best-fitting and the true m200m_obs,
    m200m_dmo and r_s.

    """
    results = {}
    fname_base = f'{TABLE_DIR}/observational_results_fgas_r_{model_fname_base}'
    model_fname_append = f'{model_fname_base}_{model_fname_range}'
    with asdf.open(
            f'{fname_base}_{model_fname_range}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree
    with asdf.open(
            f'{fname_base}_min_{model_fname_range}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results_min = af.tree
    with asdf.open(
            f'{fname_base}_plus_{model_fname_range}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results_plus = af.tree



    z = results['z']
    m500c = results['m500c']
    fbar = results['omega_b'] / results['omega_m']

    idx_z_ref = np.argmin(np.abs(z - 0.43))

    m200m_dmo = results['m200m_dmo']
    c200m_dmo = results['c200m_dmo']
    rs_dmo = results['r200m_dmo'] / results['c200m_dmo']
    m200m_WL = results['m200m_WL']
    rs_WL = results['r200m_WL'] / results['c200m_WL']
    m200m_WL_rs = results['m200m_WL_rs']
    c200m_WL_rs = results['c200m_WL_rs']
    rs_WL_rs = results['r200m_WL_rs'] / results['c200m_WL_rs']

    m200m_dmo_min = results_min['m200m_dmo']
    m200m_WL_min = results_min['m200m_WL']
    m200m_WL_rs_min = results_min['m200m_WL_rs']
    m200m_dmo_plus = results_plus['m200m_dmo']
    m200m_WL_plus = results_plus['m200m_WL']
    m200m_WL_rs_plus = results_plus['m200m_WL_rs']

    # set up style and axes
    plt.clf()
    fig = plt.figure(1)
    fig.set_size_inches(10, 9, forward=True)
    ax_dmo = fig.add_axes([0.15, 0.25, 0.8, 0.7])
    ax_c = fig.add_axes([0.15, 0.15, 0.8, 0.1])

    # plot concentration ratio
    ax_c.axhline(y=1, c=bw_color, ls='--')
    ax_c.axhspan(0.99, 1.01, facecolor=bw_color, alpha=0.3)
    ax_c.axhspan(0.95, 1.05, facecolor=bw_color, alpha=0.1)
    ax_c.plot(
        m500c, rs_WL[idx_z_ref] / rs_dmo[idx_z_ref],
        dashes=dashes_WL
    )
    ax_c.plot(
        m500c, rs_WL_rs[idx_z_ref] / rs_dmo[idx_z_ref],
        dashes=dashes_WL_rs
    )

    ax_dmo.axhline(y=1, c="k", ls="--")
    ax_dmo.axhspan(0.99, 1.01, facecolor=bw_color, alpha=0.3)
    ax_dmo.axhspan(0.98, 1.02, facecolor=bw_color, alpha=0.1)

    # plot dmo lines
    l_WL_dmo, = ax_dmo.plot(
        m500c, m200m_WL[idx_z_ref] / m200m_dmo[idx_z_ref],
        dashes=dashes_WL,
    )
    ax_dmo.fill_between(
        m500c, m200m_WL_min[idx_z_ref] / m200m_dmo_min[idx_z_ref],
        m200m_WL_plus[idx_z_ref] / m200m_dmo_plus[idx_z_ref],
        color=l_WL_dmo.get_color(), alpha=0.3
    )
    l_WL_c_dmo, = ax_dmo.plot(
        m500c, m200m_WL_rs[idx_z_ref] / m200m_dmo[idx_z_ref],
        dashes=dashes_WL_rs,
    )
    ax_dmo.fill_between(
        m500c, m200m_WL_rs_min[idx_z_ref] / m200m_dmo_min[idx_z_ref],
        m200m_WL_rs_plus[idx_z_ref] / m200m_dmo_plus[idx_z_ref],
        color=l_WL_c_dmo.get_color(), alpha=0.3
    )

    ax_dmo.text(
        0.05, 0.1, f'$z={z[idx_z_ref]:.2f}$',
        va='center', ha='left',
        transform=ax_dmo.transAxes,
        color=bw_color, fontsize=30
    )

    leg = ax_dmo.legend(
        [
            l_WL_dmo,
            l_WL_c_dmo,
         ],
        [
            r"$i=$NFW",
            r"$i=$NFW $r_\mathrm{s}$ free",
        ],
        handler_map={tuple: plot.HandlerTupleVertical()},
        loc=4, ncol=1,
        frameon=False, markerfirst=False,
        labelcolor=bw_color
    )
    leg.get_frame().set_linewidth(0)

    ax_dmo.set_ylim(0.87, 1.05)
    ax_dmo.set_xscale("log")

    ax_c.set_xscale('log')
    ax_c.set_xlabel(r'$m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax_c.set_ylabel(
        r'$r_{\mathrm{s},i} / r_\mathrm{s,dmo}$',
        fontsize=30, labelpad=15)

    ax_dmo.set_xticklabels([])

    ax_dmo.set_ylabel(
        r'$m_{\mathrm{200m},i}/m_\mathrm{200m,dmo}$',
        fontsize=30, labelpad=15)

    fname = f'm200m_dmo_vs_WL_fgas_r_{model_fname_append}'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True)
    plt.show()


def plot_mass_ratio_m_aperture(
        z_ref=0.43, R1=0.75, R2=2., Rmax=2.5,
        data_dir=TABLE_DIR,
        model_fname_base='planck2019_z_0p43_m500c_13p5-15p5_nbins_4',
        model_fname_range='R_0p75-2p5'):
    """Plot the ratio between the best-fitting aperture masses and the
    true aperture masses for the baryonic and DMO haloes.

    """
    results = {}
    fname_base = f'{TABLE_DIR}/observational_results_fgas_r_{model_fname_base}'
    with asdf.open(
            f'{fname_base}_{model_fname_range}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results['med'] = af.tree
    with asdf.open(
            f'{fname_base}_min_{model_fname_range}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results['min'] = af.tree
    with asdf.open(
            f'{fname_base}_plus_{model_fname_range}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results['max'] = af.tree

    z = results['med']['z']
    m500c = results['med']['m500c']
    fbar = results['med']['omega_b'] / results['med']['omega_m']

    idx_z_ref = np.argmin(np.abs(z - z_ref))

    for k, v in results.items():
        results[k]['M_ap_true'] = profs.sigma_mean_from_sigma(
            R=R1, Rs=results[k]['r_range'],
            sigma=results[k]['sigma_tot']
        ) * np.pi * R1**2

        results[k]['M_ap_dmo'] = profs.sigma_mean_nfw(
            R=R1, m_x=results[k]['m200m_dmo'], r_x=results[k]['r200m_dmo'],
            c_x=results[k]['c200m_dmo']
        ) * np.pi * R1**2

        results[k]['M_ap_WL'] = mock_lensing.M_ap_clowe_nfw(
            R_bins=results[k]['R_bins'], R_obs=results[k]['R_obs'],
            g_obs=results[k]['shear_red_obs'],
            R1=R1, R2=R2, Rmax=Rmax,
            sigma_crit=results[k]['sigma_crit'],
            m_x=results[k]['m200m_WL'], r_x=results[k]['r200m_WL'],
            c_x=results[k]['c200m_WL']
        )

        results[k]['M_ap_WL_rs'] = mock_lensing.M_ap_clowe_nfw(
            R_bins=results[k]['R_bins'], R_obs=results[k]['R_obs'],
            g_obs=results[k]['shear_red_obs'],
            R1=R1, R2=R2, Rmax=Rmax,
            Rs=results[k]['r_range'], sigma=results[k]['sigma_tot'],
            sigma_crit=results[k]['sigma_crit'],
            m_x=results[k]['m200m_WL'], r_x=results[k]['r200m_WL'],
            c_x=results[k]['c200m_WL']
        )

    # set up style and axes
    fig = plt.figure(figsize=(10,9))
    ax_obs = fig.add_axes([0.15, 0.65, 0.8, 0.3])
    ax_dmo = fig.add_axes([0.15, 0.15, 0.8, 0.5])

    # ax_cb = fig.add_axes([0.3, 0.83, 0.4, 0.05])

    ax_dmo.axhline(y=1, c="k", ls="--")
    ax_dmo.axhspan(0.99, 1.01, facecolor=bw_color, alpha=0.3)
    ax_dmo.axhspan(0.98, 1.02, facecolor=bw_color, alpha=0.1)

    ax_obs.axhline(y=1, c="k", ls="--")
    ax_obs.axhspan(0.99, 1.01, facecolor=bw_color, alpha=0.3)
    ax_obs.axhspan(0.98, 1.02, facecolor=bw_color, alpha=0.1)
    # plot obs results
    l_WL, = ax_obs.plot(
        m500c,
        results['med']['M_ap_WL'][idx_z_ref] / results['med']['M_ap_true'][idx_z_ref],
        dashes=dashes_WL
    )
    l_WL_rs, = ax_obs.plot(
        m500c,
        results['med']['M_ap_WL_rs'][idx_z_ref] / results['med']['M_ap_true'][idx_z_ref],
        dashes=dashes_WL_rs
    )

    ax_obs.fill_between(
        m500c,
        results['min']['M_ap_WL_rs'][idx_z_ref] / results['min']['M_ap_true'][idx_z_ref],
        results['max']['M_ap_WL_rs'][idx_z_ref] / results['max']['M_ap_true'][idx_z_ref],
        color=l_WL.get_color(), alpha=0.3
    )

    # plot DMO results
    l_t, = ax_dmo.plot(
        m500c,
        results['med']['M_ap_true'][idx_z_ref] / results['med']['M_ap_dmo'][idx_z_ref],
        ls='-', c=bw_color, lw=1
    )
    l_WL_dmo, = ax_dmo.plot(
        m500c,
        results['med']['M_ap_WL'][idx_z_ref] / results['med']['M_ap_dmo'][idx_z_ref],
        c=l_WL.get_color(), dashes=dashes_WL
    )
    l_WL_rs_dmo, = ax_dmo.plot(
        m500c,
        results['med']['M_ap_WL_rs'][idx_z_ref] / results['med']['M_ap_dmo'][idx_z_ref],
        c=l_WL_rs.get_color(), dashes=dashes_WL_rs
    )

    ax_dmo.fill_between(
        m500c,
        results['min']['M_ap_WL'][idx_z_ref] / results['min']['M_ap_dmo'][idx_z_ref],
        results['max']['M_ap_WL'][idx_z_ref] / results['max']['M_ap_dmo'][idx_z_ref],
        color=l_WL.get_color(), alpha=0.3
    )

    # plot other ratios
    l_WL_200m, = ax_obs.plot(
        m500c,
        results['med']['m200m_WL'][idx_z_ref] / results['med']['m200m_obs'][idx_z_ref],
        c=bw_color, alpha=0.5, dashes=dashes_WL
    )
    l_WL_rs_200m, = ax_obs.plot(
        m500c,
        results['med']['m200m_WL_rs'][idx_z_ref] / results['med']['m200m_obs'][idx_z_ref],
        c=bw_color, alpha=0.5, dashes=dashes_WL_rs
    )
    l_WL_200m, = ax_dmo.plot(
        m500c,
        results['med']['m200m_WL'][idx_z_ref] / results['med']['m200m_dmo'][idx_z_ref],
        c=bw_color, alpha=0.5, dashes=dashes_WL
    )
    l_WL_rs_200m, = ax_dmo.plot(
        m500c,
        results['med']['m200m_WL_rs'][idx_z_ref] / results['med']['m200m_dmo'][idx_z_ref],
        c=bw_color, alpha=0.5, dashes=dashes_WL_rs
    )

    ax_dmo.text(
        0.95, 0.05,
        (
            f'$R = {R1:.2f} \, h^{{-1}} \, \mathrm{{Mpc}}$'
            # f'$R_1 = {R1:.2f} \, h^{{-1}} \, \mathrm{{Mpc}}$ \n'
            # f'$R_2 = {R2:.2f} \, h^{{-1}} \, \mathrm{{Mpc}}$ \n'
            # f'$R_\mathrm{{max}} = {Rmax:.2f} \, h^{{-1}} \, \mathrm{{Mpc}}$'
        ),
        va='bottom', ha='right',
        transform=ax_dmo.transAxes,
        color=bw_color, fontsize=30
    )

    ax_obs.text(
        0.05, 0.05, f'$z={z[idx_z_ref]:.2f}$',
        va='bottom', ha='left',
        transform=ax_obs.transAxes,
        color=bw_color, fontsize=30
    )

    idx_ann = int(0.25 * len(m500c))
    ax_dmo.annotate(
        r'$\frac{M_\mathrm{true}(<R)}{M_\mathrm{dmo}(<R)}$',
        (
            m500c[idx_ann], results['med']['M_ap_true'][idx_z_ref][idx_ann]
            / results['med']['M_ap_dmo'][idx_z_ref][idx_ann]
        ),
        # (-50, -10), textcoords='offset points',
        (0.015, 0.6), textcoords=ax_dmo.transAxes,
        color=bw_color, fontsize=30,
        ha='left', va='center',
        arrowprops=dict(
            facecolor=bw_color, shrink=0.,
            edgecolor=bw_color,
        ),
    )


    leg = ax_obs.legend(
        [
            tuple([l_WL_dmo, l_WL_rs_dmo]),
            tuple([l_WL_200m, l_WL_rs_200m]),
         ],
        [
            r"$M_{\zeta_\mathrm{c},i}(<R)$",
            r"$m_{\mathrm{200m},i}$",
        ],
        handler_map={tuple: plot.HandlerTupleVertical()},
        loc='upper center', ncol=2,
        frameon=False, markerfirst=False,
        labelcolor=bw_color,
    )
    leg.get_frame().set_linewidth(0)
    leg = ax_dmo.legend(
        [
            tuple([l_WL_dmo, l_WL_200m]),
            tuple([l_WL_rs_dmo, l_WL_rs_200m]),
         ],
        [
            r"$i=$NFW",
            r"$i=$NFW $r_\mathrm{s}$ free",
        ],
        handler_map={tuple: plot.HandlerTupleVertical()},
        loc='upper center', ncol=2,
        frameon=False, markerfirst=False,
        labelcolor=bw_color,
    )
    leg.get_frame().set_linewidth(0)

    ax_dmo.set_ylim(0.87, 1.05)
    ax_dmo.set_xscale("log")
    ax_obs.set_ylim(0.95, 1.05)

    ticks = ax_obs.get_yticklabels()
    ticks[0].set_visible(False)

    ax_obs.set_xscale("log")
    ax_obs.set_xticklabels([])

    ax_dmo.set_xlabel(r'$m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax_dmo.set_ylabel(
        r'$M_{\zeta_\mathrm{c},i}(<R)/M_\mathrm{dmo}(<R)$',
        fontsize=25
    )
    ax_obs.set_ylabel(
        r'$M_{\zeta_\mathrm{c},i}(<R)/M_\mathrm{true}(<R)$',
        fontsize=25
    )

    fname = (
        f'm_aperture_{model_fname_base}_{model_fname_range}'
        f'_R1_{tools.num_to_str(R1, precision=2)}'
        f'_R2_{tools.num_to_str(R2, precision=2)}'
        f'_Rm_{tools.num_to_str(Rmax, precision=2)}'
    )
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def abs_to_rel(maps, means):
    """Convert absolute cosmological parameters to relative ones."""
    means = np.atleast_1d(means)
    return maps / means - 1


def sigma_8_to_S8(maps, alpha=0.2, idx_om=0, idx_s8=1):
    """Convert sigma_8 to S8"""
    maps_S8 = np.copy(maps)
    maps_S8[..., idx_s8] = maps[..., idx_s8] * (maps[..., idx_om] / 0.3)**alpha
    return maps_S8


def plot_cosmological_contours(
        fname, methods, mcut, stage='III',
        res='res_gaussian', alpha=0.2, S8=True, relative=True,
        bounds=None):
    """Plot the 2D contours for method and mcut in fnames."""
    methods_info = {
        'true': {
            'label': 'true',
            'ls': '--',
            'color': bw_color,
            'args': {'alpha': 1.},
            'filled': False
        },
        'WL': {
            'label': 'NFW',
            'ls': '-',
            'color': mpl.cm.tab10.colors[0],
            'args': {'alpha': 0.6},
            'filled': True
        },
        'WL_c': {
            'label': r'NFW $r_\mathrm{s}$ free',
            'ls': '-',
            'color': mpl.cm.tab10.colors[1],
            'args': {'alpha': 0.6},
            'filled': True
        },
    }
    if bounds is None:
        if stage is None:
            raise ValueError('if no bounds are provided, need to specify stage')
        elif stage == 'IV':
            bounds = np.array([
                [-0.15, 0.05], [-0.04, 0.005], [-0.05, 0.16]])
        elif stage == 'III':
            bounds = np.array([
                [-0.39, 0.25], [-0.049, 0.019], [-0.49, 0.49]])

    results = dict((m, {}) for m in methods)
    with asdf.open(fname, copy_arrays=True, lazy_load=False) as af:
        A_add = tools.num_to_str(af['A_survey'], unit='k', precision=1)
        mcut_add = tools.num_to_str(mcut, precision=2)
        means = np.array([af['omega_m'], af['sigma_8'], af['w0']])
        means = sigma_8_to_S8(means, alpha=alpha)
        for method in methods:
            m = af[method][mcut][res]['maps']
            if S8:
                m = sigma_8_to_S8(m, alpha=alpha)
            if relative:
                m = abs_to_rel(m, means)
            results[method]['maps'] = m
            results[method]['loglikes'] = af[method][mcut][res]['fun']

    if S8:
        s8_label = '\Delta S_8 / S_8'
    else:
        s8_label = '\Delta \sigma_8 / \sigma_8'
    if relative:
        markers = {'om': 0, 's8': 0, 'w0': 0}
    else:
        markers = {'om': means[0], 's8': means[1], 'w0': means[2]}

    samples = {
        method : getdist.mcsamples.MCSamples(
            samples=results[method]['maps'],
            names=['om', 's8', 'w0'],
            labels=[
                '\Delta \Omega_\mathrm{m} / \Omega_\mathrm{m}',
                s8_label,
                '\Delta w_0 / w_0'
            ],
            label=methods_info[method]['label']) for method in methods
    }

    # need to manually change the smoothing scales for the chains
    # only seems to work in this way, kwargs to plotter don't work
    for k, v in samples.items():
        v.smooth_scale_1D = 0.5
        v.smooth_scale_2D = 0.5

    # Triangle plot with S8
    g = plots.get_subplot_plotter(
        width_inch=12,
        subplot_size=3.9,
        subplot_size_ratio=1
    )
    g.settings.rc_sizes()
    g.settings.axis_marker_lw = 2
    g.settings.linewidth = 2

    g.triangle_plot(
        [sample for method, sample in samples.items()],
        contour_colors=[methods_info[method]['color'] for method in samples.keys()],
        contour_ls=[methods_info[method]['ls'] for method in samples.keys()],
        contour_args=[methods_info[method]['args'] for method in samples.keys()],
        filled=[methods_info[method]['filled'] for method in samples.keys()],
        markers=markers,
    )

    axs = g.subplots

    if bounds is not None:
        # set limits
        axs[0, 0].set_xlim(bounds[0])
        axs[1, 1].set_xlim(bounds[1])
        axs[2, 2].set_xlim(bounds[2])

        axs[1, 0].set_xlim(bounds[0])
        axs[1, 0].set_ylim(bounds[1])

        axs[2, 0].set_xlim(bounds[0])
        axs[2, 0].set_ylim(bounds[2])

        axs[2, 1].set_xlim(bounds[1])
        axs[2, 1].set_ylim(bounds[2])

    for t in g.legend.get_texts():
        t.set_color(bw_color)

    # for ax in axs.flatten():
    #     if ax is None:
    #         continue
    #     ax.tick_params(
    #         axis='both', which='both', left=True, right=True,
    #         top=True, bottom=True
    #     )

    # S8 multiples need to be specified
    axs[1, 1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.01))
    axs[1, 1].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.002))
    axs[1, 0].yaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
    axs[1, 0].yaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.005))
    axs[2, 1].xaxis.set_major_locator(mpl.ticker.MultipleLocator(0.02))
    axs[2, 1].xaxis.set_minor_locator(mpl.ticker.MultipleLocator(0.005))

    plt.minorticks_on()
    # turn off minor ticks and spines on 1D histograms
    axs[0, 0].tick_params(axis='y', which='minor', left=False)
    axs[1, 1].tick_params(axis='y', which='minor', left=False)
    axs[2, 2].tick_params(axis='y', which='minor', left=False)
    axs[0, 0].spines['top'].set_visible(False)
    axs[0, 0].spines['left'].set_visible(False)
    axs[0, 0].spines['right'].set_visible(False)
    axs[1, 1].spines['top'].set_visible(False)
    axs[1, 1].spines['right'].set_visible(False)
    axs[2, 2].spines['top'].set_visible(False)
    axs[2, 2].spines['right'].set_visible(False)

    if S8:
        s8_add = 'S8'
    else:
        s8_add = 's8'
    if relative:
        rel_add = 'rel'
    else:
        rel_add = 'abs'

    prms = {}
    for method in methods:
        if stage == 'III':
            prec = [2, 3, 2]
        elif stage == 'IV':
            prec = [3, 4, 3]
        else:
            prec = [2, 3, 2]

        med = np.percentile(results[method]['maps'], 50, axis=0)
        q16 = np.percentile(results[method]['maps'], 16, axis=0)
        q84 = np.percentile(results[method]['maps'], 84, axis=0)

        om_med = np.round(med[0], prec[0])
        om_q16 = np.round(np.abs(med[0] - q16[0]), prec[0])
        om_q84 = np.round(np.abs(med[0] - q84[0]), prec[0])

        s8_med = np.round(med[1], prec[1])
        s8_q16 = np.round(np.abs(med[1] - q16[1]), prec[1])
        s8_q84 = np.round(np.abs(med[1] - q84[1]), prec[1])

        w0_med = np.round(med[2], prec[2])
        w0_q16 = np.round(np.abs(med[2] - q16[2]), prec[2])
        w0_q84 = np.round(np.abs(med[2] - q84[2]), prec[2])

        prms[method] = {
            'om_med': om_med,
            'om_q16': om_q16,
            'om_q84': om_q84,
            's8_med': s8_med,
            's8_q16': s8_q16,
            's8_q84': s8_q84,
            'w0_med': w0_med,
            'w0_q16': w0_q16,
            'w0_q84': w0_q84,
        }

        method_str = method.replace('_', '')
        print(f'\\newcommand{{\\Om{method_str}{stage}}}{{\\ensuremath{{{om_med}_{{-{om_q16}}}^{{+{om_q84}}}}}}}')
        print(f'\\newcommand{{\\S{method_str}{stage}}}{{\\ensuremath{{{s8_med}_{{-{s8_q16}}}^{{+{s8_q84}}}}}}}')
        print(f'\\newcommand{{\\w{method_str}{stage}}}{{\\ensuremath{{{w0_med}_{{-{w0_q16}}}^{{+{w0_q84}}}}}}}')

    print(f'==TABLE==')
    print('& & NFW $r_\mathrm{s}$ fixed & NFW $r_\mathrm{s}$ free & true mass \\\\')
    print('\hline')
    print('\hline')
    print(f'stage {stage}-like & $\Delta \Omega_\mathrm{{m}} / \Omega_\mathrm{{m}}$ & '
          f'\\OmWL{stage} & \\OmWLc{stage} & \\Omtrue{stage} \\\\')
    print(f'& $\Delta S_8 / S_8$ & '
          f'\\SWL{stage} & \\SWLc{stage} & \\Strue{stage} \\\\')
    print(f'& $\Delta w_0 / w_0$ & '
          f'\\wWL{stage} & \\wWLc{stage} & \\wtrue{stage} \\\\')

    fname = (
        f'fit_om_{s8_add}_w0_{rel_add}_Asurvey_{A_add}deg2'
        f'_m200m_min_{mcut_add}'
    )
    plt.savefig(
        f'{PAPER_DIR}/{fname}.pdf', transparent=True
    )
    plt.show()


def plot_cosmological_1d(
        prms_names, prms_means, prms_bounds,
        method_names, method_kwargs, samples_set,
        var_names, var_labels, var_title, var_label_kwargs, var_title_kwargs,
        method_labels=None, legend_kwargs={},
        figsize=(20, 8)):
    """Create 1D PDFs for all different prms_names in samples_set for
    different var_names.

    samples_set should be of size (method_names, var_names) with
    samples for each different method at each variation. A plot will
    be made for each parameter in prms_names:


          m1  m2m3  m1 m2 m3     m2  m1m3
          /\  /\/\  |  |  /\     /|  /\/\
    var i --------  -------- ... --------
    ...
          /\  /\/\  |  |  /\     /|  /\/\
    var 2 --------  -------- ... --------
          /\  /\/\  |  |  /\     /|  /\/\
    var 1 --------  -------- ... --------
          prm 1     prm2         prm n

    Parameters
    ----------
    prms_names : list
        parameter names that should be present in samples
    var_names : list
        different variations
    method_names : list
        different methods to obtain sample
    samples_set : list of lists
        list of size method_names with size var_names samples for each method
    var_labels : list
        label for each var
    var_title : str
        title to indicate what is being varied
    prms_means : list
        mean value for each prm in prms_names
    prms_bounds : list
        boundaries for each prm in prms_names
    method_kwargs : dict
        kwargs to fill_between for each method in method_names
    figsize : tuple
        (width, height) for the figure [inches]

    Returns
    -------
    fig, axs : Figure and list of Axes for the given plot
    """
    samples_set = np.atleast_2d(samples_set)

    prm_name_options = [p.name for p in samples_set[0, 0].paramNames.names]
    for prm_name in prms_names:
        if prm_name not in prm_name_options:
            raise ValueError(f'{prm_name} not in {prm_name_options}')

    if len(samples_set.shape) > 2:
        raise ValueError('samples_set should have 2 dimensions')
    if samples_set.shape[0] != len(method_names):
        raise ValueError('samples_set dimension 0 does not match method_names')
    if samples_set.shape[1] != len(var_names):
        raise ValueError('samples_set dimension 1 does not match var_names')

    plt.clf()
    fig = plt.figure(figsize=figsize)
    axes = []
    lower = (0.15, 0.2)
    size = (0.8, 0.6)

    # axes are filled up left to right, bottom to top
    for n in range(len(prms_names)):
        dn = size[0] / len(prms_names)
        di = size[1] / len(var_names)
        axs = [
            fig.add_axes([lower[0] + n * dn, lower[1] + i * di, 0.95 * dn, di])
            for i in range(len(var_names))
        ]
        axes.append([
            plot.set_spines_labels(
                ax=ax, left=False, right=False, bottom=not bool(i),
                top=False, labels=not bool(i)) for i, ax in enumerate(axs)])

    for idx_p, axs_p in enumerate(axes):
        for idx_v, ax in enumerate(axs_p):
            leg_patches = []
            for idx_m, method_name in enumerate(method_names):
                prm_range = np.linspace(*prms_bounds[idx_p], 200)
                patch = ax.fill_between(
                    prm_range, samples_set[idx_m, idx_v].get1DDensity(
                        prms_names[idx_p], smooth_scale_1D=0.5)(prm_range),
                    **method_kwargs[method_name]
                )
                leg_patches.append(patch)
                if prms_means is not None:
                    ax.axvline(x=prms_means[idx_p], c=bw_color, ls='--')

            if idx_v == 0:
                ax.set_xlabel(f'${samples_set[idx_m, 0].parLabel(prms_names[idx_p])}$')
            if idx_v == len(axs_p) // 2 and idx_p == 0 and var_title is not None:
                ax.text(
                    s=f'{var_title}',
                    transform=ax.transAxes,
                    **var_title_kwargs
                )

            ax.set_ylim(bottom=0)
            if idx_p == 0:
                ax.text(
                    0, 0, f'{var_labels[idx_v]}',
                    transform=ax.transAxes,
                    **var_label_kwargs
            )
            # else:
            #     ax.set_xticklabels([])
            #     ax.xaxis.set_visible(False)
            #     ax.spines['bottom'].set_visible(False)

    if method_labels is not None:
        axes[0][-1].legend(
            leg_patches, method_labels,
            **legend_kwargs,
        )


    return fig, axs


def plot_cosmological_1d_mcuts(
        fname, methods, mcuts, res='res_gaussian', alpha=0.2,
        bounds=None, stage=None):
    """Plot the marginalized 1D PDFs for different mass cuts."""
    methods_info = {
        'true': {
            'label': 'true',
            'ls': '-',
            'color': bw_color,
            'alpha': 0.6,
        },
        'WL': {
            'label': 'NFW',
            'ls': '-',
            'color': mpl.cm.tab10.colors[0],
            'alpha': 0.6,
        },
        'WL_c': {
            'label': r'NFW $r_\mathrm{s}$ free',
            'ls': '-',
            'color': mpl.cm.tab10.colors[1],
            'alpha': 0.6,
        },
    }
    if bounds is None:
        if stage is None:
            raise ValueError('if no bounds are provided, need to specify stage')
        elif stage == 'IV':
            bounds = np.array([
                [-0.15, 0.05], [-0.04, 0.005], [-0.05, 0.16]])
        elif stage == 'III':
            bounds = np.array([
                [-0.39, 0.25], [-0.049, 0.019], [-0.49, 0.49]])

    methods = [method for method in methods if method in methods_info.keys()]
    results = dict((m, {mcut: {} for mcut in mcuts}) for m in methods)
    with asdf.open(fname, copy_arrays=True, lazy_load=False) as af:
        A_add = tools.num_to_str(af['A_survey'], unit='k', precision=1)
        means = np.array([af['omega_m'], af['sigma_8'], af['w0']])
        means = sigma_8_to_S8(means, alpha=alpha)
        for method in methods:
            for mcut in mcuts:
                m = af[method][mcut][res]['maps']
                m = sigma_8_to_S8(m, alpha=alpha)
                m = abs_to_rel(m, means)
                results[method][mcut]['maps'] = m
                results[method][mcut]['loglikes'] = af[method][mcut][res]['fun']

    prms_names = ['om', 's8', 'w0']
    prms_means = [0, 0, 0]
    prms_bounds = bounds
    var_names = mcuts
    var_labels = [f'${mcut:.2f}$' for mcut in mcuts]
    var_title = r'$\log_{10} m_\mathrm{200m,min}$'
    var_label_kwargs = {
        'color': bw_color,
        'ha': 'right',
        'va': 'center',
    }
    var_title_kwargs = {
        'x': -0.35,
        'y': 0,
        'color': bw_color,
        'rotation': 90,
        'ha': 'right',
        'va': 'center',
    }

    method_names = [method for method in methods]
    method_labels = [methods_info[method]['label'] for method in methods]
    method_kwargs = {m: methods_info[m] for m in methods}
    legend_kwargs = {
        'bbox_to_anchor': (1.1, 1.4, 0.8, 0.2),
        'handlelength': 2.0,
        'handletextpad': 1.0,
        'columnspacing': 2.0,
        'loc': 'upper center',
        'labelcolor': bw_color,
        'fontsize': 30,
        'ncol': 3
    }

    samples_set = []
    for idx_m, method in enumerate(methods):
        for idx_v, mcut in enumerate(mcuts):
            results[method][mcut] = getdist.MCSamples(
                samples=results[method][mcut]['maps'],
                loglikes=results[method][mcut]['loglikes'],
                names=prms_names,
                labels=[
                    '\Delta \Omega_\mathrm{m} / \Omega_\mathrm{m}',
                    '\Delta S_8 / S_8',
                    '\Delta w_0 / w_0'
                ],
                label=f'{mcut}'
            )

        samples_set.append([*results[method].values()])

    fig, axs = plot_cosmological_1d(
        prms_names=prms_names, prms_means=prms_means, prms_bounds=prms_bounds,
        method_names=method_names, method_kwargs=method_kwargs,
        samples_set=samples_set,
        var_names=var_names, var_labels=var_labels, var_title=var_title,
        var_label_kwargs=var_label_kwargs, var_title_kwargs=var_title_kwargs,
        legend_kwargs=legend_kwargs, method_labels=method_labels,
        figsize=(12, 6)
    )
    fname = (
        f'fit_om_s8_w0_Asurvey_{A_add}deg2'
        f'_1D_mcuts'
    )
    plt.savefig(
        f'{PAPER_DIR}/{fname}.pdf', transparent=True
    )
    plt.show()

    return results


def plot_cosmological_1d_gauss_vs_mixed(
        fname, mcuts=[14.0], alpha=0.2, bounds=None, stage=None):
    """Plot the marginalized 1D PDFs for different likelihoods."""
    methods_info = {
        'true_gaussian': {
            'name': 'true Gaussian',
            'method': 'true',
            'res': 'res_gaussian',
            'kwargs': {
                'label': 'true',
                'ls': '--',
                'color': bw_color,
                'alpha': 0.3,
            },
        },
        'true_mixed': {
            'name': r'mixed',
            'method': 'true',
            'res': 'res_gaussian_poisson',
            'kwargs': {
                'label': 'true',
                'ls': '-',
                'color': bw_color,
                'alpha': 0.5,
            },
        },
        'WL_gaussian': {
            'name': r'NFW Gaussian',
            'method': 'WL',
            'res': 'res_gaussian',
            'kwargs': {
                'label': 'NFW',
                'ls': '--',
                'color': mpl.cm.tab10.colors[0],
                'alpha': 0.3,
            },
        },
        'WL_mixed': {
            'name': r'mixed',
            'method': 'WL',
            'res': 'res_gaussian_poisson',
            'kwargs': {
                'label': 'NFW',
                'ls': '-',
                'color': mpl.cm.tab10.colors[0],
                'alpha': 0.5,
            },
        },
        'WL_c_gaussian': {
            'name': r'NFW $r_\mathrm{s}$ free Gaussian',
            'method': 'WL_c',
            'res': 'res_gaussian',
            'kwargs': {
                'label': r'NFW $r_\mathrm{s}$ free',
                'ls': '--',
                'color': mpl.cm.tab10.colors[1],
                'alpha': 0.3,
            },
        },
        'WL_c_mixed': {
            'name': r'mixed',
            'method': 'WL_c',
            'res': 'res_gaussian_poisson',
            'kwargs': {
                'label': r'NFW $r_\mathrm{s}$ free',
                'ls': '-',
                'color': mpl.cm.tab10.colors[1],
                'alpha': 0.5,
            },
        },
    }
    if bounds is None:
        if stage == 'IV':
            bounds = np.array([
                [-0.15, 0.05], [-0.04, 0.005], [-0.05, 0.16]])
        elif stage == 'III':
            bounds = np.array([
                [-0.39, 0.25], [-0.049, 0.019], [-0.49, 0.49]])

    results = dict((m, {mcut: {} for mcut in mcuts}) for m in methods_info.keys())
    with asdf.open(fname, copy_arrays=True, lazy_load=False) as af:
        A_add = tools.num_to_str(af['A_survey'], unit='k', precision=1)
        means = np.array([af['omega_m'], af['sigma_8'], af['w0']])
        means = sigma_8_to_S8(means, alpha=alpha)
        for method in methods_info.keys():
            for mcut in mcuts:
                method_name = methods_info[method]['method']
                results[method][mcut] = {}
                m = af[method_name][mcut][methods_info[method]['res']]['maps']
                m = sigma_8_to_S8(m, alpha=alpha)
                m = abs_to_rel(m, means)
                results[method][mcut]['maps'] = m
                results[method][mcut]['loglikes'] = af[method_name][mcut][methods_info[method]['res']]['fun']

    prms_names = ['om', 's8', 'w0']
    prms_means = [0, 0, 0]
    prms_bounds = bounds
    var_names = mcuts
    var_labels = [f'${mcut:.2f}$' for mcut in mcuts]
    var_title = r'$\log_{10} m_\mathrm{200m,min}$'
    var_label_kwargs = {
        'color': bw_color,
        'ha': 'right',
        'va': 'center',
    }
    var_title_kwargs = {
        'x': -0.05,
        'y': 0.1,
        'color': bw_color,
        'rotation': 90,
        'ha': 'right',
        'va': 'bottom',
    }
    method_names = [method for method in methods_info.keys()]
    method_labels = [methods_info[method]['name'] for method in methods_info.keys()]
    method_kwargs = {m: methods_info[m]['kwargs'] for m in methods_info.keys()}
    legend_kwargs = {
        'bbox_to_anchor': (-0.1, 1.1, 0.8, 0.2),
        'handlelength': 1.0,
        'handletextpad': 0.5,
        'columnspacing': 1.0,
        'loc': 'upper left',
        'labelcolor': bw_color,
        'fontsize': 22,
        'ncol': 3
    }

    samples_set = []
    for idx_m, method in enumerate(methods_info.keys()):
        for idx_v, mcut in enumerate(mcuts):
            results[method][mcut] = getdist.MCSamples(
                samples=results[method][mcut]['maps'],
                loglikes=results[method][mcut]['loglikes'],
                names=prms_names,
                labels=[
                    '\Delta \Omega_\mathrm{m} / \Omega_\mathrm{m}',
                    '\Delta S_8 / S_8',
                    '\Delta w_0 / w_0'
                ],
                label=f'{mcut}'
            )

        samples_set.append([*results[method].values()])

    fig, axs = plot_cosmological_1d(
        prms_names=prms_names, prms_means=prms_means, prms_bounds=prms_bounds,
        method_names=method_names, method_kwargs=method_kwargs, method_labels=method_labels,
        samples_set=samples_set,
        var_names=var_names, var_labels=var_labels, var_title=var_title,
        var_label_kwargs=var_label_kwargs, var_title_kwargs=var_title_kwargs,
        legend_kwargs=legend_kwargs,
        figsize=(10, 5.5)
    )
    fname = (
        f'fit_om_s8_w0_Asurvey_{A_add}deg2'
        f'_1D_prof_likelihood_comparison' )
    plt.savefig( f'{PAPER_DIR}/{fname}.pdf', transparent=True )
    plt.show()


def plot_cosmological_1d_profile(
        fname, mcut=14.0, methods=['true', 'WL', 'WL_c'],
        res='res_gaussian', alpha=0.2, stage=None, bounds=None):
    """Plot the marginalized 1D PDFs for different mass cuts."""
    methods_info = {
        'true': {
            'name': r'true',
            'min': 'true',
            'max': 'true',
            'med': 'true',
            'kwargs': {
                'label': 'true',
                'ls': '-',
                'color': bw_color,
                'alpha': 0.6,
            }
        },
        'WL': {
            'name': r'NFW',
            'min': 'WL_min',
            'max': 'WL_max',
            'med': 'WL',
            'kwargs': {
                'label': 'NFW',
                'ls': '-',
                'color': mpl.cm.tab10.colors[0],
                'alpha': 0.6,
            }
        },
        'WL_c': {
            'name': r'NFW $r_\mathrm{s}$ free',
            'min': 'WL_c_min',
            'max': 'WL_c_max',
            'med': 'WL_c',
            'kwargs': {
                'label': r'NFW $r_\mathrm{s}$ free',
                'ls': '-',
                'color': mpl.cm.tab10.colors[1],
                'alpha': 0.6,
            }
        },
    }
    if bounds is None:
        if stage is None:
            raise ValueError('if no bounds are provided, need to specify stage')
        elif stage == 'IV':
            bounds = np.array([
                [-0.15, 0.05], [-0.04, 0.005], [-0.05, 0.16]])
        elif stage == 'III':
            bounds = np.array([
                [-0.39, 0.25], [-0.049, 0.019], [-0.49, 0.49]])

    prof_qs = ['min', 'max', 'med']

    results = dict((m, {prof_q: {} for prof_q in prof_qs}) for m in methods)
    with asdf.open(fname, copy_arrays=True, lazy_load=False) as af:
        A_add = tools.num_to_str(af['A_survey'], unit='k', precision=1)
        zmin_add = tools.num_to_str(af['true'][mcut]['z_min'], precision=2)
        zmax_add = tools.num_to_str(af['true'][mcut]['z_max'], precision=2)
        mcut_add = tools.num_to_str(mcut, precision=2)
        means = np.array([af['omega_m'], af['sigma_8'], af['w0']])
        means = sigma_8_to_S8(means, alpha=alpha)
        for method in methods:
            for prof_q in prof_qs:
                results[method][prof_q] = {}
                m = af[methods_info[method][prof_q]][mcut][res]['maps']
                m = sigma_8_to_S8(m, alpha=alpha)
                m = abs_to_rel(m, means)
                results[method][prof_q]['maps'] = m
                results[method][prof_q]['loglikes'] = af[methods_info[method][prof_q]][mcut][res]['fun']

    prms_names = ['om', 's8', 'w0']
    prms_means = [0, 0, 0]
    prms_bounds = bounds
    var_names = prof_qs
    var_labels = ['$f_\mathrm{gas, min}$', '$f_\mathrm{gas, med}$', '$f_\mathrm{gas, max}$']
    var_title = None
    var_label_kwargs = {
        'color': bw_color,
        'ha': 'right',
        'va': 'center',
    }
    var_title_kwargs = {
        'x': -0.35,
        'y': -0.0,
        'color': bw_color,
        'rotation': 90,
        'ha': 'right',
        'va': 'center',
    }
    method_names = [method for method in methods if method in methods_info.keys()]
    method_labels = [methods_info[method]['name'] for method in methods_info.keys()]
    method_kwargs = {m: methods_info[m]['kwargs'] for m in methods_info.keys()}
    legend_kwargs = {
        'bbox_to_anchor': (1.1, 1.4, 0.8, 0.2),
        'handlelength': 2.0,
        'handletextpad': 1.0,
        'columnspacing': 2.0,
        'loc': 'upper center',
        'fontsize': 20,
        'ncol': 3
    }

    samples_set = []
    for idx_m, method in enumerate(methods):
        for prof_q in prof_qs:
            results[method][prof_q] = getdist.MCSamples(
                samples=results[method][prof_q]['maps'],
                loglikes=results[method][prof_q]['loglikes'],
                names=prms_names,
                labels=[
                    '\Delta \Omega_\mathrm{m} / \Omega_\mathrm{m}',
                    '\Delta S_8 / S_8',
                    '\Delta w_0 / w_0'
                ],
                label=f'{prof_q}'
            )

        samples_set.append([*results[method].values()])

    fig, axs = plot_cosmological_1d(
        prms_names=prms_names, prms_means=prms_means, prms_bounds=prms_bounds,
        method_names=method_names, method_kwargs=method_kwargs, method_labels=method_labels,
        samples_set=samples_set,
        var_names=var_names, var_labels=var_labels, var_title=var_title,
        var_label_kwargs=var_label_kwargs, var_title_kwargs=var_title_kwargs,
        legend_kwargs=legend_kwargs,
        figsize=(12, 6)
    )
    fname = (
        f'fit_om_s8_w0_Asurvey_{A_add}deg2'
        f'_z_{zmin_add}_{zmax_add}_m200m_min_{mcut_add}_1D_prof_var'
    )
    plt.savefig(
        f'{PAPER_DIR}/{fname}.pdf', transparent=True
    )
    plt.show()
