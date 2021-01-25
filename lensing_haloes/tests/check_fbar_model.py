from pathlib import Path

import asdf
import numpy as np
import scipy.interpolate as interp
import scipy.optimize as opt

import matplotlib as mpl
import matplotlib.pyplot as plt

import lensing_haloes.settings as settings
import lensing_haloes.data.generate_mock_lensing as mock_lensing
import lensing_haloes.data.generate_survey_results as gen_survey
import lensing_haloes.data.observational_data as obs_data
import lensing_haloes.data.process_data as process_data
import lensing_haloes.halo.profiles as profs
import lensing_haloes.util.plot as plot
import lensing_haloes.util.tools as tools

import pdb

TABLE_DIR = settings.TABLE_DIR
FIGURE_DIR = settings.FIGURE_DIR
PAPER_DIR = settings.PAPER_DIR
OBS_DIR = settings.OBS_DIR

RHO_CRIT = 2.7763458 * (10.0**11.0)  # [h^2 M_sun / Mpc^3]

dashes_WL = (15, 10)
dashes_WL_rs = (2, 2, 2, 2)
dashes_X = (5, 5)
dashes_ap = (10, 10)


def check_rho_gas_to_mr_gas(
        m500c_ref=1e14,
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_R_0p75-2p5_nbins_4'):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree

    z = results['z']
    m500c = results['m500c']
    fbar = results['omega_b'] / results['omega_m']

    r = results['r_range']
    rho_gas = results['rho_gas']
    mr_gas = results['m_gas']
    mr_tot = results['m_tot']
    mr_dmo = results['m_dm'] / (1 - fbar)

    idx_z = np.argmin(np.abs(z - 0.43))
    idx_m = np.argmin(np.abs(m500c - m500c_ref))

    mr_gas_int = profs.mr_from_rho(r=r, rs=r, rho=rho_gas[idx_z, idx_m], dlog10r=2)

    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.4])

    ax.plot(r, mr_dmo[idx_z, idx_m], c='k', ls='--', label='dmo')
    ax.plot(r, mr_tot[idx_z, idx_m], label='tot')
    l, = ax.plot(r, mr_gas[idx_z, idx_m], lw=1, label='gas')
    ax.plot(r, mr_gas_int, c=l.get_color(), lw=2, ls='--', label='gas fit')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$m(<r) \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax.legend(markerfirst=False)

    ax_r.axhline(1, c='k', ls='--')
    ax_r.plot(r, mr_tot[idx_z, idx_m] / mr_dmo[idx_z, idx_m])
    l, = ax_r.plot(r, mr_gas[idx_z, idx_m] / mr_dmo[idx_z, idx_m], lw=1)
    ax_r.plot(r, mr_gas_int / mr_dmo[idx_z, idx_m], c=l.get_color(), lw=2, ls='--')
    ax_r.set_xscale('log')
    ax_r.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')
    ax_r.set_ylabel(r'$m(<r) / m_\mathrm{dmo}(<r)$')
    ax_r.text(
        0.05, 0.5, f"$z={z[idx_z]:.2f}$\n$m_\mathrm{{500c}}={m500c[idx_m]:.2e}$",
        va='center', ha='left',
        transform=ax_r.transAxes,
        color='black', fontsize=30
    )

    plt.show()


def check_rho_tot_vs_rho_dmo(
        z_ref=0.43,
        m500c_ref=1e14,
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_R_0p75-2p5_nbins_4'):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree

    z = results['z']
    m500c = results['m500c']
    fbar = results['omega_b'] / results['omega_m']

    r = results['r_range']
    rho_gas = results['rho_gas']
    rho_dm = results['rho_dm']
    rho_dmo = results['rho_dm'] / (1 - fbar)
    rho_tot = results['rho_tot']

    idx_z = np.argmin(np.abs(z - 0.43))
    idx_m = np.argmin(np.abs(m500c - m500c_ref))

    plt.clf()
    fig = plt.figure(1)
    ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.4])

    ax.plot(r, rho_dmo[idx_z, idx_m], c='k', ls='--', label='dmo')
    ax.plot(r, rho_tot[idx_z, idx_m], label='tot')
    l, = ax.plot(r, rho_gas[idx_z, idx_m], lw=1, label='gas')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylabel(r'$\rho(r) \, [h^{2} \, \mathrm{M_\odot/Mpc^3}]$')
    ax.legend(markerfirst=False)

    ax_r.axhline(1, c='k', ls='--')
    ax_r.plot(r, rho_tot[idx_z, idx_m] / rho_dmo[idx_z, idx_m])
    l, = ax_r.plot(r, rho_gas[idx_z, idx_m] / rho_dmo[idx_z, idx_m], lw=1)
    ax_r.set_xscale('log')
    ax_r.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')
    ax_r.set_ylabel(r'$\rho(r) / \rho_\mathrm{dmo}(r)$')
    plt.show()


def check_fit_parameters_fit(
        dataset='croston+08', n_bins=4, z_ls=[0.1, 0.43, 1.0, 2.0], dlog10r=2,
        outer_norm=None):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    z_ls = np.atleast_1d(z_ls)
    # prepare figure
    plt.clf()
    plt.style.use('paper')
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
        results_all = process_data.get_rho_gas_fits_all(
            r_range=None, z_l=z_l, dlog10r=dlog10r, datasets=[dataset],
            outer_norm=outer_norm)
        results_bins = process_data.get_rho_gas_fits_bins(
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

        l_f, = ax.plot(m500c_bins, log10_rt_bins, c='k', lw=3)
        ax.plot(m500c_bins, alpha_bins, c='k', lw=3)

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


def check_fit_parameters(
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p1-2p0_m500c_13p5-15p5_nbins_4',
        ref_200m=False):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    with asdf.open(
            f'{TABLE_DIR}/model_fgas_r_{model_fname_append}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree

    z = results['z'][:]
    m500c = results['m500c'][:]
    r500c = results['r500c'][:]
    m200m = results['m200m_dmo'][:]
    r200m = results['r200m_dmo'][:]
    if ref_200m:
        log10_rt = results['log10_rt'][:] + np.log10(r500c / r200m)
        label = r'$\log_{10}r_\mathrm{t}/r_\mathrm{200m,dmo}$'
        ref_append = '_r200m_dmo'
    else:
        log10_rt = results['log10_rt'][:]
        label = r'$\log_{10}r_\mathrm{t}/r_\mathrm{500c}$'
        ref_append = '_r500c'
    alpha = results['alpha'][:]

    # prepare figure
    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(13, 8, forward=True)
    axm = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    axz = fig.add_axes([0.5, 0.1, 0.4, 0.8])
    axm_cb = fig.add_axes([0.3, 0.8, 0.15, 0.05])
    axz_cb = fig.add_axes([0.7, 0.8, 0.15, 0.05])

    cmap_m = plot.get_partial_cmap(mpl.cm.plasma_r)
    norm_m = mpl.colors.Normalize(np.log10(m500c.min()), np.log10(m500c.max()))
    sm_m = plt.cm.ScalarMappable(norm=norm_m, cmap=cmap_m)
    cmap_m = sm_m.to_rgba

    cmap_z = plot.get_partial_cmap(mpl.cm.viridis_r)
    norm_z = mpl.colors.Normalize(z.min(), z.max())
    sm_z = plt.cm.ScalarMappable(norm=norm_z, cmap=cmap_z)
    cmap_z = sm_z.to_rgba

    for idx_z, zz in enumerate(z):
        if ref_200m:
            m_ref = m200m[idx_z]
        else:
            m_ref = m500c
        axm.plot(
            m_ref, log10_rt[idx_z], marker='o', c=cmap_z(zz),
            label=label
        )
        axm.plot(
            m_ref, alpha[idx_z], marker='x', c=cmap_z(zz),
            label=r'$\alpha$'
        )
        # axm.scatter(m200m_bins, log10_rt_bins[idx_z], c='k')
        # axm.scatter(m200m_bins, alpha_bins[idx_z], c='k')

    cbm = plt.colorbar(sm_z, cax=axm_cb, orientation='horizontal')
    cbm.set_label(r'$z$')
    # cbm.set_label(r'$\log_{10} m_\mathrm{200m}$')

        # def linear_fit(log10_m200m, a, b):
        #     return a * (log10_m200m - b)

        # # opt_rt, pcov = opt.curve_fit(linear_fit, xdata=np.log10(m200m), ydata=log10_rt_all, maxfev=1000)
        # # opt_a, pcov = opt.curve_fit(linear_fit, xdata=np.log10(m200m), ydata=alpha_all, maxfev=5000)
        # opt_rt, pcov = opt.curve_fit(linear_fit, xdata=np.log10(m200m_bins), ydata=log10_rt_bins, maxfev=1000)
        # opt_a, pcov = opt.curve_fit(
        #     linear_fit, xdata=np.log10(m200m_bins), ydata=alpha_bins,
        #     p0=[-0.5, 15.5], maxfev=5000)
        # print(opt_rt, opt_a)

        # ax.scatter(
        #     m200m, log10_rt_all, marker="o", c=cmap(np.log10(m200m)),
        #     label=r'$\log_{10}r_\mathrm{t}/r_\mathrm{200m}$')
        # ax.scatter(
        #     m200m, alpha_all, marker="x", c=cmap(np.log10(m200m)),
        #     label=r'$\alpha$')
        # ax.plot(m200m_bins, log10_rt_bins, c="k")
        # ax.plot(m200m_bins, alpha_bins, c="k")
        # ax.plot(m200m, linear_fit(np.log10(m200m), *opt_rt))
        # ax.plot(m200m, linear_fit(np.log10(m200m), *opt_a))

    for idx_m, mm in enumerate(m500c):
        axz.plot(
            z, log10_rt[:, idx_m], marker='o', c=cmap_m(np.log10(mm)),
            label=label
        )
        axz.plot(
            z, alpha[:, idx_m], marker='x', c=cmap_m(np.log10(mm)),
            label=r'$\alpha$'
        )
        # axz.scatter(z_bins, log10_rt_bins[idx_m], c='k')
        # axz.scatter(z_bins, alpha_bins[idx_m], c='k')

    cbz = plt.colorbar(sm_m, cax=axz_cb, orientation='horizontal')
    cbz.set_label(r'$\log_{10} m_\mathrm{500c}$')

    axm.set_xscale('log')
    axm.set_ylim(-2, 2)
    axz.set_ylim(-2, 2)
    axm.get_xaxis().set_minor_formatter(mpl.ticker.NullFormatter())

    if ref_200m:
        axm.set_xlabel(r'$m_\mathrm{200m,dmo} \, [h^{-1} \, \mathrm{M_\odot}]$')
        axm.set_ylabel(r'$\log_{10}r_\mathrm{t}/r_\mathrm{200m,dmo}, \alpha$')
    else:
        axm.set_xlabel(r'$m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')
        axm.set_ylabel(r'$\log_{10}r_\mathrm{t}/r_\mathrm{500c}, \alpha$')

    axz.set_xlabel(r'$z$')
    axz.set_yticklabels([])
    # axm.legend(markerfirst=False)

    plt.savefig(f'{FIGURE_DIR}/fbar_r_fit_parameters{ref_append}.pdf', bbox_inches='tight')
    plt.show()


def check_fit_parameters2D(
        data_dir=TABLE_DIR,
        m_ref='m500c',
        model_fname_append='planck2019_z_0p1-2p0_m500c_13p5-15p5_nbins_4'):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    with asdf.open(
            f'{TABLE_DIR}/model_fgas_r_{model_fname_append}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree

    l10_rt_fit, a_fit, vals = gen_survey.fit_rt_alpha_model(
        data_dir=data_dir, model_fname_append=model_fname_append, m_ref=m_ref,
        diagnostic=True, n_bins=3,
    )

    z = results['z'][:]
    m500c = results['m500c'][:]
    r500c = results['r500c'][:]
    m200m = results['m200m_dmo'][:]
    r200m = results['r200m_dmo'][:]
    if m_ref == 'm500c':
        m_label = '500c'
        mx_ref = m500c
        rx_ref = r500c
        log10_rt = results['log10_rt'][:]
        log10_rt_fit = np.array([
            l10_rt_fit(None, mx_ref, 1/(zz+1), None)
            for idx_z, zz in enumerate(z)
        ])
        alpha_fit = np.array([
            a_fit(None, mx_ref, 1/(zz+1), None)
            for idx_z, zz in enumerate(z)
        ])
    elif m_ref =='m200m':
        m_label = '200m,dmo'
        mx_ref = m200m
        rx_ref = r200m
        log10_rt = results['log10_rt'][:] + np.log10(r500c / r200m)
        log10_rt_fit = np.array([
            l10_rt_fit(None, mx_ref[idx_z], 1/(zz+1), None)
            for idx_z, zz in enumerate(z)
        ])
        alpha_fit = np.array([
            a_fit(None, mx_ref[idx_z], 1/(zz+1), None)
            for idx_z, zz in enumerate(z)
        ])

    alpha = results['alpha'][:]


    log10_rt_ratio = log10_rt / log10_rt_fit - 1
    alpha_ratio = alpha / alpha_fit - 1

    # prepare figure
    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(13, 8, forward=True)
    axr = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    axa = fig.add_axes([0.5, 0.1, 0.4, 0.8])
    axr_cb = fig.add_axes([0.2, 0.825, 0.2, 0.05])
    axa_cb = fig.add_axes([0.6, 0.825, 0.2, 0.05])

    imr = axr.imshow(
        np.abs(log10_rt_ratio), extent=(
            np.log10(m500c.min()), np.log10(m500c.max()),
            z.min(), z.max()
        ), origin='lower', cmap=mpl.cm.binary,
        norm=mpl.colors.Normalize(
            vmin=0., vmax=0.2
        )
    )
    cbr = plt.colorbar(imr, cax=axr_cb, orientation='horizontal')
    axr_cb.set_xlabel(
        r'$\Delta \log_{10}(r_\mathrm{t}/r_\mathrm{x}) / \log_{10}(r_\mathrm{t}/r_\mathrm{x})$'
    )
    axr_cb.xaxis.tick_top()
    axr_cb.xaxis.set_label_position('top')

    ima = axa.imshow(
        np.abs(alpha_ratio), extent=(
            np.log10(m500c.min()), np.log10(m500c.max()),
            z.min(), z.max()
        ), origin='lower', cmap=mpl.cm.binary,
        norm=mpl.colors.Normalize(
            vmin=0., vmax=0.2
        )
    )
    cba = plt.colorbar(ima, cax=axa_cb, orientation='horizontal')
    axa_cb.set_xlabel(r'$\Delta\alpha / \alpha$')
    axa_cb.xaxis.tick_top()
    axa_cb.xaxis.set_label_position('top')

    axr.set_xlim(right=15.499)
    axr.set_xlabel(r'$m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')
    axr.set_ylabel(r'$z$')
    axa.set_xlabel(r'$m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')

    axa.set_yticklabels([])
    plt.savefig(f'{FIGURE_DIR}/fbar_r_fit_parameters_fit_residual_{m_ref}.pdf', bbox_inches='tight')
    plt.show()


def check_rho_gas_fits_all(
        datasets=['croston+08'], z_l=None, dlog10r=0,
        outer_norm=None, ids=None, plot_fit=True):
    """Plot the fractional di_bins b5_binstween the best-fitting gas profile
    bins and the true gas profile."""
    results = process_data.get_rho_gas_fits_all(
        r_range=None, z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm)

    plt.clf()
    plt.style.use('paper')
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


def check_rho_gas_fits_bins(
        dataset='croston+08', n_bins=3, n_r=15,
        z_l=0.43, dlog10r=2, outer_norm=None):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    results = process_data.get_rho_gas_fits_bins(
        datasets=[dataset], n_bins=n_bins, z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm, n_r=n_r,
    )

    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(10, 12, forward=True)
    # fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.15, 0.525, 0.8, 0.4])
    ax_r = fig.add_axes([0.15, 0.125, 0.8, 0.4])
    ax_cb = fig.add_axes([0.175, 0.65, 0.45, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=n_bins)

    ax_r.text(
        0.95, 0.95, f'$z={z_l:.2f}$',
        va='top', ha='right',
        transform=ax_r.transAxes,
        color='black', fontsize=30)
    # ax.text(
    #     0.95, 0.9, 'Croston+2008',
    #     va='center', ha='right',
    #     transform=ax.transAxes,
    #     color='black', fontsize=30)
    ax_r.axhline(y=0, ls="--", c="k")
    ax_r.axhspan(-0.01, .01, facecolor="k", alpha=0.3)
    ax_r.axhspan(-0.05, .05, facecolor="k", alpha=0.1)

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
        ld = ax.errorbar(
            rx * r500c[idx], rho_gas_med,
            yerr=(rho_gas_med - rho_gas_q16, rho_gas_q84 - rho_gas_med),
            c=cmap(idx), marker='o', elinewidth=1, lw=0,
        )
        ls_d.append(ld)
        # ax.plot(rx * r500c[idx], rho_gas_fit, c=cmap(idx), lw=2)
        # ax.fill_between(
        #     rx * r500c[idx], rho_gas_q16, rho_gas_q84,
        #     color=cmap(idx), alpha=0.2
        # )
        ax_r.plot(
            rx * r500c[idx], rho_gas_med / rho_gas_fit - 1,
            c=cmap(idx), marker='o', lw=3
        )
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
        markerfirst=False, loc=1
    )

    ax.set_xlim(left=0.01)
    ax.set_ylim(bottom=10**13.01, top=10**15.5)
    ax_r.set_xlim(left=0.01)
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
    if outer_norm is not None:
        fname_append = f"_rx_{outer_norm['rx']:d}_fbar_{outer_norm['fbar']:.2f}"
    else:
        fname_append = ''
    plt.savefig(f'{PAPER_DIR}/fbar_r_fit_rho_gas_vs_true_bins{fname_append}.pdf')
    plt.show()


def check_rho_gas_slope_fits_bins(
        dataset='croston+08', n_bins=3,
        z_l=0.43, dlog10r=2, outer_norm=None):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    results = process_data.get_rho_gas_fits_bins(
        datasets=[dataset], n_bins=n_bins, z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm,
    )

    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(10, 9, forward=True)
    # fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    ax_cb = fig.add_axes([0.4, 0.825, 0.475, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=n_bins)

    m500c = results[dataset]['m500c_bins'][:]
    r500c = results[dataset]['r500c_bins'][:]
    for idx, res in enumerate(results[dataset]['fit_results']):
        fbar = results['omega_b'] / results['omega_m']
        rx = res['rx']
        rho_gas_fit = res['rho_gas_fit']

        rho_gas_spl = interp.splrep(np.log10(rx), np.log10(rho_gas_fit))

        ax.plot(rx, interp.splev(np.log10(rx), rho_gas_spl, der=1), c=cmap(idx), lw=2)

    cb = plot.add_colorbar_indexed(
        cmap_indexed=cmap, fig=fig, ax_cb=ax_cb,
        items=np.log10(m500c),
        orientation='horizontal',
    )
    cb.set_label(r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')

    ax.set_xlim(left=0.01)
    ax.set_xscale('log')
    ax.set_xlabel(r'$r/r_\mathrm{500c}$')
    ax.set_ylabel(r'$\frac{\mathrm{d} \, \log_{10}\rho_\mathrm{gas}(r)}{\mathrm{d} \, \log_{10} r/r_\mathrm{500c}}$')
    if outer_norm is not None:
        fname_append = f"_rx_{outer_norm['rx']:d}_fbar_{outer_norm['fbar']:.2f}"
    else:
        fname_append = ''
    plt.savefig(f'{FIGURE_DIR}/fbar_r_fit_rho_gas_slope_bins{fname_append}.pdf', bbox_inches='tight')
    plt.show()


def check_mr_gas_fits_all(
        datasets=['croston+08'], z_l=None,
        outer_norm=None,
        dlog10r=2, n_int=1000):
    """Plot the fractional difference between the best-fitting enclosed
    gas mass profiles and the true ones."""
    results = process_data.get_rho_gas_fits_all(
        z_l=z_l, dlog10r=0, outer_norm=outer_norm
    )

    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax_cb = fig.add_axes([0.5, 0.65, 0.375, 0.05])

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

    ax_r.axhline(y=0, ls="--", c="k")

    for dataset in datasets:
        m500c = results[dataset]['m500c']
        z = results[dataset]['z']
        for idx, res in enumerate(results[dataset]['fit_results']):
            rx = res['rx']
            rho_gas = res['rho_gas']
            rho_gas_err = res['rho_gas_err']
            r500c = res['r_x']

            mr_gas_fit = process_data.mr_gas_from_fbar(
                r=rx*r500c, r_y=r500c, **res['opt_prms'], **res['dm_kwargs']
            )
            mr_gas = process_data.mr_gas_from_rho_gas(
                r=rx*r500c, rs=rx*r500c, rho_gas=rho_gas,
                dlog10r=dlog10r, n_int=n_int
            )
            mr_gas_err = np.mean([
                mr_gas - process_data.mr_gas_from_rho_gas(
                    r=rx*r500c, rs=rx*r500c, rho_gas=rho_gas-rho_gas_err,
                    dlog10r=dlog10r, n_int=n_int
                ),
                process_data.mr_gas_from_rho_gas(
                    r=rx*r500c, rs=rx*r500c, rho_gas=rho_gas+rho_gas_err,
                    dlog10r=dlog10r, n_int=n_int
                ) - mr_gas
            ], axis=0)

            c = cmap(np.log10(m500c[idx]))
            ax.plot(rx * r500c, mr_gas / m500c[idx], c=c, lw=3, alpha=0.5)
            ax.plot(rx * r500c, mr_gas_fit / m500c[idx], c=c, lw=2)
            ax.fill_between(
                rx * r500c,
                (mr_gas - mr_gas_err) / m500c[idx],
                (mr_gas + mr_gas_err) / m500c[idx],
                color=c, alpha=0.2
            )

            ax_r.plot(rx * r500c, mr_gas / mr_gas_fit - 1, c=c, lw=2)
            ax_r.fill_between(
                rx * r500c,
                (mr_gas - mr_gas_err) / mr_gas_fit - 1,
                (mr_gas + mr_gas_err) / mr_gas_fit -1,
                color=c, alpha=0.2
            )

    cb = fig.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_label(r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')

    ax.set_xlim(left=1e-2)
    ax.set_ylim(1.01e-5, 0.2)
    ax_r.set_xlim(left=1e-2)
    ax_r.set_ylim(-0.5, 0.5)
    ax.set_xscale('log')
    ax.set_xticklabels([])
    ax.set_yscale('log')
    ax_r.set_xscale('log')
    ax.set_ylabel(r'$m_\mathrm{gas}(<r) \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax_r.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')
    ax_r.set_ylabel(r'$m_\mathrm{gas}(<r) / m_\mathrm{gas,fit}(<r) - 1$')
    plt.savefig(f'{FIGURE_DIR}/fbar_r_fit_mr_gas_vs_true_all.pdf', bbox_inches='tight')
    plt.show()


def check_mr_gas_fits_bins(
        dataset='croston+08',
        outer_norm=None,
        n_bins=3, z_l=0.43, dlog10r=2, n_int=1000):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    results = process_data.get_rho_gas_fits_bins(
        datasets=[dataset], n_bins=n_bins, z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm,
    )

    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax_cb = fig.add_axes([0.5, 0.65, 0.375, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=n_bins)

    ax_r.axhline(y=0, ls="--", c="k")

    m500c = results[dataset]['m500c_bins'][:]
    r500c = results[dataset]['r500c_bins'][:]
    for idx, res in enumerate(results[dataset]['fit_results']):
        rx = res['rx']
        rho_gas_fit = res['rho_gas_fit']
        rho_gas_med = res['rho_gas_med']
        rho_gas_q16 = res['rho_gas_q16']
        rho_gas_q84 = res['rho_gas_q84']

        mr_gas_fit = process_data.mr_gas_from_fbar(
            r=rx*r500c[idx], r_y=r500c[idx], **res['opt_prms'], **res['dm_kwargs']
        )
        # like-for-like comparison
        # mr_gas_fit = process_data.mr_gas_from_rho_gas(
        #     r=rx*r500c[idx], rs=rx*r500c[idx], rho_gas=rho_gas_fit,
        #     dlog10r=dlog10r, n_int=n_int
        # )
        mr_gas_med = process_data.mr_gas_from_rho_gas(
            r=rx*r500c[idx], rs=rx*r500c[idx], rho_gas=rho_gas_med,
            dlog10r=dlog10r, n_int=n_int
        )
        mr_gas_q16 = process_data.mr_gas_from_rho_gas(
            r=rx*r500c[idx], rs=rx*r500c[idx], rho_gas=rho_gas_q16,
            dlog10r=dlog10r, n_int=n_int
        )
        mr_gas_q84 = process_data.mr_gas_from_rho_gas(
            r=rx*r500c[idx], rs=rx*r500c[idx], rho_gas=rho_gas_q84,
            dlog10r=dlog10r, n_int=n_int
        )

        ax.plot(rx * r500c[idx], mr_gas_med / m500c[idx], c=cmap(idx), lw=3, alpha=0.5)
        ax.plot(rx * r500c[idx], mr_gas_fit / m500c[idx], c=cmap(idx), lw=2)
        ax.fill_between(
            rx * r500c[idx], mr_gas_q16 / m500c[idx], mr_gas_q84 / m500c[idx],
            color=cmap(idx), alpha=0.2
        )

        ax_r.plot(rx * r500c[idx], mr_gas_med / mr_gas_fit - 1, c=cmap(idx), lw=2)
        ax_r.fill_between(
            rx * r500c[idx], mr_gas_q16 / mr_gas_fit - 1, mr_gas_q84 / mr_gas_fit -1,
            color=cmap(idx), alpha=0.2
        )

    cb = plot.add_colorbar_indexed(
        cmap_indexed=cmap, fig=fig, ax_cb=ax_cb,
        items=np.log10(m500c),
        orientation='horizontal',
    )
    cb.set_label(r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')

    ax.set_xlim(left=1e-2)
    ax.set_ylim(1.01e-5, 1e-1)
    ax_r.set_xlim(left=1e-2)
    ax_r.set_ylim(-0.5, 0.5)
    ax.set_xscale('log')
    ax.set_xticklabels([])
    ax.set_yscale('log')
    ax_r.set_xscale('log')
    ax.set_ylabel(r'$m_\mathrm{gas}(<r) / m_\mathrm{500c}$')
    ax_r.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')
    ax_r.set_ylabel(r'$m_\mathrm{gas}(<r) / m_\mathrm{gas,fit}(<r) - 1$')
    plt.savefig(f'{FIGURE_DIR}/fbar_r_fit_mr_gas_vs_true_bins.pdf', bbox_inches='tight')
    plt.show()


def check_fbar_fits_all(
        datasets=['croston+08'], z_l=None,
        outer_norm=None,
        dlog10r=2, n_int=1000):
    """Plot the fractional difference between the best-fitting enclosed
    gas mass profiles and the true ones."""
    results = process_data.get_rho_gas_fits_all(
        z_l=z_l, dlog10r=0, outer_norm=outer_norm)

    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(8, 10, forward=True)
    ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    ax_r = fig.add_axes([0.1, 0.1, 0.8, 0.4])
    ax_cb = fig.add_axes([0.45, 0.65, 0.425, 0.05])

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

            fbar_fit = process_data.fbar_rx(rx=rx, **res['opt_prms'])
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


def check_fbar_fits_bins(
        dataset='croston+08',
        n_bins=3, z_l=0.43, n_r=15, dlog10r=2, n_int=1000,
        outer_norm=None):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    results = process_data.get_rho_gas_fits_bins(
        datasets=[dataset], n_bins=n_bins, z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm, n_r=n_r,
    )

    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(10, 12, forward=True)
    # fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.15, 0.525, 0.8, 0.4])
    ax_r = fig.add_axes([0.15, 0.125, 0.8, 0.4])
    ax_cb = fig.add_axes([0.475, 0.65, 0.45, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=n_bins)

    ax_r.text(
        0.95, 0.95, f'$z={z_l:.2f}$',
        va='top', ha='right',
        transform=ax_r.transAxes,
        color='black', fontsize=30)
    ax_r.axhline(y=0, ls="--", c="k")
    ax_r.axhspan(-0.01, .01, facecolor="k", alpha=0.3)
    ax_r.axhspan(-0.05, .05, facecolor="k", alpha=0.1)

    m500c = results[dataset]['m500c_bins'][:]
    r500c = results[dataset]['r500c_bins'][:]
    ls_d = []
    for idx, res in enumerate(results[dataset]['fit_results']):
        rx = res['rx']
        fbar_data = res['fbar_rx']
        fbar_err = res['fbar_rx_err']
        fbar = results['omega_b'] / results['omega_m']

        fbar_fit = process_data.fbar_rx(rx=rx, **res['opt_prms'])

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
        markerfirst=True, loc=2
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


def check_fbar_fits_r_bins(
        dataset='croston+08',
        n_bins=3,
        m500c=np.logspace(13.5, 15.5, 50),
        r500cs=np.array([1, 2, 4]),
        z_l=0.43, n_r=15, dlog10r=2, n_int=1000,
        outer_norm=None):
    results_data = obs_data.load_datasets(datasets=[dataset], h_units=True)
    results_data = results_data[dataset]

    # get the best-fitting linear relation for log10_rt and alpha
    omega_b = 0.0493
    omega_m = 0.315
    fbar = omega_b / omega_m

    fit_prms = gen_survey.fit_observational_dataset(
        dataset=dataset, z=z_l, omega_b=omega_b, omega_m=omega_m,
        dlog10r=dlog10r, n_int=n_int, n_bins=n_bins, outer_norm=outer_norm,
        err=True, diagnostic=True, bins=True
    )
    log10_rt_prms = fit_prms['med']['log10_rt']
    alpha_prms = fit_prms['med']['alpha']
    log10_rt_prms_min = fit_prms['min']['log10_rt']
    log10_rt_prms_plus = fit_prms['plus']['log10_rt']

    log10_rt = gen_survey.linear_fit(np.log10(m500c), *log10_rt_prms)
    alpha = gen_survey.linear_fit(np.log10(m500c), *alpha_prms)
    log10_rt_plus = gen_survey.linear_fit(np.log10(m500c), *log10_rt_prms_plus)
    log10_rt_min = gen_survey.linear_fit(np.log10(m500c), *log10_rt_prms_min)

    # set up the figure
    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(10, 10, forward=True)
    # fig.set_size_inches(10, 13, forward=True)
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax_cb = fig.add_axes([0.6, 0.3, 0.2, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=len(r500cs))

    # ax.text(
    #     0.95, 0.95, f'$z={z_l:.2f}$',
    #     va='top', ha='right',
    #     transform=ax.transAxes,
    #     color='black', fontsize=30)

    ld = ax.errorbar(
        results_data['m500c'][:],
        results_data['fgas_500c'] / fbar,
        xerr=results_data['m500c_err'].T,
        yerr=results_data['fgas_500c_err'].T / fbar,
        lw=0, color=cmap(0), marker='o', elinewidth=2,
    )


    for idx, rx in enumerate(r500cs):
        fbar_fit = process_data.fbar_rx(
            rx=rx, log10_rt=log10_rt, alpha=alpha,
            fbar=fbar, fbar0=0)
        fbar_fit_plus = process_data.fbar_rx(
            rx=rx, log10_rt=log10_rt_plus, alpha=alpha,
            fbar=fbar, fbar0=0)
        fbar_fit_min = process_data.fbar_rx(
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
        markerfirst=True, loc=2
    )

    ax.set_ylim(top=1.1)
    ax.set_xscale('log')
    ax.set_xlabel(r'$m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax.set_ylabel(r'$f_\mathrm{bar}(<x \, r_\mathrm{500c}) / (\Omega_\mathrm{b} / \Omega_\mathrm{m})$')
    plt.savefig(f'{PAPER_DIR}/fbar_r_fit_fbar_vs_true_r_bins.pdf')
    plt.show()


def compare_model_beta_fit_all(
        dataset='croston+08', z_l=None, dlog10r=2,
        outer_norm=None):
    """Plot the fractional difference between the best-fitting gas profiles."""
    results_beta = process_data.get_rho_gas_fits_all(
        datasets=[dataset], z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm)
    results_fbar = process_data.get_rho_gas_fits_all(
        datasets=[dataset], rx_range=np.geomspace(0.15, 1.4, 20), log=True,
        z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm)

    m500c = results_beta[dataset]['m500c']

    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(13, 8, forward=True)
    ax_m = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    ax_b = fig.add_axes([0.5, 0.1, 0.4, 0.8])
    ax_cb = fig.add_axes([0.65, 0.275, 0.2, 0.05])

    cmap = plot.get_partial_cmap(mpl.cm.plasma_r)
    norm = mpl.colors.Normalize(
        vmin=np.log10(m500c.min()), vmax=np.log10(m500c.max()))
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap = sm.to_rgba

    def beta_prof(r, rc, beta, rho0):
        return rho0 / (1 + (r / rc)**2)**(1.5 * beta)

    def beta_min(prms, r, prof):
        rc, beta, rho0 = prms
        return np.sum(
            (np.log10(prof) - np.log10(beta_prof(r, rc, beta, rho0)))**2
        )

    for idx, res in enumerate(results_beta[dataset]['fit_results']):
        rx = res['rx']
        r_x = res['r_x']
        rho_gas = res['rho_gas']
        rho_gas_err = res['rho_gas_err']

        fbar_prms = results_fbar[dataset]['fit_results'][idx]['opt_prms']
        dm_kwargs = results_fbar[dataset]['fit_results'][idx]['dm_kwargs']
        rho_gas_fbar = process_data.rho_gas_from_fbar(
            r=rx*r_x, r_y=r_x, **fbar_prms, **dm_kwargs
        )

        c = cmap(np.log10(m500c[idx]))

        beta_prms, _ = opt.curve_fit(
            beta_prof, rx[rx > 0.15], rho_gas[rx > 0.15],
            p0=[0.2, 0.71, 500 * RHO_CRIT], sigma=rho_gas_err[rx > 0.15]
        )
        # res = opt.minimize(
        #     beta_min, x0=[0.2, 0.71, 500 * RHO_CRIT],
        #     args=(rx[rx > 0.15], rho_gas[rx > 0.15])
        # )
        # beta_prms = res.x

        ax_m.plot(rx, rho_gas / rho_gas_fbar - 1, c=c, lw=2)
        ax_m.fill_between(
            rx,
            (rho_gas - rho_gas_err) / rho_gas_fbar - 1,
            (rho_gas + rho_gas_err) / rho_gas_fbar - 1,
            color=c, alpha=0.2
        )
        ax_b.plot(rx, rho_gas / beta_prof(rx, *beta_prms) - 1, c=c, lw=2)
        ax_b.fill_between(
            rx,
            (rho_gas - rho_gas_err) / beta_prof(rx, *beta_prms) - 1,
            (rho_gas + rho_gas_err) / beta_prof(rx, *beta_prms) - 1,
            color=c, alpha=0.2
        )

    ax_m.axvspan(0.15, 1.3, color='g', alpha=0.3)
    ax_b.axvspan(0.15, 1.3, color='g', alpha=0.3)
    ax_m.set_title(r'$f_\mathrm{gas}(<r)$ fit')
    ax_b.set_title(r'beta profile fit')
    ax_m.axhline(y=0, ls="--", c="k")
    ax_b.axhline(y=0, ls="--", c="k")

    cb = plt.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_label(r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \,\mathrm{M_\odot}]$')

    ax_m.set_xlim(0.01, 1.3)
    ax_m.set_ylim(-0.5, 0.5)
    ax_b.set_xlim(0.0101, 1.3)
    ax_b.set_ylim(-0.5, 0.5)

    ax_m.set_xscale('log')
    ax_b.set_xscale('log')

    ax_b.set_yticklabels([])

    ax_m.set_ylabel(r'$\rho_\mathrm{gas}(r) / \rho_\mathrm{gas,fit}(r) - 1$')
    # ax_r.set_xlabel(r'$r \, [h_{70}^{-1} \, \mathrm{Mpc}]$')
    ax_m.set_xlabel(r'$r / r_\mathrm{500c}$')
    ax_b.set_xlabel(r'$r / r_\mathrm{500c}$')
    plt.savefig(f'{FIGURE_DIR}/fbar_r_fit_rho_gas_model+beta_vs_true_all.pdf', bbox_inches='tight')
    plt.show()


def compare_model_beta_fit_bins(
        dataset='croston+08', z_l=None, dlog10r=2,
        n_bins=3, n_r=15, outer_norm=None):
    """Plot the fractional difference between the best-fitting gas profiles."""
    results_beta = process_data.get_rho_gas_fits_bins(
        datasets=[dataset], n_bins=n_bins, z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm, n_r=n_r,
    )
    results_fbar = process_data.get_rho_gas_fits_bins(
        datasets=[dataset], rx_range=np.geomspace(0.15, 1.4, n_r), log=True,
        z_l=z_l, dlog10r=dlog10r,
        outer_norm=outer_norm)

    m500c = results_beta[dataset]['m500c']

    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(13, 8, forward=True)
    ax_m = fig.add_axes([0.1, 0.1, 0.4, 0.8])
    ax_b = fig.add_axes([0.5, 0.1, 0.4, 0.8])
    ax_cb = fig.add_axes([0.65, 0.275, 0.2, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=n_bins)

    def beta_prof(r, rc, beta, rho0):
        return rho0 / (1 + (r / rc)**2)**(1.5 * beta)

    def beta_min(prms, r, prof):
        rc, beta, rho0 = prms
        return np.sum(
            (np.log10(prof) - np.log10(beta_prof(r, rc, beta, rho0)))**2
        )

    for idx, res in enumerate(results_beta[dataset]['fit_results']):
        rx = res['rx']
        r_x = res['r_x']
        rho_gas = res['rho_gas']
        rho_gas_err = res['rho_gas_err']

        fbar_prms = results_fbar[dataset]['fit_results'][idx]['opt_prms']
        dm_kwargs = results_fbar[dataset]['fit_results'][idx]['dm_kwargs']
        rho_gas_fbar = process_data.rho_gas_from_fbar(
            r=rx*r_x, r_y=r_x, **fbar_prms, **dm_kwargs
        )

        beta_prms, _ = opt.curve_fit(
            beta_prof, rx[rx > 0.15], rho_gas[rx > 0.15],
            p0=[0.2, 0.71, 500 * RHO_CRIT], sigma=rho_gas_err[rx > 0.15]
        )
        # res = opt.minimize(
        #     beta_min, x0=[0.2, 0.71, 500 * RHO_CRIT],
        #     args=(rx[rx > 0.15], rho_gas[rx > 0.15])
        # )
        # beta_prms = res.x

        ax_m.plot(rx, rho_gas / rho_gas_fbar - 1, c=cmap(idx), lw=2)
        ax_m.fill_between(
            rx,
            (rho_gas - rho_gas_err) / rho_gas_fbar - 1,
            (rho_gas + rho_gas_err) / rho_gas_fbar - 1,
            color=cmap(idx), alpha=0.2
        )
        ax_b.plot(rx, rho_gas / beta_prof(rx, *beta_prms) - 1, c=cmap(idx), lw=2)
        ax_b.fill_between(
            rx,
            (rho_gas - rho_gas_err) / beta_prof(rx, *beta_prms) - 1,
            (rho_gas + rho_gas_err) / beta_prof(rx, *beta_prms) - 1,
            color=cmap(idx), alpha=0.2
        )

    ax_m.axvspan(0.15, 1.3, color='g', alpha=0.3)
    ax_b.axvspan(0.15, 1.3, color='g', alpha=0.3)
    ax_m.set_title(r'$f_\mathrm{gas}(<r)$ fit')
    ax_b.set_title(r'beta profile fit')
    ax_m.axhline(y=0, ls="--", c="k")
    ax_b.axhline(y=0, ls="--", c="k")

    cb = plot.add_colorbar_indexed(
        cmap_indexed=cmap, fig=fig, ax_cb=ax_cb,
        items=np.log10(m500c),
        orientation='horizontal',
    )
    cb.set_label(r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \, \mathrm{M_\odot}]$')

    ax_m.set_xlim(0.01, 1.3)
    ax_m.set_ylim(-0.5, 0.5)
    ax_b.set_xlim(0.0101, 1.3)
    ax_b.set_ylim(-0.5, 0.5)

    ax_m.set_xscale('log')
    ax_b.set_xscale('log')

    ax_b.set_yticklabels([])

    ax_m.set_ylabel(r'$\rho_\mathrm{gas}(r) / \rho_\mathrm{gas,fit}(r) - 1$')
    # ax_r.set_xlabel(r'$r \, [h_{70}^{-1} \, \mathrm{Mpc}]$')
    ax_m.set_xlabel(r'$r / r_\mathrm{500c}$')
    ax_b.set_xlabel(r'$r / r_\mathrm{500c}$')
    plt.savefig(f'{FIGURE_DIR}/fbar_r_fit_rho_gas_model+beta_vs_true_bins.pdf', bbox_inches='tight')
    plt.show()


def check_mass_ratio_m200m_obs(
        z_ref=0.43,
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_nbins_4_R_0p75-2p5'):
    """Plot the ratio between the best-fitting and the true m200m_obs"""
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}_min.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results_min = af.tree
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}_plus.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results_plus = af.tree

    z = results['z']
    m500c = results['m500c']
    fbar = results['omega_b'] / results['omega_m']

    idx_z_ref = np.argmin(np.abs(z - 0.43))

    m200m_obs = results['m200m_obs']
    m200m_WL = results['m200m_WL']
    m200m_WL_rs = results['m200m_WL_rs']
    m200m_obs_min = results_min['m200m_obs']
    m200m_WL_min = results_min['m200m_WL']
    m200m_WL_rs_min = results_min['m200m_WL_rs']
    m200m_obs_plus = results_plus['m200m_obs']
    m200m_WL_plus = results_plus['m200m_WL']
    m200m_WL_rs_plus = results_plus['m200m_WL_rs']

    # set up style and axes
    plt.style.use("paper")
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax_cb = fig.add_axes([0.3, 0.83, 0.4, 0.05])

    l_WL, = ax.plot(
        m500c, m200m_WL[idx_z_ref] / m200m_obs[idx_z_ref],
        dashes=dashes_WL
    )
    ax.fill_between(
        m500c, m200m_WL_min[idx_z_ref] / m200m_obs_min[idx_z_ref],
        m200m_WL_plus[idx_z_ref] / m200m_obs_plus[idx_z_ref],
        color=l_WL.get_color(), alpha=0.3
    )

    l_WL_c, = ax.plot(
        m500c, m200m_WL_rs[idx_z_ref] / m200m_obs[idx_z_ref],
        dashes=dashes_WL_rs
    )
    ax.fill_between(
        m500c, m200m_WL_rs_min[idx_z_ref] / m200m_obs_min[idx_z_ref],
        m200m_WL_rs_plus[idx_z_ref] / m200m_obs_plus[idx_z_ref],
        color=l_WL_c.get_color(), alpha=0.3
    )

    ax.axhline(y=1, c="k", ls="--")
    ax.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    ax.axhspan(0.98, 1.02, facecolor="k", alpha=0.1)

    ax.text(
        0.05, 0.05, f'$z={z[idx_z_ref]:.2f}$',
        va='center', ha='left',
        transform=ax.transAxes,
        color='black', fontsize=30)

    leg = ax.legend(
        [l_WL, l_WL_c],
        [r"NFW", r"NFW $r_\mathrm{s}$ free"],
        loc=4, frameon=True, framealpha=0.6, markerfirst=False,
    )
    leg.get_frame().set_linewidth(0)

    ax.set_ylim(0.87, 1.05)
    ax.set_xscale("log")

    ax.set_xlabel(r'$m_\mathrm{500c,true} \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax.set_ylabel(r'$m_{\mathrm{200m},i}/m_\mathrm{200m,obs,true}$')

    fname = f'm200m_obs_vs_WL_fgas_r_{model_fname_append}'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def check_mass_ratio_m_aperture(
        z_ref=0.43, R1=0.75, R2=2., Rmax=2.5,
        data_dir=TABLE_DIR,
        model_fname_base='planck2019_z_0p43_m500c_13p5-15p5_nbins_4',
        model_fname_range='R_0p75-2p5'):
    """Plot the ratio between the best-fitting and the true m200m_dmo"""
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
    plt.style.use("paper")
    fig = plt.figure(figsize=(10,9))
    ax_obs = fig.add_axes([0.15, 0.65, 0.8, 0.3])
    ax_dmo = fig.add_axes([0.15, 0.15, 0.8, 0.5])

    # ax_cb = fig.add_axes([0.3, 0.83, 0.4, 0.05])

    ax_dmo.axhline(y=1, c="k", ls="--")
    ax_dmo.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    ax_dmo.axhspan(0.98, 1.02, facecolor="k", alpha=0.1)

    ax_obs.axhline(y=1, c="k", ls="--")
    ax_obs.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    ax_obs.axhspan(0.98, 1.02, facecolor="k", alpha=0.1)
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
        ls='-', c='k', lw=1
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
        c='k', alpha=0.5, dashes=dashes_WL
    )
    l_WL_rs_200m, = ax_obs.plot(
        m500c,
        results['med']['m200m_WL_rs'][idx_z_ref] / results['med']['m200m_obs'][idx_z_ref],
        c='k', alpha=0.5, dashes=dashes_WL_rs
    )
    l_WL_200m, = ax_dmo.plot(
        m500c,
        results['med']['m200m_WL'][idx_z_ref] / results['med']['m200m_dmo'][idx_z_ref],
        c='k', alpha=0.5, dashes=dashes_WL
    )
    l_WL_rs_200m, = ax_dmo.plot(
        m500c,
        results['med']['m200m_WL_rs'][idx_z_ref] / results['med']['m200m_dmo'][idx_z_ref],
        c='k', alpha=0.5, dashes=dashes_WL_rs
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
        color='black', fontsize=30
    )

    ax_obs.text(
        0.05, 0.05, f'$z={z[idx_z_ref]:.2f}$',
        va='bottom', ha='left',
        transform=ax_obs.transAxes,
        color='black', fontsize=30)

    idx_ann = int(0.25 * len(m500c))
    ax_dmo.annotate(
        r'$\frac{M_\mathrm{true}(<R)}{M_\mathrm{dmo}(<R)}$',
        (
            m500c[idx_ann], results['med']['M_ap_true'][idx_z_ref][idx_ann]
            / results['med']['M_ap_dmo'][idx_z_ref][idx_ann]
        ),
        # (-50, -10), textcoords='offset points',
        (0.015, 0.6), textcoords=ax_dmo.transAxes,
        ha='left', va='center',
        arrowprops=dict(
            facecolor='k', shrink=0.,
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


def check_mass_ratio_m200m_dmo(
        z_ref=0.43,
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_nbins_4_R_0p75-2p5'):
    """Plot the ratio between the best-fitting and the true m200m_dmo"""
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results = af.tree
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}_min.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results_min = af.tree
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}_plus.asdf',
            lazy_load=False, copy_arrays=True) as af:
        results_plus = af.tree

    z = results['z']
    m500c = results['m500c']
    fbar = results['omega_b'] / results['omega_m']

    idx_z_ref = np.argmin(np.abs(z - 0.43))

    m200m_dmo = results['m200m_dmo']
    m200m_WL = results['m200m_WL']
    m200m_WL_rs = results['m200m_WL_rs']
    m200m_dmo_min = results_min['m200m_dmo']
    m200m_WL_min = results_min['m200m_WL']
    m200m_WL_rs_min = results_min['m200m_WL_rs']
    m200m_dmo_plus = results_plus['m200m_dmo']
    m200m_WL_plus = results_plus['m200m_WL']
    m200m_WL_rs_plus = results_plus['m200m_WL_rs']

    # set up style and axes
    plt.style.use("paper")
    fig = plt.figure(figsize=(10,9))
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
    # ax_cb = fig.add_axes([0.3, 0.83, 0.4, 0.05])

    l_WL, = ax.plot(
        m500c, m200m_WL[idx_z_ref] / m200m_dmo[idx_z_ref],
        dashes=dashes_WL
    )
    ax.fill_between(
        m500c, m200m_WL_min[idx_z_ref] / m200m_dmo_min[idx_z_ref],
        m200m_WL_plus[idx_z_ref] / m200m_dmo_plus[idx_z_ref],
        color=l_WL.get_color(), alpha=0.3
    )

    l_WL_c, = ax.plot(
        m500c, m200m_WL_rs[idx_z_ref] / m200m_dmo[idx_z_ref],
        dashes=dashes_WL_rs
    )
    ax.fill_between(
        m500c, m200m_WL_rs_min[idx_z_ref] / m200m_dmo_min[idx_z_ref],
        m200m_WL_rs_plus[idx_z_ref] / m200m_dmo_plus[idx_z_ref],
        color=l_WL_c.get_color(), alpha=0.3
    )

    ax.axhline(y=1, c="k", ls="--")
    ax.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    ax.axhspan(0.98, 1.02, facecolor="k", alpha=0.1)

    ax.text(
        0.05, 0.05, f'$z={z[idx_z_ref]:.2f}$',
        va='center', ha='left',
        transform=ax.transAxes,
        color='black', fontsize=30)


    leg = ax.legend(
        [l_WL, l_WL_c],
        [r"NFW", r"NFW $r_\mathrm{s}$ free"],
        loc=4, frameon=True, framealpha=0.6, markerfirst=False,
    )
    leg.get_frame().set_linewidth(0)

    ax.set_ylim(0.87, 1.05)
    ax.set_xscale("log")

    ax.set_xlabel(r'$m_\mathrm{500c,true} \, [h^{-1} \, \mathrm{M_\odot}]$')
    ax.set_ylabel(r'$m_{\mathrm{200m},i}/m_\mathrm{200m,dmo,true}$')

    fname = f'm200m_dmo_vs_WL_fgas_r_{model_fname_append}'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def check_mass_ratio_m200m_obs_dmo(
        z_ref=0.43,
        data_dir=TABLE_DIR,
        model_fname_base='planck2019_z_0p43_m500c_13p5-15p5_nbins_4',
        model_fname_range='R_0p75-2p5'):
    """Plot the ratio between the best-fitting and the true m200m_dmo"""
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
    plt.style.use("paper")
    fig = plt.figure(1)
    fig.set_size_inches(10, 9, forward=True)
    ax_obs = fig.add_axes([0.15, 0.6, 0.8, 0.35])
    ax_dmo = fig.add_axes([0.15, 0.25, 0.8, 0.35])
    ax_c = fig.add_axes([0.15, 0.15, 0.8, 0.1])

    # plot concentration ratio
    ax_c.axhline(y=1, c='k', ls='--')
    ax_c.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    ax_c.axhspan(0.95, 1.05, facecolor="k", alpha=0.1)
    ax_c.plot(
        m500c, rs_WL[idx_z_ref] / rs_dmo[idx_z_ref],
        dashes=dashes_WL
    )
    ax_c.plot(
        m500c, rs_WL_rs[idx_z_ref] / rs_dmo[idx_z_ref],
        dashes=dashes_WL_rs
    )

    ax_dmo.axhline(y=1, c="k", ls="--")
    ax_dmo.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    ax_dmo.axhspan(0.98, 1.02, facecolor="k", alpha=0.1)

    ax_obs.axhline(y=1, c="k", ls="--")
    ax_obs.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    ax_obs.axhspan(0.98, 1.02, facecolor="k", alpha=0.1)

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
        color='black', fontsize=30)

    # ax_dmo.text(
    #     0.05, 0.95, '(a)', weight='bold',
    #     va='top', ha='left',
    #     transform=ax_dmo.transAxes,
    #     color='black', fontsize=30)
    # ax_obs.text(
    #     0.05, 0.95, '(b)', weight='bold',
    #     va='top', ha='left',
    #     transform=ax_obs.transAxes,
    #     color='black', fontsize=30)
    # ax_c.text(
    #     0.05, 0.9, '(c)', weight='bold',
    #     va='top', ha='left',
    #     transform=ax_c.transAxes,
    #     color='black', fontsize=30)


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
    # ax_c.set_ylabel(
    #     r'$\frac{r_{\mathrm{s},i}}{r_\mathrm{s,dmo}}$',
    #     labelpad=15)

    ax_dmo.set_xticklabels([])
    ax_obs.set_xticklabels([])

    ax_dmo.set_ylabel(
        r'$m_{\mathrm{200m},i}/m_\mathrm{200m,dmo}$',
        fontsize=25, labelpad=15)
    # ax_dmo.set_ylabel(
    #     r'$\frac{m_{\mathrm{200m},i}}{m_\mathrm{200m,dmo}}$',
    #     labelpad=15)
    ax_obs.set_ylabel(
        r'$m_{\mathrm{200m},i}/m_\mathrm{200m,true}$',
        fontsize=25, labelpad=0)
    # ax_obs.set_ylabel(
    #     r'$\frac{m_{\mathrm{200m},i}}{m_\mathrm{200m,true}}$',
    #     labelpad=0)

    fname = f'm200m_dmo+obs_vs_WL_fgas_r_{model_fname_append}'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True)
    plt.show()


def check_shear_red_profile_bins(
        z_ref=0.43,
        m500c_refs=[10**13.97, 10**14.27, 10**14.52, 10**15],
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_nbins_4_R_0p75-2p5'):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
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
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(10, 12, forward=True)
    ax = fig.add_axes([0.15, 0.525, 0.8, 0.4])
    ax_r = fig.add_axes([0.15, 0.125, 0.8, 0.4])
    # ax_cb = fig.add_axes([0.45, 0.65, 0.475, 0.05])
    ax_cb = fig.add_axes([0.475, 0.45, 0.4, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=len(m500c_refs))

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
            ),
        )

    ax_r.text(
        r500c[idx_z_ref, idx_m_ref], 0.96,
        '$r_\mathrm{500c}$', va='bottom', ha='center',
        color='black', fontsize=30,
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
    ax_r.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    ax_r.axhspan(0.98, 1.02, facecolor="k", alpha=0.1)

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
        color='black', fontsize=30
    )

    ax.set_xticklabels([])
    ax_r.set_xlabel(r'$R \, [h^{-1} \, \mathrm{Mpc}]$')
    ax.set_ylabel(r'$g_\mathrm{T}(R)$')
    ax_r.set_ylabel(r'$g_\mathrm{T,fit}(R) / g_\mathrm{T,true}(R)$')

    fname=f'g_T_fit_{model_fname_append}_bins'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def check_shear_red_profile_significance_bins(
        z_ref=0.43,
        m500c_refs=[10**13.97, 10**14.27, 10**14.52, 10**15],
        sigma_e=0.25, n_mean=30,
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_nbins_4_R_0p75-2p5'):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
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

    shear_obs_err = mock_lensing.shape_noise(
        R_bins=R_bins, z_l=z_ref, cosmo=results['cosmo'],
        sigma_e=sigma_e, n_arcmin2=n_mean)

    idx_z_ref = np.argmin(np.abs(results['z'] - z_ref))
    idx_m_refs = np.array([
        np.argmin(np.abs(results['m500c'] - m500c_ref))
        for m500c_ref in m500c_refs
    ])

    # set up figure style and axes
    plt.clf()
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(10, 12, forward=True)
    ax = fig.add_axes([0.15, 0.525, 0.8, 0.4])
    ax_r = fig.add_axes([0.15, 0.125, 0.8, 0.4])
    # ax_cb = fig.add_axes([0.45, 0.65, 0.475, 0.05])
    ax_cb = fig.add_axes([0.2, 0.85, 0.4, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=len(m500c_refs))

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
            dashes=dashes_WL_rs, c=cmap(idx))
        lwc, = ax.plot(
            r_range, shear_red_WL_rs[idx_z_ref, idx_m_ref],
            dashes=dashes_WL_rs, c=cmap(idx))

        ls_d.append(ld)
        ls_t.append(lt)
        ls_w.append(lw)
        ls_wc.append(lwc)

        shear_WL_interp = interp.interp1d(
            r_range, shear_red_WL[idx_z_ref, idx_m_ref])
        for N in [1, 100, 1000, 10000]:
            SNR_WL = np.abs(
                (shear_WL_interp(R_obs) - shear_red_obs[idx_z_ref, idx_m_ref])
                         / (N**(-0.5) * shear_obs_err[0]))

            ax_r.plot(R_obs, SNR_WL, dashes=dashes_WL, c=cmap(idx))

            if idx == idx_m_refs.shape[0] - 1 and SNR_WL[-1] > 0.1:
                ax_r.text(
                    R_obs[-1], SNR_WL[-1],
                    f"$N=10^{{{np.log10(N).astype(int)}}}$",
                    ha="left", va="top")

        ax_r.annotate(
            '',
            xy=(r500c[idx_z_ref, idx_m_ref], 0.1),
            xytext=(r500c[idx_z_ref, idx_m_ref], 0.15),
            arrowprops=dict(
                facecolor=cmap(idx), shrink=0.,
            ),
        )

    # only add text for last arrow
    ax_r.text(
        r500c[idx_z_ref, idx_m_ref], 0.15,
        '$r_\mathrm{500c}$', va='bottom', ha='center',
        color='black', fontsize=30,
    )



    ax_r.text(
        0.95, 0.05, f"$z={z[idx_z_ref]:.2f}$",
        va='bottom', ha='right',
        transform=ax_r.transAxes,
        color='black', fontsize=30
    )

    ax_r.set_ylim(0.1, 4.99)
    ax_r.set_yscale('log')

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

    # ax_r.axhline(y=1, c="k", ls="--")
    # ax_r.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    # ax_r.axhspan(0.98, 1.02, facecolor="k", alpha=0.1)

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
    )

    # ax.text(
    #     0.5, 1.05,
    #     f"$m_\mathrm{{500c}} = 10^{{{np.log10(m500c[idx_m_ref]):.1f}}}"
    #     ' \, h^{-1} \, \mathrm{M_\odot}$',
    #     va='bottom', ha='center',
    #     transform=ax.transAxes,
    #     color='black', fontsize=30
    # )
    # ax_r.text(
    #     r200m[idx_z_ref, idx_m_ref], 1.025,
    #     '$r_\mathrm{200m}$', rotation=90, va='center', ha='left',
    #     color='black', fontsize=30,
    # )

    ax.set_xlim(
        left=R_bins.min() / 2,
        right=R_bins.max() + 2
    )
    ax.set_ylim(bottom=0, top=0.16)

    ax_r.set_xlim(
        left=R_bins.min() / 2,
        right=R_bins.max() + 2
    )

    ax.set_xticklabels([])
    ax_r.set_xlabel(r'$R \, [h^{-1} \, \mathrm{Mpc}]$')
    ax.set_ylabel(r'$g_\mathrm{T}(R)$')
    ax_r.set_ylabel(r'$|g_\mathrm{T,fit}(R) - g_\mathrm{T,true}(R)|/ \sigma $')

    fname=f'g_T_fit_{model_fname_append}_sigma_bins'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def plot_shear_red_profiles(
        z_ref=0.43,
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_nbins_4_R_0p75-2p5'):
    """Plot the fractional difference between the best-fitting gas profile
    and the true gas profile."""
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}.asdf',
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

    cmap = plot.get_partial_cmap(mpl.cm.plasma_r)
    norm = mpl.colors.Normalize(
        vmin=np.log10(m500c.min()), vmax=np.log10(m500c.max())
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap = sm.to_rgba

    # set up figure style and axes
    plt.style.use('paper')
    fig = plt.figure(1)
    fig.set_size_inches(10, 12, forward=True)
    ax = fig.add_axes([0.15, 0.525, 0.8, 0.4])
    ax_r = fig.add_axes([0.15, 0.125, 0.8, 0.4])
    ax_cb = fig.add_axes([0.45, 0.65, 0.475, 0.05])
    # ax_cb = fig.add_axes([0.54, 0.4425, 0.35, 0.05])

    ls_d = []
    ls_t = []
    ls_w = []
    ls_wc = []

    for idx_m, m in zip(np.log10(m500c)):
        # plot the observed shear
        lt, = ax.plot(
            r_range, shear_red_tot[idx_z_ref, idx_m], c=cmap(m)
        )

        ld = ax.errorbar(
            R_obs, shear_red_obs[idx_z_ref, idx_m],
            yerr=shear_red_err[idx_z_ref],
            marker="o", lw=0, elinewidth=1, c=cmap(m)
        )

        lw, = ax.plot(
            r_range, shear_red_WL[idx_z_ref, idx_m],
            c=cmap(m), dashes=dashes_WL
        )
        lwc, = ax.plot(
            r_range, shear_red_WL_rs[idx_z_ref, idx_m],
            c=cmap(m), dashes=dashes_WL_rs
        )

        ls_d.append(ld)
        ls_t.append(lt)
        ls_w.append(lw)
        ls_wc.append(lwc)

        ax_r.plot(
            r_range,
            shear_red_WL[idx_z_ref, idx_m] / shear_red_tot[idx_z_ref, idx_m],
            dashes=dashes_WL, c=lw.get_color()
        )
        ax_r.plot(
            r_range,
            shear_red_WL_rs[idx_z_ref, idx_m] / shear_red_tot[idx_z_ref, idx_m],
            dashes=dashes_WL_rs, c=lwc.get_color()
        )
        # ax.axvline(
        #     r200m[idx_z_ref, idx_m],
        #     c="k", ls="--"
        # )
        # ax.axvline(
        #     r500c[idx_z_ref, idx_m],
        #     c=cmap(m), ls="--"
        # )
        # ax_r.axvline(
        #     r200m[idx_z_ref, idx_m],
        #     c="k", ls="--"
        # )
        # ax_r.axvline(
        #     r500c[idx_z_ref, idx_m],
        #     c="k", ls="--"
        # )

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

    cb = plt.colorbar(sm, cax=ax_cb, orientation='horizontal')
    cb.set_label(
        r'$\log_{10} m_\mathrm{500c} \, [h^{-1} \,\mathrm{M_\odot}]$',
        labelpad=-5
    )

    # add the legend with data points, need horizontal spacing
    leg = ax.legend(
        [tuple(ls_d)[::10]],
        [r'"observed"'], loc=1,
        handler_map={tuple: plot.HandlerTuple(m500c[::10].shape[0])},
        frameon=False, markerfirst=False,
    )
    ax.add_artist(leg)

    # need empty handle to overlay with already present legend
    r = mpl.patches.Rectangle(
        (0,0), 1, 1, fill=False, edgecolor='none', visible=False
    )

    leg = ax.legend(
        [r, tuple(ls_t)[::10], tuple(ls_w)[::10], tuple(ls_wc)[::10]],
        [r"", r"true", r"NFW", r"NFW $r_\mathrm{s}$ free"],
        handler_map={tuple: plot.HandlerTupleVertical()},
        frameon=False, markerfirst=False,
    )

    # ax.text(
    #     0.5, 1.05,
    #     f"$m_\mathrm{{500c}} = 10^{{{np.log10(m500c[idx_m_ref]):.1f}}}"
    #     ' \, h^{-1} \, \mathrm{M_\odot}$',
    #     va='bottom', ha='center',
    #     transform=ax.transAxes,
    #     color='black', fontsize=30
    # )
    # ax_r.text(
    #     r500c[idx_z_ref, 0], 1.025,
    #     '$r_\mathrm{500c}$', rotation=90, va='center', ha='left',
    #     color='black', fontsize=30,
    # )
    # ax_r.text(
    #     r200m[idx_z_ref, -1], 1.025,
    #     '$r_\mathrm{200m}$', rotation=90, va='center', ha='left',
    #     color='black', fontsize=30,
    # )

    ax.set_xlim(
        left=R_bins.min() / 2,
        right=R_bins.max() + 2
    )
    ax.set_ylim(bottom=0, top=0.2)

    ax_r.set_xlim(
        left=R_bins.min() / 2,
        right=R_bins.max() + 2
    )
    ax_r.set_ylim(0.95, 1.049)

    ax_r.text(
        0.95, 0.05, f"$z={z[idx_z_ref]:.2f}$",
        va='bottom', ha='right',
        transform=ax_r.transAxes,
        color='black', fontsize=30
    )

    ax.set_xticklabels([])
    ax_r.set_xlabel(r'$R \, [h^{-1} \, \mathrm{Mpc}]$')
    ax.set_ylabel(r'$g_\mathrm{T}(R)$')
    ax_r.set_ylabel(r'$g_\mathrm{T,fit}(R) / g_\mathrm{T,true}(R)$')

    fname=f'g_T_fit_{model_fname_append}_m500c_all'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True)
    plt.show()


def check_deprojected_masses_all(
        z_ref=0.43,
        m500c_refs=[10 ** 14, 10 ** 14.5, 10 ** 15, 10 ** 15.5],
        data_dir=TABLE_DIR,
        model_fname_append='planck2019_z_0p43_m500c_13p5-15p5_nbins_4_R_0p75-2p5'):
    """Plot the ratio between the best-fitting and true deprojected masses."""
    with asdf.open(
            f'{TABLE_DIR}/results_fgas_r_{model_fname_append}.asdf',
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
    plt.style.use('paper')
    fig = plt.figure(figsize=(12, 12))
    ax0 = fig.add_axes([0.1, 0.5, 0.4, 0.4])
    ax1 = fig.add_axes([0.5, 0.5, 0.4, 0.4])
    ax2 = fig.add_axes([0.1, 0.1, 0.4, 0.4])
    ax3 = fig.add_axes([0.5, 0.1, 0.4, 0.4])
    ax_cb = fig.add_axes([0.505, 0.45, 0.23, 0.025])
    axs = [ax0, ax1, ax2, ax3]

    cmap = plot.get_partial_cmap(mpl.cm.plasma_r)
    norm = mpl.colors.Normalize(
        vmin=np.log10(m500c.min()), vmax=np.log10(m500c.max())
    )
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cmap = sm.to_rgba

    idx_z_ref = np.argmin(np.abs(z - z_ref))
    m_idcs = np.array([np.argmin(np.abs(m500c - m_ref)) for m_ref in m500c_refs])

    ls_wl = []
    ls_wl_c = []
    for idx, idx_m in enumerate(m_idcs):
        m = np.log10(m500c_refs[idx])
        ax = axs[idx]
        ax.text(
            0.05, 0.1,
            r'$\frac{m_\mathrm{500c}}{\mathrm{M_\odot}/h} = $'
            f'$10^{{{np.round(np.log10(m500c[idx_m]), 1)}}}$',
            va='bottom', ha='left',
            transform=ax.transAxes,
            color='black', fontsize=30,
            bbox=dict(boxstyle='round', facecolor='w', edgecolor="k", alpha=0.5)
        )

        ax.text(
            r500c[idx_z_ref, idx_m], 1.1,
            '$r_\mathrm{500c}$', rotation=90, va='top', ha='right',
            color=cmap(m), fontsize=30
        )
        ax.text(
            r200m[idx_z_ref, idx_m], 1.1,
            '$r_\mathrm{200m}$', rotation=90, va='top', ha='right',
            color=cmap(m), fontsize=30,
        )

        l_wl, = ax.plot(
            r_range, mr_WL[idx_z_ref, idx_m] / mr_tot[idx_z_ref, idx_m],
            dashes=dashes_WL, c=cmap(m)
        )
        l_wl_c, = ax.plot(
            r_range,
            mr_WL_rs[idx_z_ref, idx_m] / mr_tot[idx_z_ref, idx_m],
            dashes=dashes_WL_rs, c=cmap(m)
        )

        ls_wl.append(l_wl)
        ls_wl_c.append(l_wl_c)

        ax.axvline(
            x=r200m[idx_z_ref, idx_m],
            ls="--", c=cmap(m)
        )
        ax.axvspan(R_bins.min(), R_bins.max(), color="g", alpha=0.2)
        ax.axvline(x=r500c[idx_z_ref, idx_m], c="k", ls="--")

        ax.axhline(y=1, c="k", ls="--")
        ax.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
        ax.axhspan(0.95, 1.05, facecolor="k", alpha=0.1)

        ax.set_xlim(0.1, 6)
        ax.set_ylim(0.9, 1.1)
        ax.set_xscale("log")

    ticks = ax0.get_yticklabels()
    ticks[0].set_visible(False)

    ax0.set_xticklabels([])
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    ax3.set_yticklabels([])

    ax1.text(
        0.95, 0.95, f"$z={z[idx_z_ref]}$",
        va='top', ha='right',
        transform=ax1.transAxes,
        color='black', fontsize=30
    )

    leg = ax0.legend(
        [tuple(ls_wl)],
        [r"NFW"],
        handler_map={tuple: plot.HandlerTupleVertical()},
        bbox_to_anchor=(0, 0.95, 1, 0.1), loc=8,
        frameon=False, markerfirst=False,
    )
    leg.get_frame().set_linewidth(0)
    leg = ax1.legend(
        [tuple(ls_wl_c)],
        [r"NFW $r_\mathrm{s}$ free"],
        handler_map={tuple: plot.HandlerTupleVertical()},
        bbox_to_anchor=(0, 0.95, 1, 0.1), loc=8,
        frameon=False, markerfirst=False,
    )
    leg.get_frame().set_linewidth(0)

    ax2.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')
    ax3.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')

    ax0.set_ylabel(r'$m_\mathrm{NFW}(<r)/m_\mathrm{true}(<r)$')
    ax2.set_ylabel(r'$m_\mathrm{NFW}(<r)/m_\mathrm{true}(<r)$')

    fname = f'mr_enc_obs_vs_WL_reconstruction_{model_fname_append}'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True, bbox_inches='tight')
    plt.show()


def check_deprojected_masses_bins(
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
    plt.style.use('paper')
    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_axes([0.15, 0.15, 0.8, 0.8])
    ax_cb = fig.add_axes([0.525, 0.875, 0.4, 0.05])

    cmap = plot.get_partial_cmap_indexed(mpl.cm.plasma_r, N=len(m500c_refs))

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
            ),
        )

    ax.axvspan(R_bins.min(), R_bins.max(), color="g", alpha=0.2)
    ax.axhline(y=1, c="k", ls="--")
    ax.axhspan(0.99, 1.01, facecolor="k", alpha=0.3)
    ax.axhspan(0.95, 1.05, facecolor="k", alpha=0.1)

    ax.text(
        r500c[idx_z_ref, idx_m_refs[0]], 0.915,
        '$r_\mathrm{500c}$', va='bottom', ha='center',

        color='k', fontsize=30,
    )

    ax.set_xlim(0.1, 6)
    ax.set_ylim(0.9, 1.1)
    ax.set_xscale("log")

    ax.text(
        0.05, 0.05, f"$z={z[idx_z_ref]}$",
        va='center', ha='left',
        transform=ax.transAxes,
        color='black', fontsize=30
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
    )
    leg.get_frame().set_linewidth(0)

    ax.set_xlabel(r'$r \, [h^{-1} \, \mathrm{Mpc}]$')

    ax.set_ylabel(r'$m_\mathrm{NFW}(<r)/m_\mathrm{true}(<r)$')

    fname = f'mr_enc_obs_vs_WL_reconstruction_{model_fname_append}_bins'
    plt.savefig(PAPER_DIR + fname + '.pdf', transparent=True)
    plt.show()
