from copy import deepcopy
from pathlib import Path

import asdf
import numpy as np
import scipy.optimize as opt

import matplotlib as mpl
import matplotlib.pyplot as plt

import lensing_haloes.settings as settings
import lensing_haloes.data.observational_data as obs_data
import lensing_haloes.halo.profiles as profs
import lensing_haloes.util.plot as plot

import pdb

TABLE_DIR = settings.TABLE_DIR
FIGURE_DIR = settings.FIGURE_DIR
PAPER_DIR = settings.PAPER_DIR
OBS_DIR = settings.OBS_DIR

RHO_CRIT = 2.7763458 * (10.0**11.0)  # [h^2 M_sun / Mpc^3]


def check_fgas_500c(datasets=['croston+08'], groupby='z'):
    """Plot the gas fraction of the different datasets."""
    data = obs_data.load_datasets(datasets=datasets)

    plt.clf()
    plt.style.use('paper')
    cmap = plot.get_partial_cmap(mpl.cm.plasma_r)

    for dataset in datasets:
        m500c = data[dataset]['m500c'][:]
        mgas_500c = data[dataset]['mgas_500c'][:]
        c = data[dataset][groupby][:]


        plt.scatter(
            m500c, mgas_500c / m500c,
            c=c, marker=data[dataset]['marker'],
            cmap=cmap, label=dataset
        )

    cb = plt.colorbar()
    cb.set_label(f'${groupby}$', rotation=270, labelpad=35)

    plt.xlabel(r'$m_\mathrm{500c} \, [h_{70}^{-1} \, \mathrm{M_\odot}]$')
    plt.ylabel(r'$f_\mathrm{gas,500c} \, [h_{70}^{-3/2}]$')
    plt.xscale('log')
    plt.ylim(top=0.2)
    plt.legend()
    plt.show()
