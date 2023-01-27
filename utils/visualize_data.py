import os
import argparse
import uproot
import numpy as np
import pandas as pd
import awkward as ak
import matplotlib as mpl
import matplotlib.pyplot as plt
from glob import glob

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIG_SIZE = 20

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)     # fontsize of the figure title


def plot_spectrum(jet_df, outdir):
    pt_bins = np.geomspace(20, 5000, 50)
    eta_bins = np.linspace(0, jet_df.eta_gen.abs().max(), 30)

    H, _, _ = np.histogram2d(jet_df.pt_gen, jet_df.eta_gen.abs(), bins=(pt_bins, eta_bins))

    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot()

    plt.imshow(np.flipud(H.T), aspect='auto', norm=mpl.colors.LogNorm(vmin=1, vmax=H.max()))
    plt.xlabel('$p^\\mathrm{{gen}}_{T}$')
    plt.ylabel('$|\\eta^\\mathrm{{gen}}|$')
    plt.colorbar()

    xlim = plt.gca().get_xlim()
    xrange = np.linspace(xlim[0], xlim[1], 50)
    pt_ticks = [100, 1000]
    plt.xticks(ticks=np.interp(pt_ticks, pt_bins, xrange), labels=pt_ticks)

    minor_ticks = list(range(0, 100, 10)) + list(range(100, 1000, 100)) + list(range(1000, 5000, 1000))
    ax.set_xticks(ticks=np.interp(minor_ticks, pt_bins, xrange), minor=True)

    ylim = plt.gca().get_ylim()
    yrange = np.linspace(ylim[0], ylim[1], 30)
    eta_ticks = [0, 0.5, 1, 1.5, 2, 2.5]
    plt.yticks(ticks=np.interp(eta_ticks, eta_bins, yrange), labels=eta_ticks)
    
    plt.show()
    
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(outdir, ext, f'spectrum.{ext}'))
    plt.close(fig)


def plot_target(jet_df, outdir):
    plt.hist(np.log(jet_df.pt_gen / jet_df.pt), bins=50)
    plt.xlabel('$\log(p_T^\mathrm{gen} / p_T^\mathrm{reco})$')
    plt.ylabel('Fraction of jets/bin')
    plt.xlim([-1, 1])
    plt.show()
    
    for ext in ['png', 'pdf']:
        plt.savefig(os.path.join(outdir, ext, f'target.{ext}'))
