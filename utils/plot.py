import os
import argparse
import json
import itertools
import warnings
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from functools import singledispatch

MARKERS = ['o', 's', '^', 'x', 'v']
FLAVOR_MARKERS = ['o', 's', '^', 'x', 'v']
COLORS = ['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple']

SMALL_SIZE = 14
MEDIUM_SIZE = 14
BIG_SIZE = 16

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIG_SIZE)     # fontsize of the figure title


@singledispatch
def to_serializable(val):
    """Used by default."""
    return str(val)


@to_serializable.register(np.float32)
def ts_float32(val):
    """Used if *val* is an instance of numpy.float32."""
    return np.float64(val)


def plot_distrs(dataframe, names, fig_dir):
    """Plot distributions of response in a few representative bins."""

    binning = np.linspace(0.5, 1.5, num=101)
    pt_bins = [(30, np.inf), (30, 100), (100, 300), (300, 1000), (1000, np.inf)]
    eta_bins = [(0, 1.3), (1.3, 2.5)]

    histograms = {}
    for name in names:
        histograms[name] = {}
    for (ipt, pt_bin), (ieta, eta_bin) in itertools.product(
        enumerate(pt_bins), enumerate(eta_bins)
    ):
        df_bin = dataframe[
            (dataframe.pt_gen >= pt_bin[0]) & (dataframe.pt_gen < pt_bin[1])
            & (np.abs(dataframe.eta_gen) >= eta_bin[0])
            & (np.abs(dataframe.eta_gen) < eta_bin[1])
        ]
        for label, selection in [
            ('all', (df_bin.flavour != 0)),
            ('uds', (df_bin.flavour <= 3) & (df_bin.flavour != 0)),
            ('c', df_bin.flavour == 4),
            ('b', df_bin.flavour == 5),
            ('g', df_bin.flavour == 21)
        ]:
            for name in names:
                h, _ = np.histogram(df_bin[name][selection], bins=binning)
                histograms[name][ipt, ieta, label] = h

    for ipt, ieta, flavour in itertools.product(
        range(len(pt_bins)), range(len(eta_bins)), ['all', 'uds', 'c', 'b', 'g']
    ):
        fig = plt.figure(figsize=(6, 4.8))
        ax = fig.add_subplot()
        for i, name in enumerate(names):
            ax.hist(
                binning[:-1], weights=histograms[name][ipt, ieta, flavour],
                bins=binning, histtype='step', label=name, color=COLORS[i])
        ax.axvline(1., ls='dashed', lw=0.8, c='gray')
        ax.margins(x=0)
        ax.set_xlabel(
            r'$p_\mathrm{T}^\mathrm{corr}\//\/p_\mathrm{T}^\mathrm{gen}$')
        ax.set_ylabel('Jets')
        ax.legend(loc='upper right')
        if ipt == 0:
            ax.text(
                1., 1.002,
                r'${}$, $p_\mathrm{{T}}^\mathrm{{gen}} > {:g}$ GeV, '
                r'${:g} < |\eta^\mathrm{{gen}}| < {:g}$'.format(
                    flavour, pt_bins[ipt][0],
                    eta_bins[ieta][0], eta_bins[ieta][1]
                ),
                ha='right', va='bottom', transform=ax.transAxes
            )
        else:
            ax.text(
                1., 1.002,
                r'${}$, ${:g} < p_\mathrm{{T}}^\mathrm{{gen}} < {:g}$ GeV, '
                r'${:g} < |\eta^\mathrm{{gen}}| < {:g}$'.format(
                    flavour, pt_bins[ipt][0], pt_bins[ipt][1],
                    eta_bins[ieta][0], eta_bins[ieta][1]
                ),
                ha='right', va='bottom', transform=ax.transAxes
            )
        ax.tick_params(
            axis='both', which='both', direction='in', 
            bottom=True, top=True, left=True, right=True
        )
        ax.tick_params(axis='both', which='both', width=1.2)

        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(os.path.join(fig_dir, ext, f'{flavour}_pt{ipt + 1}_eta{ieta + 1}.{ext}'))
        
        if (ipt == 0) and (flavour == 'all'):
            plt.show()
        
        plt.close(fig)


def bootstrap_median(x, num=30):
    """Compute errors on median with bootstrapping."""

    if len(x) == 0:
        return np.nan

    medians = []
    for _ in range(num):
        x_resampled = np.random.choice(x, len(x))
        medians.append(np.median(x_resampled))
    return np.std(medians)


def compare_flavours(dataframe, names, fig_dir):
    """Plot median response as a function of jet flavour."""
    
    data = {}

    pt_bins = [(30, np.inf), (30, 100), (100, 300), (300, 1000), (1000, np.inf)]
    eta_bins = [(0, 1.3), (1.3, 2.5)]

    for (ipt, pt_bin), (ieta, eta_bin) in itertools.product(
        enumerate(pt_bins), enumerate(eta_bins)
    ):
        df_pteta = dataframe[
            (dataframe.pt_gen >= pt_bin[0]) & (dataframe.pt_gen < pt_bin[1])
            & (np.abs(dataframe.eta_gen) >= eta_bin[0])
            & (np.abs(dataframe.eta_gen) < eta_bin[1])
        ]
        median, median_error = {}, {}
        for name in names:
            median[name], median_error[name] = [], []
        flavours = [('u', {1}), ('d', {2}), ('s', {3}), ('c', {4}), ('b', {5}), ('g', {21})]
        for _, pdg_ids in flavours:
            df = df_pteta[df_pteta.flavour.isin(pdg_ids)]
            for name in names:
                median[name].append(df[name].median())
                median_error[name].append(bootstrap_median(df[name]))
        fig = plt.figure(figsize=(6, 4.8))
        ax = fig.add_subplot()
        offset = [0.1 * i for i in range(len(names))]
        offset = offset - np.mean(offset)
        for i, name in enumerate(names):
            if np.sum(np.isnan(median[name])):
                warnings.filterwarnings("ignore", message="Warning: converting a masked element to nan.")
            ax.errorbar(
                np.arange(len(flavours)) + offset[i], median[name], yerr=median_error[name],
                color=COLORS[i], marker=FLAVOR_MARKERS[i], ms=5, lw=0, elinewidth=0.8, label=name
            )
        ax.set_xlim(-0.5, len(flavours) - 0.5)
        ax.axhline(1, ls='dashed', lw=0.8, c='gray')
        if eta_bin == (0, 1.3):
            ax.set_yticks([0.99, 1.00, 1.01, 1.02])
        ax.set_xticks(np.arange(len(flavours)))
        xlabels = [f[0] for f in flavours]
        ax.set_xticklabels(xlabels)
        ax.legend()
        ax.set_ylabel('Median response')
        if pt_bins[ipt][1] == np.inf:
            ax.text(
                1., 1.002,
                r'$p_\mathrm{{T}}^\mathrm{{gen}} > {:g}$ GeV, '
                r'${:g} < |\eta^\mathrm{{gen}}| < {:g}$'.format(
                    pt_bins[ipt][0], eta_bin[0], eta_bin[1]
                ),
                ha='right', va='bottom', transform=ax.transAxes
            )
        else:
            ax.text(
                1., 1.002,
                r'${:g} < p_\mathrm{{T}}^\mathrm{{gen}} < {:g}$ GeV, '
                r'${:g} < |\eta^\mathrm{{gen}}| < {:g}$'.format(
                    pt_bins[ipt][0], pt_bins[ipt][1],
                    eta_bins[ieta][0], eta_bins[ieta][1]
                ),
                ha='right', va='bottom', transform=ax.transAxes
            )
        ax.tick_params(
            axis='both', which='both', direction='in', 
            bottom=True, top=True, left=True, right=True
        )
        ax.tick_params(axis='both', which='both', width=1.2)

        base_median = np.array(median['Baseline'])
        baseline = np.sum(np.abs(base_median - base_median.mean()))
        improvement = {}
        for name in names:
            median_arr = np.array(median[name])
            improvement[name] = 100 * (1 - np.sum(np.abs(median_arr - median_arr.mean())) / baseline)
            median[name] = dict(zip(xlabels, median[name]))
            median_error[name] = dict(zip(xlabels, median_error[name]))

        data[f'pt{ipt+1}eta{ieta+1}'] = {
            'improvement': improvement,
            'median': median, 
            'median_error': median_error
        }

        for ext in ['png', 'pdf', 'svg']:
            fig.savefig(os.path.join(fig_dir, ext, f'pt{ipt+1}eta{ieta+1}.{ext}'))
        
        if pt_bin == (30, np.inf):
            plt.show()
        
        plt.close(fig)

    with open(os.path.join(fig_dir, 'data.json'), 'w') as f:
        json.dump(data, f, indent='\t', default=to_serializable)


def plot_median_response(outdir, flavour_label, bins, bin_centers, eta_bin, ieta, names):
    """Plot median response as a function of pt."""

    median, median_error = {}, {}
    for name in names:
        median[name] = bins[name].median().to_numpy()
        median_error[name] = np.empty_like(median[name])
        for i, (_, df) in enumerate(bins):
            median_error[name][i] = bootstrap_median(df[name].to_numpy())

    fig = plt.figure(figsize=(6, 4.8))
    
    ax = fig.add_subplot()
    
    for i, name in enumerate(names):
        ax.errorbar(
            bin_centers, median[name], yerr=median_error[name],
            color=COLORS[i], ms=3, fmt=MARKERS[i], elinewidth=0.8, label=name
        )
    ax.axhline(1, ls='dashed', c='gray', alpha=.7)
    ax.set_xlabel(r'$p_\mathrm{T}^\mathrm{gen}$')
    ax.set_ylabel('Median response')
    ax.text(
        1., 1.002,
        '{}${:g} < |\\eta^\\mathrm{{gen}}| < {:g}$'.format(
            f'${flavour_label}$, ' if flavour_label != 'all' else '',
            eta_bin[0], eta_bin[1]
        ),
        ha='right', va='bottom', transform=ax.transAxes
    )
    ax.legend(loc='upper right')
    ax.tick_params(
        axis='both', which='both', direction='in', 
        bottom=True, top=True, left=True, right=True
    )
    ax.tick_params(axis='both', which='both', width=1.2)
    ax.set_xscale('log')

    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(outdir, ext, f'{flavour_label}_eta{ieta}.{ext}'))
        
    if flavour_label == 'all':
          plt.show()
    
    plt.close(fig)


def bootstrap_iqr(x, num=30):
    """Compute errors on IQR with bootstrapping."""

    if len(x) == 0:
        return np.nan

    iqrs = []
    for _ in range(num):
        x_resampled = np.random.choice(x, len(x))
        quantiles = np.percentile(x_resampled, [25, 75])
        iqrs.append(quantiles[1] - quantiles[0])
    return np.std(iqrs)


def compute_iqr(groups):
    """Compute IQR from series GroupBy."""
    
    q = groups.quantile([0.25, 0.75])
    iqr = q[1::2].values - q[0::2].values

    return iqr


def compute_resolution_improvement(df, names):    
    q1 = df.quantile(0.25)
    q2 = df.quantile(0.75)
    iqr = q2 - q1
    median = df.median()
    baseline = iqr['Baseline'] / median['Baseline']
    
    improvement = {}
    for name in names:
        iqr_median = iqr[name] / median[name]
        ratio = iqr_median / baseline
        improvement[name] = 100 * (1 - ratio)
    return improvement


def plot_resolution(outdir, flavour_label, bins, bin_centers, eta_bin, ieta, binning, names):
    np.seterr(invalid='ignore')
    median, iqr, iqr_error = {}, {}, {}
    for name in names:
        median[name] = bins[name].median().to_numpy()
        iqr[name] = compute_iqr(bins[name])
        iqr_error[name] = np.empty_like(iqr[name])
        for i, (_, df) in enumerate(bins):
            iqr_error[name][i] = bootstrap_iqr(df[name].to_numpy())

    fig = plt.figure(figsize=(6, 4.8))
    gs = mpl.gridspec.GridSpec(2, 1, hspace=0.02, height_ratios=[4, 1])
    axes_upper = fig.add_subplot(gs[0, 0])
    axes_lower = fig.add_subplot(gs[1, 0])

    iqr_median, iqr_median_error, ratio, improvement = {}, {}, {}, {}
    for i, name in enumerate(names):
        iqr_median[name] = iqr[name] / median[name]
        iqr_median_error[name] = iqr_error[name] / median[name]
        ratio[name] = iqr_median[name] / iqr_median['Baseline']
        improvement[name] = 100 * (1 - np.nanmean(ratio[name]))
        axes_upper.errorbar(
            bin_centers, iqr_median[name], yerr=iqr_median_error[name],
            color=COLORS[i], ms=3, marker=MARKERS[i], lw=0, elinewidth=0.8, label=name
        )
        if name != 'Baseline':
            axes_lower.plot(
                bin_centers, ratio[name], color=COLORS[i], ms=3, marker=MARKERS[i], lw=0
            )

    axes_upper.set_ylim(0, None)
    if eta_bin[0] == 0:
        axes_upper.set_ylim(0.0, 0.23)
        axes_lower.set_ylim(0.86, 1.04)
    else:
        axes_upper.set_ylim(0.0, 0.23)
        axes_lower.set_ylim(0.82, 1.04)
        axes_lower.set_yticks([0.9, 1.0])
    for axes in [axes_upper, axes_lower]:
        axes.set_xscale('log')
        axes.set_xlim(binning[0], binning[-1])
    axes_upper.xaxis.set_major_formatter(mpl.ticker.NullFormatter())
    axes_upper.xaxis.set_minor_formatter(mpl.ticker.NullFormatter())
    axes_upper.legend()
    axes_upper.text(
        1, 1.002,
        '{}${:g} < |\\eta^\\mathrm{{gen}}| < {:g}$'.format(
            f'${flavour_label}$, ' if flavour_label != 'all' else '',
            eta_bin[0], eta_bin[1]
        ),
        ha='right', va='bottom', transform=axes_upper.transAxes
    )
    axes_upper.set_ylabel('IQR / Median response')
    axes_lower.set_ylabel('Ratio')
    axes_lower.set_xlabel(r'$p_\mathrm{T}^\mathrm{gen}$')
    axes_upper.tick_params(
        axis='both', which='both', direction='in', width=1.2,
        bottom=True, top=True, left=True, right=True
    )
    axes_lower.tick_params(
        axis='both', which='both', direction='in', width=1.2,
        bottom=True, top=True, left=True, right=True
    )
    fig.align_ylabels()

    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(outdir, ext, f'{flavour_label}_eta{ieta}_iqr.{ext}'))
    
    if flavour_label == 'all':
          plt.show()
    
    plt.close(fig)

    for name in names:
        iqr_median[name] = dict(zip(bin_centers, iqr_median[name]))
        iqr_median_error[name] = dict(zip(bin_centers, iqr_median_error[name]))
        ratio[name] =  dict(zip(bin_centers, ratio[name]))

    return {
        'improvement': {},
        'per_bin': improvement,
        'iqr_median': iqr_median,
        'iqr_median_error': iqr_median_error,
        'ratio': ratio
    }

def plot_median_residual(outdir, bin_centers, flavour_labels, bins, eta_bin, ieta, names):
    """Plot difference in median response between flavours as a function of pt."""

    median, median_error, difference, error = {}, {}, {}, {}
    for name in names:
        median[name], median_error[name] = {}, {}
        for i in [0, 1]:
            median[name][i] = bins[i][name].median().to_numpy()
            median_error[name][i] = np.empty_like(median[name][i])
            for j, (_, df) in enumerate(bins[i]):
                median_error[name][i][j] = bootstrap_median(df[name].to_numpy())

        difference[name] = median[name][0] - median[name][1]
        error[name] = np.sqrt(median_error[name][0] ** 2 + median_error[name][1] ** 2)

    fig = plt.figure(figsize=(6, 4.8))
    ax = fig.add_subplot()
    for i, name in enumerate(names):
        ax.errorbar(
            bin_centers, difference[name], yerr=error[name], 
            color=COLORS[i], ms=3, fmt=MARKERS[i], elinewidth=0.8, label=name
        )
    ax.axhline(0, ls='dashed', c='gray', alpha=.7)
    ax.set_xlabel(r'$p_\mathrm{T}^\mathrm{gen}$')
    ax.set_ylabel('$R_{' + flavour_labels[0] + '}-R_{' + flavour_labels[1] + '}$')
    ax.text(
        1., 1.002,
        '${:g} < |\\eta^\\mathrm{{gen}}| < {:g}$'.format(eta_bin[0], eta_bin[1]),
        ha='right', va='bottom', transform=ax.transAxes
    )
    ax.set_xscale('log')
    ax.legend(loc='upper right')
    ax.tick_params(
        axis='both', which='both', direction='in', 
        bottom=True, top=True, left=True, right=True
    )

    for ext in ['png', 'pdf', 'svg']:
        fig.savefig(os.path.join(outdir, ext, f'{flavour_labels[0]}-{flavour_labels[1]}_eta{ieta}.{ext}'))
        
    if flavour_labels == ('uds', 'g'):
        plt.show()
    plt.close(fig)


def list_str(values):
    lst = values.split(',')
    return [val.strip() for val in lst]


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description=__doc__)
    arg_parser.add_argument('-p', '--predictions', required=True, type=list_str, help='Prediction files')
    arg_parser.add_argument('-n', '--names', required=True, type=list_str, help='Model name')
    arg_parser.add_argument('-o', '--outdir', required=True, help='Where to store plots')
    args = arg_parser.parse_args()

    try:
        os.makedirs(args.outdir)
    except FileExistsError:
        pass

    df = read_data(dict(zip(args.names, args.predictions)))

    for subdir in ['distributions', 'flavours', 'response', 'resolution', 'residual']:
        try:
            for ext in ['png', 'pdf', 'svg']:
                os.makedirs(os.path.join(args.outdir, subdir, ext))
        except FileExistsError:
            pass
    
    names = ['Baseline'] + args.names
    plot_distrs(df, names, os.path.join(args.outdir, 'distributions'))
    compare_flavours(df, names, os.path.join(args.outdir, 'flavours'))

    binning = np.geomspace(20, 3000, 20)
    bin_centers = np.sqrt(binning[:-1] * binning[1:])

    data = {}
    for (ieta, eta_bin), (flavour_label, flavour_ids) in itertools.product(
        enumerate([(0, 1.3), (1.3, 2.5)], start=1),
        [
            ('uds', {1, 2, 3}), ('c', {4}), ('b', {5}), ('g', {21}),
            ('all', {0, 1, 2, 3, 4, 5, 21})
        ]
    ):
        df_bin = df[
            (np.abs(df.eta_gen) >= eta_bin[0])
            & (np.abs(df.eta_gen) < eta_bin[1])
            & df.flavour.isin(flavour_ids)
        ]
        bins = df_bin.groupby(pd.cut(df_bin.pt_gen, binning))

        plot_median_response(
            os.path.join(args.outdir, 'response'),
            flavour_label, bins, bin_centers, eta_bin, ieta, names
        )

        data[f'{flavour_label}_eta{ieta}'] = plot_resolution(
            os.path.join(args.outdir, 'resolution'),
            flavour_label, bins, bin_centers, eta_bin, ieta, binning, names
        )

        for (ipt, pt_bin) in enumerate(
                [(30, np.inf), (30, 100), (100, 300), (300, 1000), (1000, np.inf)], start=1
            ):
            pt_bin = df_bin[
                (df_bin.pt_gen >= pt_bin[0])
                & (df_bin.pt_gen < pt_bin[1])
            ]
            data[f'{flavour_label}_eta{ieta}']['improvement'][f'pt{ipt}'] = compute_resolution_improvement(pt_bin, names)

    with open(os.path.join(args.outdir, 'resolution', 'data.json'), 'w') as f:
        json.dump(data, f, indent='\t', default=to_serializable)
    
    for (ieta, eta_bin), flavours in itertools.product(
        enumerate([(0, 1.3), (1.3, 2.5)], start=1),
        itertools.combinations([('uds', {1, 2, 3}), ('c', {4}), ('b', {5}), ('g', {21})], r=2),
    ):
        bins = []
        for i, flavour_ids in enumerate([flavours[0][1], flavours[1][1]]):
            df_bin = df[
                (np.abs(df.eta_gen) >= eta_bin[0])
                & (np.abs(df.eta_gen) < eta_bin[1])
                & df.flavour.isin(flavour_ids)
            ]
            bins.append(df_bin.groupby(pd.cut(df_bin.pt_gen, binning)))

        plot_median_residual(
            os.path.join(args.outdir, 'residual'),
            bin_centers, (flavours[0][0], flavours[1][0]), bins, eta_bin, ieta, names
        )