#!/usr/bin/env python 

from __future__ import division, print_function

import logging
import argparse
import numpy as np
import pylab
import matplotlib as mpl
import matplotlib.pyplot as plt
import cPickle as pickle

from mpl_toolkits.mplot3d import Axes3D

from blocks.main_loop import MainLoop
from blocks.log import TrainingLog

FORMAT = '[%(asctime)s] %(name)-15s %(message)s'
DATEFMT = "%H:%M:%S"
logging.basicConfig(format=FORMAT, datefmt=DATEFMT, level=logging.INFO)


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser()
    parser.add_argument("model_file", help="filename the log to plot from")
    args = parser.parse_args()

    logging.info("Loading file %s..." % args.model_file)
    with open(args.model_file, "rb") as f:
        p = pickle.load(f)

    if isinstance(p, MainLoop):
        print("GOOD LUCK, BUT PLEASE USE A LOG!")
        assert(False)
    elif isinstance(p, TrainingLog):
        log = p

    plot_tag = args.model_file[0:-8]
    df = log.to_dataframe()
    df_keys = df.keys()

    ################################
    # PLOT VARIATIONAL FREE-ENERGY #
    ################################
    nll_bound_types = [k for k in df_keys if (k.find('nll_bound') > -1)]
    nll_bound_idx = df[nll_bound_types[0]].keys()[1:-5]
    nll_bound_idx = [i for i in nll_bound_idx if i < 40000]
    nll_bound_map = {}
    for k in nll_bound_types:
        idx = np.asarray(nll_bound_idx)
        vals = np.asarray(df[k][nll_bound_idx])
        nll_bound_map[k] = [idx, vals]

    nll_plot_name = "NLL_BOUNDS_{}.png".format(plot_tag)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    min_map = {}
    for k, v in nll_bound_map.items():
        x, y = v
        y_min = np.min(y)
        ax.plot(x, y, label=k)
        ax.plot(x, ((0.0*y) + y_min), label="min({0:s})={1:.4f}".format(k,y_min))
    ax.legend()
    fig.savefig(nll_plot_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
    #####################
    # PLOT PER-STEP KLD #
    #####################
    valid_kl_keys = [k for k in df_keys if (k.find('valid_kl_') > -1)]
    valid_kl_idx = df[valid_kl_keys[0]].keys()[1:-5]
    valid_kl_map = {}
    for k in valid_kl_keys:
        idx = np.asarray(valid_kl_idx)
        vals = np.asarray(df[k][valid_kl_idx])
        valid_kl_map[k] = [idx, vals]

    kl_plot_name = "KL_TERMS_{}.png".format(plot_tag)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.hold(True)
    for k, v in valid_kl_map.items():
        x, y = v
        ax.plot(x, y, label=k)
    ax.legend()
    fig.savefig(kl_plot_name, dpi=None, facecolor='w', edgecolor='w', \
        orientation='portrait', papertype=None, format=None, \
        transparent=False, bbox_inches=None, pad_inches=0.1, \
        frameon=None)
    plt.close(fig)
