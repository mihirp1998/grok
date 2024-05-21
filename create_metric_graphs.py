#!/usr/bin/env python
# coding: utf-8

# Render metrics graphs

import csv
import logging
import os
import glob
import socket
from argparse import ArgumentParser

from collections import defaultdict

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import torch

from mpl_toolkits.axes_grid1 import make_axes_locatable
from tqdm import tqdm

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

import grok
from grok.visualization import *

from adjustText import adjust_text

# from grok_runs import RUNS

logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger("grok.view_metrics")
logger.setLevel(logging.ERROR)


limits = {
    "min_val_accuracy": 0,
    "max_val_accuracy": 100,
    "min_T": 0,  # 0
    "max_T": 100,  # 87.5
    "min_D": 0,  # 8
    "max_D": 256,  # 256
    "min_H": 0,  # 1
    "max_H": 4,  # 8
    "min_L": 0,  # 1
    "max_L": 4,  # 4
    "min_accuracy": 0,
    "max_accuracy": 100,
}

for k in limits.keys():
    metric = k.replace("min_", "").replace("max_", "")
    assert (
        limits["max_" + metric] >= limits["min_" + metric]
    ), f"invalid {metric} limits"


parser = ArgumentParser()
parser.add_argument("-i", "--image_dir", type=str, default=IMAGE_DIR)
args = parser.parse_args()


def create_loss_curves(
    metric_data,
    epochs,
    run,
    most_interesting_only=False,
    image_dir=args.image_dir,
    ds_len=None,
    cmap=DEFAULT_CMAP,
):
    scales = {
        "x": "log",
        "y": "linear",
    }


    arch = list(metric_data.keys())[0]

    ncols = 2
    nrows = 3
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    add_metric_graph(
        fig, axs[0, 0], "val_loss", metric_data, scales, cmap=cmap, ds_len=ds_len
    )
    add_metric_graph(
        fig, axs[0, 1], "val_accuracy", metric_data, scales, cmap, ds_len=ds_len
    )
    add_metric_graph(
        fig, axs[1, 0], "train_loss", metric_data, scales, cmap, ds_len=ds_len
    )
    add_metric_graph(
        fig, axs[1, 1], "train_accuracy", metric_data, scales, cmap, ds_len=ds_len
    )
    add_metric_graph(
        fig,
        axs[2, 0],
        "learning_rate",
        metric_data,
        scales,
        cmap,  # ds_len=ds_len
    )
    fig.suptitle(f"{operation} {list(data.keys())[0]}")
    fig.tight_layout()

    img_file = f"{image_dir}/loss_curves/{operation}_loss_curves_{arch}"
    if ds_len is not None:
        img_file += "_by_update"
    if most_interesting_only:
        img_file += "_most_interesting"
    img_file += ".png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


def create_max_accuracy_curves(
    metric_data, epochs, run, image_dir=args.image_dir, ds_len=None
):
    scales = {
        "x": "linear",
        "y": "linear",
    }

    ncols = 1
    nrows = 2
    fig_width = ncols * 8
    fig_height = nrows * 5
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    def get_ax(row=0, col=0, nrows=nrows, ncols=ncols, axs=axs):
        if nrows == 0:
            if ncols == 1:
                return axs
            else:
                return axs[col]
        else:
            if ncols == 1:
                return axs[row]
            else:
                return axs[row, col]

    add_extremum_graph(
        get_ax(0, 0), "val_accuracy", "max", metric_data, show_legend=False
    )
    add_extremum_graph(
        get_ax(1, 0), "train_accuracy", "max", metric_data, show_legend=False
    )
    fig.suptitle(f"{operation} {list(data.keys())[0]}")
    fig.tight_layout()

    expt = list(metric_data.keys())[0]
    img_file = f"{image_dir}/max_accuracy/{operation}_max_accuracy_{arch}.png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


VALID_OPERATORS = {
    "+": "addition",
    "-": "subtraction",
    "*": "muliplication",
    "/": "division",
    "**2+": "squarepoly",
    "**3+": "cubepoly",
    "x**2+y**2_mod_97": "quad1",
    "x**2+y**2+x*y_mod_97": "quad2",
    "x**2+y**2+x*y+x_mod_97": "quad3",
    "x**3+x*y_mod_97": "cube1",
    "x**3+x*y**2+y_mod_97": "cube2",
    "(x._value//y)if(y._value%2==1)else(x-y)_mod_97": "mix1",
    "s5": "s5",
    "s5conj": "s5conj",
    "s5aba": "s5aba",
    "+*": "even-addition_odd-multiplication",
    "+-": "even-addition_odd-subtraction",
    "pfactor" : "prime_factors",
    "2x" : "2x",
    "x**3" : "x**3",
    "2x+1" : "2x+1",
    "x+11" : "x+11",
    "sort": "sort",
    "reverse": "reverse",
    "copy": "copy",
    "caesarcipher": "caesarcipher",
    "permutev1": "permutev1",
    "permutev2": "permutev2",
    "permutev3": "permutev3",
    "strdeletev1": "strdeletev1",
    "strdeletev2": "strdeletev2",
    "caesarcipher_permutev1": "caesarcipher_permutev1"
}

def create_tsne_graphs(operation, run_dir, image_dir=args.image_dir):

    saved_pt_dir = f"{run_dir}"

    import itertools
    vocab_start_idx = 2 + len(VALID_OPERATORS.keys())
    vocab_end_idx = -len(list(itertools.permutations(range(5))))

    epoch_num = 1024
    file = saved_pt_dir + f"/epoch_{epoch_num}.ckpt"

    print(f"Loading {file}")
    saved_pt = torch.load(file)

    linear_weight = saved_pt['state_dict']['transformer.linear.weight']

    linear_weight = linear_weight[vocab_start_idx:vocab_end_idx]


    print("Doing T-SNE..")
    loss_tsne = PCA(n_components=2).fit_transform(linear_weight.detach().cpu().numpy())

    # loss_tsne = TSNE(n_components=2, init="pca").fit_transform(linear_weight.detach().cpu().numpy())
    # print("...done T-SNE.")

    ncols = 1
    nrows = 1
    fig_width = ncols * 12
    fig_height = nrows * 8
    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, figsize=(fig_width, fig_height))

    # Define colors based on residue modulo 15
    residues = np.array([(i + vocab_start_idx) % 5 for i in range(loss_tsne.shape[0])])
    cmap = plt.get_cmap('tab20')
    colors = [cmap(residue) for residue in residues]
    scatter = axs.scatter(loss_tsne[:, 0], loss_tsne[:, 1], c=residues, cmap='tab20', alpha=0.7, edgecolors='w', s=100)

    # Add color bar
    cbar = plt.colorbar(scatter, ax=axs)
    cbar.set_label('Index % 5', fontsize=14)

    # Annotate each point with its index and draw lines to (i + 15) % 97
    num_points = loss_tsne.shape[0]
    annotations = []
    for i in range(num_points):
        txt = i
        annotations.append(axs.annotate(txt, (loss_tsne[i, 0], loss_tsne[i, 1]), fontsize=10, color='black'))
        target_idx = (i + 5) % 97
        if residues[i] == residues[target_idx]:
            line_color = colors[i]
        else:
            line_color = 'gray'
        axs.plot([loss_tsne[i, 0], loss_tsne[target_idx, 0]], 
                 [loss_tsne[i, 1], loss_tsne[target_idx, 1]], color=line_color, linewidth=0.8, alpha=0.8)

    # Adjust text to avoid overlap
    adjust_text(annotations, arrowprops=dict(arrowstyle='->', color='gray', lw=0.5))

    img_file = f"{image_dir}/tsne/{operation}_temp.png"
    d = os.path.split(img_file)[0]
    os.makedirs(d, exist_ok=True)
    print(f"Writing {img_file}")
    fig.savefig(img_file)
    plt.close(fig)


create_tsne_graphs('addition', run_dir=f"{DATA_DIR}")
