# import matplotlib
# import matplotlib.cm as cm
import matplotlib.pyplot as plt

# from matplotlib import ticker

# matplotlib.rcParams["text.usetex"] = True

import numpy as np
import pandas as pd

from sklearn.manifold import Isomap, TSNE
from sklearn.decomposition import (
    PCA,
    IncrementalPCA,
    KernelPCA,
    SparsePCA,
    TruncatedSVD,
)

from loader import cols_quantitative, features, get_innova_df, get_innova_df_normalized


def make_isomap_plot():
    df = get_innova_df()

    nbd_sizes = [1, 3, 5, 10, 15]  # Sizes of neighborhoods to try
    num_nbd_sizes = len(nbd_sizes)
    num_features = len(features)
    fig, axs = plt.subplots(
        num_features,
        num_nbd_sizes,
        figsize=(num_nbd_sizes * 4, num_features * 4),
    )

    for ax, feature in zip(axs[:, 0], features):
        ax.set_ylabel(f"Color = {feature}", rotation=90, size="large")

    for col_idx, nbd_size in enumerate(nbd_sizes):
        axs[0, col_idx].set_title(f"n_neighbors = {nbd_size}")
        projected = Isomap(n_components=2, n_neighbors=nbd_size).fit_transform(
            df[cols_quantitative]
        )
        for row_idx, feature in enumerate(features):
            axs[row_idx, col_idx].scatter(
                projected[:, 0], projected[:, 1], c=df[feature]
            )

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    plt.savefig("Figures/Dim-Reduce/isomap.png")
    plt.close()


def make_tsne_plot():
    df = get_innova_df_normalized()

    perplexities = [5, 10, 15, 20, 25]
    num_perplexities = len(perplexities)
    num_features = len(features)

    fig, axs = plt.subplots(
        num_features, num_perplexities, figsize=(num_perplexities * 4, num_features * 4)
    )

    for ax, feature in zip(axs[:, 0], features):
        ax.set_ylabel(f"Color = {feature}", rotation=90, size="large")

    for col_idx, perplexity in enumerate(perplexities):
        axs[0, col_idx].set_title(f"Perplexity {perplexity}")
        tsne = TSNE(perplexity=perplexity).fit_transform(df)
        for row_idx, feature in enumerate(features):
            axs[row_idx, col_idx].scatter(tsne[:, 0], tsne[:, 1], c=df[feature])

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    plt.savefig("Figures/Dim-Reduce/tsne.png")
    plt.close()


def make_pca_plot():
    df = get_innova_df()

    methods = [
        (PCA, "PCA"),
        (IncrementalPCA, "IncrementalPCA"),
        (KernelPCA, "KernelPCA"),
        (SparsePCA, "SparsePCA"),
        (TruncatedSVD, "TruncatedSVD"),
    ]
    num_methods = len(methods)
    num_features = len(features)
    fig, axs = plt.subplots(
        num_features, num_methods, figsize=(num_methods * 4, num_features * 4)
    )

    for ax, feature in zip(axs[:, 0], features):
        ax.set_ylabel(f"Color = {feature}", rotation=90, size="large")

    for col_idx, (method, title) in enumerate(methods):
        axs[0, col_idx].set_title(title)
        projected = method(n_components=2).fit_transform(df[cols_quantitative])
        for row_idx, feature in enumerate(features):
            axs[row_idx, col_idx].scatter(
                projected[:, 0], projected[:, 1], c=df[feature]
            )

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    plt.savefig("Figures/Dim-Reduce/pca.png")
    plt.close()


if __name__ == "__main__":

    make_isomap_plot()
    make_pca_plot()
    make_tsne_plot()
