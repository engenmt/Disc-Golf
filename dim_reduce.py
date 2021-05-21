import matplotlib.pyplot as plt


from sklearn.manifold import Isomap, LocallyLinearEmbedding, SpectralEmbedding, TSNE
from sklearn.decomposition import (
    PCA,
    IncrementalPCA,
    KernelPCA,
    SparsePCA,
    TruncatedSVD,
)
from functools import partial


from loader import cols_quantitative, features, get_innova_df, get_innova_df_normalized


def make_plot(methods, file_out, dummy_method=False):
    df = get_innova_df_normalized()

    num_cols = len(methods)
    num_rows = len(features)
    assert (
        num_rows > 1
    ), f"Need to have more than one row, otherwise axs array indexing will fail!"

    if num_cols == 1:
        assert dummy_method, "Need more than one method or dummy_method=True!"
        [method] = list(methods.values())
        methods["repeat"] = method

    fig, axs = plt.subplots(
        num_rows,
        num_cols,
        figsize=(num_cols * 4, num_rows * 4),
    )

    for ax, feature in zip(axs[:, 0], features):
        ax.set_ylabel(f"Color = {feature}", rotation=90, size="large")

    for col_idx, (title, method) in enumerate(methods.items()):
        axs[0, col_idx].set_title(title)
        projected = method.fit_transform(df[cols_quantitative])
        for row_idx, feature in enumerate(features):
            axs[row_idx, col_idx].scatter(
                projected[:, 0], projected[:, 1], c=df[feature]
            )

    plt.setp(plt.gcf().get_axes(), xticks=[], yticks=[])
    fig.tight_layout()
    plt.savefig(f"Figures/Dim-Reduce/{file_out}.png")
    plt.close()


def make_isomap_plot():
    file_out = "isomap"
    nbd_sizes = [1, 3, 5, 10, 15]
    methods = {
        f"Isomap(n_neighbors = {nbd_size})": Isomap(
            n_components=2, n_neighbors=nbd_size
        )
        for nbd_size in nbd_sizes
    }
    make_plot(methods, file_out=file_out)


def make_tsne_plot():
    file_out = "tsne"
    perplexities = [5, 10, 15, 20, 25]
    methods = {
        f"TSNE(perplexity = {perplexity})": TSNE(perplexity=perplexity)
        for perplexity in perplexities
    }
    make_plot(methods, file_out=file_out)


def make_pca_plot():
    file_out = "pca"
    n_components = 2
    methods = {
        "PCA": PCA(n_components=n_components),
        "IncrementalPCA": IncrementalPCA(n_components=n_components),
        "KernelPCA": KernelPCA(n_components=n_components),
        "SparsePCA": SparsePCA(n_components=n_components),
        "TruncatedSVD": TruncatedSVD(n_components=n_components),
    }
    make_plot(methods, file_out=file_out)


def make_LLE_neighbors_plot():
    file_out = "lle_neighbors"
    nbd_sizes = range(10, 20, 2)
    methods = {
        f"LLE(n_neighbors = {nbd_size})": LocallyLinearEmbedding(
            n_neighbors=nbd_size, n_components=2
        )
        for nbd_size in nbd_sizes
    }
    make_plot(methods, file_out=file_out)


def make_LLE_methods_plot():
    file_out = "lle_methods"
    n_components = 2
    n_neighbors = 30
    LLE = partial(
        LocallyLinearEmbedding,
        n_neighbors=n_neighbors,
        n_components=n_components,
        eigen_solver="auto",
    )
    methods = {
        "LLE": LLE(method="standard"),
        "LTSA": LLE(method="ltsa"),
        "Hessian LLE": LLE(method="hessian"),
        "Modified LLE": LLE(method="modified"),
    }
    make_plot(methods, file_out=file_out)


def make_spectral_plot():
    file_out = "spectral"
    methods = {
        "Spectral NN": SpectralEmbedding(affinity="nearest_neighbors"),
        "Spectral RBF": SpectralEmbedding(affinity="rbf"),
    }
    make_plot(methods, file_out=file_out)


def make_methods_plot():
    file_out = "methods"
    n_components = 2
    n_neighbors = 20
    methods = {
        "LLE": LocallyLinearEmbedding(n_neighbors=n_neighbors),
        "Spectral NN": SpectralEmbedding(affinity="nearest_neighbors"),
        "Spectral RBF": SpectralEmbedding(affinity="rbf"),
        "PCA": PCA(n_components=n_components),
        "IncrementalPCA": IncrementalPCA(n_components=n_components),
        "KernelPCA": KernelPCA(n_components=n_components),
        "SparsePCA": SparsePCA(n_components=n_components),
        "TruncatedSVD": TruncatedSVD(n_components=n_components),
        f"TSNE(perplexity = {n_neighbors})": TSNE(perplexity=n_neighbors),
    }
    make_plot(methods, file_out=file_out)


if __name__ == "__main__":

    # make_LLE_methods_plot()
    # make_LLE_neighbors_plot()
    # make_isomap_plot()
    # make_pca_plot()
    # make_tsne_plot()
    # make_spectral_plot()
    make_methods_plot()
