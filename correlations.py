import matplotlib.pyplot as plt

from loader import cols_quantitative
from dim_reduce import get_df_pdga_quantitative
from math import ceil


def plot_all_pairs():
    df = get_df_pdga_quantitative()
    features = list(cols_quantitative)
    num_features = len(features)
    fig, axes = plt.subplots(
        num_features,
        num_features,
        figsize=(num_features * 3, num_features * 3),
        sharex="col",
        sharey="row",
        subplot_kw=dict(
            xticks=[],
            yticks=[],
        ),
        constrained_layout=True,
    )

    for idx, (col, title) in enumerate(cols_quantitative.items()):
        axes[idx, 0].set_ylabel(title, rotation=90)

        axes[0, idx].xaxis.set_label_position("top")
        axes[0, idx].set_xlabel(title)

    for col_idx, col_feature in enumerate(features):
        for row_idx, row_feature in enumerate(features[: col_idx + 1]):
            axes[row_idx, col_idx].plot(
                df[col_feature], df[row_feature], marker="o", ls="", alpha=0.05
            )

    plt.savefig(f"Figures/Correlations/all.png")


def get_significant_correlations(df, cols, sort=True, threshold=0.5):
    """Given a dataframe and a list of cols, return a list of positively correlated
    column pairs and a list of negatively correlated column pairs with their correlations.
    """
    positive_correlations = dict()
    negative_correlations = dict()
    for idx, feature_1 in enumerate(cols):
        for feature_2 in cols[idx + 1 :]:
            correlation = df[feature_1].corr(df[feature_2])
            if correlation >= threshold:
                positive_correlations[(feature_1, feature_2)] = correlation
            elif correlation <= -threshold:
                negative_correlations[(feature_1, feature_2)] = correlation

    positive_correlations = positive_correlations.items()
    negative_correlations = negative_correlations.items()
    if sort:
        positive_correlations = sorted(
            positive_correlations, key=lambda x: x[1], reverse=True
        )
        negative_correlations = sorted(negative_correlations, key=lambda x: x[1])

    return positive_correlations, negative_correlations


def plot_significant_correlations(threshold=0.5):
    df = get_df_pdga_quantitative()
    positive_correlations, negative_correlations = get_significant_correlations(
        df, list(cols_quantitative), threshold=threshold
    )

    plot_correlations(
        df, positive_correlations, fig_title="Positively correlated features"
    )
    plt.savefig(f"Figures/Correlations/positive_correlations.png")
    plt.close()

    plot_correlations(
        df,
        negative_correlations,
        fig_title="Negatively correlated features",
    )
    plt.savefig(f"Figures/Correlations/negative_correlations.png")
    plt.close()


def plot_correlations(df, feature_pairs, fig_title, num_cols=5):
    num_axes = len(feature_pairs)
    num_rows = ceil(num_axes / num_cols)
    fig, axes = plt.subplots(
        ncols=num_cols,
        nrows=num_rows,
        figsize=(num_cols * 4, num_rows * 4),
        constrained_layout=True,
    )
    fig.suptitle(fig_title, fontsize=16)

    for ax, ((feature_1, feature_2), corr) in zip(axes.flat, feature_pairs):
        ax.plot(df[feature_1], df[feature_2], marker="o", alpha=0.05, ls="")
        ax.set_title(f"r = {corr:6.4f}")
        ax.set_xlabel(cols_quantitative[feature_1])
        ax.set_ylabel(cols_quantitative[feature_2])

    # Hide unused axes
    for ax in axes.flat[len(feature_pairs) :]:
        ax.axis("off")


if __name__ == "__main__":
    plot_all_pairs()
    # plot_significant_correlations()
