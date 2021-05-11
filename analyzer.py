import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as pp

matplotlib.rcParams["text.usetex"] = True

import numpy as np
import pandas as pd

from itertools import combinations
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

cols_aux = [
    "class",
    "max_weight_vint",
    "last_year_prod",
    "cert_number",
    "date_approved",
]
cols_qualitative = ["manufacturer", "model"]
cols_quantitative = {
    "max_weight": "Max. weight (g)",
    "diameter": "Diameter (cm)",
    "height": "Height (cm)",
    "rim_depth": "Rim depth (cm)",
    "inside_rim_diameter": "Inside rim diameter (cm)",
    "rim_thickness": "Rim thickness (cm)",
    "rim_depth_to_diameter": "Rim depth to diameter ratio",
    "rim_config": "Rim configuration",
    "flexibility": "Flexibility",
}
features = {
    "speed": (1.0, 14.0),
    "glide": (0, 7),
    "turn": (-5, 2),
    "fade": (0.0, 5.0),
}


def get_innova_df():
    """Return a pd.DataFrame with the PDGA-registered physical features and flight numbers of each Innova disc."""
    discs_innova = get_df_by_mfr("Innova Champion Discs")

    numbers_innova = pd.read_csv(
        "innova.csv",
        names=["model", "speed", "glide", "turn", "fade", "abbreviation"],
        index_col="model",
        usecols=["model", "speed", "glide", "turn", "fade"],
        sep="\t",
    )

    return discs_innova.join(numbers_innova, how="inner").astype(
        {feature: "float64" for feature in features}
    )


def get_df_by_mfr(manufacturer):
    """Return a pd.DataFrame with the PDGA-registered physical features."""
    pdga = "pdga.csv"
    discs_all = pd.read_csv(
        pdga,
        header=0,
        names=cols_qualitative + list(cols_quantitative) + cols_aux,
        usecols=cols_qualitative + list(cols_quantitative),
    )
    discs_subset = discs_all[discs_all.manufacturer == manufacturer]
    discs_subset = discs_subset[["model"] + list(cols_quantitative)]
    discs_subset.reset_index(drop=True, inplace=True)
    discs_subset.set_index("model", inplace=True)

    return discs_subset


def get_col_subsets(df, cols, feature_to_predict, num_cols, threshold, degree=1):
    """Return the list of column subsets of size `num_cols` whose score for predicting
    `feature_to_predict` exceed the threshold."""

    col_subsets = []

    for col_subset in combinations(cols, num_cols):
        col_subset = list(col_subset)
        df_reduced = df[col_subset]
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(df_reduced, df[feature_to_predict])
        score = model.score(df_reduced, df[feature_to_predict])

        if score > threshold:
            col_subsets.append(col_subset)

    return col_subsets


def get_best_col_subset(df, cols, feature_to_predict, num_cols, degree=1):
    """Return the best size-`num_cols` subset of columns for predicting `feature_to_predict`."""

    best_score = None
    for col_subset in combinations(cols, num_cols):
        col_subset = list(col_subset)
        df_reduced = df[col_subset]
        model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
        model.fit(df_reduced, df[feature_to_predict])
        score = model.score(df_reduced, df[feature_to_predict])

        if best_score is None or score > best_score:
            best_score = score
            best_col_subset = col_subset

    return [best_col_subset]


def make_2d_plot(df, feature_to_predict, degree, cols):

    # Fit model, get score
    model = make_pipeline(PolynomialFeatures(degree), LinearRegression())
    model.fit(df[cols], df[feature_to_predict])
    score = model.score(df[cols], df[feature_to_predict])

    # Compute canvas bounds
    # We manually add a constant 10% margin as these are used to plot our colors.
    [x1, x2] = cols
    x1_min = min(df[x1])
    x1_max = max(df[x1])
    x1_margin = (x1_max - x1_min) * 0.1
    x1_min -= x1_margin
    x1_max += x1_margin

    x2_min = min(df[x2])
    x2_max = max(df[x2])
    x2_margin = (x2_max - x2_min) * 0.1
    x2_min -= x2_margin
    x2_max += x2_margin

    # Make predictions
    grid_points = (
        np.mgrid[x1_min:x1_max:50j, x2_min:x2_max:50j].reshape(2, -1).T
    )  # `grid_points` is a length-2500 list of ordered pairs
    Y = model.predict(grid_points).reshape(50, 50).T

    # Set up colormap keyword args
    y_min, y_max = features[feature_to_predict]
    cmap_kwargs = dict(vmin=y_min, vmax=y_max, cmap="RdYlGn_r")

    # Set up figure
    fig, ax = pp.subplots()
    fig.suptitle(None)
    ax.set_title(
        f"Predicting {feature_to_predict} with a degree-${degree}$ regression ($R^2 = {score:>6.4f}$)"
    )
    ax.set_xlabel(cols_quantitative[x1])
    ax.set_ylabel(cols_quantitative[x2])
    ax.margins(0.1)

    # Plot predictions
    img = ax.imshow(
        Y,
        extent=[x1_min, x1_max, x2_min, x2_max],
        aspect="auto",
        origin="lower",
        **cmap_kwargs,
    )

    # Plot input data
    ax.scatter(
        df[x1],
        df[x2],
        c=df[feature_to_predict],
        marker="o",
        edgecolors="black",
        alpha=0.5,
        **cmap_kwargs,
    )

    # Set up colorbar
    cb = fig.colorbar(img)
    cb.set_ticks(range(int(y_min), int(y_max) + 1))
    cb.set_label(feature_to_predict.capitalize())

    # Save and close figure
    pp.savefig(f"Figures/2d/degree-{degree}-{feature_to_predict}-{x1}-vs-{x2}.png")
    pp.close()


def analyze_2d(feature_to_predict, degree, threshold=0.9):
    df = get_innova_df()

    num_cols = 2
    col_pairs = get_col_subsets(
        df,
        cols_quantitative,
        feature_to_predict,
        num_cols=num_cols,
        threshold=threshold,
        degree=degree,
    )
    if not col_pairs:
        # No pair of columns achieved a score above the threshold
        col_pairs = get_best_col_subset(
            df, cols_quantitative, feature_to_predict, num_cols=num_cols, degree=degree
        )

    for cols_good in col_pairs:
        make_2d_plot(df, feature_to_predict, degree, cols_good)


def make_1d_plot(df, feature_to_predict, degree, col):

    # Fit model, get score.
    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(df[[col]], df[feature_to_predict])
    score = model.score(df[[col]], df[feature_to_predict])

    # Make predictions
    x_min = min(df[col])
    x_max = max(df[col])
    x_margin = (x_max - x_min) * 0.1
    x_min -= x_margin
    x_max += x_margin
    x_plot = np.linspace(x_min, x_max, 50)
    Y = model.predict(x_plot[:, np.newaxis])

    # Set up figure
    fig, ax = pp.subplots()
    fig.suptitle(None)
    ax.set_title(
        f"Predicting {feature_to_predict} with a degree-${degree}$ regression ($R^2 = {score:>6.4f}$)"
    )
    ax.set_xlabel(cols_quantitative[col])
    ax.set_ylabel(feature_to_predict.capitalize())
    ax.margins(0.10)

    # Plot input data
    ax.scatter(
        df[[col]], df[feature_to_predict], marker="o", edgecolors="black", alpha=0.25
    )
    # Plot predictions
    ax.plot(x_plot, Y)

    # Save figure
    pp.savefig(f"Figures/1d/degree-{degree}-{feature_to_predict}-{col}.png")
    pp.close()


def analyze_1d(feature_to_predict, degree, threshold=0.9):

    df = get_innova_df()

    num_cols = 1
    cols_best = get_col_subsets(
        df,
        cols_quantitative,
        feature_to_predict,
        num_cols=num_cols,
        threshold=threshold,
        degree=degree,
    )
    if not cols_best:
        # No column achieved a score above the threshold
        cols_best = get_best_col_subset(
            df, cols_quantitative, feature_to_predict, num_cols=num_cols, degree=degree
        )

    for [col] in cols_best:
        make_1d_plot(df, feature_to_predict, degree, col)


def analyze(feature_to_predict, degree):
    df = get_innova_df()
    print(f"\nPredicting {feature_to_predict} with a degree-{degree} regression.")

    model = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    model.fit(df[cols_quantitative], df[feature_to_predict])
    score = model.score(df[cols_quantitative], df[feature_to_predict])

    print(f"Score = {score}\n")


if __name__ == "__main__":

    threshold = 0.9
    for degree in [1, 2]:
        for feature_to_predict in features:
            print(f"Predicting {feature_to_predict} with a degree-{degree} regression.")
            analyze(feature_to_predict, degree=degree)
            analyze_1d(feature_to_predict, degree=degree, threshold=threshold)
            analyze_2d(feature_to_predict, degree=degree, threshold=threshold)
