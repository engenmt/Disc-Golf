import numpy as np
import pandas as pd

from collections import Counter
from itertools import combinations


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
    "rim_thickness": "Rim width (cm)",
    "rim_depth_to_diameter": "Rim depth to diameter ratio",
    "rim_config": "Rim configuration",
    "flexibility": "Flexibility",
}
features = {
    "speed": (1, 14),
    "glide": (0, 7),
    "turn": (-5, 2),
    "fade": (0, 5),
    "stability": (-5, 7),
}


def get_innova_df(include_max_weight=True):
    """Return a pd.DataFrame with the PDGA-registered physical features and flight numbers of each Innova disc."""
    discs_innova = get_df_by_mfr("Innova Champion Discs")

    numbers_innova = pd.read_csv(
        "innova.csv",
        names=["model", "speed", "glide", "turn", "fade", "abbreviation"],
        index_col="model",
        usecols=["model", "speed", "glide", "turn", "fade"],
        sep="\t",
    )
    if not include_max_weight:
        discs_innova = discs_innova.drop(columns="max_weight")

    numbers_innova["stability"] = numbers_innova["turn"] + numbers_innova["fade"]

    return discs_innova.join(numbers_innova, how="inner").astype(
        {feature: "float64" for feature in features}
    )


def get_innova_df_normalized(**kwargs):
    """Return the Innova dataframe with quantitative values between 0 and 1."""
    return normalize_df(get_innova_df(**kwargs).copy(deep=True))


def normalize_df(df):
    """Given a dataframe with quantitative columns, normalize them to lie between 0 and 1."""
    for col in df.columns:
        col_min = min(df[col])
        col_max = max(df[col])
        df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


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


if __name__ == "__main__":
    # df = get_innova_df()

    df = get_innova_df_normalized()
    df = get_innova_df_normalized(include_max_weight=False)
