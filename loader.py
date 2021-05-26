import pandas as pd

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


def get_df_innova(include_max_weight=True, include_stability=False):
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
        discs_innova.drop(columns="max_weight", inplace=True)

    if include_stability:
        numbers_innova["stability"] = numbers_innova["turn"] + numbers_innova["fade"]
    else:
        del features["stability"]

    return discs_innova.join(numbers_innova, how="inner").astype(
        {feature: "float64" for feature in features}
    )


def normalize_df(df):
    """Given a dataframe with quantitative columns, normalize them to lie between 0 and 1."""
    df = df.copy(deep=True)
    for col in df.columns:
        col_min = min(df[col])
        col_max = max(df[col])
        df[col] = (df[col] - col_min) / (col_max - col_min)
    return df


def get_df_by_mfr(manufacturer):
    """Return a pd.DataFrame with the PDGA-registered physical features."""
    discs_all = get_df_pdga().dropna()
    discs_subset = discs_all[discs_all.manufacturer == manufacturer]
    discs_subset = discs_subset[["model"] + list(cols_quantitative)]
    discs_subset.reset_index(drop=True, inplace=True)
    discs_subset.set_index("model", inplace=True)
    return discs_subset.sort_index()


def get_df_pdga():
    """Return a pd.DataFrame with all PDGA-registered discs."""
    pdga = "pdga.csv"
    discs_all = pd.read_csv(
        pdga,
        header=0,
        names=cols_qualitative + list(cols_quantitative) + cols_aux,
        usecols=cols_qualitative + list(cols_quantitative),
    )
    return discs_all


def get_df_pdga_quantitative():
    """Return a pd.DataFrame with all PDGA-registered discs and their quantitative features."""
    discs_all = get_df_pdga().dropna()
    discs_all.reset_index(drop=True, inplace=True)
    discs_all.set_index(cols_qualitative, inplace=True)
    return discs_all.sort_index()


if __name__ == "__main__":
    df = get_df_by_mfr("Innova Champion Discs")
    # df = get_df_pdga()
    # df = normalize_df(get_df_innova(include_max_weight=False))
