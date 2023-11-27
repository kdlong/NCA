import pandas as pd
from lifelines import KaplanMeierFitter
import matplotlib.pyplot as plt

def load_and_prep_data(csv_input: str) -> pd.DataFrame:
    df_raw = pd.read_csv(csv_input)

    # This is just the entry number, which Pandas will fill in automatically
    df = df_raw.copy().drop(columns="Unnamed: 0")
    
    for colname in ["sleeping_pattern", "been_pregnant_before", "country", "sleeping_pattern", "education"]:
        col = df[colname]
        df.loc[pd.isna(col), colname] = "Unreported"
        df[colname] = df[colname].astype("category")      

    df["regular_cycle"] = df["regular_cycle"].astype(bool)
    df["pregnant"] = df["outcome"] == "pregnant"
    # Dedication > 1. is impossible, must be a recording error
    df.loc[df["dedication"] > 1., "dedication"] = 1.

    return df    

def prep_data_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("outcome", axis=1)
    df["country"] = df["country"].cat.set_categories(["SE", "US", "GB", "Unreported", "Other"])
    df["country"][df["country"].isna()] = "Other"

    # Should be careful with this, a more conservative option would be to drop these, but that would bias the data on small number of cycles
    df.loc[df["cycle_length_std"].isna(), "cycle_length_std"] = df["cycle_length_std"].mean()
    df.loc[df["average_cycle_length"].isna(), "average_cycle_length"] = df["average_cycle_length"].mean()

    merge_categories(df, "been_pregnant_before", ["No, never", "Unreported",], "Yes")
    merge_categories(df, "country", ["SE", "US", "GB", "Unreported",], "Other")

    # This makes no sense and could screw up the predictions. Perhaps dropping them is better,
    # but for the sake of the importance analysis, it's ok
    df.loc[(df["intercourse_frequency"] == 0.) & df["pregnant"], "intercourse_frequency"] = df["intercourse_frequency"].mean()
     
    return df

def compare_km_cumulatives(df: pd.DataFrame, split_var: str, lower_val: float, upper_val: float) -> plt.Axes:
    kmfup = KaplanMeierFitter()
    kmfdown = KaplanMeierFitter()

    dfup = df[df[split_var] >= upper_val]
    dfdown = df[df[split_var] < lower_val]

    kmfup.fit(dfup["n_cycles_trying"], dfup["pregnant"])
    kmfdown.fit(dfdown["n_cycles_trying"], dfdown["pregnant"])
    fig = plt.figure()
    ax = fig.add_subplot()
    kmfup.plot_cumulative_density(label=f"Participants {split_var} >= {upper_val}", ax=ax)
    fig = kmfdown.plot_cumulative_density(label=f"Participants {split_var} < {lower_val}", ax=ax)
    ax.set_ylabel("Prob(conception) within $n_{cycles}$")
    ax.set_xlabel("$n_{cycles}$")
    return fig

def merge_categories(df: pd.DataFrame, catname: str, to_keep: list[str], new_cat: str) -> None:
    df[catname] = df[catname].cat.set_categories(to_keep+[new_cat])
    df.loc[df[catname].isna(), catname] = new_cat
