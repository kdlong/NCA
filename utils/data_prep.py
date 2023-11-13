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

    return df    

def prep_data_for_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop("outcome", axis=1)
    df["country"] = df["country"].cat.set_categories(["SE", "US", "GB", "Unreported", "Other"])
    df["country"][df["country"].isna()] = "Other"

    # Should be careful with this, a more conservative option would be to drop these, but that would bias the data on small number of cycles
    df.loc[df["cycle_length_std"].isna(), "cycle_length_std"] = df["cycle_length_std"].mean()
    df.loc[df["average_cycle_length"].isna(), "average_cycle_length"] = df["average_cycle_length"].mean()
     
    return df

def compare_km_cumulatives(df1: pd.DataFrame, df2: pd.DataFrame, label1: str, label2: str) -> plt.Axes:
    kmf1 = KaplanMeierFitter()
    kmf2 = KaplanMeierFitter()
    kmf1.fit(df1["n_cycles_trying"], df1["pregnant"])
    kmf2.fit(df2["n_cycles_trying"], df2["pregnant"])
    fig = plt.figure()
    ax = fig.add_subplot()
    kmf1.plot_cumulative_density(label=label1, ax=ax)
    fig = kmf2.plot_cumulative_density(label=label2, ax=ax)
    ax.set_ylabel("Prob(conception)")
    ax.set_xlabel("$n_{cycles}$")
    return fig

