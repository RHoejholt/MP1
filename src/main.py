import os
import hashlib
import glob
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# sti til data og output
data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "CSGO")
out_dir = os.path.join(os.path.dirname(__file__), "..", "outputs")
fig_dir = os.path.join(out_dir, "figures")
os.makedirs(fig_dir, exist_ok=True)


def find_csv_files(patterns=None):
    # hvis man giver patterns så brug dem, ellers tag alle csv‑filer
    if patterns:
        files = []
        for p in patterns:
            files += sorted(glob.glob(os.path.join(data_dir, p)))
    else:
        files = sorted(glob.glob(os.path.join(data_dir, "*.csv")))
    return files


def read_concat_csv(files, usecols=None, nrows=None, chunksize=None):
    # simpel læsning af csv‑filer og samling i én dataframe
    if not files:
        raise FileNotFoundError("no csv files")
    dfs = []
    for f in files:
        print("reading", f)
        if chunksize:
            it = pd.read_csv(f, usecols=usecols, chunksize=chunksize)
            part = pd.concat([c for c in it], ignore_index=True)
            if nrows:
                part = part.head(nrows)
            dfs.append(part)
        else:
            dfs.append(pd.read_csv(f, usecols=usecols, nrows=nrows))
    combined = pd.concat(dfs, ignore_index=True, sort=False)
    print("combined shape:", combined.shape)
    return combined


def inspect_df(df, name="df"):
    # lidt debuginfo
    print(f"--- inspect {name} ---")
    print("shape:", df.shape)
    print("cols:", df.columns.tolist())
    print(df.head())
    print(df.dtypes)
    print("missing (top 10):")
    print(df.isna().sum().sort_values(ascending=False).head(10))


def hash_series(s, salt="mini_project_salt"):
    # hash hver værdi i en serie
    return s.fillna("").astype(str).apply(
        lambda v: hashlib.sha256((v + salt).encode()).hexdigest()
    )


def basic_clean(df):
    # fjern ekstra mellemrum i kolonnenavne og gør tal til numerisk
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    for col in ("seconds", "tick", "round"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def anonymise_if_present(df, candidate_cols):
    # hash kolonner hvis de findes
    df = df.copy()
    for c in candidate_cols:
        if c in df.columns:
            print("anonymising", c)
            df[c] = hash_series(df[c])
    return df


def plot_weapon_counts(df, outpath=None, top_n=20):
    if "wp" not in df.columns:
        print("no wp column")
        return
    counts = df["wp"].value_counts().head(top_n)
    plt.figure(figsize=(8, 5))
    counts.plot.bar()
    plt.title("top weapons")
    plt.ylabel("count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def plot_events_per_round(df, outpath=None, sample_matches=5):
    if "round" not in df.columns:
        print("no round column")
        return
    match_col = "match_id" if "match_id" in df.columns else "file"
    ev = df.groupby([match_col, "round"]).size().reset_index(name="events")
    avg = ev.groupby("round")["events"].mean()
    plt.figure(figsize=(8, 4))
    avg.plot.line(marker="o")
    plt.title("average events per round")
    plt.xlabel("round")
    plt.ylabel("avg events")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def plot_damage_hist(df, outpath=None):
    if "damage" not in df.columns:
        print("no damage column")
        return
    plt.figure(figsize=(6, 4))
    df["damage"].dropna().astype(float).plot.hist(bins=50)
    plt.title("damage distribution")
    plt.xlabel("damage")
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath)
        plt.close()
    else:
        plt.show()


def main():
    # find csv‑filer
    files = find_csv_files(
        patterns=[
            "mm_master_demos*.csv",
            "mm_*.csv",
            "esea_meta_demos*.part*.csv",
            "mm_grenades_demos*.csv",
        ]
    )
    if not files:  # fallback
        files = find_csv_files()
    print("files:", files)

    # læs kun de første 150k rækker for hurtighed
    df = read_concat_csv(files, nrows=150000)

    inspect_df(df, "raw")
    df = basic_clean(df)

    # anonymiser id‑kolonner hvis de findes
    id_cols = [
        "attacker",
        "victim",
        "attacker_steamid",
        "victim_steamid",
        "player",
    ]
    df = anonymise_if_present(df, id_cols)

    # gem en lille prøve til csv
    sample_out = os.path.join(out_dir, "cleaned_sample_events.csv")
    df.head(20000).to_csv(sample_out, index=False)
    print("saved sample to", sample_out)

    # lav nogle simple plots
    plot_weapon_counts(df, outpath=os.path.join(fig_dir, "weapon_counts.png"))
    plot_events_per_round(df, outpath=os.path.join(fig_dir, "events_per_round.png"))
    plot_damage_hist(df, outpath=os.path.join(fig_dir, "damage_hist.png"))

    print("done, figs in", fig_dir)


if __name__ == "__main__":
    main()
