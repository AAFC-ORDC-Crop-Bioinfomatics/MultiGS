#!/usr/bin/env python3
"""
RCBD analysis for multiple experiments with common checks.

Input format (tab-delimited assumed):
expt   bloc   name   Original AAFC Designation   pedigree   trait1   trait2   ...

- expt: experiment ID (string)
- bloc: block/rep within experiment (RCBD)
- name: line/cultivar name (used as line ID)
- pedigree: parents separated by "/" (e.g. "CDC Bethune/ CDC Mons")
- columns 6+ are numeric traits

Outputs (prefix based on input file name unless overridden):
- <prefix>_anova.tsv
- <prefix>_lsmeans.tsv
- <prefix>_lsmeans_calibrated.tsv
- <prefix>_traits_pairplot.png
- <prefix>_unique_parents.tsv
"""

import argparse
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # for non-interactive environments
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols


COMMON_CHECKS = ["AAC Bright", "CDC Bethune", "CDC Glas", "CDC Kernen", "CDC Plava"]


def read_data(infile, sep="\t"):
    df = pd.read_csv(infile, sep=sep)
    # Basic sanity
    required_cols = ["expt", "bloc", "name", "pedigree"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in input.")
    return df


def get_trait_columns(df):
    # Assuming col1-5 fixed: expt, bloc, name, Original AAFC Designation, pedigree
    # traits start from col index 5 onward
    if len(df.columns) <= 5:
        raise ValueError("No trait columns found (need columns beyond col 5).")
    return df.columns[5:].tolist()


def fit_rcbd_anova(df_expt, trait):
    """
    Fit RCBD model for a given experiment & trait:
      y ~ C(name) + C(bloc)
    Returns fitted model and ANOVA table.
    """
    sub = df_expt[["bloc", "name", trait]].dropna()
    if sub.empty:
        return None, None

    # Need at least 2 lines and 2 blocks
    if sub["name"].nunique() < 2 or sub["bloc"].nunique() < 2:
        return None, None

    formula = f"{trait} ~ C(name) + C(bloc)"
    model = ols(formula, data=sub).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    return model, anova_table


def compute_lsmeans_rcbd(df_expt, model, trait):
    """
    Compute LS-means (least squares means) for each line in an RCBD:
      trait ~ C(name) + C(bloc)

    For each line, we create a small new design with that line across all blocks,
    predict, and average the predictions.
    """
    sub = df_expt[["bloc", "name", trait]].dropna()
    blocks = sorted(sub["bloc"].unique())
    lines = sorted(sub["name"].unique())

    lsmeans = {}
    for ln in lines:
        # Build hypothetical data: all blocks, same line
        new_rows = [{"name": ln, "bloc": b} for b in blocks]
        new_df = pd.DataFrame(new_rows)
        preds = model.predict(new_df)
        lsmeans[ln] = float(preds.mean())
    return lsmeans


def analyze_rcbd(df, trait_cols, out_prefix):
    """
    (1) RCBD ANOVAs for each expt & trait
    (2) LS-means per expt & line & trait
    """
    anova_rows = []
    ls_data = {}  # key=(expt, name) -> dict with metadata + traits

    for expt_id, df_ex in df.groupby("expt"):
        # Prepare meta per line in this experiment
        for name, g in df_ex.groupby("name"):
            key = (expt_id, name)
            if key not in ls_data:
                row = {
                    "expt": expt_id,
                    "name": name,
                }
                # Keep Original AAFC Designation and pedigree if present
                if "Original AAFC Designation" in g.columns:
                    row["Original AAFC Designation"] = g["Original AAFC Designation"].iloc[0]
                if "pedigree" in g.columns:
                    row["pedigree"] = g["pedigree"].iloc[0]
                ls_data[key] = row

        for trait in trait_cols:
            model, anova = fit_rcbd_anova(df_ex, trait)
            if model is None or anova is None:
                continue

            # Store ANOVA rows
            for effect, r in anova.iterrows():
                anova_rows.append({
                    "expt": expt_id,
                    "trait": trait,
                    "effect": effect,
                    "df": r.get("df", np.nan),
                    "sum_sq": r.get("sum_sq", np.nan),
                    "mean_sq": r.get("mean_sq", np.nan),
                    "F": r.get("F", np.nan),
                    "PR(>F)": r.get("PR(>F)", np.nan)
                })

            # Compute LSmeans for this trait
            ls_trait = compute_lsmeans_rcbd(df_ex, model, trait)
            for ln, val in ls_trait.items():
                key = (expt_id, ln)
                if key not in ls_data:
                    # Should not happen, but guard
                    ls_data[key] = {"expt": expt_id, "name": ln}
                ls_data[key][trait] = val

    # Save ANOVA table
    if anova_rows:
        anova_df = pd.DataFrame(anova_rows)
        anova_path = f"{out_prefix}_anova.tsv"
        anova_df.to_csv(anova_path, sep="\t", index=False)
        print(f"[INFO] Saved ANOVA results to: {anova_path}")
    else:
        print("[WARN] No ANOVA results produced (check data).")

    # Build LSmeans wide table
    if ls_data:
        ls_df = pd.DataFrame(list(ls_data.values()))
        # Ensure columns ordering similar to input: expt, name, Original, pedigree, traits...
        base_cols = ["expt", "name"]
        if "Original AAFC Designation" in ls_df.columns:
            base_cols.append("Original AAFC Designation")
        if "pedigree" in ls_df.columns:
            base_cols.append("pedigree")
        other_cols = [c for c in ls_df.columns if c not in base_cols]
        ls_df = ls_df[base_cols + sorted(other_cols)]

        ls_path = f"{out_prefix}_lsmeans.tsv"
        ls_df.to_csv(ls_path, sep="\t", index=False)
        print(f"[INFO] Saved LS-means table to: {ls_path}")
    else:
        ls_df = pd.DataFrame()
        print("[WARN] No LS-means computed.")

    return ls_df


def calibrate_across_experiments(ls_df, trait_cols, out_prefix,
                                 checks=COMMON_CHECKS):
    """
    Calibrate trait scales across experiments using common check cultivars.
    - Choose one experiment as baseline (first in sorted order).
    - For each other experiment & trait:
        baseline_trait(checks) ~ current_trait(checks), linear regression
        apply y = a + b * x to all lines in that experiment.
    """
    if ls_df.empty:
        print("[WARN] LS-means table is empty; skipping calibration.")
        return ls_df

    expts = sorted(ls_df["expt"].unique())
    if len(expts) == 0:
        print("[WARN] No experiments for calibration.")
        return ls_df

    baseline_expt = expts[0]
    print(f"[INFO] Using baseline experiment for calibration: {baseline_expt}")

    cal_df = ls_df.copy()

    for trait in trait_cols:
        if trait not in cal_df.columns:
            continue

        # Baseline values for common checks
        base_sub = cal_df[(cal_df["expt"] == baseline_expt) &
                          (cal_df["name"].isin(checks)) &
                          cal_df[trait].notna()][["name", trait]]
        base_sub = base_sub.rename(columns={trait: "baseline_trait"})

        for ex in expts:
            if ex == baseline_expt:
                continue

            cur_sub = cal_df[(cal_df["expt"] == ex) &
                             (cal_df["name"].isin(checks)) &
                             cal_df[trait].notna()][["name", trait]]
            cur_sub = cur_sub.rename(columns={trait: "current_trait"})

            merged = pd.merge(base_sub, cur_sub, on="name", how="inner")
            if len(merged) < 3:
                print(f"[WARN] Not enough common checks with data for calibration: "
                      f"expt={ex}, trait={trait} (n={len(merged)})")
                continue

            x = merged["current_trait"].values
            y = merged["baseline_trait"].values

            if np.allclose(np.var(x), 0):
                print(f"[WARN] No variation in current trait for calibration: "
                      f"expt={ex}, trait={trait}")
                continue

            # Fit linear mapping: baseline ≈ a + b * current
            b, a = np.polyfit(x, y, 1)  # y = b*x + a
            print(f"[CALIB] expt={ex}, trait={trait}: y = {a:.4f} + {b:.4f} * x")

            mask = cal_df["expt"] == ex
            cal_df.loc[mask, trait] = a + b * cal_df.loc[mask, trait]

    cal_path = f"{out_prefix}_lsmeans_calibrated.tsv"
    cal_df.to_csv(cal_path, sep="\t", index=False)
    print(f"[INFO] Saved calibrated LS-means table to: {cal_path}")
    return cal_df


def plot_trait_correlation(cal_df, trait_cols, out_prefix):
    """
    Draw a correlation matrix plot with diagonal histograms (pairplot)
    using calibrated LS-means.
    """
    if cal_df.empty:
        print("[WARN] Calibrated LS-means table is empty; skipping correlation plot.")
        return

    traits_df = cal_df[trait_cols].dropna()
    if traits_df.empty:
        print("[WARN] No complete trait rows for correlation plotting.")
        return

    sns.set(style="whitegrid")
    g = sns.pairplot(traits_df, diag_kind="hist", corner=False)
    g.fig.suptitle("Trait correlations (calibrated LS-means)", y=1.02)

    out_png = f"{out_prefix}_traits_pairplot.png"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Saved trait correlation pairplot to: {out_png}")


def extract_unique_parents(df, out_prefix):
    """
    From 'pedigree' column, split by '/', strip whitespace,
    and export a file with unique parent names.
    """
    parents = set()
    if "pedigree" not in df.columns:
        print("[WARN] 'pedigree' column not found; skipping parent extraction.")
        return

    for ped in df["pedigree"].dropna():
        parts = ped.split("/")
        for p in parts:
            p_clean = p.strip()
            if p_clean and p_clean.upper() not in {"NA", "N/A", ".", "NULL"}:
                parents.add(p_clean)

    parents = sorted(parents)
    parent_df = pd.DataFrame({"parent": parents})
    out_path = f"{out_prefix}_unique_parents.tsv"
    parent_df.to_csv(out_path, sep="\t", index=False)
    print(f"[INFO] Saved unique parent list to: {out_path}")


def main():
    parser = argparse.ArgumentParser(description="RCBD analysis across multiple experiments with common checks.")
    parser.add_argument("--in", dest="infile", required=True, help="Input data file (TSV recommended)")
    parser.add_argument("--sep", default="\t", help="Field separator for input (default: tab)")
    parser.add_argument("--prefix", default=None, help="Output prefix (default: input file basename without extension)")
    args = parser.parse_args()

    infile = args.infile
    sep = args.sep

    if args.prefix is None:
        base = os.path.basename(infile)
        prefix = os.path.splitext(base)[0]
    else:
        prefix = args.prefix

    print(f"[INFO] Reading data from: {infile}")
    df = read_data(infile, sep=sep)

    trait_cols = get_trait_columns(df)
    print(f"[INFO] Detected trait columns: {trait_cols}")

    # (1) & (2): ANOVA + LS-means
    ls_df = analyze_rcbd(df, trait_cols, prefix)

    # (3): Calibration using common checks
    cal_df = calibrate_across_experiments(ls_df, trait_cols, prefix)

    # (4): Correlation matrix + diagonal histograms
    plot_trait_correlation(cal_df, trait_cols, prefix)

    # (5): Unique parents from pedigree
    extract_unique_parents(df, prefix)


if __name__ == "__main__":
    main()

