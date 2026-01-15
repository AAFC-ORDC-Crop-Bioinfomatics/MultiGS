#!/usr/bin/env python3
"""
Two-stage MET BLUP analysis for RCBD LS-means.

Stage 2 (this script):
    - Input: LS-means per experiment (output of rcbd_flax_2025_data_analysis.py)
    - Model (per trait):
        y_ij = mu + E_i (fixed) + G_j (random) + e_ij
      where:
        E_i: experiment effect (fixed)
        G_j: genotype/line effect (random)

    - Output:
        1) Overall genotype BLUPs per trait:
           <prefix>_met_blup_overall.tsv
        2) (optional) Predicted values per expt x genotype (G+E style):
           <prefix>_met_blup_by_expt.tsv
        3) Variance component summary per trait:
           <prefix>_met_varcomp.tsv
"""

import argparse
import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ---------- Utilities ----------

META_COLS_BASE = ["expt", "name", "Original AAFC Designation", "pedigree"]


def detect_trait_columns(df):
    """Detect numeric trait columns, assuming meta columns are known."""
    meta_cols = [c for c in META_COLS_BASE if c in df.columns]
    trait_candidates = [c for c in df.columns if c not in meta_cols]
    # Keep only numeric-ish columns
    traits = []
    for c in trait_candidates:
        if np.issubdtype(df[c].dtype, np.number):
            traits.append(c)
    if not traits:
        raise ValueError("No numeric trait columns detected in LS-means file.")
    return traits, meta_cols


def safe_random_effect_value(re_dict):
    """
    statsmodels MixedLM.random_effects[group] can be:
        - dict-like (e.g. {'Group': val} or {'Intercept': val})
        - or a scalar/array
    This function extracts a single scalar random intercept.
    """
    if re_dict is None:
        return 0.0
    # dict-like
    if isinstance(re_dict, dict):
        if len(re_dict) == 0:
            return 0.0
        # take first value
        return float(list(re_dict.values())[0])
    # array-like / scalar
    try:
        arr = np.asarray(re_dict).flatten()
        if arr.size == 0:
            return 0.0
        return float(arr[0])
    except Exception:
        return float(re_dict)


# ---------- MET BLUP core ----------

def fit_met_blup_for_trait(df_ls, trait, reml=True):
    """
    Fit MET mixed model for a single trait:
        y_ij = mu + E_i (fixed) + G_j (random) + e_ij

    Returns:
        res (MixedLMResults) or None on failure.
    """
    sub = df_ls[["expt", "name", trait]].dropna()
    if sub.empty:
        print(f"[WARN] Trait '{trait}': no data (all NA). Skipping.")
        return None

    if sub["expt"].nunique() < 2:
        print(f"[WARN] Trait '{trait}': only one experiment present. "
              f"MET model not meaningful; skipping.")
        return None

    if sub["name"].nunique() < 2:
        print(f"[WARN] Trait '{trait}': fewer than 2 genotypes. Skipping.")
        return None

    try:
        # Build design matrices manually so we can keep experiment fixed
        # exog: fixed effects (intercept + experiment)
        exog = pd.get_dummies(sub["expt"], drop_first=True)
        exog = sm.add_constant(exog)

        # groups: genotypes (random intercept)
        groups = sub["name"]

        md = sm.MixedLM(endog=sub[trait].values,
                        exog=exog.values,
                        groups=groups)
        res = md.fit(reml=reml, method="lbfgs", maxiter=200, disp=False)
        print(f"[INFO] Trait '{trait}': MET model converged.")
        return res
    except Exception as e:
        print(f"[ERROR] Trait '{trait}': MET model failed to converge. {e}")
        return None


def extract_overall_genotype_blups(df_ls, trait, res):
    """
    From fitted MET model result, compute overall genotype BLUPs:

        BLUP_j = mu + u_j

    where:
        mu: intercept (overall mean in baseline experiment)
        u_j: random intercept for genotype j.
    """
    mu = float(res.fe_params[0])  # intercept is first
    sub = df_ls[["expt", "name", trait]].dropna()
    genotypes = sorted(sub["name"].unique())

    blups = {}
    for g in genotypes:
        re = res.random_effects.get(g, 0.0)
        u_j = safe_random_effect_value(re)
        blups[g] = mu + u_j

    return blups


def extract_by_expt_predictions(df_ls, trait, res):
    """
    Using the same MET model, predict line means for each expt x line
    combination where the line is observed (G+E-style predictions).

    These reflect (mu + E_i + u_j). They are NOT a separate random GxE
    component, but are useful for looking at line performance patterns
    across experiments.
    """
    sub = df_ls[["expt", "name", trait]].dropna().copy()

    # Build fixed-effect design to compute predictions manualy:
    # exog: intercept + dummy(expt, drop_first=True)
    expt_levels = sorted(sub["expt"].unique())
    baseline = expt_levels[0]

    # Create dummy matrix aligned with the model fit
    dummy = pd.get_dummies(sub["expt"], drop_first=True)
    exog = sm.add_constant(dummy)  # col 0 = intercept

    # Fixed params
    beta = res.fe_params.values  # shape (p,)

    preds = []
    for i, row in sub.iterrows():
        g = row["name"]
        e = row["expt"]

        # fixed part
        x = exog.iloc[i].values  # aligned by index
        fixed_val = float(np.dot(x, beta))

        # random part
        re = res.random_effects.get(g, 0.0)
        u_j = safe_random_effect_value(re)

        y_hat = fixed_val + u_j
        preds.append((e, g, y_hat))

    # Build dataframe
    pred_df = pd.DataFrame(preds, columns=["expt", "name", trait])
    return pred_df


import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf

def met_blup_analysis(lsmeans_file: str,
                      prefix: str,
                      gxe: bool = False,
                      sep: str = "\t") -> None:
    """
    Two-stage MET analysis on LS-means.

    Stage 1 (already done upstream):
        RCBD within-experiment -> LS-means per (expt, line).

    Stage 2 (this function):
        Mixed model across experiments:
            y_ijk = mu + expt_i (fixed) + g_j (random) + e_ij

        Optionally, G×E can be added in the future (currently not implemented;
        gxe=True will simply emit a warning and use the main-effect model).

    Outputs:
        <prefix>_MET_overall_blups.tsv
        <prefix>_MET_by_expt_preds.tsv
        <prefix>_MET_varcomps.tsv
    """

    print(f"[INFO] Reading LS-means from: {lsmeans_file}")
    df_ls = pd.read_csv(lsmeans_file, sep=sep)

    # ------------------------------------------------------------------
    # 1) Detect trait columns vs meta columns using your existing helper
    # ------------------------------------------------------------------
    trait_cols, meta_cols = detect_trait_columns(df_ls)

    # Basic sanity: require 'expt' and 'name'
    required_meta = ["expt", "name"]
    for col in required_meta:
        if col not in df_ls.columns:
            raise ValueError(f"Required column '{col}' not found in LS-means file.")

    # ------------------------------------------------------------------
    # 2) Clean / sanitize
    # ------------------------------------------------------------------
    # Strip whitespace in all string columns
    df_ls = df_ls.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Force meta columns to string (avoids weird mixed dtypes)
    for col in meta_cols:
        df_ls[col] = df_ls[col].astype(str).fillna("")

    # Convert trait columns to numeric
    for col in trait_cols:
        df_ls[col] = pd.to_numeric(df_ls[col], errors="coerce")

    # Drop rows where *all* traits are NaN
    df_ls = df_ls.dropna(subset=trait_cols, how="all")

    print("[INFO] After cleaning, missing value count per trait:")
    print(df_ls[trait_cols].isna().sum())
    print(f"[INFO] Detected trait columns for MET: {trait_cols}")
    print(f"[INFO] Meta columns: {meta_cols}")

    if gxe:
        print("[WARN] G×E random effect is not yet implemented in this Python "
              "version. Proceeding with main-effect model: "
              "expt fixed, genotype random.")

    # ------------------------------------------------------------------
    # 3) Prepare a *minimal* modeling dataframe
    #    (only experiment, genotype, and traits)
    # ------------------------------------------------------------------
    safe_cols = ["expt", "name"] + trait_cols
    safe_df = df_ls[safe_cols].copy()
    safe_df = safe_df.rename(columns={"name": "genotype"})

    # Also keep one row of additional meta (designation, pedigree) per genotype
    meta_for_geno = None
    extra_meta = [c for c in meta_cols if c not in ["expt", "name"]]
    if extra_meta:
        meta_for_geno = (
            df_ls[["name"] + extra_meta]
            .drop_duplicates(subset=["name"])
            .rename(columns={"name": "genotype"})
            .set_index("genotype")
        )

    # Containers for outputs
    overall_rows = []   # genotype-level BLUPs
    byexpt_rows = []    # expt × genotype predictions
    var_rows = []       # variance components

    # ------------------------------------------------------------------
    # 4) Loop over traits and fit mixed model
    # ------------------------------------------------------------------
    for trait in trait_cols:
        print(f"[INFO] Processing trait: {trait}")

        # Build per-trait df and drop NAs
        df_trait = safe_df[["expt", "genotype", trait]].copy()
        df_trait = df_trait.dropna(subset=[trait])
        df_trait = df_trait.rename(columns={trait: "y"})

        if df_trait.empty:
            print(f"[WARN] Trait '{trait}': all values are missing, skipping.")
            continue

        # Check types before modeling
        #print("[DEBUG] dtypes for modeling:")
        #print(df_trait.dtypes)

        # Mixed model: expt fixed, genotype random
        # y ~ mu + C(expt) + u_genotype
        try:
            md = smf.mixedlm("y ~ C(expt)", df_trait,
                             groups=df_trait["genotype"])
            mdf = md.fit(method="lbfgs", reml=True, disp=False)
        except Exception as e:
            print(f"[ERROR] Trait '{trait}': MET model failed to converge. {e}")
            continue

        # ---------------------
        # 4a) Variance components
        # ---------------------
        try:
            sigma_g2 = float(mdf.cov_re.iloc[0, 0])
        except Exception:
            sigma_g2 = np.nan

        try:
            sigma_e2 = float(mdf.scale)
        except Exception:
            sigma_e2 = np.nan

        var_rows.append(
            {
                "trait": trait,
                "sigma_g2": sigma_g2,
                "sigma_e2": sigma_e2,
                "converged": bool(mdf.converged),
                "loglik": float(mdf.llf),
            }
        )

        # ---------------------
        # 4b) Overall BLUP per genotype
        # ---------------------
        for g, re in mdf.random_effects.items():
            # random_effects[g] is usually array-like (length 1)
            # safe BLUP extraction
            if isinstance(re, (list, tuple, np.ndarray)):
                g_blup = float(re[0])
            elif isinstance(re, pd.Series):
                g_blup = float(re.iloc[0])
            else:
                g_blup = float(re)


            overall_rows.append(
                {
                    "genotype": g,
                    "trait": trait,
                    "blup": g_blup,
                }
            )

        # ---------------------
        # 4c) Fitted values per expt × genotype
        # ---------------------
        df_trait = df_trait.copy()
        df_trait["pred"] = mdf.fittedvalues.values

        for _, row in df_trait.iterrows():
            byexpt_rows.append(
                {
                    "expt": row["expt"],
                    "genotype": row["genotype"],
                    "trait": trait,
                    "pred": float(row["pred"]),
                }
            )

    # ------------------------------------------------------------------
    # 5) Assemble and export results
    # ------------------------------------------------------------------
    # --- Overall BLUPs (wide) ---
    if overall_rows:
        df_overall = pd.DataFrame(overall_rows)
        wide_overall = (
            df_overall
            .pivot(index="genotype", columns="trait", values="blup")
            .reset_index()
        )

        # Attach extra meta info if available
        if meta_for_geno is not None:
            wide_overall = wide_overall.set_index("genotype")
            wide_overall = meta_for_geno.join(wide_overall, how="left")
            wide_overall = wide_overall.reset_index()

        out_overall = f"{prefix}_MET_overall_blups.tsv"
        wide_overall.to_csv(out_overall, sep="\t", index=False)
        print(f"[INFO] Saved overall BLUPs to: {out_overall}")
    else:
        print("[WARN] No overall BLUPs computed.")

    # --- By-experiment predictions (wide) ---
    if byexpt_rows:
        df_byexpt = pd.DataFrame(byexpt_rows)
        wide_byexpt = (
            df_byexpt
            .pivot(index=["expt", "genotype"], columns="trait", values="pred")
            .reset_index()
        )

        # Attach extra meta info if available
        if meta_for_geno is not None:
            wide_byexpt = wide_byexpt.set_index("genotype")
            wide_byexpt = meta_for_geno.join(wide_byexpt, how="right")
            # Now index has genotype; expt is a column inside
            wide_byexpt = wide_byexpt.reset_index()

        out_byexpt = f"{prefix}_MET_by_expt_preds.tsv"
        wide_byexpt.to_csv(out_byexpt, sep="\t", index=False)
        print(f"[INFO] Saved by-experiment predictions to: {out_byexpt}")
    else:
        print("[WARN] No by-experiment predictions produced.")

    # --- Variance components ---
    if var_rows:
        df_var = pd.DataFrame(var_rows)
        out_var = f"{prefix}_MET_varcomps.tsv"
        df_var.to_csv(out_var, sep="\t", index=False)
        print(f"[INFO] Saved variance components to: {out_var}")
    else:
        print("[WARN] No variance components saved.")

# ---------- CLI ----------

def main():
    parser = argparse.ArgumentParser(
        description="Two-stage MET BLUP analysis for RCBD LS-means.")
    parser.add_argument("--in", dest="infile", required=True,
                        help="Input LS-means file (TSV) from stage-1 RCBD analysis.")
    parser.add_argument("--sep", default="\t",
                        help="Field separator for input (default: tab).")
    parser.add_argument("--prefix", default=None,
                        help="Output prefix (default: from input filename).")
    parser.add_argument("--no-gxe", action="store_true",
                        help="Disable per-experiment predictions (GxE-style).")
    args = parser.parse_args()

    infile = args.infile
    if args.prefix:
        prefix = args.prefix
    else:
        base = os.path.basename(infile)
        prefix = os.path.splitext(base)[0]

    gxe = not args.no_gxe

    print(f"[INFO] Reading LS-means from: {infile}")
    met_blup_analysis(infile, prefix, gxe=gxe, sep=args.sep)


if __name__ == "__main__":
    main()

