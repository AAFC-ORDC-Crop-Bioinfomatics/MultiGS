#!/usr/bin/env python3
import argparse
import warnings

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def detect_trait_columns(df: pd.DataFrame):
    """
    Detect which columns are traits vs meta columns.

    Assumptions (based on your LS-means file):
      - Meta columns: env (optional), expt, name, Original AAFC Designation, pedigree
      - Trait columns: all remaining numeric columns
    """
    meta_cols_base = ["expt", "name", "Original AAFC Designation", "pedigree"]
    meta_cols = meta_cols_base.copy()
    if "env" in df.columns:
        meta_cols = ["env"] + meta_cols

    # Keep only columns that exist in df
    meta_cols = [c for c in meta_cols if c in df.columns]

    candidate_traits = [c for c in df.columns if c not in meta_cols]

    trait_cols = []
    for c in candidate_traits:
        # Heuristic: treat as trait if it can be coerced to numeric
        try:
            s = pd.to_numeric(df[c], errors="coerce")
            if s.notna().sum() > 0:
                trait_cols.append(c)
        except Exception:
            continue

    return trait_cols, meta_cols


def clean_lsmeans(df_ls: pd.DataFrame, trait_cols):
    """
    Strip whitespace and safely coerce trait columns to numeric.
    Drop rows where all trait values are NaN.
    """

    # Strip whitespace from object columns (no applymap warning)
    for col in df_ls.columns:
        if df_ls[col].dtype == "object":
            df_ls[col] = df_ls[col].astype(str).str.strip()

    # Convert trait columns to numeric
    for col in trait_cols:
        df_ls[col] = pd.to_numeric(df_ls[col], errors="coerce")

    # Drop rows where all trait values are NaN
    df_ls = df_ls.dropna(subset=trait_cols, how="all")

    print("[INFO] After cleaning, missing value count per trait:")
    print(df_ls[trait_cols].isna().sum())

    return df_ls


def met_blup_analysis(lsmeans_file, prefix, sep="\t"):
    print(f"[INFO] Reading LS-means from: {lsmeans_file}")
    df_ls = pd.read_csv(lsmeans_file, sep=sep)

    # Detect trait/meta columns
    trait_cols, meta_cols = detect_trait_columns(df_ls)
    df_ls = clean_lsmeans(df_ls, trait_cols)

    print(f"[INFO] Detected trait columns for MET: {trait_cols}")
    print(f"[INFO] Meta columns: {meta_cols}")

    has_env = "env" in df_ls.columns

    overall_blups_records = []
    by_expt_records = []
    varcomps_records = []

    for trait in trait_cols:
        print(f"[INFO] Processing trait: {trait}")

        # Build modeling DataFrame
        df_trait = pd.DataFrame({
            "expt": df_ls["expt"].astype(str).str.strip(),
            "genotype": df_ls["name"].astype(str).str.strip(),
            "y": pd.to_numeric(df_ls[trait], errors="coerce"),
        })

        if has_env:
            df_trait["env"] = df_ls["env"].astype(str).str.strip()

        # Drop missing y
        df_trait = df_trait.dropna(subset=["y"])

        if df_trait.empty:
            print(f"[WARN] Trait '{trait}': no non-missing observations, skipping.")
            continue

        print("[DEBUG] dtypes for modeling:")
        print(df_trait[["expt", "genotype", "y"] + (["env"] if has_env else [])].dtypes)

        try:
            # Mixed model:
            #   y ~ C(expt)
            #   random intercept for genotype
            #   optional random component for env via vc_formula
            if has_env:
                md = smf.mixedlm(
                    "y ~ C(expt)",
                    df_trait,
                    groups=df_trait["genotype"],
                    vc_formula={"env": "0 + C(env)"},
                )
            else:
                md = smf.mixedlm(
                    "y ~ C(expt)",
                    df_trait,
                    groups=df_trait["genotype"],
                )

            result = md.fit(method="lbfgs", reml=False, maxiter=200, disp=False)

        except Exception as e:
            print(
                f"[ERROR] Trait '{trait}': MET model failed to converge. {e}"
            )
            continue

        # ---------------------------
        # Overall BLUPs per genotype
        # ---------------------------
        for g in sorted(df_trait["genotype"].unique()):
            re = result.random_effects.get(g, None)
            if re is None or len(re) == 0:
                g_blup = 0.0
            else:
                # re can be a Series or array-like; use iloc if Series to avoid FutureWarning
                try:
                    if hasattr(re, "iloc"):
                        g_blup = float(re.iloc[0])
                    else:
                        g_blup = float(np.asarray(re)[0])
                except Exception:
                    g_blup = float(re[0])

            overall_blups_records.append(
                {
                    "genotype": g,
                    "trait": trait,
                    "blup": g_blup,
                }
            )

        # --------------------------------
        # By-experiment (and env) predictions
        # --------------------------------
        df_trait_out = df_trait.copy()
        df_trait_out["trait"] = trait
        df_trait_out["pred"] = result.fittedvalues

        cols = []
        if has_env:
            cols.append("env")
        cols += ["expt", "genotype", "trait", "y", "pred"]

        by_expt_records.extend(df_trait_out[cols].to_dict(orient="records"))

        # -------------------------
        # Variance components
        # -------------------------
        var_g = float(result.cov_re.iloc[0,  0])
        var_resid = float(result.scale)

        # NOTE: env variance is in result.vcomp when vc_formula is used,
        # but mapping can be subtle. For now, we only record genotype and residual.
        varcomps_records.append(
            {
                "trait": trait,
                "var_genotype": var_g,
                "var_residual": var_resid,
            }
        )

    # -------------------------
    # Save outputs
    # -------------------------
    if overall_blups_records:
        df_blup = pd.DataFrame(overall_blups_records)
        out_overall = f"{prefix}_MET_overall_blups.tsv"
        df_blup.to_csv(out_overall, sep="\t", index=False)
        print(f"[INFO] Saved overall BLUPs to: {out_overall}")
    else:
        print("[WARN] No overall BLUPs computed.")

    if by_expt_records:
        df_by_expt = pd.DataFrame(by_expt_records)
        out_by_expt = f"{prefix}_MET_by_expt_preds.tsv"
        df_by_expt.to_csv(out_by_expt, sep="\t", index=False)
        print(f"[INFO] Saved by-experiment predictions to: {out_by_expt}")
    else:
        print("[WARN] No by-experiment predictions produced.")

    if varcomps_records:
        df_var = pd.DataFrame(varcomps_records)
        out_var = f"{prefix}_MET_varcomps.tsv"
        df_var.to_csv(out_var, sep="\t", index=False)
        print(f"[INFO] Saved variance components to: {out_var}")
    else:
        print("[WARN] No variance components saved.")


def main():
    parser = argparse.ArgumentParser(
        description="Two-stage MET BLUP analysis (supports optional env random effect)."
    )
    parser.add_argument(
        "--in", dest="infile", required=True, help="Input LS-means file (TSV/CSV)."
    )
    parser.add_argument(
        "--prefix", required=True, help="Prefix for output files."
    )
    parser.add_argument(
        "--sep",
        default="\t",
        help="Field separator in LS-means file (default: tab).",
    )
    # Keep gxe flag if you were using it, but it's not needed for this version
    parser.add_argument(
        "--gxe",
        action="store_true",
        help="(Reserved) Not used in this implementation.",
    )
    args = parser.parse_args()

    met_blup_analysis(args.infile, args.prefix, sep=args.sep)


if __name__ == "__main__":
    main()

