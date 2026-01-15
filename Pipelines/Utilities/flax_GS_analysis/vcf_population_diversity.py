import os
import allel
import numpy as np
import pandas as pd
from itertools import combinations

# --------------------------------------------------
# Load VCF and extract SNP metadata + genotypes
# --------------------------------------------------
def load_vcf(vcf_file):
    callset = allel.read_vcf(
        vcf_file,
        fields=['variants/CHROM', 'variants/POS',
                'variants/REF', 'variants/ALT',
                'calldata/GT'],
        alt_number=1
    )

    gt = allel.GenotypeArray(callset['calldata/GT'])

    snp_df = pd.DataFrame({
        "CHROM": callset['variants/CHROM'].astype(str),
        "POS": callset['variants/POS'],
        "REF": callset['variants/REF'].astype(str),
        "ALT": [a[0] for a in callset['variants/ALT']]
    })

    snp_df["SNP_ID"] = (
        snp_df["CHROM"] + ":" +
        snp_df["POS"].astype(str) + ":" +
        snp_df["REF"] + ":" +
        snp_df["ALT"]
    )

    return snp_df, gt


# --------------------------------------------------
# Harmonize SNPs across all populations
# --------------------------------------------------
def harmonize_snps(pop_data):
    """
    pop_data: dict {pop_name: (snp_df, gt)}
    returns: dict {pop_name: harmonized_gt}
    """
    # find common SNP_IDs
    common_snps = set.intersection(
        *[set(df["SNP_ID"]) for df, _ in pop_data.values()]
    )

    if len(common_snps) == 0:
        raise RuntimeError("No common SNPs found across VCF files.")

    harmonized = {}

    for pop, (df, gt) in pop_data.items():
        idx = df["SNP_ID"].isin(common_snps).values
        harmonized[pop] = gt.compress(idx, axis=0)

    print(f"Harmonized SNP count: {len(common_snps)}")
    return harmonized


# --------------------------------------------------
# Within-population diversity
# --------------------------------------------------

def within_population_metrics(gt):
    ac = gt.count_alleles()

    # ---- nucleotide diversity (pi), version-safe ----
    try:
        pi = allel.sequence_diversity(ac)
    except TypeError:
        pi = allel.sequence_diversity(np.arange(ac.shape[0]), ac)

    # ---- expected heterozygosity (He), robust ----
    af = ac.to_frequencies()
    valid = np.isfinite(af).all(axis=1) & (ac.sum(axis=1) > 0)
    he = (1.0 - np.sum(af[valid] ** 2, axis=1)).mean()

    # ---- observed heterozygosity (Ho) ----
    ho = gt.count_het(axis=1).mean() / gt.n_samples

    return pi, he, ho


# --------------------------------------------------
# Pairwise FST
# --------------------------------------------------
def pairwise_fst(gt1, gt2):
    """
    Pairwise Weir & Cockerham FST.
    Fully compatible with old and new scikit-allel APIs.
    """
    # concatenate genotypes
    gt_all = allel.GenotypeArray(
        np.concatenate([gt1.values, gt2.values], axis=1)
    )

    n1 = gt1.n_samples
    n2 = gt2.n_samples

    subpops = [
        list(range(0, n1)),
        list(range(n1, n1 + n2))
    ]

    res = allel.weir_cockerham_fst(gt_all, subpops)

    # ---- handle API differences ----
    if len(res) == 2:
        num, den = res
    elif len(res) == 3:
        a, b, c = res
        num = a
        den = b + c
    else:
        raise RuntimeError("Unexpected return format from weir_cockerham_fst()")

    fst = np.sum(num) / np.sum(den)
    return fst



# --------------------------------------------------
# Main
# --------------------------------------------------
def main(vcf_dir, out_prefix):
    # Load all VCFs
    pop_data = {}

    for fname in os.listdir(vcf_dir):
        if fname.endswith(".vcf") or fname.endswith(".vcf.gz"):
            pop = os.path.splitext(os.path.basename(fname))[0]
            snp_df, gt = load_vcf(os.path.join(vcf_dir, fname))
            pop_data[pop] = (snp_df, gt)

    if len(pop_data) < 2:
        raise RuntimeError("At least two VCF files are required.")

    # Harmonize SNPs
    genotypes = harmonize_snps(pop_data)

    # ----------------------------
    # Within-population results
    # ----------------------------
    within_rows = []
    for pop, gt in genotypes.items():
        pi, he, ho = within_population_metrics(gt)
        within_rows.append({
            "Population": pop,
            "Nucleotide_diversity_pi": pi,
            "Expected_heterozygosity_He": he,
            "Observed_heterozygosity_Ho": ho,
            "Num_SNPs": gt.shape[0],
            "Num_Individuals": gt.n_samples
        })

    within_df = pd.DataFrame(within_rows)
    within_df.to_csv(f"{out_prefix}_within_population.csv", index=False)

    # ----------------------------
    # Between-population FST
    # ----------------------------
    pops = list(genotypes.keys())
    fst_matrix = pd.DataFrame(
        np.zeros((len(pops), len(pops))),
        index=pops,
        columns=pops
    )

    for p1, p2 in combinations(pops, 2):
        fst = pairwise_fst(genotypes[p1], genotypes[p2])
        fst_matrix.loc[p1, p2] = fst
        fst_matrix.loc[p2, p1] = fst

    fst_matrix.to_csv(f"{out_prefix}_between_population_FST.csv")

    print("Done.")
    print(f"  {out_prefix}_within_population.csv")
    print(f"  {out_prefix}_between_population_FST.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Population genetic diversity from multiple VCF files (with SNP harmonization)"
    )
    parser.add_argument("--vcf_dir", required=True,
                        help="Folder containing VCF files (one per population)")
    parser.add_argument("--out_prefix", default="pop_diversity",
                        help="Output prefix")

    args = parser.parse_args()
    main(args.vcf_dir, args.out_prefix)

