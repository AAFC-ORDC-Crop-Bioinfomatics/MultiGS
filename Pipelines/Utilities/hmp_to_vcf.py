#!/usr/bin/env python3

import argparse
import sys
import pandas as pd
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert HapMap (.hmp) genotype file to VCF format"
    )
    parser.add_argument(
        "--hmp", required=True,
        help="Input HapMap file (.hmp or .hmp.txt)"
    )
    parser.add_argument(
        "--out", required=True,
        help="Output VCF file"
    )
    parser.add_argument(
        "--genome", default="unknown",
        help="Reference genome name (for VCF header)"
    )
    return parser.parse_args()


def genotype_to_gt(gt, ref, alt):
    """
    Convert HapMap genotype (e.g. AA, AG, NN) to VCF GT format.
    """
    if pd.isna(gt) or gt in ["NN", "N", "--"]:
        return "./."

    if len(gt) != 2:
        return "./."

    a1, a2 = gt[0], gt[1]

    if a1 == ref and a2 == ref:
        return "0/0"
    if a1 == alt and a2 == alt:
        return "1/1"
    if {a1, a2} == {ref, alt}:
        return "0/1"

    # Any other ambiguity
    return "./."


def main():
    args = parse_args()

    # Load HapMap (auto-detect delimiter)
    try:
        df = pd.read_csv(args.hmp, sep=None, engine="python")
    except Exception as e:
        sys.exit(f"ERROR: Failed to read HapMap file: {e}")

    required_cols = ["rs#", "chrom", "pos", "alleles"]
    for col in required_cols:
        if col not in df.columns:
            sys.exit(f"ERROR: HapMap file missing required column: {col}")

    sample_cols = [c for c in df.columns if c not in df.columns[:11]]

    if len(sample_cols) == 0:
        sys.exit("ERROR: No sample genotype columns detected.")

    # Write VCF header
    with open(args.out, "w") as vcf:
        vcf.write("##fileformat=VCFv4.2\n")
        vcf.write(f"##fileDate={datetime.now().strftime('%Y%m%d')}\n")
        vcf.write(f"##reference={args.genome}\n")
        vcf.write("##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">\n")

        header = [
            "#CHROM", "POS", "ID", "REF", "ALT",
            "QUAL", "FILTER", "INFO", "FORMAT"
        ] + sample_cols
        vcf.write("\t".join(header) + "\n")

        # Process each SNP
        for _, row in df.iterrows():
            chrom = row["chrom"]
            pos = row["pos"]
            snp_id = row["rs#"]

            alleles = row["alleles"].split("/")
            if len(alleles) != 2:
                continue  # skip non-biallelic

            ref, alt = alleles

            gt_values = []
            for s in sample_cols:
                gt = genotype_to_gt(row[s], ref, alt)
                gt_values.append(gt)

            vcf_row = [
                str(chrom),
                str(pos),
                snp_id,
                ref,
                alt,
                ".",
                "PASS",
                ".",
                "GT"
            ] + gt_values

            vcf.write("\t".join(vcf_row) + "\n")

    print(f"VCF written to: {args.out}")


if __name__ == "__main__":
    main()

