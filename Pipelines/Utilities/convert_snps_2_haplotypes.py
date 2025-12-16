#!/usr/bin/env python3
"""
convert_snps_2_haplotypes.py
- One VCF: call RTM-GWAS-SNPLDB to generate haplotype VCF.
- Two VCFs: harmonize by common SNPs (CHR+POS), combine samples, run RTM tool, then split hap-VCF back to train/test.

Usage:
  python convert_snps_2_haplotypes --vcf TRAIN.vcf[.gz] [--test_vcf TEST.vcf[.gz]] [--outdir results] [--prefix merged]

Dependencies:
  - scikit-allel (for reading VCFs)
  - external tool `rtm-gwas-snpldb` in PATH
"""

import os
import sys
import argparse
import subprocess
import gzip
import shutil
from typing import Tuple, List, Dict, Any

import numpy as np
import allel


# ------------------------------- Utilities -------------------------------

def _norm_chr(c: str) -> str:
    s = str(c)
    if s.lower().startswith('chr'):
        s = s[3:]
    return s

def _is_ambiguous_pair(ref: str, alt: str) -> bool:
    a, b = ref.upper(), alt.upper()
    return {a, b} in ({"A", "T"}, {"C", "G"})

def _flip_gt(gt: np.ndarray) -> np.ndarray:
    """Flip 0<->1 in a GT call; preserves -1 as missing."""
    out = gt.copy()
    mask = (out >= 0)
    out[mask] = 1 - out[mask]
    return out

def _write_vcf(path: str,
               chroms: np.ndarray,
               poss: np.ndarray,
               refs: np.ndarray,
               alts: np.ndarray,
               gt: np.ndarray,             # shape (n_var, n_samples, ploidy)
               samples: List[str]) -> None:
    """Write a simple VCF (uncompressed) with only GT format."""
    header = [
        "##fileformat=VCFv4.2",
        "##FORMAT=<ID=GT,Number=1,Type=String,Description=\"Genotype\">",
        "#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\t" + "\t".join(samples)
    ]
    with open(path, "w") as f:
        f.write("\n".join(header) + "\n")
        n_var = chroms.shape[0]
        for i in range(n_var):
            chrom = str(chroms[i])
            pos = int(poss[i])
            ref = str(refs[i])
            alt = str(alts[i])
            # Convert GT calls to strings
            gti = gt[i]  # (n_samples, ploidy)
            calls = []
            for s in range(gti.shape[0]):
                a, b = gti[s, 0], gti[s, 1]
                if a < 0 or b < 0:
                    calls.append("./.")
                else:
                    calls.append(f"{a}/{b}")
            row = [chrom, str(pos), ".", ref, alt, ".", "PASS", ".", "GT"] + calls
            f.write("\t".join(row) + "\n")


def _open_text(path: str):
    return gzip.open(path, "rt") if path.endswith(".gz") else open(path, "r")


def _write_text(path: str):
    # write plain .vcf (not gz); easy for downstream tool
    return open(path, "w")


# ---------------------------- VCF I/O & Merge ----------------------------

def _read_biallelic_snp_vcf(path: str) -> Dict[str, Any]:
    """Read a VCF and return only biallelic SNP sites + GT array."""
    fields = ['samples', 'variants/CHROM', 'variants/POS', 'variants/REF', 'variants/ALT', 'calldata/GT']
    d = allel.read_vcf(path, fields=fields, alt_number=1)  # alt_number=1 keeps only first ALT
    if d is None or 'samples' not in d:
        raise RuntimeError(f"Failed to read VCF or no samples: {path}")

    samples = list(d['samples'])
    chrom = d['variants/CHROM']
    pos = d['variants/POS']
    ref = d['variants/REF']
    alt = d['variants/ALT']
    gt = d['calldata/GT']  # shape (n_var, n_samples, 2)

    # Keep only biallelic SNP: single-character REF and ALT
    is_snp = (np.char.str_len(ref.astype(str)) == 1) & (np.char.str_len(alt.astype(str)) == 1)
    chrom = chrom[is_snp]
    pos = pos[is_snp]
    ref = ref[is_snp]
    alt = alt[is_snp]
    gt = gt[is_snp, :, :]

    return dict(samples=samples, chrom=chrom, pos=pos, ref=ref, alt=alt, gt=gt)


def _harmonize_and_combine(train: Dict[str, Any],
                           test: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """
    Harmonize allele orientation by common CHR+POS.
    Drop ambiguous A/T or C/G sites when flipping is required.
    Return arrays: chrom, pos, ref, alt, GT_combined, samples_combined
    """
    # keys
    tr_keys = [(_norm_chr(c), int(p)) for c, p in zip(train['chrom'], train['pos'])]
    te_keys = [(_norm_chr(c), int(p)) for c, p in zip(test['chrom'], test['pos'])]
    tr_index = {k: i for i, k in enumerate(tr_keys)}
    te_index = {k: i for i, k in enumerate(te_keys)}
    common = [k for k in tr_index.keys() if k in te_index]

    if not common:
        raise RuntimeError("No common SNPs by (CHR,POS) between the two VCFs.")

    tr_keep = []
    te_keep = []
    flips = []
    dropped = 0

    for k in common:
        i = tr_index[k]
        j = te_index[k]
        r1, a1 = str(train['ref'][i]).upper(), str(train['alt'][i]).upper()
        r2, a2 = str(test['ref'][j]).upper(), str(test['alt'][j]).upper()
        if r1 == r2 and a1 == a2:
            tr_keep.append(i); te_keep.append(j); flips.append(False)
        elif r1 == a2 and a1 == r2:
            # needs flipping on test; but skip ambiguous
            if _is_ambiguous_pair(r1, a1):
                dropped += 1
                continue
            tr_keep.append(i); te_keep.append(j); flips.append(True)
        else:
            # different ALT allele (e.g., true multi-allelic mismatch) -> drop
            dropped += 1

    if len(tr_keep) == 0:
        raise RuntimeError("All common SNPs were filtered out during harmonization.")

    tr_keep = np.array(tr_keep, dtype=int)
    te_keep = np.array(te_keep, dtype=int)
    flips = np.array(flips, dtype=bool)

    # Slice
    chrom = train['chrom'][tr_keep]
    pos = train['pos'][tr_keep]
    ref = train['ref'][tr_keep]
    alt = train['alt'][tr_keep]

    gt_tr = train['gt'][tr_keep, :, :]                     # (n_var, n_train, 2)
    gt_te = test['gt'][te_keep, :, :]                      # (n_var, n_test, 2)
    # flip test calls where needed
    if flips.any():
        # vectorized flip: for each variant v that needs flipping, apply transformation
        vmask = flips
        gt_te_v = gt_te[vmask, :, :]
        gt_te[vmask, :, 0] = _flip_gt(gt_te_v[:, :, 0])
        gt_te[vmask, :, 1] = _flip_gt(gt_te_v[:, :, 1])

    samples_combined = list(train['samples']) + list(test['samples'])
    gt_combined = np.concatenate([gt_tr, gt_te], axis=1)   # (n_var, n_train+n_test, 2)

    if dropped > 0:
        print(f"[Harmonize] Note: dropped {dropped} sites (ambiguous flips or allele mismatch). Kept={len(tr_keep)}")

    return chrom, pos, ref, alt, gt_combined, samples_combined


# ----------------------------- External calls ----------------------------

def find_haplotypes(vcf_path: str, out_prefix: str) -> str:
    """
    Call external RTM-GWAS SNPLDB tool:
      rtm-gwas-snpldb --vcf <vcf_path> --out <out_prefix>
    Returns path to the produced haplotype VCF (assumed '<out_prefix>.vcf' or '<out_prefix>_hap.vcf').
    """
    cmd = ["/isilon/projects/J-002035_flaxgenomics/J-001386Frank_Lab/Flax_GS_project_J-003426/GSPipeline_single_trait_models/rtm-gwas/bin/rtm-gwas-snpldb", "--vcf", vcf_path, "--out", out_prefix]
    print("[RUN]", " ".join(cmd))
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    print(p.stdout)
    if p.returncode != 0:
        raise RuntimeError(f"rtm-gwas-snpldb failed with code {p.returncode}")

    # Tool may emit <prefix>.vcf or <prefix>_hap.vcf; check both
    cand1 = out_prefix + ".vcf"
    cand2 = out_prefix + "_hap.vcf"
    if os.path.exists(cand1):
        return cand1
    if os.path.exists(cand2):
        return cand2
    # also try .vcf.gz variants
    if os.path.exists(cand1 + ".gz"):
        return cand1 + ".gz"
    if os.path.exists(cand2 + ".gz"):
        return cand2 + ".gz"
    raise RuntimeError(f"Cannot find haplotype VCF output from prefix '{out_prefix}'.")


def split_vcf_by_samples(src_vcf: str,
                         train_samples: List[str],
                         test_samples: List[str],
                         out_train_vcf: str,
                         out_test_vcf: str) -> None:
    """Split a (haplotype) VCF into two by sample lists (header-driven column selection)."""
    with _open_text(src_vcf) as inp, _write_text(out_train_vcf) as ftr, _write_text(out_test_vcf) as fte:
        for line in inp:
            if line.startswith("##"):
                ftr.write(line); fte.write(line)
            elif line.startswith("#CHROM"):
                header_cols = line.strip().split("\t")
                fixed = header_cols[:9]
                samples = header_cols[9:]
                idx_tr = [9 + samples.index(s) for s in train_samples if s in samples]
                idx_te = [9 + samples.index(s) for s in test_samples if s in samples]

                ftr.write("\t".join(fixed + train_samples) + "\n")
                fte.write("\t".join(fixed + test_samples) + "\n")
            else:
                parts = line.rstrip("\n").split("\t")
                fixed = parts[:9]
                tr_calls = [parts[i] for i in idx_tr]
                te_calls = [parts[i] for i in idx_te]
                ftr.write("\t".join(fixed + tr_calls) + "\n")
                fte.write("\t".join(fixed + te_calls) + "\n")


# --------------------------------- Main ----------------------------------

def main():
    ap = argparse.ArgumentParser(description="Build haplotype VCFs from one or two SNP VCFs (with harmonization).")
    ap.add_argument("--vcf", required=True, help="Training (or single) VCF path")
    ap.add_argument("--test_vcf", default=None, help="Optional test VCF path")
    ap.add_argument("--outdir", default="results_hap", help="Output directory")
    ap.add_argument("--prefix", default="combined", help="Base prefix for combined outputs (when two VCFs)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.test_vcf is None:
        # Single VCF → just call RTM tool
        vcf_in = args.vcf
        out_prefix = os.path.join(args.outdir, os.path.splitext(os.path.basename(vcf_in))[0] + "_hap")
        hap_vcf = find_haplotypes(vcf_in, out_prefix)
        print(f"[DONE] Haplotype VCF: {hap_vcf}")
        return

    # Two VCFs → harmonize, combine, RTM, split
    print("[STEP] Reading training VCF (biallelic SNPs)...")
    tr = _read_biallelic_snp_vcf(args.vcf)
    print(f"  train: variants={tr['gt'].shape[0]}, samples={len(tr['samples'])}")

    print("[STEP] Reading test VCF (biallelic SNPs)...")
    te = _read_biallelic_snp_vcf(args.test_vcf)
    print(f"  test : variants={te['gt'].shape[0]}, samples={len(te['samples'])}")

    print("[STEP] Harmonizing by common (CHR,POS) and combining samples...")
    chrom, pos, ref, alt, gt_comb, samples_comb = _harmonize_and_combine(tr, te)
    print(f"  kept variants={gt_comb.shape[0]}, combined samples={gt_comb.shape[1]}")

    combined_vcf = os.path.join(args.outdir, f"{args.prefix}_common_harmonized.vcf")
    print(f"[STEP] Writing combined VCF → {combined_vcf}")
    _write_vcf(combined_vcf, chrom, pos, ref, alt, gt_comb, samples_comb)

    print("[STEP] Calling RTM-GWAS SNPLDB on combined VCF...")
    hap_out_prefix = os.path.join(args.outdir, f"{args.prefix}_hap")
    combined_hap_vcf = find_haplotypes(combined_vcf, hap_out_prefix)
    print(f"  produced: {combined_hap_vcf}")

    print("[STEP] Splitting haplotype VCF back into train/test by original sample IDs...")
    out_train = os.path.join(args.outdir, f"{args.prefix}_train_hap.vcf")
    out_test = os.path.join(args.outdir, f"{args.prefix}_test_hap.vcf")
    split_vcf_by_samples(
        combined_hap_vcf,
        train_samples=tr['samples'],
        test_samples=te['samples'],
        out_train_vcf=out_train,
        out_test_vcf=out_test
    )

    print(f"[DONE] Train hap VCF: {out_train}")
    print(f"[DONE] Test  hap VCF: {out_test}")


if __name__ == "__main__":
    main()

