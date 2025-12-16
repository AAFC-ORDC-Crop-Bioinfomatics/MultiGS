#!/usr/bin/env python3
"""
Intersect two SNP VCFs and harmonize allele representation.

- Keeps sites present in BOTH files by (CHROM, POS).
- Keeps only SNPs (REF length 1 and all ALT alleles length 1).
- Uses the allele representation (REF + ALT order) from VCF1 as canonical.
- Rewrites VCF2 at common sites to match VCF1's allele representation.
  * Handles REF/ALT swaps and strand flips (complements).
  * Remaps sample GT indices in VCF2 accordingly.
- VCF1 is written as-is at common sites (no change to GT/alleles).
- Preserves header lines (#...) from each input for its own output.
- Maintains record order within each input.

Usage:
  python vcf_common_snps_harmonized.py file1.vcf[.gz] file2.vcf[.gz] \
      [--out1 file1.common.harmonized.vcf.gz] \
      [--out2 file2.common.harmonized.vcf.gz] \
      [--normalize-chr] [--biallelic-only]

Options:
  --normalize-chr    Strip leading 'chr'/'Chr' when comparing chromosome names.
  --biallelic-only   Keep only biallelic SNPs (drop multi-allelic SNPs).
"""

import argparse
import gzip
import re
import sys
from typing import Dict, List, Tuple, Optional

def open_maybe_gzip(path: str, mode: str):
    if path.endswith(".gz"):
        return gzip.open(path, mode)  # text mode: 'rt'/'wt' are fine here
    return open(path, mode, encoding="utf-8")

def normalize_chrom(chrom: str, enable: bool) -> str:
    if not enable:
        return chrom
    return re.sub(r'^(?i:chr)', '', chrom)

def is_snp(fields: List[str], biallelic_only: bool) -> bool:
    if len(fields) < 5:
        return False
    ref = fields[3]
    alt = fields[4]
    if ref == "." or alt == ".":
        return False
    # SNP check
    if len(ref) != 1:
        return False
    alts = alt.split(",")
    if any(len(a) != 1 for a in alts):
        return False
    if biallelic_only and len(alts) != 1:
        return False
    return True

def parse_alleles(fields: List[str]) -> List[str]:
    """Return [REF, ALT1, ALT2, ...]."""
    return [fields[3]] + fields[4].split(",")

COMPLEMENT = str.maketrans("ACGTacgt", "TGCAtgca")

def complement_alleles(alleles: List[str]) -> List[str]:
    return [a.translate(COMPLEMENT) for a in alleles]

def build_index_map(src: List[str], tgt: List[str]) -> Optional[Dict[int, int]]:
    """
    Map allele indices from src to tgt.
    src/tgt are lists like [REF, ALT1, ALT2, ...].
    Return dict {src_index -> tgt_index} if all src alleles exist in tgt; else None.
    """
    idx_tgt = {a: i for i, a in enumerate(tgt)}
    index_map: Dict[int, int] = {}
    for i, a in enumerate(src):
        if a not in idx_tgt:
            return None
        index_map[i] = idx_tgt[a]
    return index_map

def find_harmonization_map(alleles2: List[str], alleles1: List[str]) -> Tuple[Optional[Dict[int,int]], bool]:
    """
    Try to map VCF2 alleles to VCF1 alleles (VCF1 is canonical).
    Attempts without complement first, then with complement.
    Returns (index_map, used_complement_flag).
    """
    # Try direct
    m = build_index_map(alleles2, alleles1)
    if m is not None:
        return m, False
    # Try complemented
    alleles2c = complement_alleles(alleles2)
    m = build_index_map(alleles2c, alleles1)
    if m is not None:
        return m, True
    return None, False

def remap_gt_token(gt_tok: str, index_map: Dict[int,int]) -> str:
    """
    Remap a single GT token like '0/1', '1|2', './.' according to index_map.
    Keeps separators '/' or '|'. Missing alleles '.' remain '.'.
    """
    if gt_tok == "." or gt_tok == "./." or gt_tok == ".|.":
        return gt_tok
    sep = "/" if "/" in gt_tok else ("|" if "|" in gt_tok else None)
    if sep is None:
        parts = [gt_tok]
    else:
        parts = gt_tok.split(sep)
    new_parts: List[str] = []
    for p in parts:
        if p == "." or p == "":
            new_parts.append(p if p != "" else ".")
            continue
        try:
            i = int(p)
        except ValueError:
            # Unexpected token; keep as-is
            new_parts.append(p)
            continue
        if i not in index_map:
            # Allele index not found (e.g., allele pruned) -> set missing
            new_parts.append(".")
        else:
            new_parts.append(str(index_map[i]))
    return sep.join(new_parts) if sep else new_parts[0]

def remap_samples_gt(format_field: str, sample_fields: List[str], index_map: Dict[int,int]) -> List[str]:
    """
    Remap GT across all samples given FORMAT and per-sample columns.
    Only GT is adjusted; other fields remain untouched (order preserved).
    """
    if not sample_fields:
        return sample_fields
    fmt_keys = format_field.split(":")
    try:
        gt_idx = fmt_keys.index("GT")
    except ValueError:
        # No GT present
        return sample_fields
    new_samples = []
    for s in sample_fields:
        parts = s.split(":")
        # pad if malformed
        if len(parts) < len(fmt_keys):
            parts += [""] * (len(fmt_keys) - len(parts))
        parts[gt_idx] = remap_gt_token(parts[gt_idx], index_map)
        new_samples.append(":".join(parts))
    return new_samples

def default_out_name(path: str) -> str:
    if path.endswith(".vcf.gz"):
        return path[:-7] + ".common.harmonized.vcf.gz"
    elif path.endswith(".vcf"):
        return path[:-4] + ".common.harmonized.vcf"
    else:
        return path + ".common.harmonized.vcf"

def pass1_collect_sites_with_alleles(vcf_path: str, normalize_chr: bool, biallelic_only: bool) -> Dict[Tuple[str,int], List[str]]:
    """
    Return dict: (chrom,pos) -> alleles list [REF, ALT...]
    SNPs only (and biallelic only if requested).
    """
    d: Dict[Tuple[str,int], List[str]] = {}
    with open_maybe_gzip(vcf_path, "rt") as f:
        for line in f:
            if not line or line[0] == "#":
                continue
            fields = line.rstrip("\n").split("\t")
            if not is_snp(fields, biallelic_only):
                continue
            chrom = normalize_chrom(fields[0], normalize_chr)
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            d[(chrom, pos)] = parse_alleles(fields)
    return d

def write_filtered_vcf1(vcf_in: str, vcf_out: str, keep_sites: set, normalize_chr: bool, biallelic_only: bool) -> int:
    """
    Write VCF1 records for the harmonizable common SNP sites.
    No allele/GT changes are made for VCF1.
    """
    n = 0
    with open_maybe_gzip(vcf_in, "rt") as fin, open_maybe_gzip(vcf_out, "wt") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue
            fields = line.rstrip("\n").split("\t")
            if not is_snp(fields, biallelic_only):
                continue
            chrom = normalize_chrom(fields[0], normalize_chr)
            try:
                pos = int(fields[1])
            except ValueError:
                continue
            if (chrom, pos) in keep_sites:
                fout.write(line)
                n += 1
    return n

def write_filtered_vcf2_harmonized(
        vcf_in: str,
        vcf_out: str,
        canonical_alleles: Dict[Tuple[str,int], List[str]],
        index_maps: Dict[Tuple[str,int], Dict[int,int]],
        normalize_chr: bool,
        biallelic_only: bool
    ) -> int:

    """
    Write VCF2 records for the harmonizable common sites.
    Only sites present in index_maps are written.
    """

    n = 0

    with open_maybe_gzip(vcf_in, "rt") as fin, open_maybe_gzip(vcf_out, "wt") as fout:
        for line in fin:
            if line.startswith("#"):
                fout.write(line)
                continue

            fields = line.rstrip("\n").split("\t")
            if not is_snp(fields, biallelic_only):
                continue

            chrom = normalize_chrom(fields[0], normalize_chr)

            try:
                pos = int(fields[1])
            except ValueError:
                continue

            key = (chrom, pos)

            # 🔥 CRITICAL FIX: Only process sites that have an allele mapping
            if key not in index_maps:
                continue

            # Canonical alleles come from VCF1
            alleles1 = canonical_alleles[key]

            # Rewrite REF and ALT
            fields[3] = alleles1[0]
            fields[4] = ",".join(alleles1[1:]) if len(alleles1) > 1 else "."

            # Remap GT fields if samples present
            if len(fields) >= 10:
                fmt = fields[8]
                samples = fields[9:]

                imap = index_maps[key]   # ❗ SAFE: key is guaranteed present
                mapped_samples = remap_samples_gt(fmt, samples, imap)
                fields = fields[:9] + mapped_samples

            fout.write("\t".join(fields) + "\n")
            n += 1

    return n

def main():
    ap = argparse.ArgumentParser(description="Intersect SNPs and harmonize alleles between two VCFs.")
    ap.add_argument("vcf1", help="First VCF (.vcf or .vcf.gz) — canonical allele representation")
    ap.add_argument("vcf2", help="Second VCF (.vcf or .vcf.gz) — will be harmonized to VCF1 at common sites")
    ap.add_argument("--out1", help="Output VCF for filtered first file")
    ap.add_argument("--out2", help="Output VCF for filtered & harmonized second file")
    ap.add_argument("--normalize-chr", action="store_true",
                    help="Normalize chromosome names by removing leading 'chr'/'Chr' before comparison")
    ap.add_argument("--biallelic-only", action="store_true",
                    help="Restrict to biallelic SNPs only")
    args = ap.parse_args()

    out1 = args.out1 or default_out_name(args.vcf1)
    out2 = args.out2 or default_out_name(args.vcf2)

    # Pass 1: collect sites with alleles from both VCFs
    sites1 = pass1_collect_sites_with_alleles(args.vcf1, args.normalize_chr, args.biallelic_only)
    sites2 = pass1_collect_sites_with_alleles(args.vcf2, args.normalize_chr, args.biallelic_only)

    common_keys = set(sites1.keys()) & set(sites2.keys())

    # Determine which common sites can be harmonized, and prepare index maps for VCF2 GT remap
    harmonizable_keys = set()
    index_maps: Dict[Tuple[str,int], Dict[int,int]] = {}
    n_need_complement = 0
    n_common = len(common_keys)

    for key in common_keys:
        alleles1 = sites1[key]
        alleles2 = sites2[key]
        imap, used_comp = find_harmonization_map(alleles2, alleles1)
        if imap is not None:
            harmonizable_keys.add(key)
            index_maps[key] = imap
            if used_comp:
                n_need_complement += 1

    # Write outputs
    n1 = write_filtered_vcf1(args.vcf1, out1, harmonizable_keys, args.normalize_chr, args.biallelic_only)
    n2 = write_filtered_vcf2_harmonized(args.vcf2, out2, sites1, index_maps, args.normalize_chr, args.biallelic_only)

    print(f"[Summary] VCF1 SNP sites:                  {len(sites1)}", file=sys.stderr)
    print(f"[Summary] VCF2 SNP sites:                  {len(sites2)}", file=sys.stderr)
    print(f"[Summary] Common SNP sites (by CHR,POS):   {n_common}", file=sys.stderr)
    print(f"[Summary] Harmonizable common SNP sites:   {len(harmonizable_keys)}", file=sys.stderr)
    print(f"[Summary] Of which needed strand flip:     {n_need_complement}", file=sys.stderr)
    print(f"[Output]  {out1}: {n1} records written", file=sys.stderr)
    print(f"[Output]  {out2}: {n2} records written", file=sys.stderr)

if __name__ == "__main__":
    main()
