#!/usr/bin/env python3
"""
dpcformer_pipeline_v3.py

Wrapper pipeline around the DPCFormer / PhenotypePredictor_1v model for
genomic prediction using train/test VCF + phenotype files, in a style
similar to wheatGP_pipeline_v4.py.

Enhancements in v4 (requested)
------------------------------
1) Add --result_dir: all results and any intermediate outputs are written under this folder
2) Export per-trait accuracy summary to a TSV file
3) Export per-sample GEBVs (predictions) + observed values for all traits to one TSV file

Original logic preserved (unchanged)
------------------------------------
- Normalize chromosome names, intersect SNPs by (CHROM, POS)
- Build per-sample diploid allele matrices from VCF (REF/ALT-based)
- Intersect samples between VCF and phenotype (train/test separately)
- Encode genotypes using DPCFormer base_one_hot_encoding_1v()
- Per-chromosome SNP selection using MICSelector (training phenotypes only)
- Build 4D tensors [N, num_chrom, selected_snps, 1]
- Train single-trait model with train/val split (80/20), early stopping, evaluate on external test set
"""

import argparse
import logging
import os
import re
import sys
from datetime import datetime
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

# DPCFormer components
# All functions in the DPCFormer tool have been integrated into this wrapper program
"""
from data_process.encode import base_one_hot_encoding_1v  # 1-vector encoding
from model.model_1v_depth import PhenotypePredictor_1v
from model.early_stop import EarlyStopping
"""

# /data_process/encode.py in DPCFormer
#############################################################################################
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import mutual_info_regression

def genotype_to_dataframe(genotype_series):
    genotype_list = genotype_series.str.split(' ')
    return pd.DataFrame(genotype_list.tolist())


def base_one_hot_encoding_1v(seq_all_df):
    trans_code = {
        "AA": 0, "AT": 1, "TA": 1, "AC": 2, "CA": 2, "AG": 3, "GA": 3,
        "TT": 4, "TC": 5, "CT": 5, "TG": 6, "GT": 6, "CC": 7,
        "CG": 8, "GC": 8, "GG": 9,
        "00": -1, "A0": -1, "0A": -1, "T0": -1, "0T": -1,
        "C0": -1, "0C": -1, "G0": -1, "0G": -1
    }

    seq_all_np = seq_all_df.map(str).to_numpy()
    num_rows, num_cols = seq_all_np.shape
    output_cols = num_cols // 2
    code_arr = np.empty((num_rows, output_cols), dtype=int)

    for i in range(0, num_cols, 2):
        joint_seq = seq_all_np[:, i] + seq_all_np[:, i + 1]
        code_arr[:, i // 2] = np.vectorize(trans_code.get)(joint_seq)

    return pd.DataFrame(code_arr)


def base_one_hot_encoding_8v_dif(seq_all_df):
    base_code = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0],
                 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}

    trans_code = {}
    bases = ['A', 'C', 'G', 'T']
    for base1 in bases:
        for base2 in bases:
            trans_code[base1 + base2] = base_code[base1] + base_code[base2]

    seq_all_np = seq_all_df.values
    one_hot_arr = np.zeros((seq_all_np.shape[0], seq_all_np.shape[1] // 2, 8), dtype=np.int32)

    for row in range(seq_all_np.shape[0]):
        for base in range(0, seq_all_np.shape[1], 2):
            if base + 1 >= seq_all_np.shape[1]:
                break
            joint_seq = seq_all_np[row, base] + seq_all_np[row, base + 1]
            code = trans_code.get(joint_seq, [0] * 8)
            one_hot_arr[row, base // 2] = code

    return pd.DataFrame(one_hot_arr.reshape(one_hot_arr.shape[0], -1))


class MICSelector(BaseEstimator, TransformerMixin):
    def __init__(self, k=10000):
        self.k = k
        self.mic_scores_ = None
        self.top_k_indices_ = None

    def fit(self, X, y):
        self.mic_scores_ = mutual_info_regression(X, y, discrete_features=True, n_neighbors=7)
        self.top_k_indices_ = np.argsort(self.mic_scores_)[-self.k:]
        return self

    def transform(self, X):
        return X[:, self.top_k_indices_]

#############################################################################################

# /model/early_stop.py
class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=10, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 10
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta

    def __call__(self, val_loss):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0

#############################################################################################

# /model/model_1v_depth
import torch
import torch.nn.functional as F
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x

class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim):
        super(TransformerEncoderModel, self).__init__()
        self.positional_encoding = PositionalEncoding(d_model=input_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.positional_encoding(x)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return x

class ResConvBlockLayer(nn.Module):
    def __init__(self, in_channels, hidden, out_channels, kernel_size, dropout=0.25):
        super(ResConvBlockLayer, self).__init__()
        self.shortcut = nn.Conv1d(in_channels, out_channels, kernel_size=1, padding='same')
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=hidden, kernel_size=kernel_size, padding='same')
        self.conv2 = nn.Conv1d(in_channels=hidden, out_channels=out_channels, kernel_size=kernel_size, padding='same')
        self.bn = nn.BatchNorm1d(out_channels)
        self.maxpool = nn.MaxPool1d(kernel_size=kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = self.shortcut(x)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.bn(x)
        x += residual
        x = F.relu(x)
        x = self.maxpool(x)
        x = self.dropout(x)
        return x



class ChromosomeCNN(nn.Module):
    def __init__(self, num_snps,input_dim):
        super(ChromosomeCNN, self).__init__()
        self.res_block1 = ResConvBlockLayer(in_channels=input_dim, hidden=16, out_channels=32, kernel_size=5)
        self.res_block2 = ResConvBlockLayer(in_channels=32, hidden=32, out_channels=64, kernel_size=5)
        self.res_block3 = ResConvBlockLayer(in_channels=64, hidden=32, out_channels=16, kernel_size=5)

    def forward(self, x):
        x = self.res_block1(x)
        x = self.res_block2(x)
        x = self.res_block3(x)
        return x

class TransformerDecoder(nn.Module):
    def __init__(self, input_dim):
        super(TransformerDecoder, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=input_dim, nhead=4, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.transformer_layer, num_layers=1)
        self.fc = nn.Linear(input_dim, 1)

    def forward(self, x):
        x = x.permute(1, 0, 2)
        x = self.transformer_encoder(x)
        x = x.permute(1, 0, 2)
        return x

class PhenotypePredictor_1v(nn.Module):
    def __init__(self, num_chromosomes, snp_counts, input_dim):
        super(PhenotypePredictor_1v, self).__init__()

        self.chromosome_cnns = nn.ModuleList(
            [ChromosomeCNN(snp_counts[i],input_dim) for i in range(num_chromosomes)]
        )

        with torch.no_grad():
            sample_input = torch.randn(1, input_dim, snp_counts[0])
            sample_output = self.chromosome_cnns[0](sample_input)
            cnn_output_channels = sample_output.size(1)
            cnn_output_length = sample_output.size(2)
        self.transformer_decoder = TransformerEncoderModel(input_dim=cnn_output_channels)
        self.total_sequence_length = cnn_output_length * cnn_output_channels * num_chromosomes
        self.mlp = nn.Sequential(
            nn.Linear(self.total_sequence_length, 512),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        outputs = []
        for i, cnn in enumerate(self.chromosome_cnns):

            chromosome_data = x[:, i, :, :].permute(0, 2, 1)
            output = cnn(chromosome_data)
            outputs.append(output)

        combined_output = torch.cat(outputs, dim=2)
        combined_output = combined_output.permute(0, 2, 1)
        combined_output = self.transformer_decoder(combined_output)
        combined_output = torch.flatten(combined_output, 1)
        combined_output = self.mlp(combined_output)
        return combined_output

def print_output_shape(module, input, output):

    if isinstance(output, tuple):
        print(f"Module: {module.__class__.__name__}, Output is a tuple with {len(output)} elements")
        for i, item in enumerate(output):
            if hasattr(item, 'shape'):
                print(f"  Element {i}: Shape {item.shape}")
            else:
                print(f"  Element {i}: Not a tensor")
    else:
        print(f"Module: {module.__class__.__name__}, Output Shape: {output.shape}")

def register_hooks(model):
    hooks = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Sequential) and not isinstance(module, nn.ModuleList):
            hook = module.register_forward_hook(print_output_shape)
            hooks.append(hook)
    return hooks

if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dim = 10
    chromosome_tensors = torch.randn(1, 10, 1000, dim)
    phenotypes_tensor = torch.randn(1, 1)

    num_chromosomes = 10
    snp_counts = [1000]*num_chromosomes
    model = PhenotypePredictor_1v(num_chromosomes, snp_counts, dim).to(device)
    model.eval()
    hooks = register_hooks(model)

    chromosome_tensors = chromosome_tensors.to(device)
    output = model(chromosome_tensors)
    print(output.shape)

#############################################################################################


import random
import time

# =========================
# Reproducibility
# =========================
def set_global_seed(seed: int = 42):
    """Set seeds for reproducible results across numpy, torch, random."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info(f"Global random seed set to {seed}")


# =========================
# Logging utilities
# =========================
def setup_logging(log_dir: str) -> str:
    """Configure logging to both console and a timestamped log file."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_name = f"dpcformer_pipeline_{timestamp}.log"

    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_name)

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s %(levelname)s: %(message)s")

    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    fh = logging.FileHandler(log_path)
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logging.info("Logging initialized. Writing logs to: %s", log_path)
    return log_path


# =========================
# VCF helper functions
# =========================
def normalize_chrom(chrom: str) -> str:
    """Normalize chromosome names (e.g., 'chr1', 'Chr01', '1') to a unified form.

    Returns a string; numeric chromosomes are normalized to plain integers
    ("1", "2", ...). Non-numeric names are returned as-is (e.g. '1A', 'X', 'chrUn').
    """
    chrom = chrom.strip()
    # chr1, Chr01, etc.
    m = re.match(r"^[cC][hH][rR](\d+)$", chrom)
    if m:
        return str(int(m.group(1)))
    # pure digits: 01 -> 1
    if chrom.isdigit():
        return str(int(chrom))
    # chr1A, chrUn, etc. -> keep suffix
    m2 = re.match(r"^[cC][hH][rR](.+)$", chrom)
    if m2:
        return m2.group(1)
    return chrom


def collect_snp_keys(vcf_path: str) -> Tuple[set, Dict[Tuple[str, int], Tuple[str, str]]]:
    """Collect (normalized_chrom, pos) keys and (REF, ALT) for each variant in a VCF.

    Returns
    -------
    keys : set of (chrom, pos)
    ref_alt : dict mapping (chrom, pos) -> (REF, ALT)
    """
    keys: set[Tuple[str, int]] = set()
    ref_alt: Dict[Tuple[str, int], Tuple[str, str]] = {}
    with open(vcf_path, "r") as f:
        for line in f:
            if not line or line.startswith("##") or line.startswith("#CHROM"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 8:
                continue
            chrom_raw, pos_str, _id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                pos = int(pos_str)
            except ValueError:
                continue
            chrom = normalize_chrom(chrom_raw)
            key = (chrom, pos)
            keys.add(key)
            if key not in ref_alt:
                ref_alt[key] = (ref, alt)
    return keys, ref_alt


def get_common_snps(
    vcf_train: str, vcf_test: str
) -> Tuple[List[Tuple[str, int]], Dict[Tuple[str, int], Tuple[str, str]]]:
    """Find common SNPs by (normalized CHROM, POS) between train and test VCFs.

    Returns
    -------
    ordered_keys : list of (chrom, pos), sorted by chrom then pos
    ref_alt_train : dict mapping (chrom, pos) -> (REF, ALT) from train VCF
    """
    logging.info("Both marker_train and marker_test are VCF. Finding common SNPs by (CHROM, POS).")
    keys_train, ref_alt_train = collect_snp_keys(vcf_train)
    keys_test, _ref_alt_test = collect_snp_keys(vcf_test)
    common = keys_train.intersection(keys_test)

    def chrom_sort_key(ch: str) -> Tuple[int, str]:
        # numeric chromosomes first, in numeric order; then others lexicographically
        m = re.match(r"^(\d+)$", ch)
        if m:
            return (0, f"{int(m.group(1)):09d}")
        return (1, ch)

    ordered_keys = sorted(common, key=lambda x: (chrom_sort_key(x[0]), x[1]))
    logging.info(
        "Train VCF SNPs: %d, Test VCF SNPs: %d, Common SNPs: %d",
        len(keys_train), len(keys_test), len(ordered_keys)
    )
    return ordered_keys, ref_alt_train


def parse_vcf_samples(vcf_path: str) -> List[str]:
    """Return list of sample names from a VCF header."""
    with open(vcf_path, "r") as f:
        for line in f:
            if line.startswith("#CHROM"):
                parts = line.rstrip().split("\t")
                return parts[9:]
    raise ValueError(f"No #CHROM header line found in VCF: {vcf_path}")


def build_genotype_df_from_vcf(
    vcf_path: str,
    ordered_keys: List[Tuple[str, int]],
    key_to_index: Dict[Tuple[str, int], int],
) -> Tuple[pd.DataFrame, List[str], Dict[str, int]]:
    """Convert a VCF into a per-sample allele DataFrame for the given ordered SNP keys.

    The returned DataFrame has shape (n_samples, 2 * n_snps), cells are allele
    letters 'A', 'C', 'G', 'T', or '0' for missing, ordered consistently with
    ordered_keys. This is the input expected by base_one_hot_encoding_1v().
    """
    sample_names = parse_vcf_samples(vcf_path)
    n_samples = len(sample_names)
    n_snps = len(ordered_keys)
    logging.info(
        "Parsing VCF %s: header samples=%d, common SNPs=%d",
        vcf_path, n_samples, n_snps
    )

    alleles = np.full((n_samples, 2 * n_snps), "0", dtype="U1")
    sample_to_idx = {s: i for i, s in enumerate(sample_names)}

    with open(vcf_path, "r") as f:
        for line in f:
            if not line or line.startswith("##") or line.startswith("#CHROM"):
                continue
            parts = line.rstrip().split("\t")
            if len(parts) < 10:
                continue
            chrom_raw, pos_str, _id, ref, alt = parts[0], parts[1], parts[2], parts[3], parts[4]
            try:
                pos = int(pos_str)
            except ValueError:
                continue
            chrom = normalize_chrom(chrom_raw)
            key = (chrom, pos)
            if key not in key_to_index:
                continue
            snp_idx = key_to_index[key]
            alt_alleles = alt.split(',')

            fmt = parts[8]
            fmt_fields = fmt.split(':')
            try:
                gt_index = fmt_fields.index('GT')
            except ValueError:
                gt_index = 0  # assume GT is first if not explicitly found

            for samp_i, samp_field in enumerate(parts[9:]):
                fields = samp_field.split(':')
                if gt_index >= len(fields):
                    gt = './.'
                else:
                    gt = fields[gt_index]
                if '/' in gt:
                    a1_str, a2_str = gt.split('/')
                elif '|' in gt:
                    a1_str, a2_str = gt.split('|')
                else:
                    a1_str = a2_str = gt

                def idx_to_base(a_str: str) -> str:
                    if a_str in ('.', ''):
                        return '0'
                    try:
                        idx = int(a_str)
                    except ValueError:
                        return '0'
                    if idx == 0:
                        return ref
                    if 1 <= idx <= len(alt_alleles):
                        return alt_alleles[idx - 1]
                    return '0'

                b1 = idx_to_base(a1_str)
                b2 = idx_to_base(a2_str)
                alleles[samp_i, 2 * snp_idx] = b1
                alleles[samp_i, 2 * snp_idx + 1] = b2

    geno_df = pd.DataFrame(alleles, index=sample_names)
    return geno_df, sample_names, sample_to_idx


# =========================
# Phenotype helpers
# =========================
def read_pheno(pheno_path: str) -> Tuple[pd.DataFrame, str, List[str]]:
    """Read phenotype file with flexible separator.

    Assumes:
    - First column is sample ID
    - Remaining columns are traits (quantitative)
    """
    df = None
    for sep in [',', '\t', None]:
        try:
            df = pd.read_csv(pheno_path, sep=sep, engine='python')
            if df is not None and df.shape[1] >= 2:
                break
        except Exception:
            df = None
    if df is None or df.shape[1] < 2:
        raise ValueError(f"Could not parse phenotype file or not enough columns: {pheno_path}")

    id_col = df.columns[0]
    trait_cols = list(df.columns[1:])
    logging.info(
        "Parsed phenotype CSV/TXT %s with %d entries (ID column='%s', trait columns=%s).",
        pheno_path, df.shape[0], id_col, trait_cols
    )
    return df, id_col, trait_cols


def align_geno_pheno(
    geno_df: pd.DataFrame,
    sample_names: List[str],
    pheno_df: pd.DataFrame,
    id_col: str,
) -> Tuple[np.ndarray, pd.DataFrame, List[str]]:
    """Intersect and align genotype and phenotype by sample ID."""
    geno_ids = set(sample_names)
    pheno_ids = set(pheno_df[id_col].astype(str))
    common_ids = sorted(list(geno_ids.intersection(pheno_ids)))
    if not common_ids:
        raise ValueError("No common sample IDs found between VCF samples and phenotype file.")

    logging.info(
        "Phenotype file has %d entries; genotype (VCF) has %d samples; using %d common samples.",
        len(pheno_ids), len(geno_ids), len(common_ids)
    )

    geno_aligned = geno_df.loc[common_ids].to_numpy()
    pheno_aligned = pheno_df.set_index(id_col).loc[common_ids]
    return geno_aligned, pheno_aligned, common_ids


# =========================
# SNP grouping & selection
# =========================
def build_chrom_index_map(
    ordered_keys: List[Tuple[str, int]]
) -> Dict[str, List[int]]:
    """Build map from chromosome -> list of SNP indices (0-based) in ordered_keys."""
    chrom_to_indices: Dict[str, List[int]] = {}
    for idx, (chrom, _pos) in enumerate(ordered_keys):
        chrom_to_indices.setdefault(chrom, []).append(idx)
    return chrom_to_indices


def mic_select_per_chrom(
    geno_train_codes: np.ndarray,
    geno_test_codes: np.ndarray,
    chrom_to_indices: Dict[str, List[int]],
    y_train: np.ndarray,
    selected_snp_count: int,
) -> Tuple[torch.Tensor, torch.Tensor, List[str], List[int]]:
    """Apply MIC-based SNP selection per chromosome.

    Returns
    -------
    X_train_tensor : torch.Tensor, shape (n_train, n_chrom, selected_snps, 1)
    X_test_tensor  : torch.Tensor, shape (n_test, n_chrom, selected_snps, 1)
    chrom_list     : list of chromosomes (order used)
    snp_counts     : list of selected SNP counts per chromosome (all == selected_snp_count)
    """
    #from data_process.encode import MICSelector  # local import

    def chrom_order(c: str) -> Tuple[int, str]:
        return (0, f"{int(c):09d}") if c.isdigit() else (1, c)

    chrom_list = sorted(chrom_to_indices.keys(), key=chrom_order)
    train_chrom_tensors = []
    test_chrom_tensors = []
    snp_counts = []

    for chrom in chrom_list:
        indices = chrom_to_indices[chrom]
        X_train_chrom = geno_train_codes[:, indices]
        X_test_chrom = geno_test_codes[:, indices]

        n_snps_chrom = X_train_chrom.shape[1]
        k = min(selected_snp_count, n_snps_chrom)
        if k <= 0:
            logging.warning("Chromosome %s has no SNPs after intersection; skipping.", chrom)
            continue

        selector = MICSelector(k=k)
        selector.fit(X_train_chrom, y_train)
        sel_idx = selector.top_k_indices_
        X_train_sel = X_train_chrom[:, sel_idx]
        X_test_sel = X_test_chrom[:, sel_idx]

        if k < selected_snp_count:
            pad_train = np.zeros((X_train_sel.shape[0], selected_snp_count - k), dtype=X_train_sel.dtype)
            pad_test = np.zeros((X_test_sel.shape[0], selected_snp_count - k), dtype=X_test_sel.dtype)
            X_train_sel = np.concatenate([X_train_sel, pad_train], axis=1)
            X_test_sel = np.concatenate([X_test_sel, pad_test], axis=1)

        X_train_sel = X_train_sel[..., np.newaxis]
        X_test_sel = X_test_sel[..., np.newaxis]

        train_chrom_tensors.append(torch.tensor(X_train_sel, dtype=torch.float32))
        test_chrom_tensors.append(torch.tensor(X_test_sel, dtype=torch.float32))
        snp_counts.append(selected_snp_count)

        logging.info(
            "Chromosome %s: original SNPs=%d, selected=%d, padded_to=%d",
            chrom, n_snps_chrom, k, selected_snp_count
        )

    if not train_chrom_tensors:
        raise ValueError("No chromosomes remained after MIC selection.")

    X_train = torch.stack(train_chrom_tensors, dim=1)  # (N_train, n_chrom, snps, 1)
    X_test = torch.stack(test_chrom_tensors, dim=1)    # (N_test, n_chrom, snps, 1)
    logging.info(
        "Final genotype tensor shapes: train=%s, test=%s",
        tuple(X_train.shape), tuple(X_test.shape)
    )
    return X_train, X_test, chrom_list, snp_counts


# =========================
# Training loop
# =========================
def train_single_trait(
    X_train: torch.Tensor,
    y_train: np.ndarray,
    X_test: torch.Tensor,
    y_test: np.ndarray,
    chrom_list: List[str],
    snp_counts: List[int],
    device: torch.device,
    trait_name: str,
) -> Tuple[float, float, float, np.ndarray, np.ndarray]:
    """Train PhenotypePredictor_1v on a single trait and evaluate on the test set.

    Returns
    -------
    pearson_r, mse, mae (on original phenotype scale), preds, true
    """
    scaler = StandardScaler()
    y_train_scaled = scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
    y_test_scaled = scaler.transform(y_test.reshape(-1, 1)).flatten()

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train.numpy(), y_train_scaled, test_size=0.2, random_state=42
    )
    X_tr = torch.tensor(X_tr, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.float32)
    y_tr = torch.tensor(y_tr, dtype=torch.float32).unsqueeze(1)
    y_val = torch.tensor(y_val, dtype=torch.float32).unsqueeze(1)
    X_test_t = X_test.clone().detach()
    y_test_scaled_t = torch.tensor(y_test_scaled, dtype=torch.float32).unsqueeze(1)

    batch_size = 64
    train_ds = TensorDataset(X_tr, y_tr)
    val_ds = TensorDataset(X_val, y_val)
    test_ds = TensorDataset(X_test_t, y_test_scaled_t)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    num_chrom = len(chrom_list)
    input_dim = 1  # 1-vector encoding
    model = PhenotypePredictor_1v(num_chrom, snp_counts, input_dim=input_dim).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=20)

    max_epochs = 300
    best_val_loss = float('inf')
    best_state = None

    logging.info(
        "[Trait %s] Training model with %d chromosomes, selected_snps=%s",
        trait_name, num_chrom, snp_counts
    )

    for epoch in range(1, max_epochs + 1):
        model.train()
        total_train_loss = 0.0
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            preds = model(xb)
            loss = criterion(preds, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        avg_train_loss = total_train_loss / len(train_ds)

        model.eval()
        total_val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)
                preds = model(xb)
                loss = criterion(preds, yb)
                total_val_loss += loss.item() * xb.size(0)
        avg_val_loss = total_val_loss / len(val_ds)
        logging.info(
            "[Trait %s] Epoch %d/%d: train_loss=%.6f, val_loss=%.6f",
            trait_name, epoch, max_epochs, avg_train_loss, avg_val_loss
        )

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_state = model.state_dict()

        early_stopping(avg_val_loss)
        if early_stopping.early_stop:
            logging.info("[Trait %s] Early stopping at epoch %d", trait_name, epoch)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    preds_scaled = []
    true_scaled = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            out = model(xb)
            preds_scaled.extend(out.cpu().numpy().flatten())
            true_scaled.extend(yb.cpu().numpy().flatten())

    preds_scaled = np.array(preds_scaled)
    true_scaled = np.array(true_scaled)

    preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
    true = y_test

    mse = float(np.mean((preds - true) ** 2))
    mae = float(np.mean(np.abs(preds - true)))
    if np.std(preds) > 0 and np.std(true) > 0:
        pearson_r = float(pearsonr(preds, true)[0])
    else:
        pearson_r = float('nan')

    logging.info(
        "[Trait %s] Test results: Pearson r=%.4f, MSE=%.6f, MAE=%.6f",
        trait_name, pearson_r, mse, mae
    )
    return pearson_r, mse, mae, preds, true


# =========================
# Main pipeline
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="DPCFormer wrapper pipeline: VCF + phenotype (train/test) -> GS performance."
    )
    parser.add_argument("--marker_train", required=True, help="Training genotype VCF file path.")
    parser.add_argument("--pheno_train", required=True, help="Training phenotype CSV/TXT file path.")
    parser.add_argument("--marker_test", required=True, help="Test genotype VCF file path.")
    parser.add_argument("--pheno_test", required=True, help="Test phenotype CSV/TXT file path.")

    # NEW: result dir (required)
    parser.add_argument(
        "--result_dir", required=True,
        help="Directory to store all results and intermediate outputs."
    )

    # Keep selected_snps as before
    parser.add_argument(
        "--selected_snps", type=int, default=1000,
        help="Number of SNPs to select per chromosome via MIC (default: 1000)."
    )

    # Optional override for log dir (defaults to result_dir)
    parser.add_argument(
        "--log_dir", default=None,
        help="Directory to store pipeline log file (default: result_dir)."
    )
    return parser.parse_args()


def main() -> None:
    pipeline_start = time.time()

    args = parse_args()

    # Prepare result directory structure
    result_dir = os.path.abspath(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)

    # Logs go to log_dir if provided; else to result_dir
    log_dir = os.path.abspath(args.log_dir) if args.log_dir else result_dir
    os.makedirs(log_dir, exist_ok=True)
    setup_logging(log_dir)

    # (Optional) place any future intermediate outputs here
    intermediate_dir = os.path.join(result_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)

    set_global_seed(42)

    marker_train = args.marker_train
    marker_test = args.marker_test
    pheno_train_path = args.pheno_train
    pheno_test_path = args.pheno_test
    selected_snp_count = args.selected_snps

    logging.info("marker_train: %s", marker_train)
    logging.info("pheno_train : %s", pheno_train_path)
    logging.info("marker_test : %s", marker_test)
    logging.info("pheno_test  : %s", pheno_test_path)
    logging.info("result_dir  : %s", result_dir)
    logging.info("log_dir     : %s", log_dir)
    logging.info("intermediate_dir: %s", intermediate_dir)

    # Output files (NEW)
    accuracy_out = os.path.join(result_dir, "dpcformer_accuracy_summary.tsv")
    gebv_out = os.path.join(result_dir, "dpcformer_GEBVs.tsv")

    # Containers for outputs (NEW)
    accuracy_records: List[Dict[str, object]] = []
    gebv_records: List[Dict[str, object]] = []

    # 1) Common SNPs between train/test VCFs
    ordered_keys, _ref_alt_train = get_common_snps(marker_train, marker_test)
    if not ordered_keys:
        raise ValueError("No common SNPs found between train and test VCF files.")
    key_to_index = {k: i for i, k in enumerate(ordered_keys)}
    chrom_to_indices = build_chrom_index_map(ordered_keys)

    # 2) Per-sample allele DataFrames from VCFs
    geno_df_train, sample_names_train, _ = build_genotype_df_from_vcf(
        marker_train, ordered_keys, key_to_index
    )
    geno_df_test, sample_names_test, _ = build_genotype_df_from_vcf(
        marker_test, ordered_keys, key_to_index
    )

    # 3) Read and align phenotype files
    pheno_train_df, id_col_train, trait_cols_train = read_pheno(pheno_train_path)
    pheno_test_df, id_col_test, trait_cols_test = read_pheno(pheno_test_path)
    if id_col_train != id_col_test:
        logging.warning(
            "Train and test phenotype files have different ID columns ('%s' vs '%s'). Assuming same semantics.",
            id_col_train, id_col_test
        )

    common_traits = [t for t in trait_cols_train if t in trait_cols_test]
    if not common_traits:
        raise ValueError("No common trait columns between train and test phenotype files.")
    logging.info("Common trait columns between train/test: %s", common_traits)

    geno_train_aligned_raw, pheno_train_aligned, common_train_ids = align_geno_pheno(
        geno_df_train, sample_names_train, pheno_train_df, id_col_train
    )
    geno_test_aligned_raw, pheno_test_aligned, common_test_ids = align_geno_pheno(
        geno_df_test, sample_names_test, pheno_test_df, id_col_test
    )

    # 4) Encode genotypes using 1-vector encoding
    geno_train_aligned_df = pd.DataFrame(
        geno_train_aligned_raw, index=common_train_ids
    )
    geno_test_aligned_df = pd.DataFrame(
        geno_test_aligned_raw, index=common_test_ids
    )

    logging.info("Encoding genotypes (1-vector) for train/test...")
    geno_train_codes_df = base_one_hot_encoding_1v(geno_train_aligned_df)
    geno_test_codes_df = base_one_hot_encoding_1v(geno_test_aligned_df)

    geno_train_codes = geno_train_codes_df.to_numpy()
    geno_test_codes = geno_test_codes_df.to_numpy()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    # 5) For each trait, MIC selection + train/evaluate + export outputs (NEW)
    for trait_idx, trait_name in enumerate(common_traits):
        trait_start = time.time()
        logging.info("=== Trait %d/%d: %s ===", trait_idx + 1, len(common_traits), trait_name)

        y_train = pheno_train_aligned[trait_name].to_numpy().astype(float)
        y_test = pheno_test_aligned[trait_name].to_numpy().astype(float)

        mask_train = ~np.isnan(y_train)
        mask_test = ~np.isnan(y_test)
        if not np.all(mask_train):
            logging.info("[Trait %s] Dropping %d NaN entries from training phenotypes.", trait_name, np.sum(~mask_train))
        if not np.all(mask_test):
            logging.info("[Trait %s] Dropping %d NaN entries from test phenotypes.", trait_name, np.sum(~mask_test))

        geno_train_trait = geno_train_codes[mask_train]
        y_train_trait = y_train[mask_train]
        geno_test_trait = geno_test_codes[mask_test]
        y_test_trait = y_test[mask_test]

        # Keep aligned test IDs for exporting GEBVs (NEW)
        test_ids_trait = np.array(common_test_ids, dtype=str)[mask_test]

        X_train_tensor, X_test_tensor, chrom_list, snp_counts = mic_select_per_chrom(
            geno_train_trait,
            geno_test_trait,
            chrom_to_indices,
            y_train_trait,
            selected_snp_count=selected_snp_count,
        )

        pearson_r, mse, mae, preds, true = train_single_trait(
            X_train_tensor,
            y_train_trait,
            X_test_tensor,
            y_test_trait,
            chrom_list,
            snp_counts,
            device,
            trait_name,
        )

        # Record accuracy (NEW)
        accuracy_records.append({
            "Trait": trait_name,
            "PearsonR": pearson_r,
            "MSE": mse,
            "MAE": mae
        })

        # Record GEBVs + observed values (NEW)
        # preds/true correspond exactly to test_ids_trait order
        for sid, obs, gebv in zip(test_ids_trait, true, preds):
            gebv_records.append({
                "Sample": sid,
                "Trait": trait_name,
                "Observed": float(obs),
                "GEBV": float(gebv)
            })

        trait_elapsed = time.time() - trait_start
        logging.info("[Trait %s] Training+prediction time: %.2f seconds (%.2f minutes)",
                     trait_name, trait_elapsed, trait_elapsed / 60)

    # Write outputs (NEW)
    acc_df = pd.DataFrame(accuracy_records)
    acc_df.to_csv(accuracy_out, sep="\t", index=False)
    logging.info("Saved accuracy summary: %s", accuracy_out)

    gebv_df = pd.DataFrame(gebv_records)
    gebv_df.to_csv(gebv_out, sep="\t", index=False)
    logging.info("Saved GEBVs (Observed + Predicted): %s", gebv_out)

    # Total time
    total_elapsed = time.time() - pipeline_start
    logging.info("=============================================================")
    logging.info("DPCFormer pipeline completed.")
    logging.info("Total runtime: %.2f seconds (%.2f minutes, %.2f hours)",
                 total_elapsed, total_elapsed / 60, total_elapsed / 3600)
    logging.info("=============================================================")


if __name__ == "__main__":
    main()
