#!/usr/bin/env python3
"""
wheatGP_pipeline_v4.py

Modes:
  - convert:
      Convert inputs (VCF/CSV/TXT/PKL) to aligned PKLs and exit.
  - prediction:
      Convert if needed, align, then train + test WheatGP_base models.

Features:
- Markers: VCF (.vcf/.vcf.gz) or PKL (.pkl)
- Phenotypes: CSV/TXT (multi-trait; first column is sample ID, others are traits) or PKL (.pkl)
- Overwrite rule:
    * If marker/pheno input is PKL -> use as-is, do NOT overwrite G_*.pkl / P_*.pkl.
    * If marker/pheno input is VCF/CSV/TXT -> generate/overwrite PKL.
- Robust alignment of PKLs:
    1) G int-keys & P int-keys -> align by sorted int keys
    2) G int-keys & P name-keys -> map phenotype insertion order -> 0..n-1
    3) G name-keys & P name-keys -> align by sample names
    4) G name-keys & P int-keys -> error (ambiguous)
- VCF + VCF:
    * Find common SNPs by (CHROM, POS)
    * Restrict both train/test VCFs to these common SNPs
    * For each of train/test, restrict samples to intersection:
        VCF samples ∩ phenotype IDs
- Multi-trait:
    * Phenotype TXT/CSV may have multiple trait columns.
    * Build PKLs where phenotype values are vectors per sample.
    * Automatically loop over all traits and train one WheatGP_base per trait.
"""

#!/usr/bin/env python3
"""
wheatGP_pipeline_v7.py
"""

from collections import Counter
import argparse
import gzip
import logging
import os
import pickle
import sys

import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import TensorDataset, DataLoader, random_split
    from torch.optim.lr_scheduler import StepLR
except Exception:
    torch = None

import time
import random
import numpy as np


#########################################################################################
""" 
WheatGP model classes
These two class es are provided from the jithub website of wheatGP: wheat_GP_base.py
We dierectlyn merge theri class into this wrapper pipeline
"""
#########################################################################################

import torch
import torch.nn as nn

class ConvPart(nn.Module):
    def __init__(self):
        super(ConvPart, self).__init__()
        # First convolutional layer: input channels = 1, output channels = 2, kernel size = 1, padding = 1
        self.conv0 = nn.Conv1d(1, 2, 1, padding=1)
        # ReLU activation function after the first convolutional layer
        self.relu0 = nn.ReLU()
        self.conv1 = nn.Conv1d(2, 4, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(4, 8, 9, padding=1)
        self.relu2 = nn.ReLU()
        self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv0(x)
        x = self.relu0(x)

        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = self.drop(x)

        return x
class ShapeModule(nn.Module):
    def __init__(self):
        # Call the constructor of the parent class
        super(ShapeModule, self).__init__()

    def forward(self, x1, x2, x3, x4, x5, adjust_dim=True, concat=True):
        if adjust_dim:
            x1 = x1.unsqueeze(1)
            x2 = x2.unsqueeze(1)
            x3 = x3.unsqueeze(1)
            x4 = x4.unsqueeze(1)
            x5 = x5.unsqueeze(1)
        if concat:
            A_flat = x1.view(x1.size(0), -1)
            B_flat = x2.view(x2.size(0), -1)
            C_flat = x3.view(x3.size(0), -1)
            D_flat = x4.view(x4.size(0), -1)
            E_flat = x5.view(x5.size(0), -1)
            output = torch.cat((A_flat, B_flat, C_flat, D_flat, E_flat), dim=1)
            output = output.reshape(output.shape[0], 1, -1)

            return output
        else:

            return x1, x2, x3, x4, x5

class LSTMModule(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, batch_first=True):
        # Call the constructor of the parent class
        super(LSTMModule, self).__init__()
        # Define an LSTM layer with the specified input size, hidden size, number of layers, and batch first flag
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=batch_first)
        self.drop = nn.Dropout(0.3)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        lstm_out = self.drop(lstm_out)
        return lstm_out

class wheatGP_base(nn.Module):
    def __init__(self, lstm_dim):
        super(wheatGP_base, self).__init__()
        self.ConvPart = ConvPart()
        self.lstm = LSTMModule(lstm_dim, 128)
        self.shape_module = ShapeModule()
        self.fc = nn.Sequential(nn.Linear(128, 1))

    def forward(self, x1, x2, x3, x4, x5):
        x1, x2, x3, x4, x5 = self.shape_module(x1, x2, x3, x4, x5, adjust_dim=True, concat=False)
        A = self.ConvPart(x1)
        B = self.ConvPart(x2)
        C = self.ConvPart(x3)
        D = self.ConvPart(x4)
        E = self.ConvPart(x5)

        output = self.shape_module(A, B, C, D, E, adjust_dim=False)  # Assume the dimensions have been adjusted before
        output = self.lstm(output)

        output = output[:, -1, :]
        output = self.fc(output)

        return output

    def freeze_layers(self, freeze_conv=True, freeze_lstm=True, freeze_fc=True):
        # Freeze or unfreeze the parameters of the convolutional part
        for param in self.ConvPart.parameters():
            param.requires_grad = not freeze_conv
        # Freeze or unfreeze the parameters of the LSTM module
        for param in self.lstm.parameters():
            param.requires_grad = not freeze_lstm
        # Freeze or unfreeze the parameters of the fully connected layer
        for param in self.fc.parameters():
            param.requires_grad = not freeze_fc
#########################################################################################

def set_global_seed(seed: int = 42):
    """Set RNG seeds for reproducibility across Python/NumPy/PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[INFO] Global random seed set to {seed}")

# -------------------------------------------------------------
# Robust logging setup: console + timestamped file
# -------------------------------------------------------------
import logging, sys, os, datetime

def init_logging(log_dir: str):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"pipeline_{timestamp}.log"

    os.makedirs(log_dir, exist_ok=True)
    log_filename = os.path.join(log_dir, log_filename)

    # Create global logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    # Remove any handlers from previous initialization
    for h in list(logger.handlers):
        logger.removeHandler(h)

    # Formatter
    fmt = "%(asctime)s %(levelname)s: %(message)s"
    formatter = logging.Formatter(fmt)

    # Console handler
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(formatter)
    logger.addHandler(console)

    # File handler
    file = logging.FileHandler(log_filename, mode='w')
    file.setLevel(logging.INFO)
    file.setFormatter(formatter)
    logger.addHandler(file)

    logger.info(f"Logging initialized. Writing logs to: {log_filename}")

# Logging will be initialized in main() after parsing --result_dir
# ----------------------------------------------------------------------
# Utility IO
# ----------------------------------------------------------------------

def open_maybe_gz(path, mode='rt'):
    path = str(path)
    if path.endswith('.gz'):
        return gzip.open(path, mode)
    return open(path, mode, encoding='utf-8')

def save_pickle(obj, path):
    with open(path, 'wb') as fh:
        pickle.dump(obj, fh, protocol=4)
    logging.info(f"Wrote {path}")

def load_pickle(path):
    with open(path, 'rb') as fh:
        return pickle.load(fh)


# ----------------------------------------------------------------------
# Output directory helper
# ----------------------------------------------------------------------
def _outpath(result_dir: str, filename: str) -> str:
    """Join filename to result_dir; if filename is absolute, return as-is."""
    if filename is None:
        return None
    if os.path.isabs(filename):
        return filename
    return os.path.join(result_dir, filename)

# ----------------------------------------------------------------------
# Phenotype parsing (CSV/TXT -> DataFrame; first col = ID, others = traits)
# ----------------------------------------------------------------------

def detect_separator(file_path, n_lines=5):
    """Detect the most likely separator in a text file."""
    separators = ['\t', ',', ';', ' ']
    counts = {sep: 0 for sep in separators}
    
    try:
        with open_maybe_gz(file_path, 'rt') as f:
            lines = [next(f) for _ in range(min(n_lines, 5)) if f]
    except StopIteration:
        return '\t'  # Default to tab
    
    for line in lines:
        for sep in separators:
            counts[sep] += line.count(sep)
    
    # Return separator with highest count
    detected_sep = max(counts.items(), key=lambda x: x[1])[0]
    
    # If no separator found or whitespace wins but tabs exist, prefer tab
    if detected_sep == ' ' and counts['\t'] > 0:
        detected_sep = '\t'
    
    return detected_sep

def parse_pheno_file_to_df(pheno_path, sep=None):
    """
    Read a phenotype CSV/TXT file and return:
        df, id_col_name, trait_col_names

    Assumptions:
      - First column = sample ID
      - Remaining columns = traits (one or more)
    """
    path = str(pheno_path)
    ext = os.path.splitext(path)[1].lower()
    
    # If separator is explicitly provided, use it
    if sep is not None:
        logging.info(f"Using user-specified separator: {repr(sep)}")
        df = pd.read_csv(pheno_path, sep=sep, header=0, engine='python')
    else:
        # Try to auto-detect the separator
        try:
            # Read first line to detect separator
            with open_maybe_gz(pheno_path, 'rt') as f:
                first_line = f.readline().strip()
            
            # Count different separators
            tab_count = first_line.count('\t')
            comma_count = first_line.count(',')
            semicolon_count = first_line.count(';')
            space_count = len(first_line.split()) - 1  # Approximate
            
            logging.debug(f"Separator counts - Tabs: {tab_count}, Commas: {comma_count}, "
                         f"Semicolons: {semicolon_count}, Spaces: {space_count}")
            
            # Determine the most likely separator
            if tab_count > max(comma_count, semicolon_count, space_count/2):
                detected_sep = '\t'
                logging.info(f"Auto-detected tab separator in {pheno_path}")
            elif comma_count > max(tab_count, semicolon_count, space_count/2):
                detected_sep = ','
                logging.info(f"Auto-detected comma separator in {pheno_path}")
            elif semicolon_count > max(tab_count, comma_count, space_count/2):
                detected_sep = ';'
                logging.info(f"Auto-detected semicolon separator in {pheno_path}")
            else:
                # Default to tab for phenotype files (common in genetics)
                detected_sep = '\t'
                logging.info(f"Using default tab separator for {pheno_path}")
            
            df = pd.read_csv(pheno_path, sep=detected_sep, header=0, engine='python')
            
        except Exception as e:
            logging.warning(f"Auto-detection failed for {pheno_path}: {e}")
            # Fallback: try common separators
            for fallback_sep in ['\t', ',', ';']:
                try:
                    df = pd.read_csv(pheno_path, sep=fallback_sep, header=0, engine='python')
                    logging.info(f"Successfully parsed with fallback separator: {repr(fallback_sep)}")
                    break
                except Exception:
                    continue
            else:
                # Last resort: try generic whitespace
                try:
                    df = pd.read_csv(pheno_path, sep=r'\s+', header=0, engine='python')
                    logging.info(f"Parsed with generic whitespace separator")
                except Exception as e2:
                    raise ValueError(f"Could not parse phenotype file {pheno_path} with any separator: {e2}")
    
    if df.shape[1] < 2:
        # If only 1 column, try splitting on tabs (common issue with misnamed .csv files)
        if df.shape[1] == 1 and '\t' in df.columns[0]:
            logging.warning(f"File appears to be tab-separated but has .csv extension. Attempting to fix...")
            # Split the single column into multiple columns
            split_data = df.iloc[:, 0].str.split('\t', expand=True)
            # Use first row as header
            df = pd.DataFrame(split_data.values[1:], columns=split_data.iloc[0])
        
        if df.shape[1] < 2:
            raise ValueError(
                f"Phenotype file {pheno_path} must have at least 2 columns "
                f"(ID + at least one trait). Found columns: {list(df.columns)}"
            )

    id_col = df.columns[0]
    trait_cols = list(df.columns[1:])

    # Normalize IDs to strings without surrounding whitespace
    df[id_col] = df[id_col].astype(str).str.strip()

    logging.info(
        f"Parsed phenotype file {pheno_path} with {len(df)} entries "
        f"(ID column='{id_col}', {len(trait_cols)} traits: {trait_cols})."
    )
    return df, id_col, trait_cols


# ----------------------------------------------------------------------
# VCF helpers
# ----------------------------------------------------------------------

def parse_vcf_get_header_samples(vcf_path):
    with open_maybe_gz(vcf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#CHROM'):
                parts = line.rstrip('\n').split('\t')
                return parts[9:]
    raise ValueError(f"No #CHROM header found in VCF: {vcf_path}")

def get_vcf_marker_set(vcf_path):
    """
    Return a set of (chrom, pos) tuples from a VCF (for intersection).
    """
    markers = set()
    with open_maybe_gz(vcf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 2:
                continue
            chrom = cols[0]
            try:
                pos = int(cols[1])
            except Exception:
                continue
            markers.add((chrom, pos))
    return markers

def parse_vcf_to_genotype_dict(vcf_path, sample_subset=None, allowed_markers=None):
    """
    Parse a VCF into a dict[int -> np.array[int16]] with simple mode imputation.

    - sample_subset: list of sample names to keep (order preserved). If None, use header order.
    - allowed_markers: set of (chrom, pos) to keep; if None, keep all markers.

    Genotype coding:
        0/0 -> 1
        0/1 or 1/0 -> 2
        1/1 -> 3
        missing -> imputed to modal genotype at that marker.
    """
    header_samples = parse_vcf_get_header_samples(vcf_path)
    if sample_subset is None:
        sample_map = header_samples
    else:
        sample_map = [s for s in sample_subset if s in header_samples]
        if len(sample_map) == 0:
            raise ValueError("None of the requested phenotype sample IDs found in VCF header.")
    nsamples = len(sample_map)
    logging.info(f"Parsing VCF {vcf_path}: header samples={len(header_samples)} -> using {nsamples} samples")

    geno_lists = [[] for _ in range(nsamples)]
    marker_count = 0
    with open_maybe_gz(vcf_path, 'rt') as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            cols = line.rstrip('\n').split('\t')
            if len(cols) < 10:
                continue
            chrom = cols[0]
            try:
                pos = int(cols[1])
            except Exception:
                continue

            if allowed_markers is not None and (chrom, pos) not in allowed_markers:
                continue

            fmt = cols[8].split(':')
            try:
                gt_index = fmt.index('GT')
            except ValueError:
                continue
            sample_fields = cols[9:]
            header_gt = [sf.split(':')[gt_index] if sf else './.' for sf in sample_fields]

            for i, s in enumerate(sample_map):
                hdr_idx = header_samples.index(s)
                gt = header_gt[hdr_idx]
                if gt is None or gt in ('.', './.', '.|.'):
                    geno_lists[i].append(None)
                else:
                    alleles = gt.replace('|', '/').split('/')
                    if '.' in alleles:
                        geno_lists[i].append(None)
                    else:
                        try:
                            a = int(alleles[0])
                            b = int(alleles[1]) if len(alleles) > 1 else a
                            ssum = a + b
                            if ssum == 0:
                                geno_lists[i].append(1)
                            elif ssum == 1:
                                geno_lists[i].append(2)
                            elif ssum == 2:
                                geno_lists[i].append(3)
                            else:
                                geno_lists[i].append(3)
                        except Exception:
                            geno_lists[i].append(None)
            marker_count += 1

    if marker_count == 0:
        raise ValueError("No markers parsed from VCF: " + str(vcf_path))

    nmarkers = marker_count
    imputed = np.empty((nsamples, nmarkers), dtype=np.int16)
    for j in range(nmarkers):
        col = [geno_lists[i][j] for i in range(nsamples)]
        non_missing = [int(x) for x in col if x is not None]
        if len(non_missing) == 0:
            mode_val = 1
        else:
            mode_val = Counter(non_missing).most_common(1)[0][0]
        for i in range(nsamples):
            v = col[i]
            imputed[i, j] = np.int16(mode_val if v is None else v)

    geno_dict = {i: imputed[i, :].astype(np.int16) for i in range(nsamples)}
    return geno_dict, sample_map

# ----------------------------------------------------------------------
# Alignment helpers (implements the 4 rules; numpy-int safe)
# ----------------------------------------------------------------------

def is_int_keyed(d):
    """
    Accept both Python int and numpy integer keys.
    """
    return isinstance(d, dict) and all(isinstance(k, (int, np.integer)) for k in d.keys())

def is_str_keyed(d):
    return isinstance(d, dict) and all(isinstance(k, str) for k in d.keys())

def align_and_write_pkls(G_obj, P_obj, which, G_input_is_pkl, P_input_is_pkl, result_dir: str):
    """
    Align genotype and phenotype objects (which may be loaded from PKL or generated from VCF/CSV),
    following the 4 rules. Always write int-keyed PKLs:
        G_{which}.pkl, P_{which}.pkl, sample_map_{which}.pkl

    Returns (G_int_dict, P_int_dict, sample_map_list)
    """

    G_out = _outpath(result_dir, f'G_{which}.pkl')
    P_out = _outpath(result_dir, f'P_{which}.pkl')
    sample_map_out = f'sample_map_{which}.pkl'

    # Identify key types and normalize numpy-int keys to Python int
    G_int = is_int_keyed(G_obj)
    G_str = is_str_keyed(G_obj)
    P_int = is_int_keyed(P_obj)
    P_str = is_str_keyed(P_obj)

    if G_int:
        G_obj = {int(k): v for k, v in G_obj.items()}
    if P_int:
        P_obj = {int(k): v for k, v in P_obj.items()}

    # Recompute after normalization
    G_int = is_int_keyed(G_obj)
    G_str = is_str_keyed(G_obj)
    P_int = is_int_keyed(P_obj)
    P_str = is_str_keyed(P_obj)

    # Case 1: both int-keyed -> align by sorted keys
    if G_int and P_int:
        G_keys = sorted(G_obj.keys())
        P_keys = sorted(P_obj.keys())
        if len(G_keys) != len(P_keys):
            raise RuntimeError(
                f"Int-keyed genotype ({len(G_keys)}) and phenotype ({len(P_keys)}) sizes differ -> cannot align."
            )
        if G_keys == P_keys:
            sample_map = list(G_keys)
            save_pickle(G_obj, G_out)
            save_pickle(P_obj, P_out)
            save_pickle(sample_map, sample_map_out)
            return G_obj, P_obj, sample_map
        else:
            P_new = {}
            for k in G_keys:
                if k not in P_obj:
                    raise RuntimeError(f"Genotype key {k} not found in phenotype PKL -> cannot align.")
                P_new[k] = P_obj[k]

            if hasattr(P_obj, 'trait_names'):
                P_new.trait_names = P_obj.trait_names           
            save_pickle(G_obj, G_out)
            save_pickle(P_new, P_out)
            save_pickle(G_keys, sample_map_out)
            return G_obj, P_new, G_keys

    # Case 2: genotype int-keyed, phenotype string-keyed
    if G_int and P_str:
        G_keys = sorted(G_obj.keys())
        nG = len(G_keys)
        nP = len(P_obj)
        if nG != nP:
            raise RuntimeError(
                f"Genotype int-keyed has {nG} entries but phenotype has {nP} sample-name entries -> "
                "cannot safely auto-align."
            )
        sample_names = list(P_obj.keys())  # insertion order
        P_new = {}
        for i, s in enumerate(sample_names):
            P_new[i] = np.asarray(P_obj[s], dtype=np.float32)
        sample_map = sample_names

        if hasattr(P_obj, 'trait_names'):
            P_new.trait_names = P_obj.trait_names           

        save_pickle(G_obj, G_out)
        save_pickle(P_new, P_out)
        save_pickle(sample_map, sample_map_out)
        return G_obj, P_new, sample_map

    # Case 3: genotype string-keyed and phenotype string-keyed -> align by names
    if G_str and P_str:
        G_names = set(G_obj.keys())
        P_names = set(P_obj.keys())
        if G_names != P_names:
            inter = G_names.intersection(P_names)
            if len(inter) == 0:
                raise RuntimeError("No overlapping sample names between genotype PKL and phenotype PKL.")
            raise RuntimeError(
                "Sample name sets in genotype and phenotype PKLs are not identical -> cannot safely align. "
                f"Differences: genotypes_only={G_names - P_names}, phenos_only={P_names - G_names}"
            )
        sample_map = sorted(list(G_names))  # deterministic order
        G_new = {}
        P_new = {}
        for i, s in enumerate(sample_map):
            G_new[i] = G_obj[s]
            P_new[i] = np.asarray(P_obj[s], dtype=np.float32)

        if hasattr(P_obj, 'trait_names'):
            P_new.trait_names = P_obj.trait_names           

        save_pickle(G_new, G_out)
        save_pickle(P_new, P_out)
        save_pickle(sample_map, sample_map_out)
        return G_new, P_new, sample_map

    # Case 4: genotype string-keyed and phenotype int-keyed -> ambiguous
    if G_str and P_int:
        raise RuntimeError(
            "Genotype PKL keyed by sample names but phenotype PKL keyed by integers -> alignment ambiguous. "
            "Provide consistent PKL formats or a sample_map."
        )

    # Fallback
    raise RuntimeError(
        "Unhandled PKL formats during alignment. "
        f"G types: int? {G_int}, str? {G_str}; P types: int? {P_int}, str? {P_str}"
    )

# ----------------------------------------------------------------------
# Conversion logic (Option 2) with VCF-train / VCF-test intersection
# ----------------------------------------------------------------------

def write_GP_from_inputs(marker_input, pheno_input, which, result_dir: str, pheno_sep=None):
    """
    Generic path:
      - marker_input: VCF (.vcf/.vcf.gz) or PKL (.pkl)
      - pheno_input:  CSV/TXT (multi-trait) or PKL (.pkl)
      - which: 'train' or 'test'

    Returns: (G_int_dict, P_int_dict, sample_map_list)
    """
    marker_path = str(marker_input)
    pheno_path = str(pheno_input)

    marker_is_pkl = marker_path.lower().endswith('.pkl')
    pheno_is_pkl  = pheno_path.lower().endswith('.pkl')

    G_out = f'G_{which}.pkl'
    P_out = f'P_{which}.pkl'

    # Phenotype
    if pheno_is_pkl:
        P_obj = load_pickle(pheno_path)
        logging.info(f"Loaded phenotype PKL from {pheno_path}; will NOT overwrite {P_out}.")
        pheno_trait_cols = None
    else:
        df, id_col, trait_cols = parse_pheno_file_to_df(pheno_path, sep=pheno_sep)
        pheno_trait_cols = trait_cols
        P_obj = None  # will build after we know genotype samples if marker is VCF
         
         # Store trait names for later use
        P_obj_info = {
            'trait_names': trait_cols,
            'df': df,
            'id_col': id_col
        }

    # Marker
    if marker_is_pkl:
        G_obj = load_pickle(marker_path)
        logging.info(f"Loaded genotype PKL from {marker_path}; will NOT overwrite {G_out}.")
        sample_map = None

        # If phenotype came from CSV/TXT, just map ID -> vector over trait_cols
        if not pheno_is_pkl:
            df_idx = df.set_index(id_col)
            
            def build_pheno_dict_with_names(df, id_col, trait_cols, sample_ids):
                """Build phenotype dict and store trait names as metadata."""
                P_obj = {}
                df_idx = df.set_index(id_col)
                
                for sid in sample_ids:
                    if sid in df_idx.index:
                        P_obj[sid] = df_idx.loc[sid, trait_cols].to_numpy(dtype=np.float32)
                
                # Add trait names as an attribute
                P_obj._trait_names = trait_cols
                return P_obj

            # Then use it:
            P_obj = build_pheno_dict_with_names(df_sub, id_col, pheno_trait_cols, common_ids)


    else:
        # Marker is VCF
        header_samples = parse_vcf_get_header_samples(marker_path)

        if pheno_is_pkl:
            # P_obj from PKL; if string-keyed, intersect
            if isinstance(P_obj, dict) and all(isinstance(k, str) for k in P_obj.keys()):
                pheno_ids = list(P_obj.keys())
                common_ids = [s for s in pheno_ids if s in header_samples]
                if len(common_ids) == 0:
                    raise RuntimeError(
                        f"No common sample IDs between VCF {marker_path} and phenotype PKL {pheno_path}."
                    )
                if len(common_ids) < len(pheno_ids):
                    logging.info(
                        f"Phenotype PKL {pheno_path} has {len(pheno_ids)} entries, but only "
                        f"{len(common_ids)} have genotypes in VCF -> restricting to common samples."
                    )
                P_obj = {s: P_obj[s] for s in common_ids}
                sample_subset = common_ids
            else:
                # P_obj keyed by ints or other -> just parse all VCF samples
                sample_subset = None
        else:
            # Phenotype from CSV/TXT: intersect ID column with VCF header
            pheno_ids = list(df[id_col])
            common_ids = [s for s in pheno_ids if s in header_samples]
            if len(common_ids) == 0:
                raise RuntimeError(
                    f"No common sample IDs between VCF {marker_path} and phenotype file {pheno_path}."
                )
            if len(common_ids) < len(pheno_ids):
                logging.info(
                    f"Phenotype file {pheno_path} has {len(pheno_ids)} entries, but only "
                    f"{len(common_ids)} have genotypes in VCF -> restricting to common samples."
                )
            df_sub = df[df[id_col].isin(common_ids)].copy()
            df_sub = df_sub.set_index(id_col)
            P_obj = {
                sid: df_sub.loc[sid, pheno_trait_cols].to_numpy(dtype=np.float32)
                for sid in common_ids
            }
            sample_subset = common_ids
            
            # Store trait names in the P_obj
            P_obj.trait_names = trait_cols  # Add as attribute

        # Parse VCF for those samples
        G_obj, sample_map = parse_vcf_to_genotype_dict(
            marker_path,
            sample_subset=sample_subset,
            allowed_markers=None
        )
        save_pickle(G_obj, G_out)
        logging.info(f"Generated genotype PKL {G_out} from VCF {marker_path}")

    # Align and write
    G_int, P_int, sample_map = align_and_write_pkls(G_obj, P_obj, which, marker_is_pkl, pheno_is_pkl, result_dir=result_dir)
    logging.info(
        f"Aligned and wrote PKLs for {which}. "
        f"Saved G_{which}.pkl, P_{which}.pkl, sample_map_{which}.pkl"
    )
    return G_int, P_int, sample_map

def write_GP_from_two_vcfs_with_intersection(marker_train, pheno_train,
                                             marker_test,  pheno_test,
                                             result_dir: str,
                                             pheno_sep=None):
    """
    Special path when BOTH marker_train and marker_test are VCFs:
      - Compute SNP intersection by (CHROM, POS)
      - Restrict both train/test to common SNPs
      - For train and test separately:
          * Intersect VCF sample IDs with phenotype IDs
          * Build phenotype dicts sample_name -> trait vector
      - Align and write PKLs.

    Returns: (Gtrain, Ptrain, smtrain), (Gtest, Ptest, smtest)
    """
    marker_train = str(marker_train)
    marker_test  = str(marker_test)
    pheno_train  = str(pheno_train)
    pheno_test   = str(pheno_test)

    logging.info("Both marker_train and marker_test are VCF. "
                 "Finding common SNPs by (CHROM, POS).")

    # SNP intersection
    train_markers = get_vcf_marker_set(marker_train)
    test_markers  = get_vcf_marker_set(marker_test)
    common_markers = train_markers.intersection(test_markers)
    if len(common_markers) == 0:
        raise RuntimeError("No common SNPs between training and test VCFs.")

    logging.info(
        f"Train VCF SNPs: {len(train_markers)}, "
        f"Test VCF SNPs: {len(test_markers)}, "
        f"Common SNPs: {len(common_markers)}"
    )

    # VCF headers for sample IDs
    header_train_samples = parse_vcf_get_header_samples(marker_train)
    header_test_samples  = parse_vcf_get_header_samples(marker_test)

    # Phenotypes: train
    pheno_train_is_pkl = pheno_train.lower().endswith('.pkl')
    if pheno_train_is_pkl:
        P_train_obj = load_pickle(pheno_train)
        logging.info(f"Loaded phenotype PKL from {pheno_train}; will NOT overwrite P_train.pkl.")
        if isinstance(P_train_obj, dict) and all(isinstance(k, str) for k in P_train_obj.keys()):
            pheno_ids = list(P_train_obj.keys())
            common_ids_train = [s for s in pheno_ids if s in header_train_samples]
            if len(common_ids_train) == 0:
                raise RuntimeError(
                    f"No common sample IDs between train VCF {marker_train} and phenotype PKL {pheno_train}."
                )
            if len(common_ids_train) < len(pheno_ids):
                logging.info(
                    f"Train phenotype PKL {pheno_train} has {len(pheno_ids)} entries, but only "
                    f"{len(common_ids_train)} have genotypes in VCF -> restricting to common samples."
                )
            
            P_train_obj = {}
            for sid in common_ids_train:
                P_train_obj[sid] = df_tr_sub.loc[sid, common_trait_cols].to_numpy(dtype=np.float32)
            P_train_obj._trait_names = common_trait_cols

            # Do the same for P_test_obj
            P_test_obj = {}
            for sid in common_ids_test:
                P_test_obj[sid] = df_te_sub.loc[sid, common_trait_cols].to_numpy(dtype=np.float32)
            P_test_obj._trait_names = common_trait_cols

        else:
            common_ids_train = None
    else:
        df_tr, id_tr, trait_cols_tr = parse_pheno_file_to_df(pheno_train, sep=pheno_sep)
        pheno_ids_tr = list(df_tr[id_tr])
        common_ids_train = [s for s in pheno_ids_tr if s in header_train_samples]
        if len(common_ids_train) == 0:
            raise RuntimeError(
                f"No common sample IDs between train VCF {marker_train} and phenotype file {pheno_train}."
            )
        if len(common_ids_train) < len(pheno_ids_tr):
            logging.info(
                f"Train phenotype file {pheno_train} has {len(pheno_ids_tr)} entries, but only "
                f"{len(common_ids_train)} have genotypes in VCF -> restricting to common samples."
            )
        df_tr_sub = df_tr[df_tr[id_tr].isin(common_ids_train)].copy()
        df_tr_sub = df_tr_sub.set_index(id_tr)

    # Phenotypes: test
    pheno_test_is_pkl = pheno_test.lower().endswith('.pkl')
    if pheno_test_is_pkl:
        P_test_obj = load_pickle(pheno_test)
        logging.info(f"Loaded phenotype PKL from {pheno_test}; will NOT overwrite P_test.pkl.")
        if isinstance(P_test_obj, dict) and all(isinstance(k, str) for k in P_test_obj.keys()):
            pheno_ids = list(P_test_obj.keys())
            common_ids_test = [s for s in pheno_ids if s in header_test_samples]
            if len(common_ids_test) == 0:
                raise RuntimeError(
                    f"No common sample IDs between test VCF {marker_test} and phenotype PKL {pheno_test}."
                )
            if len(common_ids_test) < len(pheno_ids):
                logging.info(
                    f"Test phenotype PKL {pheno_test} has {len(pheno_ids)} entries, but only "
                    f"{len(common_ids_test)} have genotypes in VCF -> restricting to common samples."
                )
            P_test_obj = {s: P_test_obj[s] for s in common_ids_test}
        else:
            common_ids_test = None
    else:
        df_te, id_te, trait_cols_te = parse_pheno_file_to_df(pheno_test, sep=pheno_sep)
        pheno_ids_te = list(df_te[id_te])
        common_ids_test = [s for s in pheno_ids_te if s in header_test_samples]
        if len(common_ids_test) == 0:
            raise RuntimeError(
                f"No common sample IDs between test VCF {marker_test} and phenotype file {pheno_test}."
            )
        if len(common_ids_test) < len(pheno_ids_te):
            logging.info(
                f"Test phenotype file {pheno_test} has {len(pheno_ids_te)} entries, but only "
                f"{len(common_ids_test)} have genotypes in VCF -> restricting to common samples."
            )
        df_te_sub = df_te[df_te[id_te].isin(common_ids_test)].copy()
        df_te_sub = df_te_sub.set_index(id_te)

    # If phenotypes from CSV/TXT for both train and test, determine common trait columns
    if not pheno_train_is_pkl and not pheno_test_is_pkl:
        common_trait_cols = [c for c in trait_cols_tr if c in trait_cols_te]
        if len(common_trait_cols) == 0:
            raise RuntimeError(
                f"No common trait columns between train phenotype file {pheno_train} and "
                f"test phenotype file {pheno_test}."
            )
        logging.info(f"Common trait columns between train/test phenotypes: {common_trait_cols}")
        P_train_obj = {
            sid: df_tr_sub.loc[sid, common_trait_cols].to_numpy(dtype=np.float32)
            for sid in common_ids_train
        }
        P_test_obj = {
            sid: df_te_sub.loc[sid, common_trait_cols].to_numpy(dtype=np.float32)
            for sid in common_ids_test
        }

    # Parse VCFs restricted to common markers and common sample IDs
    G_train_obj, sm_train = parse_vcf_to_genotype_dict(
        marker_train,
        sample_subset=common_ids_train,
        allowed_markers=common_markers
    )
    save_pickle(G_train_obj, _outpath(result_dir, 'G_train.pkl'))

    G_test_obj, sm_test = parse_vcf_to_genotype_dict(
        marker_test,
        sample_subset=common_ids_test,
        allowed_markers=common_markers
    )
    save_pickle(G_test_obj, _outpath(result_dir, 'G_test.pkl'))

    logging.info("Generated G_train.pkl and G_test.pkl from VCFs restricted to common SNPs.")

    # Align and write PKLs
    Gtrain, Ptrain, smtrain = align_and_write_pkls(G_train_obj, P_train_obj, 'train', False, pheno_train_is_pkl, result_dir=result_dir)
    Gtest,  Ptest,  smtest  = align_and_write_pkls(G_test_obj,  P_test_obj,  'test',  False, pheno_test_is_pkl, result_dir=result_dir)

    logging.info("Aligned and wrote PKLs after common-SNP and common-sample filtering for train/test.")
    return (Gtrain, Ptrain, smtrain), (Gtest, Ptest, smtest)

# ----------------------------------------------------------------------
# Training + Testing with WheatGP_base (single or multi-trait)
# ----------------------------------------------------------------------

def split_genotype_data_from_dict(geno_dict):
    """
    Given G_{which}.pkl dict[int -> array_like], build 5 EQUAL group tensors.
    
    Returns: (groups_list, sorted_keys, n_markers)
        groups_list: [G1, G2, G3, G4, G5] each (n_samples, group_features)
        sorted_keys: sorted sample keys used for row order.
        n_markers: total number of SNP markers per sample.
    """
    keys = sorted(geno_dict.keys())
    if len(keys) == 0:
        raise RuntimeError("Genotype dictionary is empty.")
    
    arr = np.array([geno_dict[k] for k in keys], dtype=np.float32)
    nmarkers = arr.shape[1]
    
    # Calculate equal group sizes
    base_size = nmarkers // 5
    remainder = nmarkers % 5
    
    # Distribute remainder evenly or add to last group
    group_sizes = [base_size + (1 if i < remainder else 0) for i in range(5)]
    
    groups = []
    start = 0
    for i, size in enumerate(group_sizes):
        end = start + size
        groups.append(torch.tensor(arr[:, start:end], dtype=torch.float32))
        start = end
    
    # Verify total
    total_from_groups = sum(g.shape[1] for g in groups)
    if total_from_groups != nmarkers:
        raise RuntimeError(f"Group splitting error: {total_from_groups} != {nmarkers}")
    
    logging.info(f"Split {nmarkers} markers into groups: {group_sizes}")
    
    return groups, keys, nmarkers

def build_multi_trait_arrays(P_train, P_test, train_keys, test_keys):
    """
    Build (Y_train_all, Y_test_all) as 2D arrays [n_samples, n_traits]
    from P_train / P_test dicts.
    
    Returns:
        Y_train_all: np.ndarray (n_train, n_traits)
        Y_test_all:  np.ndarray (n_test, n_traits)
        n_traits:    int
    """
    train_vals = []
    for k in train_keys:
        v = np.asarray(P_train[k], dtype=np.float32).flatten()
        train_vals.append(v)
    test_vals = []
    for k in test_keys:
        v = np.asarray(P_test[k], dtype=np.float32).flatten()
        test_vals.append(v)

    train_vals = np.array(train_vals, dtype=np.float32)
    test_vals  = np.array(test_vals,  dtype=np.float32)

    if train_vals.ndim == 1:
        train_vals = train_vals.reshape(-1, 1)
    if test_vals.ndim == 1:
        test_vals = test_vals.reshape(-1, 1)

    if train_vals.shape[1] != test_vals.shape[1]:
        raise RuntimeError(
            f"Train and test phenotypes have different number of traits: "
            f"{train_vals.shape[1]} vs {test_vals.shape[1]}"
        )

    n_traits = train_vals.shape[1]
    return train_vals, test_vals, n_traits

def train_and_validate_single_trait(model, train_loader, val_loader,
                                    criterion, optimizer, scheduler,
                                    epochs, patience, device):
    """
    Train one WheatGP_base model for a single trait with early stopping based on
    validation Pearson's r.
    """
    best_P = float('-inf')
    best_mse = float('inf')
    best_mae = float('inf')
    best_model_state = None
    p = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for batch in train_loader:
            inputs = [b.to(device) for b in batch[:-1]]
            y_batch = batch[-1].to(device).view(-1, 1)

            optimizer.zero_grad()
            outputs = model(*inputs)
            loss = criterion(outputs, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item()

        epoch_train_loss = running_loss / max(1, len(train_loader))
        scheduler.step()

        model.eval()
        with torch.no_grad():
            Y_preds = []
            Y_trues = []
            val_running_loss = 0.0
            for batch in val_loader:
                inputs = [b.to(device) for b in batch[:-1]]
                y_val = batch[-1].to(device).view(-1, 1)
                outputs = model(*inputs)
                Y_preds.append(outputs)
                Y_trues.append(y_val)
                val_running_loss += criterion(outputs, y_val).item()

            if len(Y_preds) == 0:
                continue

            Y_preds = torch.cat(Y_preds).cpu().numpy().flatten()
            Y_trues = torch.cat(Y_trues).cpu().numpy().flatten()
            std_trues = np.std(Y_trues)
            std_preds = np.std(Y_preds)
            if std_trues == 0 or std_preds == 0:
                pearson_r = 0.0
            else:
                pearson_r = np.corrcoef(Y_trues, Y_preds)[0, 1]

            mae = F.l1_loss(
                torch.tensor(Y_preds, dtype=torch.float32),
                torch.tensor(Y_trues, dtype=torch.float32)
            ).item()
            mse = F.mse_loss(
                torch.tensor(Y_preds, dtype=torch.float32),
                torch.tensor(Y_trues, dtype=torch.float32)
            ).item()

            if pearson_r > best_P:
                best_P = pearson_r
                best_mse = mse
                best_mae = mae
                best_model_state = model.state_dict()
                p = 0
            else:
                p += 1

            if p >= patience:
                logging.info(f"Early stopping at epoch {epoch + 1}.")
                break

    logging.info(f"The best pearson_r (validation): {best_P}.")
    return best_P, best_mse, best_mae, best_model_state

def calculate_lstm_dim(group_sizes):
    """
    Calculate LSTM input dimension based on actual group sizes.
    
    Args:
        group_sizes: list of 5 integers representing markers per group
    """
    total_features = 0
    for size in group_sizes:
        # ConvPart reduces length by 4
        conv_output_length = size - 4
        features_per_group = conv_output_length * 8  # 8 output channels
        total_features += features_per_group
    
    return total_features

def extract_trait_names_from_parsing_logs(pheno_train_path, pheno_test_path):
    """
    Extract trait names from the parsing logs or directly from files.
    """
    # Try to parse the files directly to get trait names
    try:
        # Parse training file
        if pheno_train_path and os.path.exists(pheno_train_path):
            # Use your existing parse_pheno_file_to_df function
            df_train, id_col_train, train_trait_names = parse_pheno_file_to_df(pheno_train_path)
            return train_trait_names
    except:
        pass
    
    try:
        # Parse test file
        if pheno_test_path and os.path.exists(pheno_test_path):
            df_test, id_col_test, test_trait_names = parse_pheno_file_to_df(pheno_test_path)
            return test_trait_names
    except:
        pass
    
    return None


def run_prediction_flow(
    result_dir: str,
    G_train_pkl: str = 'G_train.pkl',
    P_train_pkl: str = 'P_train.pkl',
    G_test_pkl: str = 'G_test.pkl',
    P_test_pkl: str = 'P_test.pkl',
    pheno_train_path: str = None,  # Add these
    pheno_test_path: str = None  
):
    """
    Train + test WheatGP_base models using aligned PKLs.

    Assumes:
      - G_train.pkl / G_test.pkl are genotype dicts produced by
        align_and_write_pkls() / write_GP_from_two_vcfs_with_intersection().
      - P_train.pkl / P_test.pkl are phenotype dicts mapping sample_key -> scalar
        or 1D numpy array (multi-trait), as produced by the same utilities.

    This function:
      * Builds multi-trait phenotype matrices in the correct train/test sample order.
      * Implements a per-trait training loop (automatic multi-trait).
      * Uses a dynamic LSTM input size based on the number of markers:
            lstm_dim = 8 * (n_markers - 20)
      * Logs timing per trait and total runtime.
    """
    if torch is None:
        raise RuntimeError("Torch is required for prediction mode. Please install torch.")

    """
    # ------------------------------------------------------------------
    # Import WheatGP_base from local model/ directory
    #_________________-------------------------------------------------
    script_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.append(os.path.join(script_dir, 'model'))

    try:
        from WheatGP_base import wheatGP_base
    except Exception as e:
        raise RuntimeError(f"Could not import wheatGP_base from model/: {e}")
    """
    
    # ------------------------------------------------------------------
    # Load aligned PKLs
    # ------------------------------------------------------------------
    G_train = load_pickle(_outpath(result_dir, G_train_pkl))
    P_train = load_pickle(_outpath(result_dir, P_train_pkl))
    G_test  = load_pickle(_outpath(result_dir, G_test_pkl))
    P_test  = load_pickle(_outpath(result_dir, P_test_pkl))
    # Get trait names if available
    trait_names = []
    if hasattr(P_train, 'trait_names'):
        trait_names = P_train.trait_names
    elif hasattr(P_test, 'trait_names'):
        trait_names = P_test.trait_names


    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logging.info(f"Training & testing using device: {device}")

    # ------------------------------------------------------------------
    # Genotype splitting (groups correspond to split markers for conv/LSTM)
    # ------------------------------------------------------------------
    Gtrain_groups, train_keys, nmarkers_train = split_genotype_data_from_dict(G_train)
    Gtest_groups,  test_keys,  nmarkers_test  = split_genotype_data_from_dict(G_test)

    if nmarkers_train != nmarkers_test:
        raise RuntimeError(
            f"Train and test genotypes have different marker counts: "
            f"{nmarkers_train} vs {nmarkers_test}. They should share the same SNP set."
        )

    if nmarkers_train <= 20:
        raise RuntimeError(
            f"Number of markers ({nmarkers_train}) too small for ConvPart/LSTM pipeline."
        )

    # Dynamic LSTM input size derived from ConvPart/ShapeModule behaviour

    # Get group sizes from the first sample
    group_sizes = [g.shape[1] for g in Gtrain_groups]
    logging.info(f"Group sizes: {group_sizes}")
    
    # Calculate LSTM dimension based on actual group sizes
    lstm_dim = calculate_lstm_dim(group_sizes)
    logging.info(f"Calculated LSTM input dimension: {lstm_dim}")

    #n_markers = nmarkers_train
    #lstm_dim = calculate_lstm_dim(n_markers)
    
    logging.info(
        f"Dynamic LSTM input size: lstm_dim={lstm_dim} "
        f"(group size={group_sizes})"
    )

    # ------------------------------------------------------------------
    # Multi-trait phenotype matrices (shape: [n_samples, n_traits])
    # P_train / P_test are dicts: sample_key -> scalar or 1D array
    # ------------------------------------------------------------------
    Y_train_all, Y_test_all, n_traits = build_multi_trait_arrays(
        P_train, P_test, train_keys, test_keys
    )
    logging.info(f"Number of traits detected: {n_traits}")

    real_trait_names = extract_trait_names_from_parsing_logs(pheno_train_path, pheno_test_path)
    
    if real_trait_names and len(real_trait_names) == n_traits:
        trait_names = real_trait_names
        logging.info(f"Using real trait names: {trait_names}")
    else:
        trait_names = [f"trait{i}" for i in range(n_traits)]
        logging.info(f"Using default trait names: {trait_names}")
    
    
    # Use detected trait names if available
    if not trait_names:
        trait_names = [f"trait{i}" for i in range(n_traits)]
    
    logging.info(f"Number of traits detected: {n_traits}")
    logging.info(f"Trait names: {trait_names}")

    

    # ------------------------------------------------------------------
    # Load sample map for test set (for exporting per-sample predictions)
    # ------------------------------------------------------------------
    sample_map_test_path = _outpath(result_dir, "sample_map_test.pkl")
    if os.path.exists(sample_map_test_path):
        sample_map_test = load_pickle(sample_map_test_path)
    else:
        sample_map_test = None
        logging.warning("sample_map_test.pkl not found in result_dir; prediction exports will use test indices as sample IDs.")

    # Output files (NEW)
    accuracy_out = _outpath(result_dir, "wheatGP_accuracy_summary.tsv")
    pred_dir = _outpath(result_dir, "predictions_by_trait")
    os.makedirs(pred_dir, exist_ok=True)

    # Hyperparameters (kept consistent with original WheatGP code)
    learning_rate    = 0.005
    batch_size_train = 64
    batch_size_val   = 1
    weight_decay     = 0.0001
    epochs           = 300
    patience         = 50

    results = []
    global_start = time.time()

    for t in range(n_traits):
        trait_name = trait_names[t] if t < len(trait_names) else f"trait{t}"
        trait_start = time.time()
        logging.info(f"[Time] {trait_name} training started.")
        logging.info(f"=== Trait {t+1}/{n_traits} [{trait_name}] ===")

        # Raw trait vectors (including potential NaNs)
        y_train_full = Y_train_all[:, t]
        y_test_full  = Y_test_all[:, t]

        # Drop NaN phenotypes and log diagnostics
        train_mask = np.isfinite(y_train_full)
        test_mask  = np.isfinite(y_test_full)

        n_train_total = len(y_train_full)
        n_test_total  = len(y_test_full)
        n_train_used  = int(train_mask.sum())
        n_test_used   = int(test_mask.sum())

        logging.info(
            f"[{trait_name}] Train samples: total={n_train_total}, "
            f"used={n_train_used}, NaNs={n_train_total - n_train_used}"
        )
        logging.info(
            f"[{trait_name}] Test samples: total={n_test_total}, "
            f"used={n_test_used}, NaNs={n_test_total - n_test_used}"
        )

        if n_train_used < 2 or n_test_used < 2:
            logging.warning(
                f"[Trait {trait_name}] Too few non-NaN samples after filtering "
                f"(train={n_train_used}, test={n_test_used}). Skipping this trait."
            )
            results.append((t, float("nan"), float("nan"), float("nan")))
            continue

        # Filter phenotypes
        y_train = y_train_full[train_mask]
        y_test  = y_test_full[test_mask]

        # Filter genotypes to same indices
        train_idx = np.where(train_mask)[0]
        test_idx  = np.where(test_mask)[0]

        idx_train_t = torch.as_tensor(train_idx, dtype=torch.long)
        idx_test_t  = torch.as_tensor(test_idx, dtype=torch.long)

        train_G_tensors = [g.index_select(0, idx_train_t).to(device) for g in Gtrain_groups]
        test_G_tensors  = [g.index_select(0, idx_test_t).to(device) for g in Gtest_groups]

        # Log basic stats for diagnostics
        logging.info(
            f"[Trait {trait_name}] Train y stats: "
            f"mean={float(np.nanmean(y_train_full)):.4f}, "
            f"std={float(np.nanstd(y_train_full)):.4f}, "
            f"min={float(np.nanmin(y_train_full)):.4f}, "
            f"max={float(np.nanmax(y_train_full)):.4f}"
        )
        logging.info(
            f"[Trait {trait_name}] Test y stats: "
            f"mean={float(np.nanmean(y_test_full)):.4f}, "
            f"std={float(np.nanstd(y_test_full)):.4f}, "
            f"min={float(np.nanmin(y_test_full)):.4f}, "
            f"max={float(np.nanmax(y_test_full)):.4f}"
        )

        # ------------------------------------------------------------------
        # Build training dataset/loaders (using filtered indices)
        # ------------------------------------------------------------------
        train_Y_tensor = torch.tensor(y_train, dtype=torch.float32).to(device)
        train_dataset  = TensorDataset(*train_G_tensors, train_Y_tensor)

        dataset_size = len(train_dataset)
        train_size   = int(0.9 * dataset_size)
        val_size     = dataset_size - train_size
        train_ds, val_ds = random_split(
            train_dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(42),
        )

        train_loader = DataLoader(train_ds, batch_size=batch_size_train, shuffle=True)
        val_loader   = DataLoader(val_ds, batch_size=batch_size_val, shuffle=False)

        # ------------------------------------------------------------------
        # Model, loss, optimizer, scheduler
        # ------------------------------------------------------------------
        model = wheatGP_base(lstm_dim).to(device)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(
            model.parameters(), lr=learning_rate, weight_decay=weight_decay
        )
        scheduler = StepLR(optimizer, step_size=90, gamma=0.1)

        logging.info("Training stage")
        best_P, best_mse, best_mae, best_model_state = train_and_validate_single_trait(
            model, train_loader, val_loader,
            criterion, optimizer, scheduler,
            epochs, patience, device,
        )

        # Save best model for this trait
        best_model_path = _outpath(result_dir, f"best_model_{trait_name}.ckpt")
        torch.save(best_model_state, best_model_path)
        logging.info(f"Saved best model to {best_model_path}")

        # ------------------------------------------------------------------
        # Test set evaluation (using filtered test indices)
        # ------------------------------------------------------------------
        model.load_state_dict(best_model_state)
        model.eval()

        test_Y_tensor = torch.tensor(y_test, dtype=torch.float32).to(device)
        test_dataset  = TensorDataset(*test_G_tensors, test_Y_tensor)
        test_loader   = DataLoader(test_dataset, batch_size=1, shuffle=False)

        with torch.no_grad():
            Y_preds = []
            Y_trues = []
            for batch in test_loader:
                inputs = [b.to(device) for b in batch[:-1]]
                y_true_batch = batch[-1].to(device).view(-1, 1)
                outputs = model(*inputs)
                Y_preds.append(outputs)
                Y_trues.append(y_true_batch)

            if len(Y_preds) == 0:
                logging.warning(f"[Trait {trait_name}] No test predictions produced.")
                results.append((t, float("nan"), float("nan"), float("nan")))
                continue

            Y_preds = torch.cat(Y_preds).cpu().numpy().flatten()
            Y_trues = torch.cat(Y_trues).cpu().numpy().flatten()


        # --------------------------------------------------------------
        # NEW: Export per-sample predictions (GEBVs) with observed values
        # --------------------------------------------------------------
        # Map the filtered test indices back to sample IDs
        if sample_map_test is not None and len(sample_map_test) == len(test_keys):
            sample_ids_used = [str(sample_map_test[i]) for i in test_idx]
        else:
            # Fallback: use test_keys (often int indices)
            sample_ids_used = [str(test_keys[i]) for i in test_idx]

        pred_df = pd.DataFrame({
            "Sample": sample_ids_used,
            "Observed": Y_trues.astype(float),
            "GEBV": Y_preds.astype(float)
        })
        pred_out = os.path.join(pred_dir, f"wheatGP_trait_{trait_name}_predictions.tsv")
        pred_df.to_csv(pred_out, sep="\t", index=False)
        logging.info(f"[Trait {trait_name}] Saved predictions table: {pred_out}")



        # Extra diagnostics for problematic traits
        logging.info(
            f"[Trait {trait_name}] Pred y stats: "
            f"mean={float(np.mean(Y_preds)):.4f}, "
            f"std={float(np.std(Y_preds)):.4f}, "
            f"min={float(np.min(Y_preds)):.4f}, "
            f"max={float(np.max(Y_preds)):.4f}"
        )

        std_trues = np.std(Y_trues)
        std_preds = np.std(Y_preds)

        if std_trues < 1e-8 or std_preds < 1e-8:
            logging.warning(
                f"[Trait {trait_name}] Zero or near-zero variance encountered "
                f"(std_trues={std_trues:.4e}, std_preds={std_preds:.4e}). "
                f"Setting Pearson r = 0.0."
            )        

            r = 0.0
        else:
            r = float(np.corrcoef(Y_trues, Y_preds)[0, 1])

        mse = float(np.mean((Y_trues - Y_preds) ** 2))
        mae = float(np.mean(np.abs(Y_trues - Y_preds)))

        logging.info(
            f"Prediction results (test set, {trait_name}): "
            f"Pearson r={r:.4f}  MSE={mse:.6f}  MAE={mae:.6f}"
        )
        
        results.append((trait_name, r, mse, mae))

        trait_end = time.time()
        elapsed = trait_end - trait_start
        logging.info(
            f"[Time] {trait_name} finished. "
            f"Elapsed time: {elapsed/60:.2f} min ({elapsed:.1f} sec)"
        )

    # Summary table at the end
    logging.info("\n" + "="*80)
    logging.info("SUMMARY OF PREDICTION RESULTS")
    logging.info("="*80)
    for trait_name, r, mse, mae in results:
        logging.info(f"{trait_name:20s} | Pearson r: {r:7.4f} | MSE: {mse:10.6f} | MAE: {mae:10.6f}")
    logging.info("="*80)

    global_end = time.time()
    total_elapsed = global_end - global_start
    logging.info(
        f"[Time] All traits finished. Total elapsed time: "
        f"{total_elapsed/60:.2f} min ({total_elapsed:.1f} sec)"
    )



    # --------------------------------------------------------------
    # NEW: Write accuracy summary table to result_dir
    # --------------------------------------------------------------
    try:
        acc_df = pd.DataFrame(results, columns=["Trait", "PearsonR", "MSE", "MAE"])
        acc_df.to_csv(accuracy_out, sep="\t", index=False)
        logging.info(f"Saved accuracy summary table: {accuracy_out}")
    except Exception as e:
        logging.warning(f"Failed to write accuracy summary table: {e}")

    return results

# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    # Set reproducibility seed
    set_global_seed(42)
    pipeline_start_time = time.time()

    p = argparse.ArgumentParser(
        description=(
            "WheatGP pipeline v4: convert / prediction with robust PKL alignment, "
            "VCF common-SNP & common-sample handling, and automatic multi-trait support."
        )
    )
    p.add_argument('--mode', choices=['convert', 'prediction'], required=True)
    p.add_argument('--marker_train', required=True, help='VCF (.vcf/.vcf.gz) or PKL (.pkl)')
    p.add_argument('--pheno_train',  required=True, help='CSV/TXT (multi-trait) or PKL (.pkl)')
    p.add_argument('--marker_test',  required=True, help='VCF (.vcf/.vcf.gz) or PKL (.pkl)')
    p.add_argument('--pheno_test',   required=True, help='CSV/TXT (multi-trait) or PKL (.pkl)')
    p.add_argument('--result_dir', required=True, help='Directory to store all results/outputs (PKLs, models, logs, prediction tables)')
    p.add_argument('--pheno_sep', default=None,
                   help='Optional separator for phenotype CSV/TXT (e.g. ",", "\\t"). '
                        'If omitted: CSV -> ",", others -> whitespace.')
    args = p.parse_args()
    # Initialize result directory and logging (NEW)
    result_dir = os.path.abspath(args.result_dir)
    os.makedirs(result_dir, exist_ok=True)
    init_logging(result_dir)



    marker_train_is_vcf = args.marker_train.lower().endswith('.vcf') or args.marker_train.lower().endswith('.vcf.gz')
    marker_test_is_vcf  = args.marker_test.lower().endswith('.vcf')  or args.marker_test.lower().endswith('.vcf.gz')

    try:
        if marker_train_is_vcf and marker_test_is_vcf:
            (Gtrain, Ptrain, smtrain), (Gtest, Ptest, smtest) = write_GP_from_two_vcfs_with_intersection(
                args.marker_train, args.pheno_train,
                args.marker_test,  args.pheno_test,
                result_dir=result_dir,
                pheno_sep=args.pheno_sep
            )
        else:
            Gtrain, Ptrain, smtrain = write_GP_from_inputs(
                args.marker_train, args.pheno_train, 'train', result_dir=result_dir, pheno_sep=args.pheno_sep
            )
            Gtest,  Ptest,  smtest  = write_GP_from_inputs(
                args.marker_test,  args.pheno_test,  'test',  result_dir=result_dir, pheno_sep=args.pheno_sep
            )
    except Exception as e:
        logging.error("Conversion/alignment failed: " + str(e))
        sys.exit(1)

    logging.info(
        "PKL conversion/alignment complete. Files in result_dir: "
        "G_train.pkl, P_train.pkl, sample_map_train.pkl, "
        "G_test.pkl, P_test.pkl, sample_map_test.pkl"
    )

    if args.mode == 'convert':
        logging.info("Mode 'convert' finished. Exiting.")
        return

    if args.mode == 'prediction':
        logging.info("Running prediction (train+test) using PKLs.")
        try:
            run_prediction_flow(result_dir, 'G_train.pkl', 'P_train.pkl', 'G_test.pkl', 'P_test.pkl', pheno_train_path=args.pheno_train, pheno_test_path=args.pheno_test)
        except Exception as e:
            logging.error("Prediction flow failed: " + str(e))
            raise

    total_time = time.time() - pipeline_start_time
    logging.info(f"[Time] Total pipeline runtime: "
                 f"{total_time/60:.2f} min ({total_time:.1f} sec)")

if __name__ == '__main__':
    main()

