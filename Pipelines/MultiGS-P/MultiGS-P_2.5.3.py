# ===============================================================
# MultiGS-P 
# A Genomic Selection Pipeline for Multiple Single Traits 
# Using Diverse Machine Learning and Deep Learning Models 
# and Marker Types
# Supports: R_RRBLUP, R_GBLUP, RR_BLUP, BRR,
#           ElasticNet, RandomForest Regression (RFR), XGBoost, LightGBM,
#           HybridAttnMLP, DNNGS, GraphConvGS, GraphAttnGS, GraphSAGEGS (three implementationss of PyG GNN), GraphFormer      
#           FusionGS, EfficientGSFormer,
#           EnsembleGS, DeepBLUP, DeepResBLUP 
#           
# Single trait models for multiple traits
# Input: raw SNP vcf 
# Three feature views: SNPs (SNP), haplotypes (HAP), PCs (PC)
# Two Modes: Cross-validation (CV), Across-population Prediction (APP)
# Nov, 2025
# Frank YOU
# Agriculture and Agri-Food Canada
# ===============================================================

# Global timing dictionary to access performance
training_times = {}  # Structure: {model_name: {marker_view: [time1, time2, ...]}}
training_memories ={}  # Structure: {model_name: {marker_view: [used_memory1, used_memory2, ...]}}

import io
import os
import gc
import json
import argparse
import math
import configparser
import sys
import atexit
from datetime import datetime
import time
from collections import defaultdict
import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold, train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.linear_model import Ridge, ElasticNet, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.cross_decomposition import PLSRegression
from sklearn.neighbors import NearestNeighbors
from scipy.stats import pearsonr, spearmanr, chi2, skew, kurtosis, norm
import matplotlib.pyplot as plt

#=======================================================================
# R related packages
try:
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr
    from rpy2.robjects import default_converter
    from rpy2.robjects.conversion import localconverter
    
    # Import the converters but don't use activate()
    from rpy2.robjects import numpy2ri, pandas2ri
    
    R_AVAILABLE = True
    print("rpy2 successfully imported - using modern conversion API")
    
except ImportError:
    R_AVAILABLE = False
    print("rpy2 not available - R models will use Python fallback")
except Exception as e:
    R_AVAILABLE = False
    print(f"rpy2 import warning: {e} - R models will use Python fallback")

#-----------------------------------------------------------------------    
# hidel R callback and warning message: this is working well
import rpy2.rinterface_lib.callbacks
import logging

# Disable rpy2 callbacks
rpy2.rinterface_lib.callbacks.logger.setLevel(logging.ERROR)

def setup_r_silent_mode():
    """Setup R to run in silent mode"""
    if not R_AVAILABLE:
        return
    try:
        ro.r('options(warn=-1); options(show.error.messages=FALSE)')
    except:
        pass

# Call this after R packages are loaded
setup_r_silent_mode()
#-----------------------------------------------------------------------    

# Bayesian models (optional)
try:
    from sklearn.linear_model import BayesianRidge
    BAYESIAN_AVAILABLE = True
except ImportError:
    BAYESIAN_AVAILABLE = False

# For BayesA (if not available, we'll implement a simple version)
try:
    import pystan
    STAN_AVAILABLE = True
except ImportError:
    STAN_AVAILABLE = False

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# remove overhead and warning
import os
os.environ['LIGHTGBM_VERBOSE'] = '-1'  # or '0' depending on version

try:
    import lightgbm as lgb
    LGBM_AVAILABLE = True
except ImportError:
    LGBM_AVAILABLE = False


# Add these imports with the other deep learning imports
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# for GraphConvGS, GraphAttnGS, GraphSAGEGS in PyG GNN

try:
    import torch_geometric
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False

# Add to the warnings filter
import warnings
warnings.filterwarnings(
    "ignore",
    message="X does not have valid feature names, but LGBMRegressor was fitted with feature names",
    category=UserWarning
)
warnings.filterwarnings("ignore", category=UserWarning, module="torch")

# Add to the warnings filter section
warnings.filterwarnings(
    "ignore",
    message="'GT' FORMAT header not found",
    category=UserWarning,
    module="allel.io.vcf_read"
)

# silence warnings
warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
)

# ADD THESE LINES - STRING-BASED FILTERING (NO IMPORT NEEDED)
warnings.filterwarnings(
    "ignore",
    message="An input array is constant; the correlation coefficient is not defined."
)

warnings.filterwarnings(
    "ignore", 
    message="Polyfit may be poorly conditioned"
)

import warnings

import warnings

# --- Universal suppression for SciPy pearsonr warnings ---
try:
    # For SciPy >= 1.9 (new warning classes)
    from scipy.stats import PearsonRConstantInputWarning, NearConstantInputWarning
except ImportError:
    # For older SciPy versions — define dummy placeholders so import won't fail
    class NearConstantInputWarning(Warning):
        pass
    class PearsonRConstantInputWarning(Warning):
        pass

warnings.filterwarnings("ignore", message="DataFrame is highly fragmented")

# Now safely ignore them (works with old or new SciPy)
warnings.filterwarnings("ignore", category=NearConstantInputWarning)
warnings.filterwarnings("ignore", category=PearsonRConstantInputWarning)

# (optional) Hide other generic warnings too
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# silence DEBUG
class FilterDebug(io.StringIO):
    def __init__(self):
        super().__init__()
        self.consecutive_newlines = 0
        self.max_consecutive_newlines = 2  # Allow up to 2 blank lines
    
    def write(self, s):
        if "[DEBUG]" in s:
            return
        
        # Check if this is just whitespace/newlines
        if s.strip() == "":
            self.consecutive_newlines += 1
            if self.consecutive_newlines <= self.max_consecutive_newlines:
                sys.__stdout__.write(s)
        else:
            self.consecutive_newlines = 0
            sys.__stdout__.write(s)

# silence R warning message
import rpy2.rinterface_lib.callbacks
import sys

def custom_warn(x):
    if "Warning" not in x:
        sys.__stdout__.write(x)


# -----------------------------
# Helper functions
# -------------------------

import psutil
def get_memory_usage_mb():
    process = psutil.Process(os.getpid())
    mem_bytes = process.memory_info().rss  # Resident Set Size in bytes
    mem_mb = mem_bytes / (1024 ** 2)
    return mem_mb

def strip_inline_comment(val: str):
    if val is None:
        return val
    return str(val).split('#', 1)[0].strip()

def parse_int(cfg, sec, key, default):
    raw = cfg.get(sec, key, fallback=str(default))
    return int(strip_inline_comment(raw))

def parse_float(cfg, sec, key, default):
    raw = cfg.get(sec, key, fallback=str(default))
    return float(strip_inline_comment(raw))

def parse_list_of_ints(cfg, sec, key, default_list):
    raw = cfg.get(sec, key, fallback=",".join(map(str, default_list)))
    clean = strip_inline_comment(raw)
    return [int(x.strip()) for x in clean.split(',') if x.strip()]

def parse_list_of_floats(cfg, sec, key, default_list):
    raw = cfg.get(sec, key, fallback=",".join(map(str, default_list)))
    clean = strip_inline_comment(raw)
    return [float(x.strip()) for x in clean.split(',') if x.strip()]

def set_random_seeds(seed):
    np.random.seed(seed)
    # If you're not using PyTorch, you can remove the torch-related code

# -----------------------------
# Logging Tee Class
# -----------------------------
class Tee:
    def __init__(self, *streams):
        self.streams = streams
        
    def write(self, s):
        for st in self.streams:
            st.write(s)
            st.flush()
            
    def flush(self):
        for st in self.streams:
            st.flush()

# -----------------------------
# Configuration
# -----------------------------
import configparser
import re
def parse_config(config_path):
    # Disable interpolation to avoid % issues
    cfg = configparser.ConfigParser(interpolation=None)
    cfg.read(config_path)
    
    run_mode = 'prediction' if (
        cfg.has_option('Data', 'test_vcf_path') 
    ) else 'cross_validation'
    
    # Helper function to safely get values
    def safe_get(cfg, section, key, default):
        try:
            value = cfg.get(section, key, fallback=str(default))
            # Remove any inline comments
            value = str(value).split('#', 1)[0].strip()
            return value
        except Exception:
            return str(default)
    
    # -----------------------------
    # General parameters
    # -----------------------------
    params = {
        'seed': int(safe_get(cfg, 'General', 'seed', 42)),
        'n_replicates': int(safe_get(cfg, 'General', 'n_replicates', 3)),
        'n_folds': int(safe_get(cfg, 'General', 'n_folds', 5)),
        'results_dir': safe_get(cfg, 'General', 'results_dir', 'results'),
        'threads': int(safe_get(cfg, 'General', 'threads', 10)),
        'R_path': safe_get(cfg, 'General', 'R_path', 10),
        
        
        'vcf_path': safe_get(cfg, 'Data', 'vcf_path', ''),
        'phenotype_path': safe_get(cfg, 'Data', 'phenotype_path', ''),
        'n_pca_components': int(safe_get(cfg, 'Data', 'n_pca_components', 10000)),
        'pca_variance_explained': float(safe_get(cfg, 'Data', 'pca_variance_explained', 0.0)),
        'pheno_normalization': safe_get(cfg, 'Data', 'pheno_normalization', 'standard'),
        'genotype_normalization': safe_get(cfg, 'Data', 'genotype_normalization', 'none'),
        'impute_scope': safe_get(cfg, 'Data', 'impute_scope', 'train').lower(),
        'pca_fit_scope': safe_get(cfg, 'Data', 'pca_fit_scope', 'train').lower(),

        'rtm-gwas-snpldb_path': safe_get(cfg, 'Tool', 'rtm-gwas-snpldb_path', ''),
        'feature_views': [view.strip() for view in safe_get(cfg, 'FeatureView', 'feature_view', 'SNP').split(',')],
        #'feature_view': safe_get(cfg, 'FeatureView', 'feature_view', 'SNP'),
    }
    
    params['feature_view'] = params['feature_views'][0]     # sign the first marker eview as default


    # -----------------------------
    # Test data (if prediction mode)
    # -----------------------------
    if run_mode == 'prediction':
        params.update({
            'test_vcf_path': safe_get(cfg, 'Data', 'test_vcf_path', ''),
            'test_phenotype_path': safe_get(cfg, 'Data', 'test_phenotype_path', '')
        })
    
    # -----------------------------
    # Model selection
    # -----------------------------
    MODEL_CANDIDATES = ['R_RRBLUP', 'R_GBLUP', 'RRBLUP',
                        'ElasticNet',  
                        'RFR', 'BRR',
                        'XGBoost', 'LightGBM', 
                        #'CNN', 
                        'DNNGS',
                        'MLPGS','HybridAttnMLP',
                        'GraphConvGS','GraphAttnGS', 'GraphSAGEGS',
                        'GraphFormer',
                        #'Transformer',
                        'DeepResBLUP',
                        'DeepBLUP',
                        'FusionGS',
                        'EfficientGSFormer',
                        'EnsembleGS', # always list in last one so that all previous models have been trained
                        ]
    
    params['enabled_models'] = []

    if cfg.has_section('Models'):
        for m in MODEL_CANDIDATES:
            if cfg.has_option('Models', m):
                val = safe_get(cfg, 'Models', m, 'False').lower()
                if val in ['true', '1', 'yes', 'on']:
                    params['enabled_models'].append(m)
    else:
        # Fallback: if no [Models] section, enable all by default
        if not params['enabled_models']:
            print("WARNING: No models enabled. Please enable at least one model in the [Models] section of your config file.")
            print("Available models:", ', '.join(MODEL_CANDIDATES))
            sys.exit(1)  # Exit with error code 1

    # -----------------------------
    # Collect model-specific hyperparameters
    # -----------------------------
    hyperparams = {}
    for m in MODEL_CANDIDATES:
        section = f'Hyperparameters_{m}'
        if cfg.has_section(section):
            hyperparams[m] = {}
            for key, val in cfg.items(section):
                # Fix: use maxsplit keyword argument instead of positional
                val = re.split(r'[;#]', str(val), maxsplit=1)[0].strip()
                hyperparams[m][key] = val
    
    params['hyperparameters'] = hyperparams
    params['run_mode'] = run_mode
    # Debug: show parsed enabled models
    #print('[CFG] enabled_models:', params['enabled_models'])


    if 'EnsembleGS' in params['enabled_models']:
        print("EnsembleGS detected - ensuring all specified base models are enabled...")
        
        # Get EnsembleGS base models from hyperparameters
        ensemble_hyperparams = params['hyperparameters'].get('EnsembleGS', {})
        base_models_str = ensemble_hyperparams.get('base_models', 'RRBLUP,RFR,XGBoost')
        base_models = [model.strip() for model in base_models_str.split(',')]
        
        # Remove duplicates and empty strings
        base_models = list(dict.fromkeys([bm for bm in base_models if bm]))
        
        if not base_models:
            print("  ERROR: No base models specified for EnsembleGS in hyperparameters")
            print("  Please set base_models in [Hyperparameters_EnsembleGS] section")
            sys.exit(1)
        
        # Store base models for EnsembleGS
        params['ensemble_base_models'] = base_models
        
        # Check each base model and add to enabled_models if missing
        added_models = []
        invalid_models = []
        
        for base_model in base_models:
            if base_model not in MODEL_CANDIDATES:
                invalid_models.append(base_model)
                print(f"  ERROR: Base model '{base_model}' is not in MODEL_CANDIDATES")
            elif base_model not in params['enabled_models']:
                params['enabled_models'].append(base_model)
                added_models.append(base_model)
                print(f"  ✓ Added '{base_model}' to enabled models")
            else:
                print(f"  ✓ '{base_model}' is already enabled")
        
        # Report results
        if added_models:
            print(f"  Added {len(added_models)} models to enabled models: {', '.join(added_models)}")
        
        if invalid_models:
            print(f"  ERROR: {len(invalid_models)} invalid base models: {', '.join(invalid_models)}")
            print(f"  Available models: {', '.join(MODEL_CANDIDATES)}")
            sys.exit(1)
        
        # Final summary
        enabled_base_count = len([bm for bm in base_models if bm in params['enabled_models']])
        print(f"  EnsembleGS configuration:")
        print(f"    - Specified base models: {', '.join(base_models)}")
        print(f"    - Enabled base models: {enabled_base_count}/{len(base_models)}")
        print(f"    - Total enabled models: {len(params['enabled_models'])}")
        
        if enabled_base_count < 2:
            print("  WARNING: EnsembleGS has fewer than 2 enabled base models")

    # Add quiet mode
    params['quiet_mode'] = safe_get(cfg, 'General', 'quiet_mode', 'false').lower() == 'true'

    return params

def get_default_hyperparams(model):
    """Return default hyperparameters for each model"""

    MODEL_CANDIDATES = ['R_RRBLUP', 'R_GBLUP', 'RRBLUP',
                        'ElasticNet',  
                        'RFR', 'BRR',
                        'XGBoost', 'LightGBM', 
                        #'CNN', 
                        'DNNGS',
                        'MLPGS','HybridAttnMLP',
                        'GraphConvGS','GraphAttnGS', 'GraphSAGEGS',
                        'GraphFormer',
                        #'Transformer',
                        'DeepResBLUP',
                        'DeepBLUP',
                        'FusionGS',
                        'EfficientGSFormer',
                        'EnsembleGS', # always list in last one so that all previous models have been trained
                        ]
    
    defaults = {
        'ElasticNet': {'alpha': '1.0', 'l1_ratio': '0.5'},
        'LASSO': {'alpha': '1.0'},
        'RFR': {'n_estimators': '100', 'max_depth': 'None'},
        'BRR': {'alpha_1': '1e-6', 'alpha_2': '1e-6', 'lambda_1': '1e-6', 'lambda_2': '1e-6'},
        'RRBLUP': {
            'lambda_value': 'None',      # Let the model estimate optimal lambda
            'method': 'mixed_model',     # 'mixed_model' or 'direct'
            'lambda_method': 'auto',     # NEW: 'auto', 'reml', 'heritability', 'fixed'
            'tol': '1e-8'
        },
        'R_RRBLUP': {
            'method': 'ML',  # REML, ML
        },
        #'R_GBLUP': {
        #    'method': 'REML',
        #},
        'DeepResBLUP': {
            'base_model': 'RRBLUP',
            'dl_model': 'HybridAttnMLP',
            'dl_hidden_layers': '128,64',
            'dl_dropout': '0.2',
            'dl_learning_rate': '0.001',
            'dl_batch_size': '32',
            'dl_epochs': '100',
            'weight_ratio': '0.5'  # Optional: weighting between base and DL model
        },
        'XGBoost': {'n_estimators': '100', 'max_depth': '6', 'learning_rate': '0.1'},
        'LightGBM': {'n_estimators': '100', 'max_depth': '-1', 'learning_rate': '0.1', 'verbose': '-1'},
        'CNN': {'hidden_channels': '64,32', 'kernel_size': '3', 'pool_size': '2',
            'learning_rate': '0.001', 'batch_size': '32', 'epochs': '100',
            'dropout': '0.2'},
        'HybridAttnMLP': {
            # --- Architecture ---
            'cnn_channels': '64,128',
            'kernel_size': '5',
            'pool_size': '2',
            'attention_heads': '4',
            'hidden_size': '128',
            'mlp_hidden': '128,64',
            'activation': 'gelu',
            'norm': 'layer',
            'residual': 'true',
            'dropout': '0.3',
            'input_dropout': '0.05',
            # --- Optimization ---
            'learning_rate': '0.0005',
            'weight_decay': '0.0015',
            'batch_size': '16',
            'epochs': '300',
            'early_stopping_patience': '20',
            'warmup_ratio': '0.1',
            'grad_clip': '1.0',
            'seeds': '3',
            # --- Loss ---
            'use_huber': 'true',
            'huber_delta': '1.0',
            # --- SWA ---
            'swa': 'true',
            'swa_start': '0.7',
            'swa_freq': '1'
        },
        'DNNGS': {'hidden_layers': '512,256,128,64', 'learning_rate': '0.001', 
            'batch_size': '32', 'epochs': '150', 'dropout': '0.3',
            'activation': 'relu', 'batch_norm': 'true', 'weight_decay': '0.0001'}, 
        'Transformer': {'d_model': '512', 'nhead': '8', 'num_layers': '6',
            'dim_feedforward': '2048', 'dropout': '0.1',
            'learning_rate': '0.0001', 'batch_size': '32', 'epochs': '200',
            'weight_decay': '0.01', 'warmup_ratio': '0.1', 'patience': '30',
            'grad_clip': '1.0'},
        'EnsembleGS': {'base_models': 'BRR,XGBoost', 'meta_model': 'LinearRegression'},
        'GraphConvGS': {
            'hidden_channels': '128',
            'num_layers': '2',
            'hidden_mlp': '128',
            'dropout': '0.2',
            'learning_rate': '0.0005',
            'batch_size': '32',
            'epochs': '200',
            'weight_decay': '0.0',
            'top_k': '20',
            'graph_method': 'knn',  # New parameter
            'knn_metric': 'euclidean',  # New parameter
            'adj_threshold': '',  # optional absolute-corr threshold; blank = ignored
            'patience': '20'
        },
        'GraphAttnGS': {
            'hidden_channels': '128',
            'num_layers': '2',
            'heads': '4',
            'hidden_mlp': '128',
            'dropout': '0.2',
            'learning_rate': '0.0005',
            'batch_size': '32',
            'epochs': '200',
            'weight_decay': '0.0',
            'top_k': '20',
            'graph_method': 'knn',
            'knn_metric': 'euclidean',
            'adj_threshold': '',
            'patience': '20'
        },
        'GraphSAGEGS': {
            'hidden_channels': '128',
            'num_layers': '2',
            'hidden_mlp': '128',
            'dropout': '0.2',
            'learning_rate': '0.0005',
            'batch_size': '32',
            'epochs': '200',
            'weight_decay': '0.0',
            'top_k': '20',
            'graph_method': 'knn',
            'knn_metric': 'euclidean',
            'adj_threshold': '',
            'patience': '20'
        },
        'GraphFormer': {
            'sage_hidden': '128',
            'sage_layers': '2',
            'd_model': '128',
            'transformer_layers': '2',
            'nhead': '4',
            'mlp_hidden': '128',
            'dropout': '0.1',
            'learning_rate': '0.0005',
            'batch_size': '1',
            'epochs': '200',
            'weight_decay': '0.0',
            'top_k': '20',
            'graph_method': 'knn',
            'knn_metric': 'euclidean',
            'patience': '20'
        },

        'DeepBLUP': {
            'rrblup_lambda': '0.001',  # Reduced regularization
            'dl_hidden_layers': '256,128,64',  # Smaller network
            'dropout': '0.3',  # More dropout
            'activation': 'gelu',  # More stable activation
            'use_precomputed_rrblup': 'true',  # Use precomputed for stability
            'train_rrblup_layer': 'true',  # Keep RRBLUP fixed initially
            'use_skip_connection': 'true',  # Enable skip connections
            'use_batch_norm': 'true',  # Enable batch norm
            'learning_rate': '0.0001',  # Smaller learning rate
            'batch_size': '16',  # Smaller batches
            'epochs': '200',  # More epochs with early stopping
            'weight_decay': '0.001',
            'use_batch_norm': 'true',
            'use_residual_connections': 'true'
        },
        'FusionGS': {
            # CNN branch
            'cnn_channels': '32,64',
            'kernel_size': '5',
            'pool_size': '2',
            # MLP branch
            'mlp_hidden': '256,128',
            # Fusion & regularization
            'fusion_hidden': '128',
            'dropout': '0.3',
            'input_dropout': '0.05',
            # Training
            'learning_rate': '0.00075',
            'batch_size': '64',
            'epochs': '200',
            'weight_decay': '0.0005',
            'warmup_ratio': '0.1',
            'patience': '20'
        },

        'EfficientGSFormer': {
            # Patch-based transformer
            'd_model': '128',
            'nhead': '4',
            'num_layers': '2',
            'dim_feedforward': '256',
            'dropout': '0.1',
            'patch_size': '64',
            # Training
            'learning_rate': '0.0005',
            'batch_size': '32',
            'epochs': '200',
            'weight_decay': '0.0005',
            'patience': '20'
        },

    }
    return defaults.get(model, {})

# -----------------------------
# Data Loading and Processing
# -----------------------------
#=================================================
#This whole section is about SNP vcf to haplotype vcf file
#=================================================
# ------------------------------- Utilities -------------------------------
import os
import sys
import argparse
import subprocess
import gzip
import shutil
from typing import Tuple, List, Dict, Any

import numpy as np
import allel

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
def find_haplotypes(vcf_path: str, out_prefix: str, rtm_gwas_snpldb_path, threads) -> str:
    """
    Call external RTM-GWAS SNPLDB tool:
      rtm-gwas-snpldb --vcf <vcf_path> --out <out_prefix>
    Returns path to the produced haplotype VCF (assumed '<out_prefix>.vcf' or '<out_prefix>_hap.vcf').
    """
    # Convert threads to string if it's an integer
    threads_str = str(threads)

    cmd = [rtm_gwas_snpldb_path, "--vcf", vcf_path, "--thread", threads_str, "--out", out_prefix]
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

# convert train and test SNP vcf files to train and test haplotype vcf file 
def convert_snp_2_haplotypes(tr_snp_vcf, te_snp_vcf, outdir, rtm_gwas_snpldb_path, threads):
    
    # Two VCFs → harmonize, combine, RTM, split
    print("[STEP] Reading training VCF (biallelic SNPs)...")
    tr = _read_biallelic_snp_vcf(tr_snp_vcf)
    print(f"  train: variants={tr['gt'].shape[0]}, samples={len(tr['samples'])}")

    print("[STEP] Reading test VCF (biallelic SNPs)...")
    te = _read_biallelic_snp_vcf(te_snp_vcf)
    print(f"  test : variants={te['gt'].shape[0]}, samples={len(te['samples'])}")

    print("[STEP] Harmonizing by common (CHR,POS) and combining samples...")
    chrom, pos, ref, alt, gt_comb, samples_comb = _harmonize_and_combine(tr, te)
    print(f"  kept variants={gt_comb.shape[0]}, combined samples={gt_comb.shape[1]}")

    combined_vcf = os.path.join(outdir, f"train_test_merged_common_harmonized.vcf")
    print(f"[STEP] Writing combined VCF → {combined_vcf}")
    _write_vcf(combined_vcf, chrom, pos, ref, alt, gt_comb, samples_comb)


    print("[STEP] Calling RTM-GWAS SNPLDB on combined VCF...")
    hap_out_prefix = os.path.join(outdir, os.path.splitext(os.path.basename(tr_snp_vcf))[0] + "_merged_hap")
        
    combined_hap_vcf = find_haplotypes(combined_vcf, hap_out_prefix, rtm_gwas_snpldb_path, threads)
    print(f"Produced: {combined_hap_vcf}")

    print("[STEP] Splitting haplotype VCF back into train/test by original sample IDs...")
    out_train = os.path.join(outdir, f"converted_train_hap.vcf")
    out_test = os.path.join(outdir, f"converted_test_hap.vcf")
    split_vcf_by_samples(
        combined_hap_vcf,
        train_samples=tr['samples'],
        test_samples=te['samples'],
        out_train_vcf=out_train,
        out_test_vcf=out_test
    )

    print(f"[DONE] Train hap VCF: {out_train}")
    print(f"[DONE] Test  hap VCF: {out_test}")
    return out_train, out_test


    """Debug function to check data before training"""
    print(f"\n=== {model_name} DATA DEBUG ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_train shape: {y_train.shape}")
    
    # Check for constant columns
    train_std = np.std(X_train, axis=0)
    constant_cols_train = np.where(train_std == 0)[0]
    print(f"Constant columns in X_train: {len(constant_cols_train)}")
    
    test_std = np.std(X_test, axis=0)
    constant_cols_test = np.where(test_std == 0)[0]
    print(f"Constant columns in X_test: {len(constant_cols_test)}")
    
    # Check for NaN values
    nan_cols_train = np.where(np.isnan(X_train).any(axis=0))[0]
    print(f"Columns with NaN in X_train: {len(nan_cols_train)}")
    
    nan_cols_test = np.where(np.isnan(X_test).any(axis=0))[0]
    print(f"Columns with NaN in X_test: {len(nan_cols_test)}")
    
    # Check data ranges
    print(f"X_train range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"X_test range: [{X_test.min():.3f}, {X_test.max():.3f}]")
    
    if len(constant_cols_train) > 0:
        print(f"First 10 constant column indices in train: {constant_cols_train[:10]}")
    
    return constant_cols_train, constant_cols_test


    """Train scikit-learn models with enhanced data checking"""
    # Debug data before training
    constant_cols_train, constant_cols_test = _debug_haplotype_data(X_train, y_train, None, model_name)
    
    # Remove constant columns if they exist
    if len(constant_cols_train) > 0:
        print(f"[{model_name}] Removing {len(constant_cols_train)} constant columns")
        X_train = np.delete(X_train, constant_cols_train, axis=1)
    
    # Check if we still have valid data
    if X_train.shape[1] == 0:
        print(f"[{model_name}] WARNING: No features remaining after removing constant columns!")
        return None
    
    # Create and train model
    model = self.create_model(model_name, X_train)
    if model is not None:
        try:
            model.fit(X_train, y_train)
            return model
        except Exception as e:
            print(f"[{model_name}] Error during fitting: {e}")
            return None
    
    return None

#=================================================

def normalize_data(data, method='standard', feature_range=(0, 1)):
    if method == 'none' or method is None:
        return data, None

    as_df = isinstance(data, pd.DataFrame)
    columns = data.columns if as_df else None
    index = data.index if as_df else None
    values = data.values if as_df else np.asarray(data)

    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler(feature_range=feature_range)
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    scaled = scaler.fit_transform(values)
    if as_df:
        scaled = pd.DataFrame(scaled, columns=columns, index=index)
    return scaled, scaler

def inverse_normalize(data, scaler, columns=None, index=None):
    if scaler is None:
        return data
    as_df = isinstance(data, pd.DataFrame)
    values = data.values if as_df else np.asarray(data)
    original = scaler.inverse_transform(values)
    if as_df:
        original = pd.DataFrame(
            original,
            columns=columns if columns is not None else data.columns,
            index=index if index is not None else data.index
        )
    return original

def _build_snp_ids(chroms, pos):
    return [f"{c}_{p}" for c, p in zip(chroms, pos)]

def _as_base(x):
    if isinstance(x, (np.ndarray, list, tuple)):
        if len(x) == 0:
            return b'.'
        x = x[0]
    if isinstance(x, str):
        return x.encode()
    return x

def _allele_harmonize(train, test):
    tr_chr = train['variants/CHROM']; tr_pos = train['variants/POS']
    te_chr = test['variants/CHROM'];  te_pos = test['variants/POS']
    tr_ref = train['variants/REF'];   tr_alt = train['variants/ALT']
    te_ref = test['variants/REF'];    te_alt = test['variants/ALT']

    tr_ids = _build_snp_ids(tr_chr, tr_pos)
    te_ids = _build_snp_ids(te_chr, te_pos)
    te_map = {s: i for i, s in enumerate(te_ids)}

    comp = {b'A': b'T', b'T': b'A', b'C': b'G', b'G': b'C'}

    tr_keep, te_keep, flip_mask = [], [], []
    ambiguous = 0
    dropped_other = 0

    for i, sid in enumerate(tr_ids):
        j = te_map.get(sid, None)
        if j is None:
            continue
        r1, a1 = _as_base(tr_ref[i]), _as_base(tr_alt[i])
        r2, a2 = _as_base(te_ref[j]), _as_base(te_alt[j])

        if (r1 == r2) and (a1 == a2):
            tr_keep.append(i); te_keep.append(j); flip_mask.append(False)
        elif (r1 == a2) and (a1 == r2):
            tr_keep.append(i); te_keep.append(j); flip_mask.append(True)
        else:
            if (r1 in comp and a1 in comp and comp[r1] == a1):
                ambiguous += 1
            else:
                dropped_other += 1

    return (np.array(tr_keep, dtype=int),
            np.array(te_keep, dtype=int),
            np.array(flip_mask, dtype=bool),
            ambiguous, dropped_other)

# This function will generate imputed SNP data, PCs
def _impute_and_pca(Xtr_raw, Xte_raw, n_components, feature_view, variance_threshold=0.0, normalization='none',
                    impute_scope='train', pca_fit_scope='train', tag='PC', random_state=42):
    Xtr = Xtr_raw.copy()
    Xte = Xte_raw.copy()

    # Impute
    if impute_scope == 'combined':
        mu = np.nanmean(np.vstack([Xtr, Xte]), axis=0)
    else:
        mu = np.nanmean(Xtr, axis=0)
    r, c = np.where(np.isnan(Xtr));  Xtr[r, c] = mu[c]
    r, c = np.where(np.isnan(Xte));  Xte[r, c] = mu[c]

    # Normalize SNP features (optional)
    if normalization != 'none':
        if normalization == 'standard':
            scaler = StandardScaler(with_mean=True, with_std=True)
        elif normalization == 'minmax':
            scaler = MinMaxScaler()
        elif normalization == 'robust':
            scaler = RobustScaler()
        else:
            raise ValueError(f"Unknown genotype_normalization: {normalization}")

        if pca_fit_scope == 'combined':
            scaler.fit(np.vstack([Xtr, Xte]))
        else:
            scaler.fit(Xtr)
        Xtr = scaler.transform(Xtr)
        Xte = scaler.transform(Xte)

    # Determine number of components based on variance threshold
    if variance_threshold > 0:
        n_components = _determine_pca_components(Xtr, variance_threshold, n_components, feature_view, random_state)
    
    # PCA
    pca = PCA(n_components=n_components, random_state=random_state, svd_solver='auto')
    if pca_fit_scope == 'combined':
        stacked = np.vstack([Xtr, Xte])
        pcs_all = pca.fit_transform(stacked)
        if feature_view == 'PC':
            _report_pca(pca, tag=feature_view)
        ntr = Xtr.shape[0]
        tr_pcs = pcs_all[:ntr]
        te_pcs = pcs_all[ntr:]
    else:
        tr_pcs = pca.fit_transform(Xtr)
        if feature_view == 'PC':
            _report_pca(pca, tag=tag)
        te_pcs = pca.transform(Xte)

    if (feature_view == 'PC'):
        tr_df = pd.DataFrame(tr_pcs)
        te_df = pd.DataFrame(te_pcs)
    else:
        tr_df = pd.DataFrame(Xtr)
        te_df = pd.DataFrame(Xte)
    return tr_df, te_df, pca

# Add this function to report PCA information (same as in your MMOE pipeline)
def _report_pca(pca, tag='PC'):
    try:
        evr = getattr(pca, 'explained_variance_ratio_', None)
        if evr is None:
            return
        total = float(evr.sum()) * 100.0
        first = ', '.join([f"{x*100:.1f}%" for x in evr[:min(5, len(evr))]])
        k = getattr(pca, 'n_components_', getattr(pca, 'n_components', None))
        print(f"[PC] n_components={k} | variance explained (sum)={total:.2f}% | first5=[{first}]")
        
        # Also report cumulative variance
        cumulative_variance = np.cumsum(evr) * 100
        for i, cum_var in enumerate(cumulative_variance[:min(10, len(cumulative_variance))]):
            print(f"[PC] PC{i+1}: {cum_var:.1f}% cumulative")
            
    except Exception as _e:
        print(f"[WARN] PCA variance report failed: {_e}")


    try:
        evr = getattr(pca, 'explained_variance_ratio_', None)
        if evr is None:
            return
        total = float(evr.sum()) * 100.0
        first = ', '.join([f"{x*100:.1f}%" for x in evr[:min(5, len(evr))]])
        k = getattr(pca, 'n_components_', getattr(pca, 'n_components', None))
        print(f"[PC] ({tag}) n_components={k} | variance explained (sum)={total:.2f}% | first5=[{first}]")
    except Exception as _e:
        print(f"[WARN] PCA variance report failed: {_e}")

def _determine_pca_components(X, variance_threshold=0.95, max_components=100, feature_view='SNP', random_state=42):
    """
    Determine number of PC components needed to explain specified cumulative variance
    
    Args:
        X: Input data matrix
        variance_threshold: Minimum cumulative variance to explain (0.0 to 1.0)
        max_components: Maximum number of components to consider
    
    Returns:
        n_components: Number of components needed
    """
    if variance_threshold <= 0:
        n_components = min(X.shape[1], max_components)
        print(f"[PC] Using fixed number of components: {n_components}")
        return n_components
    
    if variance_threshold > 1.0:
        print(f"[PC] Warning: variance_threshold {variance_threshold} > 1.0, using 1.0")
        variance_threshold = 1.0
    
    # Fit PCA to get all components initially
    max_possible = min(X.shape[1], max_components, X.shape[0] - 1)
    pca = PCA(n_components=max_possible, random_state=random_state, svd_solver='auto')
    pca.fit(X)
    
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    
    # Find the smallest number of components that reach the cumulative variance threshold
    for i, cum_var in enumerate(cumulative_variance):
        if cum_var >= variance_threshold:
            n_components = i + 1
            explained_variance = cum_var * 100
            
            # Show variance explained by individual components
            if (feature_view == 'PC'):
                _report_pca(pca, tag=feature_view)
                individual_var = pca.explained_variance_ratio_[:n_components] * 100
                print(f"[PC] Cumulative variance threshold: {variance_threshold*100:.1f}%")
                print(f"[PC] Selected {n_components} components explaining {explained_variance:.1f}%")
                print(f"[PC] Individual component variances: {', '.join([f'{v:.1f}%' for v in individual_var])}")
            
            return n_components
    
    # If threshold not reached, use all components
    n_components = max_possible
    if (feature_view == 'PC'):
        explained_variance = cumulative_variance[-1] * 100
        print(f"[PC] Warning: Could not reach {variance_threshold*100:.1f}% cumulative variance threshold.")
        print(f"[PC] Using all {n_components} components explaining {explained_variance:.1f}%")
    
    return n_components


def _count_vcf_samples_markers(vcf_file):
    """
    Count total samples and markers in VCF file without external libraries
    """
    total_samples = 0
    total_markers = 0
    
    try:
        with open(vcf_file, 'r') as f:
            for line in f:
                # Skip comment lines except the header line with samples
                if line.startswith('##'):
                    continue
                elif line.startswith('#CHROM'):
                    # This is the header line with sample names
                    columns = line.strip().split('\t')
                    total_samples = len(columns) - 9  # First 9 columns are fixed, rest are samples
                elif not line.startswith('#'):
                    total_markers += 1
        
        return total_samples, total_markers
        
    except FileNotFoundError:
        print(f"Error: VCF file {vcf_file} not found")
        return 0, 0
    except Exception as e:
        print(f"Error reading VCF file: {e}")
        return 0, 0

def load_and_process_vcf(outdir, vcf_path, threads, n_components=100, variance_threshold=0.0, normalization='none', rtm_gwas_snpldb_path='rtm-gwas-snpldb', feature_view='SNP', tag='CV', random_state=42):
    # original SNP markers
    import allel
    callset = allel.read_vcf(vcf_path, fields=['calldata/GT', 'variants/CHROM', 'samples'], alt_number=1, fills={'GT': -1})
    gt = allel.GenotypeArray(callset['calldata/GT'])
    snp_matrix = gt.to_n_alt(fill=-1).T.astype(np.float32)
    snp_matrix[snp_matrix < 0] = np.nan
    mu = np.nanmean(snp_matrix, axis=0)
    r, c = np.where(np.isnan(snp_matrix))
    snp_matrix[r, c] = mu[c]

    if normalization != 'none':
        snp_matrix, _ = normalize_data(snp_matrix, method=normalization)
    # snp_matrix are original SNP markers

    # assign same marker matrix for all views
    marker_matrix = snp_matrix
    print ("Total SNPs = ", marker_matrix.shape[1])
    print ("Total samples = ", marker_matrix.shape[0])

    # initiate pca
    pca = 'none'

    # generate haplotype markers
    # Single VCF → just call RTM tool
    if feature_view == 'HAP':
        intermediate_data_dir = os.path.join(outdir, 'intermediate_data')
        os.makedirs(intermediate_data_dir, exist_ok=True)    

        out_prefix = os.path.join(intermediate_data_dir,os.path.splitext(os.path.basename(vcf_path))[0] + "_hap")
        hap_vcf = find_haplotypes(vcf_path, out_prefix, rtm_gwas_snpldb_path, threads)
        
        print (f"haplotype vcf file: {hap_vcf}")
        total_samples, total_markers = _count_vcf_samples_markers(hap_vcf);

        print(f"Total samples in haplotype vcf : {total_samples}")
        print(f"Total markers in haplotype vcf: {total_markers}")

        callset_hap = allel.read_vcf(hap_vcf, fields=['calldata/GT', 'variants/CHROM', 'samples'], alt_number=1, fills={'GT': -1})
        gt_hap = allel.GenotypeArray(callset_hap['calldata/GT'])
        hap_matrix = gt_hap.to_n_alt(fill=-1).T.astype(np.float32)
        hap_matrix[hap_matrix < 0] = np.nan
        mu = np.nanmean(hap_matrix, axis=0)
        r, c = np.where(np.isnan(hap_matrix))
        hap_matrix[r, c] = mu[c]

        if normalization != 'none':
            hap_matrix, _ = normalize_data(hap_matrix, method=normalization)
        marker_matrix = hap_matrix
        print ("Total haplotypes = ", marker_matrix.shape[1])
        print ("Total samples = ", marker_matrix.shape[0])

    elif feature_view == 'PC':
        # Determine components based on variance threshold
        if variance_threshold > 0:
            n_components = _determine_pca_components(snp_matrix, variance_threshold, n_components, feature_view, random_state)
    
        # generate PC markers
        pca = PCA(n_components=n_components, random_state=random_state, svd_solver='auto')
        snp_pca = pca.fit_transform(snp_matrix)
        _report_pca(pca, tag=tag)
    
        marker_matrix = snp_pca
        print ("Total PCs = ", marker_matrix.shape[1])
        print ("Total samples = ", marker_matrix.shape[0])

    return (
        pd.DataFrame(marker_matrix, index=callset['samples']),
        pca
    )

def load_and_merge_genotypes_scoped(train_vcf_path, test_vcf_path, outdir, rtm_gwas_snpldb_path, n_components, 
                                    normalization='none', variance_threshold=0.0, feature_view='SNP', threads=10,
                                    impute_scope='train', pca_fit_scope='train', random_state=42):
    # process raw SNP vcf file
    import allel
    fields = ['calldata/GT','samples','variants/CHROM','variants/POS','variants/REF','variants/ALT']
    train = allel.read_vcf(train_vcf_path, fields=fields, fills={'GT': -1}, alt_number=1)
    test  = allel.read_vcf(test_vcf_path,  fields=fields, fills={'GT': -1}, alt_number=1)

    gt_tr = allel.GenotypeArray(train['calldata/GT']).to_n_alt(fill=-1)
    gt_te = allel.GenotypeArray(test['calldata/GT']).to_n_alt(fill=-1)

    tr_keep, te_keep, flip_mask, ambiguous, dropped_other = _allele_harmonize(train, test)
    print(f"[Allele harmonization] common SNPs: {len(train['variants/POS'])} vs {len(test['variants/POS'])}, "
          f"kept: {len(tr_keep)}, flipped: {int(flip_mask.sum())}, "
          f"dropped ambiguous: {ambiguous}, dropped other: {dropped_other}")

    if len(tr_keep) == 0:
        raise ValueError("No common SNPs after harmonization.")

    Xtr_raw = gt_tr[tr_keep].astype(np.float32).T
    Xte_raw = gt_te[te_keep].astype(np.float32).T

    Xtr_raw[Xtr_raw < 0] = np.nan
    Xte_raw[Xte_raw < 0] = np.nan

    if flip_mask.any():
        Xte_raw[:, flip_mask] = 2.0 - Xte_raw[:, flip_mask]


    if feature_view == 'HAP':
        # convert  raw SNP vcf file to haplotype vcf file
        tr_hap_vcf_path, te_hap_vcf_path = convert_snp_2_haplotypes(train_vcf_path, test_vcf_path, outdir, rtm_gwas_snpldb_path, threads);
        fields = ['calldata/GT','samples','variants/CHROM','variants/POS','variants/REF','variants/ALT']
        train_hap = allel.read_vcf(tr_hap_vcf_path, fields=fields, fills={'GT': -1}, alt_number=1)
        test_hap  = allel.read_vcf(te_hap_vcf_path,  fields=fields, fills={'GT': -1}, alt_number=1)

        gt_tr_hap = allel.GenotypeArray(train_hap['calldata/GT']).to_n_alt(fill=-1)
        gt_te_hap = allel.GenotypeArray(test_hap['calldata/GT']).to_n_alt(fill=-1)

        tr_keep_hap, te_keep_hap, flip_mask_hap, ambiguous_hap, dropped_other_hap = _allele_harmonize(train_hap, test_hap)
        print(f"[Allele harmonization] common haplotyp markers: {len(train_hap['variants/POS'])} vs {len(test_hap['variants/POS'])}, "
              f"kept: {len(tr_keep_hap)}, flipped: {int(flip_mask_hap.sum())}, "
              f"dropped ambiguous: {ambiguous_hap}, dropped other: {dropped_other_hap}")

        if len(tr_keep_hap) == 0:
            raise ValueError("No common haplotype markers after harmonization.")

        Xtr_raw_hap = gt_tr_hap[tr_keep_hap].astype(np.float32).T
        Xte_raw_hap = gt_te_hap[te_keep_hap].astype(np.float32).T

        Xtr_raw_hap[Xtr_raw_hap < 0] = np.nan
        Xte_raw_hap[Xte_raw_hap < 0] = np.nan

        if flip_mask_hap.any():
            Xte_raw_hap[:, flip_mask_hap] = 2.0 - Xte_raw_hap[:, flip_mask_hap]

        tr_keep = tr_keep_hap
        te_keep = te_keep_hap

        Xtr_raw = Xtr_raw_hap
        Xte_raw = Xte_raw_hap

    # PC plot is needed, thus, for all feature view, we will do PCA
    # Convert variance_threshold to float to avoid string comparison error
    variance_threshold_float = float(variance_threshold)
    
    # tr_markers, te_markers represent markers of SNP, HAP or PC
    tr_markers, te_markers, pca = _impute_and_pca(
        Xtr_raw, 
        Xte_raw, 
        n_components, 
        feature_view,
        variance_threshold=variance_threshold_float,  # Use converted float value
        normalization=normalization,
        impute_scope=impute_scope, 
        pca_fit_scope=pca_fit_scope,
        random_state=random_state 
    )

    tr_markers.index = pd.Index(train['samples'], name='sample')
    te_markers.index = pd.Index(test['samples'],  name='sample')
    tr_data = tr_markers
    te_data = te_markers


    return tr_data, te_data, pca, Xtr_raw, Xte_raw, tr_keep, flip_mask

def prepare_genotype_data(outdir, vcf_path, phenotype_df, threads, n_components, variance_threshold=0.0, normalization='none', rtm_gwas_snpldb='rtm-gwas-snpldb', feature_view='SNP', random_state=42):
    features, pca = load_and_process_vcf(outdir, vcf_path, threads, n_components, variance_threshold, normalization, rtm_gwas_snpldb, feature_view, tag='CV', random_state=random_state)
    phenotype_df.index = phenotype_df.index.astype(str).str.upper().str.strip()
    features.index = features.index.astype(str).str.upper().str.strip()
    common = features.index.intersection(phenotype_df.index)
    if len(common) == 0:
        raise ValueError('No matching samples found between VCF and phenotype data')
    return features.loc[common], phenotype_df.loc[common]

def load_data(config):
    y_data = pd.read_csv(config['phenotype_path'], sep=None, engine='python', index_col=0)
    if y_data.isna().any().any():
        print("[INFO] Filled missing phenotype values with per-trait medians.")
        y_data = y_data.fillna(y_data.median())

    for c in y_data.columns:
        if y_data[c].nunique() == 1:
            y_data[c] = y_data[c] + np.random.normal(0, 1e-6, size=len(y_data))

    if config['pheno_normalization'] != 'none':
        y_data, pheno_scaler = normalize_data(y_data, method=config['pheno_normalization'])
    else:
        pheno_scaler = None


    # Use config seed for deterministic PCA
    random_state = config.get('seed', 42)

    print(f"[Input] Using VCF genotype file: {config['vcf_path']}")
    # x_data are original SNP features
    x_data, y_data = prepare_genotype_data(
            config['results_dir'],
            config['vcf_path'], 
            y_data, 
            config['threads'],
            config['n_pca_components'],
            config.get('pca_variance_explained', 0.0),
            config['genotype_normalization'],
            config['rtm-gwas-snpldb_path'],
            config['feature_view'],
            random_state
    )

    # may be SNP, HAP or PC deneding on feature_view setting
    if x_data.index.duplicated().any() or y_data.index.duplicated().any():
        print("Duplicates in x_data (genomic) index:", x_data.index[x_data.index.duplicated()].unique())
        print("Duplicates in y_data (phenotype) index:", y_data.index[y_data.index.duplicated()].unique())

    full = pd.concat([x_data, y_data], axis=1).fillna(0)
    label_cols = y_data.columns.tolist()
    used_cols = [c for c in full.columns if c not in label_cols]
    
    return (
        full, used_cols, label_cols,
        pheno_scaler
    )

def _parse_gamma_param(gamma_str, X_train=None, default='scale'):
    """Helper function to consistently parse gamma parameter"""
    valid_strings = {'scale', 'auto'}
    
    if gamma_str in valid_strings:
        if gamma_str == 'scale' and X_train is not None:
            # For scale: 1 / (n_features * X.var())
            return 1.0 / (X_train.shape[1] * np.var(X_train))
        elif gamma_str == 'auto' and X_train is not None:
            # For auto: 1 / n_features
            return 1.0 / X_train.shape[1]
        else:
            return gamma_str
    else:
        try:
            gamma = float(gamma_str)
            if gamma <= 0:
                print(f"Warning: gamma must be > 0, using 'auto'")
                return 'auto' if X_train is None else 1.0 / X_train.shape[1]
            return gamma
        except (ValueError, TypeError):
            print(f"Warning: Invalid gamma value '{gamma_str}', using '{default}'")
            return default

# -----------------------------
# Phenotype Analysis 
# -----------------------------
def _phenotype_basic_stats(df: pd.DataFrame) -> pd.DataFrame:
    # Compute descriptive stats + skewness/kurtosis + missing counts
    stats = pd.DataFrame({
        'count': df.count(),
        'mean': df.mean(),
        'std': df.std(ddof=1),
        'min': df.min(),
        '25%': df.quantile(0.25),
        '50% (median)': df.median(),
        '75%': df.quantile(0.75),
        'max': df.max(),
        'skew': df.apply(lambda x: skew(pd.to_numeric(x, errors='coerce').dropna()), axis=0),
        'kurtosis': df.apply(lambda x: kurtosis(pd.to_numeric(x, errors='coerce').dropna(), fisher=True), axis=0),
        'missing': df.isna().sum()
    })
    return stats

def _phenotype_histograms(df: pd.DataFrame, out_path: str, max_cols: int = 4):
    cols = list(df.columns)
    n = len(cols)
    ncols = min(max_cols, n) if n > 0 else 1
    nrows = int(math.ceil(n / ncols)) if n > 0 else 1
    plt.figure(figsize=(4*ncols, 3*nrows))
    for i, c in enumerate(cols, 1):
        ax = plt.subplot(nrows, ncols, i)
        series = pd.to_numeric(df[c], errors='coerce').dropna().values
        ax.hist(series, bins=30)
        ax.set_title(str(c))
        ax.set_xlabel('Value'); ax.set_ylabel('Count')
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

def _phenotype_corr_heatmap(df: pd.DataFrame, out_img: str, out_corr_csv: str, out_p_csv: str):
    cols = list(df.columns)
    k = len(cols)
    R = np.ones((k, k), dtype=float)
    P = np.zeros((k, k), dtype=float)
    # Compute Pearson r and p for each pair
    for i in range(k):
        xi = pd.to_numeric(df[cols[i]], errors='coerce').dropna()
        for j in range(k):
            xj = pd.to_numeric(df[cols[j]], errors='coerce').dropna()
            # Align lengths by common index
            common = xi.index.intersection(xj.index)
            if len(common) < 3:
                r, p = np.nan, np.nan
            else:
                r, p = pearsonr(xi.loc[common].values, xj.loc[common].values)
            R[i, j] = r
            P[i, j] = p
    # Save numeric outputs
    pd.DataFrame(R, index=cols, columns=cols).to_csv(out_corr_csv)
    pd.DataFrame(P, index=cols, columns=cols).to_csv(out_p_csv)
    # Heatmap with annotations (r and significance)
    def _stars(p):
        if not np.isfinite(p): return ''
        return '***' if p < 1e-3 else ('**' if p < 1e-2 else ('*' if p < 0.05 else ''))
    fig, ax = plt.subplots(figsize=(0.7*k+3, 0.7*k+3))
    im = ax.imshow(R, vmin=-1, vmax=1)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(np.arange(k)); ax.set_yticks(np.arange(k))
    ax.set_xticklabels(cols, rotation=45, ha='right'); ax.set_yticklabels(cols)
    # annotate
    for i in range(k):
        for j in range(k):
            r = R[i, j]
            p = P[i, j]
            txt = f"{r:.2f}" if np.isfinite(r) else "NA"
            sig = _stars(p)
            ax.text(j, i, f"{txt}\n{sig}", ha='center', va='center', fontsize=8)
    ax.set_title('Trait correlations (Pearson r) with significance')
    fig.tight_layout()
    fig.savefig(out_img, dpi=200)
    plt.close(fig)

def pairplot_with_corr(df):

    def significance_stars(p):
        if p < 0.001:
            return '***'
        elif p < 0.01:
            return '**'
        elif p < 0.05:
            return '*'
        else:
            return ''

    cols = df.columns
    n = len(cols)
    fig, axes = plt.subplots(n, n, figsize=(3 * n, 3 * n))
    plt.subplots_adjust(wspace=0.2, hspace=0.2)

    blues = ["#6f93ad", "#6497b1", "#86aac0", "#03396c", "#011f4b"]

    import contextlib
    import numpy as np

    for i in range(n):
        for j in range(n):
            ax = axes[i, j]

            # diagonal: histogram + normal curve
            if i == j:
                data = df[cols[i]].dropna()
                # Determine if data has enough spread for 20 bins
                with contextlib.suppress(ValueError):
                    if np.ptp(data) > 0:  # ptp = max - min
                        ax.hist(data, bins=20, color=blues[0], alpha=0.6, density=True)
                    else:
                        # Skip or use a single bin if data are constant
                        ax.hist(data, bins=1, color=blues[0], alpha=0.6, density=True)


                mu, sigma = np.mean(data), np.std(data)
                x = np.linspace(data.min(), data.max(), 200)
                ax.plot(x, norm.pdf(x, mu, sigma), color=blues[1], lw=2)
                ax.set_title(cols[i], fontsize=12, color="#011f4b", pad=8)

            # lower triangle: scatter + regression + correlation
            elif i > j:
                x = df[cols[j]].values
                y = df[cols[i]].values
                mask = ~np.isnan(x) & ~np.isnan(y)
                r, p = pearsonr(x[mask], y[mask])

                ax.scatter(x[mask], y[mask], s=10, alpha=0.6, color=blues[2])

                if np.var(x[mask]) > 0:
                    m, b = np.polyfit(x[mask], y[mask], 1)
                    x_line = np.linspace(min(x[mask]), max(x[mask]), 200)
                    ax.plot(x_line, m * x_line + b, color=blues[1], lw=1.5)

                ax.text(
                    0.5, 0.9, f"r={r:.2f}{significance_stars(p)}",
                    transform=ax.transAxes,
                    fontsize=10, fontweight='bold',
                    color="#011f4b", ha='left', va='top'
                )

            # upper triangle: empty
            else:
                ax.axis('off')

            # formatting
            if i < n - 1:
                ax.set_xticklabels([])
            if j > 0:
                ax.set_yticklabels([])

            if i == n - 1:
                ax.set_xlabel(cols[j], fontsize=11, color="black", labelpad=5)
            if j == 0:
                ax.set_ylabel(cols[i], fontsize=11, color="black", labelpad=5)

    plt.tight_layout()
    return fig  

def generate_phenotype_analysis(df: pd.DataFrame, out_dir: str, prefix: str = 'train'):
    os.makedirs(out_dir, exist_ok=True)
    # Basic stats
    stats = _phenotype_basic_stats(df)
    stats.to_csv(os.path.join(out_dir, f'phenotype_stats_{prefix}.csv'))
    # Histograms (all traits in one figure)
    _phenotype_histograms(df, os.path.join(out_dir, f'phenotype_histograms_{prefix}.png'))
    # Correlation heatmap + CSVs
    _phenotype_corr_heatmap(
        df,
        os.path.join(out_dir, f'phenotype_corr_heatmap_{prefix}.png'),
        os.path.join(out_dir, f'phenotype_corr_{prefix}.csv'),
        os.path.join(out_dir, f'phenotype_corr_pvals_{prefix}.csv')
    )

    numeric_df = df.select_dtypes(include=['number'])
    if numeric_df.shape[1] > 1:
        fig = pairplot_with_corr(numeric_df)
        fig_name = os.path.join(out_dir, f'phenotype_corr_matrix_{prefix}.png')
        fig.savefig(fig_name, dpi=200, bbox_inches='tight')
        plt.close(fig)

# -----------------------------
# Visualization Functions 

# ==== Train/Test combined scatter helpers ====
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr, t as student_t

def _align_true_pred(y_true, y_pred, idx_true=None, idx_pred=None):
    """Align by index; return numpy arrays (x, y, used_index)."""
    if isinstance(y_true, (pd.Series, pd.DataFrame)):
        s_true = pd.Series(np.asarray(y_true).squeeze(), index=y_true.index, name="true")
    else:
        if idx_true is None:
            raise ValueError("idx_true required when y_true is not a pandas Series.")
        s_true = pd.Series(np.asarray(y_true).squeeze(), index=pd.Index(idx_true), name="true")

    if isinstance(y_pred, (pd.Series, pd.DataFrame)):
        s_pred = pd.Series(np.asarray(y_pred).squeeze(), index=y_pred.index, name="pred")
    else:
        if idx_pred is None:
            raise ValueError("idx_pred required when y_pred is not a pandas Series.")
        s_pred = pd.Series(np.asarray(y_pred).squeeze(), index=pd.Index(idx_pred), name="pred")

    if not s_true.index.is_unique:
        s_true = s_true[~s_true.index.duplicated(keep="first")]
    if not s_pred.index.is_unique:
        s_pred = s_pred[~s_pred.index.duplicated(keep="first")]

    df = pd.concat([s_true, s_pred], axis=1, join="inner").dropna()
    return df["true"].to_numpy(), df["pred"].to_numpy(), df.index

def _ols_conf_band(x, y, xgrid, alpha=0.05):
    """
    Mean-response 1-alpha confidence band for OLS y ~ b0 + b1*x.
    Returns (y_hat_on_grid, lower, upper, beta).
    """
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = x.size
    X = np.c_[np.ones_like(x), x]
    XtX = X.T @ X
    XtX_inv = np.linalg.pinv(XtX)
    beta = XtX_inv @ (X.T @ y)               # [b0, b1]
    yhat = X @ beta
    resid = y - yhat
    dof = max(n - 2, 1)
    s2 = (resid @ resid) / dof               # residual variance

    Xg = np.c_[np.ones_like(xgrid), xgrid]
    se = np.sqrt(s2 * np.sum((Xg @ XtX_inv) * Xg, axis=1))

    # Student-t critical value (fallback to normal if needed)
    try:
        tcrit = float(student_t.ppf(1 - alpha/2, dof))
        if not np.isfinite(tcrit):
            raise ValueError
    except Exception:
        from math import log, sqrt
        def _approx_erfinv(u):
            a = 0.147
            sgn = np.sign(u)
            u = np.clip(u, -0.999999, 0.999999)
            ln = log(1 - u*u)
            return sgn*np.sqrt(np.sqrt((2/(np.pi*a)+ln/2)**2 - ln/a) - (2/(np.pi*a)+ln/2))
        z = sqrt(2) * _approx_erfinv(1 - alpha)
        tcrit = z

    ygrid = Xg @ beta
    lo = ygrid - tcrit * se
    hi = ygrid + tcrit * se
    return ygrid, lo, hi, beta

def plot_true_vs_predicted_train_test(
    y_train_true, y_train_pred,
    y_test_true, y_test_pred,
    output_path,
    trait_name, model_name,
    ci_alpha=0.05,
    color_train="C0",
    color_test="C1"
):
    """
    One figure: Train & Test scatter with separate OLS lines + 95% CIs, and r for each split.
    Inputs can be Series (preferred) or arrays; alignment is done by index if Series provided.
    """
    # Align each split separately (keeps only common samples of that split)
    x_tr, y_tr, _ = _align_true_pred(y_train_true, y_train_pred)
    x_te, y_te, _ = _align_true_pred(y_test_true,  y_test_pred)

    # Guard
    if len(x_tr) < 2 and len(x_te) < 2:
        raise ValueError("Not enough points to plot.")

    # Correlations
    r_tr, p_tr = (pearsonr(x_tr, y_tr) if len(x_tr) >= 2 else (np.nan, np.nan))
    r_te, p_te = (pearsonr(x_te, y_te) if len(x_te) >= 2 else (np.nan, np.nan))

    # X-range for both bands
    xmin = np.nanmin([np.nanmin(x_tr) if len(x_tr) else np.nan, np.nanmin(x_te) if len(x_te) else np.nan])
    xmax = np.nanmax([np.nanmax(x_tr) if len(x_tr) else np.nan, np.nanmax(x_te) if len(x_te) else np.nan])
    if not np.isfinite(xmin) or not np.isfinite(xmax) or xmin == xmax:
        xmin = np.nanmin([x_tr.min() if len(x_tr) else x_te.min()])
        xmax = np.nanmax([x_tr.max() if len(x_tr) else x_te.max()])
    xgrid = np.linspace(xmin, xmax, 200)

    # OLS for train
    yfit_tr = lo_tr = hi_tr = None
    if len(x_tr) >= 2:
        yfit_tr, lo_tr, hi_tr, _ = _ols_conf_band(x_tr, y_tr, xgrid, alpha=ci_alpha)

    # OLS for test
    yfit_te = lo_te = hi_te = None
    if len(x_te) >= 2:
        yfit_te, lo_te, hi_te, _ = _ols_conf_band(x_te, y_te, xgrid, alpha=ci_alpha)

    # Plot
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    plt.figure(figsize=(6.8, 6.3))

    # Scatter points
    if len(x_tr):
        plt.scatter(x_tr, y_tr, s=20, alpha=0.65, label=f"Train (r={r_tr:.3f})", color=color_train)
    if len(x_te):
        plt.scatter(x_te, y_te, s=20, alpha=0.65, label=f"Test (r={r_te:.3f})",  color=color_test)

    # Lines + bands
    if yfit_tr is not None:
        plt.plot(xgrid, yfit_tr, lw=1.8, linestyle="--", color=color_train)
        plt.fill_between(xgrid, lo_tr, hi_tr, alpha=0.18, linewidth=0, color=color_train)
    if yfit_te is not None:
        plt.plot(xgrid, yfit_te, lw=1.8, linestyle="--", color=color_test)
        plt.fill_between(xgrid, lo_te, hi_te, alpha=0.18, linewidth=0, color=color_test)

    plt.title(f"{trait_name} ({model_name})")
    plt.xlabel("True")
    plt.ylabel("Predicted")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=220)
    plt.close()

    return {"r_train": float(r_tr), "p_train": float(p_tr),
            "r_test": float(r_te), "p_test": float(p_te),
            "path": output_path}
# ==== end helpers ====


def plot_true_vs_predicted(true_values, predicted_values, output_dir, trait_name, model_name, feature_view, ci_alpha=0.05):
    """
    Scatter of True vs Predicted with OLS fitted line and 95% confidence band.
    No y=x reference; no LOWESS.
    """
    # arrays and mask
    x = np.asarray(true_values, dtype=float)
    y = np.asarray(predicted_values, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]; y = y[m]

    # correlation for title
    try:
        r, p_value = pearsonr(x, y)
    except Exception:
        r, p_value = np.nan, np.nan

    # edge case
    if x.size < 3:
        plt.figure(figsize=(6,6))
        plt.scatter(x, y, alpha=0.6, s=20)
        plt.title(f'{trait_name} ({model_name}): r={r:.3f}, p={p_value:.3g}')
        plt.xlabel('True'); plt.ylabel('Predicted')
        os.makedirs(output_dir, exist_ok=True)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'true_vs_pred_{feature_view}_{model_name}_{trait_name}.png'), dpi=220)
        plt.close()
        return r, p_value

    # create grid
    xmin, xmax = x.min(), x.max()
    xgrid = np.linspace(xmin, xmax, 200)

    # OLS line + band
    yfit, lo, hi, beta = _ols_conf_band(x, y, xgrid, alpha=ci_alpha)

    # plot
    plt.figure(figsize=(6.6,6.2))
    plt.scatter(x, y, alpha=0.6, s=20, label="Samples")
    plt.plot(xgrid, yfit, lw=1.8, linestyle="--", label="OLS fit")
    plt.fill_between(xgrid, lo, hi, alpha=0.18, linewidth=0, label=f"{int((1-ci_alpha)*100)}% CI")

    # no y=x reference line
    plt.title(f'{trait_name} ({model_name}): r={r:.3f}, p={p_value:.3g}')
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.legend()
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'true_vs_pred_{feature_view}_{model_name}_{trait_name}.png'), dpi=220)
    plt.close()
    return r, p_value

def plot_true_vs_predicted_multi(true_values, preds_by_model: dict, output_dir: str, trait_name: str, feature_view):
    """Overlay scatter for multiple models in one figure for easy comparison."""
    plt.figure(figsize=(10, 8))  # Slightly larger to accommodate more models
    
    # In the plot_true_vs_predicted_multi function:
    colors = {
        # RRBLUP family
        'RRBLUP':      '#1f77b4',  # blue
        'R_RRBLUP':    '#1f77b4',  # alias
        'GBLUP':       '#1f77b4',  # alias
        'R_GBLUP':     '#1f77b4',  # alias

        # Linear models
        'ElasticNet':  '#17becf',  # cyan
        'BRR':         '#9edae5',  # light cyan

        # Tree-based models
        'RFR':         '#8c564b',  # brown
        'XGBoost':     '#ff9896',  # light red
        'LightGBM':    '#c5b0d5',  # light purple

        # Deep learning models
        'DNNGS':        '#d62728',  # red
        'DeepResBLUP':  '#2ca02c',  # green
        'DeepBLUP':     '#bcbd22',  # yellow-green

        # Attention / Hybrid
        'HybridAttnMLP': '#ff7f0e',  # orange
        'AttnCNNGS':     '#ff6b6b',  # coral red (was commented for CNN)

        # Standard MLP GS
        'MLPGS':         '#9467bd',  # purple (Transformer color reused)

        # Graph models
        'GraphConvGS':   '#8c564b',  # brown (same as RFR—kept for consistency)
        'GraphAttnGS':   '#e377c2',  # pink
        'GraphSAGEGS':   '#7f7f7f',  # gray
        'GraphFormer':   '#aec7e8',  # light blue (previous EnsembleGS color)

        # Transformer (if needed)
        'Transformer':   '#9467bd',  # purple (not currently used)

        # Ensemble
        'EnsembleGS':    '#aec7e8',  # light blue
    }
        
    for model_name, y_pred in preds_by_model.items():
        y_true = np.asarray(true_values, dtype=float)
        y_pred = np.asarray(y_pred, dtype=float)

        mask = np.isfinite(y_true) & np.isfinite(y_pred)
        if not np.any(mask):
            continue
        y_true_m = y_true[mask]; y_pred_m = y_pred[mask]
        plt.scatter(y_true_m, y_pred_m, alpha=0.6, label=model_name, color=colors.get(model_name, None))
        try:
            c = np.polyfit(y_true_m, y_pred_m, 1)
            xline = np.linspace(float(np.min(y_true_m)), float(np.max(y_true_m)), 100)
            yline = c[0]*xline + c[1]
            plt.plot(xline, yline, ls='--', color=colors.get(model_name, None))
        except Exception:
            pass
    plt.xlabel('True'); plt.ylabel('Predicted')
    plt.title(f'{trait_name}: True vs Predicted (all models)')
    plt.legend(frameon=False)
    os.makedirs(output_dir, exist_ok=True)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'true_vs_pred_ALL_{feature_view}_{trait_name}.png'), dpi=200)
    plt.close()

def generate_plots_by_feature_view(results, out_dir, feature_view):
    os.makedirs(out_dir, exist_ok=True)
    # Flatten to a DataFrame
    rows = []
    for trait, recs in results.items():
        for r in recs:
            model = r.get('model', feature_view)
            rows.append({'trait': trait, 'PearsonR': r['PearsonR'], 'R2': r.get('R2', np.nan), 'model': model})
    df = pd.DataFrame(rows)
    if not len(df):
        return
   

    # Grouped by TRAIT: for each trait, models side-by-side
    _grouped_box_by_trait(df, 'PearsonR', f'{feature_view}_cv_grouped_box_by_trait_r.png', 
                            'Pearson r', 'CV Pearson r', out_dir)
    # Grouped by Model: for model, traits side-by-side
    _grouped_box_by_model(df, 'PearsonR', f'{feature_view}_cv_grouped_box_by_model_r.png', 
                            'Pearson r', 'CV Pearson r', out_dir)


def _grouped_box_by_trait(results_df, metric_key: str, fname: str, xlab: str, title: str, out_dir: str):
    import matplotlib.patches as mpatches
    traits = sorted(results_df['trait'].unique())
    models = sorted(results_df['model'].unique())
    n_traits = len(traits); n_models = len(models)
    height = 0.8 / max(1, n_models)
    
    # Complete color palette for all models
    colors = {
        # RRBLUP family
        'RRBLUP':      '#1f77b4',  # blue
        'R_RRBLUP':    '#1f77b4',  # alias
        'GBLUP':       '#1f77b4',  # alias
        'R_GBLUP':     '#1f77b4',  # alias

        # Linear models
        'ElasticNet':  '#17becf',  # cyan
        'BRR':         '#9edae5',  # light cyan

        # Tree-based models
        'RFR':         '#8c564b',  # brown
        'XGBoost':     '#ff9896',  # light red
        'LightGBM':    '#c5b0d5',  # light purple

        # Deep learning models
        'DNNGS':        '#d62728',  # red
        'DeepResBLUP':  '#2ca02c',  # green
        'DeepBLUP':     '#bcbd22',  # yellow-green

        # Attention / Hybrid
        'HybridAttnMLP': '#ff7f0e',  # orange
        'AttnCNNGS':     '#ff6b6b',  # coral red (was commented for CNN)

        # Standard MLP GS
        'MLPGS':         '#9467bd',  # purple (Transformer color reused)

        # Graph models
        'GraphConvGS':   '#8c564b',  # brown (same as RFR—kept for consistency)
        'GraphAttnGS':   '#e377c2',  # pink
        'GraphSAGEGS':   '#7f7f7f',  # gray
        'GraphFormer':   '#aec7e8',  # light blue (previous EnsembleGS color)

        # Transformer (if needed)
        'Transformer':   '#9467bd',  # purple (not currently used)

        # Ensemble
        'EnsembleGS':    '#aec7e8',  # light blue
    }
    
    # Alternative: assign colors automatically if model not in dictionary
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                     '#9edae5', '#aec7e8', '#ff9896', '#c5b0d5', '#ff6b6b']
    
    fig, ax = plt.subplots(figsize=(8, max(6, 0.8 * n_traits)))
    
    for i, t in enumerate(traits):
        base = i
        for j, m in enumerate(models):
            vals = results_df[(results_df['trait']==t) & (results_df['model']==m)][metric_key].dropna().values
            if len(vals) == 0:
                continue
            pos = base + (j - (n_models-1)/2)*height
            
            # Get color for this model
            color = colors.get(m)
            if color is None:
                # Assign from default colors based on model index
                model_idx = list(models).index(m) % len(default_colors)
                color = default_colors[model_idx]
            
            # Create horizontal boxplot
            bp = ax.boxplot([vals], positions=[pos], widths=height, patch_artist=True, vert=False)
            
            # Style the boxplot
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.7)
            for whisker in bp['whiskers']:
                whisker.set_color(color)
                whisker.set_alpha(0.8)
            for cap in bp['caps']:
                cap.set_color(color)
                cap.set_alpha(0.8)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)
            
            # Add scatter points with jitter (vertical jitter for horizontal plot)
            jitter = np.random.normal(0, height/6, size=len(vals))
            ax.scatter(vals, np.repeat(pos, len(vals)) + jitter, s=8, alpha=0.6, 
                      color=color, edgecolors='none', linewidth=0.3, zorder=3)
    
    # Set y-axis labels and formatting
    ax.set_yticks(range(n_traits))
    ax.set_yticklabels(traits)
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Create legend
    handles = []
    for m in models:
        color = colors.get(m)
        if color is None:
            model_idx = list(models).index(m) % len(default_colors)
            color = default_colors[model_idx]
        handles.append(mpatches.Patch(color=color, label=m, alpha=0.7))
    
    # Position legend outside the plot if there are many models
    if len(models) > 8:
        ax.legend(handles=handles, title='Model', frameon=True, 
                 bbox_to_anchor=(1.05, 1), loc='upper left', 
                 borderaxespad=0., framealpha=0.9)
    else:
        ax.legend(handles=handles, title='Model', frameon=True, 
                 loc='best', framealpha=0.9)
    
    # Adjust layout and save
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)


def _grouped_box_by_model(results_df, metric_key: str, fname: str, xlab: str, title: str, out_dir: str):
    import matplotlib.patches as mpatches
    models = sorted(results_df['model'].unique())
    traits = sorted(results_df['trait'].unique())
    n_models = len(models); n_traits = len(traits)
    height = 0.8 / max(1, n_traits)
    
    # Color palette for traits - using a distinct color scheme
    colors = {
        # Using a qualitative color palette for traits
        'Trait1': '#1f77b4',      # blue
        'Trait2': '#ff7f0e',      # orange
        'Trait3': '#2ca02c',      # green
        'Trait4': '#d62728',      # red
        'Trait5': '#9467bd',      # purple
        'Trait6': '#8c564b',      # brown
        'Trait7': '#e377c2',      # pink
        'Trait8': '#7f7f7f',      # gray
        'Trait9': '#bcbd22',      # yellow-green
        'Trait10': '#17becf',     # cyan
        'Trait11': '#ff9896',     # light red
        'Trait12': '#c5b0d5',     # light purple
        'Trait13': '#aec7e8',     # light blue
        'Trait14': '#ff6b6b',     # coral
        'Trait15': '#9edae5',     # light cyan
    }
    
    # Alternative: assign colors automatically based on trait names
    default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
                     '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                     '#ff9896', '#c5b0d5', '#aec7e8', '#ff6b6b', '#9edae5',
                     '#393b79', '#637939', '#8c6d31', '#843c39', '#7b4173']
    
    fig, ax = plt.subplots(figsize=(10, max(6, 0.6 * n_models)))
    
    for i, m in enumerate(models):
        base = i
        for j, t in enumerate(traits):
            vals = results_df[(results_df['model']==m) & (results_df['trait']==t)][metric_key].dropna().values
            if len(vals) == 0:
                continue
            pos = base + (j - (n_traits-1)/2)*height
            
            # Get color for this trait
            color = colors.get(t)
            if color is None:
                # Assign from default colors based on trait index
                trait_idx = list(traits).index(t) % len(default_colors)
                color = default_colors[trait_idx]
            
            # Create horizontal boxplot
            bp = ax.boxplot([vals], positions=[pos], widths=height, patch_artist=True, vert=False)
            
            # Style the boxplot
            for box in bp['boxes']:
                box.set_facecolor(color)
                box.set_alpha(0.7)
            for whisker in bp['whiskers']:
                whisker.set_color(color)
                whisker.set_alpha(0.8)
            for cap in bp['caps']:
                cap.set_color(color)
                cap.set_alpha(0.8)
            for median in bp['medians']:
                median.set_color('black')
                median.set_linewidth(1.5)
            
            # Add scatter points with jitter
            jitter = np.random.normal(0, height/6, size=len(vals))
            ax.scatter(vals, np.repeat(pos, len(vals)) + jitter, s=8, alpha=0.6, 
                      color=color, edgecolors='none', linewidth=0.3, zorder=3)
    
    # Set y-axis labels and formatting (models on Y-axis)
    ax.set_yticks(range(n_models))
    ax.set_yticklabels(models)
    ax.set_xlabel(xlab)
    ax.set_title(title)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Create legend for traits
    handles = []
    for t in traits:
        color = colors.get(t)
        if color is None:
            trait_idx = list(traits).index(t) % len(default_colors)
            color = default_colors[trait_idx]
        handles.append(mpatches.Patch(color=color, label=t, alpha=0.7))
    
    # Position legend
    if len(traits) > 8:
        ax.legend(handles=handles, title='Trait', frameon=True, 
                 bbox_to_anchor=(1.05, 1), loc='upper left', 
                 borderaxespad=0., framealpha=0.9)
    else:
        ax.legend(handles=handles, title='Trait', frameon=True, 
                 loc='best', framealpha=0.9)
    
    # Adjust layout and save
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, fname), dpi=200, bbox_inches='tight')
    plt.close(fig)

# Export predictions for all models and single marker view
def export_trait_predictions(config, all_predictions, test_phenotypes=None, output_dir=None):
    """
    Export predictions for each trait with all models' predictions and true values
    
    Args:
        all_predictions: Dictionary of {model_name: prediction_df}
        test_phenotypes: DataFrame of true test phenotypes (optional)
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all trait names from the first model's predictions
    if not all_predictions:
        print("No predictions to export")
        return
        
    first_model = list(all_predictions.keys())[0]
    trait_names = all_predictions[first_model].columns.tolist()
    
    # Get sample IDs from the first model
    sample_ids = all_predictions[first_model].index.tolist()
    
    for trait in trait_names:
        print(f"Exporting predictions for trait: {trait}")
        
        # Create DataFrame for this trait
        trait_data = pd.DataFrame(index=sample_ids)
        trait_data.index.name = 'Sample'
        
        # Add predictions from each model
        for model_name, pred_df in all_predictions.items():
            if trait in pred_df.columns:
                trait_data[f'{model_name}_predicted'] = pred_df[trait]
        
        # Add true values if available
        if test_phenotypes is not None and trait in test_phenotypes.columns:
            # Align with prediction samples
            common_samples = trait_data.index.intersection(test_phenotypes.index)
            if len(common_samples) > 0:
                trait_data['True_value'] = np.nan  # Initialize with NaN
                trait_data.loc[common_samples, 'True_value'] = test_phenotypes.loc[common_samples, trait]
                
                # Calculate performance metrics for each model if true values available
                for model_name in all_predictions.keys():
                    pred_col = f'{model_name}_predicted'
                    if pred_col in trait_data.columns:
                        # Get common samples with non-missing values
                        mask = trait_data[[pred_col, 'True_value']].notna().all(axis=1)
                        if mask.sum() > 0:
                            y_true = trait_data.loc[mask, 'True_value']
                            y_pred = trait_data.loc[mask, pred_col]
                            try:
                                pearson_r = pearsonr(y_true, y_pred)[0]
                                mse = mean_squared_error(y_true, y_pred)
                                trait_data[f'{model_name}_pearsonr'] = pearson_r
                                trait_data[f'{model_name}_mse'] = mse
                                # Only set once per column
                                if f'{model_name}_pearsonr' not in trait_data.columns:
                                    trait_data[f'{model_name}_pearsonr'] = pearson_r
                                if f'{model_name}_mse' not in trait_data.columns:
                                    trait_data[f'{model_name}_mse'] = mse
                            except:
                                pass
        
        # Save to file
        output_file = os.path.join(output_dir, f"predictions_{config['feature_view']}_trait_{trait}.csv")
        trait_data.to_csv(output_file, index=True)
        print(f"  Saved to: {output_file}")
        
        # Also create a summary statistics file for this trait
        if 'True_value' in trait_data.columns and not trait_data['True_value'].isna().all():
            create_trait_summary(trait_data, trait, output_dir, config)

        cols_to_keep = [c for c in trait_data.columns if c.endswith('_predicted') or c == 'True_value']
        extracted_df = trait_data.loc[:, cols_to_keep].copy()
        fig = pairplot_with_corr(extracted_df)
        plot_path = os.path.join(output_dir, f"prediction_corr_matrix_{config['feature_view']}_trait_{trait}.png")
        fig.savefig(plot_path, dpi=200, bbox_inches='tight')
        plt.close(fig)

def create_trait_summary(trait_data, trait_name, output_dir, config):
    """Create summary statistics for a trait"""
    summary_rows = []
    
    # Get model names from prediction columns
    model_names = []
    for col in trait_data.columns:
        if col.endswith('_predicted'):
            model_names.append(col.replace('_predicted', ''))
    
    for model_name in model_names:
        pred_col = f'{model_name}_predicted'
        pearson_col = f'{model_name}_pearsonr'
        mse_col = f'{model_name}_mse'
        
        if pred_col in trait_data.columns:
            # Basic stats
            pred_values = trait_data[pred_col].dropna()
            if len(pred_values) > 0:
                summary = {
                    'Model': model_name,
                    'n_predictions': len(pred_values),
                    'mean_predicted': pred_values.mean(),
                    'std_predicted': pred_values.std(),
                    'min_predicted': pred_values.min(),
                    'max_predicted': pred_values.max()
                }
                
                # Performance metrics if available
                if pearson_col in trait_data.columns:
                    summary['pearson_r'] = trait_data[pearson_col].iloc[0]  # Same value for all rows
                if mse_col in trait_data.columns:
                    summary['mse'] = trait_data[mse_col].iloc[0]
                
                summary_rows.append(summary)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = os.path.join(output_dir, f"prediction_summary_{config['feature_view']}_trait_{trait_name}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"  Summary saved to: {summary_file}")


# Export predictions for all models and marker views
def export_trait_predictions_all_marker_views(config, all_predictions, test_phenotypes=None, output_dir=None):
    """
    Export predictions for each trait with all models' predictions and true values
    from all marker views.
    
    Args:
        all_predictions: Dictionary of {model_name: {marker_view: prediction_df}}
        test_phenotypes: DataFrame of true test phenotypes (optional)
        output_dir: Directory to save output files
    """
    os.makedirs(output_dir, exist_ok=True)

    # Debug: Check test_phenotypes
    """
    print(f"[DEBUG] test_phenotypes is None: {test_phenotypes is None}")
    if test_phenotypes is not None:
        print(f"[DEBUG] test_phenotypes type: {type(test_phenotypes)}")
        print(f"[DEBUG] test_phenotypes shape: {test_phenotypes.shape if hasattr(test_phenotypes, 'shape') else 'No shape'}")
        print(f"[DEBUG] test_phenotypes columns: {test_phenotypes.columns.tolist() if hasattr(test_phenotypes, 'columns') else 'No columns'}")
        print(f"[DEBUG] test_phenotypes head:\n{test_phenotypes.head() if hasattr(test_phenotypes, 'head') else 'No head method'}")
    else:
        print("[DEBUG] test_phenotypes is None - no true values will be added")
    """

    if not all_predictions:
        print("No predictions to export for multiple marker views.")
        return
        
    # Get all trait names from the first model and first marker view
    first_model = list(all_predictions.keys())[0]
    first_marker_view = list(all_predictions[first_model].keys())[0]
    trait_names = all_predictions[first_model][first_marker_view].columns.tolist()
    
    # Get sample IDs from the first model and first marker view
    sample_ids = all_predictions[first_model][first_marker_view].index.tolist()
    
    # Debug: Check sample alignment
    if test_phenotypes is not None:
        common_samples = set(sample_ids).intersection(set(test_phenotypes.index))
        """
        print(f"[DEBUG] Total samples in predictions: {len(sample_ids)}")
        print(f"[DEBUG] Total samples in test_phenotypes: {len(test_phenotypes.index)}")
        print(f"[DEBUG] Common samples between predictions and test_phenotypes: {len(common_samples)}")
        """

    for trait in trait_names:
        print(f"Exporting all marker view and model predictions for trait: {trait}")
        
        # Create DataFrame for this trait
        trait_data = pd.DataFrame(index=sample_ids)
        trait_data.index.name = 'Sample'
        
        # Add predictions from each model and each marker view
        for model_name, marker_dict in all_predictions.items():
            for marker_view, pred_df in marker_dict.items():
                if trait in pred_df.columns:
                    column_name = f'{model_name}_{marker_view}_predicted'
                    trait_data[column_name] = pred_df[trait]
        
        # Add true values if available
        if test_phenotypes is not None and trait in test_phenotypes.columns:
            #print(f"[DEBUG] Trait '{trait}' found in test_phenotypes")
            
            # Align with prediction samples
            common_samples = trait_data.index.intersection(test_phenotypes.index)
            #print(f"[DEBUG] Common samples for trait '{trait}': {len(common_samples)}")
            
            if len(common_samples) > 0:
                trait_data['True_value'] = np.nan  # Initialize with NaN
                trait_data.loc[common_samples, 'True_value'] = test_phenotypes.loc[common_samples, trait]
                #print(f"[DEBUG] Added True_value for {len(common_samples)} samples")
        
                
                # Calculate performance metrics for each model-marker_view combination if true values available
                for model_name, marker_dict in all_predictions.items():
                    for marker_view in marker_dict.keys():
                        pred_col = f'{model_name}_{marker_view}_predicted'
                        if pred_col in trait_data.columns:
                            # Get common samples with non-missing values
                            mask = trait_data[[pred_col, 'True_value']].notna().all(axis=1)
                            if mask.sum() > 0:
                                y_true = trait_data.loc[mask, 'True_value']
                                y_pred = trait_data.loc[mask, pred_col]
                                try:
                                    pearson_r = pearsonr(y_true, y_pred)[0]
                                    mse = mean_squared_error(y_true, y_pred)
                                    # Add metrics as columns (same value for all rows)
                                    trait_data[f'{model_name}_{marker_view}_pearsonr'] = pearson_r
                                    trait_data[f'{model_name}_{marker_view}_mse'] = mse
                                except Exception as e:
                                    print(f"  Warning: Could not calculate metrics for {model_name}_{marker_view}: {e}")
            else:
                print(f"[DEBUG] WARNING: No common samples found for trait '{trait}'")
        else:
            if test_phenotypes is None:
                print(f"[DEBUG] test_phenotypes is None for trait '{trait}'")
            else:
                print(f"[DEBUG] Trait '{trait}' not found in test_phenotypes columns: {test_phenotypes.columns.tolist()}")

       # Save to file
        output_file = os.path.join(output_dir, f"predictions_all_views_trait_{trait}.csv")
        trait_data.to_csv(output_file, index=True)
        print(f"  Saved to: {output_file}")
        
        # Also create a summary statistics file for this trait
        if 'True_value' in trait_data.columns and not trait_data['True_value'].isna().all():
            create_trait_summary_all_marker_views(trait_data, trait, output_dir, config)

        # Create correlation matrix plot
        cols_to_keep = [c for c in trait_data.columns if c.endswith('_predicted') or c == 'True_value']
        if len(cols_to_keep) > 1:  # Only create plot if we have multiple columns
            extracted_df = trait_data.loc[:, cols_to_keep].copy()
            fig = pairplot_with_corr(extracted_df)
            plot_path = os.path.join(output_dir, f"prediction_corr_matrix_all_views_trait_{trait}.png")
            fig.savefig(plot_path, dpi=200, bbox_inches='tight')
            plt.close(fig)
            print(f"  Correlation plot saved to: {plot_path}")

def create_trait_summary_all_marker_views(trait_data, trait_name, output_dir, config):
    """Create summary statistics for a trait across all models and marker views"""
    summary_rows = []
    
    # Get model-marker_view combinations from prediction columns
    model_marker_combinations = []
    for col in trait_data.columns:
        if col.endswith('_predicted'):
            # Extract model and marker_view from column name (format: model_marker_view_predicted)
            parts = col.replace('_predicted', '').split('_')

            # Assuming format: model_marker_view (e.g., R_RRBLUP_SNP_predicted)
            if len(parts) >= 3:  # Need at least 3 parts: model_part1, model_part2, marker_view
                model_name = f"{parts[0]}_{parts[1]}"  # First two parts are model name
                marker_view = '_'.join(parts[2:])  # Rest is marker_view
                model_marker_combinations.append((model_name, marker_view))
            elif len(parts) >= 2:
                # Fallback: if only 2 parts, assume first is model, second is marker_view
                model_name = parts[0]
                marker_view = parts[1]
                model_marker_combinations.append((model_name, marker_view))
    
    for model_name, marker_view in model_marker_combinations:
        pred_col = f'{model_name}_{marker_view}_predicted'
        pearson_col = f'{model_name}_{marker_view}_pearsonr'
        mse_col = f'{model_name}_{marker_view}_mse'
        
        if pred_col in trait_data.columns:
            # Basic stats
            pred_values = trait_data[pred_col].dropna()
            if len(pred_values) > 0:
                summary = {
                    'Model': model_name,
                    'Marker_View': marker_view,
                    'n_predictions': len(pred_values),
                    'mean_predicted': pred_values.mean(),
                    'std_predicted': pred_values.std(),
                    'min_predicted': pred_values.min(),
                    'max_predicted': pred_values.max()
                }
                
                # Performance metrics if available
                if pearson_col in trait_data.columns:
                    summary['pearson_r'] = trait_data[pearson_col].iloc[0]  # Same value for all rows
                if mse_col in trait_data.columns:
                    summary['mse'] = trait_data[mse_col].iloc[0]
                
                summary_rows.append(summary)
    
    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_file = os.path.join(output_dir, f"prediction_summary_all_views_trait_{trait_name}.csv")
        summary_df.to_csv(summary_file, index=False)
        print(f"  Summary saved to: {summary_file}")


def plot_pc_scatter(train_pcs: pd.DataFrame, test_pcs: pd.DataFrame, pca, out_dir: str):
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(train_pcs.iloc[:,0], train_pcs.iloc[:,1], alpha=0.6, label='TRAIN')
    plt.scatter(test_pcs.iloc[:,0],  test_pcs.iloc[:,1],  alpha=0.6, label='TEST')
    # Add variance explained percentages if available
    try:
        var1 = float(pca.explained_variance_ratio_[0]) * 100.0
        var2 = float(pca.explained_variance_ratio_[1]) * 100.0
        xlab, ylab = f'PC1 ({var1:.1f}%)', f'PC2 ({var2:.1f}%)'
    except Exception:
        xlab, ylab = 'PC1', 'PC2'
    plt.xlabel(xlab); plt.ylabel(ylab); plt.legend()
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, "pc_scatter_train_vs_test.png")
    plt.tight_layout()
    plt.savefig(out, dpi=200)
    plt.close()
    print(f"Saved PC1 vs PC2 scatter to: {out}")

# ==== Distance-weighting utilities (added) ====
import numpy as _np
import matplotlib.pyplot as _plt

def _compute_grm(X):
    Xc = X - _np.mean(X, axis=0)
    m = X.shape[1]
    return (Xc @ Xc.T) / float(m)

def _compute_cross_grm(X_test, X_train):
    mu = _np.mean(X_train, axis=0)
    Xt = X_test - mu
    Xr = X_train - mu
    m = X_train.shape[1]
    return (Xt @ Xr.T) / float(m)

def _compute_sample_weights_from_similarity(S_test_train, mode="gaussian", sigma=None, k=None,
                                            agg="mean", normalize="mean1"):
    D = 1.0 - S_test_train
    if agg == "mean":
        d = D.mean(axis=0)
    elif agg == "min":
        d = D.min(axis=0)
    elif isinstance(agg, float):
        d = _np.quantile(D, agg, axis=0)
    else:
        d = D.mean(axis=0)
    if mode == "gaussian":
        if sigma is None or (isinstance(sigma, str) and sigma.lower()=="auto"):
            sigma = _np.median(d) + 1e-12
        w = _np.exp(-(d**2)/(2*sigma**2))
    elif mode == "knn":
        if k is None:
            k = max(5, int(0.1*len(d)))
        ranks = _np.argsort(_np.argsort(d))
        w = (ranks < k).astype(float)
    elif mode == "threshold":
        thr = _np.median(d)
        w = (d <= thr).astype(float)
    else:
        raise ValueError("Unknown mode")
    if normalize == "mean1":
        w = w * (len(w)/(w.sum()+1e-12))
    elif normalize == "sum1":
        w = w/(w.sum()+1e-12)
    return w

def _apply_wls_transform(X, y, w):
    sw = _np.sqrt(_np.asarray(w)).reshape(-1,1)
    Xw = X * sw
    yw = y * (sw.ravel() if y.ndim==1 else sw)
    return Xw, yw

def plot_mds_from_grm(train_geno_df, test_geno_df, out_png, out_mds_values, title="MDS (GRM-based)"):
    import numpy as np
    from numpy.linalg import eigh
    train = train_geno_df.values
    test  = test_geno_df.values
    X = np.vstack([train, test])
    G = _compute_grm(X)
    diag = np.diag(G)
    D2 = diag[:,None] + diag[None,:] - 2.0*G
    n = D2.shape[0]
    J = np.eye(n) - np.ones((n,n))/n
    B = -0.5 * (J @ D2 @ J)
    vals, vecs = eigh(B)
    idx = np.argsort(vals)[::-1]
    vals = vals[idx]; vecs = vecs[:, idx]
    k = 2
    coords = vecs[:, :k] * np.sqrt(np.maximum(vals[:k], 0))
    ntr = train.shape[0]
    c0, c1 = coords[:ntr], coords[ntr:]

    # ---- save MDS values to file ----
    ids = list(train_geno_df.index) + list(test_geno_df.index)
    groups = ["Training"] * ntr + ["Test"] * (coords.shape[0] - ntr)

    mds_df = pd.DataFrame({
        "SampleID": ids,
        "Group": groups,
        "MDS1": coords[:, 0],
        "MDS2": coords[:, 1]
    })

    mds_df.to_csv(out_mds_values, sep="\t", index=False)

    # ---- plot ----
    _plt.figure(figsize=(6.4,5.2))
    _plt.scatter(c0[:,0], c0[:,1], label="Training", alpha=0.7)
    _plt.scatter(c1[:,0], c1[:,1], label="Test", alpha=0.7)
    _plt.xlabel("MDS1"); _plt.ylabel("MDS2"); _plt.title(title); _plt.legend(); _plt.tight_layout()
    _plt.savefig(out_png, dpi=150); _plt.close()
# ==== End Distance-weighting utilities ====


# Add this class definition before the AllGSModels class
class GenomicTransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, d_model=512, nhead=8, 
                 num_layers=6, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, dropout)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(d_model, 256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.GELU(), 
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        
        # Project to transformer dimension
        x = self.input_projection(x)  # (batch_size, d_model)
        x = x.unsqueeze(1)  # (batch_size, 1, d_model) - treat as sequence of length 1
        
        # Add positional encoding and apply transformer
        x = self.pos_encoding(x)
        x = self.transformer(x)  # (batch_size, 1, d_model)
        
        # Global average pooling and output
        x = x.mean(dim=1)  # (batch_size, d_model)
        x = self.output_layers(x)  # (batch_size, output_dim)
        
        return x    

import torch
from torch.utils.data import Dataset 
class GenomicDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


"""
CNNModel is a standard 1D CNN for regression or classification, with the following components:
1. 1D Convolutional Layers
    Extract local patterns from sequential features (e.g., genomic markers).
    Kernel size = 3 (small receptive field).
2. Padding = kernel_size // 2 → preserves input length before pooling.
    Pooling Layers
    Max pooling reduces sequence length (pool_size=2) after each conv.
3. Dropout Layers
    Applied after each conv+pool block for regularization.
4. Fully Connected Layers
    Flatten CNN output → linear layers → output.
    Single hidden FC layer with ReLU + dropout.
Compared with your AttnCNNGSModel, this cannot capture long-range dependencies between distant positions/features.
"""
class CNNModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_channels=[64, 32], 
                 kernel_size=3, pool_size=2, dropout=0.2):
        super(CNNModel, self).__init__()
        
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        # Calculate input channels for each layer
        in_channels = 1  # Treat as 1D signal with 1 input channel
        current_length = input_dim
        
        for out_channels in hidden_channels:
            # Conv layer
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2)
            self.conv_layers.append(conv)
            
            # Pooling layer
            pool = nn.MaxPool1d(pool_size)
            self.pool_layers.append(pool)
            
            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
            current_length = current_length // pool_size
        
        # Calculate flattened size
        self.flattened_size = in_channels * current_length
        
        # Fully connected layers
        self.fc_layers = nn.Sequential(
            nn.Linear(self.flattened_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, output_dim)
        )
    
    def forward(self, x):
        # Add channel dimension: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Apply CNN layers
        for conv, pool, dropout in zip(self.conv_layers, self.pool_layers, self.dropout_layers):
            x = torch.relu(conv(x))
            x = pool(x)
            x = dropout(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc_layers(x)
        return x

    
        if model_name == 'EnsembleGS':
            return self._predict_stacking(X_test)

        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' is not trained.")

        m = self.models[model_name]
        if model_name == 'CNN':
            return self.predict_cnn(m, X_test)
        if model_name == 'MLPGS':
            return self.predict_mlp(m, X_test)

        # default: scikit-learn-style estimators
        return m.predict(X_test)

class AttnCNNGSModel_v0(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 cnn_channels=[128, 128, 256], kernel_size=5, pool_size=2,
                 attention_heads=8, hidden_size=256, 
                 dropout=0.5, attention_dropout=0.3):
        super(AttnCNNGSModel, self).__init__()
        
        # CNN Backbone (Set 2 inspired)
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_channels = 1  # Treat as 1D signal with 1 input channel
        current_length = input_dim
        
        for out_channels in cnn_channels:
            # Conv layer
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                           padding=kernel_size//2, padding_mode='zeros')
            self.conv_layers.append(conv)
            
            # Pooling layer
            pool = nn.MaxPool1d(pool_size)
            self.pool_layers.append(pool)
            
            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
            current_length = current_length // pool_size
        
        # Calculate CNN output size
        self.cnn_output_size = in_channels * current_length
        
        # Attention Layer (Set 1 inspired)
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        
        # Project CNN features to attention dimension
        self.cnn_to_attention = nn.Linear(self.cnn_output_size, hidden_size)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        # Add channel dimension: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Apply CNN layers
        for conv, pool, dropout in zip(self.conv_layers, self.pool_layers, self.dropout_layers):
            x = torch.relu(conv(x))
            x = pool(x)
            x = dropout(x)
        
        # Flatten CNN output: (batch, channels, length) -> (batch, channels * length)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        
        # Project to attention dimension
        x = self.cnn_to_attention(x)  # (batch, hidden_size)
        
        # Prepare for attention: add sequence dimension
        x = x.unsqueeze(1)  # (batch, 1, hidden_size)
        
        # Self-attention
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        # Remove sequence dimension and get final output
        x = x.squeeze(1)
        x = self.output_layers(x)
        
        return x

"""
AttnCNNGS: a hybrid CNN + Multi-Head Attention (Transformer-style) + MLP for regression/classification:
CNN Backbone – Extracts local feature patterns from 1D inputs (genomic or other sequence-like data).
Linear Projection – Maps CNN output to a “hidden size” suitable for multi-head attention.
Multi-Head Attention – Captures global dependencies across the (flattened) CNN features.
Feed-Forward Network (FFN) – Standard Transformer-style FFN with GELU and dropout.
Output MLP – Final mapping to the desired output dimension.
"""
class AttnCNNGSModel(nn.Module):
    def __init__(self, input_dim, output_dim, 
                 cnn_channels=[128, 128, 256], kernel_size=5, pool_size=2,
                 attention_heads=8, hidden_size=256, 
                 dropout=0.5, attention_dropout=0.3):
        super(AttnCNNGSModel, self).__init__()
        
        # CNN Backbone with BatchNorm
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()
        
        in_channels = 1  # Treat input as 1D signal with 1 channel
        current_length = input_dim
        
        for out_channels in cnn_channels:
            # Conv layer
            conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                             padding=kernel_size//2, padding_mode='zeros')
            self.conv_layers.append(conv)
            
            # BatchNorm
            bn = nn.BatchNorm1d(out_channels)
            self.bn_layers.append(bn)
            
            # Pooling
            pool = nn.MaxPool1d(pool_size)
            self.pool_layers.append(pool)
            
            # Dropout
            self.dropout_layers.append(nn.Dropout(dropout))
            
            in_channels = out_channels
            current_length = current_length // pool_size
        
        self.cnn_output_channels = in_channels
        self.cnn_output_length = current_length
        
        # Attention layer
        self.attention_heads = attention_heads
        self.hidden_size = hidden_size
        
        # Project CNN features to attention dimension
        self.cnn_to_attention = nn.Linear(self.cnn_output_channels, hidden_size)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=attention_heads,
            dropout=attention_dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm1 = nn.LayerNorm(hidden_size)
        self.layer_norm2 = nn.LayerNorm(hidden_size)
        
        # Feed-forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 4, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Output layers
        self.output_layers = nn.Sequential(
            nn.Linear(hidden_size, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        # Add channel dimension: (batch, features) -> (batch, 1, features)
        x = x.unsqueeze(1)
        
        # Apply CNN layers with BatchNorm, Pooling, Dropout
        for conv, bn, pool, dropout in zip(self.conv_layers, self.bn_layers, 
                                           self.pool_layers, self.dropout_layers):
            x = torch.relu(bn(conv(x)))
            x = pool(x)
            x = dropout(x)
        
        # Prepare sequence for attention: (batch, channels, length) -> (batch, length, channels)
        x = x.permute(0, 2, 1)  # sequence length = length dimension
        
        # Project each feature vector to hidden_size
        x = self.cnn_to_attention(x)  # shape: (batch, seq_len, hidden_size)
        
        # Multi-head self-attention
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.layer_norm1(x + attn_output)
        
        # Feed-forward
        ff_output = self.feed_forward(x)
        x = self.layer_norm2(x + ff_output)
        
        # Aggregate sequence dimension (mean pooling)
        x = x.mean(dim=1)  # shape: (batch, hidden_size)
        
        # Output layers
        x = self.output_layers(x)
        
        return x


class InputDropout(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = float(p)
    def forward(self, x):
        if not self.training or self.p <= 0.0:
            return x
        mask = torch.empty(1, x.shape[1], device=x.device).bernoulli_(1 - self.p)
        return x * mask

"""
1.  AttnCNNGS branch
    Captures local patterns via CNN + global dependencies via attention.
    BatchNorm and dropout included.
2. MLP branch
    Directly models raw input features.
    Residual connections fixed: if dimensions mismatch, linear projection aligns shapes.
3.Feature fusion
    Concatenates AttnCNNGS output + MLP output before final output layer.
    Lightweight: only one additional MLP branch; does not significantly increase parameters.
4. Regularization
    Dropout in both branches + final layer.
5. Flexibility
You can adjust CNN channels, attention heads, hidden size, MLP hidden layers, dropout.

Benefits
    Leverages both sequence-aware patterns and raw feature modeling.
    Fixes MLP residual issue: projection ensures residuals are applied consistently.
    Lightweight: avoids stacking multiple attention layers unnecessarily.
    Easy to integrate into existing genomic selection pipeline.
"""
def parse_list(x):
    if isinstance(x, list):
        return x
    if isinstance(x, str):
        return [int(i.strip()) for i in x.split(',') if i.strip()]
    raise ValueError(f"Cannot parse list from: {x}")

# simple version
class MLPGSModel(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[128, 64],
                 dropout=0.2, norm='layer', activation='gelu', residual=True, input_dropout=0.05):
        super().__init__()
        
        act_name = str(activation).lower()
        self.act = nn.GELU() if act_name == 'gelu' else nn.ReLU()
        self.residual = bool(residual)
        self.input_do = nn.Dropout(float(input_dropout))
        self.blocks = nn.ModuleList()
        self.norms  = nn.ModuleList()
        
        prev = input_dim
        for h in hidden_layers:
            self.blocks.append(nn.Sequential(
                nn.Linear(prev, h),
                self.act,
                nn.Dropout(dropout)
            ))
            n = str(norm).lower()
            if n == 'layer':
                self.norms.append(nn.LayerNorm(h))
            elif n == 'batch':
                self.norms.append(nn.BatchNorm1d(h))
            else:
                self.norms.append(nn.Identity())
            prev = h
        
        # More stable output layer
        self.output_norm = nn.LayerNorm(prev)
        self.out = nn.Linear(prev, output_dim)
        
    def forward(self, x):
        x = self.input_do(x)
        h = x
        for block, nm in zip(self.blocks, self.norms):
            z = block(h)
            z = nm(z)
            if self.residual and z.shape == h.shape:
                h = z + h
            else:
                h = z
        
        h = self.output_norm(h)
        return self.out(h)

    
    def predict(self, x):
        """Prediction method that handles inference mode"""
        self.eval()  # Set to evaluation mode
        with torch.no_grad():  # Disable gradient computation
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            return self.forward(x).cpu().numpy()

class MLPBlock(nn.Module):
    """Single MLP block with optional residual and projection to match dims."""
    def __init__(self, in_dim, out_dim, activation='gelu', dropout=0.2, norm='layer', residual=True):
        super().__init__()
        act_name = str(activation).lower()
        self.act = nn.GELU() if act_name == 'gelu' else nn.ReLU()
        self.residual = bool(residual)

        self.linear = nn.Linear(in_dim, out_dim)
        self.dropout = nn.Dropout(dropout)

        n = str(norm).lower()
        if n == 'layer':
            self.norm = nn.LayerNorm(out_dim)
        elif n == 'batch':
            # BatchNorm1d expects (batch, features) for MLP outputs
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            self.norm = nn.Identity()

        # projection to match dimensions when residual requested but dims differ
        if self.residual and in_dim != out_dim:
            self.res_proj = nn.Linear(in_dim, out_dim)
        else:
            self.res_proj = None

    def forward(self, x):
        out = self.linear(x)
        out = self.act(out)
        out = self.dropout(out)
        # BatchNorm1d expects 2D: (batch, features) — it's fine here.
        out = self.norm(out)
        if self.residual:
            res = x
            if self.res_proj is not None:
                res = self.res_proj(x)
            out = out + res
        return out

class HybridAttnMLPModel(nn.Module):
    """
    Safe, cleaned HybridAttnMLP model.
    Avoids mutable default args and re-used module instances.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 cnn_channels=None,
                 kernel_size=5,
                 pool_size=2,
                 attention_heads=4,
                 hidden_size=128,
                 mlp_hidden=None,
                 dropout=0.2,
                 input_dropout=0.0,
                 device=None):
        super().__init__()

        # avoid mutable default lists
        if cnn_channels is None:
            cnn_channels = [64, 128]
        if mlp_hidden is None:
            mlp_hidden = [128, 64]

        # optional device store
        self.device = device

        # input dropout
        self.input_dropout = nn.Dropout(float(input_dropout))

        # --- AttnCNNGS branch ---
        self.conv_layers = nn.ModuleList()
        self.pool_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.dropout_layers = nn.ModuleList()

        in_ch = 1
        seq_len = int(input_dim)
        for ch in cnn_channels:
            self.conv_layers.append(
                nn.Conv1d(in_ch, int(ch), kernel_size, padding=kernel_size // 2)
            )
            self.bn_layers.append(nn.BatchNorm1d(int(ch)))
            self.pool_layers.append(nn.MaxPool1d(pool_size))
            self.dropout_layers.append(nn.Dropout(float(dropout)))
            in_ch = int(ch)
            seq_len = seq_len // int(pool_size)

        self.attn_input_channels = in_ch
        self.attn_seq_len = seq_len
        self.cnn_to_attn = nn.Linear(in_ch, int(hidden_size))
        self.attention = nn.MultiheadAttention(
            embed_dim=int(hidden_size),
            num_heads=int(attention_heads),
            dropout=float(dropout),
            batch_first=True
        )
        # small feed-forward inside attention block
        self.layer_norm1 = nn.LayerNorm(int(hidden_size))
        self.ffn = nn.Sequential(
            nn.Linear(int(hidden_size), int(hidden_size) * 4),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(hidden_size) * 4, int(hidden_size)),
            nn.Dropout(float(dropout))
        )
        self.layer_norm2 = nn.LayerNorm(int(hidden_size))

        # --- MLP branch ---
        self.mlp_blocks = nn.ModuleList()
        prev_dim = int(input_dim)
        for h in mlp_hidden:
            # create a fresh MLPBlock instance for each layer (no shared modules)
            block = MLPBlock(prev_dim, int(h), activation='gelu', dropout=dropout, norm='layer', residual=True)
            self.mlp_blocks.append(block)
            prev_dim = int(h)

        # --- Output ---
        self.output_layer = nn.Sequential(
            nn.Linear(int(hidden_size) + prev_dim, 128),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(128, int(output_dim))
        )

    def forward(self, x):
        # x : (batch, input_dim)
        x = self.input_dropout(x)

        # --- AttnCNNGS branch ---
        attn_x = x.unsqueeze(1)  # (batch, 1, features)
        for conv, bn, pool, do in zip(self.conv_layers, self.bn_layers, self.pool_layers, self.dropout_layers):
            attn_x = conv(attn_x)           # (batch, channels, length)
            attn_x = bn(attn_x)
            attn_x = torch.relu(attn_x)
            attn_x = pool(attn_x)
            attn_x = do(attn_x)

        # to (batch, seq_len, channels)
        attn_x = attn_x.permute(0, 2, 1)
        attn_x = self.cnn_to_attn(attn_x)   # project channels -> hidden_size
        attn_out, _ = self.attention(attn_x, attn_x, attn_x)
        attn_x = self.layer_norm1(attn_x + attn_out)
        ffn_out = self.ffn(attn_x)
        attn_x = self.layer_norm2(attn_x + ffn_out)
        attn_x = attn_x.mean(dim=1)         # aggregate sequence -> (batch, hidden_size)

        # --- MLP branch ---
        mlp_x = x
        for block in self.mlp_blocks:
            mlp_x = block(mlp_x)

        # --- Concatenate and output ---
        combined = torch.cat([attn_x, mlp_x], dim=1)
        return self.output_layer(combined)


class DNNGSModel_v1(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_layers=[512, 256, 128, 64],
                 dropout=0.3, activation='relu', batch_norm=True, input_dropout=0.1):
        super(DNNGSModel, self).__init__()
        
        # Activation function
        if activation.lower() == 'relu':
            self.activation = nn.ReLU()
        elif activation.lower() == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        elif activation.lower() == 'gelu':
            self.activation = nn.GELU()
        else:
            self.activation = nn.ReLU()
        
        self.batch_norm = batch_norm
        self.input_dropout = nn.Dropout(input_dropout)
        
        # Build layers
        layers = []
        prev_size = input_dim
        
        for i, hidden_size in enumerate(hidden_layers):
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(self.activation)
            layers.append(nn.Dropout(dropout))
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_dim))
        
        self.network = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm1d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.input_dropout(x)
        return self.network(x)

# Improved version (V2) of DNNGSModel. Here directly replace the original
import torch
import torch.nn as nn
import math
class ResidualMLPBlock(nn.Module):
    """
    Residual block: Linear -> Norm -> Activation -> Dropout + shortcut.
    Works even if in_dim != out_dim via a projection shortcut.
    """
    def __init__(self, in_dim, out_dim, activation, dropout=0.1,
                 norm_type='layernorm'):
        super().__init__()

        self.fc = nn.Linear(in_dim, out_dim)

        if norm_type is None or norm_type.lower() == 'none':
            self.norm = None
        elif norm_type.lower() == 'layernorm':
            self.norm = nn.LayerNorm(out_dim)
        elif norm_type.lower() == 'batchnorm':
            # For completeness; not recommended for SNP data
            self.norm = nn.BatchNorm1d(out_dim)
        else:
            raise ValueError(f"Unknown norm_type: {norm_type}")

        self.activation = activation
        self.dropout = nn.Dropout(dropout)

        # Projection for residual if dimensions don't match
        if in_dim != out_dim:
            self.shortcut = nn.Linear(in_dim, out_dim)
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        residual = x

        out = self.fc(x)
        # Handle BatchNorm vs LayerNorm dimension order
        if isinstance(self.norm, nn.BatchNorm1d):
            out = self.norm(out)
        elif isinstance(self.norm, nn.LayerNorm):
            out = self.norm(out)

        out = self.activation(out)
        out = self.dropout(out)

        residual = self.shortcut(residual)
        out = out + residual
        return out
class DNNGSModel(nn.Module):
    """
    DNNGS v2:
      - Residual MLP blocks
      - LayerNorm (default)
      - GELU activation (default)
      - Lighter regularization tuned for GS with ~2k samples

    Interface is compatible with your existing DNNGSModel.
    """
    def __init__(self,
                 input_dim,
                 output_dim,
                 hidden_layers=[512, 256, 128],
                 dropout=0.15,
                 activation='gelu',
                 batch_norm=False,      # kept for backward compatibility, but ignored if norm_type is set
                 input_dropout=0.0,
                 norm_type='layernorm',  # 'layernorm' | 'batchnorm' | 'none'
                 residual=True):
        super().__init__()

        # -------- Activation --------
        act_name = activation.lower() if isinstance(activation, str) else 'gelu'
        if isinstance(activation, str):
            if act_name == 'relu':
                self.activation = nn.ReLU()
            elif act_name == 'leaky_relu':
                self.activation = nn.LeakyReLU(0.1)
            elif act_name == 'gelu':
                self.activation = nn.GELU()
            else:
                self.activation = nn.ReLU()
        else:
            # if user passes nn.Module instance
            self.activation = activation

        # Norm selection
        # if user still uses batch_norm=True and didn't specify norm_type, honor that
        if norm_type is None:
            if batch_norm:
                norm_type = 'batchnorm'
            else:
                norm_type = 'none'
        self.norm_type = norm_type.lower() if isinstance(norm_type, str) else 'layernorm'

        # Input dropout
        self.input_dropout = nn.Dropout(input_dropout) if input_dropout and input_dropout > 0 else nn.Identity()

        # -------- Build network as residual blocks --------
        layers = []
        prev_dim = input_dim

        for hidden_size in hidden_layers:
            if residual:
                block = ResidualMLPBlock(
                    in_dim=prev_dim,
                    out_dim=hidden_size,
                    activation=self.activation,
                    dropout=dropout,
                    norm_type=self.norm_type
                )
                layers.append(block)
            else:
                # Fallback: plain MLP layer
                seq = []
                seq.append(nn.Linear(prev_dim, hidden_size))

                if self.norm_type == 'layernorm':
                    seq.append(nn.LayerNorm(hidden_size))
                elif self.norm_type == 'batchnorm':
                    seq.append(nn.BatchNorm1d(hidden_size))

                seq.append(self.activation)
                seq.append(nn.Dropout(dropout))
                layers.append(nn.Sequential(*seq))

            prev_dim = hidden_size

        self.blocks = nn.ModuleList(layers)
        self.output_layer = nn.Linear(prev_dim, output_dim)

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Kaiming init suited for ReLU-like activations (ReLU/GELU/LeakyReLU)
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0.)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.constant_(module.weight, 1.0)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.constant_(module.bias, 0.0)

    def forward(self, x):
        # x: (batch_size, input_dim)
        x = self.input_dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.output_layer(x)
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        x = x + self.pe[:x.size(1)]  # Use sequence length dimension
        return self.dropout(x)

  
# Mixed model RRBLUP
import numpy as np
from scipy.linalg import solve, cholesky, cho_solve
import warnings

class HighPerformanceRRBLUP:
    """
    Python RR-BLUP implementation matching R rrBLUP package performance
    Uses mixed model equations with kinship matrix
    """
    
    def __init__(self, lambda_value=None, method='mixed_model', 
             lambda_method='auto', tol=1e-8):
        self.lambda_value = lambda_value
        self.method = method
        self.lambda_method = lambda_method  # 'auto', 'reml', 'heritability', 'fixed'
        self.tol = tol
        self.B = None
        self.X_mean = None
        self.y_mean = None
        self.is_fitted = False
        self.kinship_matrix = None
        self.marker_effects = None
        
    def _center_matrix(self, X):
        """Center genotype matrix like R rrBLUP"""
        return X - np.mean(X, axis=0)
    
    def _compute_kinship_improved(self, X):
        """Improved kinship computation with allele frequency adjustment"""
        X_centered = X - np.mean(X, axis=0)
        n, p = X_centered.shape
    
        # Estimate allele frequencies for better standardization
        freqs = np.mean(X, axis=0) / 2  # Assuming 0,1,2 coding
        # Avoid division by zero
        freqs = np.clip(freqs, 0.01, 0.99)
    
        # VanRaden method 1 with actual allele frequencies
        # Standardize by 2 * sum(p*(1-p)) as in VanRaden 2008
        scale_factor = 2 * np.sum(freqs * (1 - freqs))
    
        # Handle case where scale_factor is too small
        if scale_factor < 1e-10:
            scale_factor = p * 0.5  # Fallback to simple scaling
    
        W = X_centered / np.sqrt(scale_factor)
        K = W @ W.T  # This should be n x n
    
        #print(f"DEBUG: Kinship matrix shape: {K.shape}, expected: ({n}, {n})")
        return K, W, X_centered

    def _compute_kinship(self, X):
        """Compute genomic relationship matrix (VanRaden method 1)"""
        X_centered = self._center_matrix(X)
        p = X_centered.shape[1]
        
        # VanRaden method 1 - same as R rrBLUP
        # Standardize by number of markers and expected variance
        Z = X_centered / np.sqrt(p * 0.5)  # Assuming allele freq ~0.5
        K = Z @ Z.T  # Genomic relationship matrix
        
        return K, Z, X_centered
    
    def _estimate_lambda(self, K, y, Z):
        """Estimate optimal lambda using REML-like approach"""
        n = K.shape[0]
        
        # Simple heuristic that works well in practice
        # This approximates the REML estimation in R rrBLUP
        trace_K = np.trace(K)
        lambda_est = (n - 1) / trace_K if trace_K > 0 else 1.0
        
        print(f"Estimated lambda: {lambda_est:.6f}")
        return lambda_est
    
    def fit_mixed_model(self, X, y):
        """Enhanced mixed model implementation matching R rrBLUP"""
        try:
            n, p = X.shape
            #print(f"DEBUG: Input X shape: {X.shape}, y shape: {y.shape}")
            # Basic sanity check
            if n != len(y):
                raise ValueError(f"Sample count mismatch: X has {n} samples, y has {len(y)}")
        
            # Check if data appears to be PCs (already centered, low correlation)
            if self._is_pc_data(X):
                print("Detected PC data, using direct method instead of mixed model")
                return self.fit_direct_method(X, y)
    
            # Center phenotypes
            self.y_mean = np.mean(y)
            y_centered = y - self.y_mean
    
            # Compute improved kinship matrix
            K, Z, X_centered = self._compute_kinship_improved(X)
            self.kinship_matrix = K
            self.X_mean = np.mean(X, axis=0)
    
            # DEBUG: Check dimensions
            #print(f"DEBUG: K shape: {K.shape}, y_centered shape: {y_centered.shape}")
        
            # Verify kinship matrix dimensions
            if K.shape[0] != n or K.shape[1] != n:
                raise ValueError(f"Kinship matrix has wrong dimensions: {K.shape}, expected: ({n}, {n})")
    
            # Estimate lambda if not provided
            if self.lambda_value is None:
                # Use auto method by default
                self.lambda_value = self._estimate_lambda_auto(K, y_centered, X)
    
            print(f"Using lambda: {self.lambda_value:.6f}")
    
            # Mixed model equations: Solve (K + λI)u = y
            if n < 2000:  # Direct solve for small to medium datasets
                try:
                    # Use Cholesky decomposition for stability
                    lhs = K + self.lambda_value * np.eye(n)
                    L = cholesky(lhs, lower=True, check_finite=False)
                    u = cho_solve((L, True), y_centered, check_finite=False)
                except np.linalg.LinAlgError:
                    # Use SVD-based solve for numerical stability
                    U, s, Vt = np.linalg.svd(K, full_matrices=False)
                    s_regularized = s + self.lambda_value
                    u = U @ (np.diag(1.0 / s_regularized) @ (U.T @ y_centered))
            else:
                # Iterative solver for large datasets
                from scipy.sparse.linalg import cg
                lhs = K + self.lambda_value * np.eye(n)
                #u, info = cg(lhs, y_centered, tol=self.tol, maxiter=1000)   #  deprecated
                u, info = cg(lhs, y_centered, rtol=self.tol, atol=0.0, maxiter=1000)

                if info != 0:
                    warnings.warn("CG solver did not converge, using direct solve")
                    u = solve(lhs, y_centered, assume_a='pos')
    
            # Calculate marker effects: β = Z'u
            # But Z should be the centered and scaled genotype matrix
            # Recompute Z properly for marker effects
            scale_factor = 2 * np.sum(self._estimate_allele_freqs(X) * (1 - self._estimate_allele_freqs(X)))
            if scale_factor < 1e-10:
                scale_factor = p * 0.5
            Z_proper = X_centered / np.sqrt(scale_factor)
        
            self.marker_effects = Z_proper.T @ u
            self.B = self.marker_effects  # For compatibility
    
            self.is_fitted = True
            print(f"RRBLUP mixed model fitted: λ={self.lambda_value:.6f}")
            return self
    
        except Exception as e:
            print(f"Error in RRBLUP mixed model fitting: {e}")
            # Fall back to direct method
            print("Falling back to direct method...")
            return self.fit_direct_method(X, y)

        """Enhanced mixed model implementation matching R rrBLUP"""
        try:
            n, p = X.shape
        
            # Check if data appears to be PCs (already centered, low correlation)
            if self._is_pc_data(X):
                print("Detected PC data, using direct method instead of mixed model")
                return self.fit_direct_method(X, y)
        
            # Center phenotypes
            self.y_mean = np.mean(y)
            y_centered = y - self.y_mean
        
            # Compute improved kinship matrix
            K, Z, X_centered = self._compute_kinship_improved(X)
            self.kinship_matrix = K
            self.X_mean = np.mean(X, axis=0)
        
            # Estimate lambda if not provided
            if self.lambda_value is None:
                # Use auto method by default
                self.lambda_value = self._estimate_lambda_auto(K, y_centered, X)
        
            print(f"Using lambda: {self.lambda_value:.6f}")
        
            # Mixed model equations: Solve (K + λI)u = y
            if n < 2000:  # Direct solve for small to medium datasets
                try:
                    # Use Cholesky decomposition for stability
                    lhs = K + self.lambda_value * np.eye(n)
                    L = cholesky(lhs, lower=True, check_finite=False)
                    u = cho_solve((L, True), y_centered, check_finite=False)
                except np.linalg.LinAlgError:
                    # Use SVD-based solve for numerical stability
                    U, s, Vt = np.linalg.svd(K, full_matrices=False)
                    s_regularized = s + self.lambda_value
                    u = U @ (np.diag(1.0 / s_regularized) @ (U.T @ y_centered))
            else:
                # Iterative solver for large datasets
                from scipy.sparse.linalg import cg
                lhs = K + self.lambda_value * np.eye(n)
                u, info = cg(lhs, y_centered, tol=self.tol, maxiter=1000)
                if info != 0:
                    warnings.warn("CG solver did not converge, using direct solve")
                    u = solve(lhs, y_centered, assume_a='pos')
        
            # Calculate marker effects: β = Z'u
            self.marker_effects = Z.T @ u
            self.B = self.marker_effects  # For compatibility
        
            self.is_fitted = True
            print(f"RRBLUP mixed model fitted: λ={self.lambda_value:.6f}")
            return self
        
        except Exception as e:
            print(f"Error in RRBLUP mixed model fitting: {e}")
            raise

    def fit_direct_method(self, X, y):
        """Direct method for comparison"""
        try:
            n, p = X.shape
            
            # Center data
            self.y_mean = np.mean(y)
            y_centered = y - self.y_mean
            self.X_mean = np.mean(X, axis=0)
            X_centered = X - self.X_mean
            
            if self.lambda_value is None:
                # Use same heuristic as mixed model
                self.lambda_value = (n - 1) / (p * 0.5)
            
            # Standard ridge regression
            XtX = X_centered.T @ X_centered
            Xty = X_centered.T @ y_centered
            
            # Add regularization with condition number check
            I = np.eye(p)
            lhs = XtX + self.lambda_value * I
            
            # Use more stable solver
            try:
                self.B = solve(lhs, Xty, assume_a='pos', check_finite=False)
            except np.linalg.LinAlgError:
                # Add small identity for numerical stability
                lhs += 1e-10 * np.eye(p)
                self.B = solve(lhs, Xty, assume_a='pos', check_finite=False)
            
            self.marker_effects = self.B
            self.is_fitted = True
            print(f"RRBLUP direct method fitted: λ={self.lambda_value:.6f}")
            return self
            
        except Exception as e:
            print(f"Error in RRBLUP direct method fitting: {e}")
            raise
    
    def fit(self, X, y):
        """Main fit method"""
        if y.ndim > 1:
            y = y.ravel()
            
        n, p = X.shape
        print(f"HighPerformanceRRBLUP: {n} samples, {p} markers, method={self.method}")
        
        if self.method == 'mixed_model':
            return self.fit_mixed_model(X, y)
        else:
            return self.fit_direct_method(X, y)
    
    def predict(self, X):
        """Predict genetic values"""
        if not self.is_fitted:
            raise ValueError("RRBLUP model not fitted yet")
        
        X_centered = X - self.X_mean
        predictions = X_centered @ self.B + self.y_mean
        
        return predictions
    
    def get_marker_effects(self):
        """Get marker effects"""
        return self.marker_effects if self.is_fitted else None
    
    def get_kinship_matrix(self):
        """Get kinship matrix (only for mixed model)"""
        return self.kinship_matrix if self.method == 'mixed_model' else None
    
    def score(self, X, y):
        """Calculate prediction accuracy"""
        from scipy.stats import pearsonr
        y_pred = self.predict(X)
        return pearsonr(y, y_pred)[0]

    # newlly added methods    
    def _is_pc_data(self, X):
        """Check if data appears to be principal components"""
        # PCs are centered and have low correlations
        means = np.mean(X, axis=0)
        corr_matrix = np.corrcoef(X.T)
        np.fill_diagonal(corr_matrix, 0)  # Remove self-correlations
        
        is_centered = np.all(np.abs(means) < 1e-10)
        low_correlations = np.mean(np.abs(corr_matrix)) < 0.1
        
        return is_centered and low_correlations
    
    def _estimate_lambda_auto(self, K, y, X):
        """Automatically choose the best lambda estimation method"""
        n, p = X.shape
        
        if n < 1000 and p < 5000:
            # Use REML-like for smaller datasets
            return self._estimate_lambda_reml_approximation(K, y)
        else:
            # Use heritability-based for larger datasets
            return self._estimate_lambda_heritability(K, y)
    
    def _estimate_lambda_heritability(self, K, y):
        """Heritability-based lambda estimation"""
        n = len(y)
        y_var = np.var(y)
        
        if y_var < 1e-10:
            return 1.0
        
        # Estimate heritability from data
        K_trace = np.trace(K)
        K_mean = np.mean(K)
        
        # Simple heritability estimate
        h2_est = min(0.8, max(0.1, K_mean / (K_mean + 1)))  # Reasonable bounds
        
        lambda_est = (1 - h2_est) / h2_est
        
        print(f"Heritability-based lambda: h2={h2_est:.3f}, λ={lambda_est:.6f}")
        return lambda_est
    
    def _estimate_lambda_reml_approximation(self, K, y):
        """Simple REML approximation"""
        n = len(y)
        
        # Eigen decomposition of K
        eigvals, eigvecs = np.linalg.eigh(K)
        eigvals = np.maximum(eigvals, 1e-10)  # Ensure positive
        
        # Transform y to eigen space
        y_tilde = eigvecs.T @ y
        
        # Simple REML-like estimation
        lambda_candidates = np.logspace(-3, 3, 50)
        best_lambda = 1.0
        best_loglik = -np.inf
        
        for lam in lambda_candidates:
            # Log-likelihood approximation
            logdet = np.sum(np.log(eigvals + lam))
            ssq = np.sum(y_tilde**2 / (eigvals + lam))
            loglik = -0.5 * (logdet + ssq)
            
            if loglik > best_loglik:
                best_loglik = loglik
                best_lambda = lam
        
        print(f"REML-like lambda estimation: λ={best_lambda:.6f}")
        return best_lambda
    
    def _compute_kinship_improved(self, X):
        """Improved kinship computation with allele frequency adjustment"""
        X_centered = X - np.mean(X, axis=0)
        p = X_centered.shape[1]
        
        # Estimate allele frequencies for better standardization
        freqs = np.mean(X, axis=0) / 2  # Assuming 0,1,2 coding
        # Avoid division by zero
        freqs = np.clip(freqs, 0.01, 0.99)
        
        # VanRaden method 1 with actual allele frequencies
        W = (X_centered) / np.sqrt(2 * np.sum(freqs * (1 - freqs)))
        K = W @ W.T
        
        return K, W, X_centered
    
    def _estimate_allele_freqs(self, X):
        """Estimate allele frequencies safely"""
        freqs = np.mean(X, axis=0) / 2  # Assuming 0,1,2 coding
        return np.clip(freqs, 0.01, 0.99)

# ---------------------------
# Enhanced DeepResBLUP_Hybrid
# (drop-in replacement)
# ---------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

# Attempt to import PyG for sample-graph GNN (optional)
try:
    from torch_geometric.nn import GCNConv, GATConv, SAGEConv, global_mean_pool
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False

class DeepResBLUP_Hybrid:
    """
    Hybrid residual model: use pipeline's R_RRBLUP for additive baseline,
    then train a DL residual model (Transformer over markers + sample-graph GNN).
    API kept consistent with other AllGSModels models.
    """

    def __init__(self, config, hyperparams):
        self.config = config
        self.hyperparams = hyperparams
        self.base_model = None         # the R_RRBLUP model instance created by AllGSModels
        self.dl_model = None           # PyTorch DL residual model
        self.base_model_name = hyperparams.get('base_model', 'R_RRBLUP')  # enforce R_RRBLUP default

        # --- Fallback mechanism ---
        if self.base_model_name.upper() == 'R_RRBLUP' and not R_AVAILABLE:
            print("[DeepResBLUP_Hybrid] R environment unavailable. Switching base model to Python RRBLUP.")
            self.base_model_name = 'RRBLUP'

        self.dl_model_name = hyperparams.get('dl_model', 'HybridAttnMLP')
        self.is_fitted = False
        self.X_mean = None
        self.y_mean = None
        # small safety defaults for DL
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------------
    # Public API: fit / predict
    # ------------------------
    def fit(self, X, y, sample_graph=None):
        """
        X: numpy array (n_samples, n_markers)
        y: numpy array (n_samples, ) or (n_samples, n_traits)
        sample_graph: optional torch_geometric.data.Data for sample-graph (nodes = samples)
        """
        print(f"[DeepResBLUP_Hybrid] Fit: base={self.base_model_name}, dl={self.dl_model_name}")
        self.X_mean = np.mean(X, axis=0)
        self.y_mean = np.mean(y, axis=0) if y.ndim > 1 else np.mean(y)

        # ---------- 1) Build and fit base RRBLUP using AllGSModels ----------
        all_models = AllGSModels(self.config)
        
        # If R is unavailable but user still requested R_RRBLUP → override here
        if self.base_model_name.upper() == 'R_RRBLUP' and not R_AVAILABLE:
            print("[DeepResBLUP_Hybrid] Warning: R_RRBLUP requested but R is not available. Using Python RRBLUP instead.")
            self.base_model_name = 'RRBLUP'

        base = all_models.create_model(self.base_model_name, X)

        if base is None:
            raise ValueError(f"Failed to create base model: {self.base_model_name}")

        # Fit base model (multi-trait support retained)
        print("[DeepResBLUP_Hybrid] Training base (RRBLUP) model...")
        if y.ndim == 2 and y.shape[1] > 1:
            # delegate multi-trait to base model(s)
            self.base_model_traits = []
            for i in range(y.shape[1]):
                m = all_models.create_model(self.base_model_name, X)
                m.fit(X, y[:, i])
                self.base_model_traits.append(m)
            self.base_model = None
        else:
            if hasattr(base, 'fit'):
                base.fit(X, y)
            self.base_model = base

        # ---------- 2) Try to extract marker effects from base (preferred) ----------
        rr_weights = None
        print("[DeepResBLUP_Hybrid] Attempting to extract RRBLUP marker effects from base model...")
        if self.base_model is not None:
            # Try common method/property names to get marker effects
            if hasattr(self.base_model, 'get_marker_effects'):
                try:
                    rr_weights = self.base_model.get_marker_effects()
                    rr_weights = np.asarray(rr_weights).reshape(-1)
                    print("[DeepResBLUP_Hybrid] Extracted marker effects via get_marker_effects()")
                except Exception:
                    rr_weights = None
            elif hasattr(self.base_model, 'marker_effects'):
                rr_weights = np.asarray(getattr(self.base_model, 'marker_effects')).reshape(-1)
                print("[DeepResBLUP_Hybrid] Extracted marker effects via marker_effects attribute")
            elif hasattr(self.base_model, 'coef_'):
                rr_weights = np.asarray(getattr(self.base_model, 'coef_')).reshape(-1)
                print("[DeepResBLUP_Hybrid] Extracted marker effects via coef_ attribute")
            # else leave rr_weights = None (we will still get base predictions via predict)
        else:
            # multi-trait branch - not extracting global rr_weights here
            rr_weights = None

        # ---------- 3) Compute base predictions and residuals ----------
        print("[DeepResBLUP_Hybrid] Computing base predictions...")
        base_preds = self._get_base_predictions(X)
        residuals = y - base_preds  # shape (n_samples, n_traits) or (n_samples,)

        # Ensure residuals shape for DL training: (N, output_dim)
        if residuals.ndim == 1:
            residuals = residuals.reshape(-1, 1)

        # ---------- 4) Create DL residual model (marker-transformer + sample-graph GNN) ----------
        n_markers = X.shape[1]
        out_dim = residuals.shape[1]
        print("[DeepResBLUP_Hybrid] Creating DL residual model...")
        self.dl_model = self._create_dl_model(input_dim=n_markers, output_dim=out_dim, rrblup_weights=rr_weights)
        self.dl_model.to(self.device)

        # ---------- 5) Train DL model on residuals ----------
        print("[DeepResBLUP_Hybrid] Training DL residual model...")
        print("[DeepResBLUP_Hybrid] Preparing data loaders and training DL residual...")
        train_loader, val_loader = self._create_data_loaders(X, residuals, batch_size=int(self.hyperparams.get('dl_batch_size', 32)))
        gs_models = AllGSModels(self.config)
        trained = gs_models._train_pytorch_model(self.dl_model, train_loader, val_loader,
                                                learning_rate=float(self.hyperparams.get('dl_learning_rate', 1e-3)),
                                                epochs=int(self.hyperparams.get('dl_epochs', 100)),
                                                model_name='DeepResBLUP_Hybrid')
        self.dl_model = trained

        self.is_fitted = True
        print("[DeepResBLUP_Hybrid] Fit completed.")
        return self

    def predict(self, X, sample_graph=None):
        """Return base + dl_residual predictions; shapes consistent with pipeline"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")

        base_preds = self._get_base_predictions(X)
        # get DL residuals
        self.dl_model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            dl_out = self.dl_model.predict_on_numpy(X_tensor) if hasattr(self.dl_model, 'predict_on_numpy') else self.dl_model(X_tensor)
            if isinstance(dl_out, torch.Tensor):
                dl_out = dl_out.cpu().numpy()
        if dl_out.ndim == 1:
            dl_out = dl_out.reshape(-1, 1)

        combined = base_preds + dl_out
        return combined

    def get_components_info(self):
        return {
            'base_model': self.base_model_name,
            'dl_model': self.dl_model_name,
            'is_fitted': self.is_fitted
        }

    # ----------------------
    # Internal helpers below
    # ----------------------
    def _get_base_predictions(self, X):
        """Return base predictions. Handles single-trait and multi-trait base models."""
        if hasattr(self, 'base_model_traits'):
            preds = []
            for m in self.base_model_traits:
                p = m.predict(X)
                if p.ndim == 1:
                    p = p.reshape(-1, 1)
                preds.append(p)
            return np.hstack(preds)
        else:
            if self.base_model is not None and hasattr(self.base_model, 'predict'):
                p = self.base_model.predict(X)
                if p.ndim == 1:
                    p = p.reshape(-1, 1)
                return p
            else:
                # fallback zero
                return np.zeros((X.shape[0], 1))

    def _create_dl_model(self, input_dim, output_dim, rrblup_weights=None):
        """
        Build the DL residual model. This returns a PyTorch nn.Module with:
          - .forward(X_tensor) -> residuals (B, output_dim)
          - .predict_on_numpy(X_tensor) optionally used in predict()
        The DL model implemented here is a compact hybrid: marker-Transformer (marker-as-token)
        + optional sample-GNN embedding (if provided at training time). It does NOT compute RRBLUP.
        """
        dl_type = self.dl_model_name.lower()

        # If user selected simple HybridAttnMLP/CNN/DNNGS, keep those choices (assumes HybridAttnMLPModel etc. are available)
        if dl_type == 'HybridAttnMLP':
            print (f"{dl_type} is used for residual learning.")

            params = self.hyperparams  # shorthand
            model = HybridAttnMLPModel(
                input_dim=input_dim,
                output_dim=output_dim,

                # list-like hyperparameters
                mlp_hidden=parse_list(params.get('mlp_hidden', [256, 128])),
                cnn_channels=parse_list(params.get('cnn_channels', [32, 64])),

                # convolution settings
                kernel_size=int(params.get('kernel_size', 3)),
                pool_size=int(params.get('pool_size', 2)),

                # attention settings
                attention_heads=int(params.get('attention_heads', 4)),
                hidden_size=int(params.get('hidden_size', 128)),

                # regularization
                dropout=float(params.get('dropout', 0.1)),
                input_dropout=float(params.get('input_dropout', 0.0)),
            )
            
            return PTWrapper(model)  # wrap to provide predict_on_numpy

        if  dl_type == 'hybrid':
            # Build lightweight marker-sequence transformer + (optional) sample-graph readout
            print (f"{dl_type} is used for residual learning.")

            model = _ResidualHybridNet(
                n_markers=input_dim,
                output_dim=output_dim,
                d_model=int(self.hyperparams.get('dl_d_model', 128)),
                nhead=int(self.hyperparams.get('dl_nhead', 4)),
                num_layers=int(self.hyperparams.get('dl_num_layers', 2)),
                dim_feedforward=int(self.hyperparams.get('dl_dim_feedforward', 512)),
                dropout=float(self.hyperparams.get('dl_dropout', 0.1)),
                marker_chunk_size=int(self.hyperparams.get('marker_chunk_size', 1)) if self.hyperparams.get('marker_chunk_size') else None,
                marker_dropout=float(self.hyperparams.get('marker_dropout', 0.05)),
                gnn_type=self.hyperparams.get('gnn_type', 'sage') if self.hyperparams.get('gnn_type') else None,
                gnn_hidden=int(self.hyperparams.get('gnn_hidden', 128)),
                gnn_layers=int(self.hyperparams.get('gnn_layers', 2)),
                device=self.device
            )
            return model

        # Other types (CNN, DNNGS, AttnCNNGS) delegate to your existing model factories if available
        if dl_type == 'cnn':
            print (f"{dl_type} is used for residual learning.")
            return PTWrapper(CNNModel(input_dim=input_dim, output_dim=output_dim,
                                      hidden_channels=list(map(int, self.hyperparams.get('dl_hidden_layers', '64,32').split(','))),
                                      kernel_size=int(self.hyperparams.get('dl_kernel_size', 3)),
                                      pool_size=2,
                                      dropout=float(self.hyperparams.get('dl_dropout', 0.2))))
        if dl_type == 'dnngs':
            print (f"{dl_type} is used for residual learning.")
            return PTWrapper(DNNGSModel(input_dim=input_dim, output_dim=output_dim,
                                      hidden_layers=list(map(int, self.hyperparams.get('dl_hidden_layers', '512,256,128,64').split(','))),
                                      dropout=float(self.hyperparams.get('dl_dropout', 0.3)),
                                      activation='relu', batch_norm=True, input_dropout=0.1))


        if dl_type == 'attncnngs':
            print (f"{dl_type} is used for residual learning.")
            cnn_channels = list(map(int, self.hyperparams.get('dl_hidden_layers', '128,128,256').split(',')))
            dropout = float(self.hyperparams.get('dl_dropout', '0.5'))
            return AttnCNNGSModel(
                input_dim=input_dim,
                output_dim=output_dim,
                cnn_channels=cnn_channels,
                kernel_size=5,
                pool_size=2,
                attention_heads=8,
                hidden_size=256,
                dropout=dropout,
                attention_dropout=0.3
            )

        # fallback: HybridAttnMLP
        print (f"MLPGSModel is used for residual learning.")
        hidden_layers = list(map(int, self.hyperparams.get('dl_hidden_layers', '128,64').split(',')))
        return PTWrapper(MLPGSModel(input_dim=input_dim, output_dim=output_dim,
                                  hidden_layers=hidden_layers, dropout=float(self.hyperparams.get('dl_dropout', 0.2)),
                                  norm='layer', activation='gelu', residual=True, input_dropout=0.05))

    def _create_data_loaders(self, X, y, batch_size=32, val_ratio=0.1):
        """Creates PyTorch DataLoaders consistent with your pipeline's GenomicDataset"""
        if val_ratio > 0:
            X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=val_ratio, random_state=self.config.get('seed', 0))
            train_ds = GenomicDataset(X_train, y_train)
            val_ds = GenomicDataset(X_val, y_val)
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True), DataLoader(val_ds, batch_size=batch_size, shuffle=False)
        else:
            train_ds = GenomicDataset(X, y)
            return DataLoader(train_ds, batch_size=batch_size, shuffle=True), None

# -----------------------------
# Small helper classes / modules
# (these are internal; drop into same file)
# -----------------------------
class PTWrapper(nn.Module):
    """
    Wraps a plain PyTorch module (or custom model) and provides a predict_on_numpy helper used in
    the pipeline predict() method. Also ensures .to(device) behavior consistent with training helper.
    """
    def __init__(self, module):
        super().__init__()
        self.module = module

    def forward(self, x):
        return self.module(x)

    def predict_on_numpy(self, x_tensor):
        # x_tensor: torch.FloatTensor on device
        self.eval()
        with torch.no_grad():
            out = self.module(x_tensor)
        return out

# -----------------------------
# Core Residual Hybrid Net (marker-transformer + optional sample GNN)
# -----------------------------
class _ResidualHybridNet(nn.Module):
    """
    Lightweight transformer-over-markers + optional sample-graph readout.
    Designed to be a residual learner (predict residuals) so no internal RRBLUP logic.
    """
    def __init__(self, n_markers, output_dim=1,
                 d_model=128, nhead=4, num_layers=2, dim_feedforward=512, dropout=0.1,
                 marker_chunk_size=None, marker_dropout=0.05,
                 gnn_type='sage', gnn_hidden=128, gnn_layers=2, device=None):
        super().__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.n_markers = int(n_markers)
        self.marker_dropout = marker_dropout
        self.marker_chunk_size = marker_chunk_size

        # per-marker projection
        self.marker_proj = nn.Linear(1, d_model)
        # learned pos emb (small init)
        self.pos_embed = nn.Parameter(torch.randn(1, max(1, min(self.n_markers, 10000)), d_model) * 0.01)

        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=dim_feedforward, dropout=dropout,
                                                   activation='gelu', batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

        # optional sample-graph GNN
        self.gnn_enabled = (gnn_type is not None)
        if self.gnn_enabled:
            if not PYG_AVAILABLE:
                raise ImportError("torch_geometric required for GNN branch but not installed")
            self.gnn_type = gnn_type.lower()
            self.node_proj = nn.Linear(1, gnn_hidden)
            self.gnn_convs = nn.ModuleList()
            self.gnn_norms = nn.ModuleList()
            for _ in range(gnn_layers):
                if self.gnn_type == 'gcn':
                    conv = GraphConvGSConv(gnn_hidden, gnn_hidden)
                elif self.gnn_type == 'gat':
                    conv = GraphAttnGSConv(gnn_hidden, gnn_hidden)
                else:
                    conv = SAGEConv(gnn_hidden, gnn_hidden)
                self.gnn_convs.append(conv)
                self.gnn_norms.append(nn.LayerNorm(gnn_hidden))
            self.gnn_readout_dim = gnn_hidden
        else:
            self.gnn_readout_dim = 0

        # final projection head
        fused_dim = d_model + self.gnn_readout_dim
        self.head = nn.Sequential(
            nn.Linear(fused_dim, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, output_dim)
        )

        # initialization
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def _apply_marker_dropout(self, X):
        if not self.training or self.marker_dropout <= 0:
            return X
        return X * (torch.rand_like(X) > self.marker_dropout)

    def _prepare_marker_sequence(self, X):
        # X: (B, n_markers)
        B, M = X.shape
        # optional chunking by averaging
        if self.marker_chunk_size and self.marker_chunk_size > 1:
            k = self.marker_chunk_size
            pad = (k - (M % k)) % k
            if pad > 0:
                X = F.pad(X, (0, pad))
            X = X.view(B, -1, k).mean(-1)  # (B, M//k)
        L = X.shape[1]
        emb = self.marker_proj(X.unsqueeze(-1))  # (B, L, d_model)
        # pos embed slicing/repeat
        if self.pos_embed.size(1) >= emb.size(1):
            pos = self.pos_embed[:, :emb.size(1), :]
        else:
            repeat = int(np.ceil(emb.size(1) / self.pos_embed.size(1)))
            pos = self.pos_embed.repeat(1, repeat, 1)[:, :emb.size(1), :]
        return emb + pos

    def forward(self, X_markers, graph_data=None):
        """
        X_markers: torch.FloatTensor (B, n_markers) on device
        graph_data: optional torch_geometric.data.Data for sample graph (with .x and .edge_index)
        """
        # marker dropout
        Xm = self._apply_marker_dropout(X_markers) if self.training else X_markers
        emb = self._prepare_marker_sequence(Xm)      # (B, L, d_model)
        trans = self.transformer(emb)                # (B, L, d_model)
        pooled = self.pool(trans.transpose(1, 2)).squeeze(-1)  # (B, d_model)

        parts = [pooled]

        # GNN branch: graph_data.x expected to be scalar or small vector per sample node
        if self.gnn_enabled and graph_data is not None:
            x = graph_data.x
            if x.dim() == 1 or x.size(1) == 1:
                x = self.node_proj(x.view(-1, 1).float())
            for conv, ln in zip(self.gnn_convs, self.gnn_norms):
                x = conv(x, graph_data.edge_index)
                x = ln(x)
                x = F.gelu(x)
                x = F.dropout(x, p=0.2, training=self.training)
            if hasattr(graph_data, 'batch'):
                g = global_mean_pool(x, graph_data.batch)
            else:
                g = x
            parts.append(g)

        fused = torch.cat(parts, dim=-1)
        out = self.head(fused)
        return out



import rpy2.robjects as ro
from rpy2.robjects import numpy2ri, pandas2ri
from rpy2.robjects.packages import importr
import numpy as np
import warnings

class RModelBase:
    """Base class for R-based models with proper dependency handling"""
    
    def __init__(self, r_package_name, fallback_model_class=None):
        self.r_package_name = r_package_name
        self.fallback_model_class = fallback_model_class
        self.is_available = False
        self.fallback_model = None
        self.r_package = None
        
        # Initialize immediately
        self._initialize_r()
    
    def _initialize_r(self):
        """Initialize R environment"""
        if not R_AVAILABLE:
            print(f"R not available - {self.r_package_name} will use Python fallback")
            self._setup_fallback()
            return
        
        try:
            # Use localconverter for all R operations
            with localconverter(default_converter + numpy2ri.converter + pandas2ri.converter):
                
                # Check if R package is installed
                self.r_package = importr(self.r_package_name)
                self.base = importr('base')
                self.stats = importr('stats')
                self.is_available = True
                print(f"✓ R package '{self.r_package_name}' successfully loaded")
        except Exception as e:
            print(f"✗ Error loading R package '{self.r_package_name}': {e}")
            self._setup_fallback()
    
    def _setup_fallback(self):
        """Setup Python fallback model"""
        if self.fallback_model_class:
            # Initialize the fallback model immediately
            self.fallback_model = self.fallback_model_class()
            print(f"Using Python fallback: {self.fallback_model_class.__name__}")
        else:
            print(f"No fallback available for {self.r_package_name}")
    
    def _ensure_available(self):
        """Ensure model is available, raise error if not"""
        if not self.is_available and not self.fallback_model:
            raise RuntimeError(
                f"R package '{self.r_package_name}' not available and no fallback provided. "
                f"Please install in R: install.packages('{self.r_package_name}')"
            )

class R_RRBLUP(RModelBase):
    """
    Wrapper for R's rrBLUP package with Python interface
    Uses mixed.solve() function from rrBLUP package
    """
    
    def __init__(self, method="ML"):  # Use ML as default
        # Initialize with rrBLUP package and Python fallback
        super().__init__('rrBLUP', HighPerformanceRRBLUP)
        
        self.method = method
        self.is_fitted = False
        self.B = None
        self.X_mean = None
        self.y_mean = None
        self.marker_effects = None

    def fit(self, X, y):
        """Fit RR-BLUP model using R's rrBLUP package"""
        self._ensure_available()
    
        # Use fallback if R not available
        if not self.is_available:
            print("Using Python fallback for RRBLUP")
            return self.fallback_model.fit(X, y)
    
        try:
            # Convert inputs to appropriate formats
            if y.ndim > 1:
                y = y.ravel()
        
            n, p = X.shape
            print(f"R-RRBLUP: Fitting with {n} samples, {p} markers using {self.method}")
        
            # Store means for centering
            self.X_mean = np.mean(X, axis=0)
            self.y_mean = np.mean(y)
        
            # Center the data (important for R rrBLUP)
            X_centered = X - self.X_mean
            y_centered = y - self.y_mean
        
            # Use modern conversion context
            with localconverter(default_converter + numpy2ri.converter + pandas2ri.converter):
                # Convert to R objects - ensure proper dimensions
                r_X = ro.r.matrix(X_centered, nrow=n, ncol=p)
                r_y = ro.FloatVector(y_centered)
            
                # Fit RR-BLUP model using mixed.solve
                print("Calling R: mixed.solve()...")
            
                # Call mixed.solve
                self.r_model = self.r_package.mixed_solve(y=r_y, Z=r_X, method=self.method)
            
                # Debug: print the structure of the returned object
                #print(f"R model type: {type(self.r_model)}")
            
                # Get the names by calling the names() method
                r_names = list(self.r_model.names())
                #print(f"R model names: {r_names}")
            
                # Convert to a dictionary for easier access
                model_dict = dict(zip(r_names, self.r_model))
            
                # Extract marker effects
                if 'u' in model_dict:
                    marker_effects_r = model_dict['u']
                    self.marker_effects = np.array(marker_effects_r).flatten()
                    self.B = self.marker_effects
                else:
                    raise KeyError("'u' not found in mixed.solve results")
            
                # Extract variance components
                if 'Ve' in model_dict:
                    self.ve = float(model_dict['Ve'][0])
                else:
                    self.ve = 1.0  # default
                
                if 'Vu' in model_dict:
                    self.vu = float(model_dict['Vu'][0])
                else:
                    self.vu = 1.0  # default
        
            self.is_fitted = True
            h2 = self.vu / (self.vu + self.ve) if (self.vu + self.ve) > 0 else 0
            print(f"R-RRBLUP fitted: Ve={self.ve:.6f}, Vu={self.vu:.6f}, h²={h2:.3f}")
        
            return self
        
        except Exception as e:
            print(f"Error in R-RRBLUP fitting: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to Python implementation")
            # Make sure fallback model is properly initialized
            if not hasattr(self, 'fallback_model') or self.fallback_model is None:
                self.fallback_model = HighPerformanceRRBLUP()
            return self.fallback_model.fit(X, y)

    def predict(self, X):
        """Predict genetic values using fitted model"""
        if not self.is_fitted:
            raise ValueError("R-RRBLUP model not fitted yet")
    
        try:
            # DEBUG: Check what we're receiving
            #print(f"[DEBUG]R_RRBLUP: Received X shape {X.shape}, expecting test data")
        
            # Center test data using training means
            X_centered = X - self.X_mean
        
            # DEBUG: Verify dimensions
            #print(f"[DEBUG]R_RRBLUP: X_centered shape {X_centered.shape}, B shape {self.B.shape}")
        
            # Ensure we have compatible dimensions for matrix multiplication
            if X_centered.shape[1] != self.B.shape[0]:
                print(f"Warning: Dimension mismatch in R-RRBLUP prediction. Trimming markers.")
                n_markers = min(X_centered.shape[1], self.B.shape[0])
                X_centered = X_centered[:, :n_markers]
                B_trimmed = self.B[:n_markers]
                predictions = X_centered @ B_trimmed + self.y_mean
            else:
                # Normal case - dimensions match
                predictions = X_centered @ self.B + self.y_mean
        
            # Ensure proper output shape
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
        
            #print(f"[DEBUG]R_RRBLUP: Final predictions shape {predictions.shape}")
            return predictions
        
        except Exception as e:
            print(f"Error in R-RRBLUP prediction: {e}")
            import traceback
            traceback.print_exc()
            # Fallback to simple mean prediction
            return np.full((X.shape[0], 1), self.y_mean)

        """Predict genetic values using fitted model"""
        if not self.is_fitted:
            raise ValueError("R-RRBLUP model not fitted yet")
    
        try:
            # Center test data using training means
            X_centered = X - self.X_mean
        
            # Debug: check dimensions
            print(f"R-RRBLUP prediction: X_centered shape {X_centered.shape}, B shape {self.B.shape}")
        
            # Ensure we have compatible dimensions for matrix multiplication
            if X_centered.shape[1] != self.B.shape[0]:
                # This should not happen if data preprocessing is consistent
                print(f"Warning: Dimension mismatch in R-RRBLUP prediction. X_centered: {X_centered.shape}, B: {self.B.shape}")
                # Try to handle by using the first n markers where n = min(X_centered.shape[1], self.B.shape[0])
                n_markers = min(X_centered.shape[1], self.B.shape[0])
                X_centered = X_centered[:, :n_markers]
                B_trimmed = self.B[:n_markers]
                predictions = X_centered @ B_trimmed + self.y_mean
            else:
                # Normal case - dimensions match
                predictions = X_centered @ self.B + self.y_mean
        
            # Ensure proper output shape
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
        
            print(f"R-RRBLUP prediction successful: {predictions.shape}")
            return predictions
        
        except Exception as e:
            print(f"Error in R-RRBLUP prediction: {e}")
            # Fallback to simple mean prediction
            return np.full((X.shape[0], 1), self.y_mean)
    
class R_GBLUP(RModelBase):
    """
    GBLUP implementation using mixed.solve (equivalent to RRBLUP but with kinship)
    """
    
    def __init__(self):
        # Initialize with rrBLUP package
        super().__init__('rrBLUP')
        
        self.is_fitted = False
        self.B = None
        self.X_mean = None
        self.y_mean = None
        self.breeding_values = None

    def fit(self, X, y):
        """Fit GBLUP model using mixed.solve with kinship matrix"""
        self._ensure_available()
    
        # Store training data for proper GBLUP prediction
        self.X_train_original = X.copy()
 
        try:
            if y.ndim > 1:
                y = y.ravel()
        
            n, p = X.shape
            print(f"R-GBLUP: Fitting with {n} samples, {p} markers")
        
            # Store means for prediction
            self.X_mean = np.mean(X, axis=0)
            self.y_mean = np.mean(y)
        
            # Center phenotypes
            y_centered = y - self.y_mean
        
            # Use modern conversion context
            with localconverter(default_converter + numpy2ri.converter + pandas2ri.converter):
                # Convert data to R
                r_X = ro.r.matrix(X, nrow=n, ncol=p)
                r_y = ro.FloatVector(y_centered)
            
                # Compute kinship matrix using R's A.mat
                print("Computing kinship matrix with R's A.mat()...")
                r_K = self.r_package.A_mat(r_X)
            
                print("Calling R: mixed.solve() with kinship...")
                # Use mixed.solve with kinship matrix (this is GBLUP)
                self.r_model = self.r_package.mixed_solve(y=r_y, K=r_K, method="ML")
            
                # Get names and convert to dictionary
                r_names = list(self.r_model.names())
                model_dict = dict(zip(r_names, self.r_model))
            
                # Debug info
                #print(f"R GBLUP model keys: {list(model_dict.keys())}")
            
                # Store breeding values for reference
                if 'u' in model_dict:
                    gebv_r = model_dict['u']
                    self.breeding_values = np.array(gebv_r).flatten()
                    print(f"Breeding values shape: {self.breeding_values.shape}")
                else:
                    raise KeyError("'u' not found in GBLUP results")
            
                # Extract variance components
                if 'Ve' in model_dict:
                    self.ve = float(model_dict['Ve'][0])
                else:
                    self.ve = 1.0
                
                if 'Vu' in model_dict:
                    self.vu = float(model_dict['Vu'][0])
                else:
                    self.vu = 1.0
            
                # Calculate marker effects from breeding values for prediction
                # u = Z * beta, so beta = Z' * (Z*Z')^-1 * u
                # But simpler: use the relationship u = K * alpha, where alpha are the GBLUP effects
                # For prediction on new samples, we need the marker effects
                X_centered = X - self.X_mean
                p = X_centered.shape[1]
                Z = X_centered / np.sqrt(p * 0.5)  # Standardize like VanRaden
            
                # Calculate marker effects: beta = Z' * (Z*Z')^-1 * u
                # Since u = Z * beta, we can solve for beta using pseudoinverse
                try:
                    # Method 1: Direct solution using pseudoinverse
                    Z_pinv = np.linalg.pinv(Z)
                    self.B = Z_pinv @ self.breeding_values
                    print(f"Marker effects calculated via pseudoinverse, shape: {self.B.shape}")
                except:
                    # Method 2: Alternative approach using normal equations
                    try:
                        self.B = np.linalg.solve(Z.T @ Z + 1e-6 * np.eye(Z.shape[1]), Z.T @ self.breeding_values)
                        print(f"Marker effects calculated via normal equations, shape: {self.B.shape}")
                    except:
                        # Method 3: Simple approximation
                        self.B = (Z.T @ self.breeding_values) / np.trace(Z.T @ Z)
                        print(f"Marker effects calculated via approximation, shape: {self.B.shape}")
                
                h2 = self.vu / (self.vu + self.ve) if (self.vu + self.ve) > 0 else 0
                print(f"R-GBLUP fitted: Ve={self.ve:.6f}, Vu={self.vu:.6f}, h²={h2:.3f}")
        
            self.is_fitted = True
            return self
        
        except Exception as e:
            print(f"Error in R-GBLUP fitting: {e}")
            import traceback
            traceback.print_exc()
            self.is_fitted = False
            return self

    def predict(self, X_test):
        """Proper GBLUP prediction using kinship between train and test"""
        if not self.is_fitted:
            raise ValueError("R-GBLUP model not fitted yet")
    
        try:
            # DEBUG: Check input
            #print(f"[DEBUG]R_GBLUP: Received X_test shape {X_test.shape}, expecting test data")
        
            # We need the original training data for proper GBLUP prediction
            if not hasattr(self, 'X_train_original'):
                print("Warning: Using simplified GBLUP prediction (no training data stored)")
                return self._simplified_predict(X_test)
        
            X_train = self.X_train_original
            X_test_centered = X_test - self.X_mean
            X_train_centered = X_train - self.X_mean
        
            # DEBUG: Check dimensions
            #print(f"[DEBUG]R_GBLUP: X_train shape {X_train.shape}, X_test shape {X_test.shape}")
        
            # Compute genomic relationship matrices
            p = X_train_centered.shape[1]
            Z_train = X_train_centered / np.sqrt(p * 0.5)
            Z_test = X_test_centered / np.sqrt(p * 0.5)
        
            # G_train_train = Z_train @ Z_train.T / p (already used in training)
            G_test_train = Z_test @ Z_train.T / float(p)  # Relationship test to train
        
            # GBLUP prediction: y_pred = G_test_train * G_train_train^-1 * u
            predictions = G_test_train @ self.breeding_values + self.y_mean
        
            # Ensure proper output shape
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
        
            #print(f"[DEBUG]R_GBLUP: Final predictions shape {predictions.shape}")
            return predictions
        
        except Exception as e:
            print(f"Error in GBLUP kinship prediction: {e}")
            import traceback
            traceback.print_exc()
            return self._simplified_predict(X_test)

    def _simplified_predict(self, X_test):
        """Fallback prediction using marker effects"""
        try:
            #print(f"[DEBUG]R_GBLUP simplified: X_test shape {X_test.shape}")
        
            X_test_centered = X_test - self.X_mean
        
            # Ensure we have compatible dimensions
            if X_test_centered.shape[1] != self.B.shape[0]:
                print(f"Warning: Dimension mismatch. Trimming markers.")
                n_markers = min(X_test_centered.shape[1], self.B.shape[0])
                X_test_centered = X_test_centered[:, :n_markers]
                B_trimmed = self.B[:n_markers]
                predictions = X_test_centered @ B_trimmed + self.y_mean
            else:
                predictions = X_test_centered @ self.B + self.y_mean
        
            if predictions.ndim == 1:
                predictions = predictions.reshape(-1, 1)
        
            #print(f"[DEBUG]R_GBLUP simplified: Final predictions shape {predictions.shape}")
            return predictions
        except Exception as e:
            print(f"Error in simplified prediction: {e}")
            return np.full((X_test.shape[0], 1), self.y_mean)
    
def check_r_dependencies():
    """Check and provide instructions for R dependencies"""
    if not R_AVAILABLE:
        print("\n" + "="*60)
        print("R INTEGRATION SETUP REQUIRED")
        print("="*60)
        print("To use R-based models (R_RRBLUP, R_GBLUP), please install:")
        print("\n1. Install rpy2 in Python:")
        print("   pip install rpy2")
        print("\n2. Install required R packages:")
        print("   Run R and execute:")
        print("   install.packages('rrBLUP')")
        print("   install.packages('BGLR')  # For Bayesian methods")
        print("\n3. Verify installation in Python:")
        print("   python -c \"import rpy2.robjects; print('rpy2 OK')\"")
        print("="*60 + "\n")
        return False
    
    # Check if rrBLUP package is installed in R
    try:
        importr('rrBLUP')
        print("✓ R rrBLUP package is available")
        return True
    except:
        print("\n" + "="*60)
        print("R PACKAGE SETUP REQUIRED")
        print("="*60)
        print("rpy2 is installed but R packages are missing.")
        print("Please install required R packages:")
        print("\n1. Start R:")
        print("   R")
        print("\n2. Install packages:")
        print("   install.packages('rrBLUP')")
        print("   install.packages('BGLR')")
        print("   quit()")
        print("="*60 + "\n")
        return False

def set_random_seeds(seed):
    """Set all random seeds for reproducibility"""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set environment variables for additional determinism
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    # For scikit-learn models that use randomness
    try:
        import sklearn
        # sklearn doesn't have a global seed function, but we set it per model
    except:
        pass

# -----------------------------
# PyG GNN-- GraphConvGS: sample-as-node implementation with KNN
# Requires: torch_geometric
# -----------------------------
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    import torch_geometric
    from torch_geometric.data import Data as PyGData
    from torch_geometric.loader import DataLoader as PyGDataLoader
    from torch_geometric.nn import GCNConv, global_mean_pool
    PYG_AVAILABLE = True
except Exception:
    PYG_AVAILABLE = False

import math
from torch import nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch_geometric.nn import SAGEConv, global_mean_pool
from torch_geometric.data import Data as PyGData
from torch_geometric.loader import DataLoader as PyGDataLoader


def build_sample_graph_knn(X, top_k=20, metric='euclidean', self_loop=True):
    """
    Build sparse sample-sample edges using KNN.
    Returns (edge_index, edge_weight) as torch tensors for PyG.
    
    Args:
        X: Input data matrix (n_samples, n_features)
        top_k: Number of nearest neighbors to connect
        metric: Distance metric ('euclidean', 'cosine', 'manhattan')
        self_loop: Whether to add self-loops
    """
    X = np.asarray(X, dtype=np.float32)
    n_samples, n_features = X.shape

    # Handle NaNs: replace with column mean
    if np.isnan(X).any():
        col_mean = np.nanmean(X, axis=0)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(col_mean, inds[1])

    # Compute pairwise distances
    if metric == 'euclidean':
        from sklearn.metrics.pairwise import euclidean_distances
        distances = euclidean_distances(X)
    elif metric == 'cosine':
        from sklearn.metrics.pairwise import cosine_distances
        distances = cosine_distances(X)
    elif metric == 'manhattan':
        from sklearn.metrics.pairwise import manhattan_distances
        distances = manhattan_distances(X)
    else:
        raise ValueError(f"Unsupported metric: {metric}. Use 'euclidean', 'cosine', or 'manhattan'")

    # Convert distances to similarities (higher value = more similar)
    # For distances, we want closer points to have higher weights
    similarities = 1.0 / (1.0 + distances)  # Transform to [0, 1] range
    
    # Find top_k neighbors for each sample
    k = min(int(top_k), n_samples - 1)  # Ensure k is valid
    W = np.zeros_like(similarities)
    
    for i in range(n_samples):
        # Get indices of top_k neighbors (excluding self)
        indices = np.argsort(distances[i])[1:k+1]  # Skip the first (self)
        W[i, indices] = similarities[i, indices]
    
    # Symmetrize the adjacency matrix (undirected graph)
    W = np.maximum(W, W.T)
    
    # Add self-loops if requested
    if self_loop:
        np.fill_diagonal(W, 1.0)
    
    # Extract edges and weights
    src, dst = np.where(W > 0)
    edge_index = np.vstack([src, dst]).astype(np.int64)
    edge_weight = W[src, dst].astype(np.float32)

    edge_index = torch.LongTensor(edge_index)
    edge_weight = torch.FloatTensor(edge_weight)
    
    print(f"[KNN Graph] Built graph with {len(src)} edges, top_k={k}, metric={metric}")
    return edge_index, edge_weight

def build_sample_graph(X, method='knn', top_k=20, threshold=None, self_loop=True, metric='euclidean'):
    """
    Unified graph building function that supports both correlation and KNN methods.
    """
    if method == 'knn':
        return build_sample_graph_knn(X, top_k=top_k, metric=metric, self_loop=self_loop)
    elif method == 'corr':
        # Original correlation-based method (keep for backward compatibility)
        X = np.asarray(X, dtype=np.float32)
        n_samples, n_markers = X.shape

        # handle NaNs: replace with column mean
        if np.isnan(X).any():
            col_mean = np.nanmean(X, axis=0)
            inds = np.where(np.isnan(X))
            X[inds] = np.take(col_mean, inds[1])

        # compute sample-wise centered correlation (genomic relationship)
        Xc = X - X.mean(axis=0, keepdims=True)
        # Normalize by number of markers for GRM-like matrix
        G = (Xc @ Xc.T) / float(n_markers)
        W = np.abs(G)  # Use absolute correlation as edge weights

        # threshold if given
        if threshold is not None and threshold != '':
            thr = float(threshold)
            W = np.where(W >= thr, W, 0.0)

        # top_k sparsification - keep only top_k strongest connections per sample
        k = int(top_k) if top_k is not None else 0
        if k > 0 and k < n_samples:
            W_sparse = np.zeros_like(W)
            for i in range(n_samples):
                idx = np.argsort(W[i])[-k:]
                W_sparse[i, idx] = W[i, idx]
            W = np.maximum(W_sparse, W_sparse.T)  # symmetrize

        # ensure diagonal/self-loop
        if self_loop:
            for i in range(n_samples):
                if W[i, i] == 0:
                    W[i, i] = 1.0

        src, dst = np.where(W > 0)
        edge_index = np.vstack([src, dst]).astype(np.int64)
        edge_weight = W[src, dst].astype(np.float32)

        edge_index = torch.LongTensor(edge_index)
        edge_weight = torch.FloatTensor(edge_weight)
        return edge_index, edge_weight
    else:
        raise ValueError(f"Unknown graph building method: {method}")

# Single graph dataset where all samples are nodes in one graph
class SingleGraphDataset(Dataset):
    def __init__(self, X, y, edge_index, edge_weight, train_mask=None, val_mask=None):
        self.data = PyGData(
            x=torch.tensor(X, dtype=torch.float32),
            y=torch.tensor(y, dtype=torch.float32),
            edge_index=edge_index,
            edge_weight=edge_weight
        )
        if train_mask is not None:
            self.data.train_mask = torch.tensor(train_mask, dtype=torch.bool)
        if val_mask is not None:
            self.data.val_mask = torch.tensor(val_mask, dtype=torch.bool)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        return self.data

# Simple PyG model for single graph with multiple samples as nodes
class GenomicSampleGraphConvGSModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, hidden_mlp=128, dropout=0.2, output_dim=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.output_dim = output_dim

        # First GCN layer
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns.append(nn.LayerNorm(hidden_channels))
        
        # Additional GCN layers
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(nn.LayerNorm(hidden_channels))

        # MLP for final prediction - output for each node (sample)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GraphConvGS layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final prediction for each node (sample)
        out = self.mlp(x)
        return out

# Updated training function with better error handling for all GNN models
def train_single_graph_gnn(model, train_loader, val_loader=None, epochs=200, lr=5e-4, weight_decay=0.0, device=None, patience=20, verbose=True, model_name='GraphConvGS'):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    best_val = float('inf')
    best_state = model.state_dict().copy()
    wait = 0

    # Get masks
    train_mask = model.train_mask.to(device) if hasattr(model, 'train_mask') else None
    val_mask = model.val_mask.to(device) if hasattr(model, 'val_mask') else None

    for epoch in range(1, epochs+1):
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch.to(device)
            opt.zero_grad()
            preds = model(batch)
            
            # Apply train mask if available
            if train_mask is not None and train_mask.shape[0] == preds.shape[0]:
                loss = criterion(preds[train_mask], batch.y[train_mask])
            else:
                # Fallback: use all nodes
                loss = criterion(preds, batch.y)
                
            loss.backward()
            opt.step()
            train_losses.append(loss.item())
        train_loss = float(np.mean(train_losses)) if len(train_losses) else 0.0

        val_loss = None
        if val_loader is not None:
            model.eval()
            vals = []
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    preds = model(batch)
                    if val_mask is not None and val_mask.shape[0] == preds.shape[0]:
                        loss = criterion(preds[val_mask], batch.y[val_mask])
                    else:
                        loss = criterion(preds, batch.y)
                    vals.append(loss.item())
            val_loss = float(np.mean(vals)) if len(vals) else None

        if verbose and epoch % 10 == 0:
            if val_loss is None:
                print(f"    [{model_name}] Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.6f}")
            else:
                print(f"    [{model_name}] Epoch {epoch+1}/{epochs} | train_loss: {train_loss:.6f} | val_loss: {val_loss:.6f}")

        # early stopping
        if val_loader is not None and val_loss is not None:
            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = model.state_dict().copy()
                wait = 0
            else:
                wait += 1
            if wait >= patience:
                if verbose:
                    print(f"[{model_name}] Early stopping at epoch {epoch}. Best val {best_val:.6f}") 
                break
       
        # fallback: save best = lowest train loss
        if val_loader is None:
            if train_loss < best_val:
                best_val = train_loss
                best_state = model.state_dict().copy()

    # Load best model
    model.load_state_dict(best_state)
    return model

def predict_single_graph_gnn(model, loader, device=None):
    device = device or (torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
    model = model.to(device).eval()
    preds = []
    ys = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch)
            preds.append(out.cpu().numpy())
            ys.append(batch.y.cpu().numpy())
    if len(preds) == 0:
        return np.array([]), np.array([])
    return np.concatenate(preds, axis=0), np.concatenate(ys, axis=0)

class GenomicSampleGraphAttnGSModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, heads=4, 
                 hidden_mlp=128, dropout=0.2, output_dim=1):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.output_dim = output_dim
        self.heads = heads

        # First GAT layer
        self.convs.append(GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout))
        self.bns.append(nn.LayerNorm(hidden_channels * heads))
        
        # Additional GAT layers
        for _ in range(num_layers - 1):
            self.convs.append(GATConv(hidden_channels * heads, hidden_channels, heads=heads, dropout=dropout))
            self.bns.append(nn.LayerNorm(hidden_channels * heads))

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels * heads, hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GraphAttnGS layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final prediction for each node (sample)
        out = self.mlp(x)
        return out

class GenomicSampleSAGEModel(nn.Module):
    def __init__(self, in_channels, hidden_channels=128, num_layers=2, 
                 hidden_mlp=128, dropout=0.2, output_dim=1, aggr='mean'):
        super().__init__()
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.dropout = dropout
        self.output_dim = output_dim
        self.aggr = aggr

        # First GraphSAGE layer
        self.convs.append(SAGEConv(in_channels, hidden_channels, aggr=aggr))
        self.bns.append(nn.LayerNorm(hidden_channels))
        
        # Additional GraphSAGE layers
        for _ in range(num_layers - 1):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels, aggr=aggr))
            self.bns.append(nn.LayerNorm(hidden_channels))

        # MLP for final prediction
        self.mlp = nn.Sequential(
            nn.Linear(hidden_channels, hidden_mlp),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mlp, output_dim)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply GraphSAGEGS layers
        for conv, bn in zip(self.convs, self.bns):
            x = conv(x, edge_index)
            x = bn(x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        
        # Final prediction for each node (sample)
        out = self.mlp(x)
        return out

"""
Hybrid GNN + Transformer Architecture (GraphFormer-GS)
Concept: Combine GNN message passing with lightweight Transformer attention applied to node features or global representation.
Why it helps GS:
(1) Captures long-range LD interactions that GNN message passing might miss.
(2) Transformer layers can be extremely shallow (1–2 layers) to avoid overfitting small datasets.
(3) Works well when test lines differ genetically → attention helps generalize.
Implementation idea:
(1) One GNN layer (GraphConvGS/GraphAttnGS/SAGE) →
(2) Global mean pooling →
(3) 1–2 Transformer encoder layers →
(4) MLP head.
"""    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, global_mean_pool 
from sklearn.metrics import r2_score, mean_squared_error
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from scipy.stats import pearsonr

"""
    GraphFormer-GS for sample-graph (samples = nodes).
    Encoder: stacked GraphSAGEGS layers (message passing)
    Communicator: TransformerEncoder applied to node embeddings as a sequence
    Head: per-node MLP producing scalar (or multi-trait) outputs.
    Designed for single-graph training (batch size = 1): Data contains all nodes.
    Pipeline:

    GNN message passing layer (GraphConvGS/GraphAttnGS/SAGE) → captures local graph structure (short-range LD).
    Global pooling (e.g., mean pooling across nodes) → compress node embeddings into a single global feature vector.
    1–2 Transformer encoder layers → allow attention between features (long-range interactions), but kept shallow to avoid overfitting.
    MLP head → final prediction for trait(s).
    This is elegant because:
    GNN captures graph topology / local interactions.
    Transformer captures long-range dependencies that GNN alone might miss.
    MLP head maps the combined representation to trait predictions.
    Shallow Transformer (1–2 layers) avoids overfitting on small datasets.

    Current implications:
    mean agrgation (default) reduces node embeddings to a single vector per graph/sample.
    mlp maps the combined representation to the trait(s).
    You can still pass edge_index from your KNN graph.
    Advantages
    Captures short-range local LD via GNN.
    Captures long-range interactions via Transformer.
    Can generalize better to genetically distinct test lines.
    Modular: you can adjust sage_layers, transformer_layers, and mlp_hidden independently.
    """

class GraphFormerModel(nn.Module):
    def __init__(self, in_feats, out_dim=1,
                 gnn_type='SAGE', gnn_hidden=128, 
                 transformer_layers=2, d_model=128, nhead=4,
                 dim_feedforward=256, dropout=0.1, mlp_hidden=128):
        super().__init__()
        
        # GNN layer
        #self.gnn = SAGEConv(in_feats, gnn_hidden)

        # Alternative GNN types 
        if gnn_type == 'SAGE':
            self.gnn = SAGEConv(in_feats, gnn_hidden)
        elif gnn_type == 'GraphConvGS':
            self.gnn = GraphConvGSConv(in_feats, gnn_hidden)
        elif gnn_type == 'GraphAttnGS':
            self.gnn = GraphAttnGSConv(in_feats, gnn_hidden // 4, heads=4, concat=True)
        else:
            # set as default
            self.gnn = SAGEConv(in_feats, gnn_hidden)
            
        self.gnn_hidden = gnn_hidden
        self.gnn_activation = nn.ReLU()
        self.gnn_dropout = nn.Dropout(dropout)
        
        # Projection to transformer dimension
        self.gnn_to_transformer = nn.Linear(gnn_hidden, d_model)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=transformer_layers)
        
        # MLP head - now outputs per-node predictions
        self.mlp = nn.Sequential(
            nn.Linear(d_model, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, out_dim)
        )
        
        self.output_dim = out_dim
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # (1) Single GNN layer - processes all nodes
        x = self.gnn(x, edge_index)
        x = self.gnn_activation(x)
        x = self.gnn_dropout(x)
        
        # (2) Project to transformer dimension
        x = self.gnn_to_transformer(x)  # [num_nodes, d_model]
        
        # (3) Transformer encoder - treat nodes as sequence
        x = x.unsqueeze(1)  # [num_nodes, 1, d_model]
        x = self.transformer(x)  # [num_nodes, 1, d_model]
        x = x.squeeze(1)  # [num_nodes, d_model]
        
        # (4) MLP head - predictions for EACH node
        out = self.mlp(x)  # [num_nodes, out_dim]
        
        return out

import torch
import torch.nn as nn
import numpy as np

"""
integrated RRBLUP into DL model
 Input Layer: Genotypes (SNPs).
 rrBLUP Layer: A single Dense (fully connected) layer with linear activation and L2 regularization (equivalent to a ridge regression). You can even initialize the weights of this layer using the marker effects estimated from a standalone rrBLUP model (transfer learning).
 DL Tower: The output of the rrBLUP layer is then fed into one or more hidden non-linear layers (e.g., with ReLU activation).
 Output Layer: A final node that produces the prediction.
 Advantage: The entire model is trained end-to-end. The "rrBLUP layer" can efficiently capture the additive baseline, and the subsequent "DL tower" can learn non-linear transformations of this additive signal and any other complex patterns. This is a very powerful and parameter-efficient design.
"""
# -------------------------------
# Robust implementation of integrated RRBLUP -> DL model
# -------------------------------
# --- RRBLUPLayer: robust, shape-safe, optionally frozen, with L2 reg ---
class RRBLUPLayer(nn.Module):
    """
    RRBLUP-like linear layer:
      - linear mapping SNPs -> outputs
      - accepts init_weights as 1-D (input_dim) or 2-D (output_dim, input_dim) or (input_dim, output_dim)
      - supports freezing (trainable=False)
      - exposes get_regularization_loss() and get_marker_effects()
    """
    def __init__(self, input_dim, output_dim=1, lambda_value=1.0, init_weights=None, trainable=True):
        super(RRBLUPLayer, self).__init__()
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.lambda_value = float(lambda_value)
        self.trainable = bool(trainable)

        # PyTorch Linear: nn.Linear(in_features, out_features) -> note argument order
        self.linear = nn.Linear(self.input_dim, self.output_dim, bias=True)

        # Initialize weights if provided (accept multiple shapes)
        if init_weights is not None:
            if isinstance(init_weights, np.ndarray):
                w = torch.from_numpy(init_weights).float()
            elif isinstance(init_weights, torch.Tensor):
                w = init_weights.float()
            else:
                raise ValueError("init_weights must be numpy array or torch tensor")

            # Accept shapes:
            # (output_dim, input_dim) -> direct assign to weight (out, in)
            # (input_dim, output_dim) -> transpose to (out, in)
            # (input_dim,) and output_dim==1 -> unsqueeze to (1, input_dim)
            if w.ndim == 2:
                if tuple(w.shape) == (self.output_dim, self.input_dim):
                    self.linear.weight.data.copy_(w)
                elif tuple(w.shape) == (self.input_dim, self.output_dim):
                    self.linear.weight.data.copy_(w.t())
                else:
                    raise ValueError(f"init_weights shape {w.shape} incompatible with required {(self.output_dim, self.input_dim)}")
            elif w.ndim == 1:
                if w.shape[0] == self.input_dim and self.output_dim == 1:
                    self.linear.weight.data.copy_(w.unsqueeze(0))
                else:
                    raise ValueError("1-D init_weights must have length input_dim and output_dim must equal 1")
            else:
                raise ValueError("init_weights must be 1- or 2-D")

        # Freeze parameters if requested
        if not self.trainable:
            for p in self.linear.parameters():
                p.requires_grad = False

    def forward(self, x):
        # x shape: (batch, input_dim)
        return self.linear(x)

    def get_regularization_loss(self):
        """Return L2 penalty of RRBLUP weights (as scalar tensor)"""
        # multiply lambda by sum of squares
        return self.lambda_value * torch.sum(self.linear.weight ** 2)

    def get_marker_effects(self):
        """Return marker effects (numpy) as 1-D (input_dim,) if output_dim==1, else (output_dim, input_dim)"""
        w = self.linear.weight.detach().cpu().numpy()
        if self.output_dim == 1:
            return w.reshape(-1)
        return w  # (out, in)

# --- DeepBLUP: rrblup linear layer feeding a small DL tower ---
# Enhanced model
class DeepBLUP(nn.Module):
    """
    Enhanced Integrated RRBLUP + DL model with:
    - Batch Normalization
    - Residual Connections  
    - Better initialization
    """
    
    def __init__(self,
                 input_dim,
                 output_dim=1,
                 rrblup_lambda=1.0,
                 dl_hidden_layers=(256, 128, 64),
                 dropout=0.3,
                 activation='gelu',
                 use_precomputed_rrblup=True,
                 rrblup_weights=None,
                 train_rrblup_layer=True,
                 use_skip_connection=True,
                 use_batch_norm=True,
                 use_residual_connections=True):
        super(DeepBLUP, self).__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.rrblup_lambda = float(rrblup_lambda)
        self.use_batch_norm = bool(use_batch_norm)
        self.use_residual_connections = bool(use_residual_connections)
        
        # Activation function
        act_name = activation.lower()
        if act_name == 'gelu':
            self.activation = nn.GELU()
        elif act_name == 'relu':
            self.activation = nn.ReLU()
        elif act_name == 'leaky_relu':
            self.activation = nn.LeakyReLU(0.1)
        else:
            self.activation = nn.GELU()

        # RRBLUP layer
        rrblup_out_dim = dl_hidden_layers[0] if len(dl_hidden_layers) > 0 else output_dim
        self.rrblup_layer = RRBLUPLayer(
            input_dim=self.input_dim,
            output_dim=rrblup_out_dim,
            lambda_value=self.rrblup_lambda,
            init_weights=rrblup_weights,
            trainable=train_rrblup_layer
        )

        # Enhanced DL Tower with BatchNorm and Residual connections
        self.dl_blocks = nn.ModuleList()
        self.bn_layers = nn.ModuleList() if self.use_batch_norm else None
        self.residual_proj = nn.ModuleList() if self.use_residual_connections else None
        
        prev_size = rrblup_out_dim
        for i, hidden_size in enumerate(dl_hidden_layers):
            # Main linear layer
            self.dl_blocks.append(nn.Linear(prev_size, hidden_size))
            
            # Batch normalization
            if self.use_batch_norm:
                self.bn_layers.append(nn.BatchNorm1d(hidden_size))
            
            # Residual connection projection (if dimensions change)
            if self.use_residual_connections and i > 0 and hidden_size != dl_hidden_layers[i-1]:
                self.residual_proj.append(nn.Linear(dl_hidden_layers[i-1], hidden_size))
            else:
                self.residual_proj.append(nn.Identity())
            
            prev_size = hidden_size

        # Output layer
        self.output_norm = nn.BatchNorm1d(prev_size) if self.use_batch_norm else nn.Identity()
        self.output_layer = nn.Linear(prev_size, output_dim)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        # Initialize DL tower weights
        for layer in self.dl_blocks:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, 0.0)
        
        # Initialize output layer
        nn.init.xavier_uniform_(self.output_layer.weight)
        if self.output_layer.bias is not None:
            nn.init.constant_(self.output_layer.bias, 0.0)

    def forward(self, x):
        # RRBLUP layer
        rrblup_out = self.rrblup_layer(x)
        h = rrblup_out
        
        # DL tower with residual connections and batch norm
        for i, (linear_layer) in enumerate(self.dl_blocks):
            residual = h
            
            # Linear transformation
            h = linear_layer(h)
            
            # Batch normalization
            if self.use_batch_norm:
                h = self.bn_layers[i](h)
            
            # Activation
            h = self.activation(h)
            
            # Dropout
            h = self.dropout(h)
            
            # Residual connection (if enabled and dimensions compatible)
            if (self.use_residual_connections and 
                i > 0 and 
                residual.shape == h.shape):
                h = h + residual
            elif (self.use_residual_connections and 
                  i > 0 and 
                  hasattr(self.residual_proj[i], 'weight')):
                # Project residual to match dimensions
                residual_proj = self.residual_proj[i](residual)
                h = h + residual_proj
        
        # Output layer
        h = self.output_norm(h) if self.use_batch_norm else h
        output = self.output_layer(h)
        
        return output

    def get_rrblup_regularization(self):
        return self.rrblup_layer.get_regularization_loss()

    def predict(self, X):
        """Predict method for compatibility"""
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(device)
            predictions = self.forward(X_tensor)
            return predictions.cpu().numpy()

class FusionGSModel(nn.Module):
    """
    FusionGS: efficient hybrid model combining
      - a shallow 1D CNN branch (local haplotype / LD-like patterns)
      - a residual MLP branch (global/polygenic signal)
    on the same marker view.

    Designed to be much lighter than AttnCNNGS while often stronger than
    a plain MLP in practice.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 cnn_channels=None,
                 kernel_size: int = 5,
                 pool_size:   int = 2,
                 mlp_hidden=None,
                 fusion_hidden: int = 128,
                 dropout: float = 0.3,
                 input_dropout: float = 0.05):
        super().__init__()

        if cnn_channels is None:
            cnn_channels = [32, 64]
        if mlp_hidden is None:
            mlp_hidden = [256, 128]

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.kernel_size = int(kernel_size)
        self.pool_size = int(pool_size)

        # ----- CNN branch (sequence over markers) -----
        channels = [1] + list(map(int, cnn_channels))
        conv_layers = []
        for in_c, out_c in zip(channels[:-1], channels[1:]):
            conv_layers.append(
                nn.Sequential(
                    nn.Conv1d(in_c, out_c,
                              kernel_size=self.kernel_size,
                              padding=self.kernel_size // 2),
                    nn.BatchNorm1d(out_c),
                    nn.GELU(),
                    nn.MaxPool1d(self.pool_size),
                    nn.Dropout(dropout),
                )
            )
        self.cnn_layers = nn.ModuleList(conv_layers)

        # we will infer the flattened dimension with a dummy forward
        with torch.no_grad():
            dummy = torch.zeros(1, 1, self.input_dim)
            x = dummy
            for layer in self.cnn_layers:
                x = layer(x)
            self.cnn_flat_dim = x.shape[1] * x.shape[2]

        self.cnn_proj = nn.Sequential(
            nn.Linear(self.cnn_flat_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ----- MLP branch (global signal) -----
        self.input_do = nn.Dropout(float(input_dropout))
        mlp_layers = []
        prev = self.input_dim
        for h in mlp_hidden:
            h = int(h)
            mlp_layers.append(
                nn.Sequential(
                    nn.Linear(prev, h),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
            )
            prev = h
        self.mlp_layers = nn.ModuleList(mlp_layers)
        self.mlp_norm = nn.LayerNorm(prev)

        # project MLP branch to fusion_hidden as well
        self.mlp_proj = nn.Sequential(
            nn.Linear(prev, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # ----- Fusion head -----
        fused_dim = fusion_hidden * 2
        self.fusion_head = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden, output_dim),
        )

    def forward(self, x):
        # x: (batch, n_markers)
        # CNN branch
        x_cnn = x.unsqueeze(1)             # (B, 1, L)
        for layer in self.cnn_layers:
            x_cnn = layer(x_cnn)
        x_cnn = x_cnn.reshape(x_cnn.size(0), -1)
        x_cnn = self.cnn_proj(x_cnn)

        # MLP branch
        h = self.input_do(x)
        for layer in self.mlp_layers:
            h = layer(h)
        h = self.mlp_norm(h)
        h = self.mlp_proj(h)

        # Fuse
        fused = torch.cat([x_cnn, h], dim=1)
        out = self.fusion_head(fused)
        return out

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            return self.forward(x).cpu().numpy()

class EfficientGSFormerModel(nn.Module):
    """
    EfficientGSFormer: patch-based transformer for genomic selection.

    - Splits the marker vector into non-overlapping patches (tokens).
    - Linear projection of each patch → d_model.
    - 1–2 lightweight TransformerEncoder layers over tokens.
    - Mean-pool over tokens → small MLP head.

    This avoids full O(M^2) attention over all markers and is much
    lighter than a naive transformer over 10k+ SNPs.
    """
    def __init__(self,
                 input_dim: int,
                 output_dim: int,
                 d_model: int = 128,
                 nhead: int = 4,
                 num_layers: int = 2,
                 dim_feedforward: int = 256,
                 dropout: float = 0.1,
                 patch_size: int = 64):
        super().__init__()

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.d_model = int(d_model)
        self.nhead = int(nhead)
        self.num_layers = int(num_layers)
        self.dim_feedforward = int(dim_feedforward)
        self.dropout = float(dropout)
        self.patch_size = int(patch_size)

        # --- Compute number of patches (ALWAYS >= 1) ---
        self.n_patches = max(1, self.input_dim // self.patch_size)

        # positional embedding for max possible patches
        max_patches = self.n_patches
        self.pos_embedding = nn.Parameter(torch.zeros(1, max_patches, self.d_model))
        nn.init.normal_(self.pos_embedding, std=0.02)

        # number of patches (tokens); drop remainder if not divisible
        self.n_patches = max(1, self.input_dim // self.patch_size)

        # project each patch (length patch_size) → d_model
        self.patch_proj = nn.Linear(self.patch_size, self.d_model)

        # positional encoding over patches
        self.pos_encoding = PositionalEncoding(self.d_model, self.dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=self.nhead,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            batch_first=True,
            activation='gelu'
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)

        self.head = nn.Sequential(
            nn.LayerNorm(self.d_model),
            nn.Linear(self.d_model, self.d_model),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, output_dim)
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)

    def _reshape_to_patches(self, x):
        # x: (B, D)
        B, D = x.shape
        usable = self.n_patches * self.patch_size

        if usable == 0:
            # fallback for tiny feature sets (e.g., PC)
            self.n_patches = 1
            self.patch_size = x.shape[1]
            usable = self.patch_size

        if D > usable:
            x = x[:, :usable]
        elif D < usable:
            # simple zero-padding at the end if needed
            pad = usable - D
            pad_tensor = x.new_zeros(B, pad)
            x = torch.cat([x, pad_tensor], dim=1)


        x = x.view(B, self.n_patches, self.patch_size)   # (B, T, P)
        return x

    def forward(self, x):
        # x: (batch_size, n_markers)
        x = self._reshape_to_patches(x)          # (B, T, P)
        x = self.patch_proj(x)                   # (B, T, d_model)
        #x = self.pos_encoding(x)                 # add PE on tokens
        x = x + self.pos_embedding[:, :x.size(1), :]
        x = self.encoder(x)                      # (B, T, d_model)

        # global pooling over tokens
        x = x.mean(dim=1)                        # (B, d_model)
        x = self.head(x)                         # (B, output_dim)
        return x

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            if isinstance(x, np.ndarray):
                x = torch.FloatTensor(x)
            return self.forward(x).cpu().numpy()

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Optional, Sequence

class AllGSModels:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.hyperparams = config['hyperparameters']
             
    def create_model(self, model_name, X_train=None):
        """Create a model instance based on name"""
        #print(f"[DEBUG] create_model called for '{model_name}'")
        
        if model_name == 'DeepBLUP':
            # Get hyperparameters
            hp = self.hyperparams.get('DeepBLUP', {})
            
            return None
        
        elif model_name == 'DeepResBLUP':
            # Create the hybrid model
            model_hyperparams = self.hyperparams.get('DeepResBLUP', {})
            return DeepResBLUP_Hybrid(self.config, model_hyperparams)

        elif model_name == 'GraphConvGS' and TORCH_AVAILABLE:
            hidden_channels = int(self.hyperparams.get('GraphConvGS', {}).get('hidden_channels', '128'))
            num_layers = int(self.hyperparams.get('GraphConvGS', {}).get('num_layers', '2'))
            hidden_mlp = int(self.hyperparams.get('GraphConvGS', {}).get('hidden_mlp', '128'))
            dropout = float(self.hyperparams.get('GraphConvGS', {}).get('dropout', '0.2'))
            graph_method = self.hyperparams.get('GraphConvGS', {}).get('graph_method', 'knn')
            knn_metric = self.hyperparams.get('GraphConvGS', {}).get('knn_metric', 'euclidean')
            patience = int(self.hyperparams.get('GraphConvGS', {}).get('patience', '20'))
    
            return {
                'type': 'GraphConvGS',
                'hidden_channels': hidden_channels,
                'num_layers': num_layers,
                'hidden_mlp': hidden_mlp,
                'dropout': dropout,
                'graph_method': graph_method,
                'knn_metric': knn_metric,
                'patience': patience
            }
        
        elif model_name == 'GraphAttnGS' and TORCH_AVAILABLE:
            hidden_channels = int(self.hyperparams.get('GraphAttnGS', {}).get('hidden_channels', '128'))
            num_layers = int(self.hyperparams.get('GraphAttnGS', {}).get('num_layers', '2'))
            heads = int(self.hyperparams.get('GraphAttnGS', {}).get('heads', '4'))
            hidden_mlp = int(self.hyperparams.get('GraphAttnGS', {}).get('hidden_mlp', '128'))
            dropout = float(self.hyperparams.get('GraphAttnGS', {}).get('dropout', '0.2'))
            graph_method = self.hyperparams.get('GraphAttnGS', {}).get('graph_method', 'knn')
            knn_metric = self.hyperparams.get('GraphAttnGS', {}).get('knn_metric', 'euclidean')
            patience = int(self.hyperparams.get('GraphAttnGS', {}).get('patience', '20'))
    
            return {
                'type': 'GraphAttnGS',
                'hidden_channels': hidden_channels,
                'num_layers': num_layers,
                'heads': heads,
                'hidden_mlp': hidden_mlp,
                'dropout': dropout,
                'graph_method': graph_method,
                'knn_metric': knn_metric,
                'patience': patience
            }
        
        elif model_name == 'GraphSAGEGS' and TORCH_AVAILABLE:
            hidden_channels = int(self.hyperparams.get('GraphSAGEGS', {}).get('hidden_channels', '128'))
            num_layers = int(self.hyperparams.get('GraphSAGEGS', {}).get('num_layers', '2'))
            hidden_mlp = int(self.hyperparams.get('GraphSAGEGS', {}).get('hidden_mlp', '128'))
            dropout = float(self.hyperparams.get('GraphSAGEGS', {}).get('dropout', '0.2'))
            graph_method = self.hyperparams.get('GraphSAGEGS', {}).get('graph_method', 'knn')
            knn_metric = self.hyperparams.get('GraphSAGEGS', {}).get('knn_metric', 'euclidean')
            aggr = self.hyperparams.get('GraphSAGEGS', {}).get('aggr', 'mean')
            patience = int(self.hyperparams.get('GraphSAGEGS', {}).get('patience', '20'))

            return {
                'type': 'GraphSAGEGS',
                'hidden_channels': hidden_channels,
                'num_layers': num_layers,
                'hidden_mlp': hidden_mlp,
                'dropout': dropout,
                'graph_method': graph_method,
                'knn_metric': knn_metric,
                'aggr': aggr,
                'patience': patience
            }

        elif model_name == 'GraphFormer' and TORCH_AVAILABLE:
            return {
                'type': 'GraphFormer',
                'gnn_type': self.hyperparams.get('GraphFormer', {}).get('gnn_type', 'SAGE'),
                'gnn_hidden': int(self.hyperparams.get('GraphFormer', {}).get('gnn_hidden', '128')),
                'd_model': int(self.hyperparams.get('GraphFormer', {}).get('d_model', '128')),
                'transformer_layers': int(self.hyperparams.get('GraphFormer', {}).get('transformer_layers', '2')),
                'nhead': int(self.hyperparams.get('GraphFormer', {}).get('nhead', '4')),
                'mlp_hidden': int(self.hyperparams.get('GraphFormer', {}).get('mlp_hidden', '128')),
                'dropout': float(self.hyperparams.get('GraphFormer', {}).get('dropout', '0.1')),
                'graph_method': self.hyperparams.get('GraphFormer', {}).get('graph_method', 'knn'),
                'knn_metric': self.hyperparams.get('GraphFormer', {}).get('knn_metric', 'euclidean'),
                'top_k': int(self.hyperparams.get('GraphFormer', {}).get('top_k', '20')),
                'patience': int(self.hyperparams.get('GraphFormer', {}).get('patience', '20'))
            }
        

        elif model_name == 'ElasticNet':
            alpha = float(self.hyperparams.get('ElasticNet', {}).get('alpha', 1.0))
            l1_ratio = float(self.hyperparams.get('ElasticNet', {}).get('l1_ratio', 0.5))
            return ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=self.config['seed'])
            
        elif model_name == 'LASSO':
            alpha = float(self.hyperparams.get('LASSO', {}).get('alpha', 1.0))
            max_iter = int(self.hyperparams.get('LASSO', {}).get('max_iter', 1000))
            selection = self.hyperparams.get('LASSO', {}).get('selection', 'cyclic')
    
            return Lasso(
                alpha=alpha, 
                random_state=self.config['seed'],
                max_iter=max_iter,
                selection=selection,
                tol=1e-4
            )
            
        elif model_name == 'RFR':
            n_estimators = int(self.hyperparams.get('RFR', {}).get('n_estimators', 100))
            max_depth = self.hyperparams.get('RFR', {}).get('max_depth', None)
            if max_depth and max_depth != 'None':
                max_depth = int(max_depth)
            else:
                max_depth = None
            return RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                                       random_state=self.config['seed'], n_jobs=-1)
            
        elif model_name == 'BRR' and BAYESIAN_AVAILABLE:
            # Return actual BayesianRidge model
            alpha_1 = float(self.hyperparams.get('BRR', {}).get('alpha_1', 1e-6))
            alpha_2 = float(self.hyperparams.get('BRR', {}).get('alpha_2', 1e-6))
            lambda_1 = float(self.hyperparams.get('BRR', {}).get('lambda_1', 1e-6))
            lambda_2 = float(self.hyperparams.get('BRR', {}).get('lambda_2', 1e-6))
            print(f"Using alpha_1={alpha_1}, alpha_2={alpha_2}, lambda_1={lambda_1}, lambda_2={lambda_2}")
            
            return BayesianRidge(
                alpha_1=alpha_1,
                alpha_2=alpha_2,
                lambda_1=lambda_1,
                lambda_2=lambda_2
            )

        elif model_name == 'RRBLUP':
            lambda_str = self.hyperparams.get('RRBLUP', {}).get('lambda_value', 'None')
            # FIX: Properly handle 'None' string
            if lambda_str == 'None' or lambda_str is None:
                lambda_value = None
            else:
                try:
                    lambda_value = float(lambda_str)
                except (ValueError, TypeError):
                    print(f"Warning: Invalid lambda_value '{lambda_str}' for RRBLUP, using None")
                    lambda_value = None
            
            method = self.hyperparams.get('RRBLUP', {}).get('method', 'mixed_model')
            lambda_method = self.hyperparams.get('RRBLUP', {}).get('lambda_method', 'auto')  # NEW
            tol = float(self.hyperparams.get('RRBLUP', {}).get('tol', '1e-8'))
    
            return HighPerformanceRRBLUP(lambda_value=lambda_value, method=method, 
                               lambda_method=lambda_method, tol=tol)  # UPDATED
        elif model_name == 'R_RRBLUP':
            method = self.hyperparams.get('R_RRBLUP', {}).get('method', 'ML')  # Default to ML
            return R_RRBLUP(method=method)
    
        elif model_name == 'R_GBLUP':
            return R_GBLUP()
        
        elif model_name == 'XGBoost':
            try:
                import xgboost as xgb
                n_estimators = int(self.hyperparams.get('XGBoost', {}).get('n_estimators', 100))
                max_depth = int(self.hyperparams.get('XGBoost', {}).get('max_depth', 6))
                learning_rate = float(self.hyperparams.get('XGBoost', {}).get('learning_rate', 0.1))
                verbose = int(self.hyperparams.get('LightGBM', {}).get('verbose', -1))  # Get verbose parameter
                return xgb.XGBRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    random_state=self.config['seed']
                )
            except ImportError:
                print("XGBoost not available. Using RFR as fallback.")
                return RandomForestRegressor(n_estimators=100, random_state=self.config['seed'])
    
        elif model_name == 'LightGBM':
            if LGBM_AVAILABLE:
                n_estimators = int(self.hyperparams.get('LightGBM', {}).get('n_estimators', 100))
                max_depth = int(self.hyperparams.get('LightGBM', {}).get('max_depth', -1))
                learning_rate = float(self.hyperparams.get('LightGBM', {}).get('learning_rate', 0.1))
                return lgb.LGBMRegressor(
                    n_estimators=n_estimators,
                    max_depth=max_depth,
                    learning_rate=learning_rate,
                    force_col_wise=True,  # ADD THIS to remove the overhead warning
                    verbosity=-1,          # verbose=-1 to remove warning message
                    random_state=self.config['seed']
                )
            else:
                print("LightGBM not available. Using RFR as fallback.")
                return RandomForestRegressor(n_estimators=100, random_state=self.config['seed'])

        
        elif model_name == 'EnsembleGS':
            # EnsembleGS ensemble - we'll handle this specially in train_models
            return None
        
        # In the create_model method, add CNN and HybridAttnMLP cases:
        elif model_name == 'CNN' and TORCH_AVAILABLE:
            hidden_channels = list(map(int, self.hyperparams.get('CNN', {}).get('hidden_channels', '64,32').split(',')))
            kernel_size = int(self.hyperparams.get('CNN', {}).get('kernel_size', '3'))
            pool_size = int(self.hyperparams.get('CNN', {}).get('pool_size', '2'))
            dropout = float(self.hyperparams.get('CNN', {}).get('dropout', '0.2'))
    
            # We'll initialize this later when we know input/output dimensions
            return {'type': 'CNN', 'hidden_channels': hidden_channels, 'kernel_size': kernel_size,
                    'pool_size': pool_size, 'dropout': dropout}

        elif model_name == 'AttnCNNGS' and TORCH_AVAILABLE:
            cnn_channels = list(map(int, self.hyperparams.get('AttnCNNGS', {}).get('cnn_channels', '128,128,256').split(',')))
            kernel_size = int(self.hyperparams.get('AttnCNNGS', {}).get('kernel_size', '5'))
            pool_size = int(self.hyperparams.get('AttnCNNGS', {}).get('pool_size', '2'))
            attention_heads = int(self.hyperparams.get('AttnCNNGS', {}).get('attention_heads', '8'))
            hidden_size = int(self.hyperparams.get('AttnCNNGS', {}).get('hidden_size', '256'))
            dropout = float(self.hyperparams.get('AttnCNNGS', {}).get('dropout', '0.5'))
            attention_dropout = float(self.hyperparams.get('AttnCNNGS', {}).get('attention_dropout', '0.3'))

            return {'type': 'AttnCNNGS', 
                'cnn_channels': cnn_channels,
                'kernel_size': kernel_size,
                'pool_size': pool_size,
                'attention_heads': attention_heads,
                'hidden_size': hidden_size,
                'dropout': dropout,
                'attention_dropout': attention_dropout}

        elif model_name == 'MLPGS' and TORCH_AVAILABLE:
            hidden_layers = list(map(int, self.hyperparams.get('MLPGS', {}).get('hidden_layers', '128,64').split(',')))
            dropout = float(self.hyperparams.get('MLPGS', {}).get('dropout', '0.2'))
    
            return {'type': 'MLPGS',
                    'hidden_layers': hidden_layers,
                    'dropout': dropout,
                    'norm': self.hyperparams.get('MLPGS', {}).get('norm', 'layer'),
                    'activation': self.hyperparams.get('MLPGS', {}).get('activation', 'gelu'),
                    'residual': str(self.hyperparams.get('MLPGS', {}).get('residual', 'true')).lower()=='true',
                    'input_dropout': float(self.hyperparams.get('MLPGS', {}).get('input_dropout', '0.05'))}
    
        elif model_name == 'HybridAttnMLP' and TORCH_AVAILABLE:
            hidden_layers = list(map(int, self.hyperparams.get('HybridAttnMLP', {}).get('hidden_layers', '128,64').split(',')))
            dropout = float(self.hyperparams.get('HybridAttnMLP', {}).get('dropout', '0.2'))
    
            return {'type': 'HybridAttnMLP',
                    'hidden_layers': hidden_layers,
                    'dropout': dropout,
                    'norm': self.hyperparams.get('HybridAttnMLP', {}).get('norm', 'layer'),
                    'activation': self.hyperparams.get('HybridAttnMLP', {}).get('activation', 'gelu'),
                    'residual': str(self.hyperparams.get('HybridAttnMLP', {}).get('residual', 'true')).lower()=='true',
                    'input_dropout': float(self.hyperparams.get('HybridAttnMLP', {}).get('input_dropout', '0.05'))}

        elif model_name == 'DNNGS' and TORCH_AVAILABLE:
            hidden_layers = list(map(int, self.hyperparams.get('DNNGS', {}).get('hidden_layers', '512,256,128,64').split(',')))
            dropout = float(self.hyperparams.get('DNNGS', {}).get('dropout', '0.3'))
            activation = self.hyperparams.get('DNNGS', {}).get('activation', 'relu')
            batch_norm = str(self.hyperparams.get('DNNGS', {}).get('batch_norm', 'true')).lower() == 'true'
            input_dropout = float(self.hyperparams.get('DNNGS', {}).get('input_dropout', '0.1'))

            return {'type': 'DNNGS',
                    'hidden_layers': hidden_layers,
                    'dropout': dropout,
                    'activation': activation,
                    'batch_norm': batch_norm,
                    'input_dropout': input_dropout}
        
        elif model_name == 'Transformer' and TORCH_AVAILABLE:
            d_model = int(self.hyperparams.get('Transformer', {}).get('d_model', '512'))
            nhead = int(self.hyperparams.get('Transformer', {}).get('nhead', '8'))
            num_layers = int(self.hyperparams.get('Transformer', {}).get('num_layers', '6'))
            dim_feedforward = int(self.hyperparams.get('Transformer', {}).get('dim_feedforward', '2048'))
            dropout = float(self.hyperparams.get('Transformer', {}).get('dropout', '0.1'))

            return {'type': 'Transformer',
                'd_model': d_model,
                'nhead': nhead, 
                'num_layers': num_layers,
                'dim_feedforward': dim_feedforward,
                'dropout': dropout}

        elif model_name == 'FusionGS':
            # use hyperparams from [Hyperparameters_FusionGS] or defaults
            hp = self.hyperparams.get('FusionGS', {})
            return {
                'cnn_channels': parse_list(hp.get('cnn_channels', '32,64')),
                'kernel_size': int(hp.get('kernel_size', '5')),
                'pool_size': int(hp.get('pool_size', '2')),
                'mlp_hidden': parse_list(hp.get('mlp_hidden', '256,128')),
                'fusion_hidden': int(hp.get('fusion_hidden', '128')),
                'dropout': float(hp.get('dropout', '0.3')),
                'input_dropout': float(hp.get('input_dropout', '0.05')),
            }

        elif model_name == 'EfficientGSFormer':
            hp = self.hyperparams.get('EfficientGSFormer', {})
            return {
                'd_model': int(hp.get('d_model', '128')),
                'nhead': int(hp.get('nhead', '4')),
                'num_layers': int(hp.get('num_layers', '2')),
                'dim_feedforward': int(hp.get('dim_feedforward', '256')),
                'dropout': float(hp.get('dropout', '0.1')),
                'patch_size': int(hp.get('patch_size', '64')),
            }

        else:
            raise ValueError(f"Unknown model: {model_name}")

        print("[DEBUG] create_model reached end WITHOUT matching model:", model_name)

    def train_models(self, X_train, y_train):
        """Train all enabled models"""
        """
        # DEBUG: Track all RRBLUP training entries
        import inspect
        frame = inspect.currentframe()
        caller_info = inspect.getframeinfo(frame)
        print(f"DEBUG RRBLUP ENTRY: Called from {caller_info.filename}:{caller_info.lineno}")

        # DEBUG: Check for duplicate model names
        print(f"DEBUG: enabled_models = {self.config['enabled_models']}")
        from collections import Counter
        model_counts = Counter(self.config['enabled_models'])
        duplicates = [model for model, count in model_counts.items() if count > 1]
        if duplicates:
            print(f"WARNING: Duplicate model names found: {duplicates}")

        # DEBUG: Check input shapes
        print(f"DEBUG train_models: X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        if y_train.ndim == 2:
            print(f"DEBUG: Multi-trait data with {y_train.shape[1]} traits")
        else:
            print(f"DEBUG: Single trait data")
        """

        # Set random seeds at the start of training
        set_random_seeds(self.config['seed'])

        # Create a copy of data to prevent modification
        X_train_copy = X_train.copy()
        y_train_copy = y_train.copy()

        # First train all individual models
        for model_name in self.config['enabled_models']:
            # Free memory manually 
            import gc
            gc.collect()
            
            if model_name == 'EnsembleGS':
                continue

            print(f"[{self.config['feature_view'].upper()}] Training {model_name}...")
            model_start_time = time.time()  # Start timing for this modelmodel_start_time = time.time()  # Start timing for this model
            start_mem = get_memory_usage_mb()

            try:
                # Reset seeds for each model to ensure consistency
                set_random_seeds(self.config['seed'])
            
                # Use fresh data copies for each model
                X_current = X_train_copy
                y_current = y_train_copy

                if model_name == 'DeepBLUP':
                    # Don't use create_model since it returns None for this model
                    # Get hyperparameters directly
                    hp = self.hyperparams.get('DeepBLUP', {})
                    
                    # Initialize model parameters
                    input_dim = X_current.shape[1]
                    output_dim = y_current.shape[1] if y_current.ndim > 1 else 1
                    
                    # Get DL hidden layers
                    dl_hidden_layers = list(map(int, hp.get('dl_hidden_layers', '128,64').split(',')))
                    rrblup_out_dim = dl_hidden_layers[0] if len(dl_hidden_layers) > 0 else output_dim
                    
                    # Precompute RRBLUP weights if requested
                    rrblup_weights = None
                    if hp.get('use_precomputed_rrblup', 'false').lower() == 'true':
                        print("Precomputing RRBLUP weights for initialization...")

                        # If R is unavailable, switch any R-based RRBLUP request to Python RRBLUP
                        if R_AVAILABLE is False:
                            if hp.get('use_precomputed_rrblup', 'true').lower() == 'true':
                                print("[DeepBLUP] R unavailable — using Python RRBLUP for marker effect extraction.")

                        rrblup_weights = self._get_rrblup_weights_for_integrated_model(
                            X_current, y_current, output_dim, rrblup_out_dim
                        )
                        
                        if rrblup_weights is not None:
                            print(f"  Successfully computed RRBLUP weights: {rrblup_weights.shape}")
                        else:
                            print("  Warning: Could not compute RRBLUP weights, using random initialization")
                    
                    # Initialize the actual model
                    model = DeepBLUP(
                        input_dim=input_dim,
                        output_dim=output_dim,
                        rrblup_lambda=float(hp.get('rrblup_lambda', '1.0')),
                        dl_hidden_layers=dl_hidden_layers,
                        dropout=float(hp.get('dropout', '0.2')),
                        activation=hp.get('activation', 'relu'),
                        rrblup_weights=rrblup_weights,
                        train_rrblup_layer=hp.get('train_rrblup_layer', 'false').lower() == 'true'
                    )
                    
                    # Train the model
                    learning_rate = float(hp.get('learning_rate', '0.001'))
                    batch_size = int(hp.get('batch_size', '32'))
                    epochs = int(hp.get('epochs', '100'))
                    weight_decay = float(hp.get('weight_decay', '0.0001'))
                    
                    train_loader, val_loader = self._create_data_loaders(X_current, y_current, batch_size)
                    trained_model = self._train_deepblup_model(
                        model, train_loader, val_loader, learning_rate, epochs, 
                        weight_decay, model_name
                    )
                    
                    self.models[model_name] = trained_model
                    continue
                
                elif model_name == 'DeepResBLUP':
                    model = self.create_model('DeepResBLUP', X_current)
                    if model is not None:
                        model.fit(X_current, y_current)
                        self.models[model_name] = model
                    continue

                # Handle GNN models separately
                elif model_name in ['GraphConvGS', 'GraphAttnGS', 'GraphSAGEGS', 'GraphFormer'] and TORCH_AVAILABLE:
                    if not PYG_AVAILABLE:
                        print(f"torch_geometric (PyG) is not installed. Install it to use {model_name}.")
                        continue
            
                    # Pass the model_name parameter explicitly
    
                    if model_name == 'GraphFormer':
                        # build sample-graph edge_index/edge_weight exactly as GraphSAGEGS/GraphAttnGS/GraphConvGS do
                        edge_index, edge_weight = build_sample_graph(
                            X_current,
                            method=self.hyperparams.get(model_name, {}).get('graph_method', 'knn'),
                            top_k=int(self.hyperparams.get(model_name, {}).get('top_k', 20)),
                            metric=self.hyperparams.get(model_name, {}).get('knn_metric', 'euclidean')
                        )

                        # Train GraphFormer — the function should NOT take edge_index or self as a parameter
                        self.train_graphformer(X_current, y_current, edge_index, edge_weight=edge_weight, model_name='GraphFormer')

                        
                    else:
                        self._train_pyg_gnn(X_current, y_current, model_name)
                    continue
    
                elif model_name == 'RRBLUP':
                    # For multi-trait data, train separate RRBLUP for each trait
                    if y_current.ndim == 2 and y_current.shape[1] > 1:
                        self.models[model_name] = []
                        n_traits = y_current.shape[1]
                        successful_traits = 0
        
                        for i in range(n_traits):
                            try:
                                #print(f"  RRBLUP training trait {i+1}/{n_traits}...")
                                model = self.create_model('RRBLUP', X_current)
                
                                # Extract single trait - THIS IS THE KEY FIX
                                y_single_trait = y_current[:, i]
                                #print(f"  DEBUG: X_current shape: {X_current.shape}, y_single_trait shape: {y_single_trait.shape}")
                
                                model.fit(X_current, y_single_trait)
                                self.models[model_name].append(model)
                                successful_traits += 1

                            except Exception as e:
                                print(f"Error training RRBLUP for trait {i+1}: {e}")
                                import traceback
                                traceback.print_exc()
                                # Add a placeholder to maintain list structure
                                self.models[model_name].append(None)
                
                        #print(f"  RRBLUP: {successful_traits}/{n_traits} traits trained successfully")
                        continue
                    else:
                        # Single trait
                        try:
                            model = self.create_model('RRBLUP', X_current)
            
                            # Ensure y is 1D for single trait
                            if y_current.ndim == 2 and y_current.shape[1] == 1:
                                y_current = y_current.ravel()
                
                            #print(f"  DEBUG RRBLUP single: X shape: {X_current.shape}, y shape: {y_current.shape}")
                            model.fit(X_current, y_current)
                            self.models[model_name] = model
            
                        except Exception as e:
                            print(f"Error training RRBLUP: {e}")
                            import traceback
                            traceback.print_exc()
                            self.models[model_name] = None

                
                # In train_models method, for R models multi-trait case:
                elif model_name in ['R_RRBLUP', 'R_GBLUP']:
                    # For multi-trait data, train separate models for each trait
                    if y_current.ndim == 2 and y_current.shape[1] > 1:
                        self.models[model_name] = []
                        n_traits = y_current.shape[1]
                        successful_models = 0
        
                        for i in range(n_traits):
                            try:
                                model = self.create_model(model_name, X_current)
                                result = model.fit(X_current, y_current[:, i])
                
                                # Check if fitting was successful
                                if hasattr(result, 'is_fitted') and result.is_fitted:
                                    self.models[model_name].append(result)
                                    successful_models += 1
                                    print(f"  {model_name} Trait {i+1}/{n_traits} - SUCCESS")
                                else:
                                    print(f"  {model_name} Trait {i+1}/{n_traits} - FAILED (not fitted)")
                                    # Create a fallback model instead of None
                                    fallback_model = HighPerformanceRRBLUP()
                                    fallback_model.fit(X_current, y_current[:, i])
                                    self.models[model_name].append(fallback_model)
                    
                            except Exception as e:
                                print(f"Error training {model_name} for trait {i+1}: {e}")
                                # Create a fallback model instead of None
                                try:
                                    fallback_model = HighPerformanceRRBLUP()
                                    fallback_model.fit(X_current, y_current[:, i])
                                    self.models[model_name].append(fallback_model)
                                    print(f"  {model_name} Trait {i+1}/{n_traits} - FALLBACK USED")
                                except:
                                    print(f"  {model_name} Trait {i+1}/{n_traits} - COMPLETE FAILURE")
                                    # Last resort: add a model that returns zeros
                                    class ZeroModel:
                                        def __init__(self):
                                            self.is_fitted = True
                                        def predict(self, X):
                                            return np.zeros((X.shape[0], 1))
                                    self.models[model_name].append(ZeroModel())
        
                        print(f"  {model_name}: {successful_models}/{n_traits} traits trained successfully")
        
                    else:
                        # Single trait case
                        try:
                            model = self.create_model(model_name, X_current)
                            result = model.fit(X_current, y_current)
                            if hasattr(result, 'is_fitted') and result.is_fitted:
                                self.models[model_name] = result
                            else:
                                print(f"{model_name} failed to fit")
                                self.models[model_name] = None
                        except Exception as e:
                            print(f"Error training {model_name}: {e}")
                            self.models[model_name] = None
      
                elif model_name in ['BRR'] and BAYESIAN_AVAILABLE:
                    #Bayesian models handling
                    self.models[model_name] = []
                    n_traits = y_train.shape[1]
                                   
                    for i in range(n_traits):
                        model = self.create_model('BRR', X_train)
                        model.fit(X_train, y_train[:, i])
                        self.models[model_name].append(model)

                elif model_name in ['LightGBM'] and LGBM_AVAILABLE:
                    # LightGBM multi-trait handling
                    self.models[model_name] = []
                    n_traits = y_train.shape[1]
                    for i in range(n_traits):
                        model = self.create_model('LightGBM', X_train)
                        model.fit(X_train, y_train[:, i])
                        self.models[model_name].append(model)

                elif model_name in ['BayesA'] and BAYESIAN_AVAILABLE:
                    # BayesCpi and BayesA multi-trait handling
                    self.models[model_name] = []
                    n_traits = y_train.shape[1]
                    for i in range(n_traits):
                        model = self.create_model(model_name, X_train)
                        model.fit(X_train, y_train[:, i])
                        self.models[model_name].append(model)

                elif model_name == 'RFR':
                    n_estimators = int(self.hyperparams.get('RFR', {}).get('n_estimators', 100))
                    max_depth = self.hyperparams.get('RFR', {}).get('max_depth', None)
                    if max_depth and max_depth != 'None':
                        max_depth = int(max_depth)
                    else:
                        max_depth = None
    
                    # FIX: Handle single trait case by flattening y
                    if y_train.ndim == 2 and y_train.shape[1] == 1:
                        y_train_flat = y_train.ravel()
                    else:
                        y_train_flat = y_train

                    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, 
                               random_state=self.config['seed'], n_jobs=-1)
                    model.fit(X_train, y_train_flat)
                    self.models[model_name] = model

                elif model_name == 'LASSO':
                    alpha = float(self.hyperparams.get('LASSO', {}).get('alpha', 1.0))
                    model = Lasso(alpha=alpha, random_state=self.config['seed'])
    
                    # Train the model
                    model.fit(X_train, y_train)

                    # Validate predictions
                    if not self._validate_predictions(model, X_train, y_train, model_name):
                        # If constant predictions, try with stronger regularization
                        print(f"Retraining {model_name} with stronger regularization...")
                        model = Lasso(alpha=alpha * 10.0, random_state=self.config['seed'])
                        model.fit(X_train, y_train)
                    self.models[model_name] = model

                elif model_name in ['CNN', 'MLPGS', 'HybridAttnMLP',
                    'DNNGS', 'Transformer',
                    'FusionGS', 'EfficientGSFormer'] and TORCH_AVAILABLE:
                    # Get hyperparameters
                    learning_rate = float(self.hyperparams.get(model_name, {}).get('learning_rate', '0.001'))
                    batch_size = int(self.hyperparams.get(model_name, {}).get('batch_size', '32'))
                    epochs = int(self.hyperparams.get(model_name, {}).get('epochs', '100'))
    
                    # Create data loaders
                    train_loader, val_loader = self._create_data_loaders(X_train, y_train, batch_size)
    
                    # Initialize model
                    input_dim = X_train.shape[1]
                    output_dim = y_train.shape[1]
    
                    if model_name == 'CNN':
                        model_config = self.create_model('CNN', X_train)
                        model = CNNModel(input_dim, output_dim, 
                                        hidden_channels=model_config['hidden_channels'],
                                        kernel_size=model_config['kernel_size'],
                                        pool_size=model_config['pool_size'],
                                        dropout=model_config['dropout'])
                        # Use standard training for CNN
                        trained_model = self._train_pytorch_model(
                            model, train_loader, val_loader, learning_rate, epochs, model_name
                        )
                        self.models[model_name] = trained_model

                    elif model_name == 'MLPGS':  # MLPGS
                        model_config = self.create_model('MLPGS', X_train)
                        model = MLPGSModel(input_dim, output_dim,
                                     hidden_layers=model_config['hidden_layers'],
                                     dropout=model_config['dropout'],
                                     norm=model_config.get('norm', 'layer'),
                                     activation=model_config.get('activation', 'gelu'),
                                     residual=model_config.get('residual', True),
                                     input_dropout=model_config.get('input_dropout', 0.05))
                        # Use standard training for CNN
                        trained_model = self._train_pytorch_model(
                            model, train_loader, val_loader, learning_rate, epochs, model_name
                        )
                        self.models[model_name] = trained_model
 

                    elif model_name == 'HybridAttnMLP':  # HybridAttnMLP
                        params = self.create_model('HybridAttnMLP', X_train)
                        model = HybridAttnMLPModel(
                            input_dim=input_dim,
                            output_dim=output_dim,

                            # list-like hyperparameters
                            mlp_hidden=parse_list(params.get('mlp_hidden', [256, 128])),
                            cnn_channels=parse_list(params.get('cnn_channels', [32, 64])),

                            # convolution settings
                            kernel_size=int(params.get('kernel_size', 3)),
                            pool_size=int(params.get('pool_size', 2)),

                            # attention settings
                            attention_heads=int(params.get('attention_heads', 4)),
                            hidden_size=int(params.get('hidden_size', 128)),

                            # regularization
                            dropout=float(params.get('dropout', 0.1)),
                            input_dropout=float(params.get('input_dropout', 0.0)),
                        )
                        
                        # Use standard training for CNN
                        trained_model = self._train_pytorch_model(
                            model, train_loader, val_loader, learning_rate, epochs, model_name
                        )
                        self.models[model_name] = trained_model
                    
                    elif model_name == 'DNNGS_v1' and TORCH_AVAILABLE:
                        model_config = self.create_model('DNNGS', X_train)
                        model = DNNGSModel(input_dim, output_dim,
                                    hidden_layers=model_config['hidden_layers'],
                                    dropout=model_config['dropout'],
                                    activation=model_config['activation'],
                                    batch_norm=model_config['batch_norm'],
                                    input_dropout=model_config['input_dropout'])
    
                        # Get DNNGS specific parameters
                        learning_rate = float(self.hyperparams.get('DNNGS', {}).get('learning_rate', '0.001'))
                        batch_size = int(self.hyperparams.get('DNNGS', {}).get('batch_size', '32'))
                        epochs = int(self.hyperparams.get('DNNGS', {}).get('epochs', '150'))
                        weight_decay = float(self.hyperparams.get('DNNGS', {}).get('weight_decay', '0.0001'))

                        # Create data loaders
                        train_loader, val_loader = self._create_data_loaders(X_train, y_train, batch_size)
    
                        # Train with enhanced optimizer (similar to AttnCNNGS but with different defaults)
                        trained_model = self._train_advanced_pytorch_model(
                            model, train_loader, val_loader, learning_rate, epochs, model_name,
                            weight_decay=weight_decay, grad_clip=1.0, 
                            warmup_ratio=0.1, patience=20
                        )
                        self.models[model_name] = trained_model
   
                    elif model_name == 'DNNGS' and TORCH_AVAILABLE:
                        # Extract model config (hidden layers, dropout, etc.)
                        model_config = self.create_model('DNNGS', X_train)

                        # NEW: additional v2 hyperparameters (with defaults)
                        #norm_type = self.hyperparams.get('DNNGS', {}).get('norm_type', 'layernorm')
                        #residual = self.hyperparams.get('DNNGS', {}).get('residual', True)
                        norm_type = 'none'
                        residual = False

                        # Build upgraded DNNGS v2 model
                        model = DNNGSModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            hidden_layers=model_config.get('hidden_layers'),
                            dropout=model_config.get('dropout'),
                            activation=model_config.get('activation'),
                            batch_norm=model_config.get('batch_norm'),  # still supported for backward compatibility
                            input_dropout=model_config.get('input_dropout'),
                            norm_type=norm_type,
                            residual=residual
                        )

                        # Learning hyperparameters
                        learning_rate = float(self.hyperparams.get('DNNGS', {}).get('learning_rate', '0.001'))
                        batch_size = int(self.hyperparams.get('DNNGS', {}).get('batch_size', '32'))
                        epochs = int(self.hyperparams.get('DNNGS', {}).get('epochs', '300'))
                        weight_decay = float(self.hyperparams.get('DNNGS', {}).get('weight_decay', '0.000001'))
                        patience = float(self.hyperparams.get('DNNGS', {}).get('patience', '40'))
                        grad_clip=1.0
                        warmup_ratio=0.1

                        # Data loaders
                        train_loader, val_loader = self._create_data_loaders(X_train, y_train, batch_size)

                        # Train using your advanced PyTorch trainer
                        trained_model = self._train_advanced_pytorch_model(
                            model,
                            train_loader,
                            val_loader,
                            learning_rate,
                            epochs,
                            model_name,
                            weight_decay,
                            grad_clip,
                            warmup_ratio,
                            patience
                        )

                        self.models[model_name] = trained_model

                    elif model_name == 'AttnCNNGS':
                        model_config = self.create_model('AttnCNNGS', X_train)
                        model = AttnCNNGSModel(input_dim, output_dim,
                                         cnn_channels=model_config['cnn_channels'],
                                         kernel_size=model_config['kernel_size'],
                                         pool_size=model_config['pool_size'],
                                         attention_heads=model_config['attention_heads'],
                                         hidden_size=model_config['hidden_size'],
                                         dropout=model_config['dropout'],
                                         attention_dropout=model_config['attention_dropout'])

                        # Train model with enhanced optimizer for AttnCNNGS
                        # Get AttnCNNGS specific parameters (only for AttnCNNGS)
                        weight_decay = float(self.hyperparams.get('AttnCNNGS', {}).get('weight_decay', '0.001'))
                        grad_clip = float(self.hyperparams.get('AttnCNNGS', {}).get('grad_clip', '1.0'))
                        warmup_ratio = float(self.hyperparams.get('AttnCNNGS', {}).get('warmup_ratio', '0.1'))
                        patience = int(self.hyperparams.get('AttnCNNGS', {}).get('patience', '20'))
                        
                        trained_model = self._train_advanced_pytorch_model(
                            model, train_loader, val_loader, learning_rate, epochs, model_name,
                            weight_decay=weight_decay, grad_clip=grad_clip, 
                            warmup_ratio=warmup_ratio, patience=patience
                        )
                        self.models[model_name] = trained_model

                    elif model_name == 'Transformer':
                        model_config = self.create_model('Transformer', X_train)
                        model = GenomicTransformerModel(input_dim, output_dim,
                                                  d_model=model_config['d_model'],
                                                  nhead=model_config['nhead'],
                                                  num_layers=model_config['num_layers'],
                                                  dim_feedforward=model_config['dim_feedforward'],
                                                  dropout=model_config['dropout'])
                    
                        # Get Transformer specific parameters
                        weight_decay = float(self.hyperparams.get('Transformer', {}).get('weight_decay', '0.01'))
                        grad_clip = float(self.hyperparams.get('Transformer', {}).get('grad_clip', '1.0'))
                        warmup_ratio = float(self.hyperparams.get('Transformer', {}).get('warmup_ratio', '0.1'))
                        patience = int(self.hyperparams.get('Transformer', {}).get('patience', '30'))
                    
                        # Use enhanced training for Transformer (similar to AttnCNNGS)
                        trained_model = self._train_advanced_pytorch_model(
                            model, train_loader, val_loader, learning_rate, epochs, model_name,
                            weight_decay=weight_decay, grad_clip=grad_clip, 
                            warmup_ratio=warmup_ratio, patience=patience
                        )

                        self.models[model_name] = trained_model

                    elif model_name == 'FusionGS':
                        params = self.create_model('FusionGS', X_train)
                        model = FusionGSModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            cnn_channels=params['cnn_channels'],
                            kernel_size=params['kernel_size'],
                            pool_size=params['pool_size'],
                            mlp_hidden=params['mlp_hidden'],
                            fusion_hidden=params['fusion_hidden'],
                            dropout=params['dropout'],
                            input_dropout=params['input_dropout'],
                        )

                        # Use advanced trainer (cosine warmup + early stopping)
                        hp = self.hyperparams.get('FusionGS', {})
                        learning_rate = float(hp.get('learning_rate', '0.00075'))
                        batch_size = int(hp.get('batch_size', '64'))
                        epochs = int(hp.get('epochs', '200'))
                        weight_decay = float(hp.get('weight_decay', '0.0005'))
                        warmup_ratio = float(hp.get('warmup_ratio', '0.1'))
                        patience = int(hp.get('patience', '20'))

                        train_loader, val_loader = self._create_data_loaders(X_train, y_train, batch_size)

                        trained_model = self._train_advanced_pytorch_model(
                            model, train_loader, val_loader,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            model_name=model_name,
                            weight_decay=weight_decay,
                            grad_clip=1.0,
                            warmup_ratio=warmup_ratio,
                            patience=patience
                        )
                        self.models[model_name] = trained_model

                    elif model_name == 'EfficientGSFormer':
                        params = self.create_model('EfficientGSFormer', X_train)
                        model = EfficientGSFormerModel(
                            input_dim=input_dim,
                            output_dim=output_dim,
                            d_model=params['d_model'],
                            nhead=params['nhead'],
                            num_layers=params['num_layers'],
                            dim_feedforward=params['dim_feedforward'],
                            dropout=params['dropout'],
                            patch_size=params['patch_size'],
                        )

                        hp = self.hyperparams.get('EfficientGSFormer', {})
                        learning_rate = float(hp.get('learning_rate', '0.0005'))
                        batch_size = int(hp.get('batch_size', '32'))
                        epochs = int(hp.get('epochs', '200'))
                        weight_decay = float(hp.get('weight_decay', '0.0005'))
                        patience = int(hp.get('patience', '20'))

                        train_loader, val_loader = self._create_data_loaders(X_train, y_train, batch_size)

                        trained_model = self._train_advanced_pytorch_model(
                            model, train_loader, val_loader,
                            learning_rate=learning_rate,
                            epochs=epochs,
                            model_name=model_name,
                            weight_decay=weight_decay,
                            grad_clip=1.0,
                            warmup_ratio=0.1,
                            patience=patience
                        )
                        self.models[model_name] = trained_model

  
                else:
                    # Single model for all traits
                    model = self.create_model(model_name, X_train)
                    if model is not None:
                        model.fit(X_train, y_train)
                        self.models[model_name] = model

            except Exception as e:
                print(f"Error training {model_name}: {e}")

            finally:
                # Calculate and print memory usage and training time for this model
                end_mem = get_memory_usage_mb()
                mem_used = end_mem - start_mem
                print(f"[{self.config['feature_view'].upper()}] [{model_name}] memory used during training: {mem_used:.2f} MB")
                
                model_end_time = time.time()
                model_time = model_end_time - model_start_time
                print(f"[{self.config['feature_view'].upper()}] [{model_name}] training completed in {model_time/60:.2f} minutes ({model_time:.2f} seconds)\n")
                
                # Store the timing data to global variable training_times. Initial first at all levels
                if model_name not in training_times:
                    training_times[model_name] = {}
                if self.config['feature_view'] not in training_times[model_name]:
                    training_times[model_name][self.config['feature_view']] = []
                training_times[model_name][self.config['feature_view']].append(model_time)

                if model_name not in training_memories:
                    training_memories[model_name] = {}
                if self.config['feature_view'] not in training_memories[model_name]:
                    training_memories[model_name][self.config['feature_view']] = []
                training_memories[model_name][self.config['feature_view']].append(mem_used)

    
        # Train EnsembleGS ensemble AFTER ALL base models are trained
        model_name = 'EnsembleGS'
        if model_name in self.config['enabled_models']:
            model_start_time = time.time()
            start_mem = get_memory_usage_mb()
            set_random_seeds(self.config['seed'])
            
            print(f"Training {model_name} ensemble...")
            self._train_stacking(X_train_copy, y_train_copy)
 
            end_mem = get_memory_usage_mb()
            mem_used = end_mem - start_mem
            print(f"[{self.config['feature_view'].upper()}] [{'model_name'}] memory used during training: {mem_used:.2f} MB")
            stacking_time = time.time() - model_start_time
            print(f"[{self.config['feature_view'].upper()}] [{'model_name'}] training completed in {model_time/60:.2f} minutes ({stacking_time:.2f} seconds)\n")

            if model_name not in training_times:
                training_times[model_name] = {}
            if self.config['feature_view'] not in training_times[model_name]:
                training_times[model_name][self.config['feature_view']] = []
            training_times[model_name][self.config['feature_view']].append(model_time)

            if model_name not in training_memories:
                training_memories[model_name] = {}
            if self.config['feature_view'] not in training_memories[model_name]:
                training_memories[model_name][self.config['feature_view']] = []
            training_memories[model_name][self.config['feature_view']].append(mem_used)
  
    def predict(self, X_test, model_name):
        """Predict using a specific model"""
        
        #print(f"[DEBUG]AllGSModels: Predicting {model_name} with X_test shape {X_test.shape}")     

        if model_name == 'DeepResBLUP':
            model = self.models[model_name]
            return model.predict(X_test)
        
        if model_name == 'EnsembleGS':
            return self._predict_stacking(X_test)
    
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not trained")
    
        # Handle GNN models
        if model_name in ['GraphConvGS', 'GraphAttnGS', 'GraphSAGEGS', 'GraphFormer']:
            return self._predict_pyg_gnn(X_test, model_name)
        
        if model_name in ['BRR'] and BAYESIAN_AVAILABLE:
            # Special handling for models with one model per trait
            predictions = []
            for model in self.models[model_name]:
                pred = model.predict(X_test)
                predictions.append(pred)
            return np.column_stack(predictions)

        if model_name == 'RRBLUP':
            model_obj = self.models[model_name]
    
            # Check if model training failed
            if model_obj is None:
                print(f"Warning: RRBLUP model failed during training, returning zeros")
                return np.zeros((X_test.shape[0], 1))
    
            # Multi-trait case
            if isinstance(model_obj, list):
                predictions = []
                for i, model in enumerate(model_obj):
                    if model is None:
                        # Model failed for this trait, return zeros
                        print(f"Warning: RRBLUP model for trait {i+1} failed, returning zeros")
                        pred = np.zeros((X_test.shape[0], 1))
                    else:
                        try:
                            pred = model.predict(X_test)
                            pred = pred.reshape(-1, 1)  # Ensure 2D
                        except Exception as e:
                            print(f"Error predicting with RRBLUP for trait {i+1}: {e}")
                            pred = np.zeros((X_test.shape[0], 1))
                    predictions.append(pred)
        
                # Combine predictions for all traits
                result = np.hstack(predictions)
                #print(f"  RRBLUP combined predictions shape: {result.shape}")
                return result
            else:
                # Single trait
                try:
                    result = model_obj.predict(X_test)
                    # Ensure 2D output for consistency
                    if result.ndim == 1:
                        result = result.reshape(-1, 1)
                    return result
                except Exception as e:
                    print(f"Error predicting with RRBLUP: {e}")
                    return np.zeros((X_test.shape[0], 1))

        # In AllGSModels.predict method, update the R model section:
        if model_name in ['R_RRBLUP', 'R_GBLUP']:
            model_obj = self.models[model_name]
    
            # Check if model training failed
            if model_obj is None:
                print(f"Warning: {model_name} model failed during training, returning zeros")
                return np.zeros((X_test.shape[0], len(self.config.get('label_cols', [1]))))
    
            # Multi-trait case: model_obj is a list of models
            if isinstance(model_obj, list):
                #print(f"DEBUG: {model_name} - multi-trait, {len(model_obj)} traits")
                predictions = []
                for i, model in enumerate(model_obj):
                    if model is None or not hasattr(model, 'is_fitted') or not model.is_fitted:
                        print(f"Warning: {model_name} model for trait {i+1} failed, returning zeros")
                        pred = np.zeros((X_test.shape[0], 1))
                    else:
                        try:
                            #print(f"DEBUG: {model_name} trait {i+1} - predicting with X_test shape {X_test.shape}")
                            pred = model.predict(X_test)
                            # Ensure 2D shape
                            if pred.ndim == 1:
                                pred = pred.reshape(-1, 1)
                            #print(f"DEBUG: {model_name} trait {i+1} - prediction shape {pred.shape}")
                        except Exception as e:
                            print(f"Error predicting with {model_name} for trait {i+1}: {e}")
                            pred = np.zeros((X_test.shape[0], 1))
                    predictions.append(pred)
        
                # Combine predictions for all traits
                result = np.hstack(predictions)
                #print(f"DEBUG: {model_name} - combined predictions shape {result.shape}")
                return result
    
            # Single trait case
            else:
                if not hasattr(model_obj, 'is_fitted') or not model_obj.is_fitted:
                    print(f"Warning: {model_name} model not fitted, returning zeros")
                    return np.zeros((X_test.shape[0], 1))
        
                try:
                    #print(f"DEBUG: {model_name} - single trait, predicting with X_test shape {X_test.shape}")
                    result = model_obj.predict(X_test)
                    if result.ndim == 1:
                        result = result.reshape(-1, 1)
                    #print(f"DEBUG: {model_name} - prediction shape {result.shape}")
                    return result
                except Exception as e:
                    print(f"Error predicting with {model_name}: {e}")
                    return np.zeros((X_test.shape[0], 1))

        elif model_name == 'LightGBM' and LGBM_AVAILABLE:
            # LightGBM multi-trait prediction
            predictions = []
            for model in self.models[model_name]:
                pred = model.predict(X_test)
                predictions.append(pred)
            return np.column_stack(predictions)

        # In the predict method, add handling for CNN and HybridAttnMLP:
        elif model_name in ['CNN', 'MLPGS', 'HybridAttnMLP',
                            'DNNGS', 'Transformer',
                            'FusionGS', 'EfficientGSFormer'] and TORCH_AVAILABLE:
            model = self.models[model_name]
            model.eval()
    
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device)
    
            with torch.no_grad():
                X_test_tensor = torch.FloatTensor(X_test).to(device)
                predictions = model(X_test_tensor).cpu().numpy()
    
            return predictions

        # Use safe prediction for classical models
        elif model_name in ['LASSO', 'ElasticNet', 'RFR', 'BRR']:
            return self.safe_predict(self.models[model_name], X_test, model_name)

        else:
            return self.models[model_name].predict(X_test)

    def _train_pytorch_model(self, model, train_loader, val_loader=None, 
                        learning_rate=0.001, epochs=100, model_name=""):
        """Train a PyTorch model"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
    
        # Reset seeds for each model to ensure consistency
        set_random_seeds(self.config['seed'])
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        criterion = nn.MSELoss()
        try:
            use_huber = str(self.hyperparams.get(model_name, {}).get('use_huber', 'false')).lower() == 'true'
            if use_huber:
                delta = float(self.hyperparams.get(model_name, {}).get('huber_delta', '1.0'))
                criterion = nn.HuberLoss(delta=delta)
        except Exception:
            pass
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
        best_loss = float('inf')
        patience_counter = 0
        patience = 20
    
        ckpt_dir = os.path.join(self.config['results_dir'], "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        tag = f"{self.config['feature_view']}_{model_name}"
        ckpt_path = os.path.join(ckpt_dir, f"best_{tag}.pth")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
        
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                optimizer.step()
            
                train_loss += loss.item() * batch_X.size(0)
        
            train_loss /= len(train_loader.dataset)
        
            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
            
                val_loss /= len(val_loader.dataset)
                scheduler.step(val_loss)
            
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    # Save best model
                    #torch.save(model.state_dict(), f"best_{model_name}.pth")
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
                if epoch % 10 == 0:
                    print(f"    [{model_name}] Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    print(f"    [{model_name}] Epoch {epoch+1}/{epochs}, Train Loss: {train_loss:.4f}")
    
        # Load best model
        #if val_loader and os.path.exists(f"best_{model_name}.pth"):
        #    model.load_state_dict(torch.load(f"best_{model_name}.pth"))
        if val_loader and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))    
    
        return model

    def _train_advanced_pytorch_model(self, model, train_loader, val_loader=None, 
                           learning_rate=0.00075, epochs=300, model_name="",
                           weight_decay=0.001, grad_clip=1.0, 
                           warmup_ratio=0.1, patience=20):
        """Enhanced training method for AttnCNNGS with advanced optimization"""
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is not available")
    
        # Reset seeds for each model to ensure consistency
        set_random_seeds(self.config['seed'])
            
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
    
        # Use Huber loss for more robust training
        criterion = nn.HuberLoss(delta=1.0)
    
        # Optimizer with weight decay
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
        # Learning rate scheduler with warmup
        total_steps = len(train_loader) * epochs
        warmup_steps = int(total_steps * warmup_ratio)
    
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
    
        scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
        best_loss = float('inf')
        patience_counter = 0
    
        ckpt_dir = os.path.join(self.config['results_dir'], "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        tag = f"{self.config['feature_view']}_{model_name}"
        ckpt_path = os.path.join(ckpt_dir, f"best_{tag}.pth")

        for epoch in range(epochs):
            # Training
            model.train()
            train_loss = 0.0
        
            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            
                optimizer.zero_grad()
                outputs = model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
            
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            
                optimizer.step()
                scheduler.step()
            
                train_loss += loss.item() * batch_X.size(0)
        
            train_loss /= len(train_loader.dataset)
        
            # Validation
            if val_loader:
                model.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                        outputs = model(batch_X)
                        loss = criterion(outputs, batch_y)
                        val_loss += loss.item() * batch_X.size(0)
            
                val_loss /= len(val_loader.dataset)
            
                # Early stopping
                if val_loss < best_loss:
                    best_loss = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    patience_counter += 1
                    if patience_counter >= patience:
                        print(f"Early stopping at epoch {epoch+1}")
                        break
            
                if epoch % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"    [{model_name}] Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            else:
                if epoch % 10 == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    print(f"    [{model_name}] Epoch {epoch+1}/{epochs}, LR: {current_lr:.6f}, Train Loss: {train_loss:.4f}")
    
        # Load best model

        if val_loader and os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            
            # Use the actual model_name parameter instead of hardcoded "AttnCNNGS"
            # both AttnCNNGS and Transformer use this training method
            print(f"Loaded best {model_name} model with validation loss: {best_loss:.4f}")

        return model

    def _create_data_loaders(self, X_train, y_train, batch_size=32, val_ratio=0.1):
        """Create data loaders for training and validation"""
        if val_ratio > 0:
            # Split training data for validation
            X_train, X_val, y_train, y_val = train_test_split(
                X_train, y_train, test_size=val_ratio, random_state=self.config['seed']
            )
        
            train_dataset = GenomicDataset(X_train, y_train)
            val_dataset = GenomicDataset(X_val, y_val)
        
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
        
            return train_loader, val_loader
        else:
            train_dataset = GenomicDataset(X_train, y_train)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
            return train_loader, None

    # The following fuctions: newly added in the *MLP_phased2_stackfix.pl
    def _get_stacking_requested(self):
        cfg = self.hyperparams.get('EnsembleGS', {}) or {}
        raw = cfg.get('base_models', '')
        requested = [m.strip() for m in raw.split(',') if m.strip()]
        requested = [m for m in requested if m != 'EnsembleGS']  # sanitize
        meta_model = str(cfg.get('meta_model', 'linear')).lower()
        meta_alpha = float(cfg.get('meta_alpha', 0.0))
        return requested, meta_model, meta_alpha

    def _collect_available_bases(self, requested):
        trained = [k for k, v in self.models.items() if v is not None]
        print(f"EnsembleGS base models (requested): {requested}")
        print(f"Models available (trained):   {trained}")
        available = [m for m in requested if m in self.models and self.models[m] is not None]
        if len(available) == 0:
            raise ValueError("EnsembleGS: none of the requested base models are trained.")
        return available
        
    def _train_stacking(self, X_train, y_train, X_val=None, y_val=None, current_fold_seed=42):
        """
        Multi-trait aware EnsembleGS.
        - Builds meta-features from base models' predictions.
        - Fits one meta-learner per trait when y has multiple columns.
        """
        # Reset seeds for each model to ensure consistency
        set_random_seeds(self.config['seed'])
        
        requested = self.hyperparams.get('EnsembleGS', {}).get('base_models', 'BRR,HybridAttnMLP')
        requested = [m.strip() for m in requested.split(',') if m.strip() and m.strip() != 'EnsembleGS']
        meta_model = str(self.hyperparams.get('EnsembleGS', {}).get('meta_model', 'linear')).lower()
        meta_alpha = float(self.hyperparams.get('EnsembleGS', {}).get('meta_alpha', 0.0))

        # Available bases (trained this fold)
        trained = [k for k, v in self.models.items() if v is not None]
        print(f"EnsembleGS base models (requested): {requested}")
        print(f"Models available (trained):   {trained}")
        base_names = [m for m in requested if m in self.models and self.models[m] is not None]
        if len(base_names) == 0:
            raise ValueError("EnsembleGS: none of the requested base models are trained.")
        if len(base_names) == 1:
            print(f"[EnsembleGS] Only one available base model: {base_names[0]} (calibrated passthrough).")

        import numpy as np
        # Choose target/prediction source (validation preferred, else in-sample)
        if X_val is not None and y_val is not None:
            X_src = X_val
            y_src = np.asarray(y_val)
            prov = 'validation'
        else:
            print("[EnsembleGS] Using in-sample training predictions for meta-learner.")
            X_src = X_train
            y_src = np.asarray(y_train)
            prov = 'in-sample'

        # Determine #traits
        if y_src.ndim == 1:
            n_targets = 1
            y_src_2d = y_src.reshape(-1, 1)
        else:
            n_targets = y_src.shape[1]
            y_src_2d = y_src

        # Collect base predictions once (to avoid recompute per trait)
        # Each entry is either (n_samples,) or (n_samples, n_targets)
        base_preds = []
        for name in base_names:
            p = self.predict(X_src, name)
            p = np.asarray(p)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            base_preds.append(p)

        # Build and fit one meta-model per trait
        from sklearn.linear_model import Ridge
        meta_models = []
        for t in range(n_targets):
            cols = []
            for p in base_preds:
                # If the base predicted all traits, take column t; if it predicted a single column, use it
                if p.shape[1] == n_targets:
                    cols.append(p[:, [t]])
                elif p.shape[1] == 1:
                    cols.append(p)  # single-output base
                else:
                    # Fallback: take first column (defensive; should not happen with well-formed bases)
                    cols.append(p[:, [0]])
            X_meta_t = np.hstack(cols)  # (n_samples, n_bases)
            y_t = y_src_2d[:, t].ravel()

            # Meta-learner
            if meta_model in ('linear', 'ridge'):
                meta = Ridge(alpha=float(meta_alpha), fit_intercept=True)
            else:
                print(f"[EnsembleGS] Unknown meta_model='{meta_model}', falling back to Ridge.")
                meta = Ridge(alpha=float(meta_alpha), fit_intercept=True)

            meta.fit(X_meta_t, y_t)
            meta_models.append(meta)

        print(f"[EnsembleGS] Meta-features for each trait: (n_samples={X_meta_t.shape[0]}, n_bases={X_meta_t.shape[1]}) (source: {prov})")

        # Save for prediction
        self.models['EnsembleGS'] = {
            'base_names': base_names,
            'meta_models': meta_models,
            'n_targets': n_targets
        }

    def _predict_stacking(self, X):
        """
        Multi-trait aware prediction.
        Returns (n_samples,) for single-trait; (n_samples, n_traits) for multi-trait.
        """
        info = self.models.get('EnsembleGS', None)
        if not isinstance(info, dict):
            raise ValueError("EnsembleGS model not trained")

        import numpy as np
        base_names = info['base_names']
        meta_models = info['meta_models']
        n_targets = info.get('n_targets', 1)

        # Get base predictions on X
        base_preds = []
        for name in base_names:
            p = self.predict(X, name)
            p = np.asarray(p)
            if p.ndim == 1:
                p = p.reshape(-1, 1)
            base_preds.append(p)

        n_samples = base_preds[0].shape[0]

        # If single-target stacking, just assemble one meta-matrix
        if n_targets == 1:
            cols = []
            for p in base_preds:
                # If a base produced multi-col predictions accidentally, take its first column
                cols.append(p[:, [0]])
            X_meta = np.hstack(cols)  # (n_samples, n_bases)
            y_hat = meta_models[0].predict(X_meta).reshape(-1)
            return y_hat

        # Multi-target: predict one trait at a time
        Y_hat = np.zeros((n_samples, n_targets), dtype=float)
        for t in range(n_targets):
            cols = []
            for p in base_preds:
                if p.shape[1] == n_targets:
                    cols.append(p[:, [t]])
                else:
                    cols.append(p[:, [0]])
            X_meta_t = np.hstack(cols)  # (n_samples, n_bases)
            Y_hat[:, t] = meta_models[t].predict(X_meta_t).reshape(-1)

        return Y_hat

    def _predict_pyg_gnn(self, X_test, model_name):
        """Predict using PyG GNN models - UPDATED FOR GraphFormer"""
        m = self.models.get(model_name, None)
        if m is None:
            print(f"Warning: {model_name} model not trained properly")
            return np.zeros((X_test.shape[0], 1))
        
        try:
            model = m['model']
            graph_method = m.get('graph_method', 'knn')
            knn_metric = m.get('knn_metric', 'euclidean')
            top_k = int(self.hyperparams.get(model_name, {}).get('top_k', '20'))
            
            # For prediction with new samples, create a new graph
            print(f"[{model_name}] Building prediction graph with {X_test.shape[0]} samples using {graph_method.upper()}")
            test_edge_index, test_edge_weight = build_sample_graph(
                X_test, 
                method=graph_method, 
                top_k=top_k, 
                metric=knn_metric
            )
            
            # Create dummy targets for dataset
            dummy_y = np.zeros((X_test.shape[0], m.get('output_dim', 1)))
            test_ds = SingleGraphDataset(X_test, dummy_y, test_edge_index, test_edge_weight)
            test_loader = PyGDataLoader(test_ds, batch_size=1, shuffle=False)
            
            # Predict
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model = model.to(device).eval()
            
            preds = []
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    pred = model(batch)
                    preds.append(pred.cpu().numpy())
            
            if len(preds) > 0:
                preds = np.concatenate(preds, axis=0)
            else:
                preds = np.zeros((X_test.shape[0], m.get('output_dim', 1)))
            
            # Ensure output shape matches expected
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            
            return preds
            
        except Exception as e:
            print(f"Error predicting with {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((X_test.shape[0], m.get('output_dim', 1)))

    def _train_pyg_gnn(self, X_train, y_train, model_name):
        """Train PyG GNN models (GraphConvGS, GraphAttnGS, GraphSAGEGS) with KNN graph construction"""
        gnn_start_time = time.time()

        try:
            # Set seed again for this model
            set_random_seeds(self.config['seed'])
        
            # Get common hyperparameters
            lr = float(self.hyperparams.get(model_name, {}).get('learning_rate', '0.0005'))
            epochs = int(self.hyperparams.get(model_name, {}).get('epochs', '200'))
            weight_decay = float(self.hyperparams.get(model_name, {}).get('weight_decay', '0.0'))
            top_k = int(self.hyperparams.get(model_name, {}).get('top_k', '20'))
            graph_method = self.hyperparams.get(model_name, {}).get('graph_method', 'knn')
            knn_metric = self.hyperparams.get(model_name, {}).get('knn_metric', 'euclidean')
            patience = int(self.hyperparams.get(model_name, {}).get('patience', '20'))

            print(f"[{model_name}] Building sample graph with {X_train.shape[0]} samples using {graph_method.upper()} (top_k={top_k}, metric={knn_metric})")
            
            # Build graph using specified method
            edge_index, edge_weight = build_sample_graph(
                X_train, 
                method=graph_method, 
                top_k=top_k, 
                metric=knn_metric
            )
            print(f"[{model_name}] Graph built with {edge_index.shape[1]} edges")

            num_samples = X_train.shape[0]

            # 80/20 split
            idx = np.arange(num_samples)
            np.random.shuffle(idx)
            val_size = max(1, num_samples // 5)

            val_idx = idx[:val_size]
            train_idx = idx[val_size:]

            train_mask = np.zeros(num_samples, dtype=bool)
            val_mask = np.zeros(num_samples, dtype=bool)

            train_mask[train_idx] = True
            val_mask[val_idx] = True

            # Create single graph dataset
            dataset = SingleGraphDataset(
                X_train, y_train, 
                edge_index, edge_weight,
                train_mask=train_mask,
                val_mask=val_mask
            )
            train_loader = PyGDataLoader(dataset, batch_size=1, shuffle=False)
            val_loader = train_loader   # SAME dataset, but val_mask distinguishes nodes
            
            # Instantiate model
            input_dim = X_train.shape[1]
            output_dim = y_train.shape[1]
        
            # Get model configuration and instantiate appropriate model
            model_config = self.create_model(model_name, X_train)
            
            if model_name == 'GraphConvGS':
                model = GenomicSampleGraphConvGSModel(
                    in_channels=input_dim,
                    hidden_channels=model_config['hidden_channels'],
                    num_layers=model_config['num_layers'],
                    hidden_mlp=model_config['hidden_mlp'],
                    dropout=model_config.get('dropout', 0.2),
                    output_dim=output_dim
                )
            elif model_name == 'GraphAttnGS':
                model = GenomicSampleGraphAttnGSModel(
                    in_channels=input_dim,
                    hidden_channels=model_config['hidden_channels'],
                    num_layers=model_config['num_layers'],
                    heads=model_config['heads'],
                    hidden_mlp=model_config['hidden_mlp'],
                    dropout=model_config.get('dropout', 0.2),
                    output_dim=output_dim
                )
            elif model_name == 'GraphSAGEGS':
                model = GenomicSampleSAGEModel(
                    in_channels=input_dim,
                    hidden_channels=model_config['hidden_channels'],
                    num_layers=model_config['num_layers'],
                    hidden_mlp=model_config['hidden_mlp'],
                    dropout=model_config.get('dropout', 0.2),
                    output_dim=output_dim,
                    aggr=model_config.get('aggr', 'mean')
                )

            elif model_name == 'GraphFormer':
                return self.train_graphformer(X_train, y_train, edge_index, edge_weight, model_name='GraphFormer')

            #elif model_name == 'GraphFormer':
            #    return self._train_graphformer(X_train, y_train, edge_index, edge_weight, model_name='GraphFormer')

            else:
                print(f"Unknown GNN model: {model_name}")
                return

            # Train the model
            print(f"[{model_name}] Starting training for {epochs} epochs")
            trained_model = train_single_graph_gnn(
                model, train_loader, val_loader,
                epochs=epochs, lr=lr, weight_decay=weight_decay,
                patience=patience, verbose=True, model_name=model_name
            )

            # Save trained model
            self.models[model_name] = {
                'model': trained_model,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'input_dim': input_dim,
                'output_dim': output_dim,
                'graph_method': graph_method,
                'knn_metric': knn_metric
            }
            print(f"[{model_name}] Training completed successfully")
        
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            if model_name in self.models:
                del self.models[model_name]
        #finally:
            # this prin should be done in train_models function
            #gnn_time = time.time() - gnn_start_time
            #print(f"[{self.config['feature_view'].upper()}] {model_name} training completed in {gnn_time/60:.2f} minutes ({gnn_time:.2f} seconds)")

    def train_graphformer(self, X_train, y_train, edge_index, edge_weight=None, model_name='GraphFormer'):
        """Train GraphFormer model - WITH COMPREHENSIVE DEBUGGING"""
        try:
            # Set seed for reproducibility
            set_random_seeds(self.config['seed'])
            
            # Get hyperparameters with defaults
            hp = self.hyperparams.get('GraphFormer', {})
            gnn_type = hp.get('gnn_type', 'SAGE')
            gnn_hidden = int(hp.get('gnn_hidden', '128'))
            transformer_layers = int(hp.get('transformer_layers', '2'))
            d_model = int(hp.get('d_model', '128'))
            nhead = int(hp.get('nhead', '4'))
            mlp_hidden = int(hp.get('mlp_hidden', '128'))
            dropout = float(hp.get('dropout', '0.1'))
            lr = float(hp.get('learning_rate', '0.001'))
            epochs = int(hp.get('epochs', '500'))
            weight_decay = float(hp.get('weight_decay', '0.001'))
            patience = int(hp.get('patience', '30'))
            
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            # Determine output dimension
            output_dim = y_train.shape[1] if y_train.ndim > 1 else 1
            
            print(f"[{model_name}] Creating model:")
            print(f"  - Input dim: {X_train.shape[1]}, Output dim: {output_dim}")
            print(f"  - GNN type: {gnn_type}, GNN hidden: {gnn_hidden}")
            print(f"  - Transformer: {transformer_layers} layers, d_model: {d_model}")
            print(f"  - Device: {device}")
            
            # Create model
            model = GraphFormerModel(
                in_feats=X_train.shape[1],
                out_dim=output_dim,
                gnn_type=gnn_type,
                gnn_hidden=gnn_hidden,
                transformer_layers=transformer_layers,
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=d_model * 4,
                dropout=dropout,
                mlp_hidden=mlp_hidden
            ).to(device)
            
            # DEBUG: Verify model creation
            print(f"[{model_name}] Model created successfully")
            print(f"[{model_name}] Model has GNN attribute: {hasattr(model, 'gnn')}")
            if hasattr(model, 'gnn'):
                print(f"[{model_name}] GNN layer type: {type(model.gnn)}")
            
            # Create dataset and loader
            dataset = SingleGraphDataset(X_train, y_train, edge_index, edge_weight)
            loader = PyGDataLoader(dataset, batch_size=1, shuffle=False)
            
            print(f"[{model_name}] Dataset created: {len(dataset)} graphs")
            
            # Test forward pass with a single batch
            print(f"[{model_name}] Testing forward pass...")
            model.eval()
            with torch.no_grad():
                test_batch = next(iter(loader)).to(device)
                try:
                    test_output = model(test_batch)
                    print(f"[{model_name}] Forward pass successful! Output shape: {test_output.shape}")
                except Exception as e:
                    print(f"[{model_name}] Forward pass failed: {e}")
                    raise
            
            # Training setup
            optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
            criterion = nn.MSELoss()
            
            best_loss = float('inf')
            patience_counter = 0
            
            ckpt_dir = os.path.join(self.config['results_dir'], "ckpts")
            os.makedirs(ckpt_dir, exist_ok=True)
            ckpt_path = os.path.join(ckpt_dir, f"best_{self.config['feature_view']}_{model_name}.pth")
            
            print(f"[{model_name}] Starting training for {epochs} epochs...")
            
            # Training loop
            for epoch in range(epochs):
                model.train()
                total_loss = 0.0
                batch_count = 0
                
                for batch in loader:
                    batch = batch.to(device)
                    optimizer.zero_grad()
                    
                    preds = model(batch)
                    target = batch.y
                    if target.ndim == 1:
                        target = target.unsqueeze(1)
                    
                    loss = criterion(preds, target)
                    loss.backward()
                    
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    optimizer.step()
                    
                    total_loss += loss.item()
                    batch_count += 1
                
                avg_loss = total_loss / batch_count if batch_count > 0 else total_loss
                
                # Early stopping
                if avg_loss < best_loss - 1e-6:
                    best_loss = avg_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), ckpt_path)
                    if epoch % 10 == 0:
                        print(f"    [{model_name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f} *")
                else:
                    patience_counter += 1
                    if epoch % 10 == 0:
                        print(f"    [{model_name}] Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
                
                if patience_counter >= patience:
                    print(f"    [{model_name}] Early stopping at epoch {epoch+1}")
                    break
            
            # Load best model
            if os.path.exists(ckpt_path):
                model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
                print(f"[{model_name}] Loaded best model with loss: {best_loss:.6f}")
            
            # Store model
            self.models[model_name] = {
                'model': model,
                'edge_index': edge_index,
                'edge_weight': edge_weight,
                'output_dim': output_dim,
                'graph_method': hp.get('graph_method', 'knn'),
                'knn_metric': hp.get('knn_metric', 'euclidean')
            }
            
            print(f"[{model_name}] Training completed successfully")
            return self.models[model_name]
            
        except Exception as e:
            print(f"Error training GraphFormer: {e}")
            import traceback
            traceback.print_exc()
            self.models[model_name] = None
            return None

    def _validate_predictions(self, model, X_train, y_train, model_name):
        """Validate that model produces non-constant predictions"""
        try:
            # Test prediction on training data
            y_pred = model.predict(X_train)
        
            # Check if predictions are constant
            if y_pred.ndim == 1:
                is_constant = np.std(y_pred) < 1e-10
            else:
                is_constant = np.all([np.std(y_pred[:, j]) < 1e-10 for j in range(y_pred.shape[1])])
        
            if is_constant:
                print(f"Warning: {model_name} producing constant predictions. Adding regularization.")
                return False
            return True
        except:
            return True  # If validation fails, proceed anyway

    # for classical models
    def safe_predict(self, model, X, model_name):
        """Safe prediction that handles constant outputs"""
        try:
            predictions = model.predict(X)
        
            # Check for constant predictions
            if predictions.ndim == 1:
                if np.std(predictions) < 1e-10:
                    print(f"Warning: {model_name} produced constant predictions")
                    # Return mean of training data instead
                    if hasattr(model, 'y_train_means_'):
                        return np.full(X.shape[0], model.y_train_means_)
                    else:
                        return np.full(X.shape[0], 0.0)
            else:
                for j in range(predictions.shape[1]):
                    if np.std(predictions[:, j]) < 1e-10:
                        print(f"Warning: {model_name} produced constant predictions for trait {j}")
                        # Replace with small random noise around mean
                        mean_val = np.mean(predictions[:, j])
                        predictions[:, j] = mean_val + np.random.normal(0, 1e-6, predictions.shape[0])
        
            return predictions
        except Exception as e:
            print(f"Error in {model_name} prediction: {e}")
            return np.zeros((X.shape[0], 1)) if X.ndim == 1 else np.zeros((X.shape[0], self._get_output_dim()))
     
    def _train_deepblup_model(self, model, train_loader, val_loader, learning_rate, epochs, weight_decay, model_name):
        """
        Train DeepBLUP model end-to-end.

        Signature kept consistent with existing calls in train_models:
        _train_deepblup_model(self, model, train_loader, val_loader, learning_rate, epochs, weight_decay, model_name)

        Returns:
        trained model (with best validation state if val_loader provided)
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for _train_deepblup_model")

        set_random_seeds(self.config['seed'])
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)

        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # LR scheduler (reduce on plateau)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        # checkpoints directory
        ckpt_dir = os.path.join(self.config.get('results_dir', '.'), "ckpts")
        os.makedirs(ckpt_dir, exist_ok=True)
        tag = f"{self.config.get('feature_view', 'feat')}_{model_name}"
        ckpt_path = os.path.join(ckpt_dir, f"best_{tag}.pth")

        best_val = float('inf')
        patience = int(self.hyperparams.get(model_name, {}).get('patience', 30)) if hasattr(self, 'hyperparams') else 30
        patience_counter = 0

        for epoch in range(1, int(epochs) + 1):
            model.train()
            train_loss_accum = 0.0
            n_samples = 0

            for batch_idx, (batch_X, batch_y) in enumerate(train_loader):
                batch_X = batch_X.to(device).float()
                batch_y = batch_y.to(device).float()

                optimizer.zero_grad()
                outputs = model(batch_X)
                mse = criterion(outputs, batch_y)

                # add rrblup L2 penalty (if model implements it)
                reg_loss = 0.0
                if hasattr(model, 'get_rrblup_regularization'):
                    reg_loss = model.get_rrblup_regularization()
                    # ensure tensor on correct device
                    if isinstance(reg_loss, torch.Tensor):
                        reg_loss = reg_loss.to(device)
                loss = mse + 0.01 * reg_loss  # scale down rr penalty inside DL training (tunable)

                loss.backward()

                # debug gradients optionally
                # total_grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

                optimizer.step()

                batch_n = batch_X.size(0)
                train_loss_accum += mse.item() * batch_n
                n_samples += batch_n

            train_loss = train_loss_accum / max(1, n_samples)

            # Validation
            val_loss = None
            if val_loader is not None:
                model.eval()
                val_accum = 0.0
                n_val = 0
                with torch.no_grad():
                    for batch_X, batch_y in val_loader:
                        batch_X = batch_X.to(device).float()
                        batch_y = batch_y.to(device).float()
                        outputs = model(batch_X)
                        vloss = criterion(outputs, batch_y)
                        val_accum += vloss.item() * batch_X.size(0)
                        n_val += batch_X.size(0)
                val_loss = val_accum / max(1, n_val)

                # lr scheduler step
                scheduler.step(val_loss)

                # early stopping & checkpoint
                if val_loss < best_val - 1e-6:
                    best_val = val_loss
                    patience_counter = 0
                    torch.save(model.state_dict(), ckpt_path)
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    print(f"    [{model_name}] Early stopping at epoch {epoch}. Best val {best_val:.6f}")
                    break

            # occasional logging
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                if val_loss is None:
                    print(f"    [{model_name}] Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f}")
                else:
                    print(f"    [{model_name}] Epoch {epoch+1}/{epochs} - train_loss: {train_loss:.6f} - val_loss: {val_loss:.6f} - lr: {optimizer.param_groups[0]['lr']:.6e}")

        # Load best checkpoint if available
        if os.path.exists(ckpt_path):
            model.load_state_dict(torch.load(ckpt_path, map_location='cpu'))
            print(f"    [{model_name}] Loaded best model from {ckpt_path} with val_loss {best_val:.6f}")

        model.to('cpu')
        return model

    def _get_rrblup_weights_for_integrated_model(self, X, y, output_dim, rrblup_out_dim):
        """Get properly formatted RRBLUP weights using R_RRBLUP for initialization"""
        if y.ndim == 1 or y.shape[1] == 1:
            # Single trait case - use R_RRBLUP
            try:
                print("  Using R_RRBLUP for weight computation...")
                r_rrblup_model = R_RRBLUP()
                y_flat = y.ravel() if y.ndim > 1 else y
                r_rrblup_model.fit(X, y_flat)
                
                if hasattr(r_rrblup_model, 'marker_effects') and r_rrblup_model.marker_effects is not None:
                    weights_1d = r_rrblup_model.marker_effects
                    # Convert to 2D: (rrblup_out_dim, input_dim)
                    repeated_weights = np.tile(weights_1d.reshape(1, -1), (rrblup_out_dim, 1))
                    print(f"  R_RRBLUP weights computed successfully: {repeated_weights.shape}")
                    return repeated_weights
                else:
                    print("  Warning: R_RRBLUP failed to compute marker effects, falling back to Python RRBLUP")
                    # Fallback to Python RRBLUP
                    return self._get_rrblup_weights_python_fallback(X, y_flat, rrblup_out_dim)
                    
            except Exception as e:
                print(f"  Warning: R_RRBLUP failed: {e}, falling back to Python RRBLUP")
                return self._get_rrblup_weights_python_fallback(X, y.ravel(), rrblup_out_dim)
        else:
            # Multi-trait case - use R_RRBLUP for each trait
            rrblup_weights_list = []
            n_traits = min(3, y.shape[1])  # Use first few traits
            
            for i in range(n_traits):
                try:
                    print(f"  Computing R_RRBLUP weights for trait {i+1}/{n_traits}...")
                    r_rrblup_model = R_RRBLUP()
                    y_single = y[:, i].ravel()
                    r_rrblup_model.fit(X, y_single)
                    
                    if hasattr(r_rrblup_model, 'marker_effects') and r_rrblup_model.marker_effects is not None:
                        weights = r_rrblup_model.marker_effects
                        rrblup_weights_list.append(weights)
                        print(f"    Trait {i+1}: Success")
                    else:
                        print(f"    Trait {i+1}: Failed to get marker effects")
                        # Fallback to zeros
                        rrblup_weights_list.append(np.zeros(X.shape[1]))
                        
                except Exception as e:
                    print(f"    Trait {i+1}: R_RRBLUP failed: {e}")
                    # Fallback to zeros
                    rrblup_weights_list.append(np.zeros(X.shape[1]))
            
            if rrblup_weights_list:
                # Stack weights: (n_traits, input_dim)
                stacked_weights = np.stack(rrblup_weights_list, axis=0)
                
                # If RRBLUP output dimension is larger than number of traits,
                # repeat the weights to match the required dimension
                if rrblup_out_dim > len(rrblup_weights_list):
                    repeated_weights = []
                    for i in range(rrblup_out_dim):
                        trait_idx = i % len(rrblup_weights_list)
                        repeated_weights.append(rrblup_weights_list[trait_idx])
                    stacked_weights = np.stack(repeated_weights, axis=0)
                
                print(f"  R_RRBLUP multi-trait weights computed: {stacked_weights.shape}")
                return stacked_weights
            
            # Final fallback
            return self._get_rrblup_weights_python_fallback(X, y[:, 0].ravel(), rrblup_out_dim)

    def _get_rrblup_weights_python_fallback(self, X, y, rrblup_out_dim):
        """Fallback to Python RRBLUP if R_RRBLUP fails"""
        try:
            print("  Using Python RRBLUP fallback...")
            python_rrblup = HighPerformanceRRBLUP()
            python_rrblup.fit(X, y)
            weights_1d = python_rrblup.get_marker_effects()
            
            if weights_1d is not None:
                repeated_weights = np.tile(weights_1d.reshape(1, -1), (rrblup_out_dim, 1))
                return repeated_weights
        except Exception as e:
            print(f"  Python RRBLUP fallback also failed: {e}")
        
        # Last resort: random initialization
        print("  Using random initialization as last resort")
        return None

# -----------------------------
# Cross-Validation Workflow
# -----------------------------
def run_cross_validation(config):
    """Run cross-validation for all classical models"""
    
    # Initialize results storage
    results = defaultdict(list)

    # Run multiple marker views
    for i, marker_view in enumerate(config['feature_views']):
        # assign the current marker view to config parameter object
        config['feature_view'] = marker_view

        print('=' * 50)
        print(f"Running cross-validation for marker type: {marker_view}")
        print('=' * 50)

        # Load and prepare data
        data, used_cols, label_cols, pheno_scaler = load_data(config)
        
        import gc
        
        # Phenotype analysis
        if i == 0:
            try:
                generate_phenotype_analysis(data[label_cols], os.path.join(config['results_dir'], 'phenotype_analysis'), prefix='train')
            except Exception as e:
                print(f"[WARN] Phenotype pre-analysis (CV) failed: {e}")
        
        X = data[used_cols].values              # Markers: SNP, HAP or PC view
        Y = data[label_cols].values             # Phenotypes
            
        # ---------------- Stats for Prediction mode ----------------
        samples_n = len(X)
        markers_n = len(used_cols)
        traits_n  = len(label_cols)
        
        print("")
        print(f"[DATA] Total samples: {samples_n} ")
        print(f"[DATA] Feature view: {config['feature_view']} ")
        print(f"[DATA] No. markers: {markers_n}")
        print(f"[DATA] Traits: {label_cols}")
        print(f"[DATA] No. traits: {traits_n}")
        print("")
    
        # Run CV
        for replicate in range(config['n_replicates']):

            kf = KFold(n_splits=config['n_folds'], shuffle=True, 
                    random_state=config['seed'] + replicate)
            
            for fold, (train_idx, test_idx) in enumerate(kf.split(data)):
                # Start timing
                start_time = time.time()
                
                print(f"\n=== [{marker_view}] Replicate {replicate+1}/{config['n_replicates']}, Fold {fold+1}/{config['n_folds']} ===")
                
                # Split data
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = Y[train_idx], Y[test_idx]
                
                # Train all models once for this fold
                gs_models = AllGSModels(config)
                
                all_models = [m for m in config['enabled_models']]
                print("Training models: " + ", ".join(all_models))
                
                # Train models
                gs_models.train_models(X_train, y_train)
                
                # Evaluate each model
                print("")   # insert an empty line before output predciton values
                for model_name in config['enabled_models']:
                    #print(f"Predicting with {model_name}...")
                    try:
                        # Use the pre-trained models
                        y_pred = gs_models.predict(X_test, model_name)
                        #print(f"[DEBUG] {model_name} prediction shape: {y_pred.shape}")
                        #print(f"[DEBUG] {model_name} prediction range: [{y_pred.min():.3f}, {y_pred.max():.3f}]")
                        
                        # Calculate metrics for each trait
                        for j, trait in enumerate(label_cols):
                            y_true = y_test[:, j]
                            y_pred_trait = y_pred[:, j] if y_pred.ndim > 1 else y_pred
                            
                            mse = mean_squared_error(y_true, y_pred_trait)
                            rmse = np.sqrt(mse)
                            try:
                                r = pearsonr(y_true, y_pred_trait)[0]
                            except:
                                r = np.nan
                            
                            results[trait].append({
                                'replicate': replicate,
                                'fold': fold,
                                'marker_view': marker_view,
                                'model': model_name,
                                'MSE': mse,
                                'RMSE': rmse,
                                'PearsonR': r,
                                'n_samples': len(y_true)
                            })
                            
                            print(f"{trait} [{marker_view}][{model_name}]: MSE={mse:.4f}, PearsonR={r:.3f}")
                            
                    except Exception as e:
                        print(f"Error with {model_name}: {e}")
                        import traceback
                        traceback.print_exc()  # This gives detailed traceback
                        # Continue with next model instead of crashing entire pipeline
                        continue
                        
                        for j, trait in enumerate(label_cols):
                            results[trait].append({
                                'replicate': replicate,
                                'fold': fold,
                                'marker_view': marker_view,
                                'model': model_name,
                                'MSE': np.nan,
                                'RMSE': np.nan,
                                'PearsonR': np.nan,
                                'n_samples': len(test_idx)
                            })
                    gc.collect()    # gabage collection after each replicate, each fold

                # calculate used time for each replicate
                end_time = time.time()
                total_seconds = end_time - start_time
                total_minutes = total_seconds / 60
                    
                print('-' * 76)
                print(f"Execution time for Replicate {replicate+1}/{config['n_replicates']}, Fold {fold+1}/{config['n_folds']}: {total_minutes:.2f} minutes ({total_seconds:.2f} seconds)")
                print('-' * 76)

        # Generate visualizations
        generate_plots_by_feature_view(results, config['results_dir'], marker_view)

    return results, label_cols

# -----------------------------
# Prediction Workflow
# -----------------------------
def run_prediction_mode(config):
    """Run prediction on test data for all GS models"""

    # Training phenotype processing
    y_train = pd.read_csv(config['phenotype_path'], sep=None, engine='python', index_col=0)
    if y_train.isna().any().any():
        print("[INFO] Filled missing phenotype values with per-trait medians.")
        y_train = y_train.fillna(y_train.median())

    # --- HARD ALIGNMENT & DEDUP GUARDRAILS (Prediction mode) ---
    # Assume X_train is a DataFrame indexed by sample_id
    # and y_train is your phenotype DataFrame with one or more traits

    # 1) Drop/aggregate duplicate phenotype rows by sample_id
    if y_train.index.has_duplicates:
        dup_n = y_train.index.duplicated(keep=False).sum()
        print(f"[WARN] Phenotype table has {dup_n} duplicate sample IDs; aggregating by mean.")
        y_train = y_train.groupby(level=0).mean()


    # Phenotype analysis for training data
    try:
        generate_phenotype_analysis(y_train, os.path.join(config['results_dir'], 'phenotype_analysis'), prefix='train')
    except Exception as e:
        print(f"[WARN] Phenotype pre-analysis (train) failed: {e}")

    # set intermediate data foder for some intermediate files
    intermediate_data_dir = os.path.join(config['results_dir'], 'intermediate_data')
    os.makedirs(intermediate_data_dir, exist_ok=True)    

    # If test phenotypes are available, evaluate the predicted results
    # evaluation stores prediction results
    evaluation = {}
    marker_view_predictions = {}   
    test_phenotypes = None
    
    # Sart running different marker views
    for i, marker_view in enumerate(config['feature_views']):
        # assign the current marker view to config parameter object
        config['feature_view'] = marker_view

        print('=' * 50)
        print(f"Running across-population prediction (APP) for marker type: {marker_view}")
        print('=' * 50)


        impute_scope = str(config.get('impute_scope', 'train')).lower()
        pca_fit_scope = str(config.get('pca_fit_scope', 'train')).lower()
        if config['feature_view'].lower() == 'pc':
            print(f"Impute scope: {impute_scope}, PCA fit scope: {pca_fit_scope}")
        else :
            print(f"Impute scope: {impute_scope}")

        # Genotypes + harmonization + scoped transforms
        # train_geno and test_geno for feature view SNP, HAP and PCA depednign on setting
        # Xtr_raw, Xte_raw for SNPs

        train_geno, test_geno, pca, Xtr_raw, Xte_raw, keep_mask, flip_mask = load_and_merge_genotypes_scoped(
            config['vcf_path'],
            config['test_vcf_path'],
            #config['results_dir'],
            intermediate_data_dir,
            config['rtm-gwas-snpldb_path'],
            config['n_pca_components'],
            config['genotype_normalization'],
            config['pca_variance_explained'],
            config['feature_view'],
            config['threads'],
            impute_scope=impute_scope,
            pca_fit_scope=pca_fit_scope
        )
        
        original_test_ids = test_geno.index.copy()
        
        # GRM-based MDS plot
        # only draw this plot for the first marker veiew
        if i == 0:
            try:
                out_png = os.path.join(config['results_dir'], 'mds_grm_scatter_plot.png')
                out_mds_values = os.path.join(config['results_dir'], 'mds_grm_scatter_plot_values.csv')
                
                plot_mds_from_grm(train_geno, test_geno, out_png, out_mds_values, title='MDS (GRM-based)')
                print(f"Saved MDS plot to {out_png}")
            except Exception as e:
                print(f"[WARN] MDS plot failed: {e}")

        def _norm_idx(idx):
            idx = idx.astype(str)
            return idx.str.upper().str.strip()

        train_geno.index = _norm_idx(train_geno.index)
        test_geno.index  = _norm_idx(test_geno.index)
        y_train.index    = _norm_idx(y_train.index)

        # Map PCA dataframe order to raw rows
        train_all_names = train_geno.index.tolist()
        name2row = {name: i for i, name in enumerate(train_all_names)}

        # Overlap with phenotypes
        common = train_geno.index.intersection(y_train.index)
        if len(common) == 0:
            raise ValueError('No matching samples between training genotypes and phenotypes')

        train_geno = train_geno.loc[common]
        y_train    = y_train.loc[common]


        # Align raw training genotypes to the subset order
        row_idx = np.array([name2row[n] for n in train_geno.index], dtype=int)
        Xtr_raw_aligned = Xtr_raw[row_idx, :]

        # find common rows for haplotypes
        # To be finished

        # Phenotype scaling
        if config['pheno_normalization'] != 'none':
            y_train_scaled, pheno_scaler = normalize_data(y_train, method=config['pheno_normalization'])
        else:
            y_train_scaled = y_train.copy()
            pheno_scaler = None

        label_cols = y_train.columns.tolist()
        
        # ---------------- Stats for Prediction mode ----------------
        # train_geno, test_geno are DataFrames with markers as columns, samples as rows
        train_n = len(train_geno)
        test_n  = len(test_geno)

        train_cols = train_geno.columns.astype(str)
        test_cols  = test_geno.columns.astype(str)

        markers_n = len(train_cols.intersection(test_cols))
        traits_n = len(y_train.columns.astype(str))

        print(f"[DATA] Train samples: {train_n} | Test samples: {test_n}")
        print(f"[DATA] Feature view: {config['feature_view']} ")
        print(f"[DATA] Markers → common: {markers_n}")
        print(f"[DATA] Traits →: {traits_n}")
        print ("")

        # Train models and predict
        X_train = train_geno.values
        X_test = test_geno.values
        y_train_values = y_train_scaled.values if config['pheno_normalization'] != 'none' else y_train.values
        
        print("Training all models...")
        gs_models = AllGSModels(config)

        gs_models.train_models(X_train, y_train_values)
        
        all_predictions = {}
        all_train_predictions = {}
        
        # Create trait predictions directory
        trait_pred_dir = os.path.join(config['results_dir'], 'trait_predictions')
        os.makedirs(trait_pred_dir, exist_ok=True)    
        
        # Predict with each model
        for model_name in config['enabled_models']:
            #print(f"Predicting with {model_name}...")
            try:
                y_pred = gs_models.predict(X_test, model_name)
                
                # Inverse transform if needed
                if config['pheno_normalization'] != 'none':
                    y_pred = inverse_normalize(pd.DataFrame(y_pred, columns=label_cols), pheno_scaler, 
                                            columns=label_cols).values
                    #print(f"[DEBUG] After inverse transform shape: {y_pred.shape}")

                # Store predictions
                pred_df = pd.DataFrame(y_pred, columns=label_cols, index=original_test_ids)
                all_predictions[model_name] = pred_df
                #print(f"[DEBUG] {model_name} prediction DataFrame shape: {pred_df.shape}")

                # Initialize the nested structure for marker_view_predictions
                if model_name not in marker_view_predictions:
                    marker_view_predictions[model_name] = {}  # Initialize as dict
                marker_view_predictions[model_name][marker_view] = pred_df
                
                # Save to file in results/traits_predictions directory
                out_file = os.path.join(trait_pred_dir, f"prediction_{config['feature_view']}_{model_name}.csv")
                pred_df.to_csv(out_file, index=True)
                print(f"Saved {model_name} predictions to {out_file}")
                
            except Exception as e:
                print(f"Error with {model_name}: {e}")
        
        
            # ---- ALSO get TRAIN predictions per model for plotting ----
            try:
                # these three models can't get predicted values for train
                #if model_name == 'EnsembleGS' or  model_name == 'CNN' or model_name == 'HybridAttnMLP':
                if model_name == 'EnsembleGS':
                    continue
                
                model_obj = gs_models.models.get(model_name, None)
                if model_obj is None:
                    #print(f"[DEBUG] model_obj is None for {model_name}")
                    continue
                
                #print(f"[DEBUG] Processing {model_name}, type: {type(model_obj)}, TORCH_AVAILABLE: {TORCH_AVAILABLE}")

                # Handle GraphConvGS specially - it's stored as a dictionary
                #if model_name == 'GraphConvGS' and TORCH_AVAILABLE:
                if model_name in ['GraphConvGS', 'GraphAttnGS', 'GraphSAGEGS', 'GraphFormer'] and TORCH_AVAILABLE:

                    if 'model' not in model_obj:
                        continue
                    
                    model = model_obj['model']
                    edge_index = model_obj['edge_index']
                    edge_weight = model_obj.get('edge_weight', None)
            
                    # Create dataset for training predictions
                    train_ds = SingleGraphDataset(X_train, np.zeros_like(y_train_values), edge_index, edge_weight)
                    train_loader = PyGDataLoader(train_ds, batch_size=1, shuffle=False)
            
                    model.eval()
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model = model.to(device)
            
                    with torch.no_grad():
                        X_train_tensor = torch.FloatTensor(X_train).to(device)
                        # For GraphConvGS, we need to use the data loader
                        y_pred_train_list = []
                        for batch in train_loader:
                            batch = batch.to(device)
                            preds = model(batch)
                            y_pred_train_list.append(preds.cpu().numpy())
                
                        y_pred_train = np.concatenate(y_pred_train_list, axis=0)

                # Handle PyTorch models (CNN, HybridAttnMLP, DNNGS, AttnCNNGS, Transformer)
                elif model_name in ['CNN', 'MLPGS', 'HybridAttnMLP',
                                    'DNNGS', 'Transformer',
                                    'FusionGS', 'EfficientGSFormer'] and TORCH_AVAILABLE:
                    #print(f"[DEBUG] Entering PyTorch handling for {model_name}")
                    model_obj.eval()
                    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                    model_obj = model_obj.to(device)
            
                    with torch.no_grad():
                        X_train_tensor = torch.FloatTensor(X_train).to(device)
                        y_pred_train = model_obj(X_train_tensor).cpu().numpy()
                    #print(f"[DEBUG] Successfully got PyTorch predictions for {model_name}")

                # Handle per-trait model lists (BRR, LightGBM, etc.)
                elif isinstance(model_obj, list):
                    preds_list = []
                    for i, m in enumerate(model_obj):
                        try:
                            p = m.predict(X_train)
                            if p is None:
                                continue
                            preds_list.append(p.reshape(-1, 1))
                        except Exception as e:
                            print(f"[WARNING] Model {i} in {model_name} failed during predict: {e}")
                            continue

                    if len(preds_list) == 0:
                        raise RuntimeError(f"[ERROR] No predictions collected for model list '{model_name}'. "
                                        f"The model list is empty or all models failed.")

                    y_pred_train = np.hstack(preds_list)

                # Handle scikit-learn style models
                else:
                    #print(f"[DEBUG] Entering scikit-learn handling for {model_name}")
                    y_pred_train = model_obj.predict(X_train)

                # Inverse transform if needed (mirror test handling)
                if config['pheno_normalization'] != 'none':
                    y_pred_train = inverse_normalize(pd.DataFrame(y_pred_train, columns=label_cols), pheno_scaler,
                                            columns=label_cols).values

                # Make a DataFrame with TRAIN sample index
                train_pred_df = pd.DataFrame(y_pred_train, columns=label_cols, index=train_geno.index)
                all_train_predictions[model_name] = train_pred_df
                #print(f"[DEBUG] Successfully stored train predictions for {model_name}")
        
            except Exception as e:
                print(f"[WARN] Could not compute train predictions for {model_name}: {e}")
                import traceback
                traceback.print_exc()  # This will show the exact line where it fails
        
        # If test phenotypes are available, evaluate
        #evaluation = {}
        
        test_pheno_path = config.get('test_phenotype_path')
        if test_pheno_path and os.path.exists(test_pheno_path):
            y_test = pd.read_csv(test_pheno_path, sep=None, engine='python', index_col=0)

            if y_test.isna().any().any():
                y_test = y_test.fillna(y_test.median())
            y_test.index = y_test.index.astype(str).str.strip() # just in case genotype names are numbers only, we need to convert numbers to string first.

            
            # Phenotype analysis for test data
            if i == 0:
                try:
                    generate_phenotype_analysis(y_test, os.path.join(config['results_dir'], 'phenotype_analysis'), prefix='test')
                except Exception as e:
                    print(f"[WARN] Phenotype pre-analysis (test) failed: {e}")
            
            print ("\nPrediction results:")
            for model_name, pred_df in all_predictions.items():
                common_idx = pred_df.index.intersection(y_test.index)

                #print(f"[DEBUG] {model_name} - common_idx length: {len(common_idx)}")
                #print(f"[DEBUG] {model_name} - pred_df shape: {pred_df.shape}")
                #print(f"[DEBUG] {model_name} - y_test shape: {y_test.shape}")

                if len(common_idx) > 0:
                    model_eval = {}
                    for trait in label_cols:
                        if trait in y_test.columns:
                            try:
                                # Extract the values properly
                                y_true_vals = y_test.loc[common_idx, trait].values
                                y_pred_vals = pred_df.loc[common_idx, trait].values
                            
                                #print(f"[DEBUG] {model_name} {trait} - y_true_vals shape: {y_true_vals.shape}")
                                #print(f"[DEBUG] {model_name} {trait} - y_pred_vals shape: {y_pred_vals.shape}")
                                r = pearsonr(
                                    y_test.loc[common_idx, trait].values,
                                    pred_df.loc[common_idx, trait].values
                                )[0]
                                model_eval[trait] = r
                                print(f"{trait} [{marker_view}][{model_name}]: PearsonR = {r:.3f}")
                                
                                # Ensure we have valid data for correlation
                                if len(y_true_vals) > 1 and len(y_pred_vals) > 1:
                                    r = pearsonr(y_true_vals, y_pred_vals)[0]
                                    model_eval[trait] = r
                                    #print(f"{trait} [{marker_view}][{model_name}]: PearsonR = {r:.3f}")
                                else:
                                    print(f"{trait} [{marker_view}][{model_name}]: Not enough common samples for correlation")
                                    model_eval[trait] = np.nan
                            
                                # Create individual scatter plots
                                plot_true_vs_predicted(
                                    y_test.loc[common_idx, trait].values,
                                    pred_df.loc[common_idx, trait].values,
                                    os.path.join(config['results_dir'], 'scatter_plots'),
                                    trait, model_name, config['feature_view']
                                )


                                

                                # Temporiarily remove this section as some of modles can't generate 
                                #    predicted values for the training set.
                                # Create Train+Test combined scatter with separate OLS lines + CI
                                # y_train[trait] is the TRAIN truth (already aligned earlier),
                                # train_pred_df is the TRAIN predictions we just computed.
                                #train_pred_df = all_train_predictions.get(model_name, None)
                                #if train_pred_df is not None and trait in train_pred_df.columns:
                                #    out_combined = os.path.join(
                                #        config['results_dir'], 'scatter_plots',
                                #        f"true_pred_TRAIN_TEST_{trait}_{model_name}.png"
                                #    )
                                #    try:
                                #        plot_true_vs_predicted_train_test(
                                #            y_train[trait],                        # train true (Series with index)
                                #            train_pred_df[trait],                  # train pred (Series with index)
                                #            y_test.loc[common_idx, trait],         # test true (Series aligned to common_idx)
                                #            pred_df.loc[common_idx, trait],        # test pred (Series aligned to common_idx)
                                #            output_path=out_combined,
                                #            trait_name=trait,
                                #            model_name=model_name,
                                #            ci_alpha=0.05
                                #        )
                                #    except Exception as e:
                                #        print(f"[WARN] combined train/test plot failed for {model_name} {trait}: {e}")

                            except:
                                model_eval[trait] = np.nan
                    evaluation[model_name] = model_eval
                else:
                    print(f"Warning: No common samples between predictions and test data for {marker_view} - {model_name}")

            # Create combined scatter plots for each trait
            for trait in label_cols:
                if trait in y_test.columns:
                    preds_dict = {}
                    for model_name, pred_df in all_predictions.items():
                        if trait in pred_df.columns:
                            preds_dict[model_name] = pred_df.loc[common_idx, trait].values
                    
                    if preds_dict:
                        plot_true_vs_predicted_multi(
                            y_test.loc[common_idx, trait].values,
                            preds_dict,
                            os.path.join(config['results_dir'], 'scatter_plots'),
                            trait,
                            config['feature_view']
                        )
        
            # After all predictions are made, export trait-wise results
            print("\nExporting trait-wise predictions...")
        
            # Load test phenotypes if available
            #test_phenotypes = None
            #if i == 0:
            test_pheno_path = config.get('test_phenotype_path')
            if test_pheno_path and os.path.exists(test_pheno_path):
                test_phenotypes = pd.read_csv(test_pheno_path, sep=None, engine='python', index_col=0)
                if test_phenotypes.isna().any().any():
                    test_phenotypes = test_phenotypes.fillna(test_phenotypes.median())
                #test_phenotypes.index = test_phenotypes.index.str.upper().str.strip()
    
            export_trait_predictions(config, all_predictions, test_phenotypes, trait_pred_dir)
            print(f"Trait-wise prediction export for {marker_view} completed!\n")
    
    # After runs for all marker views ar done 
    # Export trait predictions for all marker views together into one file
    # Debug: Check the predictions dictionary
    #print(f"[DEBUG] Before export - marker_view_predictions type: {type(marker_view_predictions)}")
    #print(f"[DEBUG] Before export - marker_view_predictions keys: {list(marker_view_predictions.keys())}")

    if marker_view_predictions:
        for model_name in marker_view_predictions:
            #print(f"[DEBUG] Model '{model_name}' has marker views: {list(marker_view_predictions[model_name].keys())}")
            for marker_view in marker_view_predictions[model_name]:
                pred_df = marker_view_predictions[model_name][marker_view]
                #print(f"[DEBUG]   {marker_view}: shape {pred_df.shape}, columns: {pred_df.columns.tolist()}")
    #else:
        #print("[DEBUG] marker_view_predictions is EMPTY!")


    export_trait_predictions_all_marker_views(config, marker_view_predictions, test_phenotypes, trait_pred_dir)

    return marker_view_predictions, evaluation

# -----------------------------
# Results Processing
# -----------------------------

# Generate summary for all data (all marker types  and models)
def generate_statistics_report(results, label_cols):
    report = {}
    for trait in label_cols:
        report[trait] = {}
        trait_records = results[trait]
        
        # Get all unique model and marker_view combinations for this trait
        combinations = set((r['model'], r['marker_view']) for r in trait_records)
        
        for model, marker_view in combinations:
            # Filter records for this trait, model, and marker_view combination
            combo_records = [r for r in trait_records if r['model'] == model and r['marker_view'] == marker_view]
            
            if combo_records:
                # Create a nested structure: report[trait][model][marker_view]
                if model not in report[trait]:
                    report[trait][model] = {}
                
                report[trait][model][marker_view] = {
                    'MSE': calculate_statistics([r['MSE'] for r in combo_records]),
                    'RMSE': calculate_statistics([r['RMSE'] for r in combo_records]),
                    'PearsonR': calculate_statistics([r['PearsonR'] for r in combo_records]),
                    'n_samples': calculate_statistics([r['n_samples'] for r in combo_records])
                }
    
    # Print summary results
    print("\n=== FINAL RESULTS ===")
    for trait in label_cols:
        print(f"\n{trait}:")
        
        # Get all unique marker_view and model combinations for this trait
        combinations = set((r['marker_view'], r['model']) for r in results[trait])
        
        # Sort for consistent output (optional)
        combinations = sorted(combinations)
        
        for marker_view, model in combinations:
            # Filter results for this specific marker_view and model combination
            model_results = [r for r in results[trait] if r['model'] == model and r['marker_view'] == marker_view]
            
            if model_results:
                pearson_vals = [r['PearsonR'] for r in model_results if not np.isnan(r['PearsonR'])]
                if pearson_vals:
                    mean_r = np.mean(pearson_vals)
                    std_r = np.std(pearson_vals)
                    print(f"  {model} [{marker_view}]: PearsonR = {mean_r:.3f} ± {std_r:.3f}")

    
    return report

def calculate_statistics(metric_values):
    vals = np.array(metric_values, dtype=float)
    
    # Check if this is n_samples (should be integer)
    is_count_metric = np.all(vals == vals.astype(int)) if len(vals) > 0 else False
    
    stats = {
        'mean': float(np.nanmean(vals)),
        'std': float(np.nanstd(vals)),
        'min': float(np.nanmin(vals)),
        '25%': float(np.nanpercentile(vals, 25)),
        'median': float(np.nanmedian(vals)),
        '75%': float(np.nanpercentile(vals, 75)),
        'max': float(np.nanmax(vals)),
        'n': int(np.sum(~np.isnan(vals)))
    }
    
    return stats

def save_results(results, label_cols, statistics_report, config):
    out_dir = config['results_dir']
    os.makedirs(out_dir, exist_ok=True)
    
    # Save detailed results
    detailed = []
    for trait in label_cols:
        detailed.extend([{**rec, 'trait': trait} for rec in results[trait]])
    detailed_df = pd.DataFrame(detailed)
    detailed_fn = os.path.join(out_dir,  'cv_gs_detailed_results.csv')
    detailed_df.to_csv(detailed_fn, index=False)
    
    # Reformat summary statistics to include both trait and model
    summary_rows = []
    for trait in label_cols:
        for model, metrics_dict in statistics_report[trait].items():
            for metric, stats_dict in metrics_dict.items():
                for stat_name, stat_value in stats_dict.items():
                    if stat_name != 'n':  # Exclude 'n' from the table
                        # Format values based on metric type
                        if metric in ['MSE', 'RMSE', 'PearsonR']:
                            # Format to 4 decimal places
                            formatted_value = f"{stat_value:.4f}"
                        elif metric == 'n_samples':
                            # Format as integer
                            formatted_value = f"{int(stat_value)}"
                        else:
                            formatted_value = stat_value
                            
                        summary_rows.append({
                            'Trait': trait,
                            'Model': model,
                            'Metric': metric,
                            'Statistic': stat_name,
                            'Value': formatted_value
                        })
    
    summary_df = pd.DataFrame(summary_rows)
    summary_fn = os.path.join(out_dir,  'cv_gs_summary_stats_fm1.csv')
    summary_df.to_csv(summary_fn, index=False)
    
    # Save JSON report
    #json_fn = os.path.join(out_dir, "".join([config['feature_view'],'_gs_results.json']))
    json_fn = os.path.join(out_dir, 'cv_gs_results.json')
    with open(json_fn, 'w') as f:
        json.dump(statistics_report, f, indent=2)
    
    return detailed_fn, summary_fn, json_fn

# Generate summary for all data (by marker type and model)
def generate_summary_table(results, label_cols):
    """Generate a flattened summary table with mean and std"""
    summary_data = []
    
    for trait in label_cols:
        trait_records = results[trait]
        
        # Get all unique model and marker_view combinations for this trait
        combinations = set((r['model'], r['marker_view']) for r in trait_records)
        
        for model, marker_view in combinations:
            # Filter records for this trait, model, and marker_view combination
            combo_records = [r for r in trait_records if r['model'] == model and r['marker_view'] == marker_view]
            
            if combo_records:
                # Calculate PearsonR statistics
                pearson_vals = [r['PearsonR'] for r in combo_records if not np.isnan(r['PearsonR'])]
                
                if pearson_vals:
                    summary_data.append({
                        'Trait': trait,
                        'Model': model,
                        'Marker_view': marker_view,
                        'PearsonR_mean': np.mean(pearson_vals),
                        'PearsonR_std': np.std(pearson_vals),
                        'n_observations': len(pearson_vals)
                    })
    
    return pd.DataFrame(summary_data)

def safe_pearsonr(x, y):
    """
    Calculate Pearson correlation with handling for constant arrays
    Returns nan if correlation is undefined
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    # Remove NaN values
    mask = np.isfinite(x) & np.isfinite(y)
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return np.nan
    
    # Check if either array is constant
    if np.std(x) < 1e-10 or np.std(y) < 1e-10:
        return np.nan
    
    try:
        return pearsonr(x, y)[0]
    except:
        return np.nan

# -----------------------------
# PearsonR ANOVA
# -----------------------------

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for servers
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd


def _ensure_outdir(out_dir: str):
    os.makedirs(out_dir, exist_ok=True)


def _simple_boxplot(df, x_col, y_col, hue_col=None, out_path="plot.png", title=None):
    # Use matplotlib-only plotting (no seaborn), single axes, no custom colors.
    # If hue_col is provided, we plot grouped boxes side by side manually.
    plt.figure(figsize=(10, 6))
    ax = plt.gca()

    if hue_col is None:
        groups = [df[df[x_col] == k][y_col].dropna().values for k in df[x_col].unique()]
        #ax.boxplot(groups, labels=list(df[x_col].unique()))
        ax.boxplot(groups, tick_labels=list(df[x_col].unique()))
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
    else:
        x_levels = list(df[x_col].unique())
        hue_levels = list(df[hue_col].unique())
        # compute positions for grouped boxplots
        n_hue = len(hue_levels)
        width = 0.8 / max(1, n_hue)
        positions = []
        tick_positions = []

        # Collect data and positions
        pos_counter = 1
        for xi, xval in enumerate(x_levels, start=1):
            tick_positions.append(xi)
            for hj, hval in enumerate(hue_levels):
                subset = df[(df[x_col] == xval) & (df[hue_col] == hval)][y_col].dropna().values
                positions.append((subset, xi - 0.4 + hj * width + width / 2.0))

        # Draw boxplots one-by-one to control positions
        for (subset, pos) in positions:
            if len(subset) == 0:
                continue
            ax.boxplot([subset], positions=[pos], widths=width)

        # X ticks
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(x_levels)
        ax.set_xlabel(x_col)
        ax.set_ylabel(y_col)
        # Simple legend: draw proxy lines with labels
        # Since we didn't draw colored boxes, we will annotate legend text only.
        legend_text = " / ".join([str(h) for h in hue_levels])
        ax.text(0.99, 0.95, f"{hue_col}: {legend_text}", transform=ax.transAxes,
                ha="right", va="top")

    if title:
        ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def _tukey_to_dataframe(tukey_obj) -> pd.DataFrame:
    # Convert Tukey result to DataFrame
    # tukey_obj.summary() is a SimpleTable; we parse from tukey_obj._results_table.data
    table = tukey_obj._results_table.data[1:]  # skip header row
    cols = ["group1", "group2", "meandiff", "p-adj", "lower", "upper", "reject"]
    df = pd.DataFrame(table, columns=cols)
    # cast numeric cols
    for c in ["meandiff", "p-adj", "lower", "upper"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["reject"] = df["reject"].astype(str)
    return df


def run_anova(csv_path: str, out_dir: str = "results/anova", alpha: float = 0.05) -> dict:
    """
    Run ANOVA on PearsonR across models and/or traits.
    Treat fold & replicate as repeated observations (ignored as factors).

    Returns a dict with paths to generated artifacts.
    """
    _ensure_outdir(out_dir)
    artifacts = {}

    # Load data
    df = pd.read_csv(csv_path)
    needed_cols = ["model", "trait", "PearsonR"]
    missing = [c for c in needed_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns in CSV: {missing}")

    # Keep only needed columns
    df = df[needed_cols].copy()
    # Drop NaNs in PearsonR
    df = df.dropna(subset=["PearsonR"]).reset_index(drop=True)

    n_models = df["model"].nunique()
    n_traits = df["trait"].nunique()

    if n_models == 0 or n_traits == 0:
        raise ValueError("No models or traits found in the data.")

    # Fit appropriate ANOVA
    if n_models > 1 and n_traits > 1:
        formula = "PearsonR ~ C(model) * C(trait)"
    elif n_models > 1:
        formula = "PearsonR ~ C(model)"
    else:
        formula = "PearsonR ~ C(trait)"

    model = ols(formula, data=df).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)
    anova_csv = os.path.join(out_dir, "anova_table.csv")
    anova_table.to_csv(anova_csv)
    artifacts["anova_table"] = anova_csv

    # Group means table
    group_means = (
        df.groupby(["model", "trait"], dropna=False)["PearsonR"]
        .agg(["count", "mean", "std"])
        .reset_index()
        .rename(columns={"count": "n", "mean": "PearsonR_mean", "std": "PearsonR_sd"})
    )
    group_means_csv = os.path.join(out_dir, "group_means.csv")
    group_means.to_csv(group_means_csv, index=False)
    artifacts["group_means"] = group_means_csv

    # Post-hoc multiple comparisons (Tukey HSD)
    # We run on main factors if they have >1 level.
    if n_models > 1:
        tukey_model = pairwise_tukeyhsd(endog=df["PearsonR"], groups=df["model"], alpha=alpha)
        tukey_model_df = _tukey_to_dataframe(tukey_model)
        tukey_model_csv = os.path.join(out_dir, "tukey_model.csv")
        tukey_model_df.to_csv(tukey_model_csv, index=False)
        artifacts["tukey_model"] = tukey_model_csv

    if n_traits > 1:
        tukey_trait = pairwise_tukeyhsd(endog=df["PearsonR"], groups=df["trait"], alpha=alpha)
        tukey_trait_df = _tukey_to_dataframe(tukey_trait)
        tukey_trait_csv = os.path.join(out_dir, "tukey_trait.csv")
        tukey_trait_df.to_csv(tukey_trait_csv, index=False)
        artifacts["tukey_trait"] = tukey_trait_csv

    # If both >1, also compute Tukey on combinations (model_trait group) for detailed pairwise
    if n_models > 1 and n_traits > 1:
        combo = df.assign(model_trait=(df["model"].astype(str) + " | " + df["trait"].astype(str)))
        tukey_combo = pairwise_tukeyhsd(endog=combo["PearsonR"], groups=combo["model_trait"], alpha=alpha)
        tukey_combo_df = _tukey_to_dataframe(tukey_combo)
        tukey_combo_csv = os.path.join(out_dir, "tukey_model_trait_combo.csv")
        tukey_combo_df.to_csv(tukey_combo_csv, index=False)
        artifacts["tukey_model_trait_combo"] = tukey_combo_csv

    # Plots
    if n_traits > 1 and n_models > 1:
        out_plot = os.path.join(out_dir, "boxplot_model_by_trait.png")
        _simple_boxplot(df, x_col="model", y_col="PearsonR", hue_col="trait",
                        out_path=out_plot, title="PearsonR by Model × Trait")
        artifacts["plot_model_by_trait"] = out_plot
    elif n_traits > 1:
        out_plot = os.path.join(out_dir, "boxplot_trait.png")
        _simple_boxplot(df, x_col="trait", y_col="PearsonR", hue_col=None,
                        out_path=out_plot, title="PearsonR by Trait")
        artifacts["plot_trait"] = out_plot
    elif n_models > 1:
        out_plot = os.path.join(out_dir, "boxplot_model.png")
        _simple_boxplot(df, x_col="model", y_col="PearsonR", hue_col=None,
                        out_path=out_plot, title="PearsonR by Model")
        artifacts["plot_model"] = out_plot

    # README with quick interpretation tips
    readme_text = f"""
    ANOVA & Tukey HSD Results
    =========================

    Data source: {os.path.abspath(csv_path)}
    Output dir : {os.path.abspath(out_dir)}

    Factors used:
      - Models: {n_models} levels
      - Traits: {n_traits} levels
      - Design : {"Two-way with interaction (model × trait)" if (n_models > 1 and n_traits > 1) else ("One-way by model" if n_models > 1 else "One-way by trait")}

    Files:
      - anova_table.csv
      - group_means.csv
      - tukey_model.csv (if >1 model)
      - tukey_trait.csv (if >1 trait)
      - tukey_model_trait_combo.csv (if >1 model and >1 trait)
      - boxplot_*.png

    Notes:
      - All replicate × fold rows are treated as independent observations.
      - Check `anova_table.csv` for significance of factors and interaction.
      - Use Tukey CSVs to see which specific groups differ.
      - Boxplots illustrate the distribution of PearsonR across groups.

    """

    with open(os.path.join(out_dir, "README_ANOVA.txt"), "w", encoding="utf-8") as f:
        f.write(readme_text.strip())

    artifacts["readme"] = os.path.join(out_dir, "README_ANOVA.txt")
    return artifacts


# -----------------------------
# Helper functions to validate input
# -----------------------------

import os
import io
import csv
import gzip
import shutil
from typing import Dict, Optional, Tuple, List

def _resolve_path(p: Optional[str], base_dir: Optional[str]=None) -> Optional[str]:
    if not p:
        return None
    p = os.path.expanduser(p)
    if not os.path.isabs(p) and base_dir:
        p = os.path.join(base_dir, p)
    return os.path.abspath(p)

def _open_maybe_gzip(path: str, mode: str = "rt", encoding: str = "utf-8") -> io.TextIOBase:
    # Mode must be text-mode ("rt") for reading lines reliably
    if path.endswith(".gz"):
        return gzip.open(path, mode=mode, encoding=encoding, errors="replace")
    return open(path, mode=mode, encoding=encoding, errors="replace")

def _check_vcf_headers(vcf_path: str) -> Tuple[bool, List[str]]:
    """Return (ok, problems) for basic VCF header requirements."""
    problems = []
    saw_fileformat = False
    saw_columns = False
    try:
        with _open_maybe_gzip(vcf_path, "rt") as f:
            for raw in f:
                line = raw.rstrip("\n")
                if line.startswith("##fileformat=VCF"):
                    saw_fileformat = True
                if line.startswith("#CHROM"):
                    # Must at least contain up to INFO column (FORMAT/sample columns are allowed)
                    cols = line.lstrip("#").split("\t")
                    required_prefix = ["CHROM", "POS", "ID", "REF", "ALT", "QUAL", "FILTER", "INFO"]
                    if cols[:len(required_prefix)] == required_prefix:
                        saw_columns = True
                    else:
                        problems.append(
                            f"Table header columns malformed (got: {cols[:len(required_prefix)]}, "
                            f"expected prefix: {required_prefix})."
                        )
                    break  # After #CHROM we can stop
    except FileNotFoundError:
        return False, [f"File not found: {vcf_path}"]
    except OSError as e:
        return False, [f"Error reading VCF '{vcf_path}': {e}"]

    if not saw_fileformat:
        problems.append("Missing required line like '##fileformat=VCFv4.x' (required by rtm-gwas-snpldb).")
    if not saw_columns:
        problems.append("Missing/invalid table header line starting with '#CHROM'.")
    return (len(problems) == 0), problems

def _check_pheno_table(pheno_path: str) -> Tuple[bool, List[str]]:
    """Return (ok, problems) for phenotype file: header with ≥2 columns (ID + ≥1 trait)."""
    problems = []
    try:
        with open(pheno_path, "rt", encoding="utf-8", errors="replace") as f:
            # Sniff delimiter robustly (fallback to tab)
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
                delim = dialect.delimiter
            except csv.Error:
                # Default to tab if sniff fails
                delim = "\t"
            header = f.readline().strip()
            if not header:
                problems.append("Empty file or missing header line.")
                return False, problems
            cols = header.split(delim)
            if len(cols) < 2:
                problems.append(
                    f"Header must have ≥2 columns (first = Sample ID, then ≥1 trait). Found {len(cols)}."
                )
            if cols[0].strip() == "":
                problems.append("First header column (Sample ID) is empty.")
    except FileNotFoundError:
        return False, [f"File not found: {pheno_path}"]
    except OSError as e:
        return False, [f"Error reading phenotype file '{pheno_path}': {e}"]
    return (len(problems) == 0), problems

def _check_executable(tool_path_or_name: Optional[str]) -> Tuple[bool, List[str], Optional[str]]:
    """Return (ok, problems, resolved_path). Accepts absolute/relative path or a name in $PATH."""
    problems = []
    if not tool_path_or_name:
        return False, ["rtm-gwas-snpldb_path is missing in config."], None

    # If path points to a file, check executable bit; otherwise try which()
    p = os.path.expanduser(tool_path_or_name)
    if os.path.sep in p or p.startswith("."):
        abs_p = os.path.abspath(p)
        if not os.path.isfile(abs_p):
            problems.append(f"Executable not found at '{tool_path_or_name}'.")
            return False, problems, None
        if not os.access(abs_p, os.X_OK):
            problems.append(f"File exists but is not executable: '{abs_p}'.")
            return False, problems, None
        return True, [], abs_p
    else:
        found = shutil.which(tool_path_or_name)
        if not found:
            problems.append(
                f"Executable '{tool_path_or_name}' not found in PATH. "
                f"Provide a full path or add it to PATH."
            )
            return False, problems, None
        return True, [], found

def validate_pipeline_inputs(
    config: Dict,
    *,
    base_dir: Optional[str] = None,
    strict: bool = True,
) -> None:
    """
    Validate required inputs and environment for the GS pipeline.

    Parameters
    ----------
    config : dict
        Parsed config from parse_config(), expected keys:
          - 'vcf_path' (str), optional 'test_vcf_path' (str)
          - 'phenotype_path' (str), optional 'test_phenotype_path' (str)
          - 'rtm-gwas-snpldb_path' (str)
          - 'feature_view' (str): one of {'SNP','HAP','PC'} (case-insensitive)
    base_dir : str, optional
        If provided, resolve relative paths against this directory (e.g. config file dir).
    strict : bool
        If True, raise ValueError on any problem. If False, print warnings and continue.

    Raises
    ------
    ValueError
        If any validation fails (when strict=True).
    """
    errors: List[str] = []
    warnings: List[str] = []

    # ---- Feature views ----
    fv_list = config.get("feature_views") or []

    # Define valid options
    valid_views = {"SNP", "HAP", "PC"}

    # Check for empty input
    if not fv_list:
        errors.append("feature_views cannot be empty. Must be one or more of: SNP, HAP, PC.")
    else:
        # Convert to uppercase for validation
        fv_list_upper = [view.upper().strip() for view in fv_list]
        
        # Check for invalid entries
        invalid_views = [view for view in fv_list_upper if view not in valid_views]
        if invalid_views:
            errors.append(
                f"feature_views must contain only SNP, HAP, PC (case-insensitive). "
                f"Invalid entries: {invalid_views}. Got: {fv_list}."
            )
        
        # Check for duplicates (using original case for error message)
        if len(fv_list_upper) != len(set(fv_list_upper)):
            errors.append(
                f"feature_views contains duplicate entries. Got: {fv_list}."
            )
        
        # Update the config with uppercase versions
        config["feature_views"] = fv_list_upper

    # ---- External tool ----
    ok, probs, resolved_tool = _check_executable(config.get("rtm-gwas-snpldb_path"))
    if not ok:
        errors.extend([f"[rtm-gwas-snpldb] {p}" for p in probs])

    # ---- VCF(s) ----
    vcf_primary = _resolve_path(config.get("vcf_path"), base_dir)
    if not vcf_primary:
        errors.append("vcf_path is missing in config.")
    else:
        ok, probs = _check_vcf_headers(vcf_primary)
        if not ok:
            errors.extend([f"[vcf_path] {p}" for p in probs])

    vcf_test = _resolve_path(config.get("test_vcf_path"), base_dir)
    
    # Determine run mode based on test VCF presence
    run_mode = 'prediction' if vcf_test else 'cross_validation'
    config['run_mode'] = run_mode  # Set the run mode in config
    
    if vcf_test:
        ok, probs = _check_vcf_headers(vcf_test)
        if not ok:
            errors.extend([f"[test_vcf_path] {p}" for p in probs])
        else:
            print(f"[validate_pipeline_inputs] Run mode: PREDICTION (test VCF provided)")

    # ---- Phenotypes ----
    pheno_primary = _resolve_path(config.get("phenotype_path"), base_dir)
    if not pheno_primary:
        errors.append("phenotype_path is missing in config.")
    else:
        ok, probs = _check_pheno_table(pheno_primary)
        if not ok:
            errors.extend([f"[phenotype_path] {p}" for p in probs])

    pheno_test = _resolve_path(config.get("test_phenotype_path"), base_dir)
    
    # For prediction mode: test phenotype file is optional
    if run_mode == 'prediction' and pheno_test:
        ok, probs = _check_pheno_table(pheno_test)
        if not ok:
            warnings.extend([f"[test_phenotype_path] {p}" for p in probs])
        else:
            # Compare training and testing phenotype traits
            try:
                train_traits = _get_phenotype_traits(pheno_primary)
                test_traits = _get_phenotype_traits(pheno_test)
                
                # Check if number of traits matches
                if len(train_traits) != len(test_traits):
                    warnings.append(
                        f"Training and testing datasets have different number of traits: "
                        f"training has {len(train_traits)} traits, testing has {len(test_traits)} traits. "
                        f"Prediction evaluation will be skipped."
                    )
                
                # Check if trait names match
                mismatched_traits = []
                for i, (train_trait, test_trait) in enumerate(zip(train_traits, test_traits)):
                    if train_trait != test_trait:
                        mismatched_traits.append((i, train_trait, test_trait))
                
                if mismatched_traits:
                    mismatch_info = "; ".join([f"Trait {i}: '{train}' vs '{test}'" 
                                             for i, train, test in mismatched_traits])
                    warnings.append(
                        f"Trait names differ between training and testing datasets: {mismatch_info}. "
                        f"Prediction evaluation will be skipped."
                    )
                
                if len(train_traits) == len(test_traits) and not mismatched_traits:
                    print(f"[validate_pipeline_inputs] Training and testing traits match: {train_traits}")
                    
            except Exception as e:
                warnings.append(f"Could not compare training and testing traits: {e}")
    
    elif run_mode == 'prediction' and not pheno_test:
        print("[validate_pipeline_inputs] No test phenotype file provided - prediction will proceed without evaluation")
    
    # For cross-validation mode: test phenotype should not be provided
    elif run_mode == 'cross_validation' and pheno_test:
        warnings.append(
            "Test phenotype file provided but running in cross-validation mode. "
            "Test phenotypes will be ignored."
        )

    # ---- Finalize ----
    if errors and strict:
        # Aggregate into a readable report
        msg = ["Input/environment validation failed:"]
        msg += [f" - {e}" for e in errors]
        if warnings:
            msg.append("Warnings:")
            msg += [f" - {w}" for w in warnings]
        raise ValueError("\n".join(msg))
    else:
        # Optional lightweight reporting
        if warnings:
            print("[validate_pipeline_inputs] Warnings:")
            for w in warnings:
                print(" -", w)
        if not errors:
            # You may log the resolved path for the external tool
            if resolved_tool:
                print(f"[validate_pipeline_inputs] Found rtm-gwas-snpldb at: {resolved_tool}")
            print(f"[validate_pipeline_inputs] All checks passed. Run mode: {run_mode.upper()}")
        elif not strict:
            print("[validate_pipeline_inputs] Completed with issues:")
            for e in errors:
                print(" -", e)

def _get_phenotype_traits(pheno_path: str) -> List[str]:
    """
    Extract trait names from phenotype file header.
    
    Returns:
        List of trait names (column names excluding the first column which is sample ID)
    """
    try:
        with open(pheno_path, "rt", encoding="utf-8", errors="replace") as f:
            # Sniff delimiter
            sample = f.read(4096)
            f.seek(0)
            try:
                dialect = csv.Sniffer().sniff(sample, delimiters="\t,;")
                delim = dialect.delimiter
            except csv.Error:
                delim = "\t"
            
            header = f.readline().strip()
            if not header:
                raise ValueError("Empty header in phenotype file")
            
            cols = [col.strip() for col in header.split(delim)]
            if len(cols) < 2:
                raise ValueError(f"Phenotype file must have at least 2 columns (sample ID + traits). Found {len(cols)}")
            
            # Return all columns except the first one (sample ID)
            return cols[1:]
            
    except Exception as e:
        raise ValueError(f"Error reading phenotype traits from {pheno_path}: {e}")
    
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_performance_boxplots(results, label_cols):
    # Convert results to DataFrame
    data_list = []
    for trait in label_cols:
        for result in results[trait]:
            data_list.append({
                'Trait': trait,
                'Marker_View': result['marker_view'],
                'Model': result['model'],
                'PearsonR': result['PearsonR'],
                'MSE': result['MSE'],
                'RMSE': result['RMSE']
            })
    
    df = pd.DataFrame(data_list)
    
    # Create faceted boxplots for PearsonR
    plt.figure(figsize=(15, 10))
    g = sns.catplot(
        data=df, 
        x='Model', 
        y='PearsonR', 
        hue='Marker_View',
        col='Trait',
        kind='box',
        height=6, 
        aspect=1.2,
        palette='Set2'
    )
    g.set_xticklabels(rotation=45)
    g.fig.suptitle('Model Performance by Trait and Marker View (Pearson R)', y=1.02)
    plt.tight_layout()
    plt.savefig('performance_boxplots.png', dpi=200, bbox_inches='tight')
    plt.show()

def plot_grouped_boxplots(results, label_cols):
    data_list = []
    for trait in label_cols:
        for result in results[trait]:
            data_list.append({
                'Trait': trait,
                'Marker_View': result['marker_view'],
                'Model': result['model'],
                'PearsonR': result['PearsonR']
            })
    
    df = pd.DataFrame(data_list)
    
    # Create grouped boxplots
    plt.figure(figsize=(12, 8))
    
    # Create combination of Model + Marker_View for x-axis
    df['Model_View'] = df['Model'] + ' (' + df['Marker_View'] + ')'
    
    sns.boxplot(data=df, x='Trait', y='PearsonR', hue='Model_View')
    plt.xticks(rotation=45)
    plt.title('Model Performance Across Traits')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig('grouped_performance.png', dpi=200, bbox_inches='tight')
    plt.show()

def plot_horizontal_boxplot_by_trait(results, label_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for trait in label_cols:
        data_list = []
        for result in results[trait]:
            data_list.append({
                'Marker_View': result['marker_view'],
                'Model': result['model'],
                'PearsonR': result['PearsonR']
            })
        
        df = pd.DataFrame(data_list)
        
        plt.figure(figsize=(10, 8))  # Increased height for better readability
        sns.boxplot(data=df, y='Model', x='PearsonR', hue='Marker_View')
        plt.title(f'{trait}')
        plt.xlim(0, 1)  # Change from ylim to xlim
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'cv_boxplot_r_by_trait_{trait}.png'), dpi=200, bbox_inches='tight')
        plt.show()

def plot_vertical_boxplot_by_trait(results, label_cols, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for trait in label_cols:
        data_list = []
        for result in results[trait]:
            data_list.append({
                'Marker_View': result['marker_view'],
                'Model': result['model'],
                'PearsonR': result['PearsonR']
            })
        
        df = pd.DataFrame(data_list)
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x='Model', y='PearsonR', hue='Marker_View')
        plt.title(f'{trait}')
        plt.ylim(0, 1)  # Assuming Pearson R between 0-1
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'cv_boxplot_r_by_trait_{trait}.png'), dpi=200, bbox_inches='tight')
        plt.show()

def analyze_training_times(training_times_dict, output_dir=None):
    """
    Analyze and visualize training times across models and marker views
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for analysis
    rows = []
    for model_name, marker_dict in training_times_dict.items():
        for marker_view, times in marker_dict.items():
            if times:  # Only if we have timing data
                rows.append({
                    'Model': model_name,
                    'Marker_View': marker_view,
                    'Mean_Time_s': np.mean(times),
                    'Std_Time_s': np.std(times),
                    'Min_Time_s': np.min(times),
                    'Max_Time_s': np.max(times),
                    'N_Runs': len(times),
                    'Total_Time_s': np.sum(times)
                })
    
    if not rows:
        print("No timing data available for analysis")
        return None
    
    timing_df = pd.DataFrame(rows)
    
    # Print summary table
    print("\n" + "="*91)
    print("TRAINING TIME SUMMARY")
    print("="*91)
    print(timing_df.to_string(index=False))
    
    # Save to CSV
    if output_dir:
        csv_path = os.path.join(output_dir, "training_times_summary.csv")
        timing_df.to_csv(csv_path, index=False)
        print(f"\nTraining times summary saved to: {csv_path}")
    
    # Create visualization
    create_training_time_chart(timing_df, output_dir)
    
    return timing_df

def create_training_time_chart(timing_df, output_dir=None):
    """
    Create grouped bar chart of training times with marker views sorted as SNP, HAP, PC
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    
    if timing_df.empty:
        print("No data available for chart")
        return
    
    # Define the desired order for marker views
    desired_order = ['SNP', 'HAP', 'PC']
    
    # Get available marker views from data and sort according to desired order
    available_marker_views = timing_df['Marker_View'].unique()
    marker_views_sorted = [mv for mv in desired_order if mv in available_marker_views]
    
    # Add any remaining marker views that aren't in the desired order
    remaining_views = [mv for mv in available_marker_views if mv not in desired_order]
    marker_views_sorted.extend(sorted(remaining_views))
    
    # Pivot the data for plotting with sorted marker views
    pivot_df = timing_df.pivot(index='Model', columns='Marker_View', values='Mean_Time_s')
    
    # Reorder columns according to our sorted marker views
    pivot_df = pivot_df.reindex(columns=marker_views_sorted)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up bar positions
    models = pivot_df.index.tolist()
    marker_views = pivot_df.columns.tolist()
    
    x = np.arange(len(models))
    width = 0.8 / len(marker_views)  # Dynamic width based on number of marker views
    
    # Define colors for each marker view (optional)
    colors = {
        'SNP': '#1f77b4',  # blue
        'HAP': '#ff7f0e',  # orange
        'PC': '#2ca02c'    # green
    }
    
    # Create bars for each marker view in the sorted order
    bars = []
    for i, marker_view in enumerate(marker_views):
        times = pivot_df[marker_view].values
        bar_pos = x + (i - len(marker_views)/2 + 0.5) * width
        
        # Use predefined color if available, otherwise use default
        color = colors.get(marker_view, None)
        bar = ax.bar(bar_pos, times, width, label=marker_view, alpha=0.8, color=color)
        bars.append(bar)
        
        # Add value labels on bars
        for j, (pos, time_val) in enumerate(zip(bar_pos, times)):
            if not np.isnan(time_val):
                ax.text(pos, time_val + time_val*0.01, f'{time_val:.1f}s', 
                       ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Customize the chart
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Training Time (seconds)', fontsize=12, fontweight='bold')
    ax.set_title('Training Time Comparison by Model and Marker Type', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Marker Views', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    if output_dir:
        chart_path = os.path.join(output_dir, "training_times_comparison.png")
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        print(f"Training time chart saved to: {chart_path}")
    
    plt.show()

def analyze_training_memories(training_memories_dict, output_dir=None):
    """
    Analyze and visualize training memory usage across models and marker views
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    import os
    
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Convert to DataFrame for analysis
    rows = []
    for model_name, marker_dict in training_memories_dict.items():
        for marker_view, memories in marker_dict.items():
            if memories:  # Only if we have memory data
                rows.append({
                    'Model': model_name,
                    'Marker_View': marker_view,
                    'Mean_Memory_MB': np.mean(memories),
                    'Std_Memory_MB': np.std(memories),
                    'Min_Memory_MB': np.min(memories),
                    'Max_Memory_MB': np.max(memories),
                    'N_Runs': len(memories)
                })
    
    if not rows:
        print("No memory usage data available for analysis")
        return None
    
    memory_df = pd.DataFrame(rows)
    
    # Print summary table
    print("\n" + "="*80)
    print("TRAINING MEMORY USAGE SUMMARY")
    print("="*80)
    print(memory_df.to_string(index=False))
    
    # Save to CSV
    if output_dir:
        csv_path = os.path.join(output_dir, "training_memory_usage_summary.csv")
        memory_df.to_csv(csv_path, index=False)
        print(f"\nTraining memory usage summary saved to: {csv_path}")
    
    # Create visualization
    create_training_memory_chart(memory_df, output_dir)
    
    return memory_df

def create_training_memory_chart(memory_df, output_dir=None):
    """
    Create grouped bar chart of training memory usage with marker views sorted as SNP, HAP, PC
    """
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    import os
    
    if memory_df.empty:
        print("No data available for memory usage chart")
        return
    
    # Define the desired order for marker views
    desired_order = ['SNP', 'HAP', 'PC']
    
    # Get available marker views from data and sort according to desired order
    available_marker_views = memory_df['Marker_View'].unique()
    marker_views_sorted = [mv for mv in desired_order if mv in available_marker_views]
    
    # Add any remaining marker views that aren't in the desired order
    remaining_views = [mv for mv in available_marker_views if mv not in desired_order]
    marker_views_sorted.extend(sorted(remaining_views))
    
    # Pivot the data for plotting with sorted marker views
    pivot_df = memory_df.pivot(index='Model', columns='Marker_View', values='Mean_Memory_MB')
    
    # Reorder columns according to our sorted marker views
    pivot_df = pivot_df.reindex(columns=marker_views_sorted)
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Set up bar positions
    models = pivot_df.index.tolist()
    marker_views = pivot_df.columns.tolist()
    
    x = np.arange(len(models))
    width = 0.8 / len(marker_views)  # Dynamic width based on number of marker views
    
    # Define colors for each marker view (same as timing chart for consistency)
    colors = {
        'SNP': '#1f77b4',  # blue
        'HAP': '#ff7f0e',  # orange
        'PC': '#2ca02c'    # green
    }
    
    # Create bars for each marker view in the sorted order
    bars = []
    for i, marker_view in enumerate(marker_views):
        memories = pivot_df[marker_view].values
        bar_pos = x + (i - len(marker_views)/2 + 0.5) * width
        
        # Use predefined color if available, otherwise use default
        color = colors.get(marker_view, None)
        bar = ax.bar(bar_pos, memories, width, label=marker_view, alpha=0.8, color=color)
        bars.append(bar)
        
        # Add value labels on bars
        for j, (pos, memory_val) in enumerate(zip(bar_pos, memories)):
            if not np.isnan(memory_val):
                ax.text(pos, memory_val + memory_val*0.01, f'{memory_val:.0f}MB', 
                       ha='center', va='bottom', fontsize=9, rotation=90)
    
    # Customize the chart
    ax.set_xlabel('Models', fontsize=12, fontweight='bold')
    ax.set_ylabel('Memory Usage (MB)', fontsize=12, fontweight='bold')
    ax.set_title('Training Memory Usage Comparison by Model and Marker Type', 
                fontsize=14, fontweight='bold', pad=20)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.legend(title='Marker Views', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(axis='y', alpha=0.3)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save the chart
    if output_dir:
        chart_path = os.path.join(output_dir, "training_memory_usage_comparison.png")
        plt.savefig(chart_path, dpi=200, bbox_inches='tight')
        print(f"Training memory usage chart saved to: {chart_path}")
    
    plt.show()

#import logging
#def setup_logging(quiet=False, verbose=False):
#    """Setup logging based on verbosity"""
#    if quiet:
#        level = logging.WARNING
#    elif verbose:
#        level = logging.DEBUGTrue
#    else:
#        level = logging.INFO
        
#    logging.basicConfig(
#        level=level,
#        format='%(asctime)s - %(levelname)s - %(message)s',
#        datefmt='%Y-%m-%d %H:%M:%S'
#    )


# Usage throughout your code:
#logging.info("Training %s...", model_name)
#logging.debug("DEBUG: X shape: %s", X.shape)
#logging.warning("Warning: Constant predictions")

# -----------------------------
# Main Function with Logging
# -----------------------------
def main():
    # version
    version = '2.0'

    # Start timing
    start_time = time.time()

    # silence all DEBUG messsages. Comment it if you wna tot see all DEBUG message
    sys.stdout = FilterDebug()

    parser = argparse.ArgumentParser(description=f'MultiGS-P {version}: Genomic Selection Pipeline')
    parser.add_argument('-c', '--config', required=True, help='Path to configuration file')
    #parser.add_argument('-q', '--quiet', action='store_true', help='Suppress non-essential output')
    #parser.add_argument('--verbose', action='store_true', help='Show detailed output')
    
    args = parser.parse_args()
    
    # Parse config
    config = parse_config(args.config)
    
    # Override with command line args
    #if args.quiet:
    #    config['quiet_mode'] = True
    #if args.verbose:
    #    config['verbose_mode'] = True
    
    # Set up logging based on quiet/verbose
    #setup_logging(config)

    # Check if R models are requested
    r_models_requested = any(model in config['enabled_models'] 
                           for model in ['R_RRBLUP', 'R_GBLUP'])
    
    if r_models_requested:
        print("R models requested, checking dependencies...")
        r_available = check_r_dependencies()
        if not r_available:
            print("WARNING: R models requested but dependencies not met.")
            print("R models will use Python fallbacks or be skipped.")
    
    # VALIDATE: Check if any models are enabled
    if not config['enabled_models']:
        print("❌ ERROR: No models are enabled in the configuration file.")
        print("Please enable at least one model in the [Models] section.")
        sys.exit(1)

    os.makedirs(config['results_dir'], exist_ok=True)
        
    # Validate input parameters incluidng vcf file format and if the external tool  
    # rtm-gwas-snpldb_path is available
    validate_pipeline_inputs(config, base_dir=None, strict=True)

    # Setup logging
    log_path = os.path.join(config['results_dir'], 
                           f"gs_{config.get('run_mode', 'cv').lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    log_fh = open(log_path, 'w', buffering=1, encoding='utf-8')
    
    # Redirect stdout and stderr to both console and log file
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    sys.stdout = Tee(original_stdout, log_fh)
    sys.stderr = Tee(original_stderr, log_fh)
    
    # Ensure log file is closed on exit
    def cleanup():
        sys.stdout = original_stdout
        sys.stderr = original_stderr
        log_fh.close()
    atexit.register(cleanup)
    
    print(f"[Log] Writing console output to: {log_path}")
    print('=' * 66)
    print(f'MultiGS-P ({version}): GENOMIC SELECTION PIPELINE USING MACHINING LEARNING AND DEEP LEARNING MODELS')
    print('=' * 66)
    print(f"Mode: {config['run_mode'].upper()}")
    print(f"Enabled models: {', '.join(config['enabled_models'])}")
    print(f"Feature (marker) views: {config['feature_views']}")
    print(f"Results directory: {config['results_dir']}")

    if config['run_mode'] == 'prediction':
        print(f"Training marker file: {config['vcf_path']}")
        print(f"Test marker file: {config['test_vcf_path']}")
        print(f"Training pheno file: {config['phenotype_path']}")
        print(f"Test pheno file: {config['test_phenotype_path']}")
    else:
        print(f"Marker file: {config['vcf_path']}")
        print(f"Pheno file: {config['phenotype_path']}")
    print('-' * 66)
    
    try:
        if config['run_mode'] == 'prediction':
            print("Running in PREDICTION mode")
            predictions, evaluation = run_prediction_mode(config)
            print("Prediction completed successfully!")
            
        else:
            print("Running in CROSS-VALIDATION mode")
            results, label_cols = run_cross_validation(config)
            
            print(f"\nRESULT SUMMRIZATION:")

            # Generate the summary table for all haplotype markers
            summary_df = generate_summary_table(results, label_cols)
            # Display the table
            #print("\n=== SUMMARY STATISTICS ===")
            #print(summary_df.to_string(index=False))
            # save to CSV
            summary_file = os.path.join(config['results_dir'], 'cv_gs_summary_stats_fm2.csv')
            summary_df.to_csv(summary_file, index=False)
            
            # Create boxplots for all results
            plot_horizontal_boxplot_by_trait(results, label_cols, config['results_dir'])

            stats = generate_statistics_report(results, label_cols)
            detailed_fn, summary_fn, json_fn  = save_results(
                results, label_cols, stats, config)
            
            print(f"\nResults saved to:")
            print(f"  - {detailed_fn}")
            print(f"  - {summary_fn}")
            print(f"  - {json_fn}")
            print(f"  - {summary_file}")


            # perfrom ANOVA
            # after your pipeline writes cv_gs_detailed_results.csv to results_dir:
            run_anova(
                csv_path=os.path.join(config['results_dir'], 'cv_gs_detailed_results.csv'),
                out_dir=os.path.join(config['results_dir'], "anova"),
            )


    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:

        # After all model training is complete
        print("\n" + "-"*75)
        print("PERFORMANCE ANALYSIS")
        print("-"*75)
        
        # Analyze and visualize training times
        timing_summary = analyze_training_times(training_times, output_dir=config['results_dir'])

        # Analyze and visualize training memory usage
        usage_summary = analyze_training_memories(training_memories, output_dir=config['results_dir'])


        # Calculate and display total time
        end_time = time.time()
        total_seconds = end_time - start_time
        total_minutes = total_seconds / 60
        
        print('=' * 66)
        print(f"Total execution time: {total_minutes:.2f} minutes ({total_seconds:.2f} seconds)")
        print('=' * 66)
        run_prediction_mode
        # Ensure log file is properly closed
        cleanup()

if __name__ == '__main__':
    main()