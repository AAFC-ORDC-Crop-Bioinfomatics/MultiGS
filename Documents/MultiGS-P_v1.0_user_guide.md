# MultiGS-P

## A Genomic Selection Pipeline for Multiple Single Traits Using Diverse Machine Learning and Deep Learning Models and Marker Types

**MultiGS_P** is a comprehensive Python-based genomic selection pipeline for multiple single traits. It integrates **machine learning**, **deep learning**, and **classical statistical models** for genomic prediction. The pipeline supports multiple feature types, including SNPs, haplotypes, and principal components and provides both **cross-validation (CV)** and **across-population prediction (APP)** modes, enabling robust genomic selection analyses and practical applications in breeding programs.

# Table of Contents
- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Configuration](#configuration)
- [Input Files](#input-files)
- [Advanced Configuration](#advanced-configuration)
- [Usage](#usage)
- [Output](#output)
- [Troubleshooting](#troubleshooting)
- [Logging](#logging)
- [Tutorials](#tutorials)
  

# Overview

**MultiGS-P** provides an end-to-end workflow for **Genomic Selection (GS)**, from data preprocessing to model training, evaluation, and prediction. The pipeline streamlines all stages of the genomic selection process, enabling **reproducible** and **scalable** genomic prediction across diverse datasets and environments. The pipeline supports multiple **genomic marker representations** including SNPs, Haplotypes, and Principal Components along with a broad suite of **statistical, machine learning, and deep learning algorithms**, making it a complete platform for both **cross-validation** and **across-population prediction** workflows.

# Key Features
### Multiple marker types (Feature Views):
- **SNP marker (SNP)**
  - Raw SNP markers (0,1,2 encoding)
  - Direct use of individual SNP effects
- **Haplotype View (HAP)**
  - Constructed using RTM-GWAS SNPLDB tool
  -	Captures linkage disequilibrium patterns
  -	Reduces dimensionality while preserving genetic information
- **Principal Components (PC)**
  - Dimensionality reduction via PCA
  -	Configurable variance threshold for component selection
  -	Efficient representation of genetic structure
### Diverse Model Support:
- **Machine learning:** 
    - **Random Forest (RFR):** Ensemble of decision trees
    - **XGBoost:** Gradient boosting implementation
    - **LightGBM:** Light gradient boosting machine
- **Deep learning:** 
 	- **CNN:** Convolutional Neural Network for spatial patterns
    - **HybridCNN:** CNN with Transformer-like attention
    - **MLP:** Multi-Layer Perceptron with advanced features
    - **DNN:** Deep neural network
    - **Transformer:** Deep learning (attention)
    - **GCN:** Graph Convolutional Neural Network in Graph nural network (GNN) 
    - **GAT:** Graph Attention Network in GNN
    - **GraphSAGE:** Graph Sample and AggregatE in GNN
- **Statistical/linear models:** 
    - **ElasticNet:** Linear model with L1 and L2 regularization
	- **LASSO:** L1-regularized linear model
	- **BRR:** Bayesian Ridge Regression
 	- **R-RRBLUP** Ridge regression BLUP, R-based
  	- **R-GBLUP** Genomic Best Linear Unbiased Prediction, R-based
 	- **RRBLUP** Python implimentation of RRBLUP
- **Ensemble:**
    - **Stacking**: Meta-ensemble combining multiple base models
	- Any models implemented in this pipeline can be stacked.

### Comprehensive Analysis:
 - Phenotype analysis 
 - Visualization
 - Statistical reporting

### Primary Information of Supported Models:
| Model Name | Architecture / Type | Core Algorithm / Method | Key Features / Implementation | Advantages | Performance | Best Use Cases |
|------------|----------------------|--------------------------|------------------------------|------------|-------------|----------------|
| **R_RRBLUP (R)** | Linear mixed model | Ridge regression BLUP via rrBLUP (R) | R backend execution | Trusted RRBLUP implementation | Fast | GS pipelines requiring R benchmarks |
| **R_GBLUP (R)** | Linear mixed model | GBLUP using R (rrBLUP) | R backend; genomic relationship matrix | Captures population structure | Fast–moderate | Across-population prediction |
| **RRBLUP (Python)** | Linear mixed model | Equivalent to GBLUP with homogeneous SNP variance | `sklearn.linear_model.Ridge` | Classical GS baseline; stable | Very fast | Traits with polygenic architecture |
| **ElasticNet** | Linear model | L1 + L2 regularized regression | `sklearn.linear_model.ElasticNet` | Handles correlated SNPs; variable shrinkage + selection | Very fast | Baseline GS; large SNP panels |
| **LASSO** | Linear model | L1-regularized regression | `sklearn.linear_model.Lasso` | Sparse feature selection | Very fast | When only a few markers matter |
| **BRR (Bayesian Ridge)** | Bayesian linear | Bayesian regression with priors | `sklearn.linear_model.BayesianRidge` | Probabilistic; captures uncertainty | Fast–moderate | Classical benchmark; small–medium datasets |
| **RFR (Random Forest)** | Ensemble (bagging) | Randomized decision trees | `sklearn.ensemble.RandomForestRegressor` | Non-linear interactions; robust to noise | Moderate | Complex architectures; mixed effects |
| **XGBoost** | Ensemble (boosting) | Gradient boosted trees | `xgboost.XGBRegressor` | Captures nonlinearities & SNP interactions | Moderate | SNP interactions; medium–large sets |
| **LightGBM** | Ensemble (boosting) | GOSS + leaf-wise growth | `lightgbm.LGBMRegressor` | Faster and lighter than XGBoost | Fast–moderate | Very large SNP sets |
| **MLP** | Neural network | Multi-layer perceptron | Dense layers + GELU + dropout | Simple, robust NN baseline | Moderate | Medium datasets; mild nonlinearity |
| **DNN** | Deep neural network | 4–5 layer fully connected architecture | ReLU + batch norm + dropout | Learns complex genotype–phenotype patterns | Moderate–slow | Large marker sets; continuous traits |
| **CNN** | Deep learning | 1D convolutional network | 2–3 conv layers + pooling + dropout | Captures local LD/block patterns | Slowest (GPU helps) | Traits influenced by local SNP clusters |
| **HybridCNN** | Deep learning hybrid | CNN + Transformer-like attention | CNN extractor → attention → MLP | Captures local + long-range SNP interactions | Slowest (GPU recommended) | High-dimensional SNP data |
| **Transformer** | Deep learning (attention) | Transformer encoder with positional encoding | Multi-head self-attention; 6 encoder layers | Global long-range dependency modeling | Slow, GPU intensive | Traits with long-range LD |
| **GCN (Graph Convolutional Network)** | Graph neural network | Node message passing on correlation graph | PyTorch Geometric; correlation graph using SAGEConv | Learns relatedness/population structure | Slowest | Across-population prediction; sample-as-node modeling |
| **GAT (Graph Attention Network)** | Graph neural network | Attention-weighted message passing | PyG `GATConv` | Learns heterogeneous relationships | Very slow | Structured populations; relatedness-sensitive traits |
| **GraphSAGE** | Graph neural network | Aggregation-based message passing | PyG `SAGEConv` | Scales better than GAT; inductive | Very slow | Population graphs; clustering structure |
| **Stacking** | Ensemble meta-model | Level-2 regression stacking | Linear model combining ML + DL predictions | Improves robustness; reduces model variance | Depends on base models | Integrative predictions across models |


# Installation

### 1. Clone the Repository
```bash
git clone https://github.com/AAFC-ORDC-Crop-Bioinfomatics/MultiGS-P.git
cd MultiGS-P
```
### 2. Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate multigs_p
```
### 3. Installing *rtm-gwas-snpldb* tool
The rtm-gwas-snpldb tool for haplotype block identification is included in the utilities folder in this repository. The latest executable can also be downloaded separately from:
https://github.com/njau-sri/rtm-gwas

# Configuration

All pipeline settings are defined in a single `.ini` configuration file. 

A complete sample configuration file is provided. Only a few sections may need to be modified—see the user guide for details.

### Example

```ini
[General]
seed = 42
n_replicates = 1
n_folds = 5

# Pipeline will create this folder, if didn't exist
results_dir = /path/to/results_dir

[Data]
# Required for either cross-validation (CV) or across-population prediction (APP) mode
vcf_path = /path/to/train_genotype.vcf
phenotype_path = /path/to/train_phenotype.txt

# Required only for APP mode. If CV mode, this section should be commented
test_vcf_path = /path/to/test_genotype.vcf
test_phenotype_path = /path/to/test_phenotype.txt

# For data preprocessing
pheno_normalization = standard
genotype_normalization = standard

# for principal component (PC) marker type only
pca_variance_explained = 0.95  # Use enough components to explain 95 percent variance

[Tool]
# For HAP marker only
rtm-gwas-snpldb_path = /path/to/rtm-gwas-snpldb

[FeatureView]
# Marker type options: SNP, HAP, PC
# Default: SNP
feature_view = SNP,HAP,PC      # SNP,HAP,PC:  one, any two, or all three separated by ','

[Models]
# Classical GS
R_RRBLUP = True
R_GBLUP = True
RRBLUP = True
BRR = True

# Linear / Regression-based
ElasticNet = True
LASSO = True

# Tree-based
RFR = True
XGBoost = True
LightGBM = True

# Neural Network Based
CNN = True
HybridCNN = True
MLP = True
DNN = True
Transformer = True
GCN = True
GAT = True
GraphSAGE = True

# Ensemble
Stacking = True

# Hyper-Parameters for all available Models

# ================================
# Linear / Regression-based
# ================================

[Hyperparameters_R_RRBLUP]
method = REML

[Hyperparameters_R_GBLUP] 
method = REML

[Hyperparameters_RRBLUP]
lambda_value = None
method = mixed_model
lambda_method = auto    ; Options: auto, reml, heritability, fixed
tol = 1e-8

[Hyperparameters_ElasticNet]
# Reduce regularization for ElasticNet: fro, 1 to 0.1->0.01->0.001
alpha = 1.0
l1_ratio = 0.1   # toward ridge: from 0.5 to 0.1-0.3

[Hyperparameters_LASSO]
alpha = 1.0

[Hyperparameters_XGBoost]
n_estimators = 100
max_depth = 6
learning_rate = 0.1
subsample = 0.8
colsample_bytree = 0.8
random_state = 42

[Hyperparameters_LightGBM]
n_estimators = 100
max_depth = -1
learning_rate = 0.1
num_leaves = 31
subsample = 0.8
colsample_bytree = 0.8
random_state = 42

[Hyperparameters_MLP]
hidden_layers = 1024,512,256
activation = gelu
norm = layer
residual = true
input_dropout = 0.05
dropout = 0.5

learning_rate = 0.0005
weight_decay = 0.0015
batch_size = 16
epochs = 300
early_stopping_patience = 20
warmup_ratio = 0.1
grad_clip = 1.0
seeds = 3

use_huber = true
huber_delta = 1.0

swa = true
swa_start = 0.7
swa_freq = 1

[Hyperparameters_DNN]
hidden_layers = 512,256,128,64
learning_rate = 0.001
batch_size = 32
epochs = 300
dropout = 0.3
activation = gelu
batch_norm = true
weight_decay = 0.0001
input_dropout = 0.1

[Hyperparameters_CNN]
hidden_channels = 128,128,256
kernel_size = 7
pool_size = 2
learning_rate = 0.0005
batch_size = 32
epochs = 300
dropout = 0.5
weight_decay = 0.001
grad_clip = 1.0
early_stopping_patience = 30
warmup_ratio = 0.1
seeds = 3

[Hyperparameters_HybridCNN]
cnn_channels = 128,128,256
kernel_size = 5
pool_size = 2
attention_heads = 8
hidden_size = 256
learning_rate = 0.00075
batch_size = 64
epochs = 300
dropout = 0.5
attention_dropout = 0.3
weight_decay = 0.001
grad_clip = 1.0
warmup_ratio = 0.1
patience = 30

[Hyperparameters_Stacking]
base_models = BRR, MLP, DNN, GraphSAGE, HybridCNN, ElasticNet
meta_model = linear
meta_alpha = 1.0

[Hyperparameters_Transformer]
d_model = 512
nhead = 8
num_layers = 6
dim_feedforward = 2048
dropout = 0.1
learning_rate = 0.0001
batch_size = 32
epochs = 500
weight_decay = 0.01
warmup_ratio = 0.1
patience = 40
grad_clip = 1.0

[Hyperparameters_GCN]
hidden_channels = 128
num_layers = 2
hidden_mlp = 128
dropout = 0.2
learning_rate = 0.0005
epochs = 300
top_k = 20
graph_method = knn
knn_metric = euclidean
patience = 20

[Hyperparameters_GAT]
hidden_channels = 128
num_layers = 2
heads = 4
hidden_mlp = 128
dropout = 0.2
learning_rate = 0.0005
epochs = 300
top_k = 20
graph_method = knn
knn_metric = euclidean
patience = 20

[Hyperparameters_GraphSAGE]
hidden_channels = 128
num_layers = 2
hidden_mlp = 128
dropout = 0.2
learning_rate = 0.0005
epochs = 300
top_k = 20
graph_method = knn
knn_metric = euclidean
aggr = mean
patience = 20

```

# Input Files
### Genotype Data
- VCF format: Standard Variant Call Format with biallelic SNPs
- VCF file can be gziped (*.gz) 
- VCF file must have standard header lines, containing at least one header line, such as
```ini
##fileformat=VCFv4.2
```
### Phenotype Data
- CSV file or tab-delimited text file with samples as rows and traits as columns
- Support multiple traits
- First column should contain sample IDs
- Missing values are automatically imputed with trait medians

# Advanced Configuration
## PCA Configuration
```ini
[Data]
pca_variance_explained = 0.95  # Auto-select components to explain 95% variance
pca_fit_scope = train  # or 'combined' for train+test
```

## Normalization Options
```ini
[Data]
pheno_normalization = standard  # standard, minmax, robust, or none
genotype_normalization = standard
```
## Across-population prediction
You need to set additional data file for **across-population prediction"
```ini
[Data]
test_vcf_path = new_samples.vcf
test_phenotype_path = new_phenotypes.csv  # Optional for evaluation
```

## Model-specific Hyperparameters
Each model supports extensive hyperparameter tuning through the configuration file.


# Usage

Once the configuration file (`config.ini`) is prepared, the pipeline can be executed directly as a Python module.

### Run the Pipeline
You need to first activate the Conda environment (multigs_p, previously created in installation) that contains all the required libraries.

```bash
conda activate multigs_p
python MultiGS-P_1.0.pyc --config config.ini
```

### Releases

Precompiled binaries of **MultiGS-P** are available under the [Releases](../../releases) section of this repository.  To use the binary (No Environment is required):
```bash
./MultiGS-P_1.0 --config config.ini
```

# Output 
The pipeline generates a well-organized directory structure:
```bash
Results_folder/
├── phenotype_analysis/
│   ├── phenotype_stats_train.csv
│   ├── phenotype_histograms_train.png
│   └── phenotype_corr_heatmap_train.png
├── scatter_plots/
│   ├── true_vs_pred_trait_model.png
│   └── true_vs_pred_ALL_trait.png
├── trait_predictions/
│   ├── prediction_view_model.csv
│   └── predictions_view_trait_trait.csv
├── anova/
│   ├── anova_table.csv
│   ├── group_means.csv
│   └── tukey_*.csv
├── cv_boxplot_r_by_model.png
├── cv_grouped_boxplot_r.png
├── mds_grm_scatter_plot.png
└── *.log
```

# Troubleshooting
## Common Issues
1.	**VCF file errors:** Ensure VCF follows standard format with proper headers
2.	**External tool errors:** Verify RTM-GWAS SNPLDB installation and path configuration
3.	**Convergence warnings:** Adjust hyperparameters or normalization methods

# Logging
Detailed logs are saved in the results directory for debugging.


# Tutorials

Comprehensive instructions for configuration, data preparation, model training, and execution are provided in the **MultiGS-P User Guide (PDF)**.

The guide will be available in the repository’s `docs/` directory.


