# MultiGS
A Comprehensive and User-Friendly Genomic Selection platform Integrating Statistical, Machine Learning, and Novel Deep Learning Models for Breeders

For detailed MultiGS-R user guide:<a href="Documents/MultiGS-R_v1.0_user_guide.md"> MultiGS-R_v1.0_user_guide.md</a>  

A MultiGS-R tutorial is available in the file <a href="Documents/MultiGS-R_v1.0_tutorial.pdf">MultiGS-R_v1.0_tutorial.pdf</a>

For detailed MultiGS-P user guide: <a href="Documents/MultiGS-P_v1.0_user_guide.md">MultiGS-P_v1.0_user_guide.md</a>

## Table of Contents
- [Key Features](#key-features)
- [Installation](#installation)
- [License](#license)
  
## Key Features

-   **Flexible Analysis Modes:** Supports both **cross-validation** (for
    model evaluation) and independent **across-population prediction**
     (using a training set to predict a test set).

-   **Multiple Marker Views:**

    -   **SNP:** Direct use of Single Nucleotide Polymorphisms (SNPs).

    -   **HAP:** Conversion of SNPs into haplotype blocks using RTM-GWAS
         SNP-LD for potentially capturing epistatic effects.

    -   **PC:** Use of Principal Components as markers to reduce
         dimensionality and address multicollinearity.

-   **Comprehensive Data Preprocessing:** Includes sample alignment, genotype harmonization between training and test lines, and missing data imputation.
-   **Comprehensive Analysis:** Includes phenotype analysis, visualization and statistical reporting
-   **Diverse GS Modeling Methods:**

**Table 1. Linear and machine learning models implemented in MultiGS-R**

| Model name | Full name | Model category | Core algorithm | Key features | R package(s) |
|-----------|-----------|----------------|----------------|--------------|--------------|
| RR-BLUP | Ridge Regression Best Linear Unbiased Prediction | Linear mixed model | Penalized linear regression with ridge penalty | Assumes equal variance of marker effects; computationally efficient baseline | rrBLUP |
| GBLUP | Genomic Best Linear Unbiased Prediction | Linear mixed model | Genomic relationship matrix–based BLUP | Models additive genetic relationships using genomic kinship | BGLR |
| BRR | Bayesian Ridge Regression | Bayesian linear model | Bayesian ridge regression | Shrinkage of marker effects with Gaussian prior | BGLR |
| BL | Bayesian LASSO | Bayesian linear model | LASSO with Laplace prior | Allows variable shrinkage across markers | BGLR |
| BayesA | BayesA | Bayesian linear model | Marker-specific variance model | Heavy-tailed priors capture large-effect loci | BGLR |
| BayesB | BayesB | Bayesian linear model | Mixture model with spike-and-slab prior | Performs variable selection by excluding many markers | BGLR |
| BayesC | BayesC | Bayesian linear model | Modified BayesB with shared variance | Improved stability and reduced sensitivity to hyperparameters | BGLR |
| RKHS | Reproducing Kernel Hilbert Space regression | Kernel-based model / ML | Gaussian kernel regression | Captures nonlinear and epistatic effects | BGLR |
| RFR | Random Forest Regression | Machine learning | Ensemble of decision trees | Captures nonlinear interactions; robust to noise | randomForest |

**Table 2. Summary of eight linear and machine learning models implemented in MultiGS-P**

| Model | Architecture / Type | Core Algorithm / Method | Key Features | Best Use Cases |
|------|---------------------|-------------------------|--------------|----------------|
| R_RRBLUP | Linear Mixed Model (R) | Ridge regression BLUP via R package rrBLUP | Widely validated baseline | Additive traits |
| R_GBLUP | Linear Mixed Model (R) | Genomic relationship kernel BLUP | Captures population structure | Standard GS benchmark |
| RRBLUP | Linear regression (Python) | Ridge regression | Consistent with R version | Additive effects |
| ElasticNet | Linear model (L1 + L2) | Elastic-net regularization | Feature shrinkage | Sparse / noisy SNP effects |
| BRR | Bayesian linear regression | Gaussian priors | Uncertainty estimation | Moderate shrinkage traits |
| RFR | Ensemble of trees | Random Forest | Nonlinear interactions | Epistasis / nonlinear |
| XGBoost | Gradient boosting trees | Additive boosting | Handles complex patterns | Large SNP sets |
| LightGBM | Gradient boosting trees | Histogram-based boosting | Fast, scalable | High-dimensional SNPs |


**Table 3. Summary of nine deep learning model architectures implemented in MultiGS-P**

| Model | Architecture Type | Main Components | Key Properties |
|------|-------------------|-----------------|----------------|
| DNNGS | Fully connected deep neural network | Input dropout; 4 fully connected blocks (512–256–128–64) with ReLU and dropout; optional batch normalization; final linear prediction head | Efficient nonlinear modeling of genotype features; simple and fast baseline DL model |
| MLPGS | MLP with normalization and residual connections | Input dropout; 2 dense blocks (128→64) with GELU/ReLU, dropout, layer normalization; optional residual skip connections; output normalization + final dense layer | Stabilized deep MLP with improved training dynamics and gradient flow |
| GraphConvGS | Graph Convolutional Network (GCN) | KNN graph construction; 2×GCNConv + layer norm + ReLU + dropout; node-wise MLP | Models sample-to-sample similarity; effective for population-structured GS data |
| GraphAttnGS | Graph Attention Network (GAT) | KNN graph; 2×GATConv (multi-head attention) + layer norm + dropout; node MLP | Learns adaptive attention weights over neighbors; flexible modeling of heterogeneous relationships |
| GraphSAGEGS | GraphSAGE neighborhood aggregator | KNN graph; 2×SAGEConv + layer norm + dropout; node MLP | Inductive, scalable, and robust across populations; robust performance on large datasets |
| GraphFormer | GraphSAGE + Transformer hybrid | 2×SAGEConv → node embeddings → Transformer encoder → MLP | Captures both local graph structure and global node interactions for enhanced representation learning |
| DeepResBLUP | Residual hybrid (RRBLUP + DL) | Fit RRBLUP baseline; DL model fits residual signal; weighted combination of linear and nonlinear predictions | Effective additive baseline with nonlinear correction; interpretable and stable |
| DeepBLUP | Integrated BLUP-in-DL architecture | RRBLUP-like linear layer → 3 dense blocks (256–128–64) with GELU, batch norm, dropout, and residual connections; optional skip link from RRBLUP output | Deep refinement of BLUP with modern DL structure; robust performance for additive + mild nonlinear effects |
| EnsembleGS | Stacked ensemble | Trains multiple base models; collects OOF predictions; meta-learner (linear regression) for final prediction | Most robust across datasets; leverages complementary strengths of diverse model families |
 

    


## Installation

### Prerequisites

1.  **Java Runtime Environment (JRE):** Version 21 or higher must be
    installed. You can check by running `java -version` in your
    terminal.

2.  **R:** Version 3.5 or higher must be installed and accessible from
     the command line. Check with `R --version`.

3.  **Rscript:** This executable (included with R) must be in your
     system\'s PATH.

### Installing R Libraries

Before running MultiGS, you must install the required R packages. Start an R session and run the following commands:
```bash
r
```
```r
# Install required packages from CRAN
install.packages(c("rrBLUP", "BGLR", "randomForest", "e1071","ade4", "sommer", "ggplot2"))
```
An additional G2P package needs to be installed through source file. Please download it from GitHub and follow installation instruction:
<https://github.com/cma2015/G2P>

### Clone the Repository
```bash
git clone https://github.com/AAFC-ORDC-Crop-Bioinfomatics/MultiGS.git
cd MultiGS
```
### Create the Conda Environment
```bash
conda env create -f environment.yml
conda activate multigs
```
## License

This project is licensed under the terms of the **MIT License**.
