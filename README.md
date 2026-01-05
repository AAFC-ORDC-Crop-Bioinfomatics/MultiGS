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
-   **Diverse GS Modeling Methods:** Integrates several state-of-the-art
     models via R packages:
    
    -  **Linear Models:** Ridge-Regression BLUP (RR-BLUP) via rrBLUP and Genomic Best Linear Unbiased Predictio (GBLUP) via BGLR.

    -  **Kernel Methods:** Reproducing Kernel Hilbert Spaces (RKHS).

    -  **System RBayesian Approaches:** BL (Bayesian LASSO), BRR (Bayesian Ridge Regression), BayesA, BayesB, BayesC via BGLR.

    -  **Machine Learning:** Random Forest for Regression (RFR) and Classification (RFC), Support Vector Regression (SVR) and Classification (SVC).
    


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
