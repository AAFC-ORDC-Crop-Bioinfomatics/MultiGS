# MultiGS-R

## Java Pipeline for Genomic Selection of Multiple Single Traits Using R-Based Models and Diverse Marker Types

**MultiGS-R** is a powerful, flexible, and user-friendly Java-based pipeline for performing genomic selection (GS) analysis. It seamlessly integrates a wide range of popular R packages implementing both classical statistical and modern machine learning models, providing a unified platform for cross-validation and across-population prediction in plant and animal breeding programs.

The pipeline supports multiple genomic marker types (SNPs, haplotypes, and principal components) and a comprehensive suite of GS modeling algorithms, making it an all-in-one solution for breeders and researchers.

A detailed tutorial is available in the file <a href="doc/MultiGS-R_v1.0_tutorial.pdf">MultiGS-R_v1.0_tutorial.pdf</a>

## Table of Contents

1. [Introduction](#introduction)
2. [Quick Start](#quick-start)
3. [Configuration File](#configuration-file)
    - [Sample Configuration](#sample-configuration)
    - [Parameter Details](#parameter-details)
4. [Input Files](#input-files)
    - [Genotypic Data (Markers)](#genotypic-data-markers)
    - [Phenotypic Data](#phenotypic-data)
5. [Usage](#usage)
6. [Output](#output)
7. [Troubleshooting](#troubleshooting)


## Introduction

Genomic Selection accelerates genetic improvement by predicting the genetic-estimated breeding values (GEBVs) of individuals based on their genomic markers. MultiGS-R automates the complex workflow of GS, which includes data preprocessing, quality control, imputation, model training, and validation. By leveraging the robust statistical capabilities of R within a managed Java pipeline, MultiGS-R ensures reproducibility, scalability, and ease of use for both small-scale studies and large breeding populations.

## Quick Start

-  **Prepare your data:** Have your VCF marker files and phenotypic data files ready.
-  **Create a configuration file:** Copy the sample below and modify the paths to match your system and data.
-  **Run the pipeline:** 
```bash
java -jar MultiGS-R_1.0.jar /path/to/your/config.ini
```

## 

## Configuration File

The pipeline is controlled by a single configuration file using an
INI-style format.

### Sample Configuration

```
# This is a configuration file for MultiGS-R pipeline.
[Tools]
# Haplotype block identification tool 
rtm_gwas_snpldb_path = /home/user/OmniGS-R/rtm_gwas/rtm-gwas-snpldb
# R path
RScriptPath = /usr/bin/Rscript

[General]
# variance explained for selection of number of principal components
pca_variance_explained = 0.95
# result output folder
result_folder = sample_results_CV
# Number of threads for parallel computation
threads = 7
# number of replicates in CROSS-VALIDATION mode
Replicates = 2

[GS_Mode]
# Mode: CROSS-VALIDATION \| PREDICTION
mode = CROSS-VALIDATION

[Feature_view]
# Three marker types: raw SNPs (SNP), haplotypes (HAP) and principal
components (PC)
marker_type = PC

[Data]
# (training) marker file (for cross_validation or Prediction)
marker_file=/path/to/training_markers.vcf
# test marker file (required for PREDICTION mode, optional for
CROSS-VALIDATION)
test_marker_file=/path/to/test_markers.vcf
# training phenotypic data file for both modes
training_pheno_file=/path/to/training_pheno.txt
# test phenotypic data file (optional, for PREDICTION mode only)
test_pheno_file=/path/to/test_pheno.txt

[Models]
# Choose GS modeling methods: True \| False
# Parametric/linear models
RR-BLUP = True
GBLUP = True
BRR = True
BL = True
BayesA = True
BayesB = True
BayesC = True

# Non-parametric machine learning methods
RFR = True
SVR = True
RKHS = True

# Classifiers
RFC = True
SVC = True

[Hyperparameters]
# Model parameters for Bayesian methods
nIter = 12000
burnIn = 2000
```
### 

### Parameter Details

  ---------------------------------------------------------------------------------------------------------------------------------
 | Section              | Parameter               | Description                                       | Values                     |
 |----------------------|-------------------------|---------------------------------------------------|----------------------------|           
 | **Tools**            | rtm_gwas_snpldb_path    | Path to haplotype block identification tool       | File path                  | 
 |                      | RScriptPath             | Path to RScript executable                        | File path                  |
 | **General**          | pca_variance_explained  | Variance cutoff for PCA component selection       | 0.0-1.0 (e.g., 0.95)       |
 |                      | result_folder           | Output directory for results                      | Directory path             | 
 |                      | threads                 | Number of CPU threads for parallel processing     | Integer                    |
 |                      | Replicates              | Number of CV replicates                           | Integer                    |
 | **GS_Mode**          | mode                    | Analysis mode                                     | CROSS-VALIDATION or PREDICTION|
 | **Feature_view**     | marker_type             | Type of markers to use                            | SNP, HAP, or PCA           |
 | **Data**             | marker_file             | Training population VCF file                      | File path                  |
 |                      | test_marker_file        | Test population VCF file (Prediction mode)        | File path                  |
 |                      | training_pheno_file     | Training phenotype data                           | File path                  |
 |                      | test_pheno_file         | Test phenotype data (optional)                    | File path                  |
 | **Models**           | Various                 | Enable/disable specific GS models                 | True or False              |
 | **Hyperparameters**  | nIter                   | MCMC iterations for Bayesian models               | Integer (e.g., 12000)      |
 |                      | burnIn                  | MCMC burn-in period                               | Integer (e.g., 2000)       | 

## 

## Input Files

### Genotypic Data (Markers)

-   **Format:** standard VCF (Variant Call Format) with header - can be compressed (.vcf.gz) or uncompressed
-   **Requirements:**
    -   For **Cross-Validation:** One VCF file for the training population
    -   For **Prediction:** Two VCF files (training and test)

### Phenotypic Data

-   **Format:** Tab-delimited text file **with a header row**
-   **Structure:**
    -   First column: Individual/Sample IDs
    -   Subsequent columns: Phenotypic values for different traits

**Example** training_pheno.txt

```text

SampleID Yield Height Weight
sample_1 5.6 112 45
sample_2 4.8 105 42
sample_3 NA 108 44
```
*Missing values should be coded as NA. The pipeline will handle them
automatically.*

## Usage

1.  **Prepare your configuration file** following the template above

2.  **Run the pipeline:**

   ``` bash
java -jar MultiGS-R_1.0.jar /path/to/your/config.ini
```

3.  **Monitor progress:**
The pipeline will display progress in the console and write detailed logs to the output directory

**For large datasets, you may need to increase memory allocation:**

```bash

java -Xmx8g -jar MiltiGS-R_1.0.jar config.ini
```

## Output

The pipeline generates a well-organized directory structure:

```bash
result_folder/
├── gs\_\<timestamp\>.log \# Detailed log file
├── all_CV_results.txt \# Detailed CV results (CV mode)
├── CV_summary_statistics.csv \# Summary statistics (CV mode)
├── prediction_detailed_results.txt \# Model results (Prediction mode)
│
├── trait_predictions/ \# Predicted values for test set
│ └── \<Trait\>\_\<Model\>\_prediction_data.txt
├── plots/ \# Diagnostic plots
│ ├── MDS_plot.png \# Population structure
│ └── \... \# Other visualizations
├── intermediate_data/ \# Processed intermediate files
└── pheno_data/ \# Preprocessed phenotypic data
```
## Troubleshooting

-   **\"RScript not found\":**
  Verify the RScriptPath in your configuration file is correct

-   **Missing R packages:**
  Check the log file for package errors and install missing packages in R

-   **Memory errors:**
  Use -Xmx parameter to increase Java heap space  e.g., -Xmx8g for 8GB)

-   **VCF file errors:**
  Ensure your VCF files are properly formatted and have necessary header


