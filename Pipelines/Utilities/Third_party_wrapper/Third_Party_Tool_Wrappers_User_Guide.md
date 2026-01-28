# User Guide for Third-Party Tool Wrappers

Many genomic prediction and deep learning tools published by research
groups are difficult to install and run due to complex dependencies and
command-line options.\
To simplify their usage, we provide wrapper scripts that standardize
input formats, automate preprocessing, and manage execution.

Before running any wrapper, please make sure that: - All required
dependencies for the original tool are installed (Python, PyTorch, CUDA,
R, etc.). - Input genotype and phenotype files follow the formats
described in each tool's documentation. - You are running in the correct
conda / virtual environment.

------------------------------------------------------------------------

## 1. Cropformer

**Original tool:**\
GitHub: https://github.com/jiekesen/Cropformer

**Wrapper script:**\
`cropformer_pipeline.py`

**Description:**\
This wrapper prepares genotype and phenotype data, configures model
parameters, and launches the Cropformer training and prediction workflow
using a configuration file.

**How to run:**

``` bash
python cropformer_pipeline.py -c cropformer_config.ini > output.log 2>&1
```

**Notes:** - `cropformer_config.ini` contains paths to input files,
model hyperparameters, and output directories. - All runtime messages
and errors are redirected to `output.log` for easier debugging.

------------------------------------------------------------------------

## 2. DPCformer

**Original tool:**\
Project page: https://anonymous.4open.science/r/DPCformer-0B5C/README.md

**Wrapper script:**\
`dpcformer_pipeline_v1.0.py`

**Description:**\
This wrapper runs DPCformer for genomic prediction using training and
testing genotype/phenotype datasets.

**How to run:**

``` bash
python dpcformer_pipeline_v1.0.py \
  --marker_train train_genotype.vcf \
  --pheno_train train_phenotype.csv \
  --marker_test test_genotype.vcf \
  --pheno_test test_phenotype.csv \
  --result_dir results_dpc
```

**Parameters:** - `--marker_train`: VCF file for training genotypes\
- `--pheno_train`: CSV file for training phenotypes\
- `--marker_test`: VCF file for testing genotypes\
- `--pheno_test`: CSV file for testing phenotypes\
- `--result_dir`: Directory for saving predictions and model outputs

------------------------------------------------------------------------

## 3. WheatGP

**Original tool:**\
GitHub: https://github.com/Breed-AI/WheatGP

**Wrapper script:**\
`wheatGP_pipeline_v1.0.py`

**Description:**\
This wrapper supports both training and prediction modes for the WheatGP
deep learning model.

**How to run (prediction mode):**

``` bash
python wheatGP_pipeline_v1.0.py \
  --mode prediction \
  --marker_train train_genotype.vcf \
  --pheno_train train_phenotype.csv \
  --marker_test test_genotype.vcf \
  --pheno_test test_phenotype.csv \
  --result_dir result_wheatGP
```

**Notes:** - The `--mode` option can be `training` or `prediction`. -
Output includes predicted breeding values and evaluation metrics.

------------------------------------------------------------------------

## 4. DeepGS

**Original tool:**\
GitHub: https://github.com/cma2015/DeepGS

**Wrapper script:**\
`deepGS_pipeline_v1.0.py` (to be provided)

**Description:**\
DeepGS is a deep learning-based genomic selection method using
convolutional neural networks.\
The wrapper will automate data formatting, model training, and
prediction.

**Planned command format:**

``` bash
python deepGS_pipeline_v1.0.py \
  --marker_train train_genotype.vcf \
  --pheno_train train_phenotype.csv \
  --marker_test test_genotype.vcf \
  --pheno_test test_phenotype.csv \
  --result_dir result_deepGS
```

------------------------------------------------------------------------

## General Notes

1.  **Input formats**
    -   Genotype: VCF or PLINK formats (tool-dependent)
    -   Phenotype: CSV with sample IDs matching genotype files
2.  **Output**
    -   Prediction results (CSV)
    -   Trained model checkpoints
    -   Logs and evaluation metrics
3.  **Reproducibility**
    -   Set random seeds in config files when available.
    -   Record software versions and GPU information in logs.
4.  **Troubleshooting**
    -   Check log files in the result directory.
    -   Verify Python, CUDA, and PyTorch versions match tool
        requirements.
    -   Ensure sample IDs are consistent between genotype and phenotype
        files.
