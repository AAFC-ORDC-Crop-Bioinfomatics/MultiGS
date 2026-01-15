#!/bin/bash

#SBATCH --job-name=DeepGS_CPU
#SBATCH --output=%x_%j.out
#SBATCH --cluster=gpsc8
#SBATCH --partition=standard
#SBATCH --account=aafc_aac
#SBATCH --time=100:00:00
#SBATCH --cpus-per-task=1
####SBATCH --mem=200G
#SBATCH --ntasks=1
#SBATCH --comment="registry.maze.science.gc.ca/ssc-hpcs/generic-job:ubuntu22.04,tmpfs_size=1000G"

deepgs_sif=/gpfs/fs7/aafc/labs/youf_lab/code/sharedTool/docker/deepgs.sif
test_path=/home/chz000/CZ_project2/thirdPartyGS/DeepGS

cd $test_path/results_cpu

singularity exec -B $test_path ${deepgs_sif}  Rscript  $test_path/DeepGS_Independent_Test.CNN.v5.1.tmp.R \
  --tr-snp=$test_path/maize/train_hybrid42k_maize_genotype.common.txt \
  --tr-phe=$test_path/maize/train_hybrid42k_maize_phenotype.common.csv \
  --te-snp=$test_path/maize/test_hybrid42k_maize_genotype.common.txt \
  --te-phe=$test_path/maize/test_hybrid42k_maize_phenotype.common.csv \
  --device=cpu \
  --num_round=1000 \
  --lr=0.01 \
  --array_batch_size = 30

conda activate multigs


python cropformer_pipeline.py -c cropformer_config.ini > output.log 2>&1
conda activate multigs


python dpcformer_pipeline_v1.0.py --marker_train train_genotype.vcf --pheno_train train_phenotype.csv  --marker_test test_genotype.vcf --pheno_test test_phenotype.csv  --result_dir results_dpc

conda activate multigs


python wheatGP_pipeline_v1.0.py --mode prediction --marker_train train_genotype.vcf --pheno_train train_phenotype.csv  --marker_test test_genotype.vcf --pheno_test test_phenotype.csv  --result_dir result_wheatGP
