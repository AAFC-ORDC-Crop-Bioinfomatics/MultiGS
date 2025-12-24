#!/bin/bash
#$ -N W_CV_MGSP
#$ -cwd
#$ -V
#$ -pe smp 32

export PATH=/isilon/ottawa-rdc/users/youf/backup_data/sharedTool/mmoe_env/bin:/isilon/ottawa-rdc/users/youf/backup_data/sharedTool/GS_new_pipelines/GS_java/java/bin:$PATH

python /isilon/projects/J-002035_flaxgenomics/J-001386Frank_Lab/Flax_GS_project_J-003426/github/MultiGS-P/src/MultiGS-P_2.0.py -c pc_cv_config.ini