#!/bin/bash
#$ -N f_p_test
#$ -cwd
#$ -V
#$ -pe smp 16

export PATH=/isilon/ottawa-rdc/users/youf/backup_data/sharedTool/mmoe_env/bin:/isilon/ottawa-rdc/users/youf/backup_data/sharedTool/GS_new_pipelines/GS_java/java/bin:$PATH

python /isilon/projects/J-002035_flaxgenomics/J-001386Frank_Lab/Flax_GS_project_J-003426/github/MultiGS-P/src/MultiGS-P_1.0.py -c config.ini