#$ -cwd
#$ -pe smp 7

export PATH=/isilon/ottawa-rdc/users/youf/backup_data/sharedTool/GS_new_pipelines/GS_java/java/bin:$PATH
java -version
R --version

src=/isilon/projects/J-002035_flaxgenomics/J-001386Frank_Lab/Flax_GS_project_J-003426/github/MultiGS/Pipelines/MultiGS-R
java -Xmx20g -jar $src/MultiGS-R-1.0.jar  MultiGS-R_config_prediction2.ini
