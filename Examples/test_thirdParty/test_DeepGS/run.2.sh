Rscript ../DeepGS_Independent_Test.CNN.v5.1.tmp.R \
  --tr-snp=train_genotype.txt \
  --tr-phe=train_phenotype.csv \
  --te-snp=test_genotype.txt \
  --te-phe=test_phenotype.csv \
  --device=cpu \
  --num_round=1000 \
  --lr=0.01 \
  --array_batch_size=16
