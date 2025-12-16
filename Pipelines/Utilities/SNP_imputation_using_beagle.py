#!/usr/bin/env python3
"""
SNP Imputation Utility for Genomic Selection using Beagle
"""

import os
import sys
import argparse
import subprocess
import logging
from pathlib import Path
import pandas as pd
import numpy as np
from typing import Optional, Tuple, List
import tempfile
import shutil

class BeagleImputation:
    def __init__(self, beagle_path: str, java_path: str = "java", memory: str = "4g"):
        """
        Initialize Beagle imputation utility
        
        Args:
            beagle_path: Path to Beagle jar file
            java_path: Path to Java executable
            memory: Java heap memory allocation
        """
        self.beagle_path = beagle_path
        self.java_path = java_path
        self.memory = memory
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging configuration"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler('beagle_imputation.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
    
    def check_dependencies(self) -> bool:
        """Check if required dependencies are available"""
        try:
            # Check Java
            result = subprocess.run([self.java_path, "-version"], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                self.logger.error("Java not found or not working")
                return False
            
            # Check Beagle
            if not os.path.exists(self.beagle_path):
                self.logger.error(f"Beagle jar file not found: {self.beagle_path}")
                return False
                
            self.logger.info("All dependencies verified")
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking dependencies: {e}")
            return False
    
    def convert_to_vcf(self, input_file: str, output_vcf: str, 
                      format_type: str = "plink") -> bool:
        """
        Convert various formats to VCF for Beagle
        
        Args:
            input_file: Input file path
            output_vcf: Output VCF path
            format_type: Input format type ('plink', 'csv', 'hapmap')
        """
        try:
            if format_type == "plink":
                return self._plink_to_vcf(input_file, output_vcf)
            elif format_type == "csv":
                return self._csv_to_vcf(input_file, output_vcf)
            elif format_type == "hapmap":
                return self._hapmap_to_vcf(input_file, output_vcf)
            else:
                self.logger.error(f"Unsupported format: {format_type}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error converting to VCF: {e}")
            return False
    
    def _plink_to_vcf(self, prefix: str, output_vcf: str) -> bool:
        """Convert PLINK files to VCF using plink command"""
        try:
            cmd = [
                "plink", "--bfile", prefix, "--recode", "vcf", 
                "--out", output_vcf.replace(".vcf", "")
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info(f"Successfully converted PLINK to VCF: {output_vcf}")
                return True
            else:
                self.logger.error(f"PLINK conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error in PLINK conversion: {e}")
            return False
    
    def _csv_to_vcf(self, csv_file: str, output_vcf: str) -> bool:
        """Convert CSV genotype data to VCF format"""
        try:
            # This is a simplified example - adapt based on your CSV format
            df = pd.read_csv(csv_file)
            
            with open(output_vcf, 'w') as vcf:
                # Write VCF header
                vcf.write("##fileformat=VCFv4.2\n")
                vcf.write("#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT")
                
                # Add sample IDs (assuming first column is SNP info)
                samples = df.columns[1:]
                for sample in samples:
                    vcf.write(f"\t{sample}")
                vcf.write("\n")
                
                # Write genotype data
                for _, row in df.iterrows():
                    # Adapt this based on your CSV structure
                    chrom, pos, ref, alt = row[0].split('_')  # Example: chr1_12345_A_T
                    vcf.write(f"{chrom}\t{pos}\t.\t{ref}\t{alt}\t.\t.\t.\tGT")
                    
                    for genotype in row[1:]:
                        # Convert genotype to VCF format
                        gt = self._convert_genotype_to_vcf(genotype)
                        vcf.write(f"\t{gt}")
                    vcf.write("\n")
            
            self.logger.info(f"Successfully converted CSV to VCF: {output_vcf}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error in CSV conversion: {e}")
            return False
    
    def _hapmap_to_vcf(self, hapmap_file: str, output_vcf: str) -> bool:
        """Convert HapMap format to VCF"""
        try:
            # Implementation for HapMap conversion
            # This would need to be adapted based on specific HapMap format
            self.logger.info("HapMap conversion not yet implemented")
            return False
            
        except Exception as e:
            self.logger.error(f"Error in HapMap conversion: {e}")
            return False
    
    def _convert_genotype_to_vcf(self, genotype: str) -> str:
        """Convert genotype string to VCF format"""
        # Simple conversion - adapt based on your encoding
        if genotype == "0":
            return "0/0"
        elif genotype == "1":
            return "0/1"
        elif genotype == "2":
            return "1/1"
        else:
            return "./."  # Missing genotype
    
    def run_beagle(self, vcf_file: str, output_prefix: str, 
                  genetic_map: Optional[str] = None,
                  nthreads: int = 1, window: float = 40.0,
                  overlap: float = 4.0, ne: int = 1000000,
                  impute: bool = True, gt_prob: bool = True) -> bool:
        """
        Run Beagle imputation
        
        Args:
            vcf_file: Input VCF file
            output_prefix: Output file prefix
            genetic_map: Genetic map file (optional)
            nthreads: Number of threads
            window: Imputation window size in cM
            overlap: Overlap size in cM
            ne: Effective population size
            impute: Whether to perform imputation
            gt_prob: Whether to output genotype probabilities
        """
        try:
            cmd = [
                self.java_path, f"-Xmx{self.memory}", "-jar", self.beagle_path,
                f"gt={vcf_file}",
                f"out={output_prefix}",
                f"nthreads={nthreads}",
                f"window={window}",
                f"overlap={overlap}",
                f"ne={ne}",
                f"impute={str(impute).lower()}",
                f"gp={str(gt_prob).lower()}"
            ]
            
            if genetic_map:
                cmd.append(f"map={genetic_map}")
            
            self.logger.info(f"Running Beagle command: {' '.join(cmd)}")
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                self.logger.info("Beagle imputation completed successfully")
                self.logger.info(f"Output files: {output_prefix}.*")
                return True
            else:
                self.logger.error(f"Beagle imputation failed: {result.stderr}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error running Beagle: {e}")
            return False
    
    def validate_imputation(self, original_vcf: str, imputed_vcf: str) -> dict:
        """
        Validate imputation results
        
        Args:
            original_vcf: Original VCF file
            imputed_vcf: Imputed VCF file
            
        Returns:
            Dictionary with validation metrics
        """
        try:
            # Simple validation - count missing genotypes before and after
            orig_missing = self._count_missing_genotypes(original_vcf)
            imp_missing = self._count_missing_genotypes(imputed_vcf)
            
            metrics = {
                'original_missing_count': orig_missing,
                'imputed_missing_count': imp_missing,
                'imputed_snps': imp_missing - orig_missing,  # Negative if imputation reduced missing
                'reduction_percentage': ((orig_missing - imp_missing) / orig_missing * 100) if orig_missing > 0 else 0
            }
            
            self.logger.info(f"Imputation validation metrics: {metrics}")
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error validating imputation: {e}")
            return {}
    
    def _count_missing_genotypes(self, vcf_file: str) -> int:
        """Count missing genotypes in VCF file"""
        try:
            count = 0
            with open(vcf_file, 'r') as f:
                for line in f:
                    if not line.startswith('#'):
                        genotypes = line.strip().split('\t')[9:]  # Sample columns
                        count += sum(1 for gt in genotypes if './.' in gt or '.' in gt)
            return count
        except Exception as e:
            self.logger.error(f"Error counting missing genotypes: {e}")
            return 0

def main():
    parser = argparse.ArgumentParser(description='SNP Imputation using Beagle')
    parser.add_argument('--input', required=True, help='Input genotype file')
    parser.add_argument('--format', choices=['plink', 'csv', 'hapmap', 'vcf'], 
                       required=True, help='Input format')
    parser.add_argument('--beagle-jar', required=True, help='Path to Beagle jar file')
    parser.add_argument('--output-prefix', required=True, help='Output file prefix')
    parser.add_argument('--java-path', default='java', help='Path to Java executable')
    parser.add_argument('--memory', default='4g', help='Java heap memory')
    parser.add_argument('--genetic-map', help='Genetic map file')
    parser.add_argument('--threads', type=int, default=1, help='Number of threads')
    parser.add_argument('--window', type=float, default=40.0, help='Imputation window size')
    parser.add_argument('--overlap', type=float, default=4.0, help='Overlap size')
    parser.add_argument('--ne', type=int, default=1000000, help='Effective population size')
    
    args = parser.parse_args()
    
    # Initialize imputation utility
    imputer = BeagleImputation(
        beagle_path=args.beagle_jar,
        java_path=args.java_path,
        memory=args.memory
    )
    
    # Check dependencies
    if not imputer.check_dependencies():
        sys.exit(1)
    
    # Convert to VCF if needed
    if args.format != 'vcf':
        vcf_file = f"{args.output_prefix}_input.vcf"
        if not imputer.convert_to_vcf(args.input, vcf_file, args.format):
            sys.exit(1)
    else:
        vcf_file = args.input
    
    # Run Beagle imputation
    if imputer.run_beagle(
        vcf_file=vcf_file,
        output_prefix=args.output_prefix,
        genetic_map=args.genetic_map,
        nthreads=args.threads,
        window=args.window,
        overlap=args.overlap,
        ne=args.ne
    ):
        # Validate results
        imputed_vcf = f"{args.output_prefix}.vcf.gz"
        metrics = imputer.validate_imputation(vcf_file, imputed_vcf)
        
        print("\nImputation completed successfully!")
        print(f"Output files: {args.output_prefix}.*")
        print(f"Validation metrics: {metrics}")
    else:
        sys.exit(1)

if __name__ == "__main__":
    main()

