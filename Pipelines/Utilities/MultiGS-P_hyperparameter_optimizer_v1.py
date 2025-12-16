#!/usr/bin/env python3
"""
MultiGS-P Hyperparameter Optimization Utility
Author: Frank YOU
Agriculture and Agri-Food Canada
"""

import os
import sys
import argparse
import itertools
import subprocess
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import configparser
import json
import shutil
import copy
import glob

class HyperparameterOptimizer:
    def __init__(self, base_config_path, model_name, mode='cv', output_dir='hyperparameter_results'):
        self.program_path = 'MultiGS-P_1.0.py'  # Default, overridden by user input
        self.base_config_path = base_config_path
        self.model_name = model_name
        self.mode = mode.lower()
        self.output_dir = output_dir
        self.base_config = self._load_base_config()
        
        # Define parameter spaces for each model
        self.parameter_spaces = self._define_parameter_spaces()
        
    def _load_base_config(self):
        """Load the base configuration file"""
        # Preserve case of option names (so model names are not lowercased)
        config = configparser.ConfigParser()
        config.optionxform = str  # <--- preserve original case
        config.read(self.base_config_path)
        return config
    
    def _deep_copy_config(self):
        """Create a deep copy of the config parser object"""
        new_config = configparser.ConfigParser()
        new_config.optionxform = str  # <--- preserve original case
        for section in self.base_config.sections():
            new_config.add_section(section)
            for key, value in self.base_config.items(section):
                new_config.set(section, key, value)
        return new_config
    
    def _define_parameter_spaces(self):
        """Define parameter search spaces for each model"""
        spaces = {
            'ElasticNet': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0],
                'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
            },
            'LASSO': {
                'alpha': [0.001, 0.01, 0.1, 1.0, 10.0]
            },
            'RandomForest': {
                'n_estimators': [100, 200, 500],
                'max_depth': [None, 10, 20, 30],
                'max_features': ['auto', 'sqrt', 'log2']
            },
            'XGBoost': {
                'n_estimators': [100, 200, 500],
                'max_depth': [3, 6, 9],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'LightGBM': {
                'n_estimators': [100, 200, 500],
                'num_leaves': [31, 63, 127],
                'learning_rate': [0.01, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            },
            'MLP': {
                'hidden_layers': ['128,64', '256,128', '512,256,128', '1024,512,256'],
                'learning_rate': [0.0001, 0.0005, 0.001],
                'batch_size': [16, 32, 64],
                'dropout': [0.2, 0.3, 0.5],
            },
            'CNN': {
                'hidden_channels': ['64,32', '128,64', '128,128,64', '256,128,64'],
                'kernel_size': [3, 5, 7],
                'learning_rate': [0.0001, 0.0005, 0.001],
                'batch_size': [16, 32, 64],
                'dropout': [0.2, 0.3, 0.5],
            },
            'HybridCNN': {
                'cnn_channels': ['64,32', '128,64', '128,128,64', '256,128,64'],
                'kernel_size': [3, 5, 7],
                'learning_rate': [0.0001, 0.0005, 0.001],
                'batch_size': [16, 32, 64],
                'dropout': [0.2, 0.3, 0.5],
                'attention_heads': [4, 8],
            },
            'Stacking': {
                'base_models': ['BRR,ElasticNet', 'BRR,MLP', 'ElasticNet,MLP', 'BRR,ElasticNet,MLP'],
                'meta_model': ['linear', 'ridge'],
                'meta_alpha': [0.1, 1.0, 10.0]
            }
        }
        
        if self.model_name not in spaces:
            raise ValueError(f"Model {self.model_name} not supported for hyperparameter optimization")
        
        return spaces[self.model_name]
    
    def generate_parameter_combinations(self):
        """Generate all possible parameter combinations"""
        keys = list(self.parameter_spaces.keys())
        values = list(self.parameter_spaces.values())
        
        combinations = []
        for combo in itertools.product(*values):
            param_dict = dict(zip(keys, combo))
            combinations.append(param_dict)
        
        print(f"Generated {len(combinations)} parameter combinations for {self.model_name}")
        return combinations
    
    def create_config_for_combination(self, combination, combo_id):
        """Create a new config file for a specific parameter combination"""
        config = self._deep_copy_config()
        
        # Create meaningful result directory name
        param_str_parts = []
        for k, v in combination.items():
            if k == 'hidden_layers' or k == 'hidden_channels':
                # Shorten layer descriptions
                v_str = v.replace(',', '-').replace(' ', '')
                param_str_parts.append(f"{k}_{v_str}")
            elif isinstance(v, float):
                param_str_parts.append(f"{k}_{v:.4f}".replace('.', 'p'))
            elif v is None:
                param_str_parts.append(f"{k}_None")
            else:
                param_str_parts.append(f"{k}_{v}")
        
        param_str = "_".join(param_str_parts)
        # Limit length to avoid filesystem issues
        if len(param_str) > 100:
            param_str = param_str[:100]
        
        result_dir = f"{self.output_dir}/{self.model_name}_combo_{combo_id:03d}_{param_str}"
        
        # Update results directory
        # Use the exact option name as in config (case preserved because optionxform=str)
        if not config.has_section('General'):
            config.add_section('General')
        config.set('General', 'results_dir', result_dir)
        
        # Disable all models except the target model
        # Use the same option keys you want in the generated config (case preserved)
        all_models = ['ElasticNet', 'LASSO', 'RandomForest', 'BRR', 'XGBoost', 
                     'LightGBM', 'CNN', 'HybridCNN', 'MLP', 'Stacking']
        
        if not config.has_section('Models'):
            config.add_section('Models')
        
        for model in all_models:
            # write True/False as strings (optionxform preserved)
            if model == self.model_name:
                config.set('Models', model, 'True')
            else:
                config.set('Models', model, 'False')

        # Skip invalid HybridCNN combinations
        if self.model_name == 'HybridCNN':
            hidden_size = int(config.get('Hyperparameters_HybridCNN', 'hidden_size', fallback='256'))
            heads = int(combination.get('attention_heads', 1))
            if hidden_size % heads != 0:
                print(f"Skipping invalid HybridCNN combo: hidden_size {hidden_size} not divisible by attention_heads {heads}")
                return None, None

        
        # Update hyperparameters for the target model
        section_name = f'Hyperparameters_{self.model_name}'
        if not config.has_section(section_name):
            config.add_section(section_name)
        
        for param, value in combination.items():
            config.set(section_name, param, str(value))
        
        # Save the new config file
        os.makedirs(self.output_dir, exist_ok=True)
        config_path = f"{self.output_dir}/config_{self.model_name}_combo_{combo_id:03d}.ini"
        with open(config_path, 'w') as f:
            config.write(f)
        
        return config_path, result_dir
    
    def run_pipeline(self, config_path):
        """Run the MultiGS-P pipeline with the given config"""
        try:
            # Use the same Python executable that's running this script
            python_exec = sys.executable
            
            program_to_run = self.program_path
            if program_to_run.endswith('.py'):
                cmd = [python_exec, program_to_run, '-c', config_path]
            else:
                cmd = [program_to_run, '-c', config_path]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=7200
            )

            
            if result.returncode == 0:
                return True, result.stdout, result.stderr
            else:
                print(f"Pipeline failed with return code: {result.returncode}")
                if result.stderr:
                    print(f"Stderr: {result.stderr[-500:]}")
                return False, result.stdout, result.stderr
                
        except subprocess.TimeoutExpired:
            print(f"Timeout for config: {config_path}")
            return False, "", "Timeout"
        except Exception as e:
            print(f"Error running pipeline: {e}")
            return False, "", str(e)
    
    def extract_results_cv(self, result_dir, trait):
        """Extract results from CV mode"""
        # Get feature view from base config and strip inline comments
        raw_feature_view = self.base_config.get('FeatureView', 'feature_view', fallback='SNP')
        # Remove inline comment after '#' or ';' and strip whitespace
        feature_view = raw_feature_view.split('#', 1)[0].split(';', 1)[0].strip()
        summary_file = f"{result_dir}/{feature_view}_gs_summary_stats.csv"
        
        if not os.path.exists(summary_file):
            print(f"Summary file not found: {summary_file}")
            return None
        
        try:
            df = pd.read_csv(summary_file)
            # Filter for the specific trait and model, and mean PearsonR
            filtered = df[
                (df['Trait'] == trait) & 
                (df['Model'] == self.model_name) & 
                (df['Metric'] == 'PearsonR') & 
                (df['Statistic'] == 'mean')
            ]
            
            if len(filtered) > 0:
                pearson_r = float(filtered['Value'].iloc[0])
                print(f"  Extracted Pearson R: {pearson_r:.4f}")
                return pearson_r
            else:
                print(f"  No matching results found for {trait} in summary file")
                return None
        except Exception as e:
            print(f"Error reading CV results: {e}")
            return None
    
    def extract_results_prediction(self, result_dir, trait):
        """Extract results from Prediction mode"""
        raw_feature_view = self.base_config.get('FeatureView', 'feature_view', fallback='SNP')
        feature_view = raw_feature_view.split('#', 1)[0].split(';', 1)[0].strip()
        trait_predictions_dir = f"{result_dir}/trait_predictions"
        
        if not os.path.exists(trait_predictions_dir):
            print(f"Trait predictions directory not found: {trait_predictions_dir}")
            return None
        
        # Look for the summary file with correct column name
        summary_file = f"{trait_predictions_dir}/prediction_summary_{feature_view}_trait_{trait}.csv"
        
        if not os.path.exists(summary_file):
            print(f"Prediction summary file not found: {summary_file}")
            return None
        
        try:
            df = pd.read_csv(summary_file)
            
            # Try different column name patterns
            possible_columns = [
                f'{self.model_name}_pearson_r',  # Correct format
                f'{self.model_name}_pearsonr',   # Old format
                'pearson_r',                     # Generic column
                'pearsonr'                       # Old generic
            ]
            
            pearson_r = None
            for col in possible_columns:
                if col in df.columns and not pd.isna(df[col].iloc[0]):
                    pearson_r = float(df[col].iloc[0])
                    print(f"  Extracted Pearson R from column '{col}': {pearson_r:.4f}")
                    break
            
            if pearson_r is None:
                print(f"  No Pearson R column found. Available columns: {list(df.columns)}")
                # Try to find any column with 'pearson' in the name
                pearson_cols = [col for col in df.columns if 'pearson' in col.lower()]
                if pearson_cols:
                    print(f"  Potential Pearson columns: {pearson_cols}")
                    for col in pearson_cols:
                        if not pd.isna(df[col].iloc[0]):
                            pearson_r = float(df[col].iloc[0])
                            print(f"  Using column '{col}': {pearson_r:.4f}")
                            break
            
            return pearson_r
            
        except Exception as e:
            print(f"Error reading prediction results: {e}")
            return None
    
    def get_trait_names(self):
        """Get trait names from phenotype file"""
        pheno_path = self.base_config.get('Data', 'phenotype_path')
        if not pheno_path:
            print("No phenotype path found in config")
            return ['Trait1']
            
        try:
            # Try different separators
            for sep in [',', '\t', ';', ' ']:
                try:
                    df = pd.read_csv(pheno_path, sep=sep, engine='python', index_col=0)
                    if len(df.columns) > 0:
                        traits = df.columns.tolist()
                        print(f"Found {len(traits)} traits: {', '.join(traits)}")
                        return traits
                except:
                    continue
            
            # If all separators fail, try without specifying separator
            df = pd.read_csv(pheno_path, engine='python', index_col=0)
            traits = df.columns.tolist()
            print(f"Found {len(traits)} traits: {', '.join(traits)}")
            return traits
            
        except Exception as e:
            print(f"Error reading phenotype file '{pheno_path}': {e}")
            return ['Trait1']  # Fallback
    
    def run_optimization(self, max_combinations=None):
        """Run the complete hyperparameter optimization"""
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get trait names
        traits = self.get_trait_names()
        
        # Generate parameter combinations
        combinations = self.generate_parameter_combinations()
        
        # Limit combinations if specified
        if max_combinations and len(combinations) > max_combinations:
            print(f"Limiting to first {max_combinations} combinations")
            combinations = combinations[:max_combinations]
        
        # Store results
        all_results = []
        
        print(f"\nStarting hyperparameter optimization for {self.model_name}")
        print(f"Total combinations to test: {len(combinations)}")
        print("=" * 80)
        
        successful_runs = 0
        failed_runs = 0
        results_extracted = 0
        
        for i, combo in enumerate(combinations):
            print(f"\n[{i+1}/{len(combinations)}] Running combination:")
            for param, value in combo.items():
                print(f"  {param}: {value}")
            
            # Create config for this combination
            config_path, result_dir = self.create_config_for_combination(combo, i)
            if config_path is None:
                continue

            # Run pipeline
            success, stdout, stderr = self.run_pipeline(config_path)
            
            if success:
                successful_runs += 1
                print("✓ Pipeline completed successfully")
                
                # Extract results for each trait
                combo_successful = False
                for trait in traits:
                    if self.mode == 'cv':
                        pearson_r = self.extract_results_cv(result_dir, trait)
                    else:  # prediction mode
                        pearson_r = self.extract_results_prediction(result_dir, trait)
                    
                    if pearson_r is not None:
                        result_entry = {
                            'combination_id': i,
                            'trait': trait,
                            'pearson_r': pearson_r,
                            'parameters': str(combo),
                            'parameters_dict': combo,  # Keep as dict for analysis
                            'result_dir': result_dir,
                            'config_file': config_path
                        }
                        all_results.append(result_entry)
                        combo_successful = True
                        results_extracted += 1
                    else:
                        print(f"  {trait}: Could not extract results")
                
                if combo_successful:
                    print(f"  ✓ Results extracted successfully")
                else:
                    failed_runs += 1
                    print("  ✗ No valid results extracted for any trait")
            else:
                failed_runs += 1
                print("✗ Pipeline failed")
                if stderr:
                    error_lines = stderr.split('\n')
                    # Print last few error lines
                    for line in error_lines[-10:]:
                        if line.strip():
                            print(f"  Error: {line}")
            
            # Clean up config file to save space
            try:
                os.remove(config_path)
            except:
                pass
            
            # Progress update
            print(f"Progress: {i+1}/{len(combinations)} | Successful: {successful_runs} | Results: {results_extracted} | Failed: {failed_runs}")
        
        print(f"\nOptimization completed: {successful_runs} successful runs, {results_extracted} results extracted, {failed_runs} failed")
        return all_results
    
    def analyze_results(self, results):
        """Analyze and visualize the optimization results"""
        if not results:
            print("No results to analyze")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(results)
        
        # Save detailed results
        results_file = f"{self.output_dir}/{self.model_name}_hyperparameter_results.csv"
        
        # Prepare data for CSV (convert dict to string for saving)
        df_csv = df.copy()
        df_csv['parameters'] = df_csv['parameters_dict'].apply(str)
        df_csv = df_csv.drop('parameters_dict', axis=1)
        df_csv.to_csv(results_file, index=False)
        
        print(f"\nDetailed results saved to: {results_file}")
        
        # Create visualizations for each trait
        traits = df['trait'].unique()
        
        for trait in traits:
            trait_df = df[df['trait'] == trait].copy()
            
            if len(trait_df) == 0:
                continue
            
            # Sort by Pearson R
            trait_df = trait_df.sort_values('pearson_r', ascending=False)
            
            # Create visualization
            self._create_visualization(trait_df, trait)
            
            # Print top combinations
            print(f"\nTop 5 combinations for {trait}:")
            top_combinations = trait_df.head(5)
            for idx, (_, row) in enumerate(top_combinations.iterrows()):
                print(f"  Rank {idx + 1}: Pearson R = {row['pearson_r']:.4f}")
                print(f"  Parameters: {row['parameters_dict']}")
                print(f"  Result dir: {os.path.basename(row['result_dir'])}")
                print()
        
        # Create summary statistics
        summary_stats = df.groupby('trait')['pearson_r'].agg(['max', 'min', 'mean', 'std', 'count'])
        summary_file = f"{self.output_dir}/{self.model_name}_summary_statistics.csv"
        summary_stats.to_csv(summary_file)
        print(f"Summary statistics saved to: {summary_file}")
        
        # Save best parameters for each trait
        best_parameters = {}
        for trait in traits:
            trait_results = df[df['trait'] == trait]
            if len(trait_results) > 0:
                best_idx = trait_results['pearson_r'].idxmax()
                best_parameters[trait] = {
                    'pearson_r': trait_results.loc[best_idx, 'pearson_r'],
                    'parameters': trait_results.loc[best_idx, 'parameters_dict'],
                    'result_dir': trait_results.loc[best_idx, 'result_dir']
                }
        
        best_params_file = f"{self.output_dir}/{self.model_name}_best_parameters.json"
        with open(best_params_file, 'w') as f:
            # Convert to JSON-serializable format
            json_serializable = {}
            for trait, params in best_parameters.items():
                json_serializable[trait] = {
                    'pearson_r': params['pearson_r'],
                    'parameters': {k: (str(v) if v is None else v) for k, v in params['parameters'].items()},
                    'result_dir': params['result_dir']
                }
            json.dump(json_serializable, f, indent=2)
        
        print(f"Best parameters saved to: {best_params_file}")
        
        return df
    
    def _create_visualization(self, trait_df, trait):
        """Create bar chart visualization for a trait"""
        plt.figure(figsize=(14, 8))
        
        # Use combination IDs for x-axis
        x_positions = range(len(trait_df))
        pearson_values = trait_df['pearson_r'].values
        
        # Create colormap based on performance
        colors = plt.cm.viridis((pearson_values - pearson_values.min()) / (pearson_values.max() - pearson_values.min()))
        
        bars = plt.bar(x_positions, pearson_values, alpha=0.8, color=colors)
        plt.xlabel('Parameter Combination (sorted by performance)')
        plt.ylabel('Pearson R')
        plt.title(f'Hyperparameter Optimization for {self.model_name} - {trait}\n'
                 f'Best: {pearson_values[0]:.4f}, Worst: {pearson_values[-1]:.4f}')
        plt.xticks(x_positions, [f'#{i+1}' for i in range(len(trait_df))], rotation=90, fontsize=8)
        
        # Add value labels on bars (only for top 10 and bottom 10 to avoid clutter)
        for i, (bar, value) in enumerate(zip(bars, pearson_values)):
            if i < 10 or i >= len(trait_df) - 10:  # Top 10 and bottom 10
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                        f'{value:.3f}', ha='center', va='bottom', fontsize=6)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        plot_file = f"{self.output_dir}/{self.model_name}_{trait}_optimization_plot.png"
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {plot_file}")

def main():
    parser = argparse.ArgumentParser(description='MultiGS-P Hyperparameter Optimization Utility')
    parser.add_argument('-p', '--program', required=False, default='MultiGS-P_1.0.py',
                    help='Path to MultiGS-P pipeline script or executable')    
    parser.add_argument('-c', '--config', required=True, help='Base configuration file path')
    parser.add_argument('-m', '--model', required=True, 
                       choices=['ElasticNet', 'LASSO', 'RandomForest', 'XGBoost', 
                               'LightGBM', 'CNN', 'HybridCNN','MLP', 'Stacking'],
                       help='Model to optimize')
    parser.add_argument('--mode', choices=['cv', 'prediction'], default='cv',
                       help='Operation mode: cv (cross-validation) or prediction')
    parser.add_argument('-o', '--output', default='hyperparameter_results',
                       help='Output directory for results')
    parser.add_argument('--max-combinations', type=int, 
                       help='Maximum number of combinations to test (for testing)')
    
    args = parser.parse_args()
    
    print("MultiGS-P Hyperparameter Optimization Utility")
    print("=" * 50)
    
    # Check if base config exists
    if not os.path.exists(args.config):
        print(f"Error: Configuration file '{args.config}' not found")
        sys.exit(1)
    
    # Check if pipeline script exists
    if not os.path.exists(args.program):
        print(f"Error: MultiGS-P pipeline not found at '{args.program}'")
        sys.exit(1)
 
    # Initialize optimizer
    optimizer = HyperparameterOptimizer(
        base_config_path=args.config,
        model_name=args.model,
        mode=args.mode,
        output_dir=args.output
    )
    # pass program path into optimizer instance
    optimizer.program_path = args.program
    
    # Run optimization
    results = optimizer.run_optimization(max_combinations=args.max_combinations)
    
    # Analyze results
    if results:
        optimizer.analyze_results(results)
        print(f"\nOptimization completed! Results saved in: {args.output}")
    else:
        print("\nNo valid results obtained. Please check your configuration and data.")

if __name__ == '__main__':
    main()
