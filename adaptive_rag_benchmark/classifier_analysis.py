#!/usr/bin/env python3
"""
Classifier Performance Analysis

This script analyzes all classifier models trained on silver-labeled data,
extracting accuracy scores and per-class metrics for comparison across
different configurations.
"""

import json
import os
import pandas as pd
from pathlib import Path
from datetime import datetime
import re
from typing import Dict, List, Any, Tuple


def parse_model_directory_name(dir_name: str) -> Dict[str, str]:
    """
    Parse model directory name to extract configuration details.
    Format: {id}_{base_model}_{strategy}_{epochs}ep
    """
    try:
        # Split by underscores
        parts = dir_name.split('_')
        
        if len(parts) < 4:
            return {'error': f'Invalid directory format: {dir_name}'}
        
        model_id = parts[0]
        
        # Find strategy and epochs by looking for known patterns
        strategy = None
        epochs = None
        strategy_idx = -1
        
        # Look for strategy in parts
        for i, part in enumerate(parts):
            if part in ['optimized', 'original']:
                strategy = part
                strategy_idx = i
                # Next part should be epochs
                if i < len(parts) - 1:
                    epoch_part = parts[i+1]
                    if epoch_part.endswith('ep'):
                        epochs = epoch_part[:-2]  # Remove 'ep'
                break
        
        if not strategy or not epochs or strategy_idx == -1:
            return {'error': f'Could not parse strategy/epochs from: {dir_name}'}
        
        # Everything between model_id and strategy is base_model
        base_model_parts = parts[1:strategy_idx]
        base_model = '_'.join(base_model_parts) if base_model_parts else 'unknown'
        
        # Handle models with dashes by replacing underscores with dashes in known patterns
        if 'qwen' in base_model.lower():
            base_model = base_model.replace('_', '-')
        elif 'gemini' in base_model.lower():
            base_model = base_model.replace('_', '-')
        
        return {
            'model_id': model_id,
            'base_model': base_model,
            'strategy': strategy,
            'epochs': epochs,
            'full_name': dir_name
        }
    except Exception as e:
        return {'error': f'Error parsing {dir_name}: {e}'}


def load_classifier_results(base_dirs: List[str]) -> List[Dict[str, Any]]:
    """
    Load classifier results from all models in the specified directories.
    """
    all_results = []
    
    for base_dir in base_dirs:
        if not os.path.exists(base_dir):
            print(f"WARNING: Directory not found: {base_dir}")
            continue
        
        classifier_type = os.path.basename(base_dir)  # bert-large or t5-large
        print(f"\nProcessing {classifier_type} models...")
        
        model_dirs = [d for d in os.listdir(base_dir) 
                     if os.path.isdir(os.path.join(base_dir, d))]
        
        print(f"Found {len(model_dirs)} model directories")
        
        for model_dir in sorted(model_dirs):
            print(f"  {model_dir}")
            
            # Parse directory name
            config = parse_model_directory_name(model_dir)
            if 'error' in config:
                print(f"    ERROR: {config['error']}")
                continue
                
            # Build paths
            model_path = os.path.join(base_dir, model_dir)
            validation_path = os.path.join(model_path, 'validation')
            
            if not os.path.exists(validation_path):
                print(f"    ERROR: No validation directory found")
                continue
            
            # Load final results
            final_results_path = os.path.join(validation_path, 'final_eval_results.json')
            per_class_path = os.path.join(validation_path, 'final_eval_results_perClass.json')
            
            if not os.path.exists(final_results_path):
                print(f"    ERROR: No final_eval_results.json found")
                continue
            
            try:
                # Load overall accuracy
                with open(final_results_path, 'r') as f:
                    final_results = json.load(f)
                
                # Load per-class metrics
                per_class_metrics = {}
                if os.path.exists(per_class_path):
                    with open(per_class_path, 'r') as f:
                        per_class_metrics = json.load(f)
                
                # Combine all data
                result = {
                    'classifier_type': classifier_type,
                    'model_id': config['model_id'],
                    'base_model': config['base_model'],
                    'strategy': config['strategy'],
                    'epochs': int(config['epochs']),
                    'full_name': config['full_name'],
                    'final_accuracy': final_results.get('final_acc_score', 0.0),
                    'model_path': model_path
                }
                
                # Add per-class metrics
                if per_class_metrics:
                    result.update({
                        'label_a_acc': per_class_metrics.get('A (zero) acc', 0.0),
                        'label_b_acc': per_class_metrics.get('B (single) acc', 0.0),
                        'label_c_acc': per_class_metrics.get('C (multi) acc', 0.0),
                        'label_a_pred': per_class_metrics.get('A (zero) pred num', 0),
                        'label_b_pred': per_class_metrics.get('B (single) pred num', 0),
                        'label_c_pred': per_class_metrics.get('C (multi) pred num', 0),
                        'label_a_gold': per_class_metrics.get('A (zero) gold num', 0),
                        'label_b_gold': per_class_metrics.get('B (single) gold num', 0),
                        'label_c_gold': per_class_metrics.get('C (multi) gold num', 0)
                    })
                
                all_results.append(result)
                print(f"    Accuracy: {result['final_accuracy']:.2f}%")
                
            except Exception as e:
                print(f"    ERROR: Error loading results: {e}")
                continue
    
    return all_results


def create_comparison_analysis(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    Create comprehensive comparison analysis.
    """
    df = pd.DataFrame(results)
    
    if df.empty:
        print("WARNING: No results to analyze")
        return df
    
    print(f"\nCLASSIFIER PERFORMANCE ANALYSIS")
    print("=" * 70)
    print(f"Total models analyzed: {len(df)}")
    
    # Overall statistics
    print(f"\nOVERALL PERFORMANCE:")
    print(f"Best accuracy: {df['final_accuracy'].max():.2f}% ({df.loc[df['final_accuracy'].idxmax(), 'full_name']})")
    print(f"Worst accuracy: {df['final_accuracy'].min():.2f}% ({df.loc[df['final_accuracy'].idxmin(), 'full_name']})")
    print(f"Average accuracy: {df['final_accuracy'].mean():.2f}%")
    print(f"Std deviation: {df['final_accuracy'].std():.2f}%")
    
    # Analysis by classifier type
    print(f"\nBY CLASSIFIER TYPE:")
    classifier_stats = df.groupby('classifier_type')['final_accuracy'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
    for classifier_type, stats in classifier_stats.iterrows():
        print(f"  {classifier_type}: {stats['mean']:.2f}% ±{stats['std']:.2f}% (n={stats['count']}, range: {stats['min']:.1f}%-{stats['max']:.1f}%)")
    
    # Analysis by strategy
    print(f"\nBY STRATEGY:")
    strategy_stats = df.groupby('strategy')['final_accuracy'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
    for strategy, stats in strategy_stats.iterrows():
        print(f"  {strategy}: {stats['mean']:.2f}% ±{stats['std']:.2f}% (n={stats['count']}, range: {stats['min']:.1f}%-{stats['max']:.1f}%)")
    
    # Analysis by base model
    print(f"\nBY BASE MODEL:")
    base_model_stats = df.groupby('base_model')['final_accuracy'].agg(['mean', 'std', 'min', 'max', 'count']).round(2)
    for base_model, stats in base_model_stats.iterrows():
        print(f"  {base_model}: {stats['mean']:.2f}% ±{stats['std']:.2f}% (n={stats['count']}, range: {stats['min']:.1f}%-{stats['max']:.1f}%)")
    
    # Strategy comparison within each classifier type
    print(f"\nSTRATEGY COMPARISON BY CLASSIFIER:")
    for classifier_type in df['classifier_type'].unique():
        print(f"\n  {classifier_type.upper()}:")
        subset = df[df['classifier_type'] == classifier_type]
        if 'optimized' in subset['strategy'].values and 'original' in subset['strategy'].values:
            opt_mean = subset[subset['strategy'] == 'optimized']['final_accuracy'].mean()
            orig_mean = subset[subset['strategy'] == 'original']['final_accuracy'].mean()
            diff = opt_mean - orig_mean
            print(f"    Optimized: {opt_mean:.2f}%")
            print(f"    Original:  {orig_mean:.2f}%")
            print(f"    Difference: {diff:+.2f}pp ({'optimized' if diff > 0 else 'original'} performs better)")
        else:
            strategy_means = subset.groupby('strategy')['final_accuracy'].mean()
            for strategy, mean_acc in strategy_means.items():
                print(f"    {strategy}: {mean_acc:.2f}%")
    
    # Per-class performance analysis
    if 'label_a_acc' in df.columns:
        print(f"\nPER-CLASS PERFORMANCE:")
        label_cols = ['label_a_acc', 'label_b_acc', 'label_c_acc']
        for col in label_cols:
            if col in df.columns:
                label = col.split('_')[1].upper()
                mean_acc = df[col].mean()
                std_acc = df[col].std()
                print(f"  Label {label}: {mean_acc:.2f}% ±{std_acc:.2f}%")
    
    # All performers
    print(f"\nALL MODELS BY PERFORMANCE:")
    all_sorted = df.sort_values('final_accuracy', ascending=False)[['full_name', 'classifier_type', 'strategy', 'base_model', 'final_accuracy']]
    for idx, row in all_sorted.iterrows():
        print(f"  {row['final_accuracy']:5.2f}% - {row['classifier_type']} | {row['base_model']} | {row['strategy']}")
    
    return df


def save_analysis_results(df: pd.DataFrame, output_dir: str):
    """
    Save analysis results to multiple formats.
    """
    if df.empty:
        print("WARNING: No data to save")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_filename = f"classifier_analysis_{timestamp}"
    
    # Save as CSV
    csv_path = os.path.join(output_dir, f"{base_filename}.csv")
    df.to_csv(csv_path, index=False)
    print(f"\nSaved CSV: {csv_path}")
    
    # Save as JSON
    json_path = os.path.join(output_dir, f"{base_filename}.json")
    df.to_json(json_path, orient='records', indent=2)
    print(f"Saved JSON: {json_path}")
    
    # Create detailed markdown report
    md_path = os.path.join(output_dir, f"{base_filename}_report.md")
    create_markdown_report(df, md_path)
    print(f"Saved Markdown: {md_path}")
    
    return csv_path, json_path, md_path


def create_markdown_report(df: pd.DataFrame, output_path: str):
    """
    Create a detailed markdown report.
    """
    with open(output_path, 'w') as f:
        f.write("# Classifier Performance Analysis Report\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Summary
        f.write("## Executive Summary\n\n")
        f.write(f"- **Total Models Analyzed**: {len(df)}\n")
        f.write(f"- **Best Performance**: {df['final_accuracy'].max():.2f}% ({df.loc[df['final_accuracy'].idxmax(), 'full_name']})\n")
        f.write(f"- **Average Performance**: {df['final_accuracy'].mean():.2f}% ±{df['final_accuracy'].std():.2f}%\n")
        f.write(f"- **Performance Range**: {df['final_accuracy'].min():.2f}% - {df['final_accuracy'].max():.2f}%\n\n")
        
        # Classifier comparison
        f.write("## Performance by Classifier Type\n\n")
        classifier_stats = df.groupby('classifier_type')['final_accuracy'].agg(['mean', 'std', 'count']).round(2)
        f.write(classifier_stats.to_markdown())
        f.write("\n\n")
        
        # Strategy comparison
        f.write("## Performance by Strategy\n\n")
        strategy_stats = df.groupby('strategy')['final_accuracy'].agg(['mean', 'std', 'count']).round(2)
        f.write(strategy_stats.to_markdown())
        f.write("\n\n")
        
        # Base model comparison
        f.write("## Performance by Base Model\n\n")
        base_model_stats = df.groupby('base_model')['final_accuracy'].agg(['mean', 'std', 'count']).round(2)
        f.write(base_model_stats.to_markdown())
        f.write("\n\n")
        
        # Detailed results table
        f.write("## Detailed Results\n\n")
        display_cols = ['classifier_type', 'base_model', 'strategy', 'epochs', 'final_accuracy']
        if 'label_a_acc' in df.columns:
            display_cols.extend(['label_a_acc', 'label_b_acc', 'label_c_acc'])
        
        display_df = df[display_cols].round(2).sort_values('final_accuracy', ascending=False)
        f.write(display_df.to_markdown(index=False))
        f.write("\n\n")
        
        # All performers sorted by accuracy
        f.write("## All Models by Performance\n\n")
        all_sorted = df.sort_values('final_accuracy', ascending=False)[['full_name', 'final_accuracy', 'classifier_type', 'strategy', 'base_model']]
        f.write(all_sorted.to_markdown(index=False))
        f.write("\n\n")


def main():
    """
    Main analysis function.
    """
    print("Starting Classifier Performance Analysis...")
    
    # Define directories to analyze
    project_root = os.environ.get('PROJECT_ROOT', '')
    base_dirs = [
        f"{project_root}/Adaptive-RAG/classifier/outputs/bert-large",
        f"{project_root}/Adaptive-RAG/classifier/outputs/t5-large"
    ]
    
    # Load results
    results = load_classifier_results(base_dirs)
    
    if not results:
        print("ERROR: No results found to analyze")
        return None
    
    # Create analysis
    df = create_comparison_analysis(results)
    
    # Save results
    output_dir = f"{project_root}/Adaptive-RAG/adaptive_rag_benchmark/results"
    save_analysis_results(df, output_dir)
    
    print(f"\nAnalysis complete! Results saved to {output_dir}")
    return df


if __name__ == "__main__":
    df = main()
