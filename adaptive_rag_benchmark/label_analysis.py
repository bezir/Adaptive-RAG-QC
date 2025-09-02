#!/usr/bin/env python3
"""
Label Distribution Analysis

Single comprehensive script that analyzes label distribution differences between 
optimized and original strategies for each dataset, with proper handling of 
missing data and clear percentage difference calculations.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime

def robust_label_analysis():
    """
    Perform robust label distribution analysis with proper data handling.
    """
    print("ROBUST LABEL DISTRIBUTION ANALYSIS")
    print("=" * 70)
    print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Read the CSV data
    project_root = os.environ.get('PROJECT_ROOT', '')
    csv_path = f"{project_root}/Adaptive-RAG/adaptive_rag_benchmark/results/silver_labeling_statistics_20250828_130738.csv"
    df = pd.read_csv(csv_path)
    
    # Clean the data - replace NaN with 0 for label percentages
    label_cols = ['label_A_pct', 'label_B_pct', 'label_C_pct']
    for col in label_cols:
        df[col] = df[col].fillna(0.0)
    
    # Get unique datasets
    datasets = sorted(df['dataset_name'].unique())
    
    results = []
    
    print("DATASET-BY-DATASET COMPARISON\n")
    
    for dataset in datasets:
        print(f"TARGET: {dataset.upper()}")
        print("-" * 50)
        
        # Filter data for this dataset
        dataset_data = df[df['dataset_name'] == dataset].copy()
        
        # Calculate averages by strategy (across both models)
        strategy_summary = dataset_data.groupby('strategy_type')[label_cols].mean().round(1)
        
        if len(strategy_summary) == 2:  # Both strategies present
            opt = strategy_summary.loc['optimized']
            orig = strategy_summary.loc['original']
            
            # Display current distributions
            print(f"Optimized: A={opt['label_A_pct']:5.1f}%  B={opt['label_B_pct']:5.1f}%  C={opt['label_C_pct']:5.1f}%")
            print(f"Original:  A={orig['label_A_pct']:5.1f}%  B={orig['label_B_pct']:5.1f}%  C={orig['label_C_pct']:5.1f}%")
            
            # Calculate differences (optimized - original)
            diff_A = opt['label_A_pct'] - orig['label_A_pct']
            diff_B = opt['label_B_pct'] - orig['label_B_pct']
            diff_C = opt['label_C_pct'] - orig['label_C_pct']
            
            # Format difference display
            print(f"Difference: A={diff_A:+5.1f}pp B={diff_B:+5.1f}pp C={diff_C:+5.1f}pp")
            
            # Interpret the differences
            interpretations = []
            if abs(diff_A) >= 1.0:
                direction = "higher" if diff_A > 0 else "lower"
                interpretations.append(f"Label A {abs(diff_A):.1f}pp {direction}")
            if abs(diff_B) >= 1.0:
                direction = "higher" if diff_B > 0 else "lower"
                interpretations.append(f"Label B {abs(diff_B):.1f}pp {direction}")
            if abs(diff_C) >= 1.0:
                direction = "higher" if diff_C > 0 else "lower"
                interpretations.append(f"Label C {abs(diff_C):.1f}pp {direction}")
            
            if interpretations:
                print(f"Key change: {', '.join(interpretations)} in optimized")
            else:
                print("Key change: Minimal differences (<1pp)")
            
            # Store results
            results.append({
                'dataset': dataset,
                'opt_A': opt['label_A_pct'],
                'opt_B': opt['label_B_pct'],
                'opt_C': opt['label_C_pct'],
                'orig_A': orig['label_A_pct'],
                'orig_B': orig['label_B_pct'],
                'orig_C': orig['label_C_pct'],
                'diff_A': diff_A,
                'diff_B': diff_B,
                'diff_C': diff_C
            })
        else:
            print("WARNING: Missing strategy data for comparison")
        
        print()
    
    # Overall analysis
    print("=" * 70)
    print("\nOVERALL PATTERNS SUMMARY\n")
    
    if results:
        results_df = pd.DataFrame(results)
        
        # Calculate averages
        avg_diffs = {
            'A': results_df['diff_A'].mean(),
            'B': results_df['diff_B'].mean(),
            'C': results_df['diff_C'].mean()
        }
        
        print("Average differences across all datasets:")
        for label, avg_diff in avg_diffs.items():
            direction = "higher" if avg_diff > 0 else "lower"
            print(f"  Label {label}: {avg_diff:+.1f}pp ({abs(avg_diff):.1f}pp {direction} in optimized)")
        
        print(f"\nMOST SIGNIFICANT CHANGES:\n")
        
        # Find datasets with largest absolute changes for each label
        max_changes = {}
        for label in ['A', 'B', 'C']:
            col = f'diff_{label}'
            max_idx = results_df[col].abs().idxmax()
            max_row = results_df.loc[max_idx]
            max_changes[label] = {
                'dataset': max_row['dataset'],
                'diff': max_row[col],
                'abs_diff': abs(max_row[col])
            }
        
        for label, info in max_changes.items():
            if info['abs_diff'] >= 1.0:
                direction = "higher" if info['diff'] > 0 else "lower"
                print(f"Label {label}: {info['dataset']} ({info['diff']:+.1f}pp - {direction} in optimized)")
        
        # Dataset type patterns
        print(f"\nDATASET TYPE INSIGHTS:\n")
        
        # Identify single-hop vs multi-hop patterns
        single_hop_datasets = ['nq', 'squad', 'trivia']
        multi_hop_datasets = ['2wikimultihopqa', 'hotpotqa', 'musique']
        
        single_hop_results = results_df[results_df['dataset'].isin(single_hop_datasets)]
        multi_hop_results = results_df[results_df['dataset'].isin(multi_hop_datasets)]
        
        if not single_hop_results.empty:
            sh_avg_A = single_hop_results['diff_A'].mean()
            sh_avg_B = single_hop_results['diff_B'].mean()
            sh_avg_C = single_hop_results['diff_C'].mean()
            print(f"Single-hop datasets: A={sh_avg_A:+.1f}pp, B={sh_avg_B:+.1f}pp, C={sh_avg_C:+.1f}pp")
        
        if not multi_hop_results.empty:
            mh_avg_A = multi_hop_results['diff_A'].mean()
            mh_avg_B = multi_hop_results['diff_B'].mean()
            mh_avg_C = multi_hop_results['diff_C'].mean()
            print(f"Multi-hop datasets:  A={mh_avg_A:+.1f}pp, B={mh_avg_B:+.1f}pp, C={mh_avg_C:+.1f}pp")
        
        # Key insights
        print(f"\nKEY INSIGHTS:\n")
        
        if avg_diffs['B'] > 5:
            print(f"• Label B shows strong increase (+{avg_diffs['B']:.1f}pp) in optimized strategy")
            print("  → Suggests optimized approach generates more intermediate confidence levels")
        
        if avg_diffs['C'] > 2:
            print(f"• Label C moderately increases (+{avg_diffs['C']:.1f}pp) in optimized strategy")
            print("  → Indicates higher confidence/success rates with streamlined systems")
        elif avg_diffs['C'] < -2:
            print(f"• Label C decreases ({avg_diffs['C']:.1f}pp) in optimized strategy")
            print("  → Suggests trade-off in highest confidence outputs")
        
        if any(abs(avg_diffs[label]) < 1 for label in ['A', 'B', 'C']):
            stable_labels = [label for label in ['A', 'B', 'C'] if abs(avg_diffs[label]) < 1]
            print(f"• Labels {', '.join(stable_labels)} remain relatively stable (<1pp change)")
            print("  → Shows consistent behavior across strategies for these confidence levels")
    
    print("\n" + "=" * 70)
    print("Analysis complete!")
    
    return results_df if results else None

if __name__ == "__main__":
    results = robust_label_analysis()
