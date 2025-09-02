#!/usr/bin/env python3
"""
Adaptive RAG Benchmark Results Generator

Usage: python generate_results_report.py <experiment_folder_path>

This script scans an experiment folder for log files and generates a comprehensive
report with all results ranked by accuracy.
"""

import os
import re
import sys
from collections import defaultdict
from pathlib import Path

def parse_log_file(log_file_path):
    """Extract scores from a log file"""
    try:
        with open(log_file_path, 'r') as f:
            content = f.read()
        
        # Look for the aggregated results section
        f1_match = re.search(r'Average F1 Score:\s*([0-9.]+)', content)
        em_match = re.search(r'Average Exact Match:\s*([0-9.]+)', content)
        acc_match = re.search(r'Average Accuracy:\s*([0-9.]+)', content)
        steps_match = re.search(r'Average Steps:\s*([0-9.]+)', content)
        qps_match = re.search(r'Overall Performance:\s*([0-9.]+)\s*queries/second', content)
        
        # Also look for per-dataset summary if available
        dataset_match = re.search(r'Dataset\s+F1\s+EM\s+Acc\s+Steps\s+Q/s\s+Time\s*\n-+\n(\w+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)', content)
        
        if f1_match and em_match and acc_match and steps_match:
            result = {
                'experiment': os.path.basename(os.path.dirname(log_file_path)),
                'f1_score': float(f1_match.group(1)),
                'exact_match': float(em_match.group(1)),
                'accuracy': float(acc_match.group(1)),
                'avg_steps': float(steps_match.group(1)),
                'queries_per_second': float(qps_match.group(1)) if qps_match else 0.0,
                'dataset': dataset_match.group(1) if dataset_match else 'unknown'
            }
            return result
    except Exception as e:
        print(f"Warning: Error processing {log_file_path}: {e}")
    
    return None

def categorize_experiment(exp_name):
    """Categorize experiment type and extract details"""
    
    # Parse verification experiments (enhanced_verification_bert or enhanced_verification_flan)
    if exp_name.startswith('enhanced_verification_bert') or exp_name.startswith('enhanced_verification_flan'):
        # Pattern: enhanced_verification_flan_gemini_1_5_flash_8b_{classifier_id}_{model-short}_{training}_{epochs}_{dataset}
        # Extract model
        if 'gemini_1_5_flash_8b' in exp_name or 'gemini.1.5.flash.8b' in exp_name or 'gemini-1.5-8b' in exp_name:
            model = 'Gemini 1.5 Flash 8B'
        elif 'gemini_2_5_flash_lite' in exp_name or 'gemini.2.5.flash.lite' in exp_name or 'gemini-2.5-flash-lite' in exp_name:
            model = 'Gemini 2.5 Flash Lite'
        else:
            model = 'Unknown'
        
        # Extract classifier ID (should be the 4+ digit number, not model version numbers)
        classifier_match = re.search(r'_(\d{4,})_', exp_name)
        classifier_id = classifier_match.group(1) if classifier_match else 'unknown'
        
        # Categorize classifier size based on ID ranges
        if model == 'Gemini 1.5 Flash 8B':
            if classifier_id in ['1920', '1974']:
                classifier_type = 'Small'
            elif classifier_id in ['9603', '9874']:
                classifier_type = 'Medium'
            elif classifier_id in ['19206', '19749']:
                classifier_type = 'Large'
            else:
                classifier_type = 'Unknown'
        else:  # Gemini 2.5 Flash Lite
            if classifier_id in ['1681', '1794']:
                classifier_type = 'Small'
            elif classifier_id in ['8407', '8971']:
                classifier_type = 'Medium'
            elif classifier_id in ['16815', '17943']:
                classifier_type = 'Large'
            else:
                classifier_type = 'Unknown'
        
        # Extract training type and epochs
        if 'optimized' in exp_name:
            if '20ep' in exp_name:
                training = 'Optimized (20ep)'
            elif '3ep' in exp_name:
                training = 'Optimized (3ep)'
            else:
                training = 'Optimized'
        elif 'original' in exp_name:
            if '20ep' in exp_name:
                training = 'Original (20ep)'
            elif '3ep' in exp_name:
                training = 'Original (3ep)'
            else:
                training = 'Original'
        else:
            training = 'Unknown'
            
        # Determine classifier name based on experiment type
        if exp_name.startswith('enhanced_verification_bert'):
            classifier_name = 'BERT'
        elif exp_name.startswith('enhanced_verification_flan'):
            classifier_name = 'Flan-T5'
        else:
            classifier_name = 'Unknown'
        
        # Format training info: "Size, Method (epochs)"
        training_info = f"{classifier_type}, {training}"
        
        # Treat verification experiments as regular Adaptive RAG (verification shown in columns)
        full_name = f'Adaptive RAG ({model}, {classifier_name}-{classifier_type}, {training})'
        return {
            'approach': 'Adaptive RAG',
            'model': model,
            'classifier_type': classifier_name,
            'classifier_id': classifier_id,
            'training': training_info,
            'full_name': full_name,
            'is_verification': True  # Flag to identify verification experiments
        }
    
    # Parse adaptive_rag_bert and adaptive_rag_flan experiments (ML with classifiers)
    elif exp_name.startswith('adaptive_rag_bert') or exp_name.startswith('adaptive_rag_flan'):
        # Pattern: adaptive_rag_bert_{model}_{classifier_id}_{model_short}_{training}_{epochs}_{dataset}
        parts = exp_name.split('_')
        
        # Extract model
        if 'gemini.1.5.flash.8b' in exp_name or 'gemini_1.5_flash_8b' in exp_name:
            model = 'Gemini 1.5 Flash 8B'
        elif 'gemini.2.5.flash.lite' in exp_name or 'gemini_2.5_flash_lite' in exp_name:
            model = 'Gemini 2.5 Flash Lite'
        else:
            model = 'Unknown'
        
        # Extract classifier ID (should be the 4+ digit number, not model version numbers)
        classifier_match = re.search(r'_(\d{4,})_', exp_name)
        classifier_id = classifier_match.group(1) if classifier_match else 'unknown'
        
        # Categorize classifier size based on ID ranges
        if model == 'Gemini 1.5 Flash 8B':
            if classifier_id in ['1920', '1974']:
                classifier_type = 'Small'
            elif classifier_id in ['9603', '9874']:
                classifier_type = 'Medium'
            elif classifier_id in ['19206', '19749']:
                classifier_type = 'Large'
            else:
                classifier_type = 'Unknown'
        else:  # Gemini 2.5 Flash Lite
            if classifier_id in ['1681', '1794']:
                classifier_type = 'Small'
            elif classifier_id in ['8407', '8971']:
                classifier_type = 'Medium'
            elif classifier_id in ['16815', '17943']:
                classifier_type = 'Large'
            else:
                classifier_type = 'Unknown'
        
        # Extract training type and epochs
        if 'optimized' in exp_name:
            if '20ep' in exp_name:
                training = 'Optimized (20ep)'
            elif '3ep' in exp_name:
                training = 'Optimized (3ep)'
            else:
                training = 'Optimized'
        elif 'original' in exp_name:
            if '20ep' in exp_name:
                training = 'Original (20ep)'
            elif '3ep' in exp_name:
                training = 'Original (3ep)'
            else:
                training = 'Original'
        else:
            training = 'Unknown'
            
        # Determine classifier name based on experiment type
        if exp_name.startswith('adaptive_rag_bert'):
            classifier_name = 'BERT'
        elif exp_name.startswith('adaptive_rag_flan'):
            classifier_name = 'Flan-T5'
        else:
            classifier_name = 'Unknown'
        
        # Format training info: "Size, Method (epochs)"
        training_info = f"{classifier_type}, {training}"
        
        full_name = f'Adaptive RAG ({model}, {classifier_name}-{classifier_type}, {training})'
        return {
            'approach': 'Adaptive RAG',
            'model': model,
            'classifier_type': classifier_name,
            'classifier_id': classifier_id,
            'training': training_info,
            'full_name': full_name,
            'is_verification': False  # Flag to identify regular experiments
        }
    
    # Parse adaptive_rag_llm experiments (pure LLM without classifiers)
    elif exp_name.startswith('adaptive_rag_llm'):
        # Pattern: adaptive_rag_llm_{model}_{dataset}
        if 'gemini_1.5_flash_8b' in exp_name:
            model = 'Gemini 1.5 Flash 8B'
        elif 'gemini_2.5_flash_lite' in exp_name:
            model = 'Gemini 2.5 Flash Lite'
        else:
            model = 'Unknown'
            
        return {
            'approach': 'Adaptive LLM',
            'model': model,
            'classifier_type': model,  # Show LLM name in classifier column
            'classifier_id': 'N/A',
            'training': 'None',
            'full_name': f'Adaptive LLM ({model})'
        }
    
    # Parse baseline experiments  
    elif exp_name.startswith('baseline'):
        # Pattern: baseline_{model}_{dataset}_{method}
        if 'gemini_1.5_flash_8b' in exp_name:
            model = 'Gemini 1.5 Flash 8B'
        elif 'gemini_2.5_flash_lite' in exp_name:
            model = 'Gemini 2.5 Flash Lite'
        else:
            model = 'Unknown'
            
        if exp_name.endswith('_ircot'):
            method = 'IRCoT'
            approach = 'Baseline (IRCoT)'
        elif exp_name.endswith('_nor'):
            method = 'No Retrieval'
            approach = 'Baseline (No Retrieval)'
        elif exp_name.endswith('_oner'):
            method = 'One Retrieval'
            approach = 'Baseline (One Retrieval)'
        else:
            method = 'Unknown'
            approach = 'Baseline'
            
        return {
            'approach': approach,
            'model': model,
            'classifier_type': 'None',
            'classifier_id': 'N/A',
            'training': 'None',
            'full_name': f'{approach} ({model})'
        }
    
    # Handle other experiment types
    else:
        return {
            'approach': 'Other',
            'model': 'Unknown',
            'classifier_type': 'None',
            'classifier_id': 'N/A',
            'training': 'None',
            'full_name': 'Unknown Experiment'
        }

def scan_experiment_folder(folder_path):
    """Scan experiment folder for log files and extract results"""
    results = []
    
    print(f"Scanning: {folder_path}")
    
    if not os.path.exists(folder_path):
        print(f"ERROR: Folder {folder_path} does not exist!")
        return None
    
    # Walk through all subdirectories
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith('.log') and not '/logs/' in root:
                # Only process main log files, not the ones in nested logs/ directories
                log_file_path = os.path.join(root, file)
                result = parse_log_file(log_file_path)
                if result:
                    # Add categorization
                    cat = categorize_experiment(result['experiment'])
                    result.update(cat)
                    results.append(result)
    
    return results

def scan_runs_folder_for_verification(runs_folder_path):
    """Scan runs folder specifically for verification experiments"""
    verification_results = []
    
    if not os.path.exists(runs_folder_path):
        print(f"WARNING: Runs folder {runs_folder_path} does not exist!")
        return verification_results
    
    print(f"Scanning verification experiments in: {runs_folder_path}")
    
    # Walk through all subdirectories looking for enhanced_verification_ experiments
    for root, dirs, files in os.walk(runs_folder_path):
        for file in files:
            if file.endswith('.log') and 'enhanced_verification_' in file:
                log_file_path = os.path.join(root, file)
                result = parse_log_file(log_file_path)
                if result:
                    # Add categorization
                    cat = categorize_experiment(result['experiment'])
                    result.update(cat)
                    verification_results.append(result)
    
    print(f"Found {len(verification_results)} verification experiments")
    return verification_results

def generate_markdown_report(results, experiment_name):
    """Generate comprehensive markdown report"""
    if not results:
        return "# ❌ No Results Found\n\nNo valid experiment results were found in the specified folder."
    
    # Separate regular and verification experiments
    regular_results = []
    verification_results = []
    
    for result in results:
        if result.get('is_verification', False):
            verification_results.append(result)
        else:
            regular_results.append(result)
    
    print(f"Processing {len(regular_results)} regular experiments and {len(verification_results)} verification experiments")
    
    # Use only regular experiments for the main table
    results = regular_results
    
    # Round numeric values
    for result in results:
        for key in ['f1_score', 'exact_match', 'accuracy', 'avg_steps', 'queries_per_second']:
            result[key] = round(result[key], 3)
    
    # Sort results by accuracy
    results.sort(key=lambda x: (x['dataset'], -x['accuracy']))
    
    # Organize by dataset
    datasets = {}
    for result in results:
        dataset = result['dataset']
        if dataset not in datasets:
            datasets[dataset] = []
        datasets[dataset].append(result)
    
    # Calculate overall stats
    total_experiments = len(results)
    best_f1_result = max(results, key=lambda x: x['f1_score'])
    best_acc_result = max(results, key=lambda x: x['accuracy'])
    best_speed_result = max(results, key=lambda x: x['queries_per_second'])
    
    # Model comparison
    gemini_25_results = [r for r in results if 'Gemini 2.5' in r['model']]
    gemini_15_results = [r for r in results if 'Gemini 1.5' in r['model']]
    
    g25_avg_f1 = sum(r['f1_score'] for r in gemini_25_results) / len(gemini_25_results) if gemini_25_results else 0
    g25_avg_acc = sum(r['accuracy'] for r in gemini_25_results) / len(gemini_25_results) if gemini_25_results else 0
    g25_avg_qps = sum(r['queries_per_second'] for r in gemini_25_results) / len(gemini_25_results) if gemini_25_results else 0
    
    g15_avg_f1 = sum(r['f1_score'] for r in gemini_15_results) / len(gemini_15_results) if gemini_15_results else 0
    g15_avg_acc = sum(r['accuracy'] for r in gemini_15_results) / len(gemini_15_results) if gemini_15_results else 0
    g15_avg_qps = sum(r['queries_per_second'] for r in gemini_15_results) / len(gemini_15_results) if gemini_15_results else 0
    
    content = f"""# 📊 Adaptive RAG Benchmark Results - {experiment_name}

**Experiment Run:** `{experiment_name}`  
**Total Experiments:** {total_experiments}  
**Models:** Gemini 1.5 Flash 8B, Gemini 2.5 Flash Lite  
**Datasets:** {', '.join(sorted(datasets.keys()))}  
**Classifier Variants:** Small, Medium, Large (all sizes included)  
**Approaches:** Adaptive LLM, Adaptive ML, Adaptive ML + Verification, Baseline methods  
**Ranking:** All experiments ranked by **Accuracy** (highest to lowest)

---

## 🎯 Executive Summary

### 🏆 Overall Champions
- **🥇 Highest Accuracy:** `{best_acc_result['accuracy']:.3f}` - {best_acc_result['full_name']} on {best_acc_result['dataset']}
- **🎯 Highest F1 Score:** `{best_f1_result['f1_score']:.3f}` - {best_f1_result['full_name']} on {best_f1_result['dataset']}
- **⚡ Fastest System:** `{best_speed_result['queries_per_second']:.1f} q/s` - {best_speed_result['full_name']} on {best_speed_result['dataset']}

### ⚖️ Model Showdown
| Model | Avg Accuracy | Avg F1 | Avg Speed | Experiments |
|-------|--------------|--------|-----------|-------------|
| **Gemini 2.5 Flash Lite** | {g25_avg_acc:.3f} | {g25_avg_f1:.3f} | {g25_avg_qps:.1f} q/s | {len(gemini_25_results)} |
| **Gemini 1.5 Flash 8B** | {g15_avg_acc:.3f} | {g15_avg_f1:.3f} | {g15_avg_qps:.1f} q/s | {len(gemini_15_results)} |

**Winner:** {"Gemini 2.5 Flash Lite" if g25_avg_acc > g15_avg_acc else "Gemini 1.5 Flash 8B"} 🏆

---

"""
    
    # Add dataset-by-dataset analysis
    for dataset_name in sorted(datasets.keys()):
        dataset_results = datasets[dataset_name]
        dataset_results.sort(key=lambda x: x['accuracy'], reverse=True)
        
        best_result = dataset_results[0]
        worst_result = dataset_results[-1]
        avg_f1 = sum(r['f1_score'] for r in dataset_results) / len(dataset_results)
        avg_acc = sum(r['accuracy'] for r in dataset_results) / len(dataset_results)
        
        content += f"""## 📊 {dataset_name.upper()} - Complete Results

**Total Experiments:** {len(dataset_results)}  
**Accuracy Range:** {worst_result['accuracy']:.3f} - {best_result['accuracy']:.3f}  
**Average Performance:** {avg_acc:.3f} Accuracy, {avg_f1:.3f} F1  

### 🏆 ALL Results (Ranked by Accuracy)

| Rank | Approach | Model | Classifier | Training | Acc | Verif Acc | F1 | EM | Steps | Verif Steps | Q/s |
|------|----------|-------|------------|----------|-----|-----------|----|----|-------|-------------|-----|
"""
        
        # Create verification comparison mapping using the separated verification_results
        verification_map = {}
        
        # Index verification experiments by their configuration
        for ver_result in verification_results:
            if ver_result['approach'] == 'Adaptive RAG' and ver_result['dataset'] == dataset_name:
                # Create a normalized key for matching
                model_normalized = ver_result['model'].replace(' ', '_').replace('.', '_').lower()
                classifier_normalized = ver_result['classifier_type'].lower()
                training_normalized = ver_result['training'].replace(' ', '_').replace('(', '').replace(')', '').lower()
                
                key = f"{model_normalized}_{classifier_normalized}_{ver_result['classifier_id']}_{training_normalized}"
                verification_map[key] = ver_result

        
        # Add ALL experiments for this dataset
        for idx, result in enumerate(dataset_results, 1):
            medal = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉" if idx == 3 else f"{idx}"
            
            # Generate verification comparison columns
            verif_acc = "N/A"
            verif_steps = "N/A"
            
            if result['approach'] == 'Adaptive RAG':
                # Use the same normalization as above for lookup
                model_normalized = result['model'].replace(' ', '_').replace('.', '_').lower()
                classifier_normalized = result['classifier_type'].lower()
                training_normalized = result['training'].replace(' ', '_').replace('(', '').replace(')', '').lower()
                
                key = f"{model_normalized}_{classifier_normalized}_{result['classifier_id']}_{training_normalized}"

                if key in verification_map:
                    ver = verification_map[key]
                    verif_acc = f"{ver['accuracy']:.3f}"
                    verif_steps = f"{ver['avg_steps']:.1f}"

            
            content += f"| {medal} | {result['approach']} | {result['model']} | {result['classifier_type']} | {result['training']} | **{result['accuracy']:.3f}** | {verif_acc} | {result['f1_score']:.3f} | {result['exact_match']:.3f} | {result['avg_steps']:.1f} | {verif_steps} | {result['queries_per_second']:.1f} |\n"
        
        # Approach summary
        approach_groups = defaultdict(list)
        for result in dataset_results:
            approach_groups[result['approach']].append(result)
        
        content += f"""
### 📈 Approach Summary for {dataset_name.upper()}

| Approach | Count | Best Acc | Avg Acc | Best F1 | Avg F1 | Avg Speed |
|----------|-------|----------|---------|---------|--------|-----------|
"""
        
        for approach in sorted(approach_groups.keys()):
            approach_results = approach_groups[approach]
            best_acc = max(r['accuracy'] for r in approach_results)
            avg_acc = sum(r['accuracy'] for r in approach_results) / len(approach_results)
            best_f1 = max(r['f1_score'] for r in approach_results)
            avg_f1 = sum(r['f1_score'] for r in approach_results) / len(approach_results)
            avg_speed = sum(r['queries_per_second'] for r in approach_results) / len(approach_results)
            
            content += f"| {approach} | {len(approach_results)} | **{best_acc:.3f}** | {avg_acc:.3f} | **{best_f1:.3f}** | {avg_f1:.3f} | {avg_speed:.1f} |\n"
        
        content += "\n---\n\n"
    
    # Add insights
    content += f"""## 🔍 Analysis Summary

### 📊 Dataset Difficulty Ranking
"""
    
    dataset_stats = []
    for dataset_name, dataset_results in datasets.items():
        max_acc = max(r['accuracy'] for r in dataset_results)
        avg_acc = sum(r['accuracy'] for r in dataset_results) / len(dataset_results)
        max_f1 = max(r['f1_score'] for r in dataset_results)
        avg_f1 = sum(r['f1_score'] for r in dataset_results) / len(dataset_results)
        
        difficulty = "Easy" if max_acc > 0.8 else "Medium" if max_acc > 0.6 else "Hard"
        dataset_stats.append((dataset_name, max_acc, avg_acc, max_f1, avg_f1, difficulty))
    
    dataset_stats.sort(key=lambda x: x[1], reverse=True)
    
    content += "| Rank | Dataset | Max Acc | Avg Acc | Max F1 | Avg F1 | Difficulty |\n"
    content += "|------|---------|---------|---------|--------|--------|------------|\n"
    
    for idx, (dataset, max_acc, avg_acc, max_f1, avg_f1, difficulty) in enumerate(dataset_stats, 1):
        emoji = "🟢" if difficulty == "Easy" else "🟡" if difficulty == "Medium" else "🔴"
        content += f"| {idx} | **{dataset}** | {max_acc:.3f} | {avg_acc:.3f} | {max_f1:.3f} | {avg_f1:.3f} | {emoji} {difficulty} |\n"
    
    content += f"""
### 🏆 Best Performers by Dataset
"""
    
    for dataset_name in sorted(datasets.keys()):
        dataset_results = datasets[dataset_name]
        best_result = max(dataset_results, key=lambda x: x['accuracy'])
        content += f"- **{dataset_name}:** {best_result['full_name']} ({best_result['accuracy']:.3f} Acc, {best_result['f1_score']:.3f} F1)\n"
    
    content += f"""
---

## 📋 Technical Details

**Generated:** January 19, 2025  
**Ranking Criteria:** Accuracy (primary), F1 Score (secondary)  
**Total Experiments:** {total_experiments} across {len(datasets)} datasets  

**Classifier Categories:**
- **Small:** Early checkpoints (1920, 1974, 1681, 1794)
- **Medium:** Mid-range checkpoints (9603, 9874, 8407, 8971)  
- **Large:** Final checkpoints (19206, 19749, 16815, 17943)

**Training Variants:**
- **Original:** Base training approach
- **Optimized:** Enhanced training methodology  
- **Epochs:** 3ep (fast) vs 20ep (thorough)

**Verification Experiments:**
- **Adaptive RAG + Verification:** Classifier predictions verified by the generator LLM
- **Verification Process:** Initial ML classifier decision → LLM verification → Final routing decision

---

*📊 Report generated by Adaptive RAG Benchmark Results Generator*
"""
    
    return content

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print("Usage: python generate_results_report.py <experiment_folder_path>")
        print("Example: python generate_results_report.py /path/to/sequential_experiments_20250818_005830")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    experiment_name = os.path.basename(folder_path)
    
    print(f"Adaptive RAG Benchmark Results Generator")
    print(f"Target folder: {folder_path}")
    print(f"Experiment: {experiment_name}")
    
    # Scan for results
    results = scan_experiment_folder(folder_path)
    if results is None:
        sys.exit(1)
    
    print(f"Found {len(results)} regular experiments")
    
    # Also scan the same folder for verification experiments
    verification_results = scan_runs_folder_for_verification(folder_path)
    
    # Combine regular and verification results
    all_results = results + verification_results
    print(f"Total experiments (including verification): {len(all_results)}")
    
    if not all_results:
        print("ERROR: No valid results found!")
        sys.exit(1)
    
    # Organize by dataset
    datasets = set(r['dataset'] for r in all_results)
    print(f"Datasets: {', '.join(sorted(datasets))}")
    
    # Generate report
    print("Generating markdown report...")
    markdown_content = generate_markdown_report(all_results, experiment_name)
    
    # Create results directory and save report
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, "results")
    os.makedirs(results_dir, exist_ok=True)
    
    output_file = os.path.join(results_dir, f"{experiment_name}.md")
    with open(output_file, 'w') as f:
        f.write(markdown_content)
    
    print(f"Report saved to: {output_file}")
    print(f"Summary: {len(all_results)} experiments across {len(datasets)} datasets")
    
    # Show top performers
    if all_results:
        best_acc = max(all_results, key=lambda x: x['accuracy'])
        best_f1 = max(all_results, key=lambda x: x['f1_score'])
        print(f"🏆 Best Accuracy: {best_acc['accuracy']:.3f} ({best_acc['dataset']})")
        print(f"Best F1: {best_f1['f1_score']:.3f} ({best_f1['dataset']})")

if __name__ == "__main__":
    main()
