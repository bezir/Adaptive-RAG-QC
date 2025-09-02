#!/usr/bin/env python3
"""
Convert annotated dataset to classifier training format.

This script converts the output from silver labeling (annotated with systems_used, systems_succeeded, etc.)
to the format expected by the classifier training pipeline.


Output format for classifier:
{
    "id": "d41d8cd9",
    "question": "Question from hotpotqa dataset (sample 3025)",
    "answer": "A",
    "dataset_name": "hotpotqa"
}
"""

import json
import os
import argparse
import random
import glob
from typing import List, Dict, Tuple
from pathlib import Path

# Optional HuggingFace datasets import
try:
    from datasets import load_dataset
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    load_dataset = None
    print("Warning: HuggingFace datasets library not available. Install with: pip install datasets")
    print("HuggingFace dataset functionality will be disabled.")


def discover_json_files_in_folder(folder_path: str) -> List[str]:
    """Discover all JSON files in a folder and its subdirectories."""
    if not os.path.exists(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        return []
    
    if not os.path.isdir(folder_path):
        print(f"Error: Path is not a directory: {folder_path}")
        return []
    
    # Find all JSON files recursively
    json_files = []
    search_patterns = [
        os.path.join(folder_path, "*.json"),
        os.path.join(folder_path, "**/*.json")
    ]
    
    for pattern in search_patterns:
        json_files.extend(glob.glob(pattern, recursive=True))
    
    # Remove duplicates and sort
    json_files = sorted(list(set(json_files)))
    
    print(f"Discovered {len(json_files)} JSON files in {folder_path}:")
    for file_path in json_files:
        print(f"  - {file_path}")
    
    return json_files


def load_annotated_data(input_paths: List[str]) -> List[Dict]:
    """Load annotated data from multiple input files."""
    all_data = []
    
    for input_path in input_paths:
        if not os.path.exists(input_path):
            print(f"Warning: Input file not found: {input_path}")
            continue
            
        print(f"Loading data from: {input_path}")
        
        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            if isinstance(data, list):
                # Handle flat list format
                all_data.extend(data)
            elif isinstance(data, dict):
                # Handle nested format with 'individual_results' key
                if 'individual_results' in data:
                    individual_results = data['individual_results']
                    if isinstance(individual_results, list):
                        # Add dataset metadata to each item
                        dataset_name = data.get('dataset_name', 'unknown')
                        for item in individual_results:
                            if 'dataset_name' not in item:
                                item['dataset_name'] = dataset_name
                        
                        all_data.extend(individual_results)
                        print(f"Loaded {len(individual_results)} samples from 'individual_results' key")
                    else:
                        print(f"Warning: 'individual_results' in {input_path} is not a list, got {type(individual_results)}")
                else:
                    print(f"Warning: Expected 'individual_results' key in dict format for {input_path}")
            else:
                print(f"Warning: Expected list or dict in {input_path}, got {type(data)}")
                
        except json.JSONDecodeError as e:
            print(f"Error parsing JSON in {input_path}: {e}")
        except Exception as e:
            print(f"Error reading {input_path}: {e}")
    
    print(f"Loaded {len(all_data)} total samples from {len(input_paths)} files")
    return all_data


def load_annotated_data_from_hf(dataset_name: str = "bezir/AdaptiveRAGQueryComplexity", 
                               model_configs: List[str] = None) -> List[Dict]:
    """Load annotated data from HuggingFace dataset."""
    if not HF_AVAILABLE:
        raise ImportError("HuggingFace datasets library is required but not installed. Run: pip install datasets")
    
    all_data = []
    
    # Default to both model configurations if none specified
    if model_configs is None:
        model_configs = ["gemini-1.5-flash-8b", "gemini-2.5-flash-lite"]
    
    print(f"Loading data from HuggingFace dataset: {dataset_name}")
    
    for config in model_configs:
        print(f"Loading configuration: {config}")
        try:
            # Load the specific configuration
            dataset = load_dataset(dataset_name, config, split='train')
            
            # Convert HuggingFace dataset to list of dictionaries
            config_data = []
            for sample in dataset:
                # Convert the sample to a regular dict and add any missing fields
                sample_dict = dict(sample)
                
                # Ensure dataset_name is set from source_dataset if not present
                if 'dataset_name' not in sample_dict and 'source_dataset' in sample_dict:
                    sample_dict['dataset_name'] = sample_dict['source_dataset']
                
                config_data.append(sample_dict)
            
            all_data.extend(config_data)
            print(f"Loaded {len(config_data)} samples from {config} configuration")
            
        except Exception as e:
            print(f"Error loading configuration {config}: {e}")
            continue
    
    print(f"Loaded {len(all_data)} total samples from HuggingFace dataset")
    return all_data


def convert_to_classifier_format(annotated_data: List[Dict]) -> List[Dict]:
    """Convert annotated data to classifier training format."""
    classifier_data = []
    
    # Mapping from numeric complexity labels to letter format
    complexity_mapping = {0: "A", 1: "B", 2: "C"}
    
    for item in annotated_data:
        # Extract only the fields needed for classifier training
        # Handle different field name mappings
        item_id = item.get("id", item.get("sample_id", ""))
        question = item.get("question", "")
        item_dataset = item.get("dataset_name", item.get("source_dataset", "unknown"))
        
        # Handle complexity label conversion
        # Priority: complexity_label (HF dataset) > label (legacy) > answer (fallback)
        complexity_label = None
        if "complexity_label" in item:
            # HuggingFace dataset format: numeric (0, 1, 2) -> letter (A, B, C)
            numeric_label = item["complexity_label"]
            if numeric_label in complexity_mapping:
                complexity_label = complexity_mapping[numeric_label]
            else:
                print(f"Warning: Unknown complexity_label value '{numeric_label}' for item {item_id}")
                continue
        elif "label" in item:
            # Legacy format: already in A/B/C format
            complexity_label = item["label"]
        elif "answer" in item and item["answer"] in ["A", "B", "C"]:
            # Fallback: answer field contains A/B/C
            complexity_label = item["answer"]
        else:
            print(f"Warning: No valid complexity label found for item {item_id}")
            continue
        
        classifier_item = {
            "id": item_id,
            "question": question,
            "answer": complexity_label,
            "dataset_name": item_dataset
        }
        
        # Validate that we have the essential fields
        if not classifier_item["question"]:
            print(f"Warning: Skipping item with missing question: {classifier_item['id']}")
            continue
            
        # Validate answer is in expected format (A, B, or C)
        if classifier_item["answer"] not in ["A", "B", "C"]:
            print(f"Warning: Invalid complexity label '{classifier_item['answer']}' for item {classifier_item['id']}")
            continue
            
        classifier_data.append(classifier_item)
    
    print(f"Converted {len(classifier_data)} valid samples to classifier format")
    return classifier_data


def balance_labels_and_sources(data: List[Dict]) -> List[Dict]:
    """
    Balance the dataset with prioritized balancing:
    1. MANDATORY: Equal class distribution (A, B, C) - exact balance required
    2. BEST EFFORT: Dataset source distribution within each class - maximize data while improving balance
    """
    print("\n" + "="*60)
    print("PRIORITIZED BALANCING: MANDATORY CLASS + BEST-EFFORT SOURCE")
    print("="*60)
    
    # Phase 1: Analyze structure
    print("\n--- Phase 1: Analyzing Data Structure ---")
    
    # Group by class and source
    class_source_data = {}  # {class: {source: [items]}}
    class_source_counts = {}  # {class: {source: count}}
    class_totals = {"A": 0, "B": 0, "C": 0}
    
    for item in data:
        label = item.get("answer", "")
        source = item.get("dataset_name", "unknown")
        
        if label not in ["A", "B", "C"]:
            continue
            
        if label not in class_source_data:
            class_source_data[label] = {}
            class_source_counts[label] = {}
            
        if source not in class_source_data[label]:
            class_source_data[label][source] = []
            class_source_counts[label][source] = 0
            
        class_source_data[label][source].append(item)
        class_source_counts[label][source] += 1
        class_totals[label] += 1
    
    # Print current distribution
    print("Current distribution by class:")
    for label in ["A", "B", "C"]:
        print(f"  Class {label}: {class_totals[label]} total samples")
    
    print("\nCurrent distribution by (class, source):")
    all_sources = set()
    for label in ["A", "B", "C"]:
        if label in class_source_counts:
            all_sources.update(class_source_counts[label].keys())
    
    for label in ["A", "B", "C"]:
        print(f"  Class {label}:")
        if label in class_source_counts:
            for source in sorted(all_sources):
                count = class_source_counts[label].get(source, 0)
                print(f"    {source}: {count}")
        else:
            print(f"    No data found")
    
    # Phase 2: Calculate sampling strategy
    print("\n--- Phase 2: Calculating Sampling Strategy ---")
    
    min_class_total = min(class_totals[label] for label in ["A", "B", "C"] if class_totals[label] > 0)
    print(f"Minimum class total: {min_class_total}")
    print(f"Target samples per class: {min_class_total} (MANDATORY for class balance)")
    print(f"Target total samples: {min_class_total * 3}")
    
    if min_class_total == 0:
        print("Error: At least one class has no data")
        return []
    
    # Phase 3: Best-effort source-balanced sampling within each class
    print("\n--- Phase 3: Performing Best-Effort Source-Balanced Sampling ---")
    
    balanced_data = []
    random.seed(42)  # For reproducibility
    
    for label in ["A", "B", "C"]:
        print(f"\nSampling for class {label} (target: {min_class_total} samples):")
        
        if label not in class_source_data:
            print(f"  Warning: No data for class {label}")
            continue
        
        # Get all sources for this class
        available_sources = list(class_source_data[label].keys())
        source_counts = {source: len(class_source_data[label][source]) for source in available_sources}
        
        print(f"  Available sources: {source_counts}")
        
        # Calculate proportional sampling from each source
        total_available = sum(source_counts.values())
        samples_needed = min_class_total
        
        sampled_data = []
        remaining_samples = samples_needed
        
        # Sort sources by count for consistent ordering
        sorted_sources = sorted(available_sources, key=lambda s: source_counts[s], reverse=True)
        
        for i, source in enumerate(sorted_sources):
            available_from_source = source_counts[source]
            
            if i == len(sorted_sources) - 1:
                samples_from_source = min(remaining_samples, available_from_source)
            else:
                proportion = available_from_source / total_available
                ideal_samples = int(proportion * samples_needed)
                samples_from_source = min(ideal_samples, available_from_source, remaining_samples)
            
            if samples_from_source > 0:
                source_data = class_source_data[label][source]
                sampled_from_source = random.sample(source_data, samples_from_source)
                sampled_data.extend(sampled_from_source)
                remaining_samples -= samples_from_source
                print(f"    {source}: sampled {samples_from_source} / {available_from_source} samples")
            
            if remaining_samples <= 0:
                break
        
        if remaining_samples > 0:
            print(f"    Need {remaining_samples} more samples - taking from available sources")
            for source in sorted_sources:
                if remaining_samples <= 0:
                    break
                    
                source_data = class_source_data[label][source]
                # Get samples not already selected
                already_selected_ids = {item['id'] for item in sampled_data if item.get('dataset_name') == source}
                available_samples = [item for item in source_data if item['id'] not in already_selected_ids]
                
                additional_needed = min(remaining_samples, len(available_samples))
                if additional_needed > 0:
                    additional_samples = random.sample(available_samples, additional_needed)
                    sampled_data.extend(additional_samples)
                    remaining_samples -= additional_needed
                    print(f"    {source}: additional {additional_needed} samples")
        
        balanced_data.extend(sampled_data)
        print(f"  Total for class {label}: {len(sampled_data)} samples")
    
    # Shuffle the final balanced dataset
    random.shuffle(balanced_data)
    
    # Final verification
    final_counts = {"A": 0, "B": 0, "C": 0}
    final_source_counts = {}
    
    for item in balanced_data:
        label = item["answer"]
        source = item["dataset_name"]
        
        final_counts[label] += 1
        
        if source not in final_source_counts:
            final_source_counts[source] = {"A": 0, "B": 0, "C": 0}
        final_source_counts[source][label] += 1
    
    print(f"\n--- Final Balanced Dataset ---")
    print(f"Total samples: {len(balanced_data)}")
    print(f"Class distribution: {final_counts}")
    print(f"Class balance achieved: {all(count == final_counts['A'] for count in final_counts.values())}")
    print(f"Source distribution (best-effort balanced):")
    for source in sorted(final_source_counts.keys()):
        source_total = sum(final_source_counts[source].values())
        print(f"  {source}: {final_source_counts[source]} (total: {source_total})")
    
    return balanced_data


def balance_labels(data: List[Dict]) -> List[Dict]:
    """Legacy function - redirects to new dual-level balancing."""
    print("Note: Using enhanced dual-level balancing (classes + sources)")
    return balance_labels_and_sources(data)


def split_train_validation(data: List[Dict], validation_ratio: float, random_seed: int = 42) -> Tuple[List[Dict], List[Dict]]:
    """Split data into training and validation sets."""
    if validation_ratio <= 0 or validation_ratio >= 1:
        raise ValueError(f"validation_ratio must be between 0 and 1, got {validation_ratio}")
    
    # Shuffle data with fixed seed for reproducibility
    data_copy = data.copy()
    random.seed(random_seed)
    random.shuffle(data_copy)
    
    # Calculate split point
    val_size = int(len(data_copy) * validation_ratio)
    train_size = len(data_copy) - val_size
    
    train_data = data_copy[:train_size]
    val_data = data_copy[train_size:]
    
    print(f"Split into {len(train_data)} training and {len(val_data)} validation samples")
    print(f"Training ratio: {len(train_data)/len(data_copy):.3f}, Validation ratio: {len(val_data)/len(data_copy):.3f}")
    
    return train_data, val_data


def create_dataset_subset(data: List[Dict], percentage: float, random_seed: int = 42) -> List[Dict]:
    """Create a subset of the data with the specified percentage."""
    if percentage <= 0 or percentage > 1:
        raise ValueError(f"percentage must be between 0 and 1, got {percentage}")
    
    # Shuffle data with fixed seed for reproducibility
    data_copy = data.copy()
    random.seed(random_seed)
    random.shuffle(data_copy)
    
    # Calculate subset size
    subset_size = int(len(data_copy) * percentage)
    subset_data = data_copy[:subset_size]
    
    print(f"Created {percentage*100:.0f}% subset: {len(subset_data)} samples from {len(data_copy)} total")
    
    return subset_data


def create_multiple_datasets(balanced_data: List[Dict], base_output_dir: Path, dataset_name_template: str, 
                           validation_ratio: float, random_seed: int = 42):
    """Create multiple datasets with different sizes (10%, 50%, 100%)."""
    percentages = [0.1, 0.5, 1.0]
    
    print("\n" + "="*60)
    print("CREATING MULTIPLE DATASET SIZES")
    print("="*60)
    
    created_datasets = []
    
    for percentage in percentages:
        print(f"\n--- Creating {percentage*100:.0f}% dataset ---")
        
        # Create subset with different seed for each percentage to ensure variety
        subset_seed = random_seed + int(percentage * 100)  # Different seed for each subset
        subset_data = create_dataset_subset(balanced_data, percentage, subset_seed)
        
        # Calculate actual size for folder naming
        actual_size = len(subset_data)
        
        # Create dataset name with actual calculated size
        # Extract model name and type from template (format: size_model_type)
        parts = dataset_name_template.split('_', 2)  # Split into max 3 parts
        if len(parts) >= 3:
            _, model_name, type_name = parts[0], parts[1], parts[2]
            size_dataset_name = f"{actual_size}_{model_name}_{type_name}"
        else:
            # Fallback if template doesn't match expected format
            size_dataset_name = f"{actual_size}_{dataset_name_template}"
        
        # Split into train/validation
        split_seed = random_seed  # Same split seed for consistency across sizes
        train_data, val_data = split_train_validation(subset_data, validation_ratio, split_seed)
        
        # Create output directory for this size (directly under base_output_dir)
        size_output_dir = base_output_dir / size_dataset_name
        size_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save files
        train_path = size_output_dir / "train.json"
        val_path = size_output_dir / "valid.json"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        print(f"  Saved {len(train_data)} training samples to: {train_path}")
        print(f"  Saved {len(val_data)} validation samples to: {val_path}")
        
        # Print label distribution for this subset
        train_stats = analyze_dataset_distribution(train_data)
        val_stats = analyze_dataset_distribution(val_data)
        print(f"  Training label distribution: {train_stats['answer_distribution']}")
        print(f"  Validation label distribution: {val_stats['answer_distribution']}")
        
        created_datasets.append({
            'percentage': percentage,
            'actual_size': actual_size,
            'dataset_name': size_dataset_name,
            'path': size_output_dir
        })
    
    return created_datasets


def analyze_dataset_distribution(data: List[Dict]) -> Dict:
    """Analyze the distribution of answers and datasets."""
    answer_counts = {"A": 0, "B": 0, "C": 0}
    dataset_counts = {}
    
    for item in data:
        answer = item.get("answer", "")
        dataset = item.get("dataset_name", "unknown")
        
        if answer in answer_counts:
            answer_counts[answer] += 1
            
        if dataset not in dataset_counts:
            dataset_counts[dataset] = 0
        dataset_counts[dataset] += 1
    
    return {
        "total_samples": len(data),
        "answer_distribution": answer_counts,
        "dataset_distribution": dataset_counts
    }


def print_comprehensive_dataset_info(data: List[Dict], dataset_name: str, dataset_type: str = ""):
    """Print comprehensive information about the classification dataset."""
    print("\n" + "="*80)
    print(f"COMPREHENSIVE CLASSIFICATION DATASET ANALYSIS")
    if dataset_type:
        print(f"Dataset: {dataset_name} ({dataset_type})")
    else:
        print(f"Dataset: {dataset_name}")
    print("="*80)
    
    if not data:
        print("No data to analyze")
        return
    
    # Basic stats
    total_samples = len(data)
    print(f"\nBASIC STATISTICS:")
    print(f"  Total samples: {total_samples:,}")
    
    # Label distribution analysis
    label_counts = {"A": 0, "B": 0, "C": 0}
    dataset_counts = {}
    label_dataset_matrix = {}  # {label: {dataset: count}}
    
    for item in data:
        label = item.get("answer", "")
        dataset = item.get("dataset_name", "unknown")
        
        if label in label_counts:
            label_counts[label] += 1
            
        if dataset not in dataset_counts:
            dataset_counts[dataset] = 0
        dataset_counts[dataset] += 1
        
        # Build label-dataset matrix
        if label not in label_dataset_matrix:
            label_dataset_matrix[label] = {}
        if dataset not in label_dataset_matrix[label]:
            label_dataset_matrix[label][dataset] = 0
        label_dataset_matrix[label][dataset] += 1
    
    # Print label distribution
    print(f"\nCLASSIFICATION LABEL DISTRIBUTION:")
    total_labeled = sum(label_counts.values())
    for label in ["A", "B", "C"]:
        count = label_counts[label]
        percentage = (count / total_labeled * 100) if total_labeled > 0 else 0
        print(f"  Class {label}: {count:,} samples ({percentage:.1f}%)")
    
    # Check if labels are balanced
    if total_labeled > 0:
        min_count = min(label_counts.values())
        max_count = max(label_counts.values())
        balance_ratio = min_count / max_count if max_count > 0 else 0
        if balance_ratio >= 0.95:
            print(f"  Labels are well balanced (ratio: {balance_ratio:.3f})")
        elif balance_ratio >= 0.8:
            print(f"  WARNING: Labels are moderately balanced (ratio: {balance_ratio:.3f})")
        else:
            print(f"  ERROR: Labels are imbalanced (ratio: {balance_ratio:.3f})")
    
    # Print dataset source distribution
    print(f"\nDATASET SOURCE DISTRIBUTION:")
    sorted_datasets = sorted(dataset_counts.items(), key=lambda x: x[1], reverse=True)
    for dataset, count in sorted_datasets:
        percentage = (count / total_samples * 100) if total_samples > 0 else 0
        print(f"  {dataset}: {count:,} samples ({percentage:.1f}%)")
    
    # Print detailed label-dataset cross-tabulation
    print(f"\nDETAILED LABEL × DATASET DISTRIBUTION:")
    print(f"{'Dataset':<15} {'Class A':<10} {'Class B':<10} {'Class C':<10} {'Total':<10}")
    print("-" * 60)
    
    for dataset in sorted(dataset_counts.keys()):
        counts_a = label_dataset_matrix.get("A", {}).get(dataset, 0)
        counts_b = label_dataset_matrix.get("B", {}).get(dataset, 0)
        counts_c = label_dataset_matrix.get("C", {}).get(dataset, 0)
        total_dataset = counts_a + counts_b + counts_c
        
        print(f"{dataset:<15} {counts_a:<10} {counts_b:<10} {counts_c:<10} {total_dataset:<10}")
    
    # Print totals row
    total_a = label_counts["A"]
    total_b = label_counts["B"] 
    total_c = label_counts["C"]
    print("-" * 60)
    print(f"{'TOTAL':<15} {total_a:<10} {total_b:<10} {total_c:<10} {total_samples:<10}")
    
    # Dataset balance analysis within each class
    print(f"\nDATASET BALANCE WITHIN EACH CLASS:")
    for label in ["A", "B", "C"]:
        print(f"  Class {label} source distribution:")
        if label in label_dataset_matrix:
            class_total = sum(label_dataset_matrix[label].values())
            for dataset in sorted(label_dataset_matrix[label].keys()):
                count = label_dataset_matrix[label][dataset]
                percentage = (count / class_total * 100) if class_total > 0 else 0
                print(f"    {dataset}: {count} samples ({percentage:.1f}%)")
        else:
            print(f"    No data for class {label}")
    
    # Quality indicators
    print(f"\nDATASET QUALITY INDICATORS:")
    
    # Check for missing IDs
    ids = [item.get('id', '') for item in data]
    unique_ids = set(filter(None, ids))  # Filter out empty IDs
    if len(unique_ids) == len(ids) and all(ids):
        print(f"  All samples have unique IDs ({len(unique_ids):,} unique)")
    else:
        missing_ids = len(ids) - len([id for id in ids if id])
        duplicate_ids = len(ids) - len(unique_ids)
        if missing_ids > 0:
            print(f"  WARNING: {missing_ids} samples missing IDs")
        if duplicate_ids > 0:
            print(f"  WARNING: {duplicate_ids} duplicate IDs found")
    
    # Check for missing questions/answers
    missing_questions = len([item for item in data if not item.get('question', '').strip()])
    missing_answers = len([item for item in data if item.get('answer', '') not in ['A', 'B', 'C']])
    
    if missing_questions == 0:
        print(f"  All samples have questions")
    else:
        print(f"  WARNING: {missing_questions} samples missing questions")
        
    if missing_answers == 0:
        print(f"  All samples have valid answers (A/B/C)")
    else:
        print(f"  WARNING: {missing_answers} samples with invalid answers")
    
    # Average question length
    question_lengths = [len(item.get('question', '').split()) for item in data if item.get('question', '')]
    if question_lengths:
        avg_length = sum(question_lengths) / len(question_lengths)
        min_length = min(question_lengths)
        max_length = max(question_lengths)
        print(f"  Question length: avg={avg_length:.1f} words, min={min_length}, max={max_length}")
    
    print("="*80)


def save_data_and_stats(data: List[Dict], output_path: str, model_name: str):
    """Save classifier data and generate statistics."""
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save the classifier data
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Saved {len(data)} samples to: {output_path}")
    
    # Generate and save statistics
    stats = analyze_dataset_distribution(data)
    stats["model_name"] = model_name
    stats["source_file"] = output_path
    
    # Print summary
    print(f"\nDataset Statistics:")
    print(f"Total samples: {stats['total_samples']}")
    print(f"Answer distribution: {stats['answer_distribution']}")
    print(f"Dataset distribution: {stats['dataset_distribution']}")


def create_auto_split_hf_datasets(annotated_data: List[Dict], base_output_dir: Path, validation_ratio: float, random_seed: int) -> List[Dict]:
    """
    Automatically split HuggingFace dataset by labeling_model and labeling_strategy.
    Creates separate folders for each combination: {count}_{model}_{strategy}/
    """
    print("\n" + "="*60)
    print("AUTO-SPLITTING HF DATASET BY MODEL AND STRATEGY")
    print("="*60)
    
    # Group data by labeling_model and labeling_strategy
    groups = {}
    for item in annotated_data:
        model = item.get('labeling_model', 'unknown')
        strategy = item.get('labeling_strategy', 'unknown')
        key = (model, strategy)
        
        if key not in groups:
            groups[key] = []
        groups[key].append(item)
    
    print(f"Found {len(groups)} combinations of model and strategy:")
    for (model, strategy), data in groups.items():
        print(f"  📊 {model} + {strategy}: {len(data)} samples")
    
    created_datasets = []
    
    for (model, strategy), group_data in groups.items():
        print(f"\n" + "-"*50)
        print(f"PROCESSING: {model} + {strategy} ({len(group_data)} samples)")
        print("-"*50)
        
        # Convert to classifier format
        classifier_data = convert_to_classifier_format(group_data)
        if not classifier_data:
            print(f"Warning: No valid data after conversion for {model}+{strategy}")
            continue
        
        # Balance labels
        balanced_data = balance_labels(classifier_data)
        if not balanced_data:
            print(f"Warning: No balanced data for {model}+{strategy}")
            continue
        
        # Create folder name: {count}_{model}_{strategy}
        count = len(balanced_data)
        folder_name = f"{count}_{model}_{strategy}"
        dataset_output_dir = base_output_dir / folder_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Split into train/validation
        train_data, val_data = split_train_validation(balanced_data, validation_ratio, random_seed)
        
        # Save files
        train_path = dataset_output_dir / "train.json"
        val_path = dataset_output_dir / "valid.json"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Created dataset: {folder_name}")
        print(f"  📄 Training: {len(train_data)} samples → {train_path}")
        print(f"  📄 Validation: {len(val_data)} samples → {val_path}")
        
        created_datasets.append({
            'model': model,
            'strategy': strategy,
            'folder_name': folder_name,
            'path': dataset_output_dir,
            'total_samples': count,
            'train_samples': len(train_data),
            'val_samples': len(val_data),
            'train_path': train_path,
            'val_path': val_path
        })
    
    return created_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Convert annotated dataset to classifier training format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Auto-split HF dataset by model and strategy (creates folders like: 19749_gemini-1.5-8b_original/)
  python convert_to_classifier_format.py \\
    --auto_split_hf \\
    --output_dir ../classifier/data \\
    --validation_ratio 0.2

  # Default usage (uses HuggingFace dataset with both model configs, single combined folder)
  python convert_to_classifier_format.py \\
    --output_dir ../classifier/data \\
    --model_name combined \\
    --dataset_name hf_query_complexity \\
    --validation_ratio 0.2

  # Load from HuggingFace dataset (specific model config only)
  python convert_to_classifier_format.py \\
    --hf_configs gemini-1.5-flash-8b \\
    --output_dir ../classifier/data \\
    --model_name gemini-1.5-flash-8b \\
    --dataset_name hf_gemini15_8b \\
    --validation_ratio 0.2

  # Use different HuggingFace dataset
  python convert_to_classifier_format.py \\
    --hf_dataset username/other-dataset \\
    --output_dir ../classifier/data \\
    --model_name other_model \\
    --dataset_name other_dataset \\
    --validation_ratio 0.2

  # Convert single local file with validation split
  python convert_to_classifier_format.py \\
    --input_paths path/to/labeled_data.json \\
    --output_dir ../classifier \\
    --model_name gemini-2.5-flash-lite \\
    --dataset_name hotpotqa \\
    --validation_ratio 0.2

  # Convert multiple local files
  python convert_to_classifier_format.py \\
    --input_paths file1.json file2.json file3.json \\
    --output_dir output_directory \\
    --model_name model_name \\
    --dataset_name combined \\
    --validation_ratio 0.2

  # Convert all JSON files in a local folder (searches recursively)
  python convert_to_classifier_format.py \\
    --input_folder scaled_silver_labeling/predictions \\
    --output_dir ../classifier \\
    --model_name gemini-2.5-flash-lite \\
    --dataset_name all_datasets \\
    --validation_ratio 0.2

  # Create multiple dataset sizes (10%, 50%, 100%) with train/val splits
  python convert_to_classifier_format.py \\
    --input_folder scaled_silver_labeling/predictions \\
    --output_dir ../classifier \\
    --model_name gemini-2.5-flash-lite \\
    --dataset_name multi_size_dataset \\
    --validation_ratio 0.2 \\
    --multiple

  # New naming convention: creates folder named 5000_gemini-2.5-flash-lite_optimized
  python convert_to_classifier_format.py \\
    --input_folder scaled_silver_labeling/predictions/dev_5000/optimized_strategy \\
    --output_dir ../classifier/data \\
    --model_name gemini-2.5-flash-lite \\
    --type optimized \\
    --size 5000 \\
    --validation_ratio 0.2
        """
    )
    
    # Create mutually exclusive group for input sources
    input_group = parser.add_mutually_exclusive_group(required=False)
    input_group.add_argument(
        "--input_paths", 
        nargs="+",
        help="Paths to input JSON files containing annotated data"
    )
    input_group.add_argument(
        "--input_folder",
        help="Folder containing JSON files to process (searches recursively)"
    )
    input_group.add_argument(
        "--hf_dataset",
        default="bezir/AdaptiveRAGQueryComplexity",
        help="HuggingFace dataset name to load (default: 'bezir/AdaptiveRAGQueryComplexity')"
    )
    parser.add_argument(
        "--output_dir", 
        required=True,
        help="Output directory for classifier training data"
    )
    parser.add_argument(
        "--model_name", 
        required=False,
        help="Name of the model used for labeling (for metadata) - not needed when using --auto_split_hf"
    )
    parser.add_argument(
        "--dataset_name",
        help="Name for the dataset (will create folder classifier/data/dataset_name/) - optional when using --type and --size"
    )
    parser.add_argument(
        "--validation_ratio", 
        type=float, 
        default=0.2,
        help="Ratio of data to use for validation (default: 0.2)"
    )
    parser.add_argument(
        "--random_seed",
        type=int,
        default=42,
        help="Random seed for reproducible train/validation splits (default: 42)"
    )
    parser.add_argument(
        "--output_prefix",
        default="",
        help="Prefix for output filenames (default: empty)"
    )
    parser.add_argument(
        "--multiple",
        action="store_true",
        help="Create multiple dataset sizes (10%%, 50%%, 100%%) instead of single dataset"
    )
    parser.add_argument(
        "--type",
        help="Labeling method type (e.g., 'optimized', 'original') - used for folder naming"
    )
    parser.add_argument(
        "--size",
        help="Dataset size (e.g., '5000', 'full') - used for folder naming"
    )
    parser.add_argument(
        "--hf_configs",
        nargs="+",
        default=["gemini-1.5-flash-8b", "gemini-2.5-flash-lite"],
        help="Model configurations to load from HF dataset (default: both gemini-1.5-flash-8b and gemini-2.5-flash-lite)"
    )
    parser.add_argument(
        "--auto_split_hf",
        action="store_true",
        help="Automatically split HF dataset by labeling_model and labeling_strategy (creates separate folders like: count_model_strategy/)"
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.validation_ratio <= 0 or args.validation_ratio >= 1:
        raise ValueError(f"validation_ratio must be between 0 and 1, got {args.validation_ratio}")
    
    # Validate model_name requirement (not needed for auto_split_hf)
    if not args.auto_split_hf and not args.model_name:
        raise ValueError("--model_name is required unless using --auto_split_hf")
    
    # Determine dataset name based on parameters (skip if using auto_split_hf)
    if not args.auto_split_hf:
        if args.type and args.size:
            # Use new naming convention: size_model_type
            dataset_name = f"{args.size}_{args.model_name}_{args.type}"
        elif args.dataset_name:
            # Use provided dataset name
            dataset_name = args.dataset_name
        else:
            raise ValueError("Either --dataset_name OR both --type and --size must be provided (unless using --auto_split_hf)")
    else:
        dataset_name = "auto_split"  # Not used but needed to avoid errors
    
    # Determine input source and load data
    if args.input_paths:
        input_source = "input_paths"
        input_files = args.input_paths
        print(f"Using specified input paths: {input_files}")
        
        # Load and convert data
        print("\n" + "="*50)
        print("LOADING DATA FROM LOCAL FILES")
        print("="*50)
        annotated_data = load_annotated_data(input_files)
        
    elif args.input_folder:
        input_source = "input_folder"
        input_files = discover_json_files_in_folder(args.input_folder)
        if not input_files:
            print(f"Error: No JSON files found in folder: {args.input_folder}")
            return 1
        print(f"Using discovered files from folder: {args.input_folder}")
        
        # Load and convert data
        print("\n" + "="*50)
        print("LOADING DATA FROM LOCAL FOLDER")
        print("="*50)
        annotated_data = load_annotated_data(input_files)
        
    else:
        # Default to HuggingFace dataset
        input_source = "hf_dataset"
        input_files = None
        print(f"Using HuggingFace dataset (default): {args.hf_dataset}")
        print(f"Model configurations: {args.hf_configs}")
        
        # Load and convert data
        print("\n" + "="*50)
        print("LOADING DATA FROM HUGGINGFACE DATASET")
        print("="*50)
        annotated_data = load_annotated_data_from_hf(args.hf_dataset, args.hf_configs)
        
        # Check if auto-split is requested for HF dataset
        if args.auto_split_hf:
            print(f"Starting AUTO-SPLIT conversion with parameters:")
            print(f"  Input source: {input_source}")
            print(f"  HF dataset: {args.hf_dataset}")
            print(f"  HF configs: {args.hf_configs}")
            print(f"  Output directory: {args.output_dir}")
            print(f"  Validation ratio: {args.validation_ratio}")
            print(f"  Random seed: {args.random_seed}")
            print(f"  Auto-split by model+strategy: ENABLED")
            
            if not annotated_data:
                print("Error: No data loaded from HF dataset")
                return 1
            
            # Auto-split by model and strategy
            base_output_dir = Path(args.output_dir)
            created_datasets = create_auto_split_hf_datasets(
                annotated_data, base_output_dir, args.validation_ratio, args.random_seed
            )
            
            print(f"\n" + "="*60)
            print("AUTO-SPLIT SUMMARY")
            print("="*60)
            print(f"Created {len(created_datasets)} dataset combinations:")
            
            total_samples = 0
            for dataset_info in created_datasets:
                print(f"📂 {dataset_info['folder_name']}: {dataset_info['total_samples']} samples")
                print(f"  📄 Training: {dataset_info['train_samples']} samples")
                print(f"  📄 Validation: {dataset_info['val_samples']} samples")
                total_samples += dataset_info['total_samples']
            
            print(f"\nTotal samples across all combinations: {total_samples}")
            print(f"Output directory: {base_output_dir}")
            print(f"🎉 Auto-split completed successfully!")
            
            return 0
    
    # Continue with regular processing for non-auto-split cases
    print(f"Starting conversion with parameters:")
    print(f"  Input source: {input_source}")
    if input_files:
        print(f"  Input files: {len(input_files)} files")
    else:
        print(f"  HF dataset: {args.hf_dataset}")
        print(f"  HF configs: {args.hf_configs}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Model name: {args.model_name}")
    print(f"  Dataset name: {dataset_name}")
    print(f"  Type: {args.type if args.type else 'N/A'}")
    print(f"  Size: {args.size if args.size else 'N/A'}")
    print(f"  Validation ratio: {args.validation_ratio}")
    print(f"  Random seed: {args.random_seed}")
    print(f"  Multiple datasets: {args.multiple}")
    
    if not annotated_data:
        print("Error: No data loaded from input source")
        return 1
    
    print("\n" + "="*50)
    print("CONVERTING TO CLASSIFIER FORMAT")
    print("="*50)
    classifier_data = convert_to_classifier_format(annotated_data)
    
    if not classifier_data:
        print("Error: No valid data after conversion")
        return 1
    
    # Balance labels
    print("\n" + "="*50)
    print("BALANCING LABELS")
    print("="*50)
    balanced_data = balance_labels(classifier_data)
    
    # Create proper folder structure: output_dir/dataset_name/
    base_output_dir = Path(args.output_dir)
    
    if args.multiple:
        # Create multiple datasets with different sizes
        created_datasets = create_multiple_datasets(
            balanced_data, base_output_dir, dataset_name, 
            args.validation_ratio, args.random_seed
        )
        
        print(f"\nMultiple dataset creation completed successfully!")
        print(f"Output directory: {base_output_dir}")
        
        for dataset_info in created_datasets:
            print(f"📂 {dataset_info['percentage']*100:.0f}% dataset ({dataset_info['actual_size']} samples): {dataset_info['path']}")
            print(f"  📄 Training data: {dataset_info['path'] / 'train.json'}")
            print(f"  📄 Validation data: {dataset_info['path'] / 'valid.json'}")
        
        # Print comprehensive analysis for the full balanced dataset
        print_comprehensive_dataset_info(balanced_data, dataset_name, "Full Balanced Dataset")
        
        # Print final summary
        print(f"\nMULTIPLE DATASETS SUMMARY:")
        print(f"Total balanced samples: {len(balanced_data)}")
        sizes_info = [f"{d['actual_size']} ({d['percentage']*100:.0f}%)" for d in created_datasets]
        print(f"Created {len(created_datasets)} different dataset sizes: {sizes_info}")
        
    else:
        # Split into train/validation (single dataset)
        print("\n" + "="*50)
        print("SPLITTING TRAIN/VALIDATION")
        print("="*50)
        train_data, val_data = split_train_validation(balanced_data, args.validation_ratio, args.random_seed)
        
        dataset_output_dir = base_output_dir / dataset_name
        dataset_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Simple filenames: train.json and valid.json
        train_path = dataset_output_dir / "train.json"
        val_path = dataset_output_dir / "valid.json"
        
        # Save results
        print("\n" + "="*50)
        print("SAVING RESULTS")
        print("="*50)
        
        # Save training data
        with open(train_path, 'w', encoding='utf-8') as f:
            json.dump(train_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(train_data)} training samples to: {train_path}")
        
        # Save validation data
        with open(val_path, 'w', encoding='utf-8') as f:
            json.dump(val_data, f, indent=2, ensure_ascii=False)
        print(f"Saved {len(val_data)} validation samples to: {val_path}")
        
        print(f"\nConversion completed successfully!")
        print(f"Dataset directory: {dataset_output_dir}")
        print(f"📄 Training data: {train_path}")
        print(f"📄 Validation data: {val_path}")
        
        # Print comprehensive dataset analysis
        print_comprehensive_dataset_info(balanced_data, dataset_name, "Balanced Dataset")
        print_comprehensive_dataset_info(train_data, dataset_name, "Training Split")
        print_comprehensive_dataset_info(val_data, dataset_name, "Validation Split")
    
    return 0


if __name__ == "__main__":
    exit(main())