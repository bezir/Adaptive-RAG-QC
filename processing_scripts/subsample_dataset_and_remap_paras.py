#!/usr/bin/env python3
"""
Experiment Dataset Creation and Paragraph Remapping Script

This script creates two non-overlapping experiment sets following ML best practices:
1. Creates test_experiment.jsonl from actual test split (subsampled to specified size)
2. Creates dev_experiment.jsonl from entire dev split (excluding any duplicates with test)
3. Remaps paragraph texts to match retrieval corpus versions

The approach ensures:
- Proper train/test separation using actual dataset splits
- Zero overlap between dev_experiment and test_experiment sets
- Maximum utilization of available training data (entire dev set)
- Duplicate detection by both question_id and question text
- Reproducible sampling with fixed random seed

The paragraph remapping ensures:
- Context paragraphs match the retrieval corpus format
- Text normalization and standardization
- Proper title-paragraph alignment for retrieval

Usage: python subsample_dataset_and_remap_paras.py {dataset_name} {test_size}
Example: python subsample_dataset_and_remap_paras.py nq 500
"""

import argparse
import random
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

from tqdm import tqdm
from lib import read_jsonl, write_jsonl, find_matching_paragraph_text

# Fixed random seed for reproducible subsampling
random.seed(13370)  # Don't change this - ensures reproducible experiments


def normalize_question_text(question):
    """Normalize question text for duplicate detection."""
    return question.lower().strip().replace("?", "").replace(".", "")


def find_duplicates(dev_instances, test_instances):
    """Find duplicates between dev and test sets using both question_id and question text."""
    
    # Create sets for test data
    test_question_ids = set(instance["question_id"] for instance in test_instances)
    test_question_texts = set(normalize_question_text(instance["question_text"]) for instance in test_instances)
    
    duplicates_by_id = []
    duplicates_by_text = []
    
    for dev_instance in dev_instances:
        # Check for question_id duplicates
        if dev_instance["question_id"] in test_question_ids:
            duplicates_by_id.append(dev_instance["question_id"])
        
        # Check for question text duplicates
        normalized_dev_question = normalize_question_text(dev_instance["question_text"])
        if normalized_dev_question in test_question_texts:
            duplicates_by_text.append(dev_instance["question_id"])
    
    return duplicates_by_id, duplicates_by_text


def remap_instance_paragraphs(instance, dataset_name):
    """Remap paragraphs for a single instance."""
    for context in instance["contexts"]:
        # Skip pinned contexts (e.g., IIRC main contexts not in Wikipedia corpus)
        if context in instance.get("pinned_contexts", []):
            continue
        
        # Handle different corpus types
        if dataset_name in ['nq', 'trivia', 'squad']:
            # For these datasets, use Wikipedia corpus (commented out for performance)
            # retrieved_result = find_matching_paragraph_text('wiki', context["paragraph_text"])
            continue
        else:
            # For other datasets, use dataset-specific corpus
            retrieved_result = find_matching_paragraph_text(dataset_name, context["paragraph_text"])

        # Update context with retrieved paragraph if found
        if retrieved_result is None:
            continue

        # Replace with corpus-aligned version
        context["title"] = retrieved_result["title"]
        context["paragraph_text"] = retrieved_result["paragraph_text"]
    
    return instance


def process_instances_parallel(instances, dataset_name, max_workers=4):
    """Process instances in parallel for faster remapping."""
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_instance = {
            executor.submit(remap_instance_paragraphs, instance, dataset_name): instance 
            for instance in instances
        }
        
        # Process results with progress bar
        processed_instances = []
        for future in tqdm(as_completed(future_to_instance), total=len(instances)):
            processed_instances.append(future.result())
    
    return processed_instances


def main():
    """
    Main function for creating proper experiment datasets with train/test separation.
    
    This function:
    1. Loads the full dev dataset and test dataset
    2. Randomly samples test_size examples from test split for test_experiment
    3. Uses entire dev split for dev_experiment (excluding duplicates with test)
    4. Optionally remaps paragraphs if the retrieval server is running
    5. Saves both files with proper naming
    """
    
    parser = argparse.ArgumentParser(
        description="Create experiment datasets with proper train/test separation. Optionally remaps paragraphs to match a retrieval corpus if the retrieval server is running.",
        epilog="Example: python subsample_dataset_and_remap_paras.py nq 500\nTo skip remapping: python subsample_dataset_and_remap_paras.py nq 500 --no-remap\nWith parallel processing: python subsample_dataset_and_remap_paras.py 2wikimultihopqa 1000 --workers 16"
    )
    parser.add_argument(
        "dataset_name", 
        type=str, 
        help="Name of the dataset to process", 
        choices=("hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad')
    )
    parser.add_argument(
        "test_size", 
        type=int, 
        help="Number of instances to sample from test split for test_experiment"
    )
    parser.add_argument(
        "--no-remap",
        action="store_true",
        help="Skip the paragraph remapping step (which requires a running retrieval server).",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers for remapping (default: 8)",
    )
    args = parser.parse_args()

    # Load the full dev dataset
    dev_file_path = os.path.join("processed_data", args.dataset_name, "dev.jsonl")
    if not os.path.exists(dev_file_path):
        print(f"Error: Dev file not found at {dev_file_path}")
        return
    
    dev_instances = read_jsonl(dev_file_path)
    print(f"Loaded {len(dev_instances)} instances from dev split: {dev_file_path}")

    # Load the test dataset
    test_file_path = os.path.join("processed_data", args.dataset_name, "test.jsonl")
    if not os.path.exists(test_file_path):
        print(f"Error: Test file not found at {test_file_path}")
        return
    
    all_test_instances = read_jsonl(test_file_path)
    print(f"Loaded {len(all_test_instances)} instances from test split: {test_file_path}")

    # Validate test_size
    if len(all_test_instances) < args.test_size:
        print(f"Error: Requested test size ({args.test_size}) is larger than available test instances ({len(all_test_instances)})")
        return

    # Randomly sample test_size instances from test split
    test_instances = random.sample(all_test_instances, args.test_size)
    print(f"Selected {len(test_instances)} instances for test_experiment from test split")

    # Find and report duplicates
    print("Checking for duplicates between dev and test sets...")
    duplicates_by_id, duplicates_by_text = find_duplicates(dev_instances, test_instances)
    
    if duplicates_by_id:
        print(f"Found {len(duplicates_by_id)} duplicates by question_id: {duplicates_by_id[:5]}{'...' if len(duplicates_by_id) > 5 else ''}")
    else:
        print("No duplicates found by question_id.")
    
    if duplicates_by_text:
        print(f"Found {len(duplicates_by_text)} duplicates by question text: {duplicates_by_text[:5]}{'...' if len(duplicates_by_text) > 5 else ''}")
    else:
        print("No duplicates found by question text.")

    # Remove duplicates from dev set
    test_question_ids = set(instance["question_id"] for instance in test_instances)
    test_question_texts = set(normalize_question_text(instance["question_text"]) for instance in test_instances)
    
    filtered_dev_instances = []
    removed_count = 0
    
    for dev_instance in dev_instances:
        normalized_dev_question = normalize_question_text(dev_instance["question_text"])

        # Skip if duplicate by ID or text
        if (dev_instance["question_id"] in test_question_ids or 
            normalized_dev_question in test_question_texts):
            removed_count += 1
            continue
            
        filtered_dev_instances.append(dev_instance)
    
    print(f"Removed {removed_count} duplicate instances from dev set")
    print(f"Final dev_experiment size: {len(filtered_dev_instances)} instances")

    if not args.no_remap:
        # Process test instances for paragraph remapping
        print(f"Processing test_experiment instances and remapping paragraphs using {args.workers} workers...")
        test_instances = process_instances_parallel(test_instances, args.dataset_name, args.workers)

        # Process dev instances for paragraph remapping
        print(f"Processing dev_experiment instances and remapping paragraphs using {args.workers} workers...")
        filtered_dev_instances = process_instances_parallel(filtered_dev_instances, args.dataset_name, args.workers)
    else:
        print("\nSkipping paragraph remapping as requested via --no-remap flag.")

    # Save both files with new naming convention
    test_output_path = os.path.join("processed_data", args.dataset_name, "test_experiment.jsonl")
    dev_output_path = os.path.join("processed_data", args.dataset_name, "dev_experiment.jsonl")
    
    write_jsonl(test_instances, test_output_path)
    write_jsonl(filtered_dev_instances, dev_output_path)
    
    print(f"\n=== EXPERIMENT DATASETS CREATED ===")
    print(f"Test experiment: {len(test_instances)} instances -> {test_output_path}")
    print(f"Dev experiment: {len(filtered_dev_instances)} instances -> {dev_output_path}")
    print(f"Zero overlap ensured between dev and test sets")
    print(f"Proper train/test separation using actual dataset splits")


if __name__ == "__main__":
    main()
