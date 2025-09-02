#!/usr/bin/env python3
"""
SQuAD (Stanford Question Answering Dataset) Processing Script

This script processes the SQuAD dataset and converts it into a standardized
format for the Adaptive RAG system.

SQuAD is a reading comprehension dataset that:
- Contains questions based on Wikipedia articles
- Focuses on extractive question answering
- Includes positive, negative, and hard negative contexts (DPR format)
- Requires single-hop reasoning within a single paragraph
- Uses crowd-sourced question-answer pairs

The script transforms the raw SQuAD format into a unified format compatible
with the experiment framework, including:
- Standardized question-answer pairs
- Context sampling from positive/negative/hard-negative pools
- Proper supporting/non-supporting labels
- Balanced context distribution for training
"""

import os

from lib import read_jsonl, write_jsonl, read_json
import json
from tqdm import tqdm
import random
from typing import List, Dict, Tuple, Union, Any

# Fixed random seed for reproducible context sampling
random.seed(13370)  # Don't change - ensures consistent sampling across runs


def safe_sample(items: List[Any], count: int) -> List[Any]:
    """
    Safely sample items from a list, handling cases where list is smaller than requested count.
    
    Args:
        items (List[Any]): List of items to sample from
        count (int): Number of items to sample
        
    Returns:
        List[Any]: Sampled items (up to the requested count or list length)
    """
    count = min(count, len(items))
    return random.sample(items, count) if count > 0 else []


def write_squad_instances_to_filepath(raw_instances, output_directory: str, set_name: str):
    """
    Process and write SQuAD instances to a JSONL file in standardized format.
    
    This function:
    1. Transforms raw SQuAD instances into unified format
    2. Samples contexts from positive, negative, and hard-negative pools
    3. Assigns proper supporting/non-supporting labels
    4. Balances context distribution for effective training
    
    Args:
        raw_instances: List of raw SQuAD instances from the dataset
        output_directory (str): Output file path for processed instances
        set_name (str): Dataset split name (train/dev) for ID generation
    """
    
    print(f"Writing in: {output_directory}")
    print(f"Processing {len(raw_instances)} instances")
    
    with open(output_directory, "w") as output_file:

        for idx, raw_instance in tqdm(enumerate(raw_instances)):

            # Transform to standardized reading comprehension format
            processed_instance = {}
            processed_instance["dataset"] = "squad"
            processed_instance["question_id"] = 'single_squad_'+set_name+'_'+str(idx)  # Unique ID generation
            processed_instance["question_text"] = raw_instance["question"]

            # Standardize answer format (SQuAD can have multiple answer spans)
            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": raw_instance["answers"],  # Multiple possible answers
            }
            processed_instance["answers_objects"] = [answers_object]

            # Build context list with proper sampling strategy
            lst_context = []
            context_id = 0

            # Add all positive contexts (supporting evidence)
            for ctx in raw_instance['positive_ctxs']:
                dict_context = {}
                dict_context['idx'] = context_id
                dict_context['title'] = ctx['title'].strip()
                dict_context['paragraph_text'] = ctx['text'].strip()
                dict_context['is_supporting'] = True  # These contexts support the answer

                context_id = context_id + 1
                lst_context.append(dict_context)

            # Sample negative contexts (non-supporting but related)
            lst_neg_ctxs = raw_instance['negative_ctxs']
            sampled_lst_neg_ctxs = safe_sample(lst_neg_ctxs, 5)  # Sample 5 negative contexts

            for ctx in sampled_lst_neg_ctxs:
                dict_context = {}
                dict_context['idx'] = context_id
                dict_context['title'] = ctx['title'].strip()
                dict_context['paragraph_text'] = ctx['text'].strip()
                dict_context['is_supporting'] = False  # These contexts don't support the answer

                context_id = context_id + 1
                lst_context.append(dict_context)

            # Sample hard negative contexts (challenging distractors)
            lst_hardneg_ctxs = raw_instance['hard_negative_ctxs']
            sampled_lst_hardneg_ctxs = safe_sample(lst_hardneg_ctxs, 5)  # Sample 5 hard negatives

            for ctx in sampled_lst_hardneg_ctxs:
                dict_context = {}
                dict_context['idx'] = context_id
                dict_context['title'] = ctx['title'].strip()
                dict_context['paragraph_text'] = ctx['text'].strip()
                dict_context['is_supporting'] = False  # Hard negatives also don't support the answer

                context_id = context_id + 1
                lst_context.append(dict_context)

            # Assign processed contexts to the instance
            processed_instance["contexts"] = lst_context

            # Write processed instance to output file
            output_file.write(json.dumps(processed_instance) + "\n")


if __name__ == "__main__":
    """
    Main processing pipeline for SQuAD dataset.
    
    This script:
    1. Loads SQuAD dataset from raw data directory
    2. Processes training, development, and test splits
    3. If test split doesn't exist, splits dev set into dev and test portions
    4. Applies context sampling and standardization
    5. Saves processed data in JSONL format
    """
    
    input_directory = os.path.join("raw_data", "squad")
    output_directory = os.path.join("processed_data", "squad")
    os.makedirs(output_directory, exist_ok=True)

    # Process training set
    print("Processing SQuAD training set...")
    output_filepath = os.path.join(output_directory, "train.jsonl")
    input_filepath = os.path.join(input_directory, f"biencoder-squad1-train.json")
    print(f"Loading training data from {input_filepath}... (this may take a few minutes)")
    raw_instances = read_json(input_filepath)
    print(f"Loaded {len(raw_instances)} training instances")
    write_squad_instances_to_filepath(raw_instances, output_filepath, 'train')

    # Check if test set exists, if not, split dev set
    test_input_filepath = os.path.join(input_directory, f"biencoder-squad1-test.json")
    if os.path.exists(test_input_filepath):
        # Process development set
        print("Processing SQuAD development set...")
        output_filepath = os.path.join(output_directory, "dev.jsonl")
        input_filepath = os.path.join(input_directory, f"biencoder-squad1-dev.json")
        print(f"Loading development data from {input_filepath}...")
        raw_instances = read_json(input_filepath)
        print(f"Loaded {len(raw_instances)} development instances")
        write_squad_instances_to_filepath(raw_instances, output_filepath, 'dev')
    
        # Process test set
        print("Processing SQuAD test set...")
        output_filepath = os.path.join(output_directory, "test.jsonl")
        print(f"Loading test data from {test_input_filepath}...")
        raw_instances = read_json(test_input_filepath)
        print(f"Loaded {len(raw_instances)} test instances")
        write_squad_instances_to_filepath(raw_instances, output_filepath, 'test')
    else:
        # Split dev set into dev and test portions
        print("No official test set found. Splitting dev set into dev and test portions...")
        input_filepath = os.path.join(input_directory, f"biencoder-squad1-dev.json")
        print(f"Loading development data from {input_filepath}...")
        raw_instances = read_json(input_filepath)
        print(f"Loaded {len(raw_instances)} development instances")
        
        # Use fixed random seed for reproducible splits
        import random
        random.seed(13370)
        random.shuffle(raw_instances)
        
        # Split dev: 60% for dev, 40% for test
        split_point = int(len(raw_instances) * 0.6)
        dev_instances = raw_instances[:split_point]
        test_instances = raw_instances[split_point:]
        
        print(f"Created dev split: {len(dev_instances)} instances")
        print(f"Created test split: {len(test_instances)} instances")
        
        # Process development set
        print("Processing SQuAD development set...")
        output_filepath = os.path.join(output_directory, "dev.jsonl")
        write_squad_instances_to_filepath(dev_instances, output_filepath, 'dev')
        
        # Process test set  
        print("Processing SQuAD test set...")
        output_filepath = os.path.join(output_directory, "test.jsonl")
        write_squad_instances_to_filepath(test_instances, output_filepath, 'test')
    
    print("SQuAD processing completed! Created train.jsonl, dev.jsonl, and test.jsonl")