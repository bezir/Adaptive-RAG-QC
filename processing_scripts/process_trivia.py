#!/usr/bin/env python3
"""
Process TriviaQA Dataset

This script processes the raw TriviaQA dataset and converts it to our standardized format.
The script handles:
1. Loading raw TriviaQA train and dev JSON files
2. Converting to our generic Reading Comprehension format
3. Processing positive, negative, and hard negative contexts
4. Sampling negative contexts to control dataset size
5. Writing processed data to JSONL files

Input Format:
- Raw TriviaQA JSON files with biencoder format
- Contains questions, answers, and various types of contexts

Output Format:
- Standardized JSONL format with:
  - question_id, question_text, answers_objects, contexts
  - Each context marked as supporting/non-supporting
"""

import os

from lib import read_jsonl, write_jsonl, read_json
import json
from tqdm import tqdm
import random
from typing import List, Dict, Tuple, Union, Any

random.seed(13370)  # Don't change.

def safe_sample(items: List[Any], count: int) -> List[Any]:
    """
    Safely sample from a list without exceeding its length.
    
    Args:
        items: List to sample from
        count: Desired number of samples
        
    Returns:
        List of sampled items (up to count, or all items if count > len(items))
    """
    count = min(count, len(items))
    return random.sample(items, count) if count > 0 else []


def write_trivia_instances_to_filepath(raw_instances, output_directory: str, set_name: str):
    """
    Convert raw TriviaQA instances to our standardized format and write to file.
    
    This function processes each raw TriviaQA instance by:
    1. Converting question and answers to our format
    2. Processing positive contexts (supporting evidence)
    3. Sampling and adding negative contexts (distractors)
    4. Sampling and adding hard negative contexts (challenging distractors)
    
    Args:
        raw_instances: List of raw TriviaQA instances from JSON
        output_directory: Path to output JSONL file
        set_name: Name of the dataset split (train/dev) for unique ID generation
    """

    print(f"Writing processed TriviaQA data to: {output_directory}")
    print(f"Processing {len(raw_instances)} instances")     
    
    with open(output_directory, "w") as output_file:

        for idx, raw_instance in tqdm(enumerate(raw_instances), desc=f"Processing {set_name}"):

            # Create standardized instance in our Generic RC Format
            processed_instance = {}
            processed_instance["dataset"] = "trivia"
            
            # Generate unique question ID using dataset, split, and index
            processed_instance["question_id"] = f'single_trivia_{set_name}_{idx}'
            processed_instance["question_text"] = raw_instance["question"]
            
            # Note: TriviaQA has level and type metadata that we're not currently using
            # processed_instance["level"] = raw_instance["level"]
            # processed_instance["type"] = raw_instance["type"]

            # Convert answers to our standardized format
            # TriviaQA provides multiple answer strings in "answers" field
            answers_object = {
                "number": "",  # TriviaQA doesn't have numeric answers
                "date": {"day": "", "month": "", "year": ""},  # TriviaQA doesn't have date answers
                "spans": raw_instance["answers"],  # List of acceptable answer strings
            }

            processed_instance["answers_objects"] = [answers_object]

            # Process contexts (passages) for this question
            lst_context = []
            context_id = 0

            # 1. Add all positive contexts (supporting evidence)
            for ctx in raw_instance['positive_ctxs']:
                dict_context = {}
                dict_context['idx'] = context_id
                dict_context['title'] = ctx['title'].strip()
                dict_context['paragraph_text'] = ctx['text'].strip()
                dict_context['is_supporting'] = True  # These contain the answer

                context_id += 1
                lst_context.append(dict_context)

            # 2. Sample and add negative contexts (random distractors)
            # Limit to 5 to prevent dataset from becoming too large
            lst_neg_ctxs = raw_instance['negative_ctxs']
            sampled_lst_neg_ctxs = safe_sample(lst_neg_ctxs, 5)

            for ctx in sampled_lst_neg_ctxs:
                dict_context = {}
                dict_context['idx'] = context_id
                dict_context['title'] = ctx['title'].strip()
                dict_context['paragraph_text'] = ctx['text'].strip()
                dict_context['is_supporting'] = False  # These don't contain the answer

                context_id += 1
                lst_context.append(dict_context)

            # 3. Sample and add hard negative contexts (challenging distractors)
            # These are semantically similar but don't contain the answer
            # Also limit to 5 for the same reason
            lst_hardneg_ctxs = raw_instance['hard_negative_ctxs']
            sampled_lst_hardneg_ctxs = safe_sample(lst_hardneg_ctxs, 5)

            for ctx in sampled_lst_hardneg_ctxs:
                dict_context = {}
                dict_context['idx'] = context_id
                dict_context['title'] = ctx['title'].strip()
                dict_context['paragraph_text'] = ctx['text'].strip()
                dict_context['is_supporting'] = False  # These don't contain the answer

                context_id += 1
                lst_context.append(dict_context)

            processed_instance["contexts"] = lst_context

            # Write instance to JSONL file
            output_file.write(json.dumps(processed_instance) + "\n")


if __name__ == "__main__":
    """
    Main processing pipeline for TriviaQA dataset.
    
    This script:
    1. Loads TriviaQA dataset from raw data directory
    2. Processes training, development, and test splits
    3. If test split doesn't exist, splits dev set into dev and test portions
    4. Applies context sampling and standardization
    5. Saves processed data in JSONL format
    """

    # Set up directory paths
    input_directory = os.path.join("raw_data", "trivia")
    output_directory = os.path.join("processed_data", "trivia")
    os.makedirs(output_directory, exist_ok=True)

    # Process training data
    print("=" * 50)
    print("Processing TriviaQA Training Data")
    print("=" * 50)
    
    output_filepath = os.path.join(output_directory, "train.jsonl")
    input_filepath = os.path.join(input_directory, f"biencoder-trivia-train.json")
    print(f"Loading training data from {input_filepath}... (this may take a few minutes)")
    raw_instances = read_json(input_filepath)
    print(f"Loaded {len(raw_instances)} training instances")
    write_trivia_instances_to_filepath(raw_instances, output_filepath, 'train')

    # Check if test set exists, if not, split dev set
    test_input_filepath = os.path.join(input_directory, f"biencoder-trivia-test.json")
    if os.path.exists(test_input_filepath):
        # Process development data
        print("=" * 50)
        print("Processing TriviaQA Development Data")
        print("=" * 50)
        
        output_filepath = os.path.join(output_directory, "dev.jsonl")
        input_filepath = os.path.join(input_directory, f"biencoder-trivia-dev.json")
        print(f"Loading development data from {input_filepath}...")
        raw_instances = read_json(input_filepath)
        print(f"Loaded {len(raw_instances)} development instances")
        write_trivia_instances_to_filepath(raw_instances, output_filepath, 'dev')
    
        # Process test data
        print("=" * 50)
        print("Processing TriviaQA Test Data")
        print("=" * 50)
        
        output_filepath = os.path.join(output_directory, "test.jsonl")
        raw_instances = read_json(test_input_filepath)
        write_trivia_instances_to_filepath(raw_instances, output_filepath, 'test')
    else:
        # Split dev set into dev and test portions
        print("=" * 50)
        print("No official test set found. Splitting dev set into dev and test portions...")
        print("=" * 50)
        
        input_filepath = os.path.join(input_directory, f"biencoder-trivia-dev.json")
        raw_instances = read_json(input_filepath)
        
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
        
        # Process development data
        print("=" * 50)
        print("Processing TriviaQA Development Data")
        print("=" * 50)
        
        output_filepath = os.path.join(output_directory, "dev.jsonl")
        write_trivia_instances_to_filepath(dev_instances, output_filepath, 'dev')
        
        # Process test data
        print("=" * 50)
        print("Processing TriviaQA Test Data")
        print("=" * 50)
        
        output_filepath = os.path.join(output_directory, "test.jsonl")
        write_trivia_instances_to_filepath(test_instances, output_filepath, 'test')
    
    print("=" * 50)
    print("TriviaQA Processing Complete! Created train.jsonl, dev.jsonl, and test.jsonl")
    print("=" * 50)