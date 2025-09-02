#!/usr/bin/env python3
"""
HotpotQA Dataset Processing Script

This script processes the HotpotQA dataset from the Hugging Face datasets library
and converts it into a standardized format for the Adaptive RAG system.

HotpotQA is a multi-hop reading comprehension dataset that requires reasoning
across multiple paragraphs to answer questions. The dataset contains:
- Questions requiring multi-hop reasoning
- Supporting facts (evidence paragraphs)
- Distractor paragraphs (non-relevant context)
- Different difficulty levels (easy, medium, hard)
- Different question types (comparison, bridge, etc.)

The script transforms the raw HotpotQA format into a unified format compatible
with the experiment framework, including:
- Standardized question-answer pairs
- Context paragraphs with supporting/non-supporting labels
- Metadata preservation (level, type, etc.)
"""

import os
import json
from collections import Counter
from typing import List, Dict

from tqdm import tqdm
from datasets import load_dataset


def write_hotpotqa_instances_to_filepath(instances: List[Dict], full_filepath: str):
    """
    Process and write HotpotQA instances to a JSONL file in standardized format.
    
    This function transforms raw HotpotQA instances into a unified format that includes:
    - Generic reading comprehension structure
    - Context paragraphs with supporting facts labels
    - Answer objects in standardized format
    - Metadata preservation (question level, type, etc.)
    
    Args:
        instances (List[Dict]): List of raw HotpotQA instances from the dataset
        full_filepath (str): Output file path for processed instances
    """
    # Token limit for paragraph text (to prevent excessive memory usage)
    max_num_tokens = 1000  # clip later.

    # Track distribution of supporting context counts (hop sizes)
    hop_sizes = Counter()
    
    print(f"Writing in: {full_filepath}")
    with open(full_filepath, "w") as full_file:
        for raw_instance in tqdm(instances):

            # Transform to generic reading comprehension format
            processed_instance = {}
            processed_instance["dataset"] = "hotpotqa"
            processed_instance["question_id"] = raw_instance["id"]
            processed_instance["question_text"] = raw_instance["question"]
            processed_instance["level"] = raw_instance["level"]  # easy, medium, hard
            processed_instance["type"] = raw_instance["type"]    # comparison, bridge, etc.

            # Standardize answer format
            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }
            processed_instance["answers_objects"] = [answers_object]

            # Extract context and supporting facts
            raw_context = raw_instance.pop("context")
            supporting_titles = raw_instance.pop("supporting_facts")["title"]

            # Create mappings between titles and paragraph texts
            title_to_paragraph = {
                title: "".join(text) for title, text in zip(raw_context["title"], raw_context["sentences"])
            }
            paragraph_to_title = {
                "".join(text): title for title, text in zip(raw_context["title"], raw_context["sentences"])
            }

            # Identify gold (supporting) paragraphs
            gold_paragraph_texts = [title_to_paragraph[title] for title in supporting_titles]
            gold_paragraph_texts = set(list(gold_paragraph_texts))

            # Prepare all paragraph texts (remove duplicates)
            paragraph_texts = ["".join(paragraph) for paragraph in raw_context["sentences"]]
            paragraph_texts = list(set(paragraph_texts))

            # Build context objects with supporting labels
            processed_instance["contexts"] = [
                {
                    "idx": index,
                    "title": paragraph_to_title[paragraph_text].strip(),
                    "paragraph_text": paragraph_text.strip(),
                    "is_supporting": paragraph_text in gold_paragraph_texts,  # Label for multi-hop reasoning
                }
                for index, paragraph_text in enumerate(paragraph_texts)
            ]

            # Count supporting contexts for hop-size analysis
            supporting_contexts = [context for context in processed_instance["contexts"] if context["is_supporting"]]
            hop_sizes[len(supporting_contexts)] += 1

            # Truncate paragraph texts to token limit
            for context in processed_instance["contexts"]:
                context["paragraph_text"] = " ".join(context["paragraph_text"].split(" ")[:max_num_tokens])

            # Write processed instance to file
            full_file.write(json.dumps(processed_instance) + "\n")

    # Print hop-size distribution for analysis
    print(f"Hop-sizes: {str(hop_sizes)}")


if __name__ == "__main__":
    """
    Main processing pipeline for HotpotQA dataset.
    
    This script:
    1. Loads the HotpotQA dataset from Hugging Face
    2. Processes training, development, and test splits
    3. For test split: splits validation set into dev and test portions
    4. Saves them in standardized JSONL format
    """
    
    # Load HotpotQA dataset with distractor setting (includes non-supporting paragraphs)
    print("Loading HotpotQA dataset from Hugging Face... (this may take a few minutes)")
    dataset = load_dataset("hotpot_qa", "distractor")
    print(f"Loaded HotpotQA dataset - Train: {len(dataset['train'])}, Validation: {len(dataset['validation'])}")

    # Create output directory
    directory = os.path.join("processed_data", "hotpotqa")
    os.makedirs(directory, exist_ok=True)

    # Process training set
    print("Processing HotpotQA training set...")
    processed_full_filepath = os.path.join(directory, "train.jsonl")
    write_hotpotqa_instances_to_filepath(dataset["train"], processed_full_filepath)

    # Split validation set into dev and test portions (since no official test set exists)
    print("Splitting validation set into dev and test portions...")
    validation_instances = list(dataset["validation"])
    print(f"Loaded {len(validation_instances)} validation instances")
    
    # Use fixed random seed for reproducible splits
    import random
    random.seed(13370)
    random.shuffle(validation_instances)
    
    # Split validation: 60% for dev, 40% for test
    split_point = int(len(validation_instances) * 0.6)
    dev_instances = validation_instances[:split_point]
    test_instances = validation_instances[split_point:]
    
    print(f"Created dev split: {len(dev_instances)} instances")
    print(f"Created test split: {len(test_instances)} instances")

    # Process development set
    print("Processing HotpotQA development set...")
    processed_full_filepath = os.path.join(directory, "dev.jsonl")
    write_hotpotqa_instances_to_filepath(dev_instances, processed_full_filepath)
    
    # Process test set
    print("Processing HotpotQA test set...")
        processed_full_filepath = os.path.join(directory, "test.jsonl")
    write_hotpotqa_instances_to_filepath(test_instances, processed_full_filepath)
    
    print("HotpotQA processing completed! Created train.jsonl, dev.jsonl, and test.jsonl")
