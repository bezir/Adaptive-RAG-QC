#!/usr/bin/env python3
"""
2WikiMultiHopQA Dataset Processing Script

This script processes the 2WikiMultiHopQA dataset and converts it into a standardized
format for the Adaptive RAG system.

2WikiMultiHopQA is a multi-hop reading comprehension dataset that:
- Requires reasoning across multiple Wikipedia passages
- Contains questions with complex multi-hop dependencies
- Includes supporting facts and evidence paragraphs
- Provides reasoning steps/evidence chains
- Uses Wikipedia as the knowledge source

The script transforms the raw 2WikiMultiHopQA format into a unified format compatible
with the experiment framework, including:
- Standardized question-answer pairs
- Context paragraphs with supporting/non-supporting labels
- Evidence/reasoning steps preservation
- Proper indexing and metadata handling
"""

import os

from lib import read_json, write_jsonl


def main():
    """
    Main processing pipeline for 2WikiMultiHopQA dataset.
    
    This function:
    1. Processes train, dev, and test splits of the 2WikiMultiHopQA dataset
    2. Transforms the raw format into standardized structure
    3. Handles supporting facts identification
    4. Preserves reasoning steps from evidence
    5. Saves processed data in JSONL format
    """
    
    # Process training, development, and test sets
    set_names = ["train", "dev", "test"]

    input_directory = os.path.join("raw_data", "2wikimultihopqa")
    output_directory = os.path.join("processed_data", "2wikimultihopqa")
    os.makedirs(output_directory, exist_ok=True)

    for set_name in set_names:
        print(f"Processing 2WikiMultiHopQA {set_name} set...")

        processed_instances = []

        # Define input and output file paths
        input_filepath = os.path.join(input_directory, f"{set_name}.json")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        # Read raw instances from JSON file
        print(f"Loading {set_name} data from {input_filepath}... (this may take a few minutes)")
        raw_instances = read_json(input_filepath)
        print(f"Loaded {len(raw_instances)} {set_name} instances")

        for raw_instance in raw_instances:

            # Extract basic question information
            question_id = raw_instance["_id"]
            question_text = raw_instance["question"]
            raw_contexts = raw_instance["context"]

            # Identify supporting titles from supporting facts
            supporting_titles = list(set([e[0] for e in raw_instance["supporting_facts"]]))

            # Process evidence into reasoning steps
            evidences = raw_instance["evidences"]
            reasoning_steps = [" ".join(evidence) for evidence in evidences]

            # Transform context paragraphs into standardized format
            processed_contexts = []
            for index, raw_context in enumerate(raw_contexts):
                title = raw_context[0]  # Wikipedia article title
                paragraph_text = " ".join(raw_context[1]).strip()  # Paragraph sentences
                is_supporting = title in supporting_titles  # Whether this context supports the answer
                
                processed_contexts.append(
                    {
                        "idx": index,
                        "title": title.strip(),
                        "paragraph_text": paragraph_text,
                        "is_supporting": is_supporting,  # Label for multi-hop reasoning
                    }
                )

            # Standardize answer format
            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [raw_instance["answer"]],
            }
            answers_objects = [answers_object]

            # Build the processed instance in standardized format
            processed_instance = {
                "question_id": question_id,
                "question_text": question_text,
                "answers_objects": answers_objects,
                "contexts": processed_contexts,
                "reasoning_steps": reasoning_steps,  # Evidence chain for multi-hop reasoning
            }

            processed_instances.append(processed_instance)

        # Write processed instances to JSONL output file
        write_jsonl(processed_instances, output_filepath)
        print(f"Processed {len(processed_instances)} instances for {set_name}")

    print("2WikiMultiHopQA processing completed! Created train.jsonl, dev.jsonl, and test.jsonl")


if __name__ == "__main__":
    main()
