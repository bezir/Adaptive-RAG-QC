#!/usr/bin/env python3
"""
MuSiQue Dataset Processing Script

This script processes the MuSiQue (Multi-hop Questions via Single-hop Question Composition)
dataset and converts it into a standardized format for the Adaptive RAG system.

MuSiQue is a multi-hop reading comprehension dataset that provides:
- Complex questions requiring multi-hop reasoning
- Question decomposition into sub-questions
- Supporting and non-supporting paragraphs
- Reasoning steps with intermediate answers
- Compositional structure showing how sub-questions combine

The script transforms the raw MuSiQue format into a unified format compatible
with the experiment framework, including:
- Standardized question-answer pairs
- Context paragraphs with supporting/non-supporting labels
- Reasoning steps reconstruction from question decomposition
- Intermediate answer tracking and substitution
"""

import os

from lib import read_jsonl, write_jsonl


def main():
    """
    Main processing pipeline for MuSiQue dataset.
    
    This function:
    1. Processes train, dev, and test splits of the MuSiQue dataset
    2. Transforms question decomposition into reasoning steps
    3. Handles intermediate answer substitution in sub-questions
    4. Standardizes the format for the experiment framework
    """
    
    # Process training, development, and test sets
    set_names = ["train", "dev", "test"]
    input_directory = os.path.join("raw_data", "musique")
    output_directory = os.path.join("processed_data", "musique")
    os.makedirs(output_directory, exist_ok=True)

    for set_name in set_names:
        print(f"Processing MuSiQue {set_name} set...")
        processed_instances = []

        # Define input and output file paths
        input_filepath = os.path.join(input_directory, f"musique_ans_v1.0_{set_name}.jsonl")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        # Read raw instances from the dataset
        print(f"Loading {set_name} data from {input_filepath}... (this may take a few minutes)")
        raw_instances = read_jsonl(input_filepath)
        print(f"Loaded {len(raw_instances)} {set_name} instances")

        for raw_instance in raw_instances:

            # Standardize answer format
            # Test set instances may not have an 'answer' key
            answer_span = raw_instance.get("answer", "")
            answers_object = {
                "number": "",
                "date": {"day": "", "month": "", "year": ""},
                "spans": [answer_span],
            }

            # Process question decomposition into reasoning steps
            # Test set instances may not have 'question_decomposition' field
            number_to_answer = {}  # Track intermediate answers for substitution
            sentences = []         # Store reasoning steps
            
            if "question_decomposition" in raw_instance:
                # Process decomposition for train/dev sets
                for index, reasoning_step in enumerate(raw_instance["question_decomposition"]):
                    number = index + 1
                    question = reasoning_step["question"]
                    
                    # Handle intermediate answer substitution in sub-questions
                    # Replace references like "#1", "#2" with actual answers from previous steps
                    # Only check for references to previously processed steps
                    for mentioned_number in range(1, number):  # Only check previous steps
                        if f"#{mentioned_number}" in reasoning_step["question"]:
                            if mentioned_number not in number_to_answer:
                                print(f"WARNING: mentioned_number {mentioned_number} not present in number_to_answer.")
                            else:
                                # Substitute the reference with the actual answer
                                question = question.replace(f"#{mentioned_number}", number_to_answer[mentioned_number])
                    
                    # Get the answer for this reasoning step
                    answer = reasoning_step["answer"]
                    number_to_answer[number] = answer
                    
                    # Create reasoning step sentence in "question >>>> answer" format
                    sentence = " >>>> ".join([question.strip(), answer.strip()])
                    sentences.append(sentence)
            else:
                # Test set instances don't have question decomposition
                sentences = []  # Empty reasoning steps for test instances

            # Build the processed instance in standardized format
            processed_instance = {
                "question_id": raw_instance["id"],
                "question_text": raw_instance["question"],
                "contexts": [
                    {
                        "idx": index,
                        "paragraph_text": paragraph["paragraph_text"].strip(),
                        "title": paragraph["title"].strip(),
                        # Test set paragraphs may not have 'is_supporting' field
                        "is_supporting": paragraph.get("is_supporting", False),  # Default to False for test set
                    }
                    for index, paragraph in enumerate(raw_instance["paragraphs"])
                ],
                "answers_objects": [answers_object],
                "reasoning_steps": sentences,  # Decomposed reasoning chain (empty for test set)
            }
            processed_instances.append(processed_instance)

        # Write processed instances to output file
        write_jsonl(processed_instances, output_filepath)
        print(f"Processed {len(processed_instances)} instances for {set_name}")

    print("MuSiQue processing completed! Created train.jsonl, dev.jsonl, and test.jsonl")


if __name__ == "__main__":
    main()
