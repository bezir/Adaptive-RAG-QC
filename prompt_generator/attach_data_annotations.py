#!/usr/bin/env python3
"""
Data Annotation Attachment System for Adaptive RAG

This script implements a sophisticated annotation attachment system that links experimental
instances with their corresponding reasoning annotations for the Adaptive RAG system.
It handles complex matching between annotation references and actual paragraph content
using both local context and external retrieval mechanisms.

Key Components:
1. Text Matching Engine
   - Fuzzy text similarity scoring using RapidFuzz
   - Exact title matching with case normalization
   - High-precision threshold (95%) for content integrity
   - Robust handling of formatting variations

2. Retrieval Integration
   - Elasticsearch backend for paragraph resolution
   - Multi-corpus support (HotpotQA, 2WikiMultiHopQA, MuSiQue, IIRC)
   - Configurable retrieval parameters
   - Deduplication mechanisms for result quality

3. Annotation Processing Pipeline
   - Question text alignment with fuzzy matching
   - Answer extraction from chain-of-thought reasoning
   - Multi-stage paragraph matching (local → retrieval)
   - Comprehensive validation and error reporting

4. Data Validation System
   - Consistency checking across data sources
   - Answer alignment verification
   - Reasoning step completeness validation
   - Experimental instance integrity checks

Scientific Workflow:
The annotation attachment process follows a systematic pipeline designed to ensure
high-quality training data for few-shot learning experiments:

1. Load annotation data from Jsonnet configuration files
2. Build annotation indices for efficient lookup
3. For each experimental instance:
   a. Match with corresponding annotation by question_id
   b. Validate question text similarity using fuzzy matching
   c. Extract and validate answer from reasoning chain
   d. Resolve paragraph content references through multi-stage matching
   e. Complete instance with validated annotation data
4. Persist annotated results for downstream processing

The system supports complex multi-hop reasoning scenarios where reasoning steps
reference specific paragraphs that must be resolved to actual content through
both local context search and external retrieval when necessary.
"""

from typing import List, Dict
import argparse
import json
import re
import os

import requests
from rapidfuzz import fuzz
from tqdm import tqdm
import _jsonnet

from lib import get_retriever_address


def read_jsonl(file_path: str) -> List[Dict]:
    """
    Read and parse JSON Lines format files for experimental data loading.
    
    This function provides standardized JSONL file reading capabilities for
    experimental data processing pipelines. Ensures consistent data loading
    across different components of the annotation system.
    
    Args:
        file_path (str): Path to JSONL file containing experimental data
        
    Returns:
        List[Dict]: List of parsed JSON objects representing experimental instances
    """
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_jsonl(instances: List[Dict], file_path: str):
    """
    Write experimental instances to JSON Lines format with progress indication.
    
    This function provides standardized JSONL file writing capabilities with
    progress indication for experimental data persistence. Ensures consistent
    data serialization across different components of the annotation system.
    
    Args:
        instances (List[Dict]): List of experimental instances to write
        file_path (str): Output file path for JSONL data
    """
    print(f"Writing {len(instances)} lines in: {file_path}")
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def _find_matching_paragraphs(query_title: str, query_text_substring: str, db_paragraphs: List[Dict]) -> List[Dict]:
    """
    Find paragraphs matching specified title and text content using fuzzy matching algorithms.
    
    This function implements sophisticated paragraph matching that combines exact title
    matching with fuzzy text similarity scoring. Essential for linking annotation
    references to actual paragraph content in knowledge bases.
    
    Scientific Rationale:
    - Exact title matching ensures domain accuracy
    - Fuzzy text matching handles minor formatting variations
    - High similarity threshold (95%) ensures content integrity
    - Prevents false matches through strict validation
    
    Algorithm:
    1. Iterate through database paragraphs
    2. Check for exact title match (case-insensitive)
    3. Compute fuzzy similarity score for text content
    4. Accept matches only if both conditions are satisfied
    5. Return all valid matches for further processing
    
    Args:
        query_title (str): Target paragraph title for exact matching
        query_text_substring (str): Target text content for fuzzy matching
        db_paragraphs (List[Dict]): Database of candidate paragraphs
        
    Returns:
        List[Dict]: List of paragraphs matching both title and text criteria
        
    Matching Criteria:
        - Title: Exact match (case-insensitive, whitespace-normalized)
        - Text: Partial fuzzy match with >95% similarity score
        
    Text Similarity:
        Uses RapidFuzz's partial_ratio which finds the best matching substring
        and computes normalized similarity score. Robust to minor formatting
        differences while maintaining high precision.
    """
    # Input validation to ensure proper data types
    assert isinstance(query_title, str)
    assert isinstance(query_text_substring, str)

    # Initialize collection for matching paragraphs
    matching_paragraphs = []
    
    # Iterate through database paragraphs for matching
    for paragraph in db_paragraphs:
        # Exact title matching with case and whitespace normalization
        title_exact_match = query_title.lower().strip() == paragraph["title"].lower().strip()
        
        # Fuzzy text matching with high similarity threshold
        paragraph_text_match_score = fuzz.partial_ratio(query_text_substring, paragraph["paragraph_text"])

        # Accept paragraph only if both conditions satisfied
        if title_exact_match and paragraph_text_match_score > 95:
            matching_paragraphs.append(paragraph)

    return matching_paragraphs


class Retriever:
    """
    Elasticsearch-based retrieval system for finding relevant paragraphs in knowledge corpora.
    
    This class implements a sophisticated retrieval interface that connects to an
    Elasticsearch backend for finding paragraphs matching specified criteria. 
    Supports domain-specific retrieval across multiple knowledge bases with
    precise control over retrieval parameters.
    
    Scientific Capabilities:
    - Elasticsearch integration for efficient large-scale retrieval
    - Multi-corpus support for domain-specific knowledge bases
    - Configurable retrieval parameters for experimental control
    - Deduplication mechanisms to prevent redundant results
    - Robust error handling for system reliability
    
    Supported Corpora:
    - HotpotQA: Multi-hop reasoning over Wikipedia
    - 2WikiMultiHopQA: Complex multi-hop questions
    - MuSiQue: Multi-step reasoning questions
    - IIRC: Incomplete information reading comprehension
    
    Retrieval Process:
    1. Accept query parameters and constraints
    2. Format request for Elasticsearch backend
    3. Execute retrieval via HTTP API
    4. Process and deduplicate results
    5. Return structured paragraph objects
    """
    
    def __init__(self, host: str, port: int, source_corpus_name: str) -> None:
        """
        Initialize retrieval system with backend configuration.
        
        Args:
            host (str): Elasticsearch host address
            port (int): Elasticsearch port number
            source_corpus_name (str): Target knowledge corpus identifier
        """
        self._host = host
        self._port = port
        
        # Validate corpus name against supported knowledge bases
        assert source_corpus_name in ("hotpotqa", "2wikimultihopqa", "musique", "iirc")
        self._source_corpus_name = source_corpus_name

    def retrieve(self, allowed_title: str, query_text: str) -> List[Dict]:
        """
        Retrieve relevant paragraphs from knowledge corpus using Elasticsearch.
        
        This method implements sophisticated retrieval that finds paragraphs
        matching specified title and text content. Uses Elasticsearch's
        ranking capabilities to find the most relevant content.
        
        Args:
            allowed_title (str): Specific Wikipedia title to search within
            query_text (str): Text content to match against paragraphs
            
        Returns:
            List[Dict]: List of retrieved paragraph objects with metadata
            
        Retrieval Parameters:
            - Method: Elasticsearch-based retrieval
            - Max hits: 50 (configurable for recall vs. precision trade-off)
            - Document type: Paragraph-level text
            - Title filtering: Exact title matching
            - Corpus: Dataset-specific knowledge base
            
        Deduplication Strategy:
            Prevents duplicate results by tracking both title and text content.
            Essential for maintaining result quality and preventing redundancy.
            
        Error Handling:
            Raises exceptions for failed retrieval requests to ensure system
            reliability and prevent silent failures in experimental pipelines.
        """
        # Configure retrieval parameters for Elasticsearch
        params = {
            "retrieval_method": "retrieve_from_elasticsearch",  # Specify retrieval backend
            "query_text": query_text,                          # Text content to match
            "max_hits_count": 50,                             # Maximum results to return
            "document_type": "paragraph_text",                # Document granularity
            "allowed_titles": [allowed_title],                # Title filtering constraint
            "corpus_name": self._source_corpus_name,          # Knowledge corpus selection
        }
        
        # Construct API endpoint URL
        url = self._host.rstrip("/") + ":" + str(self._port) + "/retrieve"
        
        # Execute retrieval request via HTTP API
        result = requests.post(url, json=params)

        # Initialize collections for deduplication
        selected_titles = []
        selected_paras = []
        unique_retrieval = []
        
        # Process retrieval results with validation and deduplication
        if result.ok:
            # Parse JSON response from Elasticsearch
            result = result.json()
            retrieval = result["retrieval"]

            # Process each retrieved paragraph
            for retrieval_item in retrieval:
                # Validate corpus consistency
                if retrieval_item["corpus_name"] != self._source_corpus_name:
                    raise Exception(
                        f"The retrieved corpus name {retrieval_item['corpus_name']} "
                        f"doesn't match {self._source_corpus_name}."
                    )

                # Apply deduplication based on title and text content
                # This was changed post-hoc to include both conditions for robustness
                if (  
                    retrieval_item["title"] in selected_titles and retrieval_item["paragraph_text"] in selected_paras
                ):
                    continue

                # Add to result collection with deduplication tracking
                selected_titles.append(retrieval_item["title"])
                selected_paras.append(retrieval_item["paragraph_text"])
                unique_retrieval.append(retrieval_item)
        else:
            # Handle retrieval failures with informative error messages
            raise Exception("Retrieval request did not succeed.")

        return unique_retrieval


def attach_data_annotations(
    processed_data: List[Dict],
    annotations: List[Dict],
    retriever: Retriever,
) -> List[Dict]:
    """
    Attach comprehensive data annotations to processed experimental instances.
    
    This function implements a sophisticated annotation attachment system that
    links experimental instances with their corresponding reasoning annotations.
    Handles complex matching between annotation references and actual paragraph
    content using both local context and external retrieval.
    
    Scientific Process:
    1. Validation and consistency checking across data sources
    2. Question text alignment with fuzzy matching tolerance
    3. Answer extraction and validation from reasoning chains
    4. Paragraph content resolution through multi-stage matching
    5. Experimental instance completion with full annotation data
    
    Key Features:
    - Fuzzy text matching for question alignment
    - Answer extraction from chain-of-thought reasoning
    - Multi-stage paragraph matching (local → retrieval)
    - Comprehensive validation and error reporting
    - Dataset-specific handling for different corpus types
    
    Args:
        processed_data (List[Dict]): Original experimental instances
        annotations (List[Dict]): Reasoning annotations to attach
        retriever (Retriever): Retrieval system for paragraph resolution
        
    Returns:
        List[Dict]: Annotated experimental instances with complete reasoning data
        
    Annotation Structure:
        Each annotation contains:
        - question_id: Unique identifier for matching
        - question_text: Potentially updated question text
        - reasoning_steps: Chain of reasoning with evidence
        - answer: Final answer extracted from reasoning
        
    Matching Algorithm:
        1. Build annotation index by question_id
        2. For each processed instance:
           a. Find corresponding annotation
           b. Validate question text similarity
           c. Extract reasoning steps and answer
           d. Resolve paragraph content references
           e. Complete instance with annotation data
    """
    # Build annotation index for efficient lookup
    id_to_annotation = {annotation["question_id"]: annotation for annotation in annotations}
    
    # Validate annotation uniqueness
    assert len(id_to_annotation) == len(annotations), "Looks like there are duplicate qid annotations."

    # Initialize collection for annotated instances
    annotated_processed_data = []
    
    # Process each experimental instance with progress indication
    for instance in tqdm(processed_data):
        # Lookup corresponding annotation
        annotation = id_to_annotation.pop(instance["question_id"], None)

        # Skip instances without annotations
        if not annotation:
            continue

        # Validate question ID consistency
        assert instance["question_id"] == annotation["question_id"]
        question_id = instance["question_id"]

        # Validate question text similarity with fuzzy matching
        question_match_score = fuzz.ratio(instance["question_text"], annotation["question_text"])
        if question_match_score < 95:
            print(
                "WARNING the following questions may not be same. Check manually : "
                f'{instance["question_text"]} >>> {annotation["question_text"]}'
            )

        # Update instance with annotation data
        instance["question_text"] = annotation["question_text"]
        instance["reasoning_steps"] = annotation["reasoning_steps"]
        reasoning_steps = instance["reasoning_steps"]

        # Extract answer from final reasoning step using regex pattern
        answer_regex = r".*answer is: (.*)\."
        assert re.match(answer_regex, reasoning_steps[-1]["cot_sent"])
        extracted_answer = re.sub(answer_regex, r"\1", reasoning_steps[-1]["cot_sent"])

        # Validate answer consistency with gold standard
        gold_answer = instance["answers_objects"][0]["spans"][0]
        if extracted_answer != gold_answer:
            print(
                f"WARNING: The extracted answer doesn't perfectly match the gold answer. "
                f"{extracted_answer} != {gold_answer}"
            )

        # Ensure answer consistency across reasoning paradigms
        # This ensures CoT answers match direct answers for fair comparison
        gold_answer = extracted_answer
        instance["answers_objects"][0]["spans"][0] = gold_answer

        # Determine context source based on dataset characteristics
        if retriever._source_corpus_name == "iirc":

            # Use "pinned_contexts" for fixed reading comprehension contexts
            context_paragraphs = instance.get("pinned_contexts", [])
        else:
            # Other datasets: "contexts" contains full paragraph content
            context_paragraphs = instance["contexts"]

        # Validate paragraph formatting consistency
        for paragraph in instance["contexts"]:
            assert not paragraph["paragraph_text"].startswith("Title: ")
            assert not paragraph["paragraph_text"].startswith("Wikipedia Title: ")

        # Process reasoning steps to resolve paragraph content
        text_populated_reasoning_steps = []
        for reasoning_step in reasoning_steps:
            # Validate reasoning step structure
            assert len(reasoning_step["paragraphs"]) == 1  # Single paragraph per step
            gold_paragraph = reasoning_step["paragraphs"][0]

            # Validate annotation completeness
            assert "title" in gold_paragraph, f"Field `title` missing in annotation for {question_id}"
            assert "text_substring" in gold_paragraph, f"Field `text_substring` missing in annotation for {question_id}"

            # Handle empty references (reasoning steps without specific evidence)
            if not gold_paragraph["title"] or not gold_paragraph["text_substring"]:
                assert not gold_paragraph["title"] and not gold_paragraph["text_substring"]
                gold_paragraph["paragraph_text"] = None
                text_populated_reasoning_steps.append(reasoning_step)
                continue

            # Phase 1: Attempt local context matching
            matching_paragraphs = _find_matching_paragraphs(
                gold_paragraph["title"], gold_paragraph["text_substring"], context_paragraphs
            )

            # Validate matching uniqueness
            assert len(matching_paragraphs) < 2

            # Phase 2: Fallback to retrieval-based matching
            if not matching_paragraphs:
                # Retrieve candidate paragraphs from knowledge base
                retrieved_paragraphs = retriever.retrieve(gold_paragraph["title"], gold_paragraph["text_substring"])
                # Apply matching to retrieved candidates
                matching_paragraphs = _find_matching_paragraphs(
                    gold_paragraph["title"], gold_paragraph["text_substring"], retrieved_paragraphs
                )

            # Handle unresolvable paragraph references
            if not matching_paragraphs:
                print("WARNING: Couldn't find any match for the annotated paragraph.")
                continue

            # Validate successful matching
            assert len(matching_paragraphs) == 1
            matching_paragraph = matching_paragraphs[0]

            # Validate title consistency
            assert gold_paragraph["title"].lower() == matching_paragraph["title"].lower()
            
            # Populate paragraph content from matched result
            gold_paragraph["paragraph_text"] = matching_paragraph["paragraph_text"]

            # Add completed reasoning step
            text_populated_reasoning_steps.append(reasoning_step)

        # Validate reasoning step completion
        assert len(text_populated_reasoning_steps) == len(reasoning_steps)
        
        # Update instance with populated reasoning steps
        instance["reasoning_steps"] = text_populated_reasoning_steps
        
        # Add completed instance to result collection
        annotated_processed_data.append(instance)

    return annotated_processed_data


def main():
    """
    Main execution function for the data annotation attachment pipeline.
    
    This function orchestrates the complete annotation attachment process,
    including data loading, retrieval system initialization, and annotation
    processing. Implements a robust pipeline for experimental data preparation.
    
    Pipeline Process:
    1. Parse command-line arguments for dataset selection
    2. Load annotation data from Jsonnet configuration files
    3. Load processed experimental data from JSONL files
    4. Initialize retrieval system with appropriate backend
    5. Execute annotation attachment with progress monitoring
    6. Persist annotated results for downstream processing
    
    Scientific Workflow:
    - Systematic data loading with validation
    - Configurable retrieval backend integration
    - Comprehensive annotation processing
    - Progress monitoring for large-scale processing
    - Standardized output formatting
    """
    # Parse command-line arguments for dataset selection
    parser = argparse.ArgumentParser(description="Attach annotations to the processed data.")
    parser.add_argument(
        "dataset_name", type=str, help="dataset_name", 
        choices={"hotpotqa", "2wikimultihopqa", "musique", "iirc"}
    )
    args = parser.parse_args()

    # Configure file paths following standardized project structure
    annotations_file_path = os.path.join("prompt_generator", "data_annotations", args.dataset_name + ".jsonnet")
    processed_data_file_path = os.path.join("processed_data", args.dataset_name, "train.jsonl")
    output_file_path = os.path.join("processed_data", args.dataset_name, "annotated_only_train.jsonl")

    # Load annotation data from Jsonnet configuration
    # Jsonnet provides programmatic configuration with enhanced readability
    annotations = json.loads(_jsonnet.evaluate_file(annotations_file_path))
    
    # Load processed experimental data
    processed_data = read_jsonl(processed_data_file_path)

    # Initialize retrieval system with backend configuration
    retriever_address_config = get_retriever_address()

    # Create retriever instance with dataset-specific configuration
    retriever = Retriever(
        host=retriever_address_config["host"],
        port=retriever_address_config["port"],
        source_corpus_name=args.dataset_name,
    )

    # Execute annotation attachment process
    attached_data_annotations = attach_data_annotations(processed_data, annotations, retriever)
    
    # Persist annotated results for downstream processing
    write_jsonl(attached_data_annotations, output_file_path)


if __name__ == "__main__":
    main()
