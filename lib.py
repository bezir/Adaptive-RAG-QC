#!/usr/bin/env python3
"""
Utility Library for Adaptive RAG System

This library provides essential utility functions for the experiment infrastructure:
- Server address configuration and management
- Dataset inference from file paths
- File path resolution and management
- JSON/JSONL file I/O operations
- Text matching and retrieval utilities

The library serves as the foundation for experiment configuration, data processing,
and server communication across the entire RAG system.
"""

import os
import json
from typing import List, Dict
from pathlib import Path

import _jsonnet
from rapidfuzz import fuzz
import requests


def get_retriever_address(suffix: str = ""):
    """
    Get retriever server address configuration from .retriever_address.jsonnet file.
    
    Args:
        suffix (str): Optional suffix for address configuration keys
        
    Returns:
        dict: Dictionary containing 'host' and 'port' for retriever server
        
    Raises:
        Exception: If retriever address config file is not found
    """
    retriever_address_config_filepath = ".retriever_address.jsonnet"
    if not os.path.exists(retriever_address_config_filepath):
        raise Exception(f"Retriver address filepath ({retriever_address_config_filepath}) not available.")
    
    retriever_address_config_ = json.loads(_jsonnet.evaluate_file(retriever_address_config_filepath))
    retriever_address_config = {
        "host": retriever_address_config_["host" + suffix],
        "port": retriever_address_config_["port" + suffix],
    }
    return retriever_address_config


def get_llm_server_address(llm_port_num: str):
    """
    Get LLM server address configuration from .llm_server_address.jsonnet file.
    
    Args:
        llm_port_num (str): Port number to use for LLM server connection
        
    Returns:
        dict: Dictionary containing 'host' and 'port' for LLM server
        
    Raises:
        Exception: If LLM server address config file is not found
    """
    llm_server_address_config_filepath = ".llm_server_address.jsonnet"
    if not os.path.exists(llm_server_address_config_filepath):
        raise Exception(f"LLM Server address filepath ({llm_server_address_config_filepath}) not available.")
    
    llm_server_address_config = json.loads(_jsonnet.evaluate_file(llm_server_address_config_filepath))
    llm_server_address_config = {key: str(value) for key, value in llm_server_address_config.items()}
    
    # Override port with provided port number
    llm_server_address_config['port'] = llm_port_num
    return llm_server_address_config


def get_roscoe_server_address(suffix: str = ""):
    """
    Get ROSCOE server address configuration from .roscoe_server_address.jsonnet file.
    ROSCOE is used for reasoning evaluation and chain-of-thought assessment.
    
    Args:
        suffix (str): Optional suffix for address configuration keys
        
    Returns:
        dict: Dictionary containing 'host' and 'port' for ROSCOE server
        
    Raises:
        Exception: If ROSCOE server address config file is not found
    """
    roscoe_server_address_config_filepath = ".roscoe_server_address.jsonnet"
    if not os.path.exists(roscoe_server_address_config_filepath):
        raise Exception(f"Retriver address filepath ({roscoe_server_address_config_filepath}) not available.")
    
    roscoe_server_address_config_ = json.loads(_jsonnet.evaluate_file(roscoe_server_address_config_filepath))
    roscoe_server_address_config = {
        "host": roscoe_server_address_config_["host" + suffix],
        "port": roscoe_server_address_config_["port" + suffix],
    }
    return roscoe_server_address_config


def infer_dataset_from_file_path(file_path: str) -> str:
    """
    Infer dataset name from file path by matching against known dataset names.
    
    Supported datasets:
    - hotpotqa: Multi-hop reasoning dataset
    - 2wikimultihopqa: Wikipedia multi-hop QA dataset
    - musique: Multi-hop reasoning with unanswerable questions
    - nq: Natural Questions dataset
    - trivia: TriviaQA dataset
    - squad: Stanford Question Answering Dataset
    
    Args:
        file_path (str): Path to the file containing dataset
        
    Returns:
        str: Inferred dataset name
        
    Raises:
        Exception: If no dataset matches or multiple datasets match
    """
    matching_datasets = []
    file_path = str(file_path)
    
    # List of supported datasets
    supported_datasets = [
        "hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad'
    ]
    
    for dataset in supported_datasets:
        if dataset.lower() in file_path.lower():
            matching_datasets.append(dataset)
    
    if not matching_datasets:
        raise Exception(f"Dataset couldn't be inferred from {file_path}. No matches found.")
    if len(matching_datasets) > 1:
        raise Exception(f"Dataset couldn't be inferred from {file_path}. Multiple matches found.")
    
    return matching_datasets[0]


def infer_source_target_prefix(config_filepath: str, evaluation_path: str) -> str:
    """
    Generate source-target prefix for cross-dataset evaluation scenarios.
    
    This function creates a prefix that indicates when a model trained on one dataset
    is being evaluated on a different dataset (e.g., "hotpotqa_to_musique__").
    
    Args:
        config_filepath (str): Path to the experiment configuration file
        evaluation_path (str): Path to the evaluation dataset
        
    Returns:
        str: Source-target prefix in format "source_to_target__"
    """
    source_dataset = infer_dataset_from_file_path(config_filepath)
    target_dataset = infer_dataset_from_file_path(evaluation_path)
    source_target_prefix = "_to_".join([source_dataset, target_dataset]) + "__"
    return source_target_prefix


def get_config_file_path_from_name_or_path(experiment_name_or_path: str) -> str:
    """
    Resolve experiment configuration file path from either a name or full path.
    
    If given a name (without .jsonnet extension), searches for matching config files.
    If given a path (with .jsonnet extension), returns the path directly.
    
    Args:
        experiment_name_or_path (str): Either experiment name or full path to config
        
    Returns:
        str: Resolved path to the configuration file
        
    Raises:
        Exception: If config file cannot be found or multiple matches exist
    """
    if not experiment_name_or_path.endswith(".jsonnet"):
        # It's an experiment name - search for matching config file
        assert (
            len(experiment_name_or_path.split(os.path.sep)) == 1
        ), "Experiment name shouldn't contain any path separators."
        
        # Search for config files matching the experiment name
        matching_result = list(Path(".").rglob("**/*" + experiment_name_or_path + ".jsonnet"))
        matching_result = [
            _result
            for _result in matching_result
            if os.path.splitext(os.path.basename(_result))[0] == experiment_name_or_path
        ]
        
        # Filter out backup files
        matching_result = [i for i in matching_result if 'backup' not in str(i)]
        
        assert len(matching_result) == 1 
        
        if len(matching_result) != 1:
            exit(f"Couldn't find one matching path with the given name ({experiment_name_or_path}).")
        config_filepath = matching_result[0]
    else:
        # It's already a file path
        config_filepath = experiment_name_or_path
    
    return config_filepath


# JSON/JSONL File I/O Utilities

def read_json(file_path: str) -> Dict:
    """
    Read and parse a JSON file.
    
    Args:
        file_path (str): Path to the JSON file
        
    Returns:
        Dict: Parsed JSON data
    """
    with open(file_path, "r") as file:
        instance = json.load(file)
    return instance


def read_jsonl(file_path: str) -> List[Dict]:
    """
    Read and parse a JSONL (JSON Lines) file.
    
    Args:
        file_path (str): Path to the JSONL file
        
    Returns:
        List[Dict]: List of parsed JSON objects, one per line
    """
    with open(file_path, "r") as file:
        instances = [json.loads(line.strip()) for line in file.readlines() if line.strip()]
    return instances


def write_json(instance: Dict, file_path: str):
    """
    Write a dictionary to a JSON file.
    
    Args:
        instance (Dict): Dictionary to write to file
        file_path (str): Path where to write the JSON file
    """
    with open(file_path, "w") as file:
        json.dump(instance, file)


def write_jsonl(instances: List[Dict], file_path: str):
    """
    Write a list of dictionaries to a JSONL (JSON Lines) file.
    
    Args:
        instances (List[Dict]): List of dictionaries to write
        file_path (str): Path where to write the JSONL file
    """
    with open(file_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


def find_matching_paragraph_text(corpus_name: str, original_paragraph_text: str) -> str:
    """
    Find matching paragraph text in the retrieval corpus using fuzzy matching.
    
    This function uses the retriever server to find the most similar paragraph
    in the specified corpus and returns it if the similarity is above threshold.
    
    Args:
        corpus_name (str): Name of the corpus to search in
        original_paragraph_text (str): Original paragraph text to match
        
    Returns:
        dict or None: Dictionary with 'title' and 'paragraph_text' if match found,
                     None if no good match or retrieval fails
    """
    # Get retriever server configuration
    retriever_address_config = get_retriever_address()
    retriever_host = str(retriever_address_config["host"])
    retriever_port = str(retriever_address_config["port"])

    # Prepare retrieval request parameters
    params = {
        "query_text": original_paragraph_text,
        "retrieval_method": "retrieve_from_elasticsearch",
        "max_hits_count": 1,
        "corpus_name": corpus_name,
    }

    # Make retrieval request
    url = retriever_host.rstrip("/") + ":" + str(retriever_port) + "/retrieve"
    result = requests.post(url, json=params)

    # Handle retrieval failure
    if not result.ok:
        print("WARNING: Something went wrong in the retrieval. Skiping this mapping.")
        return None

    result = result.json()
    retrieval = result["retrieval"]

    # Validate corpus name in results
    for item in retrieval:
        assert item["corpus_name"] == corpus_name

    # Extract retrieved information
    retrieved_title = retrieval[0]["title"]
    retrieved_paragraph_text = retrieval[0]["paragraph_text"]

    # Calculate similarity using fuzzy matching
    match_ratio = fuzz.partial_ratio(original_paragraph_text, retrieved_paragraph_text)
    
    # Return match if similarity is above threshold (95%)
    if match_ratio > 95:
        return {"title": retrieved_title, "paragraph_text": retrieved_paragraph_text}
    else:
        print(f"WARNING: Couldn't map the original paragraph text to retrieved one ({match_ratio}).")
        return None
