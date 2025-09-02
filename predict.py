#!/usr/bin/env python3
"""
Prediction Script for Adaptive RAG System

This script runs configurable inference on specified experiment configurations and datasets.
It handles the complete prediction pipeline including:
- Setting up experiment directories and file paths
- Configuring environment variables for retriever and LLM server connections
- Running the inference process with appropriate parameters
- Automatically triggering evaluation after prediction
- Maintaining experiment reproducibility through git hashing and config backups

The script is designed to work with the commaqa.inference.configurable_inference module
and integrates with the broader experiment management system.
"""

import os
import shutil
import subprocess
import argparse

from lib import (
    get_retriever_address,
    get_llm_server_address,
    infer_source_target_prefix,
    get_config_file_path_from_name_or_path,
)


def get_git_hash() -> str:
    """
    Get the current git commit hash for reproducibility tracking.
    
    Returns:
        str: Current git commit hash
    """
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def main():
    """
    Main prediction function that orchestrates the entire prediction pipeline.
    
    This function:
    1. Parses command-line arguments
    2. Sets up prediction directory structure
    3. Configures server connections
    4. Runs inference with the specified configuration
    5. Triggers evaluation automatically
    6. Maintains experiment reproducibility records
    """
    parser = argparse.ArgumentParser(description="Run configurable_inference on given config and dataset.")
    
    # Core experiment configuration
    parser.add_argument("experiment_name_or_path", type=str, help="Experiment name or path to config file")
    parser.add_argument("evaluation_path", type=str, help="Path to evaluation dataset file")
    
    # Optional configuration parameters
    parser.add_argument(
        "--prediction-suffix", 
        type=str, 
        help="Optional suffix for the prediction directory name", 
        default=""
    )
    parser.add_argument("--dry-run", action="store_true", default=False, help="Print commands without executing")
    parser.add_argument("--skip-evaluation", type=str, default="", help="Skip evaluation step")
    parser.add_argument("--force", action="store_true", default=False, help="Force prediction even if results exist")
    parser.add_argument(
        "--variable-replacements",
        type=str,
        help="JSON string for jsonnet local variable replacements",
        default="",
    )
    parser.add_argument("--silent", action="store_true", help="Run in silent mode")
    
    # Required parameters for experiment tracking
    parser.add_argument('--set_name', type=str, help="Dataset split name (e.g., test)", required=True)
    parser.add_argument('--llm_port_num', type=str, help="LLM server port number", required=True)
    
    args = parser.parse_args()

    # Resolve experiment configuration file path
    config_filepath = get_config_file_path_from_name_or_path(args.experiment_name_or_path)
    experiment_name = os.path.splitext(os.path.basename(config_filepath))[0]
    
    # Create prediction directory structure
    prediction_directory = os.path.join("predictions", args.set_name, experiment_name + args.prediction_suffix)
    os.makedirs(prediction_directory, exist_ok=True)

    # Generate prediction file names with source-target prefix for cross-dataset evaluation
    prediction_filename = os.path.splitext(os.path.basename(args.evaluation_path))[0]
    prediction_filename = infer_source_target_prefix(config_filepath, args.evaluation_path) + prediction_filename
    prediction_filepath = os.path.join(prediction_directory, "prediction__" + prediction_filename + ".json")

    # Check if prediction already exists and is complete
    if os.path.exists(prediction_filepath) and not args.force:
        from run import is_experiment_complete

        metrics_file_path = os.path.join(prediction_directory, "evaluation_metrics__" + prediction_filename + ".json")
        if is_experiment_complete(config_filepath, prediction_filepath, metrics_file_path, args.variable_replacements):
            exit(f"The prediction_file_path {prediction_filepath} already exists and is complete. Pass --force.")

    # Set up environment variables for server connections
    env_variables = {}
    
    # Configure retriever server connection
    retriever_address = get_retriever_address()
    env_variables["RETRIEVER_HOST"] = str(retriever_address["host"])
    env_variables["RETRIEVER_PORT"] = str(retriever_address["port"])
    
    # Configure LLM server connection
    llm_server_address = get_llm_server_address(args.llm_port_num)
    env_variables["LLM_SERVER_HOST"] = str(llm_server_address["host"])
    env_variables["LLM_SERVER_PORT"] = str(llm_server_address["port"])

    # Build environment variables string for command execution
    env_variables_str = " ".join([f"{key}={value}" for key, value in env_variables.items()]).strip()

    # Construct the prediction command
    predict_command = " ".join(
        [
            env_variables_str,
            "python -m commaqa.inference.configurable_inference",
            f"--config {config_filepath}",
            f"--input {args.evaluation_path}",
            f"--output {prediction_filepath}",
        ]
    ).strip()

    # Add optional flags
    if args.silent:
        predict_command += " --silent"

    if args.variable_replacements:
        predict_command += f" --variable-replacements '{args.variable_replacements}'"

    print(f"Run predict_command: \n{predict_command}\n")

    # Execute prediction command
    if not args.dry_run:
        subprocess.call(predict_command, shell=True)

    # Save git hash for reproducibility
    git_hash_filepath = os.path.join(prediction_directory, "git_hash__" + prediction_filename + ".txt")

    # Backup configuration file for reproducibility
    backup_config_filepath = os.path.join(prediction_directory, "config__" + prediction_filename + ".jsonnet")
    shutil.copyfile(config_filepath, backup_config_filepath)

    # Run evaluation automatically after prediction (unless skipped)
    if not args.skip_evaluation:
        
        # Run standard evaluation
        evaluate_command = " ".join([
            "python evaluate.py", 
            str(config_filepath), 
            str(args.evaluation_path), 
            '--set_name', args.set_name, 
            '--llm_port_num', args.llm_port_num
        ]).strip()

        print(f"Run evaluate_command: \n{evaluate_command}\n")

        if not args.dry_run:
            subprocess.call(evaluate_command, shell=True)
        
        # Run official evaluation
        evaluate_command = " ".join([
            "python evaluate.py", 
            str(config_filepath), 
            str(args.evaluation_path), 
            "--official", 
            '--set_name', args.set_name, 
            '--llm_port_num', args.llm_port_num
        ]).strip()

        print(f"Run evaluate_command: \n{evaluate_command}\n")

        if not args.dry_run:
            subprocess.call(evaluate_command, shell=True)


if __name__ == "__main__":
    main()
