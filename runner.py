#!/usr/bin/env python3
"""
Experiment Runner Wrapper for Adaptive RAG System

This script provides a convenient wrapper around run.py to simplify experimentation
with different RAG systems, models, and datasets. It handles command-line argument
parsing, experiment naming, and dynamic command construction for various operations.

Supported systems:
- ircot: Interactive Retrieval with Chain-of-Thought
- ircot_qa: Interactive Retrieval with Chain-of-Thought for QA
- oner: One-step Retrieval
- oner_qa: One-step Retrieval for QA
- nor_qa: No Retrieval for QA (baseline)

Supported models:
- gemini-2.5-flash-lite
- gemini-1.5-flash-8b

Supported datasets:
- hotpotqa: Multi-hop reasoning dataset
- 2wikimultihopqa: Wikipedia multi-hop QA dataset
- musique: Multi-hop reasoning with unanswerable questions
- nq: Natural Questions dataset
- trivia: TriviaQA dataset
- squad: Stanford Question Answering Dataset
"""

import argparse
import json
import os
import subprocess


def main():
    """
    Main function that parses command-line arguments and constructs the appropriate
    run.py command for the specified experiment configuration.
    """
    parser = argparse.ArgumentParser(description="Wrapper around run.py to make experimentation easier.")
    
    # Core experiment configuration arguments
    parser.add_argument("system", type=str, choices=("ircot", "ircot_qa", "oner", "oner_qa", "nor_qa"))
    parser.add_argument("model", type=str, choices=("gemini-2.5-flash-lite", "gemini-1.5-flash-8b"))
    
    # Dataset configuration - supports both single datasets and cross-dataset evaluation
    all_datasets = ["hotpotqa", "2wikimultihopqa", "musique", 'nq', 'trivia', 'squad']
    all_datasets += ["_to_".join([dataset_a, dataset_b]) for dataset_a in all_datasets for dataset_b in all_datasets]
    parser.add_argument("dataset", type=str, choices=all_datasets)
    
    # Command to execute
    parser.add_argument(
        "command",
        type=str,
        help="command",
        choices={
            "print",           # Print experiment configuration
            "write",           # Write experiment configuration files
            "verify",          # Verify experiment setup
            "predict",         # Run prediction on dataset
            "evaluate",        # Evaluate predictions
            "track",           # Track experiment progress
            "summarize",       # Summarize experiment results
            "ground_truth_check",  # Check ground truth alignment
            "backup",          # Backup experiment data
            "print_backup",    # Print backup information
            "recover_backup",  # Recover from backup
            "delete_predictions",  # Delete prediction files
        },
    )
    
    # LLM server configuration
    parser.add_argument(
        "--llm_port_num",
        type=str,
        help="Port number for LLM server connection",
        default="8010",
    )
    
    # Prompt configuration
    parser.add_argument(
        "--prompt_set",
        type=str,
        help="Prompt set to use for experiments",
        choices={"1", "2", "3", "aggregate"},
        default="1",
    )
    
    # Execution control flags
    parser.add_argument("--dry_run", action="store_true", default=False, help="Print commands without executing")
    parser.add_argument("--use_backup", action="store_true", default=False, help="Use backup data for experiments")
    parser.add_argument("--skip_evaluation_path", action="store_true", default=False, help="Skip evaluation path setup")
    parser.add_argument("--eval_test", action="store_true", default=False, help="Evaluate on test set instead of dev set")
    parser.add_argument("--sample_size", type=int, help="Number of samples to use for evaluation")
    parser.add_argument("--best", action="store_true", default=False, help="Use best hyperparameters")
    parser.add_argument("--skip_if_exists", action="store_true", default=False, help="Skip evaluation if results already exist")
    parser.add_argument("--only_print", action="store_true", default=False, help="Only print results for evaluation")
    parser.add_argument("--force", action="store_true", default=False, help="Force prediction even if results exist")
    parser.add_argument("--official", action="store_true", default=False, help="Use official evaluation metrics")
    
    args = parser.parse_args()

    # Handle cross-dataset evaluation (e.g., train on hotpotqa, evaluate on musique)
    if "_to_" in args.dataset:
        train_dataset, eval_dataset = args.dataset.split("_to_")
    else:
        train_dataset = eval_dataset = args.dataset

    # Generate experiment name based on system, model, and dataset
    experiment_name = "_".join([args.system, args.model.replace("-", "_"), args.dataset])
    instantiation_scheme = args.system

    # Determine dataset split (test or dev with sample size)
    set_name = "test" if args.eval_test else 'dev_' + str(args.sample_size)
    
    # Build base run command
    run_command_array = [
        f"python run.py {args.command} {experiment_name} --instantiation_scheme {instantiation_scheme} --prompt_set {args.prompt_set} --set_name {set_name} --llm_port_num {args.llm_port_num}",
    ]

    # Add best hyperparameters flag for relevant commands
    if args.command in ("write", "predict", "evaluate", "print", "summarize") and args.best:
        run_command_array += ["--best"]

    # Add no-diff flag for write command to avoid showing differences
    if args.command == "write":
        run_command_array += ["--no_diff"]

    # Add evaluation path for commands that need it
    if (
        args.command in ("predict", "evaluate", "track", "summarize", "ground_truth_check")
        and not args.skip_evaluation_path
    ) or args.best:
        set_name = "test" if args.eval_test else 'dev_' + str(args.sample_size)
        evaluation_path = os.path.join("processed_data", eval_dataset, f"{set_name}_subsampled.jsonl")
        run_command_array += [f"--evaluation_path {evaluation_path}"]

    # Add variable replacements for cross-dataset evaluation
    if (
        args.command in ("predict", "summarize") or (args.command == "write" and args.best)
    ) and train_dataset != eval_dataset:
        variable_replacements = {"retrieval_corpus_name": f'"{eval_dataset}"'}
        variable_replacements_str = json.dumps(variable_replacements).replace(" ", "")
        run_command_array += ["--variable_replacements", f"'{variable_replacements_str}'"]

    # Add skip_if_exists and silent flags for prediction
    if args.command in ("predict"):
        run_command_array.append("--skip_if_exists --silent")

    # Add backup flag for commands that support it
    if args.command in ("predict", "evaluate", "track", "summarize", "ground_truth_check") and args.use_backup:
        run_command_array += ["--use_backup"]

    # Add force flag for prediction command
    if args.command == "predict" and args.force:
        run_command_array += ["--force"]

    # Add skip_if_exists flag for evaluation command
    if args.command == "evaluate" and args.skip_if_exists:
        run_command_array += ["--skip_if_exists"]

    # Add only_print flag for evaluation command
    if args.command == "evaluate" and args.only_print:
        run_command_array += ["--only_print"]

    # Add official evaluation flag for evaluation and summarization
    if args.command in ("evaluate", "summarize") and args.official:
        run_command_array += ["--official"]

    # Sanity check: ensure train dataset is in experiment name
    assert train_dataset in experiment_name

    # Print experiment information
    print("", flush=True)
    message = f"Experiment Name: {experiment_name}"
    print("*" * len(message), flush=True)
    print(message, flush=True)
    print("*" * len(message), flush=True)

    # Execute the constructed command
    run_command_str = " ".join(run_command_array)
    print(run_command_str + "\n", flush=True)
    if not args.dry_run:
        subprocess.call(run_command_str, shell=True)


if __name__ == "__main__":
    main()
