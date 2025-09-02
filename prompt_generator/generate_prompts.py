#!/usr/bin/env python3
"""
Prompt Generation Orchestration for Adaptive RAG Experiments

This script implements a comprehensive prompt generation pipeline for systematic
evaluation of retrieval-augmented generation systems. It creates experimental
prompts across multiple dimensions including reasoning paradigms, context
configurations, and model architectures.

Core Experimental Design:
1. Factorial Experimental Framework
   - Full factorial design across model types, reasoning paradigms, and context levels
   - Systematic variation of distractor counts for robustness evaluation
   - Comprehensive coverage of experimental conditions for fair comparison

2. Task-Specific Prompt Generation
   - Question Answering: Direct and chain-of-thought reasoning evaluation
   - Open Retrieval: Zero-shot information retrieval assessment
   - Context Manipulation: Various context provision strategies

3. Model Architecture Support
   - Codex: Code-trained models with strong reasoning capabilities
   - FLAN-T5: Instruction-tuned models optimized for natural language tasks
   - Architecture-specific prompt formatting for optimal performance

4. Dataset Compatibility
   - HotpotQA: Multi-hop reasoning over Wikipedia
   - 2WikiMultiHopQA: Complex multi-hop question answering
   - MuSiQue: Multi-step reasoning with unanswerable questions

The script generates prompts following established practices in few-shot learning
and prompt engineering research, enabling systematic evaluation of different
retrieval strategies and reasoning approaches in controlled experimental settings.
"""

import os
import argparse
from typing import List, Dict

from prompt_generator.common import QAPromptGenerator, NoContextOpenRetrieverPromptGenerator


def get_qa_prompt_generator_args_and_names(dataset_name: str) -> List[Dict]:
    """
    Generate comprehensive configuration sets for Question Answering (QA) prompt generation experiments.
    
    This function implements a systematic experimental design for multi-hop reasoning tasks,
    creating configurations that span different reasoning paradigms, contextual information levels,
    and model architectures. The approach follows established practices in few-shot learning
    and prompt engineering for large language models.
    
    Scientific Rationale:
    - Evaluates both direct answer generation and chain-of-thought (CoT) reasoning
    - Tests robustness across different context configurations (no context vs. gold+distractors)
    - Compares performance between different model architectures (Codex vs. FLAN-T5)
    - Uses controlled distractor injection to measure reasoning robustness
    
    Args:
        dataset_name (str): The target dataset identifier for domain-specific configurations
        
    Returns:
        List[Dict]: A comprehensive list of experimental configurations, each containing:
            - generator_args: Parameters for the prompt generator instantiation
            - name: Systematic filename following naming conventions
            - max_paragraph_tokens: Token limit for paragraph truncation (fixed at 250)
    
    Experimental Design:
    The function creates a full factorial design across multiple dimensions:
    1. Model Architecture: {Codex, FLAN-T5}
    2. Reasoning Type: {Direct, Chain-of-Thought}
    3. Context Type: {No Context, Gold with Distractors}
    4. Distractor Count: {0, 1, 2, 3} (when applicable)
    """
    # Token limit set to 250 based on empirical analysis of optimal context length
    # This balance ensures sufficient context while maintaining model attention efficiency
    max_paragraph_tokens = 250  # keep it fixed 250.
    
    # Initialize experimental configuration storage
    prompt_generator_args_and_names = []
    
    # Model architectures under evaluation - representing different paradigms:
    # - Codex: Code-trained model with strong reasoning capabilities
    # - FLAN-T5: Instruction-tuned model optimized for natural language tasks
    # - Gemini-2.5-Flash-Lite: Google's instruction-tuned model with advanced reasoning
    # - Qwen: Alibaba's instruction-tuned model for multilingual understanding
    model_names = ["codex", "flan_t5", "gemini-2.5-flash-lite", "qwen"]
    
    # Iterate through all model architectures to ensure comprehensive evaluation
    for model_name in model_names:
        
        # Experimental factors for reasoning methodology:
        # - "direct": Traditional answer generation without explicit reasoning steps
        # - "cot": Chain-of-thought prompting encouraging step-by-step reasoning
        for qa_type in ("direct", "cot"):
            
            # Context configuration experimental conditions:
            # - "no": Zero-shot setting with no additional context
            # - "gold_with_distractors": Few-shot with gold standard + noise injection
            for context_type in ("no", "gold_with_distractors"):
                
                # Distractor count experimental design:
                # - 0 distractors: Pure gold standard context
                # - 1-3 distractors: Increasing noise levels to test robustness
                distractor_counts = (0,) if context_type == "no" else (1, 2, 3)
                
                # Iterate through distractor configurations to evaluate noise robustness
                for distractor_count in distractor_counts:
                    
                    # Logical constraint enforcement: no-context implies zero distractors
                    if distractor_count == 0:
                        assert context_type == "no"
                    
                    # Construct experimental configuration dictionary
                    prompt_generator_args = {
                        "qa_type": qa_type,              # Reasoning methodology
                        "context_type": context_type,    # Context provision strategy
                        "distractor_count": distractor_count,  # Noise injection level
                        "model_name": model_name,        # Target model architecture
                    }
                    
                    # Dataset-specific configuration adjustments:
                    # No special handling needed for current datasets
                    
                    # Generate systematic filename following experimental naming convention
                    # Format: {context_description}_{reasoning_type}_{model_name}.txt
                    context_type_ = f"gold_with_{distractor_count}_distractors"
                    if not distractor_count:
                        context_type_ = "no"
                    
                    # Systematic filename construction for experimental organization
                    prompt_name = f"{context_type_}_context_{qa_type}_qa_{model_name}.txt"
                    
                    # Package complete experimental configuration
                    prompt_generator_args_and_names.append(
                        {
                            "generator_args": prompt_generator_args,
                            "name": prompt_name,
                            "max_paragraph_tokens": max_paragraph_tokens,
                        }
                    )
    
    return prompt_generator_args_and_names


def get_no_context_open_retrieval_prompt_generator_args_and_names(dataset_name: str) -> List[Dict]:
    """
    Generate configurations for open-domain retrieval experiments without contextual grounding.
    
    This function creates experimental setups for evaluating Large Language Models' ability
    to perform information retrieval tasks in zero-shot settings. The approach tests whether
    models can identify relevant information sources without explicit context provision.
    
    Scientific Motivation:
    Open-domain retrieval represents a fundamental challenge in AI systems, requiring models
    to leverage their parametric knowledge to identify relevant information sources. This
    experimental design evaluates retrieval capabilities across different model architectures
    without providing explicit contextual hints.
    
    Experimental Paradigm:
    - Zero-shot retrieval: No example retrievals provided
    - Cross-architecture evaluation: Tests generalization across model types
    - Parametric knowledge utilization: Relies on pre-training knowledge
    
    Args:
        dataset_name (str): Target dataset for domain-specific retrieval evaluation
        
    Returns:
        List[Dict]: Experimental configurations for retrieval evaluation containing:
            - generator_args: Model-specific parameters
            - name: Systematic filename for experimental organization
            - max_paragraph_tokens: Consistent token limit (250)
    """
    # Maintain consistent token limits across all experimental conditions
    max_paragraph_tokens = 250
    prompt_generator_args_and_names = []
    
    # Configuration for Codex-based retrieval evaluation
    # Codex's code-training may provide advantages in structured retrieval tasks
    prompt_name = "no_context_open_llm_retrieval_codex.txt"
    prompt_generator_args_and_names.append(
        {"generator_args": {"model_name": "codex"}, "name": prompt_name, "max_paragraph_tokens": max_paragraph_tokens}
    )
    
    # Configuration for FLAN-T5-based retrieval evaluation
    # FLAN-T5's instruction-tuning may provide advantages in natural language retrieval
    prompt_name = "no_context_open_llm_retrieval_flan_t5.txt"
    prompt_generator_args_and_names.append(
        {"generator_args": {"model_name": "flan_t5"}, "name": prompt_name, "max_paragraph_tokens": max_paragraph_tokens}
    )
    
    # Configuration for Gemini-2.5-Flash-Lite-based retrieval evaluation
    # Gemini-2.5-Flash-Lite's advanced reasoning may provide advantages in complex retrieval tasks
    prompt_name = "no_context_open_llm_retrieval_gemini.txt"
    prompt_generator_args_and_names.append(
        {"generator_args": {"model_name": "gemini-2.5-flash-lite"}, "name": prompt_name, "max_paragraph_tokens": max_paragraph_tokens}
    )
    
    # Configuration for Qwen-based retrieval evaluation
    # Qwen's multilingual training may provide advantages in diverse retrieval scenarios
    prompt_name = "no_context_open_llm_retrieval_qwen.txt"
    prompt_generator_args_and_names.append(
        {"generator_args": {"model_name": "qwen"}, "name": prompt_name, "max_paragraph_tokens": max_paragraph_tokens}
    )
    
    return prompt_generator_args_and_names


def main():
    """
    Main execution function implementing the comprehensive prompt generation pipeline.
    
    This function orchestrates the entire experimental setup generation process,
    handling dataset-specific configurations and task-specific prompt generation.
    The pipeline follows established practices in systematic experimentation for
    natural language processing research.
    
    Experimental Pipeline:
    1. Parse command-line arguments for dataset selection
    2. Configure input/output paths following project structure
    3. Determine applicable tasks based on dataset characteristics
    4. Generate comprehensive experimental configurations
    5. Instantiate and execute prompt generators
    6. Persist generated prompts for downstream evaluation
    
    Scientific Approach:
    - Systematic experimental design across multiple dimensions
    - Reproducible configuration generation
    - Standardized file organization for experimental tracking
    - Comprehensive coverage of experimental conditions
    """
    
    # Command-line argument parsing for dataset selection
    # Supports major multi-hop reasoning datasets in the literature
    parser = argparse.ArgumentParser(description="Generate prompts.")
    parser.add_argument(
        "dataset_name", type=str, help="dataset_name", 
        choices={"hotpotqa", "2wikimultihopqa", "musique"}
    )
    args = parser.parse_args()
    
    # Configure input data path following standardized project structure
    # Uses annotated training data for few-shot example generation
    input_file_path = os.path.join("processed_data", args.dataset_name, "annotated_only_train.jsonl")
    
    # Configure output directory for systematic experimental organization
    output_directory = os.path.join("prompts", args.dataset_name)
    
    # Determine applicable tasks based on dataset characteristics
    # All current datasets support QA tasks
    task_names = ["qa"]
    
    # Iterate through all applicable tasks for comprehensive evaluation
    for task_name in task_names:
        
        # Task-specific configuration selection using strategy pattern
        if task_name == "qa":
            # Question Answering task configuration
            args_name_generator = get_qa_prompt_generator_args_and_names
            prompt_generator_cls = QAPromptGenerator
        elif task_name == "no_context_open_retrieval":
            # Open-domain retrieval task configuration
            args_name_generator = get_no_context_open_retrieval_prompt_generator_args_and_names
            prompt_generator_cls = NoContextOpenRetrieverPromptGenerator
        else:
            # Defensive programming: catch unsupported task configurations
            raise Exception(f"Invalid task_name {task_name}")
        
        # Generate all experimental configurations for the current task
        for prompt_args_and_name in args_name_generator(args.dataset_name):
            
            # Extract generator parameters and add input file path
            generator_args = prompt_args_and_name["generator_args"]
            generator_args["input_file_path"] = input_file_path
            
            # Instantiate task-specific prompt generator with experimental configuration
            prompt_generator = prompt_generator_cls(**generator_args)
            
            # Configure output file path following naming convention
            output_file_name = prompt_args_and_name["name"]
            output_file_path = os.path.join(output_directory, output_file_name)
            
            # Clean up configuration dictionary to prevent parameter leakage
            prompt_args_and_name.pop("generator_args")
            prompt_args_and_name.pop("name")
            prompt_args_and_name.pop("max_paragraph_tokens")
            
            # Defensive check for configuration completeness
            if prompt_args_and_name:
                raise Exception("Looks like prompt_args_and_name has extra unused args.")
            
            # Ensure output directory exists
            os.makedirs(output_directory, exist_ok=True)
            
            # Generate and persist prompts with progress indication
            print(f"Writing in {output_file_path}")
            with open(output_file_path, "w") as file:
                # Execute prompt generation and write to file
                file.write(prompt_generator.generate())


if __name__ == "__main__":
    main()
