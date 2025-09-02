#!/usr/bin/env python3
"""
Prompt Loader for Adaptive-RAG

This module provides functionality to dynamically load few-shot examples from 
the existing prompt infrastructure.
"""

import os
import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

class PromptLoader:
    """
    Loads and extracts few-shot examples from existing prompt files
    
    This class leverages the existing prompt infrastructure in /prompts/ and /base_configs/
    to dynamically load examples instead of using hardcoded ones.
    """
    
    def __init__(self, prompts_base_path: str = None):
        """
        Initialize the prompt loader
        
        Args:
            prompts_base_path: Base path to prompts directory (defaults to auto-detect)
        """
        if prompts_base_path is None:
            # Auto-detect the prompts path relative to this file
            current_dir = Path(__file__).parent
            self.prompts_base_path = current_dir.parent.parent / "prompts"
        else:
            self.prompts_base_path = Path(prompts_base_path)
            
        self.logger = logging.getLogger(__name__)
        
        # Validate prompts directory exists
        if not self.prompts_base_path.exists():
            raise FileNotFoundError(f"Prompts directory not found: {self.prompts_base_path}")
    
    def get_few_shot_examples(self, 
                             model_name: str, 
                             dataset: str = None, 
                             context_type: str = "gold_with_1_distractors",
                             qa_type: str = "cot",
                             num_examples: int = None) -> str:
        """
        Get few-shot examples for a specific model and dataset configuration
        
        Args:
            model_name: Name of the model
            dataset: Dataset name from Adaptive-RAG paper (e.g., "squad", "nq", "trivia", "hotpotqa", "2wikimultihopqa", "musique")
            context_type: Context type
            qa_type: QA type
            num_examples: Number of examples to extract
            
        Returns:
            String containing formatted few-shot examples
        """
        # Normalize model name
        normalized_model = self._normalize_model_name(model_name)
        
        # Get appropriate dataset based on Adaptive-RAG paper classification
        target_dataset = self._get_safe_dataset_for_examples(dataset)
        
        # Use the provided context_type directly (unified approach)
        
        # Construct prompt file path - use enhanced version if available
        prompt_filename = f"{context_type}_context_{qa_type}_qa_{normalized_model}_enhanced.txt"
        prompt_path = self.prompts_base_path / target_dataset / prompt_filename
        
        # Fallback to original file if enhanced version doesn't exist
        if not prompt_path.exists():
            self.logger.info(f"Enhanced prompt not found at {prompt_path}, falling back to original")
            prompt_filename = f"{context_type}_context_{qa_type}_qa_{normalized_model}.txt"
            prompt_path = self.prompts_base_path / target_dataset / prompt_filename
        
        # Direct prompt file loading - no fallbacks needed
        
        if not prompt_path.exists():
            raise FileNotFoundError(f"Prompt file not found: {prompt_path}. Expected IRCOT paper structure.")
        
        # IRCOT paper: "pack as many demonstrations as possible within the model's context length limit"
        max_examples = self._get_max_examples_for_model(normalized_model) if num_examples is None else num_examples
        return self._extract_examples_from_file(prompt_path, max_examples)
    
    def get_ircot_examples(self, 
                          model_name: str, 
                          dataset: str = None,
                          num_examples: int = None) -> str:
        """
        Get IRCOT-specific examples for final answer extraction following IRCOT paper
        
        Args:
            model_name: Name of the model
            dataset: Dataset name from Adaptive-RAG paper
            num_examples: Number of examples to extract (None = use model default)
            
        Returns:
            String containing formatted IRCOT examples for final answer extraction
        """
        # Get regular examples which already have proper IRCOT format
        return self.get_few_shot_examples(
            model_name=model_name,
            dataset=dataset,
            context_type="gold_with_1_distractors",
            qa_type="cot",
            num_examples=num_examples
        )
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to match prompt file naming convention"""
        model_lower = model_name.lower()
        
        if "gemini" in model_lower:
            return "gemini"
        elif "qwen" in model_lower:
            return "qwen"
        elif "flan" in model_lower or "t5" in model_lower:
            return "flan_t5"
        elif "codex" in model_lower or "gpt" in model_lower:
            return "codex"
        else:
            # Default to gemini for unknown models
            return "gemini"
    
    def _extract_examples_from_file(self, file_path: Path, num_examples: int) -> str:
        """Extract specified number of examples from a prompt file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by metadata headers to find individual examples
            examples = []
            sections = re.split(r'# METADATA:', content)
            
            for section in sections[1:]:  # Skip first empty section
                if not section.strip():
                    continue
                    
                # Extract the Q: A: part
                qa_match = re.search(r'Q: (.+?)\nA: (.+?)(?=\n\n|\n# |$)', section, re.DOTALL)
                if qa_match:
                    question = qa_match.group(1).strip()
                    answer = qa_match.group(2).strip()
                    
                    # Format as a clean example
                    examples.append(f"Q: {question}\nA: {answer}")
                    
                    if len(examples) >= num_examples:
                        break
            
            if not examples:
                self.logger.warning(f"No examples found in {file_path}")
                raise FileNotFoundError(f"No examples found in prompt file: {file_path}")
            
            # Join examples with appropriate delimiter
            delimiter = "\n\n\n" if "gemini" in str(file_path) else "\n\n"
            return delimiter.join(examples)
            
        except Exception as e:
            self.logger.error(f"Error reading prompt file {file_path}: {e}")
            raise FileNotFoundError(f"Failed to read prompt file {file_path}: {e}")
    
    def _get_safe_dataset_for_examples(self, dataset: str) -> str:
        """
        Get a safe dataset for few-shot examples based on Adaptive-RAG paper classification
        
        Adaptive-RAG paper uses these datasets:
        - Single-hop: SQUAD v1.1, Natural Questions (nq), TriviaQA (trivia)  
        - Multi-hop: MuSiQue (musique), HotpotQA (hotpotqa), 2WikiMultiHopQA (2wikimultihopqa)
        """
        if not dataset:
            return "squad"  # Default safe choice
            
        # Normalize dataset name
        dataset_lower = dataset.lower()
        
        # Map variations to standard names
        dataset_mapping = {
            "squad": "squad",
            "squad_v1": "squad", 
            "squad_v1.1": "squad",
            "natural_questions": "nq",
            "nq": "nq",
            "trivia": "trivia",
            "triviaqa": "trivia",
            "musique": "musique",
            "hotpot": "hotpotqa",
            "hotpotqa": "hotpotqa",
            "2wiki": "2wikimultihopqa",
            "2wikimultihopqa": "2wikimultihopqa"
        }
        
        return dataset_mapping.get(dataset_lower, "squad")
    
    def _get_max_examples_for_model(self, model_name: str) -> int:
        """
        Get maximum examples based on actual model context limits
        """
        if "gemini" in model_name.lower():
            return 15  # Large 128K context - can handle many examples
        elif "flan" in model_name.lower():
            return 6   # Current deployment limit (per IRCOT paper)
        else:
            return 8   # Default for unknown models
    