"""
Base Labeler Interface for Scaled Silver Labeling

This module defines the abstract base class for all labeling strategies
in the scaled silver labeling system.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Tuple
import logging
import re
import string
import time
import json

import sys
import os
from pathlib import Path
from datetime import datetime

# Add parent directory to path for synthetic checker import
current_dir = Path(__file__).parent
utils_dir = current_dir.parent / "utils"
sys.path.append(str(utils_dir))


from synthetic_checker import SyntheticChecker, SyntheticCheckResult
SYNTHETIC_CHECKER_AVAILABLE = True

import hashlib

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.common import (
    get_timestamp, validate_dataset_name, validate_model_name, 
    validate_strategy_name, get_dataset_type, normalize_answer
)
from ssl_logging.base_logger import BaseLogger




class BaseLabeler:
    """
    Base class for silver labeling strategies.
    
    This class provides the common interface and functionality for all
    silver labeling implementations in the Adaptive-RAG system.
    
    Current evaluation flow for each system (NOR, ONER, IRCOT):
    1. Try exact match against ground truths
    2. If exact match fails, immediately run synthetic checker (LLM judge)
    """
    
    def __init__(self, server_manager, logger):
        """
        Initialize the base labeler
        
        Args:
            server_manager: LLMServerManager instance for running systems
            logger: Logger instance for this labeler
        """
        self.server_manager = server_manager
        self.logger = logger
        
        # Performance tracking
        self.performance_stats = {
            'total_samples': 0,
            'successful_labels': 0,
            'label_distribution': {'A': 0, 'B': 0, 'C': 0, 'DISCARDED': 0},
            'system_success_rates': {},
            'total_annotation_time': 0.0
        }
        
        # Per-session tracking
        self.session_annotation_time = 0.0
        
        # Parallel processing configuration
        self.parallel_config = {
            'max_workers': 5,
            'auto_calculate': True
        }
        
        # Model type classification
        self.generative_models = ['gemini-2.5-flash-lite', 'gemini-1.5-flash-8b']
        self.flan_t5_models = ['flan-t5']
        
        # Regex pattern for structured answer extraction  
        # Improved to handle various formats and edge cases
        self.answer_regex = re.compile(r".*(?:so\s+)?(?:the\s+)?answer\s+is:?\s*(.*?)(?:\s*\.?\s*$)", re.IGNORECASE | re.DOTALL)
        
        # Track discarded samples for analysis
        self.discarded_samples = []
        
        # Initialize synthetic checker
        self.synthetic_checker = None
        if SYNTHETIC_CHECKER_AVAILABLE:
            try:
                # Initialize synthetic checker with API key from environment
                # This uses Google's Gemini API directly, not a local server
                self.synthetic_checker = SyntheticChecker()
                self.logger.info("Synthetic checker initialized")
            except Exception as e:
                self.logger.warning(f"Failed to initialize synthetic checker: {e}")
                self.synthetic_checker = None
        else:
            self.logger.warning("Synthetic checker not available")
        
        # Few-shot examples for different model types
        self.few_shot_examples = {
            'gemini': {
                'with_context': [
                    {
                        'question': 'Who won the 2024 Formula 1 World Championship?',
                        'context': 'Max Verstappen won the 2024 Formula 1 World Drivers\' Championship, securing his fourth consecutive title.',
                        'answer': 'Max Verstappen secured his fourth consecutive title by winning the 2024 Formula 1 World Drivers\' Championship. Answer is: Max Verstappen'
                    },
                    {
                        'question': 'What year was Shakespeare born?',
                        'context': 'William Shakespeare was an English playwright and poet widely regarded as the greatest writer in the English language. He was born in Stratford-upon-Avon in 1564 and died in 1616. During his lifetime, he wrote approximately 39 plays and 154 sonnets, including famous works like Romeo and Juliet, Hamlet, and Macbeth. He is often called England\'s national poet and the "Bard of Avon".',
                        'answer': 'William Shakespeare was born in Stratford-upon-Avon in 1564. Answer is: 1564'
                    }
                ],
                'without_context': [
                    {
                        'question': 'Who won the 2024 Formula 1 World Championship?',
                        'answer': 'Max Verstappen secured his fourth consecutive title by winning the 2024 Formula 1 World Drivers\' Championship. Answer is: Max Verstappen'
                    },
                    {
                        'question': 'What year was Shakespeare born?',
                        'answer': 'William Shakespeare was born in Stratford-upon-Avon in 1564. Answer is: 1564'
                    }
                ]
            },
            'qwen': {
                'with_context': [
                    {
                        'question': 'Who won the 2024 Formula 1 World Championship?',
                        'context': 'Max Verstappen won the 2024 Formula 1 World Drivers\' Championship, securing his fourth consecutive title.',
                        'answer': 'Max Verstappen secured his fourth consecutive title by winning the 2024 Formula 1 World Drivers\' Championship. Answer is: Max Verstappen'
                    },
                    {
                        'question': 'What year was Shakespeare born?',
                        'context': 'William Shakespeare was an English playwright and poet widely regarded as the greatest writer in the English language. He was born in Stratford-upon-Avon in 1564 and died in 1616. During his lifetime, he wrote approximately 39 plays and 154 sonnets, including famous works like Romeo and Juliet, Hamlet, and Macbeth. He is often called England\'s national poet and the "Bard of Avon".',
                        'answer': 'William Shakespeare was born in Stratford-upon-Avon in 1564. Answer is: 1564'
                    }
                ],
                'without_context': [
                    {
                        'question': 'Who won the 2024 Formula 1 World Championship?',
                        'answer': 'Max Verstappen secured his fourth consecutive title by winning the 2024 Formula 1 World Drivers\' Championship. Answer is: Max Verstappen'
                    },
                    {
                        'question': 'What year was Shakespeare born?',
                        'answer': 'William Shakespeare was born in Stratford-upon-Avon in 1564. Answer is: 1564'
                    }
                ]
            }
        }
    
    def _is_generative_model(self, model_name: str) -> bool:
        """
        Check if the model is a generative model (GPT, Gemini, Qwen)
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if it's a generative model, False otherwise
        """
        model_name_lower = model_name.lower()
        return any(gen_model.lower() in model_name_lower for gen_model in self.generative_models) or \
               'gpt' in model_name_lower or 'gemini' in model_name_lower or 'qwen' in model_name_lower
    
    def _is_flan_t5_model(self, model_name: str) -> bool:
        """
        Check if the model is a FLAN-T5 model
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if it's a FLAN-T5 model, False otherwise
        """
        model_name_lower = model_name.lower()
        return any(flan_model.lower() in model_name_lower for flan_model in self.flan_t5_models)
    
    def _is_qwen_model(self, model_name: str) -> bool:
        """
        Check if the model is a Qwen model
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if it's a Qwen model, False otherwise
        """
        return 'qwen' in model_name.lower()
    
    def _is_gemini_model(self, model_name: str) -> bool:
        """
        Check if the model is a Gemini model
        
        Args:
            model_name: Name of the model
            
        Returns:
            True if it's a Gemini model, False otherwise
        """
        return 'gemini' in model_name.lower()
    
    def _extract_structured_answer(self, response: str) -> str:
        """
        Extract answer from structured response using regex
        
        Args:
            response: Raw response from model
            
        Returns:
            Extracted answer or original response if no match
        """
        response = response.strip()
        
        # Try to match "Answer is: ..." pattern
        match = self.answer_regex.search(response)
        if match:
            extracted = match.group(1).strip()
            # Remove trailing punctuation
            extracted = extracted.rstrip('.')
            # Remove leading comma if present (common formatting issue)
            extracted = extracted.lstrip(', ')
            return extracted
        
        # Additional fallback for malformed responses (e.g., starting with comma)
        if response.startswith(', '):
            # Remove leading comma and try to extract meaningful content
            cleaned = response[2:].strip()
            # If it contains answer-like content, extract it
            if 'answer is' in cleaned.lower():
                # Try recursive extraction on cleaned version
                return self._extract_answer_from_response(cleaned)
            else:
                # Take the first meaningful sentence
                sentences = cleaned.split('.')
                if sentences:
                    return sentences[0].strip()
        
        # Final fallback: return original response
        return response
    
    def _evaluate_answer(self, predicted: str, ground_truths: List[str], model_name: str = None) -> Tuple[bool, str]:
        """
        Evaluate if predicted answer matches any ground truth using exact match on normalized text.
        
        This is the FIRST step in evaluation. If this returns False, the synthetic checker
        is immediately called to perform semantic evaluation.
        
        Args:
            predicted: Predicted answer from system
            ground_truths: List of acceptable ground truth answers
            model_name: Name of the model (for determining extraction method)
            
        Returns:
            Tuple of (is_correct, matched_ground_truth) where:
            - is_correct: True if answer matches any ground truth, False if synthetic check needed
            - matched_ground_truth: The ground truth that was matched (for logging)
        """
        if not predicted or not ground_truths:
            return False, ""
        
        # For generative models, extract structured answer first
        if model_name and self._is_generative_model(model_name):
            predicted = self._extract_structured_answer(predicted)
        
        # Normalize predicted answer
        pred_normalized = normalize_answer(predicted)
        if not pred_normalized:
            return False, ""
        
        # Check against all ground truths (max-over-ground-truths pattern)
        for ground_truth in ground_truths:
            if not ground_truth:
                continue
                
            gt_normalized = normalize_answer(ground_truth)
            if not gt_normalized:
                continue
                
            # First try exact match for all models with structured extraction
            if pred_normalized == gt_normalized:
                return True, ground_truth
        
        return False, ""

    def _extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """
        Extract all ground truth answers from sample, handling different dataset formats
        
        Args:
            sample: Sample data from dataset
            
        Returns:
            List of ground truth answer strings (all acceptable answers)
        """
        # Try the direct answer field first (for backwards compatibility)
        if 'answer' in sample and sample['answer']:
            # For backward compatibility, wrap single answer in list
            return [str(sample['answer'])]
        
        # Try the standardized answers_objects structure
        if 'answers_objects' in sample and sample['answers_objects']:
            answers_obj = sample['answers_objects'][0]
            if 'spans' in answers_obj and answers_obj['spans']:
                # Return ALL spans, not just the first one
                return [str(span) for span in answers_obj['spans']]
        
        # Fallback: empty list
        return []
    
    def set_parallel_config(self, max_workers: int = None, auto_calculate: bool = True):
        """
        Configure parallel processing settings
        
        Args:
            max_workers: Maximum number of worker threads for parallel processing
            auto_calculate: Whether to automatically calculate workers based on sample size
        """
        if max_workers is not None:
            self.parallel_config['max_workers'] = max_workers
            self.parallel_config['auto_calculate'] = False
        
        self.parallel_config['auto_calculate'] = auto_calculate
        
        self.logger.info(f"Parallel configuration updated: max_workers={self.parallel_config['max_workers']}, auto_calculate={auto_calculate}")
    
    def _get_few_shot_examples(self, model_name: str, with_context: bool = True) -> str:
        """
        Generate few-shot examples for the given model
        
        Args:
            model_name: Name of the model
            with_context: Whether to include context in examples
            
        Returns:
            Formatted few-shot examples string
        """
        if self._is_gemini_model(model_name):
            model_examples = self.few_shot_examples['gemini']
        elif self._is_qwen_model(model_name):
            model_examples = self.few_shot_examples['qwen']
        else:
            return ""  # No few-shot examples for FLAN-T5
        
        # Get the appropriate examples based on context setting
        context_key = 'with_context' if with_context else 'without_context'
        examples = model_examples.get(context_key, [])
        
        formatted_examples = []
        for example in examples:
            if with_context:
                if self._is_qwen_model(model_name):
                    formatted_examples.append(
                        f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer is: {example['answer']}"
                    )
                else:  # Gemini
                    formatted_examples.append(
                        f"Context: {example['context']}\n\nQuestion: {example['question']}\n\nAnswer is: {example['answer']}"
                    )
            else:
                if self._is_qwen_model(model_name):
                    formatted_examples.append(
                        f"Question: {example['question']}\n\nAnswer is: {example['answer']}"
                    )
                else:  # Gemini
                    formatted_examples.append(
                        f"Question: {example['question']}\n\nAnswer is: {example['answer']}"
                    )
        
        return "\n\n\n" + "\n\n\n".join(formatted_examples) + "\n\n\n"

    @abstractmethod
    def label_samples(self, dataset_name: str, sample_size: int, model_name: str, oner_max_docs: int = None, ircot_max_docs: int = None, **kwargs) -> Dict[str, Any]:
        """
        Main method to run the silver labeling process.
        
        Args:
            dataset_name: Name of the dataset to process
            sample_size: Number of samples to label
            model_name: Name of the model to use
            
        Returns:
            Dictionary with labeling results and statistics
        """
        pass
    
    @abstractmethod
    def get_required_systems(self, dataset_name: str) -> List[str]:
        """
        Get the list of systems required for this labeling strategy
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of system names required
        """
        pass
    
    def validate_inputs(self, dataset_name: str, sample_size: int, model_name: str) -> None:
        """
        Validate input parameters
        
        Args:
            dataset_name: Name of the dataset
            sample_size: Number of samples to process
            model_name: Name of the model
            
        Raises:
            ValueError: If any input is invalid
        """
        if not dataset_name or not isinstance(dataset_name, str):
            raise ValueError("dataset_name must be a non-empty string")
        
        if not isinstance(sample_size, int) or sample_size <= 0:
            raise ValueError("sample_size must be a positive integer")
        
        if not model_name or not isinstance(model_name, str):
            raise ValueError("model_name must be a non-empty string")
    
    def update_performance_stats(self, 
                               label: str,
                               systems_used: List[str],
                               systems_succeeded: List[str],
                               success: bool) -> None:
        """
        Update performance statistics
        
        Args:
            label: The assigned label ('A', 'B', 'C')
            systems_used: List of systems that were called
            systems_succeeded: List of systems that succeeded
            success: Whether the labeling was successful
        """
        self.performance_stats['total_samples'] += 1
        
        if success:
            self.performance_stats['successful_labels'] += 1
            if label in self.performance_stats['label_distribution']:
                self.performance_stats['label_distribution'][label] += 1
        else:
            self.performance_stats['failed_labels'] += 1
        
        # Update system call statistics
        for system in systems_used:
            if system in self.performance_stats['system_calls']:
                self.performance_stats['system_calls'][system] += 1
        
        # Update system success statistics
        for system in systems_succeeded:
            if system in self.performance_stats['system_successes']:
                self.performance_stats['system_successes'][system] += 1
    
    def create_result_entry(self, 
                          sample: Dict[str, Any],
                          label: str,
                          reasoning: str,
                          systems_used: List[str],
                          systems_succeeded: List[str],
                          system_answers: Dict[str, str] = None,
                          annotation_time: float = 0.0,
                          steps: int = 0,
                          synthetic_check_result: Optional[SyntheticCheckResult] = None,
                          primary_system: str = None,
                          match_type: str = None,
                          system_steps: Dict[str, int] = None,
                          system_results: Dict[str, Any] = None,
                          processing_time: float = 0.0) -> Dict[str, Any]:
        """
        Create a standardized result entry
        
        Args:
            sample: Original sample data
            label: Assigned complexity label ('A', 'B', 'C')
            reasoning: Explanation for the label assignment
            systems_used: List of systems that were executed
            systems_succeeded: List of systems that answered correctly
            system_answers: Dictionary mapping system names to their answers
            annotation_time: Time taken to annotate this sample
            steps: Total number of retrieval-and-generate steps (as defined in paper)
            synthetic_check_result: Result from synthetic answer checking
            primary_system: The primary system used for this sample
            match_type: Type of match found (exact, semantic, etc.)
            system_steps: Dictionary mapping system names to their step counts
            system_results: Complete system results for detailed analysis
            processing_time: Time taken for processing this sample
            
        Returns:
            Dictionary containing the result entry
        """
        # Determine primary answer and raw_answer from system results
        primary_answer = ""
        primary_raw_answer = ""
        
        if system_answers:
            # Use the explicitly provided primary_system if available
            if primary_system and primary_system in system_answers:
                selected_system = primary_system
            else:
                # Fallback: map label to corresponding system
                label_to_system = {
                    'A': 'nor_qa',      # Label A = NOR system 
                    'B': 'oner_qa',     # Label B = ONER system
                    'C': 'ircot_qa'     # Label C = IRCOT system
                }
                
                selected_system = label_to_system.get(label)
                
                # If label-based selection fails, use first successful system
                if not selected_system or selected_system not in system_answers:
                    if systems_succeeded:
                        selected_system = systems_succeeded[0]
                    elif systems_used:
                        selected_system = systems_used[0]
                
            if selected_system and selected_system in system_answers:
                system_result = system_answers[selected_system]
                if isinstance(system_result, dict):
                    primary_answer = system_result.get('answer', '')
                    primary_raw_answer = system_result.get('raw_answer', '')
                else:
                    # Backwards compatibility: system_result might be a string
                    primary_answer = str(system_result)
                    primary_raw_answer = str(system_result)
        
        # Get ground truths and find which one matches the answer (if any)
        ground_truths = self._extract_ground_truth(sample)
        matched_gt = ""
        if primary_answer and ground_truths:
            _, matched_gt = self._evaluate_answer(primary_answer, ground_truths, "default")
        
        result = {
            'sample_id': self._get_sample_id(sample),
            'question': sample.get('question_text', ''),
            'ground_truth': matched_gt if matched_gt else ground_truths[0] if ground_truths else "",
            'answer': primary_answer,
            'raw_answer': primary_raw_answer,
            'label': label,
            'reasoning': reasoning,
            'systems_used': systems_used,
            'systems_succeeded': systems_succeeded,
            'steps': steps,
            'timestamp': datetime.now().isoformat(),
            'overridden': True if label == 'D' else False
        }
        
        # Add system answers if provided
        if system_answers:
            result['system_answers'] = system_answers
        
        # Add synthetic check information if available
        if synthetic_check_result:
            result['synthetic_check'] = {
                'is_semantically_correct': synthetic_check_result.is_semantically_correct,
                'processing_time': synthetic_check_result.processing_time,
                'source_pipeline': synthetic_check_result.source_pipeline,
                'error': synthetic_check_result.error
            }
            
        return result
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """
        Get a summary of performance statistics
        
        Returns:
            Dictionary with performance metrics
        """
        total_samples = self.performance_stats['total_samples']
        
        # Calculate success rate
        success_rate = (self.performance_stats['successful_labels'] / 
                       max(1, total_samples))
        
        # Calculate label distribution percentages
        label_percentages = {}
        for label, count in self.performance_stats['label_distribution'].items():
            label_percentages[label] = (count / max(1, total_samples)) * 100
        
        # Calculate system efficiency
        system_efficiency = {}
        for system in self.performance_stats['system_calls']:
            calls = self.performance_stats['system_calls'][system]
            successes = self.performance_stats['system_successes'][system]
            system_efficiency[system] = (successes / max(1, calls)) * 100
        
        return {
            'total_samples': total_samples,
            'success_rate': success_rate,
            'label_distribution': self.performance_stats['label_distribution'],
            'label_percentages': label_percentages,
            'system_calls': self.performance_stats['system_calls'],
            'system_successes': self.performance_stats['system_successes'],
            'system_efficiency': system_efficiency
        }
    
    def log_labeling_start(self, dataset_name: str, sample_size: int, model_name: str) -> None:
        """
        Log the start of a labeling task
        
        Args:
            dataset_name: Name of the dataset to process
            sample_size: Number of samples to label
            model_name: Name of the model to use
        """
        self.logger.info(f"Starting labeling task: {dataset_name} with {sample_size} samples using {model_name}")
        self.logger.info(f"Strategy: {self.__class__.__name__}")
        
        # Initialize progress tracking in logger if supported
        if hasattr(self.logger, 'set_total_samples'):
            self.logger.set_total_samples(sample_size)
        
        # Reset performance stats for this task
        self.performance_stats = {
            'total_samples': 0,
            'successful_labels': 0,
            'failed_labels': 0,
            'label_distribution': {'A': 0, 'B': 0, 'C': 0},
            'system_calls': {'nor_qa': 0, 'oner_qa': 0, 'ircot_qa': 0},
            'system_successes': {'nor_qa': 0, 'oner_qa': 0, 'ircot_qa': 0}
        }
    
    def update_stats(self, **kwargs) -> None:
        """
        Update performance statistics
        
        Args:
            **kwargs: Statistics to update (total_samples_processed, successful_labels, 
                     failed_labels, discarded_samples, processing_time, etc.)
        """
        # Ensure all required keys exist in performance_stats (defensive programming)
        if 'discarded_samples' not in self.performance_stats:
            self.performance_stats['discarded_samples'] = 0
            
        if 'total_samples' not in self.performance_stats:
            self.performance_stats['total_samples'] = 0
            
        if 'successful_labels' not in self.performance_stats:
            self.performance_stats['successful_labels'] = 0
            
        if 'failed_labels' not in self.performance_stats:
            self.performance_stats['failed_labels'] = 0
            
        if 'total_samples_processed' in kwargs:
            self.performance_stats['total_samples'] += kwargs['total_samples_processed']
        
        if 'successful_labels' in kwargs:
            self.performance_stats['successful_labels'] += kwargs['successful_labels']
        
        if 'failed_labels' in kwargs:
            self.performance_stats['failed_labels'] += kwargs['failed_labels']
        
        if 'discarded_samples' in kwargs:
            self.performance_stats['discarded_samples'] += kwargs['discarded_samples']
        
        if 'processing_time' in kwargs:
            self.performance_stats['processing_time'] = kwargs['processing_time']
        
        # Log progress periodically
        total = self.performance_stats.get('total_samples', 0)
        if total > 0 and total % 50 == 0:
            success_rate = self.performance_stats.get('successful_labels', 0) / total * 100
            discarded_rate = self.performance_stats.get('discarded_samples', 0) / total * 100
            self.logger.info(f"Progress: {total} samples processed, {success_rate:.1f}% success rate, {discarded_rate:.1f}% discarded due to resource exhaustion")
    
    def aggregate_results(self, individual_results: List[Dict[str, Any]], 
                         dataset_name: str, model_name: str) -> Dict[str, Any]:
        """
        Aggregate individual sample results into a final result dictionary
        
        Args:
            individual_results: List of results from processing individual samples
            dataset_name: Name of the dataset
            model_name: Name of the model used
            
        Returns:
            Dictionary containing aggregated results and metadata
        """
        performance_summary = self.get_performance_summary()
        
        # Count label distribution (now including "DISCARDED" for discarded samples)
        label_counts = {'A': 0, 'B': 0, 'C': 0, 'DISCARDED': 0, 'ERROR': 0, 'unknown': 0}
        for result in individual_results:
            label = result.get('label', 'unknown')
            if label in label_counts:
                label_counts[label] += 1
            else:
                label_counts['unknown'] += 1
        
        # Calculate totals from individual results
        total_annotation_time = sum(result.get('annotation_time', 0.0) for result in individual_results)
        total_steps = sum(result.get('steps', 0) for result in individual_results)
        total_samples = len(individual_results)
        avg_steps_per_sample = total_steps / total_samples if total_samples > 0 else 0.0

        return {
            'dataset_name': dataset_name,
            'model_name': model_name,
            'total_samples': total_samples,
            'individual_results': individual_results,
            'label_distribution': label_counts,
            'total_annotation_time': total_annotation_time,
            'total_steps': total_steps,
            'avg_steps_per_sample': avg_steps_per_sample,
            'timestamp': get_timestamp()
        }
    
    def log_labeling_end(self, dataset_name: str, results: Dict[str, Any]) -> None:
        """
        Log the end of a labeling task
        
        Args:
            dataset_name: Name of the dataset that was processed
            results: Final results dictionary
        """
        total_samples = results.get('total_samples', 0)
        label_distribution = results.get('label_distribution', {})
        discarded_count = label_distribution.get('D', 0)
        labeled_count = total_samples - discarded_count
        
        self.logger.info(f"Labeling task completed: {dataset_name}")
        self.logger.info(f"Total samples processed: {total_samples}")
        self.logger.info(f"Samples labeled: {labeled_count}")
        self.logger.info(f"Samples discarded: {discarded_count}")
        
        # Log timing and retriever usage
        total_annotation_time = results.get('total_annotation_time', 0.0)
        total_steps = results.get('total_steps', 0)
        avg_steps_per_sample = results.get('avg_steps_per_sample', 0.0)
        self.logger.info(f"Total annotation time: {total_annotation_time:.2f} seconds")
        self.logger.info(f"Total steps: {total_steps}")
        self.logger.info(f"Average steps per sample: {avg_steps_per_sample:.2f}")
        if total_samples > 0:
            avg_time_per_sample = total_annotation_time / total_samples
            self.logger.info(f"Average annotation time per sample: {avg_time_per_sample:.2f} seconds")
        
        if 'label_distribution' in results:
            self.logger.info(f"Label distribution: {results['label_distribution']}")
        
        if 'performance_stats' in results:
            performance = results['performance_stats']
            self.logger.info(f"Success rate: {performance.get('success_rate', 0):.2%}")
        
        # Reset discarded samples for next task (keeping for backward compatibility)
        self.discarded_samples = []
    
    def log_sample_result(self, sample: Dict[str, Any], result: Dict[str, Any]) -> None:
        """
        Log the result of processing a sample
        
        Args:
            sample: Original sample data
            result: Processing result
        """
        sample_id = self._get_sample_id(sample)
        self.logger.info(f"Processed sample {sample_id}: label={result.get('label')}, "
                        f"reasoning={result.get('reasoning')}")
    
    def _get_sample_id(self, sample: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a sample - preserves original question IDs
        
        Args:
            sample: Sample data
            
        Returns:
            Original question ID if available, otherwise a generated hash
        """
        # First priority: Return original question ID directly if available
        for field in ['question_id', 'id', '_id', 'qid']:
            if field in sample and sample[field]:
                original_id = str(sample[field]).strip()
                if original_id:  # Make sure it's not empty
                    return original_id
        
        # Fallback: Create hash only if no original ID exists
        # This preserves backward compatibility for samples without IDs
        id_fields = []
        
        # Use question text as fallback (try both field names)
        if 'question_text' in sample:
            id_fields.append(sample['question_text'][:100])  # First 100 chars
        elif 'question' in sample:
            id_fields.append(sample['question'][:100])  # First 100 chars
        
        # Create hash from combined fields
        if id_fields:
            combined = '|'.join(id_fields)
            return hashlib.md5(combined.encode()).hexdigest()
        
        # Final fallback - use entire sample (risky but better than nothing)
        return hashlib.md5(json.dumps(sample, sort_keys=True).encode()).hexdigest()
    
    def _create_error_result(self, sample: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        Create a result entry for failed processing
        
        Args:
            sample: Original sample data
            error_message: Description of the error
            
        Returns:
            Error result dictionary
        """
        # For error cases, use the first ground truth as fallback
        ground_truths = self._extract_ground_truth(sample)
        
        return {
            'sample_id': self._get_sample_id(sample),
            'question': sample.get('question_text', ''),
            'ground_truth': ground_truths[0] if ground_truths else "",
            'answer': '',
            'raw_answer': '',
            'label': 'ERROR',
            'reasoning': f"Processing failed: {error_message}",
            'systems_used': [],
            'systems_succeeded': [],
            'error': True,
            'timestamp': datetime.now().isoformat(),
            'overridden': False
        } 

    def create_discarded_sample_entry(self, 
                                     sample: Dict[str, Any],
                                     system_name: str,
                                     predicted_answer: str,
                                     ground_truth: str,
                                     reason: str) -> Dict[str, Any]:
        """
        Create a standardized discarded sample entry
        
        Args:
            sample: Original sample data
            system_name: Name of the system that produced the answer
            predicted_answer: Answer from the system
            ground_truth: Ground truth answer
            reason: Reason for discarding
            
        Returns:
            Standardized discarded sample dictionary
        """
        return {
            'sample_id': self._get_sample_id(sample),
            'question': sample.get('question_text', ''),
            'predicted_answer': predicted_answer,
            'ground_truth': ground_truth,
            'system_name': system_name,
            'reason': reason,
            'timestamp': datetime.now().isoformat(),
        }
    
    def log_discarded_samples(self):
        """
        Log information about discarded samples for analysis
        """
        if not self.discarded_samples:
            return
        
        self.logger.info(f"=== DISCARDED SAMPLES ANALYSIS ===")
        self.logger.info(f"Total discarded samples: {len(self.discarded_samples)}")
        
        # Group by reason
        reason_counts = {}
        for sample in self.discarded_samples:
            reason = sample.get('reason', 'unknown')
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        
        self.logger.info(f"Discard reasons: {reason_counts}")
        
        # Log sample details
        for sample in self.discarded_samples:
            self.logger.info(
                f"DISCARDED [{sample['reason']}] | "
                f"ID: {sample['sample_id']} | "
                f"System: {sample['system_name']} | "
                f"Q: {sample['question'][:100]}{'...' if len(sample['question']) > 100 else ''} | "
                f"Predicted: {sample['predicted_answer']} | "
                f"GT: {sample['ground_truth']}"
            ) 