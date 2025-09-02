"""
Optimized labeling Strategy

This module implements the optimized labeling strategy:
- Single-hop datasets: Run NOR only. Try exact match first, then synthetic check if needed. If works -> 'A', else -> 'B'
- Multi-hop datasets: Run NOR and ONER. For each system, try exact match first, then synthetic check if needed. If NOR works -> 'A', if ONER works -> 'B', else -> 'C'

Open Source Models are in beta so needs more testing.
"""

import os
import sys
import logging
import time
import hashlib
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scaled_silver_labeling.labeling.base_labeler import BaseLabeler
from scaled_silver_labeling.data.dataset_processor import DatasetProcessor
from scaled_silver_labeling.utils.common import (
    get_timestamp, validate_dataset_name, get_dataset_type,
    SINGLE_HOP_DATASETS, MULTI_HOP_DATASETS
)
from transformers import AutoTokenizer

# Add Qwen-specific constants
QWEN_MODELS = ["Qwen/Qwen2.5-7B-Instruct"]
FLAN_T5_MODELS = ["flan-t5-large", "flan-t5-xl", "flan-t5-xxl"]

class OptimizedLabeler(BaseLabeler):
    """
    Optimized labeling strategy implementation
    
    This strategy uses different approaches based on dataset characteristics:
    - Single-hop datasets (NQ, TriviaQA, SQuAD): Only run NOR system with exact match first, then synthetic check if needed
    - Multi-hop datasets (HotpotQA, MuSiQue, 2WikiMultiHopQA): Run NOR and ONER systems, each with exact match first, then synthetic check if needed
    
    This is more efficient than the original strategy as it avoids running
    unnecessary systems and synthetic checks when exact matches succeed.
    """
    
    def __init__(self, server_manager, logger):
        """
        Initialize the optimized labeler
        
        Args:
            server_manager: LLM server manager for handling requests
            logger: Logger instance for recording operations
        """
        super().__init__(server_manager, logger)
        self.single_hop_datasets = SINGLE_HOP_DATASETS
        self.multi_hop_datasets = MULTI_HOP_DATASETS
        self.dataset_processor = DatasetProcessor()
    
    def _is_qwen_model(self, model_name: str) -> bool:
        """Check if the model is a Qwen model requiring chat template"""
        return any(qwen_model.lower() in model_name.lower() for qwen_model in QWEN_MODELS)
    
    def _is_flan_t5_model(self, model_name: str) -> bool:
        """Check if the model is a FLAN-T5 model"""
        return any(flan_model in model_name for flan_model in FLAN_T5_MODELS)
    
    def _get_qwen_tokenizer(self):
        """Get the tokenizer for Qwen models."""
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    def _create_qwen_prompt(self, question: str, context: str = None) -> str:
        """
        Create a Qwen-specific prompt with chat template and few-shot examples
        
        Args:
            question: The question to ask
            context: Optional context from retrieved documents
            
        Returns:
            Formatted prompt string for Qwen models
        """
        if context:
            # With context (for ONER system)
            system_message = "Answer the question using both the provided context documents AND your parametric knowledge. If the context doesn't contain sufficient information, rely on your parametric knowledge to provide the best answer. Always provide a substantive answer rather than saying you don't have enough information. Provide a brief explanation, then end your response with 'Answer is: ' followed by your concise answer. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples("qwen", with_context=True)
            user_message = f"{few_shot_examples}Using the following information: {context}\n\nQuestion: {question}\n\nAnswer is:"
        else:
            # Without context (for NOR system)
            system_message = "Answer the question using your parametric knowledge from training. Provide a brief explanation, then end your response with 'Answer is: ' followed by your concise answer. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples("qwen", with_context=False)
            user_message = f"{few_shot_examples}Question: {question}\n\nAnswer is:"
        
        # Format as chat template
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        tokenizer = self._get_qwen_tokenizer()
        max_length = 32000 
        tokens = tokenizer.encode(prompt, truncation=True, max_length=max_length)
        return tokenizer.decode(tokens)
    
    def _create_gemini_prompt(self, question: str, context: str = None) -> str:
        """
        Create a Gemini-specific prompt with structured answer format and few-shot examples
        
        Args:
            question: The question to answer
            context: Optional context from retrieval
            
        Returns:
            Formatted prompt string for Gemini models
        """
        if context:
            instruction = f"Answer the question using both the provided context documents AND your parametric knowledge. If the context doesn't contain sufficient information, rely on your parametric knowledge to provide the best answer. Always provide a substantive answer rather than saying you don't have enough information. Provide a brief explanation, then end your response with 'Answer is: [your concise answer]'. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples("gemini", with_context=True)
            prompt = f"{instruction}{few_shot_examples}Context: {context}\n\nQuestion: {question}\n\nAnswer is:"
        else:
            instruction = f"Answer the question using your parametric knowledge from training. Provide a brief explanation, then end your response with 'Answer is: [your concise answer]'. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples("gemini", with_context=False)
            prompt = f"{instruction}{few_shot_examples}Question: {question}\n\nAnswer is:"
        return prompt
    
    def _create_flan_t5_prompt(self, question: str, context: str = None) -> str:
        """
        Create a FLAN-T5 specific prompt
        
        Args:
            question: The question to answer
            context: Optional context from retrieval
            
        Returns:
            Formatted prompt string for FLAN-T5 models
        """
        if context:
            instruction = "Use both the provided context and your knowledge to answer. If context is insufficient, rely on your training knowledge. Always provide a substantive answer."
            return f"{instruction}\n\nQuestion: {question}\nContext: {context}\nAnswer is:"
        else:
            instruction = "Use your training knowledge to answer the question."
            return f"{instruction}\n\nQuestion: {question}\nAnswer is:"
    
    def _extract_answer_from_qwen_response(self, response: str) -> str:
        """
        Extract answer from Qwen's response using structured format
        
        Args:
            response: Raw response from Qwen model
            
        Returns:
            Extracted answer string
        """
        # Use the base class structured extraction
        return self._extract_structured_answer(response)
    
    def _extract_answer_from_gemini_response(self, response: str) -> str:
        """
        Extract answer from Gemini's response using structured format
        
        Args:
            response: Raw response from Gemini model
            
        Returns:
            Extracted answer string
        """
        # Use the base class structured extraction
        return self._extract_structured_answer(response)
    
    def _extract_answer_from_flan_t5_response(self, response: str) -> str:
        """
        Extract answer from FLAN-T5 response (usually already concise)
        
        Args:
            response: Raw response from FLAN-T5 model
            
        Returns:
            Cleaned answer string
        """
        return response.strip()
    
    def _create_model_specific_prompt(self, question: str, model_name: str, context: str = None) -> str:
        """
        Create model-specific prompt based on model type
        
        Args:
            question: The question to answer
            model_name: Name of the model
            context: Optional context from retrieval
            
        Returns:
            Formatted prompt string
        """
        if self._is_qwen_model(model_name):
            return self._create_qwen_prompt(question, context)
        elif self._is_gemini_model(model_name):
            return self._create_gemini_prompt(question, context)
        elif self._is_flan_t5_model(model_name):
            return self._create_flan_t5_prompt(question, context)
        else:
            # Default to FLAN-T5 style for unknown models
            return self._create_flan_t5_prompt(question, context)
    
    def _extract_model_specific_answer(self, response: str, model_name: str) -> str:
        """
        Extract answer using model-specific method
        
        Args:
            response: Raw model response
            model_name: Name of the model
            
        Returns:
            Extracted answer string
        """
        if self._is_qwen_model(model_name):
            return self._extract_answer_from_qwen_response(response)
        elif self._is_gemini_model(model_name):
            return self._extract_answer_from_gemini_response(response)
        elif self._is_flan_t5_model(model_name):
            return self._extract_answer_from_flan_t5_response(response)
        else:
            # Default to FLAN-T5 extraction
            return self._extract_answer_from_flan_t5_response(response)
    
    def label_samples(self, dataset_name: str, sample_size: int, model_name: str, oner_max_docs: int = None, ircot_max_docs: int = None, samples: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Label samples using the optimized labeling strategy
        
        Args:
            dataset_name: Name of the dataset to process
            sample_size: Number of samples to process
            model_name: Name of the model to use for labeling
            
        Returns:
            Dictionary containing labeled samples and metadata
        """
        # Validate inputs
        validation_errors = self.validate_inputs(dataset_name, sample_size, model_name)
        if validation_errors:
            raise ValueError(f"Input validation failed: {validation_errors}")
        
        # Store current dataset for logging
        self.current_dataset = dataset_name
        
        # Set model-specific defaults based on architecture (following Adaptive-RAG paper)
        if oner_max_docs is None:
            if self._is_flan_t5_model(model_name):
                oner_max_docs = 15  # Encoder-decoder models can handle more documents
            else:
                oner_max_docs = 6   # Decoder-only models (Gemini/Qwen) use fewer documents
        self.oner_max_docs = oner_max_docs
            
        if ircot_max_docs is None:
            if self._is_flan_t5_model(model_name):
                ircot_max_docs = 6   # T5 models use 6 for multi-hop
            else:
                ircot_max_docs = 3   # Iterative models use 3 for multi-hop
        self.ircot_max_docs = ircot_max_docs
        
        self.log_labeling_start(dataset_name, sample_size, model_name)
        start_time = time.time()
        
        # Use provided filtered samples or load from dataset processor
        self.logger.info(f"Using provided filtered samples ({len(samples)} samples)")

        
        # Determine required systems based on dataset type
        required_systems = self.get_required_systems(dataset_name)
        dataset_type = get_dataset_type(dataset_name)
        
        self.logger.info(f"Using optimized strategy for {dataset_type} dataset {dataset_name}")
        self.logger.info(f"Required systems: {required_systems}")
        
        # Log start of Q&A interactions
        if hasattr(self.logger, 'log_qa_interaction'):
            self.logger.info(f"Starting Q&A logging for {dataset_name} with {sample_size} samples")
        
        # Process samples with optimized strategy
        individual_results = []
        
        # Use configured max workers
        max_workers = min(self.parallel_config['max_workers'], len(samples))
        
        # Process samples in parallel across available servers
        self.logger.info(f"Processing {len(samples)} samples with {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(self._process_sample, sample, dataset_name, model_name, required_systems): sample
                for sample in samples
            }
            
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    individual_results.append(result)  # Always append results
                    
                    # Update statistics
                    if result and result.get('label') == 'DISCARDED':
                        self.update_stats(
                            total_samples_processed=1,
                            successful_labels=0,
                            failed_labels=0,
                            discarded_samples=1
                        )
                    else:
                        self.update_stats(
                            total_samples_processed=1,
                            successful_labels=1 if result and result['label'] in ['A', 'B', 'C'] else 0,
                            failed_labels=0  # Optimized labeler no longer fails samples except for real errors
                        )
                    
                    # Log progress with timing info
                    if len(individual_results) % 100 == 0:
                        avg_steps = sum(r.get('steps', 0) for r in individual_results) / len(individual_results)
                        avg_time = sum(r.get('processing_time', 0) for r in individual_results) / len(individual_results)
                        self.logger.info(f"Progress: {len(individual_results)} samples processed, {len(individual_results)/len(samples)*100:.1f}% complete, avg steps: {avg_steps:.2f}, avg time: {avg_time:.2f}s")
                
                except Exception as e:
                    self.logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {str(e)}")
                    individual_results.append(self._create_error_result(sample, str(e)))
                    self.update_stats(failed_labels=1)
            
            # Ensure all futures complete and executor shuts down properly
            executor.shutdown(wait=True)
        
        # Aggregate results
        processing_time = time.time() - start_time
        self.update_stats(processing_time=processing_time)
        
        aggregated_results = self.aggregate_results(individual_results, dataset_name, model_name)
        
        aggregated_results.update({
            'strategy_type': 'optimized',
            'dataset_type': dataset_type,
            'systems_used': required_systems,
            'systems_skipped': self._get_skipped_systems(dataset_name),
            'processing_time_seconds': processing_time,
            'samples_per_second': len(samples) / processing_time if processing_time > 0 else 0,
            'early_stopping_enabled': True,
            'run_all_systems': False,
        })
        
        # Save LLM interactions if logger supports it
        if hasattr(self.logger, 'save_interactions'):
            session_metadata = {
                'strategy': 'optimized',
                'dataset': dataset_name,
                'model': model_name,
                'sample_size': len(samples),
                'dataset_type': dataset_type,
                'systems_used': required_systems,
                'systems_skipped': self._get_skipped_systems(dataset_name),
                'processing_time_seconds': processing_time,
                'early_stopping_enabled': True,
            }
            self.logger.save_interactions(session_metadata=session_metadata)
        
        self.log_labeling_end(dataset_name, aggregated_results)
        return aggregated_results
    
    def get_required_systems(self, dataset_name: str) -> List[str]:
        """
        Get list of systems needed for this dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of required system names
        """
        if dataset_name in self.single_hop_datasets:
            return ["nor_qa"]  # Only NOR for single-hop
        else:
            return ["nor_qa", "oner_qa"]  # NOR and ONER for multi-hop
    
    def _process_sample(self, 
                       sample: Dict[str, Any], 
                       dataset_name: str, 
                       model_name: str,
                       required_systems: List[str]) -> Dict[str, Any]:
        """
        Process a single sample using the optimized strategy
        
        Args:
            sample: Sample data with question, context, and answer
            dataset_name: Name of the dataset being processed
            model_name: Name of the model to use
            required_systems: List of systems to run for this dataset type
            
        Returns:
            Dictionary with labeling result
        """
        import time
        
        # Start timing annotation for this sample
        annotation_start_time = time.time()
        sample_processing_time = 0.0
        system_steps_map = {}
        
        sample_id = self._get_sample_id(sample)
        ground_truths = self._extract_ground_truth(sample)
        
        self.logger.debug(f"Processing sample {sample_id} with systems: {required_systems}")
        
        system_results = {}
        dataset_type = self._get_dataset_type(dataset_name)
        
        if dataset_type == 'single_hop':
            # Single-hop strategy: Only run NOR
            result = self._run_system(sample, "nor_qa", model_name, dataset_name)
            system_results["nor_qa"] = result
            
            # Check for resource exhausted error
            if result.get('resource_exhausted', False) or result.get('answer', '').strip() == '__RESOURCE_EXHAUSTED__':
                self.logger.warning(f"Resource exhausted error for sample {sample_id} - discarding sample")
                return self._create_discarded_result(sample, "Resource exhausted during NOR system call")
            
            # Track steps and processing time from this system
            system_steps_map["nor_qa"] = result.get('steps', 0)
            sample_processing_time += result.get('processing_time', 0.0)
            
            # Apply single-hop labeling logic
            label, reasoning, primary_system, match_type = self._apply_single_hop_labeling_logic(
                system_results, ground_truths, model_name, sample
            )
            
            # Check if LLM call failed during labeling logic
            if label is None and reasoning == "__LLM_FAILED__":
                self.logger.warning(f"LLM call failed during single-hop labeling - discarding sample {self._get_sample_id(sample)}")
                return self._create_discarded_result(sample, "LLM call failed during single-hop synthetic check")
            
        else:
            # Multi-hop strategy: Run NOR, then ONER if needed
            # Try NOR first
            nor_result = self._run_system(sample, "nor_qa", model_name, dataset_name)
            system_results["nor_qa"] = nor_result
            
            # Check for resource exhausted error in NOR
            if nor_result.get('resource_exhausted', False) or nor_result.get('answer', '').strip() == '__RESOURCE_EXHAUSTED__':
                self.logger.warning(f"Resource exhausted error for sample {sample_id} during NOR - discarding sample")
                return self._create_discarded_result(sample, "Resource exhausted during NOR system call")
            
            # Track steps and processing time from NOR system
            system_steps_map["nor_qa"] = nor_result.get('steps', 0)
            sample_processing_time += nor_result.get('processing_time', 0.0)
            
            # Check if NOR succeeded with exact match first (quick check before running ONER)
            is_correct, matched_gt = self._evaluate_answer(nor_result.get('answer', ''), ground_truths, model_name)
            if is_correct:
                # NOR worked with exact match, no need to run ONER
                label, reasoning, primary_system, match_type = 'A', "NOR system answered correctly (exact match)", 'nor_qa', 'exact_match'
            else:
                # NOR failed exact match, try synthetic check for NOR (use raw answer)
                question = sample.get('question_text', '') if sample else ''
                is_semantic_correct, semantic_gt = self._check_answer_with_synthetic(
                    question=question,
                    answer=nor_result.get('raw_answer', ''),
                    ground_truths=ground_truths,
                    system_name="NOR"
                )
                
                # Check if LLM call failed during synthetic checking
                if is_semantic_correct is None and semantic_gt == "__LLM_FAILED__":
                    self.logger.warning(f"LLM call failed during NOR synthetic check - discarding sample {self._get_sample_id(sample)}")
                    return self._create_discarded_result(sample, "LLM call failed during NOR synthetic check")
                
                if is_semantic_correct:
                    # NOR worked with semantic match, no need to run ONER
                    label, reasoning, primary_system, match_type = 'A', "NOR system answered correctly (LLM semantic match)", 'nor_qa', 'llm_match'
                else:
                    # NOR failed both checks, try ONER
                    oner_result = self._run_system(sample, "oner_qa", model_name, dataset_name)
                    system_results["oner_qa"] = oner_result
                    
                    # Check for resource exhausted error in ONER
                    if oner_result.get('resource_exhausted', False) or oner_result.get('answer', '').strip() == '__RESOURCE_EXHAUSTED__':
                        self.logger.warning(f"Resource exhausted error for sample {sample_id} during ONER - discarding sample")
                        return self._create_discarded_result(sample, "Resource exhausted during ONER system call")
                    
                    # Check for server unavailable error in ONER
                    if oner_result.get('server_unavailable', False) or oner_result.get('answer', '').strip() == '__SERVER_UNAVAILABLE__':
                        error_detail = oner_result.get('error', 'Server unavailable')
                        self.logger.warning(f"Server unavailable error for sample {sample_id} during ONER - discarding sample: {error_detail}")
                        return self._create_discarded_result(sample, f"Server unavailable during ONER system call: {error_detail}")
                    
                    # Track steps and processing time from ONER system
                    system_steps_map["oner_qa"] = oner_result.get('steps', 0)
                    sample_processing_time += oner_result.get('processing_time', 0.0)
                    
                    # Apply ONER-specific logic (exact then synthetic)
                    label, reasoning, primary_system, match_type = self._apply_multi_hop_labeling_logic(
                        system_results, ground_truths, model_name, sample
                    )
                    
                    # Check if LLM call failed during labeling logic
                    if label is None and reasoning == "__LLM_FAILED__":
                        self.logger.warning(f"LLM call failed during multi-hop labeling - discarding sample {self._get_sample_id(sample)}")
                        return self._create_discarded_result(sample, "LLM call failed during multi-hop synthetic check")
        
        # Collect system answers for the result entry (both processed and raw)
        system_answers = {}
        for sys, res in system_results.items():
            system_answers[sys] = {
                'answer': res.get('answer', ''),
                'raw_answer': res.get('raw_answer', '')
            }
        
        # Calculate annotation time for this sample
        annotation_time = time.time() - annotation_start_time
        
        # Update logger with sample completion and label count
        if hasattr(self.logger, 'update_sample_completion'):
            sample_id = self._get_sample_id(sample)
            self.logger.update_sample_completion(sample_id, label)
        
        # Calculate total steps as sum of all systems used (nor + oner + ircot)
        total_steps = sum(system_steps_map.values())
        
        # Always create a result entry (no discarded samples)
        return self.create_result_entry(
            sample=sample,
            label=label,
            reasoning=reasoning,
            systems_used=list(system_results.keys()),
            systems_succeeded=[sys for sys, res in system_results.items() 
                             if self._evaluate_answer(res.get('answer', ''), ground_truths, model_name)[0]],
            system_answers=system_answers,
            annotation_time=annotation_time,
            steps=total_steps,
            system_steps=system_steps_map,
            processing_time=sample_processing_time,
            primary_system=primary_system,
            match_type=match_type,
            system_results=system_results
        )
    
    def _apply_single_hop_labeling_logic(self, 
                                        system_results: Dict[str, Any],
                                        ground_truths: List[str],
                                        model_name: str,
                                        sample: Dict[str, Any] = None) -> tuple[str, str]:
        """
        Apply single-hop labeling logic: Only use NOR system
        
        Args:
            system_results: Results from systems
            ground_truths: List of ground truths
            model_name: Name of the model for evaluation
            sample: Original sample data (for discarded sample logging)
            
        Returns:
            Tuple of (label, reasoning, primary_system, match_type) where:
            - label can be 'A' or 'B' (no discarded samples)
            - reasoning explains the labeling decision
            - primary_system indicates which system's answer to use
            - match_type indicates 'exact_match' or 'llm_match'
        """
        # For single-hop datasets, only check NOR system
        nor_result = system_results.get('nor_qa', {})
        nor_answer = nor_result.get('answer', '')
        nor_raw_answer = nor_result.get('raw_answer', '')
        question = sample.get('question_text', '') if sample else ''
        
        self.logger.debug(f"🔍 Single-hop labeling for NOR system:")
        self.logger.debug(f"   Question: {question[:100]}...")
        self.logger.debug(f"   NOR answer: '{nor_answer}' (raw: '{nor_raw_answer}')")
        self.logger.debug(f"   Ground truths: {ground_truths}")
        
        # Step 1: Try exact match for NOR system (use processed answer)
        is_correct, matched_gt = self._evaluate_answer(nor_answer, ground_truths, model_name)
        if is_correct:
            self.logger.debug(f"✅ NOR exact match succeeded with: '{matched_gt}'")
            return 'A', "NOR system answered correctly (exact match)", 'nor_qa', 'exact_match'
        
        self.logger.debug(f"❌ NOR exact match failed, trying synthetic check...")
        
        # Step 2: Try synthetic check for NOR system if exact match failed (use raw answer)
        is_semantic_correct, semantic_gt = self._check_answer_with_synthetic(
            question=question,
            answer=nor_raw_answer,
            ground_truths=ground_truths,
            system_name="NOR"
        )
        
        # Check if LLM call failed during synthetic checking
        if is_semantic_correct is None and semantic_gt == "__LLM_FAILED__":
            self.logger.warning(f"LLM call failed during NOR synthetic check - sample needs to be discarded")
            return None, "__LLM_FAILED__", None, None
        
        if is_semantic_correct:
            self.logger.debug(f"✅ NOR semantic match succeeded with: '{semantic_gt}'")
            return 'A', "NOR system answered correctly (LLM semantic match)", 'nor_qa', 'llm_match'
        
        self.logger.debug(f"❌ NOR semantic match also failed")
        
        # Step 3: Both exact and semantic checks failed, default to B for single-hop
        return 'B', "NOR system failed both exact and semantic checks, single-hop dataset defaults to B", 'oner_qa', 'fallback'

    def _apply_multi_hop_labeling_logic(self, 
                                       system_results: Dict[str, Any],
                                       ground_truths: List[str],
                                       model_name: str,
                                       sample: Dict[str, Any] = None) -> tuple[str, str]:
        """
        Apply multi-hop labeling logic: Check NOR first, then ONER
        
        Args:
            system_results: Results from systems
            ground_truths: List of ground truths
            model_name: Name of the model for evaluation
            sample: Original sample data
            
        Returns:
            Tuple of (label, reasoning, primary_system, match_type) where:
            - label can be 'A', 'B' or 'C'
            - reasoning explains the labeling decision
            - primary_system indicates which system's answer to use
            - match_type indicates 'exact_match', 'llm_match', or 'fallback'
        """
        question = sample.get('question_text', '') if sample else ''
        
        self.logger.debug(f"🔍 Multi-hop labeling:")
        self.logger.debug(f"   Question: {question[:100]}...")
        self.logger.debug(f"   Ground truths: {ground_truths}")
        
        # ======== STEP 1: Check NOR system ========
        nor_result = system_results.get('nor_qa', {})
        nor_answer = nor_result.get('answer', '')
        nor_raw_answer = nor_result.get('raw_answer', '')
        
        self.logger.debug(f"   NOR answer: '{nor_answer}' (raw: '{nor_raw_answer}')")
        
        # Try exact match for NOR system first
        is_correct, matched_gt = self._evaluate_answer(nor_answer, ground_truths, model_name)
        if is_correct:
            self.logger.debug(f"✅ NOR exact match succeeded with: '{matched_gt}'")
            return 'A', "NOR system answered correctly (exact match)", 'nor_qa', 'exact_match'
        
        self.logger.debug(f"❌ NOR exact match failed, trying NOR synthetic check...")
        
        # Try synthetic check for NOR system if exact match failed
        is_semantic_correct, semantic_gt = self._check_answer_with_synthetic(
            question=question,
            answer=nor_raw_answer,
            ground_truths=ground_truths,
            system_name="NOR"
        )
        
        # Check if LLM call failed during synthetic checking
        if is_semantic_correct is None and semantic_gt == "__LLM_FAILED__":
            self.logger.warning(f"LLM call failed during NOR synthetic check - sample needs to be discarded")
            return None, "__LLM_FAILED__", None, None
        
        if is_semantic_correct:
            self.logger.debug(f"✅ NOR semantic match succeeded with: '{semantic_gt}'")
            return 'A', "NOR system answered correctly (LLM semantic match)", 'nor_qa', 'llm_match'
        
        self.logger.debug(f"❌ NOR semantic match failed")
        
        # ======== STEP 2: Check ONER system ========
        oner_result = system_results.get('oner_qa', {})
        oner_answer = oner_result.get('answer', '')
        oner_raw_answer = oner_result.get('raw_answer', '')
        
        self.logger.debug(f"   ONER answer: '{oner_answer}' (raw: '{oner_raw_answer}')")
        
        # Try exact match for ONER system
        is_correct, matched_gt = self._evaluate_answer(oner_answer, ground_truths, model_name)
        if is_correct:
            self.logger.debug(f"✅ ONER exact match succeeded with: '{matched_gt}'")
            return 'B', "ONER system answered correctly (exact match)", 'oner_qa', 'exact_match'
        
        self.logger.debug(f"❌ ONER exact match failed, trying ONER synthetic check...")
        
        # Try synthetic check for ONER system if exact match failed
        is_semantic_correct, semantic_gt = self._check_answer_with_synthetic(
            question=question,
            answer=oner_raw_answer,
            ground_truths=ground_truths,
            system_name="ONER"
        )
        
        # Check if LLM call failed during synthetic checking
        if is_semantic_correct is None and semantic_gt == "__LLM_FAILED__":
            self.logger.warning(f"LLM call failed during ONER synthetic check - sample needs to be discarded")
            return None, "__LLM_FAILED__", None, None
        
        if is_semantic_correct:
            self.logger.debug(f"✅ ONER semantic match succeeded with: '{semantic_gt}'")
            return 'B', "ONER system answered correctly (LLM semantic match)", 'oner_qa', 'llm_match'
        
        self.logger.debug(f"❌ ONER semantic match failed")
        
        # Both NOR and ONER failed - default to multi-hop label
        self.logger.debug(f"❌ Both NOR and ONER failed all checks, defaulting to C")
        return 'C', "Both NOR and ONER failed, defaulting to multi-hop", 'ircot_qa', 'fallback'

    def _check_answer_with_synthetic(self, question: str, answer: str, ground_truths: List[str], system_name: str) -> tuple[bool, str]:
        """
        Perform direct synthetic checking
        
        Args:
            question: The question being asked
            answer: The system's answer to evaluate
            ground_truths: List of acceptable ground truth answers  
            system_name: Name of the system that produced the answer
            
        Returns:
            Tuple of (is_semantically_correct, best_ground_truth_matched)
            Special case: Returns (None, "__LLM_FAILED__") if LLM call fails and sample should be discarded
        """
        self.logger.debug(f"🔍 Starting synthetic check for {system_name}")
        
        if not self.synthetic_checker:
            self.logger.warning(f"❌ Synthetic checker not available for {system_name}")
            return False, ""
        
        if not answer or not question:
            self.logger.warning(f"❌ Missing answer or question for {system_name} synthetic check")
            return False, ""
        
        try:
            # Check against all ground truths (for multi-answer datasets like TriviaQA)
            for i, gt in enumerate(ground_truths):
                if not gt:
                    continue
                
                self.logger.debug(f"   🤖 Checking semantic equivalence:")
                self.logger.debug(f"      Answer: '{answer}'")
                self.logger.debug(f"      Ground Truth {i+1}: '{gt}'")
                    
                result = self.synthetic_checker.check_semantic_correctness(
                    question=question,
                    answer=answer,
                    ground_truth=gt,
                    source_pipeline=system_name.lower()
                )
                
                # Check if LLM call failed during synthetic checking
                if result.error:
                    self.logger.warning(f"❌ LLM call failed during synthetic check for {system_name} - discarding sample: {result.error}")
                    return None, "__LLM_FAILED__"
                
                self.logger.debug(f"      Result: {'✅ SEMANTICALLY CORRECT' if result.is_semantically_correct else '❌ Not equivalent'}")
                
                if result.is_semantically_correct:
                    self.logger.debug(f"✅ LLM semantic match found for {system_name}: '{answer}' ≡ '{gt}'")
                    return True, gt
                    
            self.logger.debug(f"❌ No semantic match found for {system_name} after checking {len(ground_truths)} ground truths")
            return False, ""
                    
        except Exception as e:
            self.logger.error(f"❌ Synthetic checker failed for {system_name}: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            # Treat exception as LLM failure and discard sample
            return None, "__LLM_FAILED__"

    def _retry_request_if_needed(self, request_data: Dict[str, Any], model_name: str, 
                                initial_response: Dict[str, Any], initial_latency: float, 
                                initial_raw_answer: str, system_name: str, max_retries: int = 3) -> tuple:
        """
        Enhanced retry mechanism with multiple attempts and exponential backoff
        Handles both empty answers and failed requests (timeouts, errors)
        
        Args:
            request_data: The request data to retry
            model_name: Name of the model
            initial_response: Initial response from server
            initial_latency: Initial request latency
            initial_raw_answer: Initial raw answer
            system_name: Name of the system (NOR/ONER) for logging
            max_retries: Maximum number of retry attempts
            
        Returns:
            Tuple of (final_raw_answer, final_response, final_latency)
        """
        raw_answer = initial_raw_answer
        response = initial_response
        latency = initial_latency
        
        # Check if we need to retry
        needs_retry = (
            not raw_answer or  # Empty answer
            not response.get('success', True) or  # Failed request
            'error' in response  # Error in response
        )
        
        if not needs_retry:
            return raw_answer, response, latency
            
        # Log the reason for retry
        if not raw_answer:
            retry_reason = "empty answer"
        elif not response.get('success', True):
            retry_reason = "failed request"
        else:
            retry_reason = f"error: {response.get('error', 'unknown')}"
            
        # Try up to max_retries times with exponential backoff
        for retry_attempt in range(1, max_retries + 1):
            self.logger.warning(f"{system_name} retry {retry_attempt}/{max_retries} due to {retry_reason}...")
            
            # Shorter backoff intervals: 1s, 5s, 10s
            retry_delays = [1, 5, 10]
            delay = retry_delays[min(retry_attempt - 1, len(retry_delays) - 1)]
            time.sleep(delay)
            
            retry_start = time.time()
            try:
                retry_response = self.server_manager.process_request(request_data, model_name)
                retry_latency = time.time() - retry_start
                retry_raw_answer = retry_response.get('answer', '').strip()
                
                # Check if retry was successful
                retry_success = (
                    retry_raw_answer and  # Got a non-empty answer
                    retry_response.get('success', True) and  # Request succeeded
                    'error' not in retry_response  # No error in response
                )
                
                if retry_success:
                    self.logger.info(f"{system_name} retry {retry_attempt} successful: got valid response")
                    return retry_raw_answer, retry_response, retry_latency
                else:
                    if not retry_raw_answer:
                        self.logger.warning(f"{system_name} retry {retry_attempt} still returned empty response")
                    elif not retry_response.get('success', True):
                        self.logger.warning(f"{system_name} retry {retry_attempt} still failed: {retry_response.get('error', 'unknown')}")
                    
            except Exception as e:
                self.logger.error(f"{system_name} retry {retry_attempt} failed with exception: {e}")
                # Continue to next retry attempt
                
        # All retries failed
        self.logger.error(f"{system_name} system failed after {max_retries} retry attempts, using original response")
        return raw_answer, response, latency

    def _run_system(self, sample: Dict[str, Any], system_name: str, model_name: str, dataset_name: str = None) -> Dict[str, Any]:
        """
        Run a specific QA system on a sample
        
        Args:
            sample: Sample data with question and context
            system_name: Name of the system to run ('nor_qa', 'oner_qa')
            model_name: Name of the model to use
            dataset_name: Name of the dataset being processed
            
        Returns:
            Dictionary with system response
        """
        try:
            if system_name == 'nor_qa':
                # NOR = No Retrieval, just LLM
                result = self._run_nor_system(sample, model_name)
            elif system_name == 'oner_qa':
                # ONER = One Retrieval step + LLM
                result = self._run_oner_system(sample, model_name, dataset_name)
            else:
                raise ValueError(f"Unknown system: {system_name}")
            
            self.logger.debug(f"System {system_name} response: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error running system {system_name}: {str(e)}")
            return {
                'answer': '',
                'system': system_name,
                'error': str(e),
                'success': False
            }
    
    def _run_nor_system(self, sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Run NOR system: No Retrieval, just LLM with model-specific prompting
        """
        question = sample.get('question_text', '') 
        sample_id = self._get_sample_id(sample)
        ground_truths = self._extract_ground_truth(sample)
        
        # Create model-specific prompt
        prompt = self._create_model_specific_prompt(question, model_name)
        
        request_data = {
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.0
        }
        
        start_time = time.time()
        response = self.server_manager.process_request(request_data, model_name)
        latency = time.time() - start_time
        
        # Check for resource exhausted error
        if response.get('resource_exhausted', False) or response.get('answer', '').strip() == '__RESOURCE_EXHAUSTED__':
            self.logger.warning(f"Resource exhausted error for sample {sample_id} in NOR system - returning special discard result")
            return {
                'answer': '__RESOURCE_EXHAUSTED__',
                'raw_answer': '__RESOURCE_EXHAUSTED__',
                'system': 'nor_qa',
                'server_id': response.get('server_id'),
                'processing_time': latency,
                'success': False,
                'resource_exhausted': True,
                'steps': 0,
                'error': 'Resource exhausted - sample discarded'
            }
        
        # Check for server unavailable error
        if response.get('server_unavailable', False) or response.get('answer', '').strip() == '__SERVER_UNAVAILABLE__':
            error_detail = response.get('error', 'Server unavailable')
            self.logger.warning(f"Server unavailable error for sample {sample_id} in NOR system - returning special discard result: {error_detail}")
            return {
                'answer': '__SERVER_UNAVAILABLE__',
                'raw_answer': '__SERVER_UNAVAILABLE__',
                'system': 'nor_qa',
                'server_id': response.get('server_id'),
                'processing_time': latency,
                'success': False,
                'server_unavailable': True,
                'steps': 0,
                'error': f'Server unavailable - sample discarded: {error_detail}'
            }
        
        raw_answer = response.get('answer', '').strip()
        
        # Enhanced retry mechanism with multiple attempts (handles both empty answers and failures)
        raw_answer, response, latency = self._retry_request_if_needed(
            request_data, model_name, response, latency, raw_answer, "NOR"
        )
        
        # Extract answer using model-specific method
        answer = self._extract_model_specific_answer(raw_answer, model_name)
        
        # Evaluate answer correctness
        is_correct, matched_gt = self._evaluate_answer(answer, ground_truths, model_name)
        
        # Log Q&A interaction with ground truth
        if hasattr(self, 'logger') and hasattr(self.logger, 'log_qa_interaction'):
            self.logger.log_qa_interaction(
                server_id=response.get('server_id', 'unknown'),
                question=question,
                answer=answer,
                system_type='nor_qa',
                dataset_name=getattr(self, 'current_dataset', 'unknown'),
                sample_id=sample_id,
                latency=latency,
                request_data=request_data,
                ground_truth=matched_gt if matched_gt else ground_truths[0] if ground_truths else "",
                is_correct=is_correct,
                # Add model-specific metadata
                model_name=model_name,
                is_qwen_model=self._is_qwen_model(model_name),
                raw_answer=raw_answer  # Store raw answer for debugging
            )
        
        return {
            'answer': answer,
            'raw_answer': raw_answer,  # Store both for debugging
            'system': 'nor_qa',
            'server_id': response.get('server_id'),
            'processing_time': response.get('processing_time', latency),
            'success': True,
            'steps': 0  # NOR: 0 steps (no retrieval-generate cycle)
        }
    
    def _run_oner_system(self, sample: Dict[str, Any], model_name: str, dataset_name: str = None) -> Dict[str, Any]:
        """
        Run ONER system: One Retrieval step + LLM with model-specific prompting
        """
        question = sample.get('question_text', '')
        sample_id = self._get_sample_id(sample)
        ground_truths = self._extract_ground_truth(sample)
        
        # Step 1: Retrieve documents
        retrieved_docs = self._retrieve_documents(question, dataset_name, max_docs=self.oner_max_docs, system_type="oner")
        
        # Step 2: Create context from retrieved documents
        context = self._format_retrieved_docs(retrieved_docs)
        
        # Step 3: Create model-specific prompt with context
        prompt = self._create_model_specific_prompt(question, model_name, context)
        
        request_data = {
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.0
        }
        
        start_time = time.time()
        response = self.server_manager.process_request(request_data, model_name)
        latency = time.time() - start_time
        
        # Check for resource exhausted error
        if response.get('resource_exhausted', False) or response.get('answer', '').strip() == '__RESOURCE_EXHAUSTED__':
            self.logger.warning(f"Resource exhausted error for sample {sample_id} in ONER system - returning special discard result")
            return {
                'answer': '__RESOURCE_EXHAUSTED__',
                'raw_answer': '__RESOURCE_EXHAUSTED__',
                'system': 'oner_qa',
                'server_id': response.get('server_id'),
                'processing_time': latency,
                'success': False,
                'resource_exhausted': True,
                'retrieved_docs': len(retrieved_docs),
                'steps': 1,
                'error': 'Resource exhausted - sample discarded'
            }
        
        # Check for server unavailable error
        if response.get('server_unavailable', False) or response.get('answer', '').strip() == '__SERVER_UNAVAILABLE__':
            error_detail = response.get('error', 'Server unavailable')
            self.logger.warning(f"Server unavailable error for sample {sample_id} in ONER system - returning special discard result: {error_detail}")
            return {
                'answer': '__SERVER_UNAVAILABLE__',
                'raw_answer': '__SERVER_UNAVAILABLE__',
                'system': 'oner_qa',
                'server_id': response.get('server_id'),
                'processing_time': latency,
                'success': False,
                'server_unavailable': True,
                'retrieved_docs': len(retrieved_docs),
                'steps': 1,
                'error': f'Server unavailable - sample discarded: {error_detail}'
            }
        
        raw_answer = response.get('answer', '').strip()
        
        # Enhanced retry mechanism with multiple attempts (handles both empty answers and failures)
        raw_answer, response, latency = self._retry_request_if_needed(
            request_data, model_name, response, latency, raw_answer, "ONER"
        )
        
        # Extract answer using model-specific method
        answer = self._extract_model_specific_answer(raw_answer, model_name)
        
        # Evaluate answer correctness
        is_correct, matched_gt = self._evaluate_answer(answer, ground_truths, model_name)
        
        # Log Q&A interaction with ground truth
        if hasattr(self, 'logger') and hasattr(self.logger, 'log_qa_interaction'):
            self.logger.log_qa_interaction(
                server_id=response.get('server_id', 'unknown'),
                question=question,
                answer=answer,
                system_type='oner_qa',
                dataset_name=dataset_name or getattr(self, 'current_dataset', 'unknown'),
                sample_id=sample_id,
                latency=latency,
                request_data=request_data,
                ground_truth=matched_gt if matched_gt else ground_truths[0] if ground_truths else "",
                is_correct=is_correct,
                retrieved_documents=retrieved_docs,
                # Add model-specific metadata
                model_name=model_name,
                is_qwen_model=self._is_qwen_model(model_name),
                raw_answer=raw_answer  # Store raw answer for debugging
            )
        
        return {
            'answer': answer,
            'raw_answer': raw_answer,  # Store both for debugging
            'system': 'oner_qa',
            'server_id': response.get('server_id'),
            'processing_time': response.get('processing_time', latency),
            'success': True,
            'retrieved_docs': len(retrieved_docs),
            'steps': 1  # ONER: 1 step (one retrieval-generate cycle)
        }
    
    def _get_corpus_name(self, dataset_name: str) -> str:
        """
        Get the appropriate corpus name for the dataset
        """
        # Map dataset names to their corresponding corpus names
        corpus_mapping = {
            'hotpotqa': 'hotpotqa',
            '2wikimultihopqa': '2wikimultihopqa', 
            'musique': 'musique',
            'squad': 'wiki',  # Single-hop datasets use wiki corpus
            'nq': 'wiki',
            'trivia': 'wiki'
        }
        
        # Default to dataset name if not in mapping
        return corpus_mapping.get(dataset_name.lower(), dataset_name.lower())

    def _retrieve_documents(self, question: str, dataset_name: str = None, max_docs: int = 15, system_type: str = "oner") -> List[Dict[str, Any]]:
        """
        Retrieve documents using the retriever server
        
        Args:
            question: The question to retrieve documents for
            dataset_name: Name of the dataset to determine corpus
            max_docs: Maximum number of documents to retrieve (overrides system defaults)
            system_type: Type of system ("oner") to determine optimal retrieval count
            
        Returns:
            List of retrieved documents
            
        Note:
            Based on Adaptive-RAG paper findings for T5 models:
            - ONER (Single-step): 15 documents (optimal for single-hop questions)
            - The optimized labeler only uses ONER for retrieval-based systems
        """
        try:
            # Determine optimal retrieval count based on system type if max_docs not explicitly set
            if max_docs is None:  # Default value, use system-specific optimization
                max_docs = self.oner_max_docs

            
            # Use the determined max_docs for retrieval
            corpus_name = self._get_corpus_name(dataset_name)
            
            # Call retriever server
            retriever_url = "http://localhost:8000/retrieve"
            payload = {
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": question,
                "max_hits_count": max_docs,
                "corpus_name": corpus_name
            }
            
            response = requests.post(retriever_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                retrieval = result.get('retrieval', [])
                self.logger.debug(f"Retrieved {len(retrieval)} documents for question: {question[:50]}... (system: {system_type}, max_docs: {max_docs})")
                return retrieval
            else:
                self.logger.error(f"Retriever server error: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.debug(f"Error calling retriever server: {e}")  # Non-critical, system continues without retrieval
            return []
    
    def _format_retrieved_docs(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into context string
        """
        if not docs:
            return "No relevant documents found."
        
        context_parts = []
        for i, doc in enumerate(docs):  # Use all retrieved documents
            title = doc.get('title', f'Document {i+1}')
            text = doc.get('text', doc.get('paragraph_text', ''))
            if text:
                # Truncate each document to prevent context overflow
                truncated_text = text[:3000] + "..." if len(text) > 3000 else text # Optimized for Gemini 2.5: better info quality at low cost
                context_parts.append(f"[{title}] {truncated_text}")
        
        return "\n\n".join(context_parts)

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
            'timestamp': datetime.now().isoformat()
        }

    def _create_discarded_result(self, sample: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Create a result entry for discarded samples (e.g., due to resource exhausted or server unavailable errors)
        
        Args:
            sample: Original sample data
            reason: Reason for discarding the sample
            
        Returns:
            Discarded result dictionary
        """
        ground_truths = self._extract_ground_truth(sample)
        
        # Determine the type of failure based on the reason
        is_resource_exhausted = 'resource exhausted' in reason.lower() or 'resource_exhausted' in reason.lower()
        is_server_unavailable = 'server unavailable' in reason.lower() or 'server_unavailable' in reason.lower()
        
        result = {
            'sample_id': self._get_sample_id(sample),
            'question': sample.get('question_text', ''),
            'ground_truth': ground_truths[0] if ground_truths else "",
            'answer': '',
            'raw_answer': '',
            'label': 'DISCARDED',
            'reasoning': f"Sample discarded: {reason}",
            'systems_used': [],
            'systems_succeeded': [],
            'discarded': True,
            'error': False,  # This is not an error, it's an intentional discard
            'timestamp': datetime.now().isoformat()
        }
        
        # Add specific failure type flags
        if is_resource_exhausted:
            result['resource_exhausted'] = True
        if is_server_unavailable:
            result['server_unavailable'] = True
            
        return result

    def _get_skipped_systems(self, dataset_name: str) -> List[str]:
        """
        Get list of systems that are skipped for efficiency
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of skipped system names
        """
        if dataset_name.lower() in ['squad', 'nq', 'trivia']:
            # Single-hop datasets skip ONER and IRCOT
            return ['oner_qa', 'ircot_qa']
        else:
            # Multi-hop datasets skip only IRCOT (early stopping)
            return ['ircot_qa']

    def _get_dataset_type(self, dataset_name: str) -> str:
        """
        Determine if dataset is single-hop or multi-hop
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            'single_hop' or 'multi_hop'
        """
        single_hop_datasets = ['squad', 'nq', 'trivia']
        multi_hop_datasets = ['hotpotqa', '2wikimultihopqa', 'musique']
        
        dataset_lower = dataset_name.lower()
        
        if dataset_lower in single_hop_datasets:
            return 'single_hop'
        elif dataset_lower in multi_hop_datasets:
            return 'multi_hop'
        else:
            # Default to multi-hop for unknown datasets
            self.logger.warning(f"Unknown dataset {dataset_name}, defaulting to multi-hop")
            return 'multi_hop'

