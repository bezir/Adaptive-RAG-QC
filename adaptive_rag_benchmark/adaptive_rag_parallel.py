#!/usr/bin/env python3
"""
Parallel Adaptive RAG Pipeline with Multi-Worker Support

This script provides massive speedup through:
1. Multiple workers for parallel query processing
2. Port range support for multiple LLM instances (local models)
3. Parallel API calls for Gemini
4. Concurrent system execution (NOR, ONER, IRCOT)

Usage:
    # Gemini (10 parallel API calls)
    python adaptive_rag_parallel.py --model gemini --dataset hotpotqa --workers 10 ...
    
    # Local LLMs (10 servers on ports 8010-8019)
    python adaptive_rag_parallel.py --model flan-t5-xl --dataset hotpotqa --port-range 8010-8019 ...
"""

import argparse
import json
import os
import sys
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError
import logging
import subprocess
import tempfile
import concurrent.futures
import threading
import multiprocessing as mp

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from collections import defaultdict
from tqdm import tqdm

# Add project root to path for scaled_silver_labeling imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Conditional import for classifier (only needed when not using --force)

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
TORCH_AVAILABLE = True

import google.generativeai as genai
import requests
import numpy as np
import string
from functools import wraps

# Import IRCOT Bridge Adapter with robust IRCOT implementation for real step counting
from scaled_silver_labeling.adapters.ircot_bridge_adapter import IRCoTBridgeAdapter

# Import the robust IRCoT engine directly for true parallel processing
# Note: Import will be done after logger is defined to avoid NameError
ROBUST_IRCOT_AVAILABLE = False

# Retry decorator with exponential backoff
def retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=60.0, backoff_factor=2.0):
    """
    Retry decorator with exponential backoff for API calls
    
    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Initial delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)  
        backoff_factor: Factor to multiply delay by on each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            delay = base_delay
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    
                    # Don't retry on the last attempt
                    if attempt == max_retries:
                        break
                    
                    # Log the retry attempt
                    logger = logging.getLogger(__name__)
                    logger.warning(f"Attempt {attempt + 1}/{max_retries + 1} failed for {func.__name__}: {e}. Retrying in {delay:.1f}s...")
                    
                    # Wait before retrying
                    time.sleep(delay)
                    
                    # Exponential backoff with jitter
                    delay = min(delay * backoff_factor + random.uniform(0, 1), max_delay)
            
            # If all retries failed, raise the last exception
            raise last_exception
        return wrapper
    return decorator

def setup_experiment_logging(experiment_output_dir: str, verbose: bool = False):
    """Setup logging to save detailed logs to experiment folder and clean console output"""
    # Create logs directory
    log_dir = os.path.join(experiment_output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    
    # Clear existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # File handler for detailed logs
    detailed_log_file = os.path.join(log_dir, 'detailed_experiment.log')
    file_handler = logging.FileHandler(detailed_log_file)
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(file_formatter)
    
    # Console handler for essential progress only
    console_handler = logging.StreamHandler()
    if verbose:
        console_handler.setLevel(logging.INFO)
    else:
        console_handler.setLevel(logging.WARNING)  # Only warnings and errors
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(console_formatter)
    
    # Configure root logger
    logging.root.setLevel(logging.DEBUG)
    logging.root.addHandler(file_handler)
    logging.root.addHandler(console_handler)
    
    # Reduce verbosity for specific noisy loggers
    logging.getLogger('httpx').setLevel(logging.WARNING)
    logging.getLogger('urllib3').setLevel(logging.WARNING)
    logging.getLogger('requests').setLevel(logging.WARNING)
    logging.getLogger('google.generativeai').setLevel(logging.WARNING)
    
    return detailed_log_file

# Set up logging (will be reconfigured per experiment)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Now import RobustIRCoT after logger is defined
try:
    from robust_ircot.core.engine import RobustIRCoT
    ROBUST_IRCOT_AVAILABLE = True
    logger.info("✅ RobustIRCoT engine is available")
except ImportError as e:
    ROBUST_IRCOT_AVAILABLE = False
    logger.warning(f"RobustIRCoT not available: {e}, will use fallback implementation")

@dataclass
class WorkerConfig:
    """Configuration for a single worker"""
    worker_id: int
    port: Optional[int] = None  # None for Gemini
    is_available: bool = True
    ircot_engine: Optional[Any] = None  # Pre-initialized IRCoT engine for this worker

class LLMClassifier:
    """Uses the generator LLM itself for one-shot query complexity classification"""
    
    def __init__(self, model: str, workers: List[WorkerConfig]):
        self.model = model
        self.workers = workers
        self.is_gemini = self._is_gemini_model(model)
        self.classification_stats = {
            'total_classifications': 0,
            'A_classifications': 0,
            'B_classifications': 0,
            'C_classifications': 0,
            'failed_classifications': 0
        }
    
    def _is_gemini_model(self, model_name: str) -> bool:
        """Check if the model is a Gemini model"""
        return 'gemini' in model_name.lower()
    
    def _is_qwen_model(self, model_name: str) -> bool:
        """Check if the model is a Qwen model"""
        return 'qwen' in model_name.lower()
    
    def _is_flan_t5_model(self, model_name: str) -> bool:
        """Check if the model is a FLAN-T5 model"""
        model_name_lower = model_name.lower()
        return 'flan-t5' in model_name_lower or 'flan_t5' in model_name_lower
    
    def _create_classification_prompt(self, question: str) -> str:
        """Create classification prompt based on model type"""
        
        # Base instruction for all models
        base_instruction = """You are classifying questions by complexity for a Retrieval-Augmented Generation (RAG) system based on your internal knowledge.

Classification categories:
- A (Simple): Questions you can answer confidently using only your parametric knowledge without any external context
- B (Medium): Questions that need one retrieval step (retrieving multiple documents in a single search) combined with your parametric knowledge  
- C (Complex): Questions requiring multiple retrieval steps or iterative reasoning to connect information from multiple queries

IMPORTANT: Questions may seem complex at first but could be simpler based on your existing knowledge. Carefully assess:
- What information do you already know from your training?
- What specific gaps do you have that require external retrieval?
- Can you fill multiple information gaps with your existing knowledge, or do you need retrieval?
- If you need retrieval, can one comprehensive search provide all missing information, or do you need multiple separate searches?

Examples:
- A: "What is the capital of France?" (can answer with parametric knowledge)
- B: "What is the population of Tokyo in 2023?" (may need one search for current data)
- B: "Who was the mayor of Kocaeli, Turkiye in 1972?" (may need one search for specific historical fact)
- C: "Compare the economic policies of the leaders who were in power during the 2008 financial crisis in USA and UK" (need multiple searches and cross-source reasoning)

Question: {question}

Based on the question above, classify it as A, B, or C. Carefully consider your existing knowledge first:
1. Do I know enough to answer this completely with my parametric knowledge? → A
2. Do I have most information but need one targeted search to fill specific gaps? → B  
3. Do I need multiple separate searches or complex reasoning across multiple sources? → C

Respond with ONLY the letter: A, B, or C"""

        if self._is_gemini_model(self.model):
            prompt = base_instruction.format(question=question)
            
        elif self._is_qwen_model(self.model):
            # Simpler prompt for Qwen to ensure clear response
            prompt = f"""Classify this question into one of three categories:
A: Simple question (can answer from parametric knowledge)
B: Medium question (needs one retrieval step)  
C: Complex question (needs multiple retrieval steps)

Question: {question}

Reply with just the letter A, B, or C.
Classification:"""
            
        else:  # FLAN-T5
            prompt = f"""Classify this question by complexity:

Question: {question}

A = Can answer with parametric knowledge only
B = Need one retrieval step  
C = Need multiple retrieval steps

Classification:"""
        
        return prompt
    
    def _call_llm_for_classification(self, prompt: str, worker: WorkerConfig = None) -> str:
        """Call the LLM for classification"""
        try:
            if self._is_gemini_model(self.model):
                # Use Gemini API
                import google.generativeai as genai
                genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(
                    prompt,
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.0,
                        max_output_tokens=5,  # Very short response expected
                        top_p=1.0
                    )
                )
                return response.text.strip().upper()
            
            else:
                # Use local LLM server
                if worker is None:
                    worker = self.workers[0]  # Use first available worker
                
                params = {
                    "prompt": prompt,
                    "max_length": 5,  # Very short response expected
                    "temperature": 0.0,
                    "do_sample": False,
                    "top_k": 50,
                    "top_p": 1.0,
                    "num_return_sequences": 1,
                    "keep_prompt": False
                }
                
                # For Qwen, we need to use the correct endpoint and response parsing
                response = requests.get(
                    f"http://localhost:{worker.port}/generate/",  # Note the trailing slash
                    params=params,
                    timeout=30  # Shorter timeout for classification
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Qwen returns generated_texts list
                    generated_text = ""
                    if 'generated_texts' in result and result['generated_texts']:
                        generated_text = result['generated_texts'][0]
                    elif 'text' in result:
                        generated_text = result['text']
                    elif 'choices' in result and result['choices']:
                        generated_text = result['choices'][0].get('text', '')
                    
                    # Extract only the actual assistant response for classification
                    clean_answer = self._extract_assistant_response(generated_text)
                    clean_text = clean_answer.strip().upper()
                    
                    # Extract just the letter if the response contains extra text
                    for letter in ['A', 'B', 'C']:
                        if letter in clean_text:
                            return letter
                    
                    logger.warning(f"No valid classification found in response: '{clean_answer}' (raw: '{generated_text[:100]}...')")
                    return ""
                else:
                    logger.error(f"Classification call failed: HTTP {response.status_code}")
                    return ""
        
        except Exception as e:
            logger.error(f"LLM classification call failed: {e}")
            return ""
    
    def classify_query(self, question: str, worker: WorkerConfig = None) -> str:
        """
        Classify a single query using the LLM
        
        Args:
            question: The question text
            worker: Worker config for local LLM calls (optional)
        
        Returns:
            Classification ('A', 'B', or 'C')
        """
        self.classification_stats['total_classifications'] += 1
        
        # Create classification prompt
        prompt = self._create_classification_prompt(question)
        
        # Call LLM for classification
        response = self._call_llm_for_classification(prompt, worker)
        
        # Parse response and return classification
        if 'A' in response:
            self.classification_stats['A_classifications'] += 1
            return 'A'
        elif 'B' in response:
            self.classification_stats['B_classifications'] += 1
            return 'B'
        elif 'C' in response:
            self.classification_stats['C_classifications'] += 1
            return 'C'
        else:
            # Default to medium complexity if parsing fails
            logger.warning(f"Failed to parse classification response: '{response}'. Defaulting to 'B'")
            self.classification_stats['failed_classifications'] += 1
            self.classification_stats['B_classifications'] += 1
            return 'B'
    
    def _extract_assistant_response(self, full_response: str) -> str:
        """
        Extract only the assistant's actual response from the full conversation structure.
        
        The Qwen server returns the entire conversation including system prompts, examples,
        and tags. We need to extract only the final assistant response.
        
        Args:
            full_response: The complete response including system prompts, examples, etc.
            
        Returns:
            Clean assistant response text
        """
        if not full_response:
            return ""
        
        # Split by lines and look for the last "assistant" section
        lines = full_response.split('\n')
        
        # Find the last occurrence of "assistant" (case-insensitive)
        assistant_start_idx = -1
        for i in reversed(range(len(lines))):
            line = lines[i].strip().lower()
            if line == 'assistant' or line.startswith('assistant'):
                assistant_start_idx = i + 1
                break
        
        if assistant_start_idx >= 0 and assistant_start_idx < len(lines):
            # Extract everything after the last "assistant" tag
            assistant_response = '\n'.join(lines[assistant_start_idx:]).strip()
            
            # Remove any remaining tags or artifacts
            assistant_response = self._clean_response_artifacts(assistant_response)
            
            return assistant_response
        
        # Fallback: If no clear assistant section found, try to extract answer pattern
        # Look for "Answer is:" pattern which is common in our prompts
        answer_patterns = [
            "Answer is:",
            "answer is:",
            "Answer:",
            "answer:"
        ]
        
        for pattern in answer_patterns:
            if pattern in full_response:
                # Extract everything after the pattern
                parts = full_response.split(pattern, 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    # Take only the first meaningful sentence/phrase
                    answer = self._clean_response_artifacts(answer)
                    return answer
        
        # Final fallback: Return the last meaningful line
        meaningful_lines = [line.strip() for line in lines if line.strip() and 
                          not line.strip().lower() in ['system', 'user', 'assistant']]
        
        if meaningful_lines:
            last_line = meaningful_lines[-1]
            return self._clean_response_artifacts(last_line)
        
        return full_response.strip()
    
    def _clean_response_artifacts(self, text: str) -> str:
        """
        Clean response artifacts like metadata, extra formatting, etc.
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned response text
        """
        if not text:
            return ""
        
        # Remove common artifacts
        artifacts_to_remove = [
            'system\n',
            'user\n', 
            'assistant\n',
            '\nsystem',
            '\nuser',
            '\nassistant'
        ]
        
        cleaned = text
        for artifact in artifacts_to_remove:
            cleaned = cleaned.replace(artifact, '')
        
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove any leading/trailing quotes or brackets that might be artifacts
        cleaned = cleaned.strip('\"\'[]{}()')
        
        return cleaned.strip()
    
    def _extract_assistant_response(self, full_response: str) -> str:
        """
        Extract only the assistant's actual response from the full conversation structure.
        
        The Qwen server returns the entire conversation including system prompts, examples,
        and tags. We need to extract only the final assistant response.
        
        Args:
            full_response: The complete response including system prompts, examples, etc.
            
        Returns:
            Clean assistant response text
        """
        if not full_response:
            return ""
        
        # Split by lines and look for the last "assistant" section
        lines = full_response.split('\n')
        
        # Find the last occurrence of "assistant" (case-insensitive)
        assistant_start_idx = -1
        for i in reversed(range(len(lines))):
            line = lines[i].strip().lower()
            if line == 'assistant' or line.startswith('assistant'):
                assistant_start_idx = i + 1
                break
        
        if assistant_start_idx >= 0 and assistant_start_idx < len(lines):
            # Extract everything after the last "assistant" tag
            assistant_response = '\n'.join(lines[assistant_start_idx:]).strip()
            
            # Remove any remaining tags or artifacts
            assistant_response = self._clean_response_artifacts(assistant_response)
            
            return assistant_response
        
        # Fallback: If no clear assistant section found, try to extract answer pattern
        # Look for "Answer is:" pattern which is common in our prompts
        answer_patterns = [
            "Answer is:",
            "answer is:",
            "Answer:",
            "answer:"
        ]
        
        for pattern in answer_patterns:
            if pattern in full_response:
                # Extract everything after the pattern
                parts = full_response.split(pattern, 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    # Take only the first meaningful sentence/phrase
                    answer = self._clean_response_artifacts(answer)
                    return answer
        
        # Final fallback: Return the last meaningful line
        meaningful_lines = [line.strip() for line in lines if line.strip() and 
                          not line.strip().lower() in ['system', 'user', 'assistant']]
        
        if meaningful_lines:
            last_line = meaningful_lines[-1]
            return self._clean_response_artifacts(last_line)
        
        return full_response.strip()

    def classify_queries_batch(self, queries: List[Dict]) -> List[str]:
        """
        Classify multiple queries in parallel
        
        Args:
            queries: List of query dictionaries
            
        Returns:
            List of classifications
        """
        if self._is_gemini_model(self.model):
            # For Gemini, use ThreadPoolExecutor for parallel API calls
            predictions = [None] * len(queries)
            
            def classify_single(index_query_pair):
                index, query = index_query_pair
                question_text = query.get('question_text', query.get('question', ''))
                return index, self.classify_query(question_text)
            
            max_workers = min(len(self.workers), len(queries))
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all classification tasks
                future_to_index = {
                    executor.submit(classify_single, (i, query)): i
                    for i, query in enumerate(queries)
                }
                
                # Collect results with progress bar
                with tqdm(total=len(queries), desc="LLM Classification") as pbar:
                    for future in concurrent.futures.as_completed(future_to_index):
                        try:
                            index, classification = future.result(timeout=60)
                            predictions[index] = classification
                            pbar.update(1)
                        except concurrent.futures.TimeoutError:
                            index = future_to_index[future]
                            logger.warning(f"Classification timeout for query {index}")
                            predictions[index] = 'B'  # Default
                            pbar.update(1)
                        except Exception as e:
                            index = future_to_index[future]
                            logger.warning(f"Classification error for query {index}: {e}")
                            predictions[index] = 'B'  # Default
                            pbar.update(1)
            
            return predictions
        else:
            # For local LLMs, distribute across workers
            queries_per_worker = len(queries) // len(self.workers)
            extra_queries = len(queries) % len(self.workers)
            
            query_batches = []
            start_idx = 0
            
            for i in range(len(self.workers)):
                batch_size = queries_per_worker + (1 if i < extra_queries else 0)
                if batch_size > 0:
                    query_batches.append((
                        queries[start_idx:start_idx + batch_size],
                        self.workers[i],
                        start_idx
                    ))
                    start_idx += batch_size
            
            predictions = [None] * len(queries)
            
            def classify_batch(batch_worker_offset):
                batch, worker, offset = batch_worker_offset
                batch_predictions = []
                for query in batch:
                    question_text = query.get('question_text', query.get('question', ''))
                    classification = self.classify_query(question_text, worker)
                    batch_predictions.append(classification)
                return offset, batch_predictions
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(self.workers)) as executor:
                # Submit all batch tasks
                future_to_batch = {
                    executor.submit(classify_batch, batch_data): batch_data
                    for batch_data in query_batches
                }
                
                # Collect results with progress bar
                with tqdm(total=len(queries), desc="LLM Classification") as pbar:
                    for future in concurrent.futures.as_completed(future_to_batch):
                        try:
                            offset, batch_predictions = future.result(timeout=300)
                            for i, pred in enumerate(batch_predictions):
                                predictions[offset + i] = pred
                            pbar.update(len(batch_predictions))
                        except Exception as e:
                            batch_data = future_to_batch[future]
                            batch, worker, offset = batch_data
                            logger.error(f"Worker {worker.worker_id} classification failed: {e}")
                            # Fill with defaults
                            for i in range(len(batch)):
                                predictions[offset + i] = 'B'
                            pbar.update(len(batch))
            
            return predictions
    
    def get_classification_stats(self) -> Dict[str, Any]:
        """Get classification statistics"""
        return self.classification_stats.copy()


class ClassificationVerifier:
    """Handles verification of classifier predictions using the same LLM that will generate answers"""
    
    def __init__(self, model: str, workers: List[WorkerConfig]):
        self.model = model
        self.workers = workers
        self.is_gemini = self._is_gemini_model(model)
        self.verification_stats = {
            'A_to_B_upgrades': 0,
            'A_to_C_upgrades': 0,
            'B_to_C_upgrades': 0,
            'total_verifications': 0,
            'A_kept': 0,
            'B_kept': 0,
            'C_no_verification': 0
        }
    
    def _is_gemini_model(self, model_name: str) -> bool:
        """Check if the model is a Gemini model"""
        return 'gemini' in model_name.lower()
    
    def _is_qwen_model(self, model_name: str) -> bool:
        """Check if the model is a Qwen model"""
        return 'qwen' in model_name.lower()
    
    def _is_flan_t5_model(self, model_name: str) -> bool:
        """Check if the model is a FLAN-T5 model"""
        model_name_lower = model_name.lower()
        return 'flan-t5' in model_name_lower or 'flan_t5' in model_name_lower
    
    def _create_verification_prompt(self, question: str, current_classification: str) -> str:
        """Create verification prompt based on current classification and model type"""
        
        if current_classification == 'A':
            # A -> B verification: Can you answer without any context?
            if self._is_gemini_model(self.model):
                prompt = f"""You are about to answer a question using only your parametric knowledge (no external context/retrieval).

Question: {question}

Can you confidently answer this question using ONLY your internal knowledge without needing any external context or information retrieval? 

Consider:
- Do you have sufficient knowledge about this topic?
- Is this a factual question you can answer definitively?
- Are there no ambiguities that would require additional context?

Respond with ONLY:
- "CONFIDENT" if you can answer confidently without any external context
- "NEED_CONTEXT" if you need external information to provide a good answer

Your response:"""
            
            elif self._is_qwen_model(self.model):
                prompt = f"""<|im_start|>system
You are about to answer a question using only your parametric knowledge (no external context/retrieval). Evaluate if you can answer confidently.
<|im_end|>
<|im_start|>user
Question: {question}

Can you confidently answer this question using ONLY your internal knowledge without needing any external context or information retrieval? 

Respond with ONLY:
- "CONFIDENT" if you can answer confidently without any external context
- "NEED_CONTEXT" if you need external information to provide a good answer
<|im_end|>
<|im_start|>assistant
"""
            
            else:  # FLAN-T5
                prompt = f"""Question: {question}

Can you answer this question confidently using only your internal knowledge without any external context?
Answer with CONFIDENT or NEED_CONTEXT:"""
        
        elif current_classification == 'B':
            # B -> C verification: Can you answer with single-step retrieval?
            if self._is_gemini_model(self.model):
                prompt = f"""You are about to answer a question using single-step retrieval (one search) plus your parametric knowledge.

Question: {question}

Can you confidently answer this question with just ONE retrieval step (single search) combined with your parametric knowledge?

Consider:
- Is this a single-hop question that needs only one piece of information?
- Can one search provide sufficient context to answer completely?
- Does this NOT require connecting information from multiple sources?

Respond with ONLY:
- "SINGLE_HOP" if one retrieval step is sufficient
- "MULTI_HOP" if you need multiple retrieval steps to connect information

Your response:"""
            
            elif self._is_qwen_model(self.model):
                prompt = f"""<|im_start|>system
You are about to answer a question using single-step retrieval plus your parametric knowledge. Evaluate if one retrieval step is sufficient.
<|im_end|>
<|im_start|>user
Question: {question}

Can you confidently answer this question with just ONE retrieval step (single search) combined with your parametric knowledge?

Respond with ONLY:
- "SINGLE_HOP" if one retrieval step is sufficient
- "MULTI_HOP" if you need multiple retrieval steps to connect information
<|im_end|>
<|im_start|>assistant
"""
            
            else:  # FLAN-T5
                prompt = f"""Question: {question}

Can you answer this question with just one retrieval step plus your knowledge?
Answer with SINGLE_HOP or MULTI_HOP:"""
        
        else:
            # Classification 'C' - no verification needed
            return ""
        
        return prompt
    
    def _call_llm_for_verification(self, prompt: str, worker: WorkerConfig = None) -> str:
        """Call the LLM (same model used for generation) for verification with timeout and retry handling"""
        max_attempts = 3
        base_timeout = 15  # 15 seconds per attempt
        
        for attempt in range(max_attempts):
            try:
                if self._is_gemini_model(self.model):
                    # Use Gemini API with thread-safe timeout
                    import google.generativeai as genai
                    
                    def _call_gemini_with_timeout():
                        genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
                        model = genai.GenerativeModel(self.model)
                        response = model.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.0,
                                max_output_tokens=100,  # Allow for brief analysis + decision
                                top_p=1.0
                            )
                        )
                        return response.text.strip().upper()
                    
                    # Use ThreadPoolExecutor for thread-safe timeout handling
                    with ThreadPoolExecutor(max_workers=1) as executor:
                        future = executor.submit(_call_gemini_with_timeout)
                        try:
                            return future.result(timeout=base_timeout)
                        except FuturesTimeoutError:
                            raise TimeoutError("Verification API call timed out")
                
                else:
                    # Use local LLM server
                    if worker is None:
                        worker = self.workers[0]  # Use first available worker
                    
                    params = {
                        "prompt": prompt,
                        "max_length": 100,  # Allow for brief analysis + decision
                        "temperature": 0.0,
                        "do_sample": False,
                        "top_k": 50,
                        "top_p": 1.0,
                        "num_return_sequences": 1,
                        "keep_prompt": False
                    }
                    
                    response = requests.get(
                        f"http://localhost:{worker.port}/generate",
                        params=params,
                        timeout=base_timeout  # Strict timeout per attempt
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        generated_text = result.get('choices', [{}])[0].get('text', '')
                        if not generated_text:
                            generated_text = result.get('text', '')
                        return generated_text.strip().upper()
                    else:
                        logger.warning(f"Verification call failed (attempt {attempt + 1}/{max_attempts}): HTTP {response.status_code}")
                        
            except (TimeoutError, requests.exceptions.Timeout) as e:
                logger.warning(f"Verification timeout (attempt {attempt + 1}/{max_attempts}): {e}")
            except Exception as e:
                logger.warning(f"Verification call failed (attempt {attempt + 1}/{max_attempts}): {e}")
            
            # Wait before retry (quick backoff)
            if attempt < max_attempts - 1:
                wait_time = 1 + attempt  # 1s, 2s, 3s
                time.sleep(wait_time)
        
        # All attempts failed - discard this verification
        logger.error(f"LLM verification failed after {max_attempts} attempts - discarding sample")
        return ""
    
    def _create_enhanced_verification_prompt(self, question: str, current_classification: str, classifier_info: Dict = None) -> str:
        """
        Create enhanced verification prompt with ML model understanding context
        
        Args:
            question: The question text
            current_classification: Current classification ('A', 'B', or 'C')
            classifier_info: Dictionary with model architecture, probabilities, etc.
        
        Returns:
            Enhanced verification prompt
        """
        # Extract classifier information if available
        if classifier_info:
            model_arch = classifier_info.get('model_architecture', 'Unknown ML Model')
            probabilities = classifier_info.get('probabilities', [0.33, 0.33, 0.34])
            confidence = classifier_info.get('confidence', max(probabilities) if probabilities else 0.5)
            classifier_path = classifier_info.get('classifier_path', 'Unknown')
        else:
            model_arch = 'Unknown ML Model'
            probabilities = [0.33, 0.33, 0.34]
            confidence = 0.5
            classifier_path = 'Unknown'
        
        # Create base context about Adaptive RAG system
        system_context = """ADAPTIVE RAG SYSTEM OVERVIEW:
You are evaluating a classification made by a machine learning model in an Adaptive RAG (Retrieval-Augmented Generation) system.

The system has three complexity levels:
- **A (Simple)**: Answer using only parametric knowledge (no retrieval needed)
- **B (Medium)**: Answer using single-step retrieval + parametric knowledge  
- **C (Complex)**: Answer using multi-hop reasoning with multiple retrieval steps

The goal is to route questions to the most appropriate (and cost-effective) approach."""

        # Format probability distribution
        prob_a, prob_b, prob_c = probabilities[:3] if len(probabilities) >= 3 else [0.33, 0.33, 0.34]
        prob_text = f"A: {prob_a:.3f} ({prob_a*100:.1f}%), B: {prob_b:.3f} ({prob_b*100:.1f}%), C: {prob_c:.3f} ({prob_c*100:.1f}%)"
        
        # Create the enhanced prompt based on model type
        if self._is_gemini_model(self.model):
            prompt = f"""{system_context}

**MACHINE LEARNING MODEL ANALYSIS:**
- **Architecture**: {model_arch}
- **Predicted Classification**: {current_classification}
- **Probability Distribution**: {prob_text}
- **Confidence Score**: {confidence:.3f} ({confidence*100:.1f}%)


**QUESTION TO EVALUATE:**
{question}

**YOUR TASK:**
Analyze this question to determine the optimal routing approach. Use your parametric knowledge as the primary guide, with the classifier's probability distribution as additional context.

**YOUR ANALYSIS:**
1. **Can you answer this question confidently from your parametric knowledge?** If yes → **CHANGE_A**
2. **Do you need to look up specific facts or entities?** If yes → **CHANGE_B**  
3. **Do you need to connect information from multiple sources?** If yes → **CHANGE_C**

**CLASSIFIER CONTEXT:**
The ML model predicted {current_classification} with {confidence:.1f}% confidence.
Probability distribution: {prob_text}

**NOTE:** The probability distribution is provided as helpful context. While ML models can sometimes be overly confident on incorrect predictions, this information may help identify uncertainty patterns in borderline cases.

**INSTRUCTIONS:**
Analyze the ML model's decision, then provide your final recommendation.

You may provide brief analysis, but MUST end your response with exactly one of these decisions:
- **KEEP_{current_classification}**: Keep the original classification (it's correct)
- **CHANGE_A**: Route to simple parametric knowledge approach
- **CHANGE_B**: Route to single-step retrieval approach  
- **CHANGE_C**: Route to multi-hop reasoning approach

Your response:"""
        
        elif self._is_qwen_model(self.model):
            prompt = f"""<|im_start|>system
{system_context}

You are an expert in machine learning, probability theory, and question-answering systems. Analyze the ML model's classification decision.
<|im_end|>
<|im_start|>user
**MACHINE LEARNING MODEL ANALYSIS:**
- **Architecture**: {model_arch}
- **Predicted Classification**: {current_classification}
- **Probability Distribution**: {prob_text}
- **Confidence Score**: {confidence:.3f} ({confidence*100:.1f}%)

**QUESTION TO EVALUATE:**
{question}

**YOUR TASK:**
Analyze this question to determine optimal routing, using your parametric knowledge as the primary guide.

**YOUR ANALYSIS:**
1. Can you answer this from your training knowledge? → **CHANGE_A**
2. Need to look up specific facts? → **CHANGE_B**
3. Need multi-hop reasoning? → **CHANGE_C**

**CLASSIFIER CONTEXT:** Predicted {current_classification} ({confidence:.1f}% confidence).
Probability distribution: {prob_text}

**NOTE:** Use the probability distribution as helpful context. ML models may sometimes be overly confident, so trust your parametric knowledge when making the final decision.

End your response with exactly one of these decisions:
- **KEEP_{current_classification}**: Keep the original classification
- **CHANGE_A**: Route to simple parametric knowledge
- **CHANGE_B**: Route to single-step retrieval
- **CHANGE_C**: Route to multi-hop reasoning
<|im_end|>
<|im_start|>assistant
"""
        
        else:  # FLAN-T5 and others
            prompt = f"""{system_context}

ML Model Analysis:
- Architecture: {model_arch}
- Predicted: {current_classification}
- Probabilities: {prob_text}
- Confidence: {confidence:.3f}

Question: {question}

Analyze using your parametric knowledge: Can you answer this from training knowledge (A), need single lookup (B), or multi-hop reasoning (C)? Classifier predicted {current_classification} ({confidence:.1f}%) with probabilities {prob_text} - use as helpful context.

End your response with exactly one decision: KEEP_{current_classification}, CHANGE_A, CHANGE_B, or CHANGE_C:"""
        
        return prompt
    
    def _parse_enhanced_verification_response(self, response: str, original_classification: str) -> str:
        """
        Parse the enhanced verification response to extract final classification
        
        Args:
            response: LLM response
            original_classification: Original classification to fall back to
        
        Returns:
            Final classification ('A', 'B', or 'C')
        """
        response_upper = response.upper()
        
        # Look for specific change commands first
        if "CHANGE_A" in response_upper:
            return 'A'
        elif "CHANGE_B" in response_upper:
            return 'B'
        elif "CHANGE_C" in response_upper:
            return 'C'
        elif f"KEEP_{original_classification}" in response_upper:
            return original_classification
        
        # Look for decision patterns in analytical responses
        if "ROUTE TO SIMPLE" in response_upper or "SIMPLE PARAMETRIC" in response_upper:
            return 'A'
        elif "ROUTE TO SINGLE-STEP" in response_upper or "SINGLE-STEP RETRIEVAL" in response_upper:
            return 'B'
        elif "ROUTE TO MULTI-HOP" in response_upper or "MULTI-HOP REASONING" in response_upper:
            return 'C'
        
        # Look for recommendation patterns
        if "RECOMMEND A" in response_upper or "CLASSIFICATION A" in response_upper:
            return 'A'
        elif "RECOMMEND B" in response_upper or "CLASSIFICATION B" in response_upper:
            return 'B'
        elif "RECOMMEND C" in response_upper or "CLASSIFICATION C" in response_upper:
            return 'C'
        
        # Look for "keep" or "maintain" patterns
        if ("KEEP" in response_upper or "MAINTAIN" in response_upper or "CORRECT" in response_upper) and original_classification in response_upper:
            return original_classification
        
        # Look for upgrade/change patterns
        if "UPGRADE TO B" in response_upper or "CHANGE TO B" in response_upper:
            return 'B'
        elif "UPGRADE TO C" in response_upper or "CHANGE TO C" in response_upper:
            return 'C'
        elif "UPGRADE TO A" in response_upper or "CHANGE TO A" in response_upper:
            return 'A'
        
        # If response mentions the original classification positively, keep it
        if original_classification in response_upper and any(word in response_upper for word in ["APPROPRIATE", "CORRECT", "OPTIMAL", "SUITABLE"]):
            return original_classification
        
        # Ultimate fallback: keep original
        logger.debug(f"Could not parse verification response definitively: {response[:200]}... Keeping original: {original_classification}")
        return original_classification
    
    def verify_classification(self, question: str, current_classification: str, 
                            classifier_info: Dict = None, worker: WorkerConfig = None) -> str:
        """
        Enhanced verification using ML model understanding - asks LLM to evaluate
        the classifier's prediction with full context about model architecture,
        probability distributions, and adaptive RAG system understanding.
        
        Args:
            question: The question text
            current_classification: Current classification ('A', 'B', or 'C')
            classifier_info: Dict containing:
                - 'model_architecture': Type of classifier (e.g., 'BERT-Large', 'FLAN-T5-Large')
                - 'probabilities': [prob_A, prob_B, prob_C] probability distribution
                - 'confidence': Confidence score (max probability)
                - 'classifier_path': Path to the trained classifier
            worker: Worker config for local LLM calls (optional)
        
        Returns:
            The (potentially updated) classification
        """
        self.verification_stats['total_verifications'] += 1
        
        # No verification needed for 'C' (highest complexity)
        if current_classification == 'C':
            self.verification_stats['C_no_verification'] += 1
            return current_classification
        
        # Create enhanced verification prompt with ML model context
        prompt = self._create_enhanced_verification_prompt(question, current_classification, classifier_info)
        if not prompt:
            return current_classification
        
        # Call LLM for verification
        response = self._call_llm_for_verification(prompt, worker)
        
        # Parse response for the final decision
        final_classification = self._parse_enhanced_verification_response(response, current_classification)
        
        # Update statistics
        if final_classification != current_classification:
            if current_classification == 'A' and final_classification == 'B':
                self.verification_stats['A_to_B_upgrades'] += 1
                logger.debug(f"Enhanced verification: A->B for question: {question[:50]}...")
            elif current_classification == 'A' and final_classification == 'C':
                self.verification_stats['A_to_C_upgrades'] += 1
                logger.debug(f"Enhanced verification: A->C for question: {question[:50]}...")
            elif current_classification == 'B' and final_classification == 'C':
                self.verification_stats['B_to_C_upgrades'] += 1
                logger.debug(f"Enhanced verification: B->C for question: {question[:50]}...")
        else:
            if current_classification == 'A':
                self.verification_stats['A_kept'] += 1
            elif current_classification == 'B':
                self.verification_stats['B_kept'] += 1
        
        return final_classification
    
    def get_verification_stats(self) -> Dict[str, Any]:
        """Get verification statistics"""
        return self.verification_stats.copy()

class ParallelQueryProcessor:
    """Handles parallel processing of queries across multiple workers"""
    
    def _is_gemini_model(self, model_name: str) -> bool:
        """Check if the model is a Gemini model"""
        return 'gemini' in model_name.lower()
    
    def _is_qwen_model(self, model_name: str) -> bool:
        """Check if the model is a Qwen model"""
        return 'qwen' in model_name.lower()
    
    def _is_flan_t5_model(self, model_name: str) -> bool:
        """Check if the model is a FLAN-T5 model"""
        model_name_lower = model_name.lower()
        return 'flan-t5' in model_name_lower or 'flan_t5' in model_name_lower
    
    def __init__(self, model: str, dataset: str, system: str, workers: List[WorkerConfig], server_manager: Optional["LLMServerManager"] = None):
        self.model = model
        self.dataset = dataset
        self.system = system
        self.workers = workers
        self.server_manager = server_manager
        self.is_gemini = model.startswith('gemini')  # Keep for backward compatibility
        self.uses_api = self._is_gemini_model(model)  # For now, only Gemini uses API directly
        self.results = {}
        self.progress_lock = threading.Lock()
        self.queries_processed = 0
        
        # Add retrieval concurrency limiter for IRCOT
        # Limit to 40 concurrent retrieval requests to match worker count
        self.retrieval_semaphore = threading.Semaphore(120)  # Increased for IRCoT: 40 workers × 3 avg iterations
        
        # Pre-initialize RobustIRCoT engines for each worker
        # This avoids initialization overhead for each query
        if system == 'ircot_qa' and ROBUST_IRCOT_AVAILABLE and self._is_qwen_model(model):
            logger.info("🚀 Pre-initializing RobustIRCoT engines for workers...")
            for worker in self.workers:
                if worker.port:
                    try:
                        worker.ircot_engine = RobustIRCoT(
                            model=self.model,
                            dataset=self.dataset,
                            retriever_config={
                                "host": "http://localhost",
                                "port": 8000
                            },
                            logger=logger,
                            server_manager=self.server_manager,  # Pass server manager for port recovery
                            assigned_port=worker.port  # Pass assigned port for parallel processing
                        )
                        logger.info(f"✅ Initialized RobustIRCoT for worker {worker.worker_id} on port {worker.port}")
                    except Exception as e:
                        logger.error(f"Failed to initialize RobustIRCoT for worker {worker.worker_id}: {e}")
                        worker.ircot_engine = None
        
         # Initialize few-shot examples
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
                        'context': 'William Shakespeare was an English playwright and poet widely regarded as the greatest writer in the English language. He was born in Stratford-upon-Avon in 1564 and died in 1616.',
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
                        'context': 'William Shakespeare was an English playwright and poet widely regarded as the greatest writer in the English language. He was born in Stratford-upon-Avon in 1564 and died in 1616.',
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
        
        # Initialize robust IRCOT adapter configuration for real step counting
        self.ircot_adapter = None
        # Removed adapter caching - each query gets fresh adapter to prevent contamination 
        if system == 'ircot_qa':
            try:
                # Set retriever configuration in environment
                os.environ["RETRIEVER_HOST"] = "http://localhost"
                os.environ["RETRIEVER_PORT"] = "8000"
                
                # Create one adapter instance to verify setup works
                test_adapter = IRCoTBridgeAdapter(logger=logger)
                self.ircot_adapter = test_adapter  # Keep for compatibility
                logger.info(f"🚀 Initialized IRCOT adapter configuration for {model} on {dataset}")
                logger.info(f"✅ Using improved retrieval and reasoning with per-worker isolation")
            except Exception as e:
                logger.warning(f"Failed to initialize robust IRCOT adapter: {e}. IRCOT queries will use fallback.")
        
        # Retriever port for ONER system
        self.retriever_port = 8000
    
    def _get_few_shot_examples(self, model_name: str, with_context: bool = True) -> str:
        """
        Generate few-shot examples for the given model matching silver labeling format
        
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
    
    def _create_gemini_prompt(self, question: str, context: str = None) -> str:
        """Create a Gemini-specific prompt with structured answer format and few-shot examples"""
        if context:
            instruction = f"Answer the question using both the provided context documents AND your parametric knowledge. If the context doesn't contain sufficient information, rely on your parametric knowledge to provide the best answer. Always provide a substantive answer rather than saying you don't have enough information. Provide a brief explanation, then end your response with 'Answer is: [your concise answer]'. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples(self.model, with_context=True)
            prompt = f"{instruction}{few_shot_examples}Context: {context}\n\nQuestion: {question}\n\nAnswer is:"
        else:
            instruction = f"Answer the question using your parametric knowledge from training. Provide a brief explanation, then end your response with 'Answer is: [your concise answer]'. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples(self.model, with_context=False)
            prompt = f"{instruction}{few_shot_examples}Question: {question}\n\nAnswer is:"
        return prompt
    
    def _create_qwen_prompt(self, question: str, context: str = None) -> str:
        """Create a Qwen-specific prompt with chat template and few-shot examples"""
        if context:
            # With context (for ONER system)
            system_message = "Answer the question using both the provided context documents AND your parametric knowledge. If the context doesn't contain sufficient information, rely on your parametric knowledge to provide the best answer. Always provide a substantive answer rather than saying you don't have enough information. Provide a brief explanation, then end your response with 'Answer is: [your concise answer]'. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples(self.model, with_context=True)
            user_message = f"{few_shot_examples}Context: {context}\n\nQuestion: {question}\n\nAnswer is:"
        else:
            # Without context (for NOR system)
            system_message = "Answer the question using your parametric knowledge from training. Provide a brief explanation, then end your response with 'Answer is: [your concise answer]'. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples(self.model, with_context=False)
            user_message = f"{few_shot_examples}Question: {question}\n\nAnswer is:"
        
        # Format as chat template
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        return prompt
    
    def _create_flan_t5_prompt(self, question: str, context: str = None) -> str:
        """Create a FLAN-T5 prompt with simple format matching silver labeling"""
        if self.system == 'nor_qa':
            # No retrieval QA - just question, no context at all
            prompt = f"Question: {question}\nAnswer is:"
        elif self.system == 'oner_qa':
            # One-step retrieval QA
            prompt = f"Question: {question}\nContext: {context if context else ''}\nProvide a detailed answer based on the context:\nAnswer is:"
        elif self.system == 'ircot_qa':
            # Iterative retrieval with chain of thought - add reasoning prefix for FLAN-T5
            # Use a simpler prompt that doesn't confuse the model
            if context:
                prompt = f"Question: {question}\nContext: {context}\nAnswer is:"
            else:
                prompt = f"Question: {question}\nAnswer is:"
        else:
            # Default prompt
            prompt = f"Question: {question}\nContext: {context if context else ''}\nAnswer is:"
        return prompt
        

    def _get_worker_ircot_adapter(self, worker_id: str = None, query_id: str = None):
        """Create a fresh IRCoT adapter instance for each query to prevent contamination"""
        if not worker_id:
            import threading
            worker_id = f"worker_{threading.current_thread().ident}"
        
        # Create fresh adapter per query, not per worker
        # This eliminates context contamination between different queries
        if not query_id:
            import time
            query_id = f"{worker_id}_{int(time.time() * 1000000)}"
        
        logger.info(f"🔍 DEBUG: Creating fresh IRCoT adapter for query {query_id}")
        
        try:
            from scaled_silver_labeling.adapters.ircot_bridge_adapter import IRCoTBridgeAdapter
            # Always create fresh instance - no reuse to prevent contamination
            adapter = IRCoTBridgeAdapter(logger=logger)
            adapter_instance_id = id(adapter)
            logger.info(f"✅ DEBUG: Created FRESH IRCoT adapter for query {query_id} (instance ID: {adapter_instance_id})")
            return adapter
        except Exception as e:
            logger.error(f"Failed to create IRCoT adapter for query {query_id}: {e}")
            return None

    @retry_with_backoff(max_retries=3, base_delay=1.0, max_delay=30.0)
    def process_query_direct(self, query: Dict) -> Tuple[str, str, int]:
        """Process a single query directly using appropriate model API with silver labeling format"""
        question_text = query.get('question_text', query.get('question', ''))
        question_id = query.get('question_id', query.get('id', 'unknown'))
        
        # DEBUG: Log worker and question mapping
        import threading
        worker_id = f"worker_{threading.current_thread().ident}"
        logger.info(f"🎯 DEBUG: {worker_id} processing question {question_id}")
        logger.info(f"📝 DEBUG: Question text: {question_text[:100]}...")
        
        # Track actual steps based on retrieval operations
        steps = 0
        
        # Prepare context based on system
        context = None
        if self.system in ['oner_qa', 'ircot_qa']:
            if 'contexts' in query and query['contexts']:
                context = "\n\n".join([
                    f"Wikipedia Title: {ctx.get('title', '')}\n{ctx.get('paragraph_text', '')}"
                    for ctx in query['contexts'][:5]  # Use top 5 contexts
                ])
                # Count retrieval step if context was retrieved
                if context and self.system == 'oner_qa':
                    steps = 1
            elif self.system == 'oner_qa':
                context = self.get_retrieval_context(question_text)
                if context:  # Count retrieval step
                    steps = 1
        
        # For IRCOT, we need to call the robust IRCOT pipeline to get real step counts
        if self.system == 'ircot_qa':
            logger.debug(f"[IRCOT] Processing query ID: {query.get('question_id', 'unknown')}")
            logger.debug(f"[IRCOT] Question: {question_text[:100]}...")
            
            # Get fresh adapter per query to prevent contamination
            worker_adapter = self._get_worker_ircot_adapter(worker_id=worker_id, query_id=question_id)
            if worker_adapter:
                # Use robust IRCOT Bridge Adapter with enhanced retrieval and reasoning
                try:
                    adapter_instance_id = id(worker_adapter)
                    logger.info(f"🧠 DEBUG: {worker_id} using adapter {adapter_instance_id} for question {question_id}")
                    
                    result = worker_adapter.run_ircot_system(
                        sample=query,
                        model_name=self.model,
                        dataset_name=self.dataset
                    )
                    answer = result.get('answer', '')
                    steps = result.get('steps', 0)
                    
                    # DEBUG: Check for contamination keywords
                    contamination_keywords = ['Emmanuelle Seigner', 'Roman Polanski', 'Am I Wrong', 'Étienne de Crécy']
                    for keyword in contamination_keywords:
                        if keyword in answer:
                            logger.error(f"🚨 CONTAMINATION DETECTED: {worker_id} got contaminated answer for {question_id}")
                            logger.error(f"🚨 Question: {question_text[:100]}...")
                            logger.error(f"🚨 Contaminated answer contains: {keyword}")
                            logger.error(f"🚨 Adapter instance: {adapter_instance_id}")
                            break
                    
                    # Log if robust implementation was used
                    if result.get('robust_ircot', False):
                        logger.info(f"✅ [IRCOT] Robust implementation used for query: {steps} iterations, answer: {answer[:50]}...")
                    else:
                        logger.warning(f"⚠️ [IRCOT] Fallback implementation used for query: {steps} iterations")
                    
                    logger.debug(f"[IRCOT] Final answer: {answer}")
                    logger.debug(f"[IRCOT] Steps taken: {steps}")
                    
                    return query.get('question_id', query.get('id', '')), answer, steps
                except Exception as e:
                    logger.error(f"❌ [IRCOT] Robust adapter failed: {e}. Using fallback.")
            else:
                logger.warning(f"⚠️ [IRCOT] No IRCOT adapter available, using fallback")
            
            # Fallback: treat as multi-hop retrieval with estimated steps
            logger.warning(f"[IRCOT] Using FALLBACK mode - implementing iterative retrieval")
            if not context:
                # Use iterative retrieval for IRCOT
                context, steps = self.get_iterative_retrieval_context(question_text)
                logger.debug(f"[IRCOT] FALLBACK: Retrieved context through {steps} iterations, length: {len(context) if context else 0}")
            else:
                # If context already provided, count it as 1 step
                steps = 1
            # Continue with normal processing below
        
        # For NOR and ONER, use the existing logic
        
        # Generate model-specific prompt
        if self._is_gemini_model(self.model):
            prompt = self._create_gemini_prompt(question_text, context)
        elif self._is_qwen_model(self.model):
            prompt = self._create_qwen_prompt(question_text, context)
        else:  # FLAN-T5 and other models
            prompt = self._create_flan_t5_prompt(question_text, context)
        
        # Call appropriate API
        try:
            if self._is_gemini_model(self.model):
                import google.generativeai as genai
                genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
                model = genai.GenerativeModel(self.model)
                response = model.generate_content(prompt)
                answer = response.text.strip()
            else:
                # For non-Gemini models, this should use the local server approach
                # but for now, we'll return empty to indicate it should use batch processing
                logger.warning(f"Direct processing for {self.model} not implemented, use batch processing")
                return query.get('question_id', query.get('id', '')), "", steps
            
            return query.get('question_id', query.get('id', '')), answer, steps
        except Exception as e:
            logger.error(f"API error for {self.model}: {e}")
            return query.get('question_id', query.get('id', '')), "", steps
    
    @retry_with_backoff(max_retries=2, base_delay=0.5, max_delay=5.0)
    def get_retrieval_context(self, question: str) -> str:
        """Get retrieval context for a question (simplified)"""
        try:
            import requests
            payload = {
                "query_text": question,  # Correct parameter name
                "max_hits_count": 5,     # Correct parameter name
                "retrieval_method": "retrieve_from_elasticsearch",
                "corpus_name": self.dataset  # Use dataset as corpus name
            }
            # Use semaphore to limit concurrent retrieval requests
            with self.retrieval_semaphore:
                response = requests.post(
                    "http://localhost:8000/retrieve",
                    json=payload,
                    timeout=60  # Increased timeout for better stability
                )
            if response.status_code == 200:
                result = response.json()
                retrieval_results = result.get('retrieval', [])
                return " ".join([r.get('text', r.get('paragraph_text', '')) for r in retrieval_results[:3]])
            else:
                raise requests.exceptions.HTTPError(f"Retrieval service returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"Retrieval context failed for question: {e}")
            raise  # Re-raise to trigger retry
        return ""
    
    def _extract_assistant_response(self, full_response: str) -> str:
        """
        Extract only the assistant's actual response from the full conversation structure.
        
        The Qwen server returns the entire conversation including system prompts, examples,
        and tags. We need to extract only the final assistant response.
        
        Args:
            full_response: The complete response including system prompts, examples, etc.
            
        Returns:
            Clean assistant response text
        """
        if not full_response:
            return ""
        
        # Split by lines and look for the last "assistant" section
        lines = full_response.split('\n')
        
        # Find the last occurrence of "assistant" (case-insensitive)
        assistant_start_idx = -1
        for i in reversed(range(len(lines))):
            line = lines[i].strip().lower()
            if line == 'assistant' or line.startswith('assistant'):
                assistant_start_idx = i + 1
                break
        
        if assistant_start_idx >= 0 and assistant_start_idx < len(lines):
            # Extract everything after the last "assistant" tag
            assistant_response = '\n'.join(lines[assistant_start_idx:]).strip()
            
            # Remove any remaining tags or artifacts
            assistant_response = self._clean_response_artifacts(assistant_response)
            
            return assistant_response
        
        # Fallback: If no clear assistant section found, try to extract answer pattern
        # Look for "Answer is:" pattern which is common in our prompts
        answer_patterns = [
            "Answer is:",
            "answer is:",
            "Answer:",
            "answer:"
        ]
        
        for pattern in answer_patterns:
            if pattern in full_response:
                # Extract everything after the pattern
                parts = full_response.split(pattern, 1)
                if len(parts) > 1:
                    answer = parts[1].strip()
                    # Take only the first meaningful sentence/phrase
                    answer = self._clean_response_artifacts(answer)
                    return answer
        
        # Final fallback: Return the last meaningful line
        meaningful_lines = [line.strip() for line in lines if line.strip() and 
                          not line.strip().lower() in ['system', 'user', 'assistant']]
        
        if meaningful_lines:
            last_line = meaningful_lines[-1]
            return self._clean_response_artifacts(last_line)
        
        return full_response.strip()
    
    def _clean_response_artifacts(self, text: str) -> str:
        """
        Clean response artifacts like metadata, extra formatting, etc.
        
        Args:
            text: Raw response text
            
        Returns:
            Cleaned response text
        """
        if not text:
            return ""
        
        # Remove common artifacts
        artifacts_to_remove = [
            'system\n',
            'user\n', 
            'assistant\n',
            '\nsystem',
            '\nuser',
            '\nassistant'
        ]
        
        cleaned = text
        for artifact in artifacts_to_remove:
            cleaned = cleaned.replace(artifact, '')
        
        # Remove excessive whitespace
        cleaned = ' '.join(cleaned.split())
        
        # Remove any leading/trailing quotes or brackets that might be artifacts
        cleaned = cleaned.strip('\"\'[]{}()')
        
        return cleaned.strip()
    
    def process_query_batch(self, queries: List[Dict], worker: WorkerConfig) -> Dict[str, Any]:
        """Process a batch of queries using a local LLM server with silver labeling prompts.

        This creates proper silver labeling prompts and sends them directly to the local LLM server
        on the given worker's port, returning a mapping of qid -> {'answer': str, 'steps': int}.
        """
        if worker.port is None:
            # Should not be called for Gemini
            return {}

        results = {}
        
        try:
            import requests
            
            for query in queries:
                question_text = query.get('question_text', query.get('question', ''))
                qid = query.get('question_id', query.get('id', ''))
                
                # Track actual steps
                steps = 0
                
                # Prepare context based on system
                context = None
                if self.system in ['oner_qa', 'ircot_qa']:
                    if 'contexts' in query and query['contexts']:
                        context = "\n\n".join([
                            f"Wikipedia Title: {ctx.get('title', '')}\n{ctx.get('paragraph_text', '')}"
                            for ctx in query['contexts'][:5]  # Use top 5 contexts
                        ])
                        if context and self.system == 'oner_qa':
                            steps = 1
                    elif self.system == 'oner_qa':
                        context = self.get_retrieval_context(question_text)
                        if context:
                            steps = 1
                
                # For IRCOT, we need to call the robust IRCOT pipeline
                if self.system == 'ircot_qa':
                    logger.debug(f"[IRCOT-BATCH] Processing query ID: {qid}")
                    logger.debug(f"[IRCOT-BATCH] Question: {question_text[:100]}...")
                    
                    if self.ircot_adapter:
                        # Use robust IRCOT Bridge Adapter with enhanced retrieval and reasoning
                        try:
                            logger.debug(f"[IRCOT-BATCH] Using robust IRCOT adapter")
                            result = self.ircot_adapter.run_ircot_system(
                                sample=query,
                                model_name=self.model,
                                dataset_name=self.dataset
                            )
                            answer = result.get('answer', '')
                            steps = result.get('steps', 0)
                            
                            # Log if robust implementation was used
                            if result.get('robust_ircot', False):
                                logger.debug(f"✅ [IRCOT-BATCH] Robust implementation used for {qid}: {steps} iterations")
                            else:
                                logger.warning(f"⚠️ [IRCOT-BATCH] Fallback implementation used for {qid}: {steps} iterations")
                            
                            results[qid] = {'answer': answer, 'steps': steps}
                            continue
                        except Exception as e:
                            logger.error(f"❌ [IRCOT-BATCH] Robust adapter failed for {qid}: {e}. Using fallback.")
                    else:
                        logger.warning(f"⚠️ [IRCOT-BATCH] No IRCOT adapter available for {qid}, using fallback")

                    # Fallback: use estimated steps
                    logger.warning(f"[IRCOT-BATCH] Using FALLBACK mode for {qid} - implementing iterative retrieval")
                    
                    # Ensure we have context through iterative retrieval
                    if not context:
                        context, steps = self.get_iterative_retrieval_context(question_text)
                        logger.debug(f"[IRCOT-BATCH] FALLBACK: Retrieved context through {steps} iterations for {qid}")
                    else:
                        # If context already provided, count it as 1 step
                        steps = 1
                
                # Generate model-specific prompt using silver labeling format
                if self._is_gemini_model(self.model):
                    prompt = self._create_gemini_prompt(question_text, context)
                elif self._is_qwen_model(self.model):
                    prompt = self._create_qwen_prompt(question_text, context)
                else:  # FLAN-T5 and other models
                    prompt = self._create_flan_t5_prompt(question_text, context)
                
                # Use different approaches based on model type
                if self._is_qwen_model(self.model) and self.server_manager:
                    # For Qwen models, use server manager with retry mechanism
                    max_server_retries = 3
                    generated_text = ""
                    
                    for retry in range(max_server_retries):
                        try:
                            # Use server manager for load balancing and port recovery
                            request_data = {
                                "prompt": prompt,
                                "max_tokens": 200,
                                "temperature": 0.0,
                                "do_sample": False
                            }
                            
                            response = self.server_manager.process_request(request_data, self.model)
                            
                            if response.get('success') == True:
                                raw_answer = response.get('answer', '').strip()
                                # Extract only the actual assistant response
                                generated_text = self._extract_assistant_response(raw_answer)
                                logger.debug(f"✅ Qwen server manager succeeded for query {qid} on attempt {retry + 1}")
                                break
                            else:
                                error_msg = response.get('error', 'Server request failed')
                                logger.warning(f"⚠️ Qwen server manager failed for query {qid} on attempt {retry + 1}: {error_msg}")
                                if retry < max_server_retries - 1:
                                    # Wait before retry with incremental backoff
                                    time.sleep(0.5 * (retry + 1))
                                    
                        except Exception as e:
                            logger.warning(f"⚠️ Qwen server manager exception for query {qid} on attempt {retry + 1}: {e}")
                            if retry < max_server_retries - 1:
                                time.sleep(0.5 * (retry + 1))
                    
                    results[qid] = {
                        'answer': generated_text,
                        'steps': steps
                    }
                    
                else:
                    # For other local LLMs (FLAN-T5, etc.), use direct HTTP calls
                    params = {
                        "prompt": prompt,
                        "max_length": 200,  # Reasonable length for answers
                        "temperature": 0.0,  # Deterministic for consistency
                        "do_sample": False,  # Deterministic sampling
                        "top_k": 50,
                        "top_p": 1.0,
                        "num_return_sequences": 1,
                        "keep_prompt": False
                    }
                    
                    # Retry HTTP requests with exponential backoff
                    for attempt in range(3):  # Max 3 attempts
                        try:
                            response = requests.get(
                                f"http://localhost:{worker.port}/generate/",  # Note: trailing slash for Qwen compatibility
                                params=params,
                                timeout=60  # 60 second timeout per query
                            )
                            break  # Success, exit retry loop
                        except (requests.exceptions.RequestException, requests.exceptions.Timeout) as e:
                            if attempt == 2:  # Last attempt
                                logger.error(f"Worker {worker.worker_id} HTTP request failed after 3 attempts for query {qid}: {e}")
                                results[qid] = {'answer': '', 'steps': steps}
                                continue  # Skip to next query
                            else:
                                # Wait before retry with exponential backoff
                                wait_time = (2 ** attempt) + random.uniform(0, 1)
                                logger.warning(f"Worker {worker.worker_id} HTTP request attempt {attempt + 1} failed for query {qid}: {e}. Retrying in {wait_time:.1f}s...")
                                time.sleep(wait_time)
                                continue
                    else:
                        # This else block executes only if the for loop completed without break
                        # (i.e., all attempts failed)
                        continue
                    
                    if response.status_code == 200:
                        result = response.json()
                        
                        # Handle different response formats
                        generated_text = ""
                        
                        # Qwen format: {"generated_texts": ["response"], ...}
                        if 'generated_texts' in result and result['generated_texts']:
                            generated_text = result['generated_texts'][0]
                        # OpenAI-style format: {"choices": [{"text": "response"}], ...}
                        elif 'choices' in result and result['choices']:
                            generated_text = result['choices'][0].get('text', '')
                        # Simple format: {"text": "response"}
                        elif 'text' in result:
                            generated_text = result['text']
                        
                        # Extract only the actual assistant response for local LLMs
                        if self._is_qwen_model(self.model):
                            clean_answer = self._extract_assistant_response(generated_text)
                        else:
                            clean_answer = generated_text.strip()
                        
                        results[qid] = {
                            'answer': clean_answer,
                            'steps': steps
                        }
                    else:
                        logger.error(f"Worker {worker.worker_id} failed for query {qid}: HTTP {response.status_code}")
                        results[qid] = {'answer': '', 'steps': steps}

            return results

        except Exception as e:
            logger.error(f"Worker {worker.worker_id} exception: {e}")
            return {}
    
    def process_single_query(self, query: Dict, worker: WorkerConfig) -> Tuple[str, str, int]:
        """Process a single query with a specific worker - enables true parallel processing"""
        if self._is_gemini_model(self.model):
            # For Gemini: Use direct API processing
            return self.process_query_direct(query)
        elif self._is_qwen_model(self.model) and self.server_manager:
            # Direct server processing to bypass server manager lock
            # Use worker's pre-assigned port directly for true parallelism
            return self.process_single_query_direct_qwen(query, worker)
        else:
            # For other local LLMs: Process single query through batch method
            batch_results = self.process_query_batch([query], worker)
            qid = query.get('question_id', query.get('id', ''))
            if qid in batch_results:
                result = batch_results[qid]
                return qid, result.get('answer', ''), result.get('steps', 0)
            else:
                return qid, '', 0

    def process_single_query_direct_qwen(self, query: Dict, worker: WorkerConfig) -> Tuple[str, str, int]:
        """
        Process a single query directly with Qwen server to bypass server manager lock.
        This enables true parallel processing by eliminating the global lock bottleneck.
        """
        qid = query.get('question_id', query.get('id', ''))
        question_text = query.get('question_text', query.get('question', ''))
        
        try:
            # Track actual steps
            steps = 0
            context = None
            
            # Prepare context based on system
            if self.system == 'oner_qa':
                # ONER: Use pre-computed contexts if available, otherwise retrieve
                if 'contexts' in query and query['contexts']:
                    context = "\n\n".join([
                        f"Wikipedia Title: {ctx.get('title', '')}\n{ctx.get('paragraph_text', '')}"
                        for ctx in query['contexts'][:5]  # Use top 5 contexts
                    ])
                    if context:
                        steps = 1
                else:
                    context = self.get_retrieval_context(question_text)
                    if context:
                        steps = 1
            elif self.system == 'ircot_qa':
                # IRCOT: Use the RobustIRCoT engine with assigned port for true parallel processing
                logger.debug(f"[IRCOT-PARALLEL] Processing {qid} with RobustIRCoT engine on port {worker.port}")
                
                if ROBUST_IRCOT_AVAILABLE and worker.port:
                    try:
                        # Use pre-initialized RobustIRCoT engine for this worker
                        ircot_engine = worker.ircot_engine
                        if not ircot_engine:
                            # Fallback: Create a fresh RobustIRCoT instance if not pre-initialized
                            logger.warning(f"Creating RobustIRCoT on-demand for worker {worker.worker_id}")
                            ircot_engine = RobustIRCoT(
                                model=self.model,
                                dataset=self.dataset,
                                retriever_config={
                                    "host": "http://localhost",
                                    "port": 8000
                                },
                                logger=logger,
                                server_manager=self.server_manager,  # Pass server manager for port recovery
                                assigned_port=worker.port  # Pass assigned port for parallel processing
                            )
                        
                        # Extract pre-computed contexts if available
                        initial_contexts = query.get('contexts', [])
                        
                        # Run the robust IRCoT engine
                        result = ircot_engine.run(
                            question=question_text,
                            config_overrides={
                                "initial_retrieval_k": 6,
                                "iterative_retrieval_k": 3,
                                "max_iterations": 5,
                                "max_total_docs": 18,
                                "enable_final_reader": True,
                                "temperature": 0.0
                            },
                            initial_contexts=initial_contexts if initial_contexts else None
                        )
                        
                        # Extract answer and steps from the result
                        final_answer = result.get('answer', '')
                        steps = result.get('iteration_count', 0)
                        
                        logger.debug(f"✅ RobustIRCoT completed for {qid}: {steps} iterations")
                        return qid, final_answer, steps
                        
                    except Exception as e:
                        logger.error(f"RobustIRCoT engine failed for {qid}: {e}")
                        # Fallback to custom implementation
                        context, steps = self._get_robust_ircot_context(
                            question_text, 
                            max_docs=6, 
                            max_iterations=5,
                            initial_contexts=query.get('contexts', [])
                        )
                else:
                    # Fallback to custom implementation if RobustIRCoT not available
                    logger.warning(f"RobustIRCoT not available for {qid}, using fallback")
                    context, steps = self._get_robust_ircot_context(
                        question_text, 
                        max_docs=6, 
                        max_iterations=5,
                        initial_contexts=query.get('contexts', [])
                    )
            
            # For NOR and ONER systems, or if IRCoT bridge adapter failed (fallback)
            if self.system != 'ircot_qa' or context is not None:
                # Generate model-specific prompt
                if self._is_qwen_model(self.model):
                    prompt = self._create_qwen_prompt(question_text, context)
                else:
                    prompt = self._create_flan_t5_prompt(question_text, context)
                
                # Direct HTTP call to worker's assigned port (bypasses server manager lock)
                import requests
                url = f"http://localhost:{worker.port}/generate/"
                params = {
                    "prompt": prompt,
                    "max_length": 512,  # INCREASED for better answers
                    "temperature": 0.0,
                    "do_sample": False
                }
                
                response = requests.get(url, params=params, timeout=90)
                
                if response.status_code == 200:
                    result = response.json()
                    # Handle Qwen response format
                    generated_text = ""
                    if 'generated_texts' in result and result['generated_texts']:
                        generated_text = result['generated_texts'][0]
                    elif 'text' in result:
                        generated_text = result['text']
                    
                    # Extract only the actual assistant response, not the full conversation
                    clean_answer = self._extract_assistant_response(generated_text)
                    
                    return qid, clean_answer.strip(), steps
                else:
                    logger.error(f"Direct Qwen request failed for {qid}: HTTP {response.status_code}")
                    return qid, '', steps
                
        except Exception as e:
            logger.error(f"Direct Qwen processing failed for {qid}: {e}")
            return qid, '', 0

    def process_query_batch_direct(self, queries: List[Dict], worker: WorkerConfig) -> Dict[str, Any]:
        """Process queries using appropriate method based on model type"""
        results = {}
        
        if self._is_gemini_model(self.model):
            # For Gemini: Process queries one by one using API calls
            for query in queries:
                qid, answer, steps = self.process_query_direct(query)
                results[qid] = {'answer': answer, 'steps': steps}
                
                with self.progress_lock:
                    self.queries_processed += 1
        else:
            # For local LLMs: Use batch processing with silver labeling prompts
            return self.process_query_batch(queries, worker)
            
        return results
    
    def process_queries_parallel(self, queries: List[Dict], max_workers: int = None) -> Dict[str, Any]:
        """Process all queries in parallel across workers"""
        if not queries:
            return {}
            
        if max_workers is None:
            max_workers = len(self.workers)
            
        # For Gemini, we can use ThreadPoolExecutor for true parallel API calls
        if self._is_gemini_model(self.model):
            all_results = {}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all queries
                future_to_query = {
                    executor.submit(self.process_query_direct, query): query
                    for query in queries
                }
                
                # Progress bar with timeout handling
                with tqdm(total=len(queries), desc=f"Processing {self.system}") as pbar:
                    try:
                        for future in concurrent.futures.as_completed(future_to_query, timeout=300):  # 5 minute timeout
                            query = future_to_query[future]
                            try:
                                qid, answer, steps = future.result(timeout=120)  # 2 minute timeout per query
                                all_results[qid] = {'answer': answer, 'steps': steps}
                                pbar.update(1)
                            except concurrent.futures.TimeoutError:
                                logger.error(f"Timeout processing query {query.get('question_id', query.get('id', 'unknown'))}")
                                # Add empty result for timeout
                                qid = query.get('question_id', query.get('id', 'unknown'))
                                all_results[qid] = {'answer': '', 'steps': 0}
                                pbar.update(1)
                            except Exception as e:
                                logger.error(f"Error processing query {query.get('question_id', query.get('id', 'unknown'))}: {e}")
                                # Add empty result for error
                                qid = query.get('question_id', query.get('id', 'unknown'))
                                all_results[qid] = {'answer': '', 'steps': 0}
                                pbar.update(1)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Overall timeout reached for {self.system} processing. Cancelling remaining futures.")
                        # Cancel remaining futures and add empty results
                        for future, query in future_to_query.items():
                            if not future.done():
                                future.cancel()
                                qid = query.get('question_id', query.get('id', 'unknown'))
                                if qid not in all_results:
                                    all_results[qid] = {'answer': '', 'steps': 0}
                                    pbar.update(1)
            
            return all_results
        else:
            # True parallel processing - distribute individual queries to workers
            # Instead of batching, process each query individually for maximum concurrency
            all_results = {}
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit each query individually with round-robin worker assignment
                future_to_query = {}
                for i, query in enumerate(queries):
                    worker = self.workers[i % len(self.workers)]  # Round-robin assignment
                    future = executor.submit(self.process_single_query, query, worker)
                    future_to_query[future] = (query, worker)
                
                # Progress bar with timeout handling
                with tqdm(total=len(queries), desc=f"Processing {self.system}") as pbar:
                    try:
                        for future in concurrent.futures.as_completed(future_to_query, timeout=600):  # 10 minute timeout
                            query, worker = future_to_query[future]
                            try:
                                qid, answer, steps = future.result(timeout=300)  # 5 minute timeout per query
                                all_results[qid] = {'answer': answer, 'steps': steps}
                                pbar.update(1)
                            except concurrent.futures.TimeoutError:
                                logger.error(f"Timeout processing query on worker {worker.worker_id}")
                                qid = query.get('question_id', query.get('id', 'unknown'))
                                all_results[qid] = {'answer': '', 'steps': 0}
                                pbar.update(1)
                            except Exception as e:
                                logger.error(f"Error processing query on worker {worker.worker_id}: {e}")
                                qid = query.get('question_id', query.get('id', 'unknown'))
                                all_results[qid] = {'answer': '', 'steps': 0}
                                pbar.update(1)
                    except concurrent.futures.TimeoutError:
                        logger.error(f"Overall timeout reached for {self.system} processing. Cancelling remaining futures.")
                        # Handle timeout for remaining queries
                        for future, (query, worker) in future_to_query.items():
                            if not future.done():
                                future.cancel()
                                qid = query.get('question_id', query.get('id', 'unknown'))
                                if qid not in all_results:
                                    all_results[qid] = {'answer': '', 'steps': 0}
                                    pbar.update(1)
            
            return all_results

    def get_iterative_retrieval_context(self, question: str, max_iterations: int = 3) -> Tuple[str, int]:
        """
        Get retrieval context using iterative retrieval (for IRCOT fallback).
        Returns the combined context and the number of retrieval steps taken.
        """
        try:
            import requests
            
            all_contexts = []
            seen_titles = set()
            current_query = question
            actual_steps = 0
            
            logger.debug(f"[IRCOT-ITERATIVE] Starting iterative retrieval for: {question[:100]}...")
            
            for iteration in range(max_iterations):
                logger.debug(f"[IRCOT-ITERATIVE] Iteration {iteration + 1}/{max_iterations}, Query: {current_query[:100]}...")
                
                # Generate worker identification for request isolation
                import threading
                worker_id = f"worker_{threading.current_thread().ident}"
                
                # Retrieve documents with proper parameters
                payload = {
                    "query_text": current_query,  # Correct parameter name
                    "max_hits_count": 3,          # Correct parameter name
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "corpus_name": self.dataset,   # Use dataset as corpus name
                    "worker_id": worker_id  # Add worker identification for isolation
                }
                
                logger.debug(f"🔍 FALLBACK RETRIEVAL: {worker_id} querying '{current_query[:50]}...'")
                
                # Use semaphore to limit concurrent retrieval requests
                with self.retrieval_semaphore:
                    response = requests.post(
                        "http://localhost:8000/retrieve",
                        json=payload,
                        timeout=60  # Increased timeout for better stability
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    retrieval_results = result.get('retrieval', [])
                    new_docs_found = False
                    
                    for doc in retrieval_results:
                        title = doc.get('title', '')
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            all_contexts.append(f"Title: {title}\n{doc.get('text', doc.get('paragraph_text', ''))}")
                            new_docs_found = True
                    
                    actual_steps += 1
                    
                    # If no new documents found, terminate early
                    if not new_docs_found:
                        logger.debug(f"[IRCOT-ITERATIVE] No new documents found, terminating at iteration {iteration + 1}")
                        break
                    
                    # Update query for next iteration based on what we've found
                    if iteration < max_iterations - 1:
                        # Create a follow-up query based on gathered information
                        context_summary = " ".join([ctx[:100] for ctx in all_contexts[-2:]])  # Last 2 contexts
                        current_query = f"{question} Based on: {context_summary}"
                else:
                    logger.warning(f"[IRCOT-ITERATIVE] Retrieval failed at iteration {iteration + 1}: HTTP {response.status_code}")
                    break
            
            # Combine all contexts
            combined_context = "\n\n".join(all_contexts[:6])  # Limit to 6 documents total
            logger.debug(f"[IRCOT-ITERATIVE] Completed: {actual_steps} iterations, {len(all_contexts)} documents")
            
            return combined_context, actual_steps
            
        except Exception as e:
            logger.error(f"[IRCOT-ITERATIVE] Iterative retrieval failed: {e}")
            # Fall back to single retrieval
            context = self.get_retrieval_context(question)
            return context, 1 if context else 0

    def _get_robust_ircot_context(self, question: str, max_docs: int = 6, max_iterations: int = 5, 
                                 initial_contexts: List = None) -> Tuple[str, int]:
        """
        Implement the same iterative retrieval pattern as RobustIRCoT in scaled_silver_labeling
        but maintain parallelism by avoiding server manager locks.
        
        Config matches RobustIRCoT:
        - initial_retrieval_k: max_docs (6)
        - iterative_retrieval_k: max(3, max_docs // 2) (3) 
        - max_iterations: 5
        - max_total_docs: max_docs * 3 (18)
        """
        try:
            import requests
            
            all_contexts = []
            seen_titles = set()
            current_query = question
            actual_steps = 0
            
            # Use same parameters as RobustIRCoT
            iterative_retrieval_k = max(3, max_docs // 2)  # 3 docs per iteration
            max_total_docs = max_docs * 3  # 18 total docs max
            
            logger.debug(f"[ROBUST-IRCOT] Starting with config: initial_k={max_docs}, iterative_k={iterative_retrieval_k}, max_iter={max_iterations}")
            
            # Use initial contexts if available (same as RobustIRCoT)
            if initial_contexts:
                logger.debug(f"[ROBUST-IRCOT] Using {len(initial_contexts)} pre-computed contexts")
                for ctx in initial_contexts[:max_docs]:
                    title = ctx.get('title', '')
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        all_contexts.append(f"Wikipedia Title: {title}\n{ctx.get('paragraph_text', '')}")
                actual_steps = 1  # Initial retrieval counts as 1 step
            
            # Iterative retrieval (same pattern as RobustIRCoT)
            for iteration in range(max_iterations):
                if len(all_contexts) >= max_total_docs:
                    logger.debug(f"[ROBUST-IRCOT] Reached max_total_docs ({max_total_docs}), stopping")
                    break
                    
                logger.debug(f"[ROBUST-IRCOT] Iteration {iteration + 1}/{max_iterations}, Query: {current_query[:100]}...")
                
                # Generate worker identification for request isolation
                import threading
                worker_id = f"worker_{threading.current_thread().ident}"
                
                # Retrieve documents with RobustIRCoT parameters
                payload = {
                    "query_text": current_query,
                    "max_hits_count": iterative_retrieval_k,  # 3 docs per iteration
                    "retrieval_method": "retrieve_from_elasticsearch",
                    "corpus_name": self.dataset,
                    "worker_id": worker_id
                }
                
                logger.debug(f"🔍 ROBUST-IRCOT: {worker_id} querying '{current_query[:50]}...' for {iterative_retrieval_k} docs")
                
                # Use semaphore to limit concurrent retrieval requests
                with self.retrieval_semaphore:
                    response = requests.post(
                        "http://localhost:8000/retrieve",
                        json=payload,
                        timeout=60
                    )
                
                if response.status_code == 200:
                    result = response.json()
                    retrieval_results = result.get('retrieval', [])
                    new_docs_found = False
                    iteration_docs = 0
                    
                    for doc in retrieval_results:
                        if len(all_contexts) >= max_total_docs:
                            break
                        title = doc.get('title', '')
                        if title and title not in seen_titles:
                            seen_titles.add(title)
                            all_contexts.append(f"Title: {title}\n{doc.get('text', doc.get('paragraph_text', ''))}")
                            new_docs_found = True
                            iteration_docs += 1
                    
                    if new_docs_found:
                        actual_steps += 1
                        logger.debug(f"[ROBUST-IRCOT] Iteration {iteration + 1}: Found {iteration_docs} new docs, total: {len(all_contexts)}")
                    
                    # If no new documents found, terminate early (same as RobustIRCoT)
                    if not new_docs_found:
                        logger.debug(f"[ROBUST-IRCOT] No new documents found, terminating at iteration {iteration + 1}")
                        break
                    
                    # Update query for next iteration based on gathered information (same as RobustIRCoT)
                    if iteration < max_iterations - 1 and len(all_contexts) < max_total_docs:
                        # Create a follow-up query based on gathered information
                        context_summary = " ".join([ctx[:100] for ctx in all_contexts[-2:]])  # Last 2 contexts
                        current_query = f"{question} Context: {context_summary[:200]}"
                else:
                    logger.warning(f"[ROBUST-IRCOT] Retrieval failed at iteration {iteration + 1}: HTTP {response.status_code}")
                    break
            
            # Combine all contexts (limit to max_total_docs)
            combined_context = "\n\n".join(all_contexts[:max_total_docs])
            logger.debug(f"[ROBUST-IRCOT] Completed: {actual_steps} steps, {len(all_contexts)} documents")
            
            return combined_context, actual_steps
            
        except Exception as e:
            logger.error(f"[ROBUST-IRCOT] Failed: {e}")
            # Fall back to simple iterative retrieval
            return self.get_iterative_retrieval_context(question)

def _is_gemini_model_global(model_name: str) -> bool:
    """Global helper to check if the model is a Gemini model"""
    return 'gemini' in model_name.lower()

def _is_qwen_model_global(model_name: str) -> bool:
    """Global helper to check if the model is a Qwen model"""
    return 'qwen' in model_name.lower() or 'Qwen' in model_name

def parse_port_range(port_range: str) -> List[int]:
    """Parse port range string (e.g., '8010-8019') into list of ports"""
    if '-' in port_range:
        start, end = port_range.split('-')
        return list(range(int(start), int(end) + 1))
    else:
        return [int(port_range)]

def check_llm_server(port: int) -> bool:
    """Check if LLM server is running on given port"""
    try:
        import requests
        # Try health endpoint first
        try:
            response = requests.get(f"http://localhost:{port}/health", timeout=2)
            if response.status_code == 200 and 'detail' not in response.text:
                return True
        except:
            pass
        
        # For Qwen servers, try root endpoint (they respond with server info)
        try:
            response = requests.get(f"http://localhost:{port}/", timeout=2)
            if response.status_code == 200 and ('Qwen' in response.text or 'server' in response.text.lower()):
                return True
        except:
            pass
        
        return False
    except:
        return False

def create_qwen_server_manager(port_range: str, model_name: str = "Qwen/Qwen2.5-3B-Instruct") -> Optional["LLMServerManager"]:
    """
    Create a Qwen server manager for port recovery and load balancing
    
    Args:
        port_range: Port range string (e.g., '8010-8049' for all 40 servers)
        model_name: Qwen model name
        
    Returns:
        LLMServerManager instance or None if creation fails
    """
    try:
        # Parse port range
        ports = parse_port_range(port_range)
        
        # Create temporary config for server manager
        import tempfile
        import json
        
        # Check which ports are actually available
        available_ports = []
        for port in ports:
            if check_llm_server(port):
                available_ports.append(port)
        
        if not available_ports:
            logger.warning(f"No Qwen servers found on ports {ports[0]}-{ports[-1]}")
            return None
        
        # Create server config for available ports
        server_config = {
            "llm_servers": []
        }
        
        for i, port in enumerate(available_ports):
            # Estimate GPU ID (6 servers per GPU typically)
            gpu_id = (port - 8010) // 6 + 1
            
            server_config["llm_servers"].append({
                "id": f"server_{port}",
                "model": model_name,
                "host": "localhost", 
                "port": port,
                "gpu_id": gpu_id,
                "timeout": 120  # 2 minute timeout for benchmark
            })
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(server_config, f, indent=2)
            config_path = f.name
        
        # Initialize server manager
        from scaled_silver_labeling.servers.llm_server_manager import LLMServerManager
        server_manager = LLMServerManager(config_path)
        
        # Clean up temp file
        os.unlink(config_path)
        
        logger.info(f"✅ Created Qwen server manager with {len(available_ports)} servers on ports: {available_ports}")
        return server_manager
        
    except Exception as e:
        logger.error(f"Failed to create Qwen server manager: {e}")
        return None

def initialize_workers(model: str, port_range: Optional[str], num_workers: int) -> Tuple[List[WorkerConfig], Optional["LLMServerManager"]]:
    """Initialize worker configurations and server manager"""
    # Set environment variable to disable caching in parallel mode
    os.environ['ADAPTIVE_RAG_PARALLEL'] = 'true'
    
    workers = []
    server_manager = None
    
    if _is_gemini_model_global(model):
        # For Gemini, create workers without ports
        for i in range(num_workers):
            workers.append(WorkerConfig(worker_id=i, port=None, is_available=True))
        logger.info(f"✅ Initialized {num_workers} Gemini API workers")
        
        # Initialize API call tracking
        from commaqa.models.gemini_generator import GeminiGenerator
        GeminiGenerator.reset_call_stats()
        logger.info("📊 Started API call rate tracking")
        
    elif _is_qwen_model_global(model):
        # For Qwen models, use server manager for port recovery and load balancing
        if not port_range:
            raise ValueError("--port-range required for Qwen models")
            
        logger.info(f"🤖 Initializing Qwen model: {model}")
        server_manager = create_qwen_server_manager(port_range, model)
        
        if not server_manager:
            raise RuntimeError(f"Failed to create Qwen server manager for ports {port_range}")
        
        # Create workers based on available servers
        available_servers = server_manager.list_servers()
        if not available_servers:
            raise RuntimeError(f"No Qwen servers available in range {port_range}")
        
        # Create workers for available servers (limit to num_workers)
        for i, server_info in enumerate(available_servers[:num_workers]):
            worker = WorkerConfig(worker_id=i, port=server_info['port'], is_available=True)
            workers.append(worker)
        
        logger.info(f"✅ Initialized {len(workers)} Qwen workers with server manager on ports: {[w.port for w in workers]}")
        
    else:
        # For other local LLMs (FLAN-T5, etc.), use simple port checking
        if not port_range:
            raise ValueError("--port-range required for non-Gemini models")
            
        ports = parse_port_range(port_range)
        available_ports = []
        
        logger.info(f"🔍 Checking LLM servers on ports {ports[0]}-{ports[-1]}...")
        for port in ports:
            if check_llm_server(port):
                available_ports.append(port)
                logger.info(f"  ✅ Port {port}: Available")
            else:
                logger.info(f"  ❌ Port {port}: Not available")
        
        if not available_ports:
            raise RuntimeError(f"No LLM servers found on ports {ports[0]}-{ports[-1]}")
            
        # Create workers for available ports
        for i, port in enumerate(available_ports[:num_workers]):
            workers.append(WorkerConfig(worker_id=i, port=port, is_available=True))
            
        logger.info(f"✅ Initialized {len(workers)} workers on ports: {[w.port for w in workers]}")
    
    return workers, server_manager

def load_test_queries(dataset_name: str, max_queries: int = None, seed: int = 42) -> List[Dict]:
    """Load test queries from dataset with reproducible sampling"""
    # ONLY use test_experiment.jsonl - this has the proper ground truth format
    test_file = f"processed_data/{dataset_name}/test_experiment.jsonl"
    if not os.path.exists(test_file):
        raise FileNotFoundError(f"test_experiment.jsonl not found for dataset {dataset_name}. "
                              f"Please ensure {test_file} exists with proper ground truth format.")
    
    queries = []
    with open(test_file, 'r') as f:
        for line in f:
            queries.append(json.loads(line))
    
    if max_queries and max_queries < len(queries):
        # Set seed for reproducible sampling
        random.seed(seed)
        queries = random.sample(queries, max_queries)
        
    return queries

def apply_probability_threshold(probabilities: torch.Tensor, threshold: float, stats: Dict[str, int]) -> List[str]:
    """
    Apply probability-based classification with threshold favoring complex systems.
    
    Args:
        probabilities: Tensor of shape (batch_size, 3) with probabilities for [A, B, C]
        threshold: Threshold value (e.g., 0.1 for 10%)
        stats: Dictionary to track upgrade statistics
    
    Returns:
        List of predicted labels after applying threshold logic
    
    Logic:
        - A must beat both B and C by threshold to stay A, otherwise upgrade to highest of B/C
        - B must beat C by threshold to stay B, otherwise upgrade to C
        - C always stays C (highest complexity)
    """
    batch_predictions = []
    label_map = ['A', 'B', 'C']
    
    for prob_row in probabilities:
        prob_A, prob_B, prob_C = prob_row.cpu().numpy()
        stats['total_classifications'] += 1
        
        # Get the class with highest probability
        max_idx = torch.argmax(prob_row).item()
        original_prediction = label_map[max_idx]
        
        final_prediction = original_prediction
        
        if original_prediction == 'A':
            # A must beat both B and C by threshold
            if not (prob_A >= prob_B + threshold and prob_A >= prob_C + threshold):
                # Upgrade to the higher of B or C
                if prob_B >= prob_C:
                    final_prediction = 'B'
                    stats['A_to_B_upgrades'] += 1
                else:
                    final_prediction = 'C'
                    stats['A_to_C_upgrades'] += 1
            else:
                stats['no_upgrades'] += 1
                
        elif original_prediction == 'B':
            # B must beat C by threshold
            if not (prob_B >= prob_C + threshold):
                final_prediction = 'C'
                stats['B_to_C_upgrades'] += 1
            else:
                stats['no_upgrades'] += 1
                
        else:  # original_prediction == 'C'
            # C is already most complex, no upgrade needed
            stats['no_upgrades'] += 1
        
        batch_predictions.append(final_prediction)
    
    return batch_predictions

def load_queries_from_dev_path(dev_path: str, dataset_name: str, max_queries: int = None, seed: int = 42) -> List[Dict]:
    """Load queries from dev_path by filtering dataset instances by IDs"""
    logger = logging.getLogger(__name__)
    
    # Load the valid.json file to get the IDs for the specific dataset
    logger.info(f"Loading IDs from dev path: {dev_path}")
    with open(dev_path, 'r') as f:
        dev_data = json.load(f)
    
    # Filter to get only IDs for the specified dataset
    dataset_ids = set()
    for item in dev_data:
        if item.get('dataset_name') == dataset_name:
            dataset_ids.add(item['id'])
    
    logger.info(f"Found {len(dataset_ids)} {dataset_name} instances in dev path")
    
    # Load the dev_experiment.jsonl file for the dataset
    dev_experiment_file = f"processed_data/{dataset_name}/dev_experiment.jsonl"
    if not os.path.exists(dev_experiment_file):
        raise FileNotFoundError(f"dev_experiment.jsonl not found for dataset {dataset_name}. "
                              f"Please ensure {dev_experiment_file} exists.")
    
    # Filter queries based on the IDs from dev_path
    queries = []
    with open(dev_experiment_file, 'r') as f:
        for line in f:
            query = json.loads(line)
            if query['question_id'] in dataset_ids:
                queries.append(query)
    
    logger.info(f"Loaded {len(queries)} queries for {dataset_name} from dev_experiment.jsonl")
    
    if max_queries and max_queries < len(queries):
        # Set seed for reproducible sampling
        random.seed(seed)
        queries = random.sample(queries, max_queries)
        logger.info(f"Sampled down to {len(queries)} queries")
        
    return queries

def classify_queries(queries: List[Dict], classifier_path: str = None, verify_classification: bool = False, 
                    model: str = None, workers: List[WorkerConfig] = None, force: str = None, 
                    verify_probabilities: float = None, classification_llm: bool = False) -> Dict[str, List[Dict]]:
    """
    Classify queries by complexity or force all queries to use a specific strategy.
    
    Args:
        queries: List of queries to classify
        classifier_path: Path to trained classifier (ignored if force or classification_llm is used)
        verify_classification: Whether to verify predictions (ignored if force or classification_llm is used)
        model: Model for verification and LLM classification
        workers: Workers for verification and LLM classification
        force: Force strategy ('nor', 'oner', 'ircot') to bypass classifier
        verify_probabilities: Threshold percentage for probability-based upgrading 
                              (favors complex systems when probability difference is insufficient)
        classification_llm: Use the generator LLM itself for classification instead of a trained classifier
    """
    logger = logging.getLogger(__name__)
    
    # Initialize classifier metadata for verification
    classifier_metadata = {}
    
    # Validate mutually exclusive verification options
    if verify_classification and verify_probabilities is not None:
        raise ValueError("Cannot use both --verify_classification and --verify_probabilities simultaneously")
    
    if force:
        # Force all queries to use the specified strategy
        force_map = {
            'nor': ('A', 'nor_qa'),
            'oner': ('B', 'oner_qa'), 
            'ircot': ('C', 'ircot_qa')
        }
        
        if force not in force_map:
            raise ValueError(f"Invalid force option: {force}. Must be one of: {list(force_map.keys())}")
        
        complexity_label, system_name = force_map[force]
        
        logger.info(f"🎯 FORCE MODE: All {len(queries)} queries will use {system_name} strategy")
        
        # Create queries_by_system with all queries in the forced system
        queries_by_system = {
            'nor_qa': [],
            'oner_qa': [],
            'ircot_qa': []
        }
        
        # Add all queries to the forced system
        for query in queries:
            query['complexity_prediction'] = complexity_label
            query['system_used'] = system_name
            queries_by_system[system_name].append(query)
        
        # Print distribution
        logger.info("📊 Forced Query Distribution:")
        for system, system_queries in queries_by_system.items():
            if len(system_queries) > 0:
                logger.info(f"  {system}: {len(system_queries)} (100.0%)")
            else:
                logger.info(f"  {system}: 0 (0.0%)")
        
        return queries_by_system
    
    if classification_llm:
        # LLM-based classification mode
        if not model or not workers:
            raise ValueError("model and workers are required when using --classification_llm")
        
        logger.info(f"🤖 LLM CLASSIFICATION MODE: Using {model} for one-shot query classification")
        
        # Create LLM classifier
        llm_classifier = LLMClassifier(model, workers)
        
        # Classify all queries using LLM
        logger.info(f"🧠 Classifying {len(queries)} queries using {model}...")
        predictions = llm_classifier.classify_queries_batch(queries)
        
        # Get classification statistics
        classification_stats = llm_classifier.get_classification_stats()
        logger.info("📊 LLM Classification Results:")
        logger.info(f"  Total classifications: {classification_stats['total_classifications']}")
        logger.info(f"  A (Simple): {classification_stats['A_classifications']}")
        logger.info(f"  B (Medium): {classification_stats['B_classifications']}")
        logger.info(f"  C (Complex): {classification_stats['C_classifications']}")
        logger.info(f"  Failed/Defaulted: {classification_stats['failed_classifications']}")
        
        # Group by system using LLM predictions
        queries_by_system = {
            'nor_qa': [],
            'oner_qa': [],
            'ircot_qa': []
        }
        
        complexity_map = {'A': 'nor_qa', 'B': 'oner_qa', 'C': 'ircot_qa'}
        
        for query, pred in zip(queries, predictions):
            system = complexity_map.get(pred, 'oner_qa')  # Default to medium complexity
            # Add prediction info to query for tracking
            query['complexity_prediction'] = pred
            query['system_used'] = system
            queries_by_system[system].append(query)
        
        # Print distribution
        total = len(queries)
        logger.info("📊 Final Query Distribution (LLM Classification):")
        for system, system_queries in queries_by_system.items():
            percentage = len(system_queries) / total * 100
            logger.info(f"  {system}: {len(system_queries)} ({percentage:.1f}%)")
        
        # Add classification stats to return value
        queries_by_system['_llm_classification_stats'] = classification_stats
        
        return queries_by_system
    
    # Original classifier-based logic
    if not classifier_path:
        raise ValueError("classifier_path is required when not using force or classification_llm mode")
    
    logger.info(f"🧠 Loading classifier from: {classifier_path}")
    
    import torch
    
    # Detect model type from path
    if 'bert' in classifier_path.lower():
        model_type = 'bert'
    else:
        model_type = 't5'
    
    logger.info(f"🔍 DEBUG: Detected model_type = '{model_type}' from path: {classifier_path}")
    
    # Load tokenizer first
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(classifier_path)
    logger.info(f"✅ Loaded tokenizer: {type(tokenizer)}")
    
    if model_type == 't5':
        # T5 models use seq2seq generation for classification - import locally to avoid conflicts
        from transformers import AutoModelForSeq2SeqLM
        logger.info(f"🔍 DEBUG: Loading T5 model with AutoModelForSeq2SeqLM")
        model_cls = AutoModelForSeq2SeqLM.from_pretrained(classifier_path)
        logger.info(f"✅ Loaded T5 seq2seq classifier: {type(model_cls)}")
    else:
        # BERT models use traditional classification heads  
        from transformers import AutoModelForSequenceClassification
        logger.info(f"🔍 DEBUG: Loading BERT model with AutoModelForSequenceClassification")
        model_cls = AutoModelForSequenceClassification.from_pretrained(classifier_path)
        logger.info(f"✅ Loaded BERT classification head: {type(model_cls)}")
    
    # Use GPU device 9 to avoid memory conflicts
    device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
    model_cls = model_cls.to(device)
    model_cls.eval()
    
    # Classify in batches
    batch_size = 32
    predictions = []
    probability_stats = {
        'total_classifications': 0,
        'A_to_B_upgrades': 0,
        'A_to_C_upgrades': 0,
        'B_to_C_upgrades': 0,
        'no_upgrades': 0
    }
    
    for i in tqdm(range(0, len(queries), batch_size), desc="Classifying queries"):
        batch_queries = queries[i:i+batch_size]
        batch_texts = [q.get('question_text', q.get('question', '')) for q in batch_queries]
        
        if model_type == 't5':
            # T5 classification: add prefix and use logits approach (not text parsing!)
            prefixed_texts = [f"classify: {text}" for text in batch_texts]
            inputs = tokenizer(
                prefixed_texts,
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors='pt'
            )
            
            device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            # Generate predictions using logits approach (like training)
            with torch.no_grad():
                outputs = model_cls.generate(
                    input_ids=inputs["input_ids"],
                    attention_mask=inputs["attention_mask"],
                    return_dict_in_generate=True,
                    output_scores=True,  # Critical: get logits instead of text
                    max_length=30,
                    num_beams=1,
                    do_sample=False
                )
                
                # Extract probabilities from logits (like training validation)
                batch_predictions = []
                if outputs.scores:
                    scores = outputs.scores[0]  # First generated token scores
                    
                    # Get A/B/C token IDs
                    a_token_id = tokenizer('A').input_ids[0]
                    b_token_id = tokenizer('B').input_ids[0] 
                    c_token_id = tokenizer('C').input_ids[0]
                    
                    # Extract probabilities for A/B/C tokens
                    probs = torch.nn.functional.softmax(
                        torch.stack([
                            scores[:, a_token_id],
                            scores[:, b_token_id],
                            scores[:, c_token_id],
                        ]), dim=0,
                    ).detach().cpu().numpy()
                    
                    # Get predictions from highest probability
                    preds_labels = np.argmax(probs, 0)
                    label_map = {0: 'A', 1: 'B', 2: 'C'}
                    batch_predictions = [label_map[pred] for pred in preds_labels]
                    
                    # Store probability information for verification (if enabled)
                    if verify_classification:
                        for j, pred in enumerate(batch_predictions):
                            global_idx = i * batch_size + j
                            if global_idx < len(queries):  # Safety check
                                # Store classifier info for this query
                                a_prob, b_prob, c_prob = probs[0][j], probs[1][j], probs[2][j]
                                classifier_metadata[global_idx] = {
                                    'model_architecture': 'FLAN-T5-Large' if 't5-large' in classifier_path.lower() else 'FLAN-T5-Base',
                                    'probabilities': [float(a_prob), float(b_prob), float(c_prob)],
                                    'confidence': float(max([a_prob, b_prob, c_prob]))
                                }
                    
                    # Log first few for debugging
                    if i == 0:  # First batch only
                        logger.info("🔍 T5 Logits-based Classification (first batch):")
                        for j, (pred, text) in enumerate(zip(batch_predictions[:3], batch_texts[:3])):
                            a_prob, b_prob, c_prob = probs[0][j], probs[1][j], probs[2][j]
                            logger.info(f"  Q{j+1}: {pred} (A:{a_prob:.3f}, B:{b_prob:.3f}, C:{c_prob:.3f}) - {text[:50]}{'...' if len(text) > 50 else ''}")
                else:
                    # Fallback if no scores available
                    logger.warning("No scores available from T5 model, defaulting all to 'A'")
                    batch_predictions = ['A'] * len(batch_texts)
                    
                    # Store fallback metadata for verification (if enabled)
                    if verify_classification:
                        for j in range(len(batch_texts)):
                            global_idx = i * batch_size + j
                            if global_idx < len(queries):  # Safety check
                                classifier_metadata[global_idx] = {
                                    'model_architecture': 'FLAN-T5-Large' if 't5-large' in classifier_path.lower() else 'FLAN-T5-Base',
                                    'probabilities': [0.33, 0.33, 0.34],  # Fallback probabilities
                                    'confidence': 0.34
                                }
                    
        else:
            # BERT classification: traditional logits approach
            inputs = tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=384,
                return_tensors='pt'
            )
            
            device = torch.device('cuda:9' if torch.cuda.is_available() else 'cpu')
            inputs = {k: v.to(device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = model_cls(**inputs)
                logits = outputs.logits
                
                # Always calculate probabilities for verification
                probabilities = torch.softmax(logits, dim=-1)
                
                if verify_probabilities is not None:
                    # Use probability-based classification with threshold
                    batch_predictions = apply_probability_threshold(probabilities, verify_probabilities / 100.0, probability_stats)
                else:
                    # Standard argmax classification
                    preds = torch.argmax(logits, dim=-1)
                    # Convert to labels
                    label_map = {0: 'A', 1: 'B', 2: 'C'}
                    batch_predictions = [label_map.get(pred.item(), 'A') for pred in preds]
                
                # Store probability information for verification (if enabled)
                if verify_classification:
                    batch_probs = probabilities.cpu().numpy()
                    for j, (pred, probs) in enumerate(zip(batch_predictions, batch_probs)):
                        global_idx = i * batch_size + j
                        if global_idx < len(queries):  # Safety check
                            # Store classifier info for this query
                            classifier_metadata[global_idx] = {
                                'model_architecture': 'BERT-Large' if 'bert-large' in classifier_path.lower() else 'BERT-Base',
                                'probabilities': [float(probs[0]), float(probs[1]), float(probs[2])],
                                'confidence': float(max(probs))
                            }

        
        predictions.extend(batch_predictions)
    
    # Verification step (if enabled)
    verified_predictions = predictions.copy()
    verifier = None
    verification_stats = None
    
    # Log probability-based statistics if used
    if verify_probabilities is not None:
        total_upgrades = (probability_stats['A_to_B_upgrades'] + 
                         probability_stats['A_to_C_upgrades'] + 
                         probability_stats['B_to_C_upgrades'])
        logger.info("📊 Probability-based Classification Results:")
        logger.info(f"  Threshold used: {verify_probabilities}%")
        logger.info(f"  Total classifications: {probability_stats['total_classifications']}")
        logger.info(f"  A -> B upgrades: {probability_stats['A_to_B_upgrades']}")
        logger.info(f"  A -> C upgrades: {probability_stats['A_to_C_upgrades']}")
        logger.info(f"  B -> C upgrades: {probability_stats['B_to_C_upgrades']}")
        logger.info(f"  No upgrades: {probability_stats['no_upgrades']}")
        logger.info(f"  Total upgrades: {total_upgrades} ({total_upgrades/probability_stats['total_classifications']*100:.1f}%)")
    
    if verify_classification and model and workers:
        logger.info("🔍 Verification enabled - checking classifier predictions with generator LLM...")
        verifier = ClassificationVerifier(model, workers)
        
        # Prepare verification tasks (skip 'C' predictions as they don't need verification)
        verification_tasks = []
        verification_indices = []
        
        for i, (query, initial_pred) in enumerate(zip(queries, predictions)):
            if initial_pred != 'C':  # Only verify A and B predictions
                question_text = query.get('question_text', query.get('question', ''))
                verification_tasks.append((question_text, initial_pred, i))  # Include query index
                verification_indices.append(i)
        
        logger.info(f"📊 Verifying {len(verification_tasks)} out of {len(queries)} predictions in parallel...")
        
        # Process verifications in parallel
        def verify_single(task_data):
            question_text, initial_pred, query_idx = task_data
            # Get classifier metadata for this query
            metadata = classifier_metadata.get(query_idx, None)
            return verifier.verify_classification(question_text, initial_pred, metadata)
        
        # Use available workers for parallel verification
        max_workers = min(len(workers), len(verification_tasks)) if verification_tasks else 1
        verification_results = [None] * len(verification_tasks)
        
        if verification_tasks:
            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all verification tasks
                future_to_index = {
                    executor.submit(verify_single, task): i 
                    for i, task in enumerate(verification_tasks)
                }
                
                # Collect results with progress bar
                with tqdm(total=len(verification_tasks), desc="Verifying classifications") as pbar:
                    for future in concurrent.futures.as_completed(future_to_index):
                        task_index = future_to_index[future]
                        try:
                            result = future.result(timeout=30)  # 30 second timeout per verification (3 attempts * 15s = 45s max)
                            verification_results[task_index] = result
                        except concurrent.futures.TimeoutError:
                            logger.warning(f"Verification timeout for task {task_index} - discarding sample after 30s")
                            # Keep original prediction on timeout
                            verification_results[task_index] = verification_tasks[task_index][1]
                        except Exception as e:
                            logger.warning(f"Verification error for task {task_index}: {e}")
                            # Keep original prediction on error
                            verification_results[task_index] = verification_tasks[task_index][1]
                        pbar.update(1)
        
        # Apply verification results back to the original predictions
        for i, verified_result in enumerate(verification_results):
            original_index = verification_indices[i]
            verified_predictions[original_index] = verified_result
        
        # Log verification statistics
        verification_stats = verifier.get_verification_stats()
        logger.info("📊 Enhanced Verification Results:")
        logger.info(f"  Total verifications: {verification_stats['total_verifications']}")
        logger.info(f"  A -> B upgrades: {verification_stats['A_to_B_upgrades']}")
        logger.info(f"  A -> C upgrades: {verification_stats['A_to_C_upgrades']}")
        logger.info(f"  B -> C upgrades: {verification_stats['B_to_C_upgrades']}")
        logger.info(f"  A kept: {verification_stats['A_kept']}")
        logger.info(f"  B kept: {verification_stats['B_kept']}")
        logger.info(f"  C (no verification): {verification_stats['C_no_verification']}")
    elif verify_classification:
        logger.warning("⚠️  Verification requested but model/workers not provided - skipping verification")
    
    # Group by system using verified predictions
    queries_by_system = {
        'nor_qa': [],
        'oner_qa': [],
        'ircot_qa': []
    }
    
    complexity_map = {'A': 'nor_qa', 'B': 'oner_qa', 'C': 'ircot_qa'}
    
    for query, pred in zip(queries, verified_predictions):
        system = complexity_map.get(pred, 'nor_qa')
        # Add prediction info to query for tracking
        query['complexity_prediction'] = pred
        query['system_used'] = system
        queries_by_system[system].append(query)
    
    # Print distribution
    total = len(queries)
    logger.info("📊 Final Query Distribution:")
    for system, system_queries in queries_by_system.items():
        percentage = len(system_queries) / total * 100
        logger.info(f"  {system}: {len(system_queries)} ({percentage:.1f}%)")
    
    # Add verification stats to return value if verification was used
    if verification_stats:
        queries_by_system['_verification_stats'] = verification_stats
    
    # Add probability statistics to return value if probability verification was used
    if verify_probabilities is not None:
        queries_by_system['_probability_stats'] = probability_stats
    
    return queries_by_system

def run_parallel_adaptive_rag(
    queries_by_system: Dict[str, List[Dict]], 
    model: str,
    dataset: str,
    workers: List[WorkerConfig],
    output_dir: str,
    all_queries: List[Dict] = None,
    server_manager: Optional["LLMServerManager"] = None
) -> Dict[str, Any]:
    """Run adaptive RAG with parallel processing"""
    
    all_predictions = {}
    system_times = {}
    
    # Create ground truth and question mapping
    ground_truth_map = {}
    question_map = {}
    if all_queries:
        for query in all_queries:
            qid = query.get('question_id', query.get('id', ''))
            question_text = query.get('question_text', query.get('question', ''))
            answers = []
            
            # Extract answers based on dataset format
            if 'answers_objects' in query and query['answers_objects']:
                for ans_obj in query['answers_objects']:
                    if 'spans' in ans_obj:
                        answers.extend(ans_obj['spans'])
            elif 'answers' in query:
                if isinstance(query['answers'], list):
                    answers = query['answers']
                else:
                    answers = [query['answers']]
            elif 'answer' in query:
                answers = [query['answer']]
            
            # Clean up answers
            clean_answers = []
            for ans in answers:
                if isinstance(ans, str):
                    clean_ans = ans.strip('"').strip("'").strip()
                    clean_answers.append(clean_ans)
                else:
                    clean_answers.append(str(ans))
            
            ground_truth_map[qid] = clean_answers
            question_map[qid] = question_text
    
    # Process each system in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_system = {}
        
        for system, queries in queries_by_system.items():
            if queries:
                processor = ParallelQueryProcessor(model, dataset, system, workers, server_manager)
                future = executor.submit(processor.process_queries_parallel, queries)
                future_to_system[future] = (system, processor, len(queries))
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_system):
            system, processor, num_queries = future_to_system[future]
            start_time = time.time()
            
            try:
                predictions = future.result()
                elapsed = time.time() - start_time
                system_times[system] = elapsed
                
                # Add system info to predictions
                for qid, pred_data in predictions.items():
                    # pred_data is now {'answer': str, 'steps': int}
                    answer = pred_data.get('answer', '') if isinstance(pred_data, dict) else pred_data
                    steps = pred_data.get('steps', 0) if isinstance(pred_data, dict) else 0
                    
                    # Get ground truth and question for this query
                    ground_truth = ground_truth_map.get(qid, [])
                    question_text = question_map.get(qid, '')
                    
                    all_predictions[qid] = {
                        'question': question_text,
                        'prediction': answer,
                        'ground_truth': ground_truth,
                        'steps': steps,
                        'system_used': system,
                        'complexity': {'nor_qa': 'A', 'oner_qa': 'B', 'ircot_qa': 'C'}[system]
                    }
                
                logger.info(f"✅ {system} completed: {num_queries} queries in {elapsed:.1f}s ({elapsed/num_queries:.1f}s per query)")
                
            except Exception as e:
                logger.error(f"❌ {system} failed: {e}")
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, f"{dataset}_predictions.json")
    
    with open(output_file, 'w') as f:
        json.dump(all_predictions, f, indent=2)
    
    return all_predictions

def save_benchmark_results(evaluation_results: Dict[str, Any], dataset_name: str, output_dir: str):
    """Save benchmark evaluation results to a JSON file for each dataset"""
    import json
    import os
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create the benchmark results file path
    benchmark_file = os.path.join(output_dir, f"{dataset_name}_benchmark_results.json")
    
    # Prepare the benchmark results data
    benchmark_data = {
        'dataset': dataset_name,
        'evaluation_metrics': {
            'f1_score': evaluation_results.get('f1_score', 0.0),
            'exact_match': evaluation_results.get('exact_match', 0.0),
            'accuracy': evaluation_results.get('accuracy', 0.0),
            'accuracy_type': evaluation_results.get('accuracy_type', 'Unknown'),
            'avg_steps': evaluation_results.get('avg_steps', 0.0)
        },
        'runtime_metrics': evaluation_results.get('runtime_metrics', {}),
        'verification_stats': evaluation_results.get('verification_stats', None),
        'query_distribution': evaluation_results.get('query_distribution', {}),
        'per_query_details': evaluation_results.get('per_query_details', []),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())
    }
    
    # Save the benchmark results
    with open(benchmark_file, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    
    print(f"Benchmark results saved to: {benchmark_file}")
    return benchmark_file

def normalize_answer(s: str) -> str:
    """Normalize answer string for evaluation"""
    import string
    import re
    
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)
    
    def white_space_fix(text):
        return ' '.join(text.split())
    
    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)
    
    def lower(text):
        return text.lower()
    
    return white_space_fix(remove_articles(remove_punc(lower(s))))

def extract_answer_from_prediction(prediction: str) -> str:
    """Extract answer from model prediction"""
    import re
    
    if not prediction:
        return ""
    
    # Look for "Answer is:" pattern
    if "Answer is:" in prediction:
        answer = prediction.split("Answer is:")[-1].strip()
        # Remove trailing periods and clean up
        answer = re.sub(r'\.$', '', answer).strip()
        return answer
    
    # Look for other answer patterns
    patterns = [
        r'[Tt]he answer is:?\s*(.+?)(?:\.|$)',
        r'[Aa]nswer:?\s*(.+?)(?:\.|$)', 
        r'Therefore,?\s*(.+?)(?:\.|$)',
        r'Based on.*?,\s*(.+?)(?:\.|$)',
        r'\*\*(.+?)\*\*',  # Bold text often contains answers
        r'Answer is:\s*\*\*(.+?)\*\*',  # Answer is: **answer**
    ]
    
    for pattern in patterns:
        match = re.search(pattern, prediction, re.IGNORECASE | re.DOTALL)
        if match:
            answer = match.group(1).strip()
            # Clean up the answer
            answer = re.sub(r'^[*\-\s]+', '', answer)  # Remove leading asterisks, dashes, spaces
            answer = re.sub(r'[*\-\s]+$', '', answer)  # Remove trailing asterisks, dashes, spaces
            answer = re.sub(r'\.$', '', answer).strip()  # Remove trailing period
            if answer and len(answer) < 100:  # Reasonable answer length
                return answer
    
    # If no patterns match, try to extract the last meaningful sentence
    sentences = prediction.strip().split('.')
    for sentence in reversed(sentences):
        sentence = sentence.strip()
        if sentence and len(sentence) < 100 and not sentence.startswith(('Based on', 'Therefore', 'The provided')):
            return sentence
    
    # Fallback: use first 50 characters
    return prediction.strip()[:50]

def calculate_f1_score(prediction: str, ground_truths: List[str]) -> float:
    """Calculate F1 score between prediction and ground truth answers"""
    prediction_tokens = normalize_answer(prediction).split()
    if not prediction_tokens:
        return 0.0
    
    best_f1 = 0.0
    for gold_answer in ground_truths:
        gold_tokens = normalize_answer(gold_answer).split()
        if not gold_tokens:
            continue
            
        # Calculate precision, recall, and F1
        common_tokens = set(prediction_tokens) & set(gold_tokens)
        if len(common_tokens) == 0:
            f1 = 0.0
        else:
            precision = len(common_tokens) / len(prediction_tokens)
            recall = len(common_tokens) / len(gold_tokens)
            f1 = 2 * precision * recall / (precision + recall)
        
        best_f1 = max(best_f1, f1)
    
    return best_f1

def calculate_exact_match(prediction: str, ground_truths: List[str]) -> bool:
    """Calculate exact match between prediction and ground truth answers"""
    pred_norm = normalize_answer(prediction)
    for gold_answer in ground_truths:
        gold_norm = normalize_answer(gold_answer)
        if pred_norm == gold_norm:
            return True
    return False

@retry_with_backoff(max_retries=2, base_delay=1.0, max_delay=10.0)
def calculate_llm_accuracy(prediction: str, ground_truths: List[str], question: str = "", model_name: str = "gemini-2.5-flash-lite") -> bool:
    """
    Check if prediction is correct using LLM synthetic check with question context.
    This provides better semantic alignment by including the original question.
    
    Args:
        prediction: The predicted answer
        ground_truths: List of acceptable ground truth answers
        question: The original question for context
        model_name: LLM model to use for checking (default: gemini-2.5-flash-lite)
    
    Returns:
        bool: True if LLM determines the answer is correct
    """
    import google.generativeai as genai
    
    # Configure Gemini API
    genai.configure(api_key=os.environ.get('GOOGLE_API_KEY'))
    model = genai.GenerativeModel(model_name)
    
    # Create improved prompt with question context
    question_context = f"Question: {question}\n\n" if question else ""
    
    # For multi-answer datasets, find the best matching ground truth
    best_ground_truth = ""
    if len(ground_truths) > 1:
        best_ground_truth = f"""
Note: Multiple valid answers exist for this question. Consider if the prediction matches ANY of these acceptable answers:
{', '.join([f'"{gt}"' for gt in ground_truths])}"""
    
    prompt = f"""You are evaluating whether a predicted answer is semantically equivalent to the ground truth answer(s) in the context of the original question.

{question_context}Ground Truth Answer(s): {', '.join(ground_truths)}
Predicted Answer: {prediction}{best_ground_truth}

Determine if the predicted answer is semantically equivalent to any of the ground truth answers in the context of the question.
Consider the answer correct if:
1. It answers the question correctly with the same meaning as the ground truth
2. It contains the key information from any ground truth answer
3. It is semantically equivalent even if worded differently (e.g., "USA" vs "United States")
4. Minor formatting, punctuation, or abbreviation differences should be ignored
5. For questions with multiple valid answers, the prediction matches ANY of the ground truths

Examples of correct matches:
- "Sea, Air, and Land" ≈ "Sea, Air, Land" (punctuation difference)
- "United States" ≈ "USA" ≈ "America" (semantic equivalence)
- "18th century" ≈ "18th" (context makes meaning clear)

Respond with ONLY 'YES' if the answer is correct, or 'NO' if it is incorrect.

Is the predicted answer correct?"""

    try:
        response = model.generate_content(prompt)
        result = response.text.strip().upper()
        
        # Check if response is YES
        return result == "YES" or result.startswith("YES")
        
    except Exception as e:
        logger.warning(f"LLM accuracy check failed: {e}. Falling back to contains check.")
        # Fallback to contains check if LLM fails
        return calculate_accuracy_contains(prediction, ground_truths)

def calculate_accuracy_contains(prediction: str, ground_truths: List[str]) -> bool:
    """Calculate if prediction contains the ground truth answer"""
    pred_norm = normalize_answer(prediction).lower()
    for gold_answer in ground_truths:
        gold_norm = normalize_answer(gold_answer).lower()
        if gold_norm in pred_norm:
            return True
    return False

def batch_calculate_llm_accuracy(prediction_question_answer_tuples: List[Tuple[str, str, List[str]]], workers: List[WorkerConfig]) -> List[bool]:
    """Calculate LLM accuracy for multiple prediction-question-answer tuples in parallel with question context"""
    if not workers or not prediction_question_answer_tuples:
        return [False] * len(prediction_question_answer_tuples)
    
    def calculate_single_accuracy(tuple_data):
        prediction, question, gold_answers = tuple_data
        return calculate_llm_accuracy(prediction, gold_answers, question)
    
    # Use ThreadPoolExecutor with the same number of workers
    max_workers = min(len(workers), len(prediction_question_answer_tuples))
    results = [False] * len(prediction_question_answer_tuples)
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit all tasks
        future_to_index = {
            executor.submit(calculate_single_accuracy, tuple_data): i 
            for i, tuple_data in enumerate(prediction_question_answer_tuples)
        }
        
        # Collect results with progress bar and timeout handling
        try:
            for future in tqdm(concurrent.futures.as_completed(future_to_index, timeout=300), 
                              total=len(future_to_index), 
                              desc="LLM accuracy checks", 
                              ncols=80):
                index = future_to_index[future]
                try:
                    results[index] = future.result(timeout=60)  # 1 minute timeout per accuracy check
                except concurrent.futures.TimeoutError:
                    logger.warning(f"LLM accuracy check timed out for item {index}")
                    results[index] = False
                except Exception as e:
                    logger.warning(f"LLM accuracy check failed for item {index}: {e}")
                    results[index] = False
        except concurrent.futures.TimeoutError:
            logger.error("Overall timeout reached for LLM accuracy checks. Cancelling remaining futures.")
            # Cancel remaining futures
            for future, index in future_to_index.items():
                if not future.done():
                    future.cancel()
                    if results[index] is None:  # Only set if not already processed
                        results[index] = False
    
    return results

def evaluate_predictions(predictions: Dict[str, Any], queries: List[Dict], use_llm_accuracy: bool = True, workers: List[WorkerConfig] = None) -> Dict[str, float]:
    """Evaluate predictions against ground truth answers using F1, EM, Accuracy, and Steps metrics
    
    Args:
        predictions: Dictionary of predictions with steps
        queries: List of queries with ground truth
        use_llm_accuracy: Whether to use LLM-based accuracy check (default: True)
        workers: List of worker configurations for parallel LLM accuracy checks
    
    Returns:
        Dictionary with evaluation metrics
    """
    print(f"Preparing evaluation data...")
    
    # Create mappings of query ID to ground truth answers and questions
    ground_truth = {}
    question_map = {}
    for query in tqdm(queries, desc="Processing ground truth", ncols=80):
        qid = query.get('question_id', query.get('id', ''))
        question_text = query.get('question_text', query.get('question', ''))
        question_map[qid] = question_text
        answers = []
        
        # Extract answers based on dataset format
        if 'answers_objects' in query and query['answers_objects']:
            for ans_obj in query['answers_objects']:
                if 'spans' in ans_obj:
                    answers.extend(ans_obj['spans'])
        elif 'answers' in query:
            if isinstance(query['answers'], list):
                answers = query['answers']
            else:
                answers = [query['answers']]
        elif 'answer' in query:
            answers = [query['answer']]
        
        # Clean up answers
        clean_answers = []
        for ans in answers:
            if isinstance(ans, str):
                # Remove quotes from answers like '"Loch Lomond"'
                clean_ans = ans.strip('"').strip("'").strip()
                clean_answers.append(clean_ans)
            else:
                clean_answers.append(str(ans))
        
        ground_truth[qid] = clean_answers
    
    # Initialize counters (overall and per-system)
    total = 0
    f1_scores = []
    exact_matches = 0
    llm_accuracy = 0
    total_steps = 0
    
    # Per-system metrics
    per_system_metrics = {}
    
    print(f"Evaluating {len(predictions)} predictions...")
    
    # Process each prediction with progress bar
    predictions_items = list(predictions.items())
    
    # Prepare data for batch processing
    valid_predictions = []
    prediction_question_answer_tuples = []
    failed_samples = []  # Track failed samples
    
    for qid, pred_data in predictions_items:
        if qid not in ground_truth:
            continue
        
        prediction = extract_answer_from_prediction(pred_data.get('prediction', ''))
        gold_answers = ground_truth[qid]
        question = question_map.get(qid, '')
        steps = pred_data.get('steps', 0)
        system_used = pred_data.get('system_used', 'unknown')
        
        # Check if this sample failed (API failure or resource exhaustion)
        is_failed = (
            pred_data.get('discarded', False) or 
            pred_data.get('resource_exhausted', False) or
            prediction.strip() in ['__API_FAILED__', '__RESOURCE_EXHAUSTED__'] or
            '__API_FAILED__' in prediction or
            '__RESOURCE_EXHAUSTED__' in prediction
        )
        
        if is_failed:
            failed_samples.append({
                'question_id': qid,
                'question': question,
                'error_type': pred_data.get('error', 'API failure'),
                'system_used': system_used
            })
            print(f"WARNING: Excluding failed sample {qid}: {pred_data.get('error', 'API failure')}")
            continue  # Skip failed samples from evaluation
        
        valid_predictions.append((qid, prediction, gold_answers, steps, system_used))
        if use_llm_accuracy:
            prediction_question_answer_tuples.append((prediction, question, gold_answers))
    
    # Batch calculate LLM accuracy if needed
    llm_accuracy_results = []
    if use_llm_accuracy and workers:
        print(f"Running parallel LLM accuracy checks with {len(workers)} workers...")
        llm_accuracy_results = batch_calculate_llm_accuracy(prediction_question_answer_tuples, workers)
    elif use_llm_accuracy:
        print(f"WARNING: No workers provided - falling back to sequential LLM accuracy checks")
        llm_accuracy_results = [calculate_llm_accuracy(pred, ans, question) for pred, question, ans in 
                               tqdm(prediction_question_answer_tuples, desc="LLM accuracy checks", ncols=80)]
    
    # Process all predictions with pre-computed accuracy results
    for i, (qid, prediction, gold_answers, steps, system_used) in enumerate(
        tqdm(valid_predictions, desc="Computing metrics", ncols=80)):
        
        total += 1
        
        # Initialize system metrics if not seen before
        if system_used not in per_system_metrics:
            per_system_metrics[system_used] = {
                'f1_scores': [],
                'exact_matches': 0,
                'llm_accuracy': 0,
                'total_steps': 0,
                'count': 0
            }
        
        # Calculate metrics
        f1 = calculate_f1_score(prediction, gold_answers)
        em = calculate_exact_match(prediction, gold_answers)
        
        # Use pre-computed LLM accuracy result
        acc = False
        if use_llm_accuracy and i < len(llm_accuracy_results):
            acc = llm_accuracy_results[i]
            if acc:
                llm_accuracy += 1
                per_system_metrics[system_used]['llm_accuracy'] += 1
        
        # Update overall metrics
        f1_scores.append(f1)
        if em:
            exact_matches += 1
            per_system_metrics[system_used]['exact_matches'] += 1
        total_steps += steps
        
        # Update per-system metrics
        per_system_metrics[system_used]['f1_scores'].append(f1)
        per_system_metrics[system_used]['total_steps'] += steps
        per_system_metrics[system_used]['count'] += 1
    
    # Calculate final metrics
    if total == 0:
        return {
            'f1_score': 0.0, 
            'exact_match': 0.0, 
            'accuracy': 0.0,
            'avg_steps': 0.0,
            'accuracy_type': 'LLM',
            'per_system': {},
            'failed_samples': failed_samples,
            'total_processed': len(predictions),
            'failed_count': len(failed_samples),
            'valid_count': 0
        }
    
    # Calculate per-system final metrics
    per_system_final = {}
    for system, metrics in per_system_metrics.items():
        count = metrics['count']
        if count > 0:
            per_system_final[system] = {
                'f1_score': sum(metrics['f1_scores']) / len(metrics['f1_scores']) if metrics['f1_scores'] else 0.0,
                'exact_match': metrics['exact_matches'] / count,
                'accuracy': metrics['llm_accuracy'] / count,
                'avg_steps': metrics['total_steps'] / count,
                'count': count
            }
    
    overall_metrics = {
        'f1_score': sum(f1_scores) / len(f1_scores) if f1_scores else 0.0,
        'exact_match': exact_matches / total,
        'accuracy': llm_accuracy / total,
        'avg_steps': total_steps / total,
        'accuracy_type': 'LLM',
        'per_system': per_system_final,
        'failed_samples': failed_samples,
        'total_processed': len(predictions),
        'failed_count': len(failed_samples),
        'valid_count': total
    }
    
    return overall_metrics

def main():
    parser = argparse.ArgumentParser(description="Parallel Adaptive RAG with multi-worker support")
    
    # Core arguments
    parser.add_argument("--classifier_path", type=str, required=False,
                       help="Path to trained query complexity classifier (required unless --force is used)")
    parser.add_argument("--model", type=str, required=True,
                       help="LLM model name (e.g., gemini, flan-t5-xl)")
    parser.add_argument("--dataset", type=str, required=False,
                       help="Dataset name (e.g., hotpotqa). Not required if --all is used.")
    parser.add_argument("--all", action="store_true",
                       help="Run on all 6 datasets: hotpotqa, musique, 2wikimultihopqa, trivia, nq, squad")
    parser.add_argument("--output_dir", type=str, default="predictions/parallel_adaptive_rag",
                       help="Output directory for predictions")
    parser.add_argument("--experiment_name", type=str, required=False,
                       help="Experiment name for organizing results (e.g., 500samples_optimized)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducible sampling (default: 42)")
    
    # Force strategy parameter (bypasses classifier)
    parser.add_argument("--force", type=str, choices=['nor', 'oner', 'ircot'], required=False,
                       help="Force all queries to use a specific strategy: 'nor' (no retrieval), 'oner' (one retrieval), or 'ircot' (iterative reasoning)")
    
    # LLM-based classification parameter
    parser.add_argument("--classification_llm", action="store_true",
                       help="Use the generator LLM itself for one-shot text classification instead of a trained classifier model")
    
    # Parallel processing arguments
    parser.add_argument("--workers", type=int, default=10,
                       help="Number of parallel workers (default: 10)")
    parser.add_argument("--port-range", type=str, default=None,
                       help="Port range for LLM servers (e.g., 8010-8019). Required for non-Gemini models")
    parser.add_argument("--max_queries", type=int, default=500,
                       help="Maximum queries to process (default: 500)")
    
    # Verification arguments
    parser.add_argument("--verify_classification", action="store_true",
                       help="Enable classification verification using generator LLM (default: False)")
    parser.add_argument("--verify_probabilities", type=float, default=None,
                       help="Enable probability-based classification with threshold percentage (e.g., 10 for 10%). "
                            "Favors complex systems when probability difference is insufficient. "
                            "Cannot be used with --verify_classification.")
    parser.add_argument("--verbose", action="store_true",
                       help="Enable verbose logging to console (default: False, logs saved to files)")
    
    # Development/Test data arguments
    parser.add_argument("--dev_path", type=str, required=False,
                       help="Path to dev valid.json file for filtering dataset instances by IDs")
    parser.add_argument("--test", action="store_true",
                       help="Use test data instead of dev data (alternative to --dev_path)")
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.all and not args.dataset:
        parser.error("Either --dataset or --all must be specified")
    
    if args.all and args.dataset:
        parser.error("Cannot specify both --dataset and --all")
    
    # Validate dev_path/test arguments
    if args.dev_path and args.test:
        parser.error("Cannot specify both --dev_path and --test")
    
    if args.dev_path and not os.path.exists(args.dev_path):
        parser.error(f"Dev path file does not exist: {args.dev_path}")
    
    if args.dev_path and not args.dataset:
        parser.error("--dataset must be specified when using --dev_path")
    
    # Validate force vs classifier_path vs classification_llm
    classification_modes = sum([bool(args.force), bool(args.classifier_path), bool(args.classification_llm)])
    if classification_modes == 0:
        parser.error("One of --classifier_path, --force, or --classification_llm must be specified")
    elif classification_modes > 1:
        parser.error("Only one of --classifier_path, --force, or --classification_llm can be specified")
    
    if args.verify_classification and args.force:
        parser.error("Cannot use --verify_classification with --force (no classification to verify)")
    
    if args.verify_classification and args.classification_llm:
        parser.error("Cannot use --verify_classification with --classification_llm (LLM classification is already self-verified)")
    
    if args.verify_probabilities is not None and args.force:
        parser.error("Cannot use --verify_probabilities with --force (no classification to modify)")
    
    if args.verify_probabilities is not None and args.classification_llm:
        parser.error("Cannot use --verify_probabilities with --classification_llm (LLM classification doesn't use probabilities)")
    
    if args.verify_classification and args.verify_probabilities is not None:
        parser.error("Cannot use both --verify_classification and --verify_probabilities simultaneously")
    
    if args.verify_probabilities is not None and (args.verify_probabilities < 0 or args.verify_probabilities > 100):
        parser.error("--verify_probabilities must be between 0 and 100")
    
    # Define all datasets
    ALL_DATASETS = ['hotpotqa', 'musique', '2wikimultihopqa', 'trivia', 'nq', 'squad']
    datasets_to_run = ALL_DATASETS if args.all else [args.dataset]
    
    print(f"Parallel Adaptive RAG Pipeline")
    print(f"Dataset(s): {', '.join(datasets_to_run)}")
    print(f"🤖 Model: {args.model}")
    if args.experiment_name:
        print(f"🧪 Experiment: {args.experiment_name}")
    if args.force:
        strategy_map = {'nor': 'No Retrieval', 'oner': 'One Retrieval', 'ircot': 'IRCoT'}
        print(f"Forced Strategy: {strategy_map[args.force]} (bypassing classifier)")
    elif args.classification_llm:
        print(f"🤖 LLM Classifier: {args.model} (one-shot classification)")
    else:
        print(f"🧠 Classifier: {args.classifier_path}")
    print(f"⚡ Workers: {args.workers}")
    if args.port_range:
        print(f"Port range: {args.port_range}")
    print(f"Max queries: {args.max_queries}")
    if args.dev_path:
        print(f"Dev path: {args.dev_path} (filtering instances)")
    elif args.test:
        print(f"🧪 Data mode: Test data")
    else:
        print(f"🧪 Data mode: Test data (default)")
    if not args.force and not args.classification_llm:
        if args.verify_classification:
            print(f"Verification: LLM-based verification enabled")
        elif args.verify_probabilities is not None:
            print(f"Verification: Probability-based verification enabled (threshold: {args.verify_probabilities}%)")
        else:
            print(f"Verification: Disabled")
    elif args.classification_llm:
        print(f"Verification: LLM classification (self-verified)")
    print(f"🎲 Seed: {args.seed}")
    
    # Initialize workers and server manager
    workers, server_manager = initialize_workers(args.model, args.port_range, args.workers)
    
    # Run for each dataset
    all_results = {}
    total_start_time = time.time()
    
    for dataset_idx, dataset_name in enumerate(datasets_to_run):
        print(f"\n{'='*60}")
        print(f"Processing Dataset {dataset_idx + 1}/{len(datasets_to_run)}: {dataset_name}")
        print(f"{'='*60}")
        
        # Load queries
        dataset_start_time = time.time()
        print(f"\nLoading queries...")
        
        if args.dev_path:
            # Use dev_path to filter queries
            queries = load_queries_from_dev_path(args.dev_path, dataset_name, args.max_queries, args.seed)
            print(f"Using dev_path filtering: {args.dev_path}")
        elif args.test:
            # Use test data (same as current default behavior)
            queries = load_test_queries(dataset_name, args.max_queries, args.seed)
            print(f"Using test data")
        else:
            # Default behavior: use test data
            queries = load_test_queries(dataset_name, args.max_queries, args.seed)
            print(f"Using test data (default)")
        
        load_time = time.time() - dataset_start_time
        print(f"Loaded {len(queries)} queries ({load_time:.1f}s)")
        
        # Classify queries
        classify_start_time = time.time()
        if args.classification_llm:
            print(f"\n🤖 Classifying queries using LLM ({args.model})...")
        else:
            print(f"\n🧠 Classifying queries...")
            if args.verify_classification:
                print(f"LLM-based verification enabled - this may take additional time...")
            elif args.verify_probabilities is not None:
                print(f"Probability-based verification enabled (threshold: {args.verify_probabilities}%)")
        
        queries_by_system = classify_queries(
            queries, 
            args.classifier_path, 
            verify_classification=args.verify_classification,
            model=args.model if (args.verify_classification or args.classification_llm) else None,
            workers=workers if (args.verify_classification or args.classification_llm) else None,
            force=args.force,
            verify_probabilities=args.verify_probabilities,
            classification_llm=args.classification_llm
        )
        
        classify_time = time.time() - classify_start_time
        print(f"Classification completed ({classify_time:.1f}s)")
        
        print(f"\nQuery distribution:")
        verification_stats = queries_by_system.pop('_verification_stats', None)
        probability_stats = queries_by_system.pop('_probability_stats', None)
        llm_classification_stats = queries_by_system.pop('_llm_classification_stats', None)
        for system, system_queries in queries_by_system.items():
            if system not in ['_verification_stats', '_probability_stats']:  # Skip stats
                print(f"  {system}: {len(system_queries)} queries")
        
        # Create experiment and dataset-specific output directory
        if args.experiment_name:
            dataset_output_dir = f"{args.output_dir}/{args.experiment_name}/{dataset_name}"
        else:
            dataset_output_dir = f"{args.output_dir}/{dataset_name}"
        
        # Setup experiment-specific logging for this dataset
        if args.experiment_name:
            experiment_log_dir = f"{args.output_dir}/{args.experiment_name}"
        else:
            experiment_log_dir = args.output_dir
        
        log_file = setup_experiment_logging(experiment_log_dir, verbose=args.verbose)
        if not args.verbose:
            print(f"Detailed logs saved to: {log_file}")
            print(f"Use --verbose flag to see detailed output in console")
        
        # Run parallel adaptive RAG
        print(f"\n🏃 Running parallel adaptive RAG...")
        start_time = time.time()
        
        predictions = run_parallel_adaptive_rag(
            queries_by_system,
            args.model,
            dataset_name,
            workers,
            dataset_output_dir,
            queries,  # Pass all queries to include ground truth
            server_manager  # Pass server manager for Qwen models
        )
        
        elapsed = time.time() - start_time
        
        # Evaluate predictions
        eval_start_time = time.time()
        print(f"\nEvaluating predictions...")
        evaluation_results = evaluate_predictions(predictions, queries, use_llm_accuracy=True, workers=workers)
        eval_time = time.time() - eval_start_time
        print(f"Evaluation completed ({eval_time:.1f}s)")
        
        # Add verification statistics to evaluation results if available
        if verification_stats:
            evaluation_results['verification_stats'] = verification_stats
        
        # Add runtime and efficiency metrics to evaluation results
        queries_per_second = len(predictions) / elapsed
        evaluation_results['runtime_metrics'] = {
            'total_time_seconds': elapsed,
            'total_time_minutes': elapsed / 60,
            'queries_processed': len(predictions),
            'queries_per_second': queries_per_second,
            'time_per_query_seconds': elapsed / len(predictions) if len(predictions) > 0 else 0,
            'workers_used': args.workers,
            'verification_enabled': args.verify_classification,
            'dataset_name': dataset_name,
            'seed_used': args.seed
        }
        
        # Store results for this dataset
        all_results[dataset_name] = evaluation_results
        
        # Save benchmark results for this dataset
        save_benchmark_results(evaluation_results, dataset_name, dataset_output_dir)
        
        # Calculate dataset completion time
        dataset_total_time = time.time() - dataset_start_time
        
        # Print individual dataset summary
        print(f"\nResults for {dataset_name}:")
        print(f"  F1 Score:     {evaluation_results['f1_score']:.3f}")
        print(f"  Exact Match:  {evaluation_results['exact_match']:.3f}")
        accuracy_type = evaluation_results.get('accuracy_type', 'Unknown')
        print(f"  Accuracy ({accuracy_type}): {evaluation_results['accuracy']:.3f}")
        print(f"  Avg Steps:    {evaluation_results['avg_steps']:.1f}")
        print(f"  Performance:  {evaluation_results['runtime_metrics']['queries_per_second']:.1f} queries/second")
        print(f"  Processing:   {evaluation_results['runtime_metrics']['total_time_seconds']:.1f}s")
        print(f"  Total time:   {dataset_total_time:.1f}s")
        
        # Print failed samples information
        failed_count = evaluation_results.get('failed_count', 0)
        total_processed = evaluation_results.get('total_processed', 0)
        valid_count = evaluation_results.get('valid_count', 0)
        
        if failed_count > 0:
            print(f"  WARNING: Failed samples: {failed_count}/{total_processed} ({failed_count/total_processed*100:.1f}%)")
            print(f"  Valid samples:  {valid_count}/{total_processed} ({valid_count/total_processed*100:.1f}%)")
            
            # List the first few failed samples
            failed_samples = evaluation_results.get('failed_samples', [])
            if failed_samples:
                print(f"  🚨 Failed examples:")
                for i, sample in enumerate(failed_samples[:3]):  # Show first 3
                    print(f"    {i+1}. ID: {sample.get('question_id', 'N/A')} - {sample.get('error_type', 'Unknown error')}")
                if len(failed_samples) > 3:
                    print(f"    ... and {len(failed_samples) - 3} more")
        else:
            print(f"  All samples processed successfully: {valid_count}/{total_processed}")
        
        # Print verification statistics if verification was used
        if verification_stats:
            total_upgrades = (verification_stats['A_to_B_upgrades'] + 
                             verification_stats['A_to_C_upgrades'] +
                             verification_stats['B_to_C_upgrades'])
            print(f"  Enhanced Verification: {total_upgrades}/{verification_stats['total_verifications']} upgrades ({total_upgrades/verification_stats['total_verifications']*100:.1f}%)")
            print(f"    A→B: {verification_stats['A_to_B_upgrades']}, A→C: {verification_stats['A_to_C_upgrades']}, B→C: {verification_stats['B_to_C_upgrades']}")
        
        # Print probability statistics if probability verification was used
        if probability_stats:
            total_prob_upgrades = (probability_stats['A_to_B_upgrades'] + 
                                  probability_stats['A_to_C_upgrades'] + 
                                  probability_stats['B_to_C_upgrades'])
            print(f"  Probability threshold ({args.verify_probabilities}%): {total_prob_upgrades}/{probability_stats['total_classifications']} upgrades ({total_prob_upgrades/probability_stats['total_classifications']*100:.1f}%)")
        
        # Print LLM classification statistics if LLM classification was used
        if llm_classification_stats:
            total_llm_classifications = llm_classification_stats['total_classifications']
            failed_rate = llm_classification_stats['failed_classifications'] / total_llm_classifications * 100 if total_llm_classifications > 0 else 0
            print(f"  LLM Classification: {total_llm_classifications} total, {llm_classification_stats['failed_classifications']} failed ({failed_rate:.1f}%)")
        
        # Progress indicator for multiple datasets
        if len(datasets_to_run) > 1:
            remaining_datasets = len(datasets_to_run) - (dataset_idx + 1)
            if remaining_datasets > 0:
                estimated_remaining_time = dataset_total_time * remaining_datasets
                print(f"Progress: {dataset_idx + 1}/{len(datasets_to_run)} datasets completed")
                print(f"Estimated remaining time: {estimated_remaining_time/60:.1f} minutes")
    
    # Calculate total elapsed time
    total_elapsed = time.time() - total_start_time
    
    # Report API call statistics for Gemini models
    if _is_gemini_model_global(args.model):
        from commaqa.models.gemini_generator import GeminiGenerator
        stats = GeminiGenerator.get_call_stats()
        print(f"\nAPI Call Statistics:")
        print(f"  Total API calls: {stats['total_calls']}")
        print(f"  Failed calls: {stats['failed_calls']}")
        print(f"  Retry calls: {stats['retry_calls']}")
        print(f"  Success rate: {stats['success_rate']:.1%}")
        print(f"  Rate: {stats['calls_per_minute']:.1f} calls/min ({stats['calls_per_second']:.1f} calls/sec)")
        print(f"  Rate limit buffer: {stats['rate_limit_buffer']:.0f} calls/min remaining")
        if stats['calls_per_minute'] > 3500:  # Warn if approaching limit
            print(f"  WARNING: Approaching rate limit! Consider reducing workers or adding delays.")
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE PIPELINE COMPLETE!")
    print(f"{'='*80}")
    print(f"Datasets processed: {len(datasets_to_run)}")
    print(f"Total time: {total_elapsed:.1f}s ({total_elapsed/60:.1f} minutes)")
    
    # Calculate aggregated metrics
    total_queries = sum(results['runtime_metrics']['queries_processed'] for results in all_results.values())
    avg_f1 = sum(results['f1_score'] for results in all_results.values()) / len(all_results)
    avg_em = sum(results['exact_match'] for results in all_results.values()) / len(all_results)
    avg_acc = sum(results['accuracy'] for results in all_results.values()) / len(all_results)
    avg_steps = sum(results['avg_steps'] for results in all_results.values()) / len(all_results)
    
    print(f"\nAggregated Results Across All Datasets:")
    print(f"  Total queries processed: {total_queries}")
    print(f"  Average F1 Score:        {avg_f1:.3f}")
    print(f"  Average Exact Match:     {avg_em:.3f}")
    print(f"  Average Accuracy:        {avg_acc:.3f}")
    print(f"  Average Steps:           {avg_steps:.1f}")
    print(f"  Overall Performance:     {total_queries/total_elapsed:.1f} queries/second")
    
    print(f"\nPer-Dataset Summary:")
    print(f"{'Dataset':<20} {'F1':<8} {'EM':<8} {'Acc':<8} {'Steps':<8} {'Q/s':<8} {'Time':<8}")
    print(f"{'-'*76}")
    for dataset_name, results in all_results.items():
        print(f"{dataset_name:<20} {results['f1_score']:<8.3f} {results['exact_match']:<8.3f} "
              f"{results['accuracy']:<8.3f} {results['avg_steps']:<8.1f} {results['runtime_metrics']['queries_per_second']:<8.1f} "
              f"{results['runtime_metrics']['total_time_seconds']:<8.1f}")
    
    # Print failed samples summary across all datasets
    total_failed = sum(results.get('failed_count', 0) for results in all_results.values())
    total_processed_all = sum(results.get('total_processed', 0) for results in all_results.values())
    
    if total_failed > 0:
        print(f"\n🚨 Failed Samples Summary Across All Datasets:")
        print(f"  Total failed: {total_failed}/{total_processed_all} ({total_failed/total_processed_all*100:.1f}%)")
        
        # List failed samples by dataset
        for dataset_name, results in all_results.items():
            failed_count = results.get('failed_count', 0)
            if failed_count > 0:
                dataset_total = results.get('total_processed', 0)
                print(f"  {dataset_name}: {failed_count}/{dataset_total} failed")
                
                # Show a few example failures for each dataset
                failed_samples = results.get('failed_samples', [])
                for i, sample in enumerate(failed_samples[:2]):  # Show first 2 per dataset
                    print(f"    - {sample.get('question_id', 'N/A')}: {sample.get('error_type', 'Unknown error')}")
                if len(failed_samples) > 2:
                    print(f"    ... and {len(failed_samples) - 2} more")
    else:
        print(f"\nNo failed samples across all datasets!")
    
    # Save aggregated results
    if args.experiment_name:
        experiment_output_dir = f"{args.output_dir}/{args.experiment_name}"
    else:
        experiment_output_dir = args.output_dir
    
    print(f"\n💾 Results saved to: {experiment_output_dir}")
    print(f"🎲 Seed used: {args.seed} (for reproducibility)")
    if args.experiment_name:
        print(f"🧪 Experiment: {args.experiment_name}")
    
    os.makedirs(experiment_output_dir, exist_ok=True)
    aggregated_results = {
        'experiment_metadata': {
            'experiment_name': args.experiment_name,
            'classifier_path': args.classifier_path,
            'model': args.model,
            'verification_enabled': args.verify_classification,
            'classification_llm': args.classification_llm,
            'seed_used': args.seed,
            'max_queries_per_dataset': args.max_queries,
            'workers_used': args.workers,
            'dev_path': args.dev_path,
            'use_test_data': args.test,
            'force_strategy': args.force,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime()),
            'total_runtime_seconds': total_elapsed
        },
        'aggregated_metrics': {
            'total_queries': total_queries,
            'average_f1_score': avg_f1,
            'average_exact_match': avg_em,
            'average_accuracy': avg_acc,
            'average_steps': avg_steps,
            'total_time_seconds': total_elapsed,
            'overall_queries_per_second': total_queries/total_elapsed,
            'datasets_processed': list(datasets_to_run)
        },
        'per_dataset_results': all_results
    }
    
    with open(os.path.join(experiment_output_dir, 'aggregated_results.json'), 'w') as f:
        json.dump(aggregated_results, f, indent=2)
    
    print(f"\nComprehensive evaluation results saved to: {os.path.join(experiment_output_dir, 'aggregated_results.json')}")

if __name__ == "__main__":
    main() 