#!/usr/bin/env python3
"""
Synthetic Ground Truth Checker

This module provides semantic evaluation of question-answer pairs using Gemini-2.5-flash-lite
as a judge to determine if answers are semantically correct even when exact string matching fails.

Key features:
- true/false evaluation
- Runs immediately after exact match failures
- Pipeline-aware labeling: answers are mapped based on which pipeline produced the correct answer
  * nor_qa → A (No retrieval)
  * oner_qa → B (One retrieval step)
  * ircot_qa → C (Iterative retrieval chain-of-thought)
- Parallel processing with configurable batch size for efficient evaluation
"""

import time
import logging
import os
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed


from dotenv import load_dotenv
# Look for .env file in project root (go up 2 levels from utils/)
env_path = os.path.join(os.path.dirname(__file__), '..', '..', '.env')
load_dotenv(env_path)


import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold, GenerationConfig
GENAI_AVAILABLE = True

# Set up logging
logger = logging.getLogger(__name__)

# Suppress verbose HTTP request logs from Google AI libraries
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)


@dataclass
class SyntheticCheckResult:
    """Result of synthetic semantic evaluation"""
    is_semantically_correct: bool
    processing_time: float
    error: Optional[str] = None
    rescued_to_label: Optional[str] = None  # The new label when rescued (A, B, or C)
    source_pipeline: Optional[str] = None  # Which pipeline produced the correct answer (nor_qa, oner_qa, ircot_qa)


class SyntheticChecker:
    """
    Semantic evaluation checker using Gemini-2.5-flash-lite as a judge.
    
    This checker evaluates whether an answer is semantically correct for a given question,
    even if it doesn't match the ground truth exactly through string comparison.
    """
    
    def __init__(self, api_key: Optional[str] = None, batch_size: int = 8, model_name: str = "gemini-2.5-flash-lite", max_retries: int = 3):
        """
        Initialize the synthetic checker
        
        Args:
            api_key: Google API key for Gemini. If None, will try to get from environment
            batch_size: Number of parallel evaluations to process at once
            model_name: Gemini model to use for evaluation
            max_retries: Maximum number of retry attempts for failed requests
        """
        self.batch_size = batch_size
        self.max_retries = max_retries
        
        if model_name not in ["gemini-2.5-flash-lite", "gemini-1.5-flash-8b"]:
            logger.warning(f"Unknown model {model_name}, defaulting to gemini-2.5-flash")
            model_name = "gemini-2.5-flash"
        
        self.model_name = model_name
        
        # Configure Gemini API
        if not GENAI_AVAILABLE:
            logger.warning("Google Generative AI not available. Install with: pip install google-generativeai")
            self.model = None
        else:
            if api_key:
                genai.configure(api_key=api_key)
            
            # Initialize the model
            try:
                self.model = genai.GenerativeModel(self.model_name)
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini model: {e}. Check API key and quota.")
                self.model = None
        
        # Judge prompt template for semantic evaluation
        self.judge_prompt_template = """
You are an expert fact-checker evaluating answer correctness.

EVALUATION PROCESS:
1. EXTRACT the core factual entity/name/date that directly answers the question
2. CHECK if that core entity matches what's expected
3. IGNORE extra context, formatting, explanations, and minor descriptive differences
4. PAY ATTENTION to question type: singular vs plural expectations

DECISION CRITERIA:
✓ CORRECT: The answer contains the correct core fact/entity/date to answer the question
✗ INCORRECT: The answer's core entity/fact/date is factually wrong or missing

Examples:
- "1976" matches "released in 1976" for release year questions
- "March 13" matches "March 13, 2019" for day/month questions, unless specifically ask for the year as well.

Examples on Singular Questions:
- If question asks "Which [singular]..." or "What [singular]...", look for the ONE correct entity
- When multiple entities are mentioned, check if the reasoning clearly identifies the correct one
- Accept answers that mention multiple entities IF the correct one is clearly identified as the answer

ACCEPTABLE VARIATIONS:
- Different descriptive terms for the same entity (e.g., "Paris city" vs "Paris town" vs "Paris")
- Different wordings (e.g., "Rome" vs "Rome is the city" vs "Therefore, Rome")
- Additional context or explanation that doesn't contradict the core entity
- Different formats (quoted, unquoted, with reasoning, conclusions, etc.)

EXAMPLES:
Q: "What is the capital of France?"
Ground truth: "Paris"
- "Paris" → CORRECT (direct answer)
- "The capital is Paris, located in northern France" → CORRECT (correct + context)
- "Paris is a city" → CORRECT (same entity, different description)
- "Lyon" → INCORRECT (wrong entity)
- "I don't know" → INCORRECT (no answer)

Q: "Which team won the championship?"
Ground truth: "Lakers"
- "Lakers" → CORRECT (direct answer)
- "The championship was contested by Lakers and Celtics, with Lakers winning" → CORRECT (mentions both but identifies Lakers as winner)
- "Lakers and Celtics both played" → INCORRECT (doesn't identify Lakers as the winner)

Q: "What town is located between X and Y?"
Ground truth: "Springfield is a town"
- "Springfield" → CORRECT (core entity correct)
- "Therefore, Springfield is the city" → CORRECT (entity correct, description differs)
- "Boston" → INCORRECT (wrong entity)

Question: {question}

Answer to evaluate: {answer}

Ground truth reference: {ground_truth}

IMPORTANT: You must respond with EXACTLY ONE WORD: either "true" (if correct) or "false" (if incorrect). No other text.

Response:"""

    def _make_gemini_request(self, prompt: str) -> Dict[str, Any]:
        """
        Make a direct request to the Gemini API with retry mechanism and safety settings
        
        Args:
            prompt: The prompt to send to Gemini
            
        Returns:
            Response dictionary from Gemini API
        """
        if self.model is None:
            return {
                'answer': '',
                'success': False,
                'error': 'Gemini model not initialized'
            }
        
        safety_settings = None
        
        # Configure generation parameters with temperature=0 for consistent evaluation
        generation_config = GenerationConfig(
            temperature=0.0       # Low temperature for consistent evaluation
        )
        
        # Retry mechanism with exponential backoff
        for attempt in range(self.max_retries):
            try:
                # Generate response with temperature=0 for consistency
                response = self.model.generate_content(
                    prompt,
                    generation_config=generation_config
                )
                
                # Check if we have a valid response with text
                try:
                    if response.text and response.text.strip():
                        return {
                            'answer': response.text.strip(),
                            'success': True,
                            'error': None
                        }
                except ValueError:
                    # response.text can throw ValueError if response is blocked
                    pass
                
                # If we get here, response was blocked or empty
                    # Check if content was blocked by safety filters
                    if hasattr(response, 'prompt_feedback') and response.prompt_feedback:
                        if hasattr(response.prompt_feedback, 'block_reason'):
                            error_msg = f"Content blocked by safety filters: {response.prompt_feedback.block_reason}"
                        else:
                            error_msg = "Content potentially blocked by safety filters"
                    elif hasattr(response, 'candidates') and response.candidates:
                        candidate = response.candidates[0]
                        if hasattr(candidate, 'finish_reason'):
                            if candidate.finish_reason == 'SAFETY':
                                error_msg = "Response blocked by safety filters"
                            else:
                                error_msg = f"Response generation stopped: {candidate.finish_reason}"
                        else:
                            error_msg = "Empty response from Gemini"
                    else:
                        error_msg = "Empty response from Gemini"
                    
                    # Don't retry for safety blocks, return immediately
                    if "safety" in error_msg.lower() or "blocked" in error_msg.lower():
                        logger.warning(f"Content blocked by safety filters: {prompt[:100]}...")
                        return {
                            'answer': '',
                            'success': False,
                            'error': error_msg
                        }
                    
                    # For other empty responses, continue to retry
                    if attempt == self.max_retries - 1:
                        return {
                            'answer': '',
                            'success': False,
                            'error': error_msg
                        }
                    
            except Exception as e:
                error_msg = str(e)
                logger.warning(f"Gemini API request failed (attempt {attempt + 1}/{self.max_retries}): {error_msg}")
                
                # Don't retry for authentication or quota errors
                if any(keyword in error_msg.lower() for keyword in ['api key', 'authentication', 'quota', 'permission']):
                    return {
                        'answer': '',
                        'success': False,
                        'error': error_msg
                    }
                
                # If this is the last attempt, return the error
                if attempt == self.max_retries - 1:
                    logger.error(f"All {self.max_retries} attempts failed for Gemini API request")
                    return {
                        'answer': '',
                        'success': False,
                        'error': error_msg
                    }
                
                # Shorter backoff intervals: 1s, 5s, 10s
                retry_delays = [1, 5, 10]
                wait_time = retry_delays[min(attempt, len(retry_delays) - 1)]
                logger.info(f"Retrying in {wait_time} seconds... (attempt {attempt + 1}/{self.max_retries})")
                time.sleep(wait_time)
        
        # This should never be reached, but just in case
        return {
            'answer': '',
            'success': False,
            'error': 'Maximum retries exceeded'
        }

    def _parse_judge_response(self, response_text: str) -> bool:
        """
        Parse the judge's simple true/false response
        
        Args:
            response_text: Raw response text from Gemini
            
        Returns:
            Boolean indicating if answer is semantically correct
        """
        response_clean = response_text.strip().lower()
        
        # Direct true/false check
        if 'true' in response_clean or response_clean == 'true':
            return True
        elif 'false' in response_clean or response_clean == 'false':
            return False
        else:
            # Default to false if unclear
            logger.warning(f"Unclear response, defaulting to false: {response_text[:50]}...")
            return False

    def _map_pipeline_to_label(self, pipeline: str) -> str:
        """
        Map pipeline name to corresponding label
        
        Args:
            pipeline: Pipeline name (nor_qa, oner_qa, ircot_qa)
            
        Returns:
            Corresponding label (A, B, C)
        """
        pipeline_mapping = {
            'nor_qa': 'A',    # No retrieval
            'oner_qa': 'B',   # One retrieval step
            'ircot_qa': 'C'   # Iterative retrieval chain-of-thought
        }
        return pipeline_mapping.get(pipeline, None)

    def check_semantic_correctness(self, 
                                 question: str, 
                                 answer: str, 
                                 ground_truth: str = None,
                                 source_pipeline: str = None) -> SyntheticCheckResult:
        """
        Check if an answer is semantically correct for a given question
        
        Args:
            question: The question being asked
            answer: The answer to evaluate
            ground_truth: Optional ground truth for reference
            source_pipeline: Which pipeline produced this answer (nor_qa, oner_qa, ircot_qa)
            
        Returns:
            SyntheticCheckResult with evaluation details
        """
        start_time = time.time()
        
        # Prepare the prompt
        prompt = self.judge_prompt_template.format(
            question=question.strip(),
            answer=answer.strip(),
            ground_truth=ground_truth.strip() if ground_truth else "Not provided"
        )
        
        # Make the request to Gemini
        response = self._make_gemini_request(prompt)
        processing_time = time.time() - start_time
        
        if not response['success']:
            return SyntheticCheckResult(
                is_semantically_correct=False,
                processing_time=processing_time,
                error=response['error'],
                rescued_to_label=None,
                source_pipeline=source_pipeline
            )
        
        # Parse the judge's response
        is_correct = self._parse_judge_response(response['answer'])
        
        
        logger.debug(f"Synthetic check result: {is_correct}")
        
        return SyntheticCheckResult(
            is_semantically_correct=is_correct,
            processing_time=processing_time,
            source_pipeline=source_pipeline
        )

    def _check_single_pair(self, pair: Dict[str, str]) -> SyntheticCheckResult:
        """
        Helper method to check a single question-answer pair (for parallel processing)
        
        Args:
            pair: Dict with 'question', 'answer', optional 'ground_truth', and optional 'source_pipeline'
            
        Returns:
            SyntheticCheckResult object
        """
        question = pair.get('question', '')
        answer = pair.get('answer', '')
        ground_truth = pair.get('ground_truth', None)
        source_pipeline = pair.get('source_pipeline', None)
        
        return self.check_semantic_correctness(
            question=question, 
            answer=answer, 
            ground_truth=ground_truth,
            source_pipeline=source_pipeline,
        )

    def batch_check_semantic_correctness(self, 
                                       question_answer_pairs: List[Dict[str, str]]) -> List[SyntheticCheckResult]:
        """
        Check multiple question-answer pairs for semantic correctness using parallel processing
        
        Args:
            question_answer_pairs: List of dicts with 'question', 'answer', optional 'ground_truth', and optional 'source_pipeline'
            
        Returns:
            List of SyntheticCheckResult objects
        """
        if not question_answer_pairs:
            return []
            
        logger.info(f"Running synthetic checks on {len(question_answer_pairs)} question-answer pairs "
                   f"with batch size {self.batch_size}")
        
        results = []
        total_pairs = len(question_answer_pairs)
        
        # Process in batches with parallel execution
        for batch_start in range(0, total_pairs, self.batch_size):
            batch_end = min(batch_start + self.batch_size, total_pairs)
            batch = question_answer_pairs[batch_start:batch_end]
            
            logger.debug(f"Processing batch {batch_start//self.batch_size + 1}: "
                        f"items {batch_start+1}-{batch_end} of {total_pairs}")
            
            # Use ThreadPoolExecutor for parallel processing within batch
            with ThreadPoolExecutor(max_workers=min(len(batch), self.batch_size)) as executor:
                # Submit all tasks in the batch
                future_to_pair = {
                    executor.submit(self._check_single_pair, pair): pair 
                    for pair in batch
                }
                
                # Collect results as they complete
                batch_results = []
                for future in as_completed(future_to_pair):
                    try:
                        result = future.result()
                        batch_results.append(result)
                    except Exception as e:
                        pair = future_to_pair[future]
                        logger.error(f"Error processing pair {pair.get('question', '')[:50]}...: {e}")
                        # Create error result
                        batch_results.append(SyntheticCheckResult(
                            is_semantically_correct=False,
                            processing_time=0.0,
                            error=str(e),
                            source_pipeline=pair.get('source_pipeline', None)
                        ))
                
                # Sort results to maintain original order within batch
                # (Note: order may not be preserved across threads, but that's usually okay)
                results.extend(batch_results)
            
            # Log progress
            logger.info(f"Completed synthetic checks: {len(results)}/{total_pairs}")
        
        logger.info(f"Finished processing all {len(results)} question-answer pairs")
        return results


def create_synthetic_checker(api_key: Optional[str] = None, batch_size: int = 8, model_name: str = "gemini-2.5-flash-lite", max_retries: int = 3) -> SyntheticChecker:
    """
    Factory function to create a SyntheticChecker instance
    
    Args:
        api_key: Google API key for Gemini. If None, will try to get from environment
        batch_size: Number of parallel evaluations to process at once
        model_name: Gemini model to use for evaluation (gemini-2.5-flash, gemini-2.5-flash-lite, or gemini-1.5-flash-8b)
        max_retries: Maximum number of retry attempts for failed requests
        
    Returns:
        Configured SyntheticChecker instance
    """
    return SyntheticChecker(api_key=api_key, batch_size=batch_size, model_name=model_name, max_retries=max_retries)
