#!/usr/bin/env python3
"""
Gemini Generator for Adaptive RAG System

This module provides a comprehensive Gemini-based text generation interface
"""

import os
import time
import logging
import json
import re
import random
import threading
from typing import List, Tuple, Optional, Dict, Any, Union
from functools import lru_cache
from dotenv import load_dotenv
from google import genai
from google.genai import types, Client
from commaqa.inference.prompt_reader import fit_prompt_into_given_limit

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# Keywords to check for in responses to detect potential contamination
contamination_keywords = []  # Empty list - no contamination checking needed

def gemini_call(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: List[str],
    n: int,
    thinking_budget: int,
) -> Dict[str, Any]:
    """Make a Gemini API call without caching for fresh results every time"""
    # Always use direct call - no caching to ensure fresh results
    return _direct_gemini_call(
        prompt=prompt,
        model=model,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
        stop=stop,
        n=n,
        thinking_budget=thinking_budget,
    )

def _direct_gemini_call(
    prompt: str,
    model: str,
    temperature: float,
    max_tokens: int,
    top_p: float,
    stop: List[str],
    n: int,
    thinking_budget: int,
) -> Dict[str, Any]:
    """Direct Gemini API call without caching"""
    logger.debug(f"🔍 _direct_gemini_call DEBUG: Received parameters:")
    logger.debug(f"  prompt: {type(prompt)} (len={len(prompt) if prompt else 0})")
    logger.debug(f"  model: {type(model)} = {model}")
    logger.debug(f"  temperature: {type(temperature)} = {temperature}")
    logger.debug(f"  max_tokens: {type(max_tokens)} = {max_tokens}")
    logger.debug(f"  top_p: {type(top_p)} = {top_p}")
    logger.debug(f"  stop: {type(stop)} = {stop}")
    logger.debug(f"  n: {type(n)} = {n}")
    logger.debug(f"  thinking_budget: {type(thinking_budget)} = {thinking_budget}")
    
    client = genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))
    
    # Handle stop sequences carefully - Gemini is picky about this
    stop_sequences = None
    if stop and len(stop) > 0:
        # Filter out empty strings and common problematic sequences
        filtered_stop = [s for s in stop if s and s not in ["\n", "\r", "\r\n"]]
        stop_sequences = filtered_stop if filtered_stop else None
    
    logger.debug(f"🔍 CONFIG DEBUG: Creating GenerateContentConfig with:")
    logger.debug(f"  temperature: {temperature}")
    logger.debug(f"  max_output_tokens: {max_tokens}")
    logger.debug(f"  top_p: {top_p}")
    logger.debug(f"  thinking_budget: {thinking_budget}")
    logger.debug(f"  stop_sequences: {stop_sequences}")
    logger.debug(f"  candidate_count: {n}")
    
    try:
        # Flexible safety settings - disable all content filtering for maximum flexibility
        flexible_safety_settings = [
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HARASSMENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_HATE_SPEECH, 
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
            types.SafetySetting(
                category=types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT,
                threshold=types.HarmBlockThreshold.BLOCK_NONE
            ),
        ]
        
        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            top_p=top_p,
            thinking_config=types.ThinkingConfig(thinking_budget=thinking_budget) if thinking_budget > 0 else None,
            stop_sequences=stop_sequences,
            candidate_count=n,
            safety_settings=flexible_safety_settings,
        )
        logger.debug(f"🔍 CONFIG DEBUG: Successfully created GenerateContentConfig")
    except Exception as e:
        logger.error(f"🚨 CONFIG DEBUG: Failed to create GenerateContentConfig: {e}")
        logger.error(f"🚨 CONFIG DEBUG: Exception type: {type(e)}")
        raise
    
    # Gemini API expects contents as a list of content objects, not a raw string
    contents = [{"role": "user", "parts": [{"text": prompt}]}]
    
    response = client.models.generate_content(
        model=model,
        contents=contents,
        config=config
    )
    
    # DEBUG: Log full response structure to understand None responses
    logger.debug(f"🔍 GEMINI RESPONSE DEBUG:")
    logger.debug(f"  response type: {type(response)}")
    logger.debug(f"  response.text: {response.text}")
    logger.debug(f"  response.candidates: {getattr(response, 'candidates', 'NOT_FOUND')}")
    
    # Check for candidates and their finish reasons
    if hasattr(response, 'candidates') and response.candidates:
        for i, candidate in enumerate(response.candidates):
            logger.debug(f"  candidate[{i}].content: {getattr(candidate, 'content', 'NOT_FOUND')}")
            logger.debug(f"  candidate[{i}].finish_reason: {getattr(candidate, 'finish_reason', 'NOT_FOUND')}")
            if hasattr(candidate, 'safety_ratings'):
                logger.debug(f"  candidate[{i}].safety_ratings: {candidate.safety_ratings}")
    
    return {
        "text": response.text,
        "candidates": [{"text": response.text}],
        "model": model,
        "usage": getattr(response, 'usage', None),
        "finish_reason": getattr(response.candidates[0], 'finish_reason', None) if hasattr(response, 'candidates') and response.candidates else None,
        "safety_ratings": getattr(response.candidates[0], 'safety_ratings', None) if hasattr(response, 'candidates') and response.candidates else None
    }

@lru_cache(maxsize=1)
def get_gemini_client():
    """Get a cached Gemini client."""
    return genai.Client(api_key=os.getenv('GOOGLE_API_KEY'))

class GeminiGenerator:
    """
    Comprehensive Gemini text generator that provides the same interface as GPT3Generator
    but uses Google's Gemini-2.5-Flash-Lite model with all advanced features.
    """
    
    # Class-level call tracking for rate limit monitoring (thread-safe)
    _total_api_calls = 0
    _failed_api_calls = 0
    _retry_api_calls = 0
    _start_time = None
    _last_call_time = 0
    _stats_lock = threading.Lock()  # Thread safety for statistics and rate limiting
    
    @classmethod
    def get_call_stats(cls):
        """Get API call statistics (thread-safe)"""
        with cls._stats_lock:
            elapsed = time.time() - cls._start_time if cls._start_time else 0
            calls_per_minute = (cls._total_api_calls / elapsed * 60) if elapsed > 0 else 0
            calls_per_second = cls._total_api_calls / elapsed if elapsed > 0 else 0
            return {
                "total_calls": cls._total_api_calls,
                "failed_calls": cls._failed_api_calls,
                "retry_calls": cls._retry_api_calls,
                "success_rate": (cls._total_api_calls - cls._failed_api_calls) / max(cls._total_api_calls, 1),
                "elapsed_seconds": elapsed,
                "calls_per_minute": calls_per_minute,
                "calls_per_second": calls_per_second,
                "rate_limit_buffer": max(0, 4000 - calls_per_minute)  # Remaining capacity
            }
    
    @classmethod
    def reset_call_stats(cls):
        """Reset API call statistics (thread-safe)"""
        with cls._stats_lock:
            cls._total_api_calls = 0
            cls._failed_api_calls = 0
            cls._retry_api_calls = 0
            cls._start_time = time.time()
            cls._last_call_time = 0
    
    @classmethod
    def _apply_rate_limit(cls):
        """Apply small delay to respect rate limits (thread-safe)"""
        with cls._stats_lock:
            current_time = time.time()
            if cls._last_call_time > 0:
                time_since_last = current_time - cls._last_call_time
                # Target: ~25 calls/second max (more conservative for session limits) 
                min_interval = 0.04  # 40ms between calls
                if time_since_last < min_interval:
                    sleep_time = min_interval - time_since_last
                    # Release lock during sleep to avoid blocking other threads
                    cls._last_call_time = current_time + sleep_time  # Pre-set to prevent race
                    cls._stats_lock.release()
                    time.sleep(sleep_time)
                    cls._stats_lock.acquire()
            cls._last_call_time = time.time()
    
    def __init__(
        self,
        model: str = "gemini-2.5-flash-lite",
        temperature: float = 0.0,
        max_tokens: int = 1000,
        top_p: float = 0.9,
        stop: Union[str, List[str]] = None,
        api_key: Optional[str] = None,
        retry_after_n_seconds: int = 2,
        sleep_if_needed: bool = True,
        timeout: int = 30,
        thinking_budget: int = 0,
        **kwargs
    ):
        """
        Initialize Gemini Generator with comprehensive parameters.
        
        Args:
            model: Gemini model to use (default: "gemini-2.5-flash-lite")
            temperature: Sampling temperature (0.0 to 1.0)
            max_tokens: Maximum tokens to generate
            top_p: Top-p sampling parameter
            stop: Stop sequences
            api_key: Google API key (auto-detected if not provided)
            retry_after_n_seconds: Seconds to wait before retrying failed requests
            sleep_if_needed: Whether to sleep when rate limited
            timeout: Request timeout in seconds
            thinking_budget: Thinking budget for Gemini (0 = no thinking)
            **kwargs: Additional parameters
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.stop = stop or ["\n"]
        self.api_key = api_key
        self.retry_after_n_seconds = retry_after_n_seconds
        self.sleep_if_needed = sleep_if_needed
        self.timeout = timeout
        self.thinking_budget = thinking_budget
        
        # Set compatibility parameters for OpenAI GPT interface
        self.n = 1  # Gemini generates one response at a time
        self.best_of = 1  # Not applicable to Gemini
        self.logprobs = 0  # Gemini doesn't provide logprobs like OpenAI
        self.remove_method = "first"  # Default method for token removal
        
        # Set model token limits - both Gemini models support 128K context window
        # Note: gemini-1.5-flash-8b is deprecated but still supported for backward compatibility
        self.model_tokens_limit = 128000  # 128K tokens for Gemini models
        
        # Validate API key
        if not os.getenv('GOOGLE_API_KEY'):
            raise ValueError("GOOGLE_API_KEY not found in environment variables. Please set it in .env file.")
        
        # Only log initialization once per unique configuration to reduce spam
        config_key = f"{model}_{temperature}"
        if not hasattr(GeminiGenerator, '_logged_configs'):
            GeminiGenerator._logged_configs = set()
        
        if config_key not in GeminiGenerator._logged_configs:
            # Add deprecation warning for gemini-1.5-flash-8b
            if "gemini-1.5-flash-8b" in model.lower():
                logger.warning(f"⚠️  Model {model} is deprecated by Google. Consider migrating to gemini-2.5-flash-lite for continued support.")
            logger.info(f"Initialized GeminiGenerator with model: {model}, temperature: {temperature}")
            GeminiGenerator._logged_configs.add(config_key)
        else:
            logger.debug(f"Reusing GeminiGenerator configuration: {model}, temperature: {temperature}")
    
    # Class-level token cache to avoid repeated API calls
    _token_cache = {}
    _cache_max_size = 1000
    
    def _count_tokens(self, text: str) -> int:
        """Count tokens using Gemini API with caching"""
        # Create cache key from text hash for fast lookup
        import hashlib
        cache_key = hashlib.md5(text.encode()).hexdigest()
        
        # Check cache first
        if cache_key in GeminiGenerator._token_cache:
            return GeminiGenerator._token_cache[cache_key]
        
        try:
            client = get_gemini_client()
            total_tokens = client.models.count_tokens(model=self.model, contents=text)
            token_count = total_tokens.total_tokens
            
            # Cache the result
            if len(GeminiGenerator._token_cache) >= GeminiGenerator._cache_max_size:
                # Remove oldest entry if cache is full
                GeminiGenerator._token_cache.pop(next(iter(GeminiGenerator._token_cache)))
            GeminiGenerator._token_cache[cache_key] = token_count
            
            return token_count
        except Exception as e:
            logger.warning(f"Failed to count tokens with Gemini API: {e}, using approximation")
            # Fallback to rough approximation: ~4 chars per token
            fallback_count = len(text) // 4
            # Cache fallback too
            GeminiGenerator._token_cache[cache_key] = fallback_count
            return fallback_count

    def _fit_gemini_prompt_into_limit(
        self,
        original_prompt: str,
        model_length_limit: int,
        estimated_generation_length: int,
        demonstration_delimiter: str = "\n\n\n",
        shuffle: bool = False,
        remove_method: str = "first",  # first, last, random, largest
        last_is_test_example: bool = True,
    ) -> str:
        """
        Fit prompt into model limits using Gemini API for token counting.
        
        This follows the same pattern as fit_prompt_into_given_limit but uses
        Gemini API for accurate token counting instead of HuggingFace tokenizers.
        """
        # Early exit optimization: if prompt is clearly short, skip token counting
        if len(original_prompt) < model_length_limit * 2:  # ~2 chars per token conservative estimate
            return original_prompt
            
        assert remove_method in (
            "first",
            "last", 
            "random",
            "largest",
        ), "The remove_method must be from first, last, random, largest."

        # ---- 1. Split the prompt into individual demonstrations ----
        demonstrations = original_prompt.strip().split(demonstration_delimiter)
        demonstrations = [demonstration.strip() for demonstration in demonstrations if demonstration.strip()]

        if len(demonstrations) <= 1:
            print("EXTREME WARNING: Found only one demonstration/example.")
            return original_prompt

        # Calculate sizes for all demonstrations using Gemini API
        demonstration_sizes = [self._count_tokens(demonstration) for demonstration in demonstrations]

        # ---- 2. Separate the test example (if applicable) ----
        test_example = None
        test_example_size = 0
        if last_is_test_example:
            test_example = demonstrations.pop(-1)
            test_example_size = demonstration_sizes.pop(-1)

        # ---- 3. Iteratively remove demonstrations until the prompt fits ----
        while True:
            updated_length = sum(demonstration_sizes) + test_example_size + estimated_generation_length
            if updated_length < model_length_limit or not demonstration_sizes:
                break

            if remove_method == "first":
                remove_index = 0
            elif remove_method == "last":
                remove_index = -1
            elif remove_method == "random":
                remove_index = random.randint(0, len(demonstrations) - 1)
            elif remove_method == "largest":
                remove_index = demonstration_sizes.index(max(demonstration_sizes))
            else:
                raise Exception(f"Unexpected remove_method: {remove_method}.")

            demonstrations.pop(remove_index)
            demonstration_sizes.pop(remove_index)

            assert len(demonstrations) == len(demonstration_sizes)

        # ---- 4. Reconstruct the final prompt ----
        if shuffle:
            random.shuffle(demonstrations)

        if last_is_test_example:
            updated_prompt = demonstration_delimiter.join(demonstrations + [test_example])
        else:
            updated_prompt = demonstration_delimiter.join(demonstrations)

        # ---- 5. Final emergency truncation if still too long ----
        # This happens if the test example alone is too big for the context window.
        final_tokens = self._count_tokens(updated_prompt)
        if final_tokens + estimated_generation_length > model_length_limit:
            print("EXTREME WARNING: Even after removing all demonstrations, the prompt is still too long.")
            # Emergency truncation - truncate the test example itself
            if test_example:
                # Rough truncation by character count as last resort
                target_chars = int((model_length_limit - estimated_generation_length) * 4)  # ~4 chars per token
                if len(updated_prompt) > target_chars:
                    updated_prompt = updated_prompt[:target_chars]

        return updated_prompt.strip()
    
    def generate_text_sequence(self, prompt: str) -> List[Tuple[str, float]]:
        """
        Generate text sequence using Gemini API.
        
        Args:
            prompt: Input prompt string
            
        Returns:
            List of (text, score) tuples where lower score is better
        """
        # Clean prompt (similar to GPT3Generator)
        prompt = prompt.rstrip()
        
        # Fit prompt into token limit using Gemini API for accurate token counting
        prompt = self._fit_gemini_prompt_into_limit(
            original_prompt=prompt,
            model_length_limit=self.model_tokens_limit,
            estimated_generation_length=self.max_tokens,
            demonstration_delimiter="\n\n\n",
            shuffle=False,
            remove_method=self.remove_method,
            last_is_test_example=True,
        )
        
        # Prepare API call arguments (excluding freq/presence penalties - not needed for Gemini)
        arguments = {
            "model": self.model,
            "prompt": prompt,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop,
            "n": self.n,
            "thinking_budget": self.thinking_budget,
        }

        # Thread-safe API call tracking 
        import threading
        worker_id = f"worker_{threading.current_thread().ident}"
        
        # Retry logic with specific exponential backoff intervals
        success = False
        max_attempts = 3  # 3 retries as requested
        retry_delays = [1, 5, 10]  # Specific delays: 1s, 5s, 10s
        
        for attempt in range(max_attempts):
            try:
                logger.debug(f"Gemini API attempt {attempt + 1}/{max_attempts} for {worker_id}")
                
                # Track API call statistics (thread-safe)
                with GeminiGenerator._stats_lock:
                    if GeminiGenerator._start_time is None:
                        GeminiGenerator.reset_call_stats()
                    GeminiGenerator._total_api_calls += 1
                    if attempt > 0:
                        GeminiGenerator._retry_api_calls += 1
                
                # Apply rate limiting delay
                GeminiGenerator._apply_rate_limit()
                
                # Log current rate for monitoring
                stats = GeminiGenerator.get_call_stats()
                if GeminiGenerator._total_api_calls % 10 == 0:  # Log every 10 calls
                    logger.info(f"📊 API Rate: {stats['calls_per_minute']:.1f}/min, {stats['calls_per_second']:.1f}/sec, Buffer: {stats['rate_limit_buffer']:.0f}/min")
                
                response = gemini_call(**arguments)
                
                # CRITICAL: Check for None responses before marking as successful
                if response:
                    # Check main response text
                    main_text = response.get("text")
                    if main_text is None:
                        finish_reason = response.get("finish_reason")
                        safety_ratings = response.get("safety_ratings")
                        logger.warning(f"🚨 GEMINI DEBUG: Main response returned None text")
                        logger.warning(f"🚨 FINISH REASON: {finish_reason}")
                        logger.warning(f"🚨 SAFETY RATINGS: {safety_ratings}")
                        raise Exception(f"Gemini API returned None response - finish_reason: {finish_reason}, safety: {safety_ratings}")
                    
                    # Check candidates if they exist
                    if "candidates" in response and response["candidates"]:
                        for index, candidate in enumerate(response["candidates"]):
                            cand_text = candidate.get("text")
                            if cand_text is None:
                                finish_reason = response.get("finish_reason")
                                safety_ratings = response.get("safety_ratings")
                                logger.warning(f"🚨 GEMINI DEBUG: Candidate {index} returned None text")
                                logger.warning(f"🚨 FINISH REASON: {finish_reason}")
                                logger.warning(f"🚨 SAFETY RATINGS: {safety_ratings}")
                                raise Exception(f"Gemini API returned None response for candidate {index} - finish_reason: {finish_reason}, safety: {safety_ratings}")
                
                # DEBUG: Check response for contamination
                if response and 'choices' in response:
                    for i, choice in enumerate(response['choices']):
                        if choice and 'text' in choice:
                            choice_text = choice['text']
                            for keyword in contamination_keywords:
                                if keyword in choice_text:
                                    logger.error(f"🚨 GEMINI RESPONSE CONTAMINATION: {worker_id} choice {i} contains {keyword}")
                                    logger.error(f"🚨 Response text: {choice_text[:200]}...")
                                    break
                
                logger.debug(f"Gemini API success for {worker_id}")
                success = True
                break
            except Exception as exception:
                success = False
                with GeminiGenerator._stats_lock:
                    GeminiGenerator._failed_api_calls += 1
                logger.warning(f"Gemini API attempt {attempt + 1}/{max_attempts} failed: {str(exception)}")
                
                # Specific backoff delays: wait before next attempt
                if attempt < max_attempts - 1:  # Don't wait after last attempt
                    delay = retry_delays[attempt]
                    logger.warning(f"⏳ Waiting {delay}s before retry {attempt + 2}/{max_attempts}")
                    import time
                    time.sleep(delay)
                
                # Check for resource exhausted error and return special response to indicate sample should be discarded
                exception_str = str(exception).lower()
                if "resource exhausted" in exception_str or "quota exceeded" in exception_str or "429" in exception_str:
                    logger.warning(f"Resource exhausted error detected - sample will be discarded: {exception}")
                    # Return a list of tuples format expected by IRCoT with special marker
                    return [("__RESOURCE_EXHAUSTED__", 999.0)]
                
                # Handle token limit exceeded
                client = get_gemini_client()
                # Format contents properly for token counting
                contents_for_count = [{"role": "user", "parts": [{"text": prompt}]}]
                prompt_num_tokens = client.models.count_tokens(model=self.model, contents=contents_for_count).total_tokens
                if prompt_num_tokens + arguments["max_tokens"] > self.model_tokens_limit > prompt_num_tokens:
                    last_used_max_tokens = arguments["max_tokens"]
                    updated_max_tokens = self.model_tokens_limit - prompt_num_tokens
                    arguments["max_tokens"] = updated_max_tokens
                    if last_used_max_tokens == updated_max_tokens:
                        break
                    logger.warning(
                        f"WARNING: (Round {attempt + 1}) Decreasing max_tokens from "
                        f"{last_used_max_tokens} to {updated_max_tokens} and retrying."
                    )
                    continue
                
                # If this is the last attempt, don't sleep
                if attempt >= max_attempts - 1:
                    break
                
                # Handle rate limiting with specific backoff intervals
                if self.retry_after_n_seconds is None:
                    import traceback
                    logger.error(f"Gemini API call failed: {traceback.format_exc()}")
                    raise exception
                
                # Use specific delay intervals
                if attempt < len(retry_delays):
                    total_delay = retry_delays[attempt]
                else:
                    total_delay = retry_delays[-1]  # Use last delay if we exceed the list
                
                logger.warning(f"Gemini API failed (attempt {attempt + 1}/{max_attempts}). "
                             f"Retrying in {total_delay}s (specific interval)")
                time.sleep(total_delay)
        
        if not success:
            logger.error(f"🚨 GEMINI DEBUG: {worker_id} FAILED after {max_attempts} attempts - discarding sample")
            # Return special marker to indicate sample should be discarded
            return [("__API_FAILED__", 999.0)]
        
        # Process response and return list of (text, score) tuples
        output_seq_score = []
        if "candidates" in response and response["candidates"]:
            for index, candidate in enumerate(response["candidates"]):
                text = candidate.get("text", "")
                # None check already done in retry loop - just ensure we have text
                if text is None:
                    text = ""  # Fallback to empty string if somehow None slipped through
                # Use index as score (lower is better, so first candidate gets score 0)
                score = float(index)
                output_seq_score.append((text, score))
        else:
            # Fallback to main response text
            text = response.get("text", "")
            # None check already done in retry loop - just ensure we have text
            if text is None:
                text = ""  # Fallback to empty string if somehow None slipped through
            output_seq_score.append((text, 0.0))
        
        # Ensure we have at least one result
        if not output_seq_score:
            output_seq_score.append(("", 100.0))
        
        # Sort by score (lower is better) and return
        return sorted(output_seq_score, key=lambda x: x[1])
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the current model configuration"""
        return {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "top_p": self.top_p,
            "stop": self.stop,
            "model_tokens_limit": self.model_tokens_limit,
            "thinking_budget": self.thinking_budget,
        }
    
    def estimate_tokens(self, text: str) -> int:
        """Estimate number of tokens in text"""
        client = get_gemini_client()
        return client.models.count_tokens(model=self.model, contents=text).total_tokens
    
    def __repr__(self) -> str:
        return f"GeminiGenerator(model='{self.model}', temperature={self.temperature})" 