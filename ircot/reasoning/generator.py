#!/usr/bin/env python3
"""
Reasoning Generator - Handles LLM generation for IRCOT reasoning steps.

This module provides:
- Model-agnostic generation interface
- Support for Gemini, Beta: Qwen, and FLAN-T5
- Retry logic and error handling
- Response post-processing
"""

import os
import logging
import requests
import time
from typing import Dict, Any, List, Optional
import json
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import model-specific generators
from commaqa.models.gemini_generator import GeminiGenerator
from commaqa.models.llm_client_generator import LLMClientGenerator

# Import the adapter pattern
try:
    from ..adapters.generator_adapter import create_generator_adapter
except ImportError:
    # Fallback if adapter not available yet
    def create_generator_adapter(generator, model_name):
        return generator


class ReasoningGenerator:
    """
    Generates reasoning steps using LLMs.
    
    Supports:
    - Multiple model backends
    - Retry logic
    - Response formatting
    - Temperature control
    """
    
    def __init__(self,
                 model: str,
                 logger: Optional[logging.Logger] = None,
                 server_manager=None,
                 assigned_port: Optional[int] = None,
                 **kwargs):
        """
        Initialize the reasoning generator.
        
        Args:
            model: Model name (e.g., "gemini", "qwen", "flan-t5-xxl")
            logger: Logger instance
            server_manager: LLM server manager for load balancing (optional, for Qwen/local LLMs)
            assigned_port: Pre-assigned port for parallel processing (overrides server_manager)
        """
        self.model = model  # Don't convert to lowercase to preserve exact model names
        self.logger = logger or logging.getLogger(__name__)
        self.server_manager = server_manager  # Store server manager for load balancing
        self.assigned_port = assigned_port  # Store assigned port for parallel processing
        
        # Initialize the appropriate generator and wrap with adapter
        raw_generator = self._initialize_generator()
        
        # Check if we should use parallel adapter for Qwen
        if assigned_port and "qwen" in model.lower():
            # Use parallel adapter for true parallelism
            try:
                from ..adapters.parallel_generator_adapter import create_parallel_generator_adapter
                self.generator = create_parallel_generator_adapter(
                    raw_generator, model, assigned_port=assigned_port, server_manager=server_manager
                )
                self.logger.info(f"✅ Using parallel adapter for {model} on port {assigned_port}")
            except ImportError:
                # Fallback to regular adapter
                self.generator = create_generator_adapter(raw_generator, model, server_manager=server_manager)
                self.logger.warning("Parallel adapter not available, using regular adapter")
        else:
            # Use regular adapter for all other cases
            self.generator = create_generator_adapter(raw_generator, model, server_manager=server_manager)
            if server_manager:
                self.logger.info(f"✅ Using adapter interface with server manager for model: {model}")
            else:
                self.logger.info(f"✅ Using adapter interface for model: {model}")
        
        self.logger.info(f"Initialized reasoning generator for model: {model}")
    
    def _initialize_generator(self):
        """Initialize the appropriate generator based on model type."""
        model_lower = self.model.lower()
        if "gemini" in model_lower:
            return GeminiGenerator(
                model="gemini-2.5-flash-lite",
                temperature=0.0,
                max_tokens=150,
                stop=["\n"]
            )
        elif "qwen" in model_lower:
            # Set environment variables for LLM server connection
            import os
            os.environ["LLM_SERVER_HOST"] = "localhost"
            os.environ["LLM_SERVER_PORT"] = str(int(os.getenv("LLM_SERVER_PORT", "8010")))
            # Use the actual model name passed (should contain the full Qwen model name)
            return LLMClientGenerator(
                model_name=self.model if "/" in self.model else "Qwen/Qwen2.5-3B-Instruct",
                temperature=0.0,
                max_length=512,  # Increased for complete answers
                eos_text="\n"
            )
        else:  # FLAN-T5
            # Set environment variables for LLM server connection
            import os
            os.environ["LLM_SERVER_HOST"] = "localhost"
            os.environ["LLM_SERVER_PORT"] = str(int(os.getenv("LLM_SERVER_PORT", "8010")))
            model_mapping = {
                "flan-t5-xl": "google/flan-t5-xl",
                "flan-t5-xxl": "google/flan-t5-xxl",
                "flan-t5-large": "google/flan-t5-large",
            }
            model_name = model_mapping.get(model_lower, "google/flan-t5-xxl")
            
            return LLMClientGenerator(
                model_name=model_name,
                temperature=0.0,
                max_length=100,
                eos_text="\n"
            )
    
    def generate(self,
                 prompt: str,
                 temperature: float = 0.0,
                 max_tokens: int = 150,
                 stop_sequences: Optional[List[str]] = None,
                 retry_count: int = 3) -> str:
        """
        Generate reasoning text.
        
        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            stop_sequences: Stop sequences
            retry_count: Number of retries on failure
            
        Returns:
            Generated text
        """
        if not prompt:
            self.logger.warning("Empty prompt provided to generator")
            return ""
        
        # Use adapter interface (handles parameter mapping automatically)
        
        # Retry logic
        last_error = None
        for attempt in range(retry_count):
            try:
                # Check if using adapter interface (has generate method with our signature)
                if hasattr(self.generator, 'generate') and hasattr(self.generator, 'retry_count'):
                    # This is an adapter - use the unified interface
                    result = self.generator.generate(
                        prompt=prompt,
                        temperature=temperature,
                        max_tokens=max_tokens,
                        stop_sequences=stop_sequences
                    )
                else:
                    # Direct generator - use the original method
                    result = self._call_generator(prompt)
                
                # Clean and return result
                cleaned_result = self._clean_generation(result)
                
                if cleaned_result:
                    return cleaned_result
                else:
                    self.logger.warning(f"Empty generation on attempt {attempt + 1}")
                    
            except Exception as e:
                last_error = e
                self.logger.warning(f"Generation attempt {attempt + 1} failed: {e}")
                if attempt < retry_count - 1:
                    time.sleep(1 * (attempt + 1))  # Exponential backoff
        
        # All attempts failed
        self.logger.error(f"All generation attempts failed. Last error: {last_error}")
        return ""
    
    def _call_generator(self, prompt: str) -> str:
        """Call the underlying generator - this is now handled by adapters."""
        # This method should not be called when using adapters
        # But kept as fallback for compatibility
        if hasattr(self.generator, 'generate_text_sequence'):
            # Gemini generator returns list of (text, score) tuples
            results = self.generator.generate_text_sequence(prompt)
            if results:
                return results[0][0]  # Return text from first result
            return ""
        elif hasattr(self.generator, 'generate'):
            # Fallback for other generator types
            return self.generator.generate(prompt)
        else:
            raise ValueError(f"Unknown generator interface for {self.model}: {type(self.generator)}")
    
    def _clean_generation(self, text: str) -> str:
        """Clean generated text."""
        if not text:
            return ""
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Qwen-specific cleaning for metadata artifacts
        if "qwen" in self.model.lower():
            import re
            # Remove reference patterns like "(Reference: Wikipedia Title: ...)"
            text = re.sub(r'\(Reference:\s*Wikipedia\s*Title:\s*[^)]*\)', '', text).strip()
            
            # Remove metadata patterns like "# METADATA: {...}"
            text = re.sub(r'#\s*METADATA:\s*\{[^}]*\}', '', text).strip()
            
            # Remove standalone Wikipedia Title references
            text = re.sub(r'Wikipedia\s*Title:\s*[^\n]*', '', text).strip()
            
            # Remove incomplete reference patterns
            text = re.sub(r'\(Reference:[^)]*$', '', text).strip()
        
        # Remove common prefixes that might be repeated
        prefixes_to_remove = ["A:", "Answer:", "Response:"]
        for prefix in prefixes_to_remove:
            if text.startswith(prefix):
                text = text[len(prefix):].strip()
        
        # Remove quotes if the entire text is quoted
        if len(text) > 2 and text[0] == text[-1] and text[0] in ['"', "'"]:
            text = text[1:-1]
        
        return text
    

    
    def batch_generate(self,
                       prompts: List[str],
                       temperature: float = 0.0,
                       max_tokens: int = 150) -> List[str]:
        """
        Generate for multiple prompts (sequential for now).
        
        Args:
            prompts: List of prompts
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            
        Returns:
            List of generated texts
        """
        results = []
        
        for prompt in prompts:
            result = self.generate(
                prompt=prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )
            results.append(result)
        
        return results 