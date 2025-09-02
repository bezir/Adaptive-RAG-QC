#!/usr/bin/env python3
"""
Gemini Server Adapter for Scaled Silver Labeling

This module provides an adapter that allows the labeling system
to work with Gemini models by mimicking the LLM server interface.
"""

import os
import json
import logging
import time
from typing import Dict, List, Optional, Any
from pathlib import Path
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from commaqa.models.gemini_generator import GeminiGenerator
from scaled_silver_labeling.servers.llm_server_manager import LLMServerManager, ServerInfo

logger = logging.getLogger(__name__)

class GeminiServerAdapter:
    """
    Adapter for interacting with a local Gemini server.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", rate_limit_delay: float = 0.1, timeout: int = 30):
        self.model_name = model_name
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        self.request_count = 0
        self.last_request_time = 0
        
        # Thread-safe lock for rate limiting
        self.rate_limit_lock = threading.Lock()
        
        self.generator = GeminiGenerator(
            model=model_name,
            temperature=0.0,
            max_tokens=300,
            top_p=1.0,
            retry_after_n_seconds=5,
            thinking_budget=0
        )
        
        # Create mock server info for compatibility
        self.server_info = ServerInfo(
            id="gemini_server_1",
            model=model_name,
            host="api.gemini.com",
            port=443,
            gpu_id=-1,
            timeout=timeout
        )
        
        # Initialize request statistics
        self.server_info.total_requests = 0
        self.server_info.failed_requests = 0
        self.server_info.last_used = 0
        
        logger.info(f"Initialized Gemini server adapter with model: {model_name}")
        logger.info(f"  Rate limit delay: {rate_limit_delay}s")
        logger.info(f"  Timeout: {timeout}s")
    
    def _respect_rate_limit(self):
        """Ensure rate limiting between requests"""
        with self.rate_limit_lock:
            current_time = time.time()
            if self.last_request_time > 0:
                elapsed = current_time - self.last_request_time
                if elapsed < self.rate_limit_delay:
                    sleep_time = self.rate_limit_delay - elapsed
                    time.sleep(sleep_time)
            self.last_request_time = time.time()
    
    def process_request(self, request_data: Dict[str, Any], retry_count: int = 0) -> Dict[str, Any]:
        """
        Process a request using Gemini API
        
        Args:
            request_data: Request data containing prompt and parameters
            retry_count: Current retry attempt (internal use)
            
        Returns:
            Response in the format expected by the scaled labeling system
        """
        # Respect rate limiting
        self._respect_rate_limit()
        
        try:
            prompt = request_data.get("prompt", "")
            
            # Generate response using Gemini
            start_time = time.time()
            response = self.generator.generate_text_sequence(prompt)
            processing_time = time.time() - start_time
            
            # Extract text from response
            generated_text = ""
            if response and len(response) > 0:
                generated_text = response[0][0]  # Get first response text (tuple format)
                
                # Check for API failure indicators
                if generated_text and generated_text.strip() in ["__RESOURCE_EXHAUSTED__", "__API_FAILED__"]:
                    logger.warning(f"API failure detected: {generated_text.strip()} - marking sample for discard")
                    return {
                        'answer': '__RESOURCE_EXHAUSTED__',
                        'server_id': self.server_info.id,
                        'model': self.model_name,
                        'processing_time': processing_time,
                        'success': False,
                        'resource_exhausted': True,
                        'error': 'Resource exhausted - sample should be discarded'
                    }
                
                # Clean up the response (preserve all patterns for extraction)
                if generated_text:
                    generated_text = generated_text.strip()
                    
                    # No cleanup of answer patterns - let extraction handle all formatting
            
            # Check if we got an empty response and should retry
            if not generated_text and retry_count < 2:
                logger.warning(f"Empty response received, retrying (attempt {retry_count + 1}/2)...")
                time.sleep(1)  # Brief delay before retry
                return self.process_request(request_data, retry_count + 1)
            
            # Update server stats
            self.server_info.total_requests += 1
            self.server_info.last_used = time.time()
            self.request_count += 1
            
            # Log request stats periodically
            if self.request_count % 10 == 0:
                logger.debug(f"Processed {self.request_count} requests, avg time: {processing_time:.2f}s")
            
            # Return in the EXACT format expected by the labeling system
            return {
                'answer': generated_text,  # This is the key field that the labeling system expects
                'server_id': self.server_info.id,
                'model': self.model_name,
                'processing_time': processing_time,
                'success': True
            }
            
        except Exception as e:
            logger.error(f"Gemini request failed: {e}")
            self.server_info.failed_requests += 1
            
            # Check for resource exhausted error and mark for discard
            exception_str = str(e).lower()
            if "resource exhausted" in exception_str or "quota exceeded" in exception_str or "429" in exception_str:
                logger.warning(f"Resource exhausted error detected - sample will be discarded: {e}")
                return {
                    'answer': '__RESOURCE_EXHAUSTED__',
                    'server_id': self.server_info.id,
                    'model': self.model_name,
                    'processing_time': 0.0,
                    'success': False,
                    'resource_exhausted': True,
                    'error': f'Resource exhausted - sample should be discarded: {str(e)}'
                }
            
            # For specific errors, try to retry once if we haven't already
            should_retry = (
                retry_count < 2 and (
                    "AutoTokenizer" in str(e) or
                    "timeout" in str(e).lower() or
                    "connection" in str(e).lower() or
                    "rate" in str(e).lower()
                )
            )
            
            if should_retry:
                logger.warning(f"Retryable error detected, retrying (attempt {retry_count + 1}/2): {e}")
                time.sleep(1)  # Brief delay before retry
                return self.process_request(request_data, retry_count + 1)
            
            # Return error response in expected format
            return {
                'answer': '',  # Empty answer for failures
                'server_id': self.server_info.id,
                'model': self.model_name,
                'processing_time': 0.0,
                'success': False,
                'error': str(e)
            }
    
    def get_server_info(self) -> Dict[str, Any]:
        """Get server information"""
        return {
            "id": self.server_info.id,
            "model": self.server_info.model,
            "host": self.server_info.host,
            "port": self.server_info.port,
            "status": "active",
            "total_requests": getattr(self.server_info, 'total_requests', 0),
            "failed_requests": getattr(self.server_info, 'failed_requests', 0),
            "rate_limit_delay": self.rate_limit_delay,
            "timeout": self.timeout
        }

class GeminiAPIServerManager(LLMServerManager):
    """
    Manages multiple parallel Gemini API servers.
    """
    def __init__(self, model_name: str = "gemini-2.5-flash-lite", max_parallel_calls: int = 5,
                 start_port: int = 8010, timeout: int = 30, log_dir: str = "gemini_server_logs", 
                 rate_limit_delay: float = 0.1):
        if not model_name:
            raise ValueError("Model name cannot be empty.")
        self.max_parallel_calls = max_parallel_calls
        self.rate_limit_delay = rate_limit_delay
        self.timeout = timeout
        
        self.gemini_adapter = GeminiServerAdapter(
            model_name=model_name,
            rate_limit_delay=rate_limit_delay,
            timeout=timeout
        )
        
        self.servers = {
            "gemini_server_1": self.gemini_adapter.server_info
        }
        self.server_ids = ["gemini_server_1"]
        
        # Initialize lock for thread safety
        self.lock = threading.Lock()
        
        # Create thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_calls)
        
        logger.info(f"Initialized Gemini LLM Server Manager with model: {model_name}")
        logger.info(f"  Max parallel calls: {max_parallel_calls}")
        logger.info(f"  Rate limit delay: {rate_limit_delay}s")
        logger.info(f"  Timeout: {timeout}s")
    
    def get_available_server(self, model_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get available server (always returns Gemini adapter)
        
        Args:
            model_name: Optional model name filter
            
        Returns:
            Server information dictionary
        """
        return self.gemini_adapter.get_server_info()
    
    def process_request(self, request_data: Dict[str, Any], model_name: str = None) -> Dict[str, Any]:
        """
        Process request using Gemini
        
        Args:
            request_data: Request data
            model_name: Optional model name
            
        Returns:
            Response from Gemini
        """
        response = self.gemini_adapter.process_request(request_data)
        
        # Log the response for debugging
        logger.debug(f"Gemini response: {response}")
        
        return response
    
    def process_batch_requests(self, requests: List[Dict[str, Any]], model_name: str = None) -> List[Dict[str, Any]]:
        """
        Process multiple requests in parallel
        
        Args:
            requests: List of request data dictionaries
            model_name: Optional model name
            
        Returns:
            List of responses from Gemini
        """
        logger.info(f"Processing batch of {len(requests)} requests with {self.max_parallel_calls} parallel calls")
        
        # Submit all requests to thread pool
        future_to_request = {
            self.executor.submit(self.process_request, request, model_name): request 
            for request in requests
        }
        
        # Collect results as they complete
        results = []
        for future in as_completed(future_to_request):
            try:
                result = future.result(timeout=self.timeout)
                results.append(result)
                
                # Log progress
                if len(results) % 10 == 0:
                    logger.info(f"Completed {len(results)}/{len(requests)} requests")
                    
            except Exception as e:
                logger.error(f"Batch request failed: {e}")
                
                # Check for resource exhausted error and mark for discard
                exception_str = str(e).lower()
                if "resource exhausted" in exception_str or "quota exceeded" in exception_str or "429" in exception_str:
                    logger.warning(f"Resource exhausted error in batch processing - sample will be discarded: {e}")
                    results.append({
                        'answer': '__RESOURCE_EXHAUSTED__',
                        'server_id': self.gemini_adapter.server_info.id,
                        'model': self.gemini_adapter.model_name,
                        'processing_time': 0.0,
                        'success': False,
                        'resource_exhausted': True,
                        'error': f'Resource exhausted - sample should be discarded: {str(e)}'
                    })
                else:
                    results.append({
                        'answer': '',
                        'server_id': self.gemini_adapter.server_info.id,
                        'model': self.gemini_adapter.model_name,
                        'processing_time': 0.0,
                        'success': False,
                        'error': str(e)
                    })
        
        return results
    
    def get_server_status(self) -> Dict[str, Any]:
        """Get status of all servers"""
        return {
            "gemini_server_1": self.gemini_adapter.get_server_info(),
            "parallel_config": {
                "max_parallel_calls": self.max_parallel_calls,
                "rate_limit_delay": self.rate_limit_delay,
                "timeout": self.timeout
            }
        }
    
    def shutdown_all(self):
        """Shutdown all servers and cleanup resources"""
        logger.info("Shutting down Gemini server manager...")
        
        # Shutdown thread pool
        if hasattr(self, 'executor'):
            self.executor.shutdown(wait=True)
            logger.info("Thread pool shutdown completed")
        
        logger.info("Shutdown complete")
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        adapter_info = self.gemini_adapter.get_server_info()
        
        return {
            "total_requests": adapter_info.get("total_requests", 0),
            "failed_requests": adapter_info.get("failed_requests", 0),
            "success_rate": (adapter_info.get("total_requests", 0) - adapter_info.get("failed_requests", 0)) / max(adapter_info.get("total_requests", 1), 1),
            "parallel_calls": self.max_parallel_calls,
            "rate_limit_delay": self.rate_limit_delay,
            "timeout": self.timeout
        } 