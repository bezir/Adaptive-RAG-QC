#!/usr/bin/env python3
"""
Parallel Generator Adapter for IRCoT

This adapter provides true parallel generation for local LLM servers
by bypassing server manager locks and using direct port assignments.
"""

import logging
import os
import random
import requests
import threading
import time
from typing import List, Optional, Any
from .generator_adapter import GeneratorAdapter, GeminiGeneratorAdapter

logger = logging.getLogger(__name__)


class ParallelQwenAdapter(GeneratorAdapter):
    """
    Parallel-friendly adapter for Qwen that bypasses server manager locks.
    Uses direct HTTP calls to assigned ports for true parallelism.
    """
    
    def __init__(self, generator: Any, model_name: str = "", assigned_port: Optional[int] = None, server_manager=None):
        """
        Initialize with a pre-assigned port for this worker.
        
        Args:
            generator: The base generator (can be ignored for Qwen)
            model_name: Model name
            assigned_port: Pre-assigned port for this worker
            server_manager: LLMServerManager for port recovery
        """
        super().__init__(generator, model_name)
        self.assigned_port = assigned_port
        self.base_url = f"http://localhost:{assigned_port}" if assigned_port else "http://localhost:8010"
        self.server_manager = server_manager  # For port recovery
        self.failed_ports = set()  # Track failed ports to avoid re-using
        self.max_port_failures = 3  # Max failures before getting new port
        self.port_failure_count = 0
        
        # IRCoT-specific optimizations for parallel performance  
        self.request_count = 0
        self.consecutive_failures = 0
        
        logger.info(f"🚀 ParallelQwenAdapter initialized on port {assigned_port} with IRCoT optimizations")
        
    def generate(self, 
                prompt: str, 
                temperature: float = 0.0,
                max_tokens: int = 150,
                stop_sequences: Optional[List[str]] = None,
                **kwargs) -> str:
        """Generate using direct HTTP call to assigned port with intelligent pacing."""
        
        # NO REQUEST PACING: Let parallel workers run freely on different ports
        self.request_count += 1
        
        # Relaxed circuit breaker: Only skip after many failures to maintain parallelism
        if self.consecutive_failures >= 5:  # Increased from 2 to 5
            logger.warning(f"🔴 Circuit breaker OPEN for port {self.assigned_port} ({self.consecutive_failures} failures)")
            return ""
        
        # Realistic timeouts for complex multi-hop IRCoT reasoning  
        if max_tokens >= 1000:  # Long generation requests (IRCoT)
            base_timeout = 120  # 2 minutes for complex multi-hop reasoning
        elif len(prompt) > 1000:  # Complex IRCoT reasoning  
            base_timeout = 90   # 1.5 minutes for complex prompts  
        else:  # Medium/simple requests
            base_timeout = 60   # 1 minute for standard requests
            
        logger.debug(f"🧠 IRCoT req #{self.request_count} on port {self.assigned_port}: {len(prompt)} chars, timeout={base_timeout}s")
        
        # Retry logic for robustness
        max_retries = 3
        base_delay = 2.0
        
        for attempt in range(max_retries):
            try:
                # 🔧 FIXED: Use POST with JSON body instead of GET with query params
                payload = {
                    "prompt": prompt,
                    "max_tokens": max_tokens,  # Use max_tokens (not max_length) to match server
                    "temperature": temperature,
                    "do_sample": temperature > 0,
                    "top_k": 50,
                    "top_p": 1.0,
                    "num_return_sequences": 1,
                    "keep_prompt": False
                }
                
                # Use the adaptive timeout calculated earlier
                timeout = base_timeout
                
                # 🚀 FIXED: POST request with JSON body (matching server expectation)
                response = requests.post(
                    f"{self.base_url}/generate/",
                    json=payload,
                    timeout=timeout
                )
                
                if response.status_code == 200:
                    result = response.json()
                    # Handle Qwen response format
                    if 'generated_texts' in result and result['generated_texts']:
                        text = result['generated_texts'][0]
                    elif 'text' in result:
                        text = result['text']
                    else:
                        text = ""
                    
                    cleaned_text = self._clean_qwen_artifacts(text)
                    if cleaned_text:  # Only return if we got non-empty text
                        # SUCCESS: Reset circuit breaker
                        self.consecutive_failures = 0
                        logger.debug(f"✅ IRCoT success on port {self.assigned_port}, req #{self.request_count}")
                        return cleaned_text
                    else:
                        logger.warning(f"Empty generation from Qwen on attempt {attempt + 1}")
                        self.consecutive_failures += 1
                else:
                    logger.error(f"Qwen HTTP {response.status_code} on attempt {attempt + 1}")
                    self.consecutive_failures += 1
                    
            except requests.exceptions.Timeout:
                logger.warning(f"Qwen timeout on attempt {attempt + 1} (port {self.assigned_port})")
                self.consecutive_failures += 1
            except requests.exceptions.ConnectionError:
                logger.warning(f"Qwen connection error on attempt {attempt + 1} (port {self.assigned_port})")
                self.consecutive_failures += 1
            except Exception as e:
                logger.error(f"Qwen error on attempt {attempt + 1}: {e}")
                self.consecutive_failures += 1
            
            # Exponential backoff between retries
            if attempt < max_retries - 1:
                delay = base_delay * (2 ** attempt)
                time.sleep(delay)
        
        # All retries failed on current port
        logger.error(f"All {max_retries} attempts failed for Qwen on port {self.assigned_port}")
        
        # NON-BLOCKING PORT RECOVERY: Immediate port switch + background recovery
        if self.server_manager and self.port_failure_count < self.max_port_failures:
            self.port_failure_count += 1
            failed_port = self.assigned_port
            logger.warning(f"🔄 Port {failed_port} failed {self.port_failure_count} times, immediate recovery...")
            
            # Mark port as failed and start background recovery (non-blocking)
            self.failed_ports.add(failed_port)
            if self.port_failure_count <= 2:  # Only try restart for first 2 failures
                self._schedule_background_recovery(failed_port)
            
            # IMMEDIATE: Get a new healthy port quickly (no long waits for parallelism)
            new_server = self._get_healthy_port_with_wait(max_wait_time=10)  # Only wait 10s max
            if new_server:
                self.assigned_port = new_server
                self.base_url = f"http://localhost:{self.assigned_port}"
                logger.info(f"✅ Immediate recovery: {failed_port} → {self.assigned_port}")
                
                # Try the request immediately on the new port
                try:
                    response = requests.get(
                        f"{self.base_url}/generate/",
                        params={
                            "prompt": prompt,
                            "max_length": max_tokens,
                            "temperature": temperature,
                            "do_sample": temperature > 0,
                            "top_k": 50,
                            "top_p": 1.0,
                            "num_return_sequences": 1,
                            "keep_prompt": False
                        },
                        timeout=min(timeout, 60)  # Cap timeout at 60s to prevent long hangs
                    )
                    
                    if response.status_code == 200:
                        result = response.json()
                        if 'generated_texts' in result and result['generated_texts']:
                            text = result['generated_texts'][0]
                        elif 'text' in result:
                            text = result['text']
                        else:
                            text = ""
                        
                        cleaned_text = self._clean_qwen_artifacts(text)
                        if cleaned_text:
                            self.port_failure_count = 0  # Reset on success
                            self.consecutive_failures = 0  # Reset circuit breaker
                            logger.info(f"✅ Immediate recovery successful on port {self.assigned_port}")
                            return cleaned_text
                except Exception as e:
                    logger.error(f"New port {self.assigned_port} also failed: {e}")
            else:
                logger.error("❌ No healthy ports available even after waiting")
        
        return ""
    
    def _get_healthy_port(self) -> Optional[int]:
        """Get a new healthy port from server manager, avoiding failed ports."""
        if not self.server_manager:
            return None
            
        # Try to get a healthy server
        max_attempts = 5
        for _ in range(max_attempts):
            server_info = self.server_manager.get_available_server()
            if server_info and server_info.get('port') not in self.failed_ports:
                return server_info['port']
        
        logger.error(f"Failed to find healthy port after {max_attempts} attempts")
        return None
    
    def _get_healthy_port_with_wait(self, max_wait_time: int = 300) -> Optional[int]:
        """
        Get a healthy port, waiting if all ports are busy.
        
        Args:
            max_wait_time: Maximum time to wait for a port (seconds)
        
        Returns:
            Port number or None if timeout
        """
        start_time = time.time()
        wait_interval = 5  # Start with 5 seconds
        
        while time.time() - start_time < max_wait_time:
            port = self._get_healthy_port()
            if port:
                return port
            
            # No ports available, wait and try again
            logger.info(f"⏳ All ports busy, waiting {wait_interval}s before retry...")
            time.sleep(wait_interval)
            
            # Exponential backoff up to 30 seconds
            wait_interval = min(wait_interval * 1.5, 30)
        
        logger.error(f"❌ No healthy ports available after {max_wait_time}s")
        return None
    
    def _schedule_background_recovery(self, failed_port: int):
        """
        Schedule background recovery for a failed port (non-blocking).
        
        Args:
            failed_port: Port number that failed
        """
        def recover_port():
            try:
                logger.info(f"🔧 Background recovery started for port {failed_port}")
                success = self._restart_server(failed_port)
                if success:
                    # Remove from failed ports so it can be used again
                    self.failed_ports.discard(failed_port)
                    logger.info(f"✅ Background recovery completed for port {failed_port}")
                else:
                    logger.error(f"❌ Background recovery failed for port {failed_port}")
            except Exception as e:
                logger.error(f"❌ Background recovery error for port {failed_port}: {e}")
        
        # Start recovery in background thread
        recovery_thread = threading.Thread(target=recover_port, daemon=True)
        recovery_thread.start()
    
    def _restart_server(self, port: int) -> bool:
        """
        Restart the server on the given port.
        
        Args:
            port: Port of the server to restart
            
        Returns:
            True if restart successful, False otherwise
        """
        try:
            import subprocess
            import sys
            from pathlib import Path
            
            # Find the server startup script
            project_root = Path(__file__).parent.parent.parent  # Go up to Adaptive-RAG directory
            startup_script = project_root / "scaled_silver_labeling" / "scripts" / "server_startup.py"
            
            if not startup_script.exists():
                logger.error(f"Server startup script not found: {startup_script}")
                return False
            
            logger.info(f"🛑 Stopping failed server on port {port}")
            
            # Step 1: Kill any existing process on this port (force cleanup)
            try:
                # Find and kill process using the port
                kill_cmd = f"lsof -ti:{port} | xargs -r kill -9"
                subprocess.run(kill_cmd, shell=True, capture_output=True, timeout=10)
                time.sleep(2)  # Give it time to clean up
            except Exception as e:
                logger.warning(f"Could not kill existing process on port {port}: {e}")
            
            logger.info(f"🚀 Starting new server on port {port}")
            
            # Step 2: Start new server (determine GPU based on port)
            # Port mapping: 8010-8019 -> GPU 0, 8020-8029 -> GPU 1, etc.
            gpu_id = (port - 8010) // 10
            
            # Step 3: Start the server using the startup script
            start_cmd = [
                sys.executable,
                str(startup_script),
                "start_single",
                "--port", str(port),
                "--model", "Qwen/Qwen2.5-3B-Instruct",
                "--gpu-id", str(gpu_id)
            ]
            
            logger.debug(f"Restart command: {' '.join(start_cmd)}")
            
            # Start the server in background
            process = subprocess.Popen(
                start_cmd,
                cwd=str(project_root),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                preexec_fn=os.setsid  # Create new process group
            )
            
            # Wait 120 seconds for server and model to fully settle
            logger.info(f"⏳ Waiting 120 seconds for server and model to settle on port {port}...")
            time.sleep(120)
            
            # Now check if the port recovered
            try:
                response = requests.get(f"http://localhost:{port}/", timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Server restart successful on port {port}")
                    return True
                else:
                    logger.error(f"❌ Server restart failed - port {port} returned status {response.status_code}")
                    return False
            except requests.exceptions.RequestException as e:
                logger.error(f"❌ Server restart failed - port {port} not responding after 120s: {e}")
                return False
            return False
            
        except Exception as e:
            logger.error(f"❌ Server restart failed for port {port}: {e}")
            return False
    
    def _clean_qwen_artifacts(self, text: str) -> str:
        """Clean Qwen-specific metadata artifacts from generated text."""
        if not text:
            return ""
        
        import re
        # Remove reference patterns
        text = re.sub(r'\(Reference:\s*Wikipedia\s*Title:\s*[^)]*\)', '', text).strip()
        # Remove metadata patterns
        text = re.sub(r'#\s*METADATA:\s*\{[^}]*\}', '', text).strip()
        # Remove standalone Wikipedia Title references
        text = re.sub(r'Wikipedia\s*Title:\s*[^\n]*', '', text).strip()
        # Remove incomplete reference patterns
        text = re.sub(r'\(Reference:[^)]*$', '', text).strip()
        # Remove Q: and A: prefixes
        text = re.sub(r'^[QA]:\s*', '', text).strip()
        
        return text


def create_parallel_generator_adapter(generator: Any, 
                                    model_name: str = "", 
                                    assigned_port: Optional[int] = None,
                                    server_manager=None) -> GeneratorAdapter:
    """
    Factory function to create parallel-friendly generator adapters.
    
    Args:
        generator: The generator instance
        model_name: Model name
        assigned_port: Pre-assigned port for Qwen workers
        server_manager: Server manager for port recovery
        
    Returns:
        Appropriate GeneratorAdapter instance
    """
    model_lower = model_name.lower()
    
    if "qwen" in model_lower and assigned_port is not None:
        # Use parallel adapter for Qwen with assigned ports
        logger.info(f"🎯 Creating ParallelQwenAdapter for port {assigned_port}")
        return ParallelQwenAdapter(generator, model_name, assigned_port, server_manager)
    elif "gemini" in model_lower:
        # Gemini doesn't need parallelization fixes
        logger.info(f"🎯 Creating GeminiGeneratorAdapter for {model_name}")
        return GeminiGeneratorAdapter(generator, model_name)
    else:
        # Default to parallel adapter for other local models
        logger.info(f"🎯 Creating ParallelQwenAdapter (default) for {model_name}")
        return ParallelQwenAdapter(generator, model_name, assigned_port or 8010, server_manager)
