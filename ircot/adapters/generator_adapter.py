"""
Generator Adapter for IRCOT System

This adapter provides a unified interface for different generator types:
- Gemini generators (max_tokens, stop)
- LLMClientGenerator (max_length, eos_text)
- Future generator types

It handles parameter mapping, retry logic, and port availability automatically.
"""

import time
import random
import logging
from typing import List, Optional, Union, Any, Dict
from abc import ABC, abstractmethod

try:
    from ..utils.server_health import get_server_manager, get_available_server
except ImportError:
    # Fallback if server health utils not available
    def get_available_server():
        return "localhost", 8010
    def get_server_manager():
        return None

logger = logging.getLogger(__name__)


class GeneratorAdapter(ABC):
    """Abstract base class for generator adapters."""
    
    def __init__(self, generator: Any, model_name: str = ""):
        self.generator = generator
        self.model_name = model_name.lower()
        self.retry_count = 3
        self.retry_delay = 1.0
        
    @abstractmethod
    def generate(self, 
                prompt: str, 
                temperature: float = 0.0,
                max_tokens: int = 150,
                stop_sequences: Optional[List[str]] = None,
                **kwargs) -> str:
        """Generate text with unified interface."""
        pass
        
    def _exponential_backoff(self, attempt: int) -> float:
        """Calculate exponential backoff delay."""
        return self.retry_delay * (2 ** attempt) + random.uniform(0, 1)


class GeminiGeneratorAdapter(GeneratorAdapter):
    """Adapter for Gemini generators (preserves original Gemini interface)."""
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.0,
                max_tokens: int = 150,
                stop_sequences: Optional[List[str]] = None,
                **kwargs) -> str:
        """Generate using original Gemini interface to avoid breaking existing functionality."""
        
        for attempt in range(self.retry_count):
            try:
                # Update generator settings
                if hasattr(self.generator, 'temperature'):
                    self.generator.temperature = temperature
                if hasattr(self.generator, 'max_tokens'):
                    self.generator.max_tokens = max_tokens
                if stop_sequences and hasattr(self.generator, 'stop'):
                    self.generator.stop = stop_sequences
                
                # Use the method that Gemini generator actually has
                if hasattr(self.generator, 'generate_text_sequence'):
                    # Gemini's primary method - returns list of (text, score) tuples
                    results = self.generator.generate_text_sequence(prompt)
                    if results and len(results) > 0:
                        result = results[0][0]  # Get text from first result
                    else:
                        result = ""
                elif hasattr(self.generator, 'generate'):
                    # Fallback method
                    result = self.generator.generate(prompt)
                else:
                    raise ValueError(f"Gemini generator has no known generate method: {type(self.generator)}")
                
                if result and result.strip():
                    return result.strip()
                    
            except Exception as e:
                logger.warning(f"🔄 Gemini generation attempt {attempt + 1} failed: {e}")
                if attempt < self.retry_count - 1:
                    delay = self._exponential_backoff(attempt)
                    logger.info(f"⏱️ Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    
        logger.error("❌ All Gemini generation attempts failed")
        return ""


class LLMClientGeneratorAdapter(GeneratorAdapter):
    """Adapter for LLMClientGenerator (max_length, eos_text interface) with server health management."""
    
    def __init__(self, generator: Any, model_name: str = "", server_manager=None):
        super().__init__(generator, model_name)
        # Use provided server_manager or fallback to get_server_manager()
        self.server_manager = server_manager or get_server_manager()
    
    def generate(self, 
                prompt: str, 
                temperature: float = 0.0,
                max_tokens: int = 150,
                stop_sequences: Optional[List[str]] = None,
                **kwargs) -> str:
        """Generate using LLMClientGenerator interface with server health management."""
        
        for attempt in range(self.retry_count):
            try:
                # Update generator settings using correct parameter names
                if hasattr(self.generator, 'temperature'):
                    self.generator.temperature = temperature
                if hasattr(self.generator, 'max_length'):
                    self.generator.max_length = max_tokens
                if stop_sequences and hasattr(self.generator, 'eos_text'):
                    self.generator.eos_text = stop_sequences[0] if stop_sequences else "\n"
                    
                # Generate with server health awareness
                result = self._generate_with_health_check(prompt)
                
                if result and result.strip():
                    return result.strip()
                    
            except Exception as e:
                logger.warning(f"🔄 LLM Client generation attempt {attempt + 1} failed: {e}")
                
                # Mark current server as potentially failed
                if self.server_manager and hasattr(self.generator, 'server_url'):
                    try:
                        port = int(self.generator.server_url.split(':')[-1].split('/')[0])
                        self.server_manager.mark_server_failed(port)
                    except:
                        pass
                
                if attempt < self.retry_count - 1:
                    delay = self._exponential_backoff(attempt)
                    logger.info(f"⏱️ Retrying in {delay:.2f}s...")
                    time.sleep(delay)
                    
                    # Try to get a new healthy server for retry
                    self._try_new_server()
                    
        logger.error("❌ All LLM Client generation attempts failed")
        return ""
    
    def _generate_with_health_check(self, prompt: str) -> str:
        """Generate with server health checking and load balancing."""
        try:
            # Use server manager for load balancing if available
            if self.server_manager:
                # ServerHealthManager uses get_healthy_server(), LLMServerManager uses get_available_server()
                if hasattr(self.server_manager, 'get_healthy_server'):
                    server_info = self.server_manager.get_healthy_server()
                    logger.debug(f"🔧 DEBUG: Got ServerHealthManager server_info type: {type(server_info)}")
                elif hasattr(self.server_manager, 'get_available_server'):
                    server_info = self.server_manager.get_available_server()
                    logger.debug(f"🔧 DEBUG: Got LLMServerManager server_info type: {type(server_info)}")
                else:
                    server_info = None
                    logger.debug("🔧 DEBUG: Unknown server manager type")
                if server_info:
                    # Temporarily update the generator's connection to use the selected server
                    original_host = getattr(self.generator, 'host', None)
                    original_port = getattr(self.generator, 'port', None)
                    
                    # Handle both ServerInfo dataclass (ServerHealthManager) and dict (LLMServerManager)
                    if isinstance(server_info, dict):
                        # LLMServerManager returns dict
                        host = server_info['host']
                        port = server_info['port']
                        server_id = server_info.get('id', f"port_{port}")
                        logger.debug(f"🔧 DEBUG: Using dict server_info - host: {host}, port: {port}")
                    else:
                        # ServerHealthManager returns ServerInfo dataclass
                        host = server_info.host
                        port = server_info.port
                        server_id = f"{host}:{port}"
                        logger.debug(f"🔧 DEBUG: Using dataclass server_info - host: {host}, port: {port}")
                    
                    # Update LLMClientGenerator to use the selected server
                    if hasattr(self.generator, 'host'):
                        self.generator.host = host
                        logger.debug(f"🔧 DEBUG: Updated generator.host to {host}")
                    if hasattr(self.generator, 'port'):
                        self.generator.port = port
                        logger.debug(f"🔧 DEBUG: Updated generator.port to {port}")
                    
                    # Also update environment variables for LLMClientGenerator
                    import os
                    os.environ["LLM_SERVER_HOST"] = host
                    os.environ["LLM_SERVER_PORT"] = str(port)
                    logger.debug(f"🔧 DEBUG: Updated env vars - LLM_SERVER_HOST={host}, LLM_SERVER_PORT={port}")
                    
                    logger.debug(f"🎯 Using server {server_id} ({host}:{port}) for load balancing")
                    
                    try:
                        # Generate with selected server
                        result = self._perform_generation(prompt)
                        logger.debug(f"🔧 DEBUG: Generation successful with server {server_id}")
                        
                        # Mark server as successful
                        if hasattr(self.server_manager, 'mark_server_success'):
                            # Use port from extracted values
                            self.server_manager.mark_server_success(port)
                        
                        return result
                    except Exception as e:
                        logger.debug(f"🔧 DEBUG: Generation failed with server {server_id}: {type(e).__name__}: {str(e)}")
                        # Mark server as failed
                        if hasattr(self.server_manager, 'mark_server_failure'):
                            # Use port from extracted values
                            self.server_manager.mark_server_failure(port)
                            logger.debug(f"🔧 DEBUG: Marked server {server_id} as failed")
                        raise e
                    finally:
                        # Restore original connection settings
                        if original_host and hasattr(self.generator, 'host'):
                            self.generator.host = original_host
                        if original_port and hasattr(self.generator, 'port'):
                            self.generator.port = original_port
                else:
                    logger.warning("⚠️ No available servers from server manager, falling back to default connection")
                    return self._perform_generation(prompt)
            else:
                # No server manager, use direct connection
                return self._perform_generation(prompt)
                
        except Exception as e:
            if "connection" in str(e).lower() or "timeout" in str(e).lower():
                logger.warning(f"🔌 Connection issue detected: {e}")
                # Try to switch to a healthy server if using server manager
                if self.server_manager and self._try_new_server():
                    logger.info("🔄 Retrying with new server...")
                    return self._generate_with_health_check(prompt)  # Recursive call with health check
            raise e
    
    def _perform_generation(self, prompt: str) -> str:
        """Perform the actual generation request."""
        # Use the correct method for LLMClientGenerator
        if hasattr(self.generator, 'generate_text_sequence'):
            # LLMClientGenerator returns list of (text, score) tuples
            results = self.generator.generate_text_sequence(prompt)
            if results and len(results) > 0:
                raw_text = results[0][0]  # Return the text from first result
                return self._clean_qwen_artifacts(raw_text)
            return ""
        elif hasattr(self.generator, 'generate'):
            raw_text = self.generator.generate(prompt)
            return self._clean_qwen_artifacts(raw_text)
        else:
            raise ValueError(f"Unknown generator interface: {type(self.generator)}")
    
    def _clean_qwen_artifacts(self, text: str) -> str:
        """Clean Qwen-specific metadata artifacts from generated text."""
        if not text or "qwen" not in self.model_name.lower():
            return text
        
        import re
        # Remove reference patterns like "(Reference: Wikipedia Title: ...)"
        text = re.sub(r'\(Reference:\s*Wikipedia\s*Title:\s*[^)]*\)', '', text).strip()
        
        # Remove metadata patterns like "# METADATA: {...}"
        text = re.sub(r'#\s*METADATA:\s*\{[^}]*\}', '', text).strip()
        
        # Remove standalone Wikipedia Title references
        text = re.sub(r'Wikipedia\s*Title:\s*[^\n]*', '', text).strip()
        
        # Remove incomplete reference patterns
        text = re.sub(r'\(Reference:[^)]*$', '', text).strip()
        
        # Remove Q: and A: prefixes that might leak from prompts
        text = re.sub(r'^[QA]:\s*', '', text).strip()
        
        return text
    
    def _try_new_server(self) -> bool:
        """Try to switch to a new healthy server."""
        try:
            if not self.server_manager:
                logger.debug("🔧 DEBUG: No server manager for server switching")
                return False
                
            # Get a healthy server - handle both manager types
            if hasattr(self.server_manager, 'get_healthy_server'):
                server_info = self.server_manager.get_healthy_server()
                logger.debug(f"🔧 DEBUG: _try_new_server got ServerHealthManager server_info: {type(server_info)}")
            elif hasattr(self.server_manager, 'get_available_server'):
                server_info = self.server_manager.get_available_server()
                logger.debug(f"🔧 DEBUG: _try_new_server got LLMServerManager server_info: {type(server_info)}")
            else:
                logger.debug("🔧 DEBUG: Server manager has no server retrieval methods")
                return False
                
            if not server_info:
                logger.warning("⚠️ No healthy servers available")
                return False
            
            # Handle both dict and dataclass server_info
            if isinstance(server_info, dict):
                host = server_info['host']
                port = server_info['port']
                logger.debug(f"🔧 DEBUG: _try_new_server using dict server - host: {host}, port: {port}")
            else:
                host = server_info.host
                port = server_info.port
                logger.debug(f"🔧 DEBUG: _try_new_server using dataclass server - host: {host}, port: {port}")
            
            # Update generator's server URL if possible
            new_url = f"http://{host}:{port}"
            if hasattr(self.generator, 'server_url'):
                old_url = self.generator.server_url
                self.generator.server_url = new_url
                logger.info(f"🔄 Switched from {old_url} to {new_url}")
                return True
            elif hasattr(self.generator, 'base_url'):
                old_url = self.generator.base_url
                self.generator.base_url = new_url
                logger.info(f"🔄 Switched from {old_url} to {new_url}")
                return True
            else:
                logger.debug("⚠️ Cannot update server URL - generator doesn't support it")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Failed to switch server: {e}")
            return False


def create_generator_adapter(generator: Any, model_name: str = "", server_manager=None) -> GeneratorAdapter:
    """
    Factory function to create the appropriate generator adapter.
    
    Args:
        generator: The generator instance
        model_name: Model name to determine adapter type
        server_manager: LLM server manager for load balancing (optional, for Qwen/local LLMs)
        
    Returns:
        Appropriate GeneratorAdapter instance
    """
    model_lower = model_name.lower()
    
    # Check generator type and model name - be specific about Qwen vs Gemini
    if "qwen" in model_lower or "flan" in model_lower:
        # Only use LLM Client adapter for Qwen and FLAN models with server manager for load balancing
        if server_manager:
            logger.info(f"🎯 Creating LLM Client adapter with server manager for model: {model_name}")
        else:
            logger.info(f"🎯 Creating LLM Client adapter (no server manager) for model: {model_name}")
        return LLMClientGeneratorAdapter(generator, model_name, server_manager=server_manager)
    elif "gemini" in model_lower:
        # Always use Gemini adapter for Gemini models (no server manager needed)
        logger.info(f"🎯 Creating Gemini adapter for model: {model_name}")
        return GeminiGeneratorAdapter(generator, model_name)
    else:
        # Default to Gemini adapter for unknown models to maintain compatibility
        logger.info(f"🎯 Creating Gemini adapter (default) for model: {model_name}")
        return GeminiGeneratorAdapter(generator, model_name)

