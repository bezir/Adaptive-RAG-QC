import json
import logging
import time
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import threading
import random
import requests
import queue
from concurrent.futures import ThreadPoolExecutor
import asyncio

# Get the logger for this module
logger = logging.getLogger(__name__)

@dataclass
class ServerInfo:
    """Server information with request queuing support"""
    id: str
    model: str
    host: str
    port: int
    gpu_id: int
    timeout: int = 30 
    total_requests: int = 0
    failed_requests: int = 0
    last_used: float = 0.0
    
    # Request queuing and synchronization
    request_semaphore: threading.Semaphore = None
    request_queue: queue.Queue = None
    is_processing: bool = False
    avg_response_time: float = 5.0
    consecutive_failures: int = 0
    
    # Circuit breaker functionality
    is_circuit_open: bool = False
    circuit_open_time: float = 0.0
    circuit_breaker_timeout: int = 60  # Time to wait before trying again (seconds)
    max_consecutive_failures: int = 3  # Max failures before opening circuit
    
    def __post_init__(self):
        """Initialize thread-safe components after dataclass creation"""
        if self.request_semaphore is None:
            self.request_semaphore = threading.Semaphore(1)  # Only 1 concurrent request per server
        if self.request_queue is None:
            self.request_queue = queue.Queue(maxsize=10)  # Max 10 queued requests per server
    
    def should_try_server(self) -> bool:
        """Check if this server should be tried based on circuit breaker state"""
        import time
        import logging
        if not self.is_circuit_open:
            return True
        
        # Check if circuit breaker timeout has passed
        if time.time() - self.circuit_open_time > self.circuit_breaker_timeout:
            self.is_circuit_open = False
            self.consecutive_failures = 0  # Reset for retry
            logger = logging.getLogger(__name__)
            logger.info(f"🟡 Circuit breaker HALF-OPEN for server {self.id} - attempting recovery")
            return True
        
        return False
    
    def mark_success(self):
        """Mark successful request - close circuit breaker if open"""
        import logging
        was_open = self.is_circuit_open
        self.consecutive_failures = 0
        self.is_circuit_open = False
        
        if was_open:
            logger = logging.getLogger(__name__)
            logger.info(f"🟢 Circuit breaker CLOSED for server {self.id} - server recovered")
    
    def mark_failure(self):
        """Mark failed request - potentially open circuit breaker"""
        import logging
        self.consecutive_failures += 1
        if self.consecutive_failures >= self.max_consecutive_failures and not self.is_circuit_open:
            self.is_circuit_open = True
            self.circuit_open_time = time.time()
            logger = logging.getLogger(__name__)
            logger.warning(f"🔴 Circuit breaker OPENED for server {self.id} after {self.consecutive_failures} failures - will retry in {self.circuit_breaker_timeout}s")


class LLMServerManager:
    """
    Simple LLM server manager for silver labeling.
    
    Manages servers without complex load balancing or health checking.
    """
    
    def __init__(self, config_path: str = None):
        """Initialize the server manager"""
        self.config_path = Path(config_path) if config_path else None
        self.servers: Dict[str, ServerInfo] = {}
        self.server_ids: List[str] = []
        self.lock = threading.Lock()
        
        # Load configuration if provided
        if self.config_path and self.config_path.exists():
            self._load_config()
        else:
            logger.info("No config file provided, initializing empty server list")
            self.servers = {}  # Keep it as dict, not list
        
        logger.info(f"Initialized LLM Server Manager with {len(self.servers)} servers")
    
    def _load_config(self):
        """Load server configuration from JSON file"""
        try:
            with open(self.config_path, 'r') as f:
                config = json.load(f)
            
            for server_config in config.get('llm_servers', []):
                server = ServerInfo(
                    id=server_config['id'],
                    model=server_config['model'],
                    host=server_config['host'],
                    port=server_config['port'],
                    gpu_id=server_config['gpu_id'],
                    timeout=server_config.get('timeout', 30)
                )
                # Ensure __post_init__ is called for thread-safe components
                server.__post_init__()
                self.servers[server.id] = server
                self.server_ids.append(server.id)
                
            logger.info(f"Loaded configuration for {len(self.servers)} servers")
            
        except Exception as e:
            logger.error(f"Failed to load server configuration: {e}")
            self.servers = {}  # Keep it as dict, not list
    
    def _load_server_health_status(self) -> Dict[str, Any]:
        """
        Load server health status from health monitor
        
        Returns:
            Dictionary with responsive_servers, hung_servers, etc.
        """
        status_file = Path('scaled_silver_labeling/logs/server_status.json')
        
        if not status_file.exists():
            logger.debug("No server health status file found, assuming all servers healthy")
            return {
                'responsive_servers': [s.port for s in self.servers.values()],
                'hung_servers': [],
                'failure_counts': {},
                'last_updated': 0
            }
        
        try:
            with open(status_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Failed to load server health status: {e}")
            return {
                'responsive_servers': [s.port for s in self.servers.values()],
                'hung_servers': [],
                'failure_counts': {},
                'last_updated': 0
            }
    
    def _filter_healthy_servers(self) -> List[str]:
        """
        Get list of healthy server IDs based on health monitoring data and circuit breaker state
        
        Returns:
            List of server IDs that are currently responsive and have working circuit breakers
        """
        health_status = self._load_server_health_status()
        responsive_ports = set(health_status.get('responsive_servers', []))
        hung_ports = set(health_status.get('hung_servers', []))
        
        healthy_server_ids = []
        circuit_broken_servers = 0
        
        for server_id, server in self.servers.items():
            # Skip servers that are explicitly marked as hung
            if server.port in hung_ports:
                logger.debug(f"Skipping server {server_id} - marked as hung")
                continue
            
            # Include servers that are either responsive OR not explicitly monitored
            # (fallback for when health monitoring is not available)
            if server.port in responsive_ports or not responsive_ports:
                # Additional check: exclude servers with too many recent failures
                if server.consecutive_failures >= 5:  # More aggressive failure threshold
                    logger.debug(f"Skipping server {server_id} - too many consecutive failures ({server.consecutive_failures})")
                    continue
                    
                if server.is_circuit_open:
                    circuit_broken_servers += 1
                    
                healthy_server_ids.append(server_id)
        
        # Log circuit breaker status for debugging
        if circuit_broken_servers > 0:
            logger.info(f"Health check: {len(healthy_server_ids)} servers available, {circuit_broken_servers} have circuit breakers open")
        
        if not healthy_server_ids:
            logger.warning("No healthy servers found from health monitoring, falling back to basic server list")
            # Still apply circuit breaker filtering even in fallback
            for server_id, server in self.servers.items():
                if server.consecutive_failures < 5:  # Only include servers with few failures
                    healthy_server_ids.append(server_id)
        
        logger.debug(f"Found {len(healthy_server_ids)} healthy servers out of {len(self.server_ids)} total")
        return healthy_server_ids

    def get_available_server(self, model_name: str = None) -> Optional[Dict[str, Any]]:
        """
        Get an available server (health-aware server selection with circuit breaker)
        
        Args:
            model_name: Optional model name filter
            
        Returns:
            Server info dictionary or None
        """
        if not self.servers:
            return None
        
        # Get healthy servers based on health monitoring
        with self.lock:
            healthy_server_ids = self._filter_healthy_servers()
            
            if not healthy_server_ids:
                logger.warning("No healthy servers available")
                return None
            
            # Filter out servers with open circuit breakers
            available_server_ids = []
            for server_id in healthy_server_ids:
                server = self.servers[server_id]
                if server.should_try_server():
                    available_server_ids.append(server_id)
                else:
                    logger.debug(f"Server {server_id} circuit breaker is open, skipping")
            
            if not available_server_ids:
                logger.warning("No available servers (all have circuit breakers open)")
                return None
            
            # Select a random available server
            server_id = random.choice(available_server_ids)
            server = self.servers[server_id]
            server.last_used = time.time()
            
            logger.debug(f"Selected available server {server.id} on port {server.port}")
            
            return {
                'id': server.id,
                'model': server.model,
                'host': server.host,
                'port': server.port,
                'gpu_id': server.gpu_id,
                'timeout': server.timeout
            }
    
    def release_server(self, server_id: str):
        """
        Release server back to pool (simplified - just update last used time)
        
        Args:
            server_id: Server ID to release
        """
        with self.lock:
            server = self.servers.get(server_id)
            if server:
                server.last_used = time.time()
    
    def process_request(self, request_data: Dict[str, Any], model_name: str = None, retry_count: int = 0) -> Dict[str, Any]:
        """
        Process a single request with per-server queuing to ensure only one request per server
        
        Args:
            request_data: Request data to process
            model_name: Optional model name
            retry_count: Current retry attempt (internal use)
            
        Returns:
            Server response data
        """
        server_info = self.get_available_server(model_name)
        
        if not server_info:
            # Check if all servers have circuit breakers open
            with self.lock:
                total_servers = len(self.servers)
                open_circuits = sum(1 for s in self.servers.values() if s.is_circuit_open)
                busy_servers = sum(1 for s in self.servers.values() if s.is_processing)
                
            # Return server failure response instead of raising exception
            if open_circuits > 0:
                logger.error(f"All servers unavailable - {open_circuits}/{total_servers} have circuit breakers open due to failures")
                return {
                    'answer': '__SERVER_UNAVAILABLE__',
                    'server_id': 'none',
                    'model': model_name or 'unknown',
                    'processing_time': 0.0,
                    'success': False,
                    'server_unavailable': True,
                    'error': f'All servers unavailable - {open_circuits}/{total_servers} servers have circuit breakers open due to failures'
                }
            else:
                logger.error(f"All servers unavailable - all servers unhealthy or busy (busy: {busy_servers}/{total_servers})")
                return {
                    'answer': '__SERVER_UNAVAILABLE__',
                    'server_id': 'none', 
                    'model': model_name or 'unknown',
                    'processing_time': 0.0,
                    'success': False,
                    'server_unavailable': True,
                    'error': f'All servers unavailable - all servers unhealthy or busy (busy: {busy_servers}/{total_servers})'
                }
        
        server = self.servers[server_info['id']]
        
        # Use semaphore to ensure only one request per server
        logger.debug(f"Acquiring semaphore for server {server.id}")
        acquired = server.request_semaphore.acquire(timeout=15)  # Wait up to 15s for server availability
        
        if not acquired:
            logger.warning(f"Failed to acquire semaphore for server {server.id} within 15s, trying different server")
            # Try to find a different server
            for attempt in range(3):
                server_info = self.get_available_server(model_name)
                if server_info and server_info['id'] != server.id:
                    server = self.servers[server_info['id']]
                    acquired = server.request_semaphore.acquire(timeout=10)
                    if acquired:
                        break
                time.sleep(1)
            
            if not acquired:
                logger.error("All servers busy - could not acquire processing slot within timeout")
                return {
                    'answer': '__SERVER_UNAVAILABLE__',
                    'server_id': 'none',
                    'model': model_name or 'unknown',
                    'processing_time': 0.0,
                    'success': False,
                    'server_unavailable': True,
                    'busy': True,
                    'error': 'All servers busy - could not acquire processing slot within timeout'
                }
        
        start_time = time.time()
        try:
            # Mark server as processing
            server.is_processing = True
            server.total_requests += 1
            
            logger.debug(f"Processing request on server {server.id} (queue size: {server.request_queue.qsize()})")
            
            # Make actual HTTP request to the server
            url = f"http://{server_info['host']}:{server_info['port']}/generate/"
            
            # Get parameters
            temperature = request_data.get("temperature", 0.0)
            do_sample = request_data.get("do_sample", True)
            
            # Fix parameter conflict: when temperature=0.0, do_sample should be False
            if temperature == 0.0:
                do_sample = False
            
            # Prepare request parameters (GET request with query params)
            params = {
                "prompt": request_data.get("prompt", ""),
                "max_length": request_data.get("max_tokens", 100),
                "temperature": temperature,
                "do_sample": do_sample
            }
            
            # Use server-specific timeout (increased from 5s)
            response = requests.get(
                url,
                params=params,
                timeout=server.timeout  # Use per-server timeout (default 60s)
            )
            
            if response.status_code == 200:
                result = response.json()
                processing_time = time.time() - start_time
                
                # Update server performance metrics
                server.avg_response_time = (server.avg_response_time * 0.8) + (processing_time * 0.2)
                server.mark_success()  # Reset failure count and close circuit breaker
                
                # Extract generated text from response
                generated_texts = result.get('generated_texts', [])
                answer = generated_texts[0] if generated_texts else ''
                
                # Clean Qwen responses - extract only assistant's response
                if model_name and 'qwen' in model_name.lower() and answer:
                    # Remove the prompt and extract only the assistant's response
                    if '<|im_start|>assistant' in answer:
                        # Split by assistant tag and take the part after it
                        assistant_parts = answer.split('<|im_start|>assistant')
                        if len(assistant_parts) > 1:
                            # Get the last assistant response
                            assistant_response = assistant_parts[-1]
                            # Remove any end tags
                            if '<|im_end|>' in assistant_response:
                                assistant_response = assistant_response.split('<|im_end|>')[0]
                            answer = assistant_response.strip()
                    elif 'assistant\n' in answer:
                        # Alternative format
                        assistant_parts = answer.split('assistant\n')
                        if len(assistant_parts) > 1:
                            answer = assistant_parts[-1].strip()
                
                # Check if we got an empty response and should retry
                if not answer.strip() and retry_count < 2:
                    logger.warning(f"Empty response received from {server.id}, retrying (attempt {retry_count + 1}/2)...")
                    time.sleep(1)  # Brief delay before retry
                    return self.process_request(request_data, model_name, retry_count + 1)
                
                response_data = {
                    'answer': answer,
                    'server_id': server.id,
                    'model': server.model,
                    'processing_time': processing_time,
                    'success': True,
                    'retry_count': retry_count,
                    'avg_response_time': server.avg_response_time
                }
                
                logger.debug(f"Request processed by {server.id} in {processing_time:.2f}s (avg: {server.avg_response_time:.2f}s)")
                return response_data
            else:
                raise Exception(f"Server returned status {response.status_code}: {response.text}")
            
        except Exception as e:
            server.failed_requests += 1
            server.mark_failure()  # Increment failures and potentially open circuit breaker
            logger.error(f"Request failed on {server.id}: {e} (consecutive failures: {server.consecutive_failures})")
            
            # Handle timeout errors with circuit breaker
            if "timeout" in str(e).lower():
                # Circuit breaker is already opened by mark_failure() above
                logger.error(f"Server {server.id} timed out after {server.timeout}s - circuit breaker {'opened' if server.is_circuit_open else 'still evaluating'}")
                
                # Return server failure response instead of raising exception for timeouts
                return {
                    'answer': '__SERVER_UNAVAILABLE__',
                    'server_id': server.id,
                    'model': server.model,
                    'processing_time': time.time() - start_time,
                    'success': False,
                    'server_unavailable': True,
                    'timeout': True,
                    'error': f'Server {server.id} timeout ({server.timeout}s) - failing to prevent congestion'
                }
            
            # Handle connection errors as server unavailable (don't retry)
            if ("connection refused" in str(e).lower() or 
                "connection error" in str(e).lower() or
                "failed to establish" in str(e).lower()):
                # Connection refused means server is not available
                logger.error(f"Server {server.id} connection refused - server unavailable")
                return {
                    'answer': '__SERVER_UNAVAILABLE__',
                    'server_id': server.id,
                    'model': server.model,
                    'processing_time': time.time() - start_time,
                    'success': False,
                    'server_unavailable': True,
                    'connection_refused': True,
                    'error': f'Server {server.id} connection refused - server unavailable: {str(e)}'
                }
            
            # Only retry other non-timeout errors once with exponential backoff
            should_retry = (
                retry_count < 1 and 
                server.consecutive_failures < 5 and  # Don't retry if server is consistently failing
                (
                    "rate" in str(e).lower() or
                    "502" in str(e) or  # Bad Gateway
                    "503" in str(e)     # Service Unavailable
                )
            )
            
            if should_retry:
                backoff_time = 1  # 1 second backoff for single retry
                logger.warning(f"Retryable error detected on {server.id}, retrying in {backoff_time}s (attempt {retry_count + 1}/1): {e}")
                time.sleep(backoff_time)
                return self.process_request(request_data, model_name, retry_count + 1)
            
            raise
            
        finally:
            # Always release server processing state and semaphore
            server.is_processing = False
            server.last_used = time.time()
            server.request_semaphore.release()
            logger.debug(f"Released semaphore for server {server.id}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get basic server statistics"""
        total_requests = sum(s.total_requests for s in self.servers.values())
        total_failures = sum(s.failed_requests for s in self.servers.values())
        
        return {
            'total_servers': len(self.servers),
            'total_requests': total_requests,
            'total_failures': total_failures,
            'success_rate': (total_requests - total_failures) / max(1, total_requests),
            'servers_by_gpu': self._get_gpu_stats()
        }
    
    def _get_gpu_stats(self) -> Dict[int, int]:
        """Get server count by GPU (simplified)"""
        gpu_stats = {}
        for server in self.servers.values():
            gpu_id = server.gpu_id
            gpu_stats[gpu_id] = gpu_stats.get(gpu_id, 0) + 1
        
        return gpu_stats
    
    def get_server_info(self, server_id: str) -> Optional[Dict[str, Any]]:
        """Get info about specific server"""
        server = self.servers.get(server_id)
        if not server:
            return None
        
        return {
            'id': server.id,
            'model': server.model,
            'host': server.host,
            'port': server.port,
            'gpu_id': server.gpu_id,
            'total_requests': server.total_requests,
            'failed_requests': server.failed_requests,
            'last_used': server.last_used
        }
    
    def list_servers(self) -> List[Dict[str, Any]]:
        """List all servers"""
        return [self.get_server_info(server_id) for server_id in self.server_ids]
    
    def get_healthy_server(self) -> Optional[Dict[str, Any]]:
        """
        Get a healthy server (compatible with generator_adapter expectations)
        
        Returns:
            Server info dictionary with health-aware selection or None
        """
        return self.get_available_server()


# Utility functions
def create_server_manager(config_path: str = None) -> LLMServerManager:
    """Create a server manager"""
    return LLMServerManager(config_path)

def get_default_config_path() -> str:
    """Get default configuration path"""
    return str(Path(__file__).parent.parent / "config" / "server_config.json") 