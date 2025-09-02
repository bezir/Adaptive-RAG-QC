#!/usr/bin/env python3
"""
LLM Server Startup Script

This script manages LLM servers for scaled silver labeling:
- Generates configuration files for multiple servers across GPUs
- Starts and stops LLM servers using the existing FastAPI infrastructure
- Supports load balancing across multiple GPUs

Usage:
    # Generate config for 10 GPUs with 4 models per GPU
    python server_startup.py generate-config --gpu_ids 0,1,2,3,4,5,6,7,8,9 --models_per_gpu 4 --model flan-t5-xl --config /tmp/flan_t5_xl_config.json
    
    # Start servers using the config
    python server_startup.py start --config /tmp/flan_t5_xl_config.json --model flan-t5-xl
    
    # Stop all servers from config
    python server_startup.py stop --config /tmp/flan_t5_xl_config.json
    
    # Check server status
    python server_startup.py status --config /tmp/flan_t5_xl_config.json
"""

import argparse
import json
import os
import subprocess
import sys
import time
import signal
import logging
import requests
import socket
import atexit
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up to Adaptive-RAG directory
sys.path.insert(0, str(project_root))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ServerProcess:
    """Information about a running server process"""
    port: int
    gpu_id: int
    model: str
    process: subprocess.Popen
    pid: int

class LLMServerStarter:
    """Manages starting and stopping LLM servers"""
    
    def __init__(self):
        self.running_servers: Dict[int, ServerProcess] = {}
        self.server_processes: List[subprocess.Popen] = []
        self._lock = threading.RLock()  # Reentrant lock for thread safety
        self._cleanup_timer = None
        
        # Set up signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
        
        # Register cleanup on exit and start periodic cleanup
        atexit.register(self._emergency_cleanup)
        self._start_periodic_cleanup()
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down servers...")
        self.stop_all_servers()
        sys.exit(0)
    
    @contextmanager
    def _thread_safe_operation(self):
        """Context manager for thread-safe operations on shared data structures"""
        self._lock.acquire()
        try:
            yield
        finally:
            self._lock.release()
    
    def _start_periodic_cleanup(self):
        """Start periodic cleanup of dead processes"""
        self._cleanup_dead_processes()
        # Schedule next cleanup in 30 seconds
        self._cleanup_timer = threading.Timer(60.0, self._start_periodic_cleanup)
        self._cleanup_timer.daemon = True
        self._cleanup_timer.start()
    
    def _cleanup_dead_processes(self):
        """Clean up processes that have died without proper cleanup"""
        with self._thread_safe_operation():
            dead_ports = []
            
            for port, server in list(self.running_servers.items()):
                if server.process.poll() is not None:
                    logger.warning(f"Found dead process on port {port}, cleaning up")
                    dead_ports.append(port)
                    
                    # Clean up zombie process
                    try:
                        server.process.wait(timeout=0.1)
                    except subprocess.TimeoutExpired:
                        pass
            
            # Remove dead servers from tracking
            for port in dead_ports:
                if port in self.running_servers:
                    server = self.running_servers[port]
                    self._cleanup_server_tracking_unsafe(port, server)
    
    def _cleanup_server_tracking_unsafe(self, port: int, server: ServerProcess):
        """Clean up server tracking data structures (call within lock)"""
        if port in self.running_servers:
            del self.running_servers[port]
        if server.process in self.server_processes:
            self.server_processes.remove(server.process)
    
    def _emergency_cleanup(self):
        """Emergency cleanup on script exit"""
        try:
            if self._cleanup_timer:
                self._cleanup_timer.cancel()
            self.stop_all_servers()
        except Exception as e:
            print(f"Emergency cleanup error: {e}", file=sys.stderr)
    
    def check_port_available(self, port: int, host: str = 'localhost') -> bool:
        """Check if a port is available for binding"""
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
                return True
        except OSError as e:
            logger.debug(f"Port {port} not available: {e}")
            return False
    
    def validate_ports_available(self, config: Dict[str, Any]) -> List[int]:
        """Validate that all configured ports are available"""
        servers = config.get('llm_servers', [])
        unavailable_ports = []
        
        for server in servers:
            port = server['port']
            if not self.check_port_available(port):
                unavailable_ports.append(port)
        
        if unavailable_ports:
            logger.error(f"Ports already in use: {unavailable_ports}")
        
        return unavailable_ports
    
    def generate_config(self, gpu_ids: List[int], models_per_gpu: int, model: str, 
                       start_port: int = 8010, timeout: int = 500) -> Dict[str, Any]:
        """
        Generate server configuration for multiple GPUs
        
        Args:
            gpu_ids: List of GPU IDs to use
            models_per_gpu: Number of model instances per GPU
            model: Model name to use
            start_port: Starting port number
            timeout: Request timeout in seconds
            
        Returns:
            Configuration dictionary
        """
        servers = []
        current_port = start_port
        
        for gpu_id in gpu_ids:
            for i in range(models_per_gpu):
                server_config = {
                    "id": f"server_{current_port}",
                    "model": model,
                    "host": "localhost",
                    "port": current_port,
                    "gpu_id": gpu_id,
                    "timeout": timeout
                }
                servers.append(server_config)
                current_port += 1
        
        config = {
            "llm_servers": servers,
            "metadata": {
                "total_servers": len(servers),
                "gpu_count": len(gpu_ids),
                "models_per_gpu": models_per_gpu,
                "model": model,
                "port_range": f"{start_port}-{current_port-1}",
                "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
        }
        
        logger.info(f"Generated config for {len(servers)} servers:")
        logger.info(f"  GPUs: {gpu_ids}")
        logger.info(f"  Models per GPU: {models_per_gpu}")
        logger.info(f"  Model: {model}")
        logger.info(f"  Port range: {start_port}-{current_port-1}")
        
        return config
    
    def save_config(self, config: Dict[str, Any], config_path: str):
        """Save configuration to JSON file"""
        os.makedirs(os.path.dirname(config_path), exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
    
    def load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from JSON file"""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        return config
    
    def start_server(self, server_config: Dict[str, Any], model: str) -> Optional[ServerProcess]:
        """
        Start a single LLM server instance
        
        Args:
            server_config: Server configuration
            model: Model name to override config model
            
        Returns:
            ServerProcess instance if successful, None otherwise
        """
        port = server_config['port']
        gpu_id = server_config['gpu_id']
        model_name = model or server_config['model']
        
        # Validate model name
        valid_models = [
            "flan-t5-base", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl",
            "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"
        ]
        
        if model_name not in valid_models:
            logger.error(f"Invalid model: {model_name}. Valid models: {valid_models}")
            return None
        
        logger.info(f"Starting server on port {port} (GPU {gpu_id}) with model {model_name}")
        
        # Set up environment variables
        env = os.environ.copy()
        env['MODEL_NAME'] = model_name
        env['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        project_root = os.environ.get('PROJECT_ROOT', '')
        env['HF_HOME'] = f"{project_root}/Adaptive-RAG/.cache/huggingface"
        env['HF_DATASETS_CACHE'] = f"{project_root}/Adaptive-RAG/.cache/huggingface"
        
        # Path to the serve.py script
        serve_script = project_root / "llm_server" / "serve.py"
        
        if not serve_script.exists():
            logger.error(f"Server script not found: {serve_script}")
            return None
        
        try:
            # Start the server using uvicorn
            cmd = [
                "python", "-m", "uvicorn",
                f"llm_server.serve:app",
                "--host", "0.0.0.0",
                "--port", str(port),
                "--workers", "1",
                "--log-level", "info"
            ]
            
            logger.debug(f"Starting server with command: {' '.join(cmd)}")
            logger.debug(f"Working directory: {project_root}")
            logger.debug(f"Environment: CUDA_VISIBLE_DEVICES={gpu_id}, MODEL_NAME={model_name}")
            
            # Create log directory if it doesn't exist
            log_dir = "/tmp/llm_logs"
            os.makedirs(log_dir, exist_ok=True)
            log_file_path = f"{log_dir}/server_{port}.log"
            
            # Open log file for writing (this avoids pipe buffer issues)
            log_file = open(log_file_path, 'w')
            
            process = subprocess.Popen(
                cmd,
                cwd=str(project_root),
                env=env,
                stdout=log_file,
                stderr=log_file,  # Log stderr to same file for debugging  
                preexec_fn=os.setsid  # Create new process group for clean shutdown
            )
            
            # Note: log_file will be closed when the process terminates
            
            # Wait a moment for the server to start
            time.sleep(3)
            
            # Check if process is still running
            if process.poll() is not None:
                logger.error(f"Server failed to start on port {port}")
                # Try to read from log file for debugging
                try:
                    log_file.close()  # Ensure file is closed before reading
                    with open(log_file_path, 'r') as f:
                        log_content = f.read()
                        if log_content:
                            logger.error(f"Server log: {log_content[-500:]}")  # Last 500 chars
                        else:
                            logger.error("No log content found")
                except Exception as e:
                    logger.error(f"Could not read log file: {e}")
                return None
            
            server_process = ServerProcess(
                port=port,
                gpu_id=gpu_id,
                model=model_name,
                process=process,
                pid=process.pid
            )
            
            # Thread-safe update of shared data structures
            with self._thread_safe_operation():
                self.running_servers[port] = server_process
                self.server_processes.append(process)
            
            logger.info(f"✅ Server started successfully on port {port} (PID: {process.pid})")
            return server_process
            
        except Exception as e:
            logger.error(f"Failed to start server on port {port}: {e}")
            return None
    
    def wait_for_server_ready(self, port: int, timeout: int = 60) -> bool:
        """
        Wait for server to be ready to accept requests
        
        Args:
            port: Server port
            timeout: Timeout in seconds
            
        Returns:
            True if server is ready, False if timeout
        """
        url = f"http://localhost:{port}/"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    logger.info(f"✅ Server on port {port} is ready")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(2)
        
        logger.warning(f"⚠️  Server on port {port} not ready after {timeout}s")
        return False
    
    def _test_server_generation(self, started_servers: List[ServerProcess]) -> None:
        """Test generation functionality on all started servers."""
        test_prompt = "What is 2+2?"
        success_count = 0
        
        def test_single_server(server: ServerProcess) -> bool:
            """Test generation on a single server."""
            try:
                import requests
                url = f"http://localhost:{server.port}/generate/"
                params = {"prompt": test_prompt, "max_tokens": 10}
                
                response = requests.get(url, params=params, timeout=10)
                if response.status_code == 200:
                    data = response.json()
                    generated_text = data.get('generated_texts', [''])[0]
                    if generated_text and len(generated_text.strip()) > 0:
                        logger.info(f"✅ Port {server.port}: Generation test passed - '{generated_text[:30]}...'")
                        return True
                    else:
                        logger.error(f"❌ Port {server.port}: Empty generation response")
                        return False
                else:
                    logger.error(f"❌ Port {server.port}: HTTP {response.status_code}")
                    return False
                    
            except Exception as e:
                logger.error(f"❌ Port {server.port}: Generation test failed - {e}")
                return False
        
        logger.info(f"🧪 Testing generation on {len(started_servers)} servers...")
        
        # Test servers in parallel for faster execution
        with ThreadPoolExecutor(max_workers=8) as executor:
            future_to_server = {
                executor.submit(test_single_server, server): server
                for server in started_servers
            }
            
            for future in as_completed(future_to_server):
                server = future_to_server[future]
                try:
                    if future.result():
                        success_count += 1
                except Exception as e:
                    logger.error(f"Exception testing server {server.port}: {e}")
        
        logger.info(f"🎯 Generation test results: {success_count}/{len(started_servers)} servers passed")
        
        if success_count == len(started_servers):
            logger.info("🎉 All servers are generating correctly!")
        else:
            failed_count = len(started_servers) - success_count
            logger.warning(f"⚠️  {failed_count} servers failed generation tests")
    
    def start_single_server(self, port: int, model: str, gpu_id: int, timeout: int = 500) -> bool:
        """
        Start a single server on specific port (used by health monitor for auto-restart)
        
        Args:
            port: Port number for the server
            model: Model name 
            gpu_id: GPU ID to use
            timeout: Request timeout in seconds
            
        Returns:
            True if server started successfully, False otherwise
        """
        # Create server config on the fly
        server_config = {
            'port': port,
            'gpu_id': gpu_id,
            'model': model,
            'timeout': timeout
        }
        
        logger.info(f"Starting single server on port {port} (GPU {gpu_id}) with model {model}")
        
        # Start the server
        server_process = self.start_server(server_config, model)
        
        if server_process:
            # Wait for server to be ready
            if self.wait_for_server_ready(port, timeout=30):
                logger.info(f"✅ Single server on port {port} is ready and responsive")
                return True
            else:
                logger.error(f"❌ Single server on port {port} started but not responsive")
                # Clean up the failed server
                if port in self.running_servers:
                    process = self.running_servers[port].process
                    if process and process.poll() is None:
                        process.terminate()
                    del self.running_servers[port]
                return False
        else:
            logger.error(f"❌ Failed to start single server on port {port}")
            return False
    
    def start_servers(self, config: Dict[str, Any], model: str = None, 
                     max_parallel: int = 4, wait_for_ready: bool = True) -> bool:
        """
        Start all servers from configuration
        
        Args:
            config: Server configuration
            model: Model name to override config
            max_parallel: Maximum number of servers to start in parallel
            wait_for_ready: Whether to wait for servers to be ready
            
        Returns:
            True if all servers started successfully
        """
        servers = config.get('llm_servers', [])
        if not servers:
            logger.error("No servers configured")
            return False
        
        # Validate port availability before starting
        unavailable_ports = self.validate_ports_available(config)
        if unavailable_ports:
            logger.error(f"Cannot start servers - ports in use: {unavailable_ports}")
            return False
        
        logger.info(f"Starting {len(servers)} servers with max {max_parallel} parallel...")
        logger.info(f"Port validation passed - all {len(servers)} ports available")
        
        # Start servers in batches
        started_servers = []
        failed_servers = []
        
        def start_server_batch(server_configs):
            """Start a batch of servers"""
            batch_results = []
            with ThreadPoolExecutor(max_workers=max_parallel) as executor:
                futures = {
                    executor.submit(self.start_server, server_config, model): server_config
                    for server_config in server_configs
                }
                
                for future in as_completed(futures):
                    server_config = futures[future]
                    try:
                        result = future.result()
                        if result:
                            batch_results.append(result)
                        else:
                            failed_servers.append(server_config['port'])
                    except Exception as e:
                        logger.error(f"Exception starting server {server_config['port']}: {e}")
                        failed_servers.append(server_config['port'])
            
            return batch_results
        
        # Process servers in batches to avoid overwhelming the system
        batch_size = max_parallel
        for i in range(0, len(servers), batch_size):
            batch = servers[i:i + batch_size]
            logger.info(f"Starting batch {i//batch_size + 1}: ports {[s['port'] for s in batch]}")
            
            batch_started = start_server_batch(batch)
            started_servers.extend(batch_started)
            
            # Small delay between batches
            if i + batch_size < len(servers):
                time.sleep(5)
        
        logger.info(f"Started {len(started_servers)} servers successfully")
        if failed_servers:
            logger.warning(f"Failed to start servers on ports: {failed_servers}")
        
        # Wait for servers to be ready if requested
        if wait_for_ready and started_servers:
            logger.info("Waiting for servers to be ready...")
            ready_count = 0
            
            with ThreadPoolExecutor(max_workers=10) as executor:
                future_to_port = {
                    executor.submit(self.wait_for_server_ready, server.port): server.port 
                    for server in started_servers
                }
                
                for future in as_completed(future_to_port):
                    port = future_to_port[future]
                    try:
                        if future.result():
                            ready_count += 1
                    except Exception as e:
                        logger.error(f"Error checking server {port}: {e}")
            
            logger.info(f"✅ {ready_count}/{len(started_servers)} servers are ready")
            
            # Test generation on all servers if requested
            if wait_for_ready and started_servers:
                logger.info("🧪 Testing generation on all servers...")
                self._test_server_generation(started_servers)
        
        return len(failed_servers) == 0
    
    def stop_server(self, port: int) -> bool:
        """Stop a single server with robust cleanup"""
        with self._thread_safe_operation():
            if port not in self.running_servers:
                logger.warning(f"Server on port {port} not found in running servers")
                return False
            
            server = self.running_servers[port]
        
        logger.info(f"Stopping server on port {port} (PID: {server.pid})")
        
        try:
            # Check if process still exists
            if server.process.poll() is not None:
                logger.info(f"Process {server.pid} already terminated")
                with self._thread_safe_operation():
                    self._cleanup_server_tracking_unsafe(port, server)
                return True
            
            # Attempt graceful shutdown
            if self._graceful_shutdown(server):
                with self._thread_safe_operation():
                    self._cleanup_server_tracking_unsafe(port, server)
                logger.info(f"✅ Server on port {port} stopped gracefully")
                return True
            
            # Force kill if graceful failed
            if self._force_kill(server):
                with self._thread_safe_operation():
                    self._cleanup_server_tracking_unsafe(port, server)
                logger.info(f"✅ Server on port {port} stopped (force killed)")
                return True
                
            logger.error(f"Failed to stop server on port {port}")
            return False
            
        except Exception as e:
            logger.error(f"Error stopping server on port {port}: {e}")
            # Still try to clean up tracking
            try:
                with self._thread_safe_operation():
                    self._cleanup_server_tracking_unsafe(port, server)
            except Exception:
                pass
            return False
    
    def _graceful_shutdown(self, server: ServerProcess) -> bool:
        """Attempt graceful shutdown with proper error handling"""
        try:
            # Check if process group exists
            try:
                pgid = os.getpgid(server.process.pid)
                os.killpg(pgid, signal.SIGTERM)
                logger.debug(f"Sent SIGTERM to process group {pgid}")
            except (ProcessLookupError, OSError) as e:
                # Process group doesn't exist or other OS error, try direct process kill
                logger.debug(f"Process group operation failed ({e}), trying direct termination")
                server.process.terminate()
            
            # Wait for graceful shutdown
            try:
                server.process.wait(timeout=10)
                return True
            except subprocess.TimeoutExpired:
                logger.debug(f"Process {server.pid} did not terminate gracefully")
                return False
                
        except Exception as e:
            logger.error(f"Error in graceful shutdown: {e}")
            return False
    
    def _force_kill(self, server: ServerProcess) -> bool:
        """Force kill with proper error handling"""
        try:
            logger.warning(f"Force killing server on port {server.port}")
            
            try:
                pgid = os.getpgid(server.process.pid)
                os.killpg(pgid, signal.SIGKILL)
                logger.debug(f"Sent SIGKILL to process group {pgid}")
            except (ProcessLookupError, OSError) as e:
                logger.debug(f"Process group kill failed ({e}), trying direct kill")
                server.process.kill()
            
            try:
                server.process.wait(timeout=5)
                return True
            except subprocess.TimeoutExpired:
                logger.error(f"Process {server.pid} did not terminate after SIGKILL")
                return False
                
        except Exception as e:
            logger.error(f"Error in force kill: {e}")
            return False
    
    def stop_servers(self, config: Dict[str, Any]) -> bool:
        """Stop all servers from configuration"""
        servers = config.get('llm_servers', [])
        if not servers:
            logger.warning("No servers configured to stop")
            return True
        
        ports = [s['port'] for s in servers]
        logger.info(f"Stopping {len(ports)} servers: {ports}")
        
        success_count = 0
        for port in ports:
            if self.stop_server(port):
                success_count += 1
        
        logger.info(f"Stopped {success_count}/{len(ports)} servers")
        return success_count == len(ports)
    
    def stop_all_servers(self):
        """Stop all running servers"""
        if not self.running_servers:
            logger.info("No servers to stop")
            return
        
        ports = list(self.running_servers.keys())
        logger.info(f"Stopping all {len(ports)} running servers")
        
        for port in ports:
            self.stop_server(port)
    
    def check_server_status(self, port: int) -> Dict[str, Any]:
        """Check status of a single server"""
        status = {
            "port": port,
            "running": False,
            "accessible": False,
            "model": None,
            "error": None
        }
        
        # Check if we have the process
        if port in self.running_servers:
            server = self.running_servers[port]
            status["running"] = server.process.poll() is None
            status["model"] = server.model
            
            if not status["running"]:
                status["error"] = "Process terminated"
        
        # Check if server is accessible
        try:
            response = requests.get(f"http://localhost:{port}/", timeout=5)
            if response.status_code == 200:
                status["accessible"] = True
                data = response.json()
                if "model" in data.get("message", ""):
                    # Extract model name from message
                    message = data["message"]
                    if "server for" in message:
                        model_part = message.split("server for ")[-1].split(".")[0]
                        status["model"] = model_part
        except requests.exceptions.RequestException as e:
            status["error"] = str(e)
        
        return status
    
    def check_servers_status(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Check status of all servers from configuration"""
        servers = config.get('llm_servers', [])
        if not servers:
            return {"servers": [], "summary": {"total": 0, "running": 0, "accessible": 0}}
        
        ports = [s['port'] for s in servers]
        logger.info(f"Checking status of {len(ports)} servers...")
        
        statuses = []
        with ThreadPoolExecutor(max_workers=10) as executor:
            future_to_port = {
                executor.submit(self.check_server_status, port): port 
                for port in ports
            }
            
            for future in as_completed(future_to_port):
                port = future_to_port[future]
                try:
                    status = future.result()
                    statuses.append(status)
                except Exception as e:
                    statuses.append({
                        "port": port,
                        "running": False,
                        "accessible": False,
                        "model": None,
                        "error": f"Status check failed: {e}"
                    })
        
        # Sort by port
        statuses.sort(key=lambda x: x["port"])
        
        # Calculate summary
        total = len(statuses)
        running = sum(1 for s in statuses if s["running"])
        accessible = sum(1 for s in statuses if s["accessible"])
        
        return {
            "servers": statuses,
            "summary": {
                "total": total,
                "running": running,
                "accessible": accessible
            }
        }


def parse_gpu_ids(gpu_ids_str: str) -> List[int]:
    """Parse GPU IDs from comma-separated string"""
    try:
        return [int(gpu_id.strip()) for gpu_id in gpu_ids_str.split(',')]
    except ValueError as e:
        raise ValueError(f"Invalid GPU IDs format: {gpu_ids_str}. Expected comma-separated integers.") from e


def main():
    parser = argparse.ArgumentParser(
        description="LLM Server Startup Manager",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate configuration for 10 GPUs with 4 models each
  python server_startup.py generate-config --gpu_ids 0,1,2,3,4,5,6,7,8,9 --models_per_gpu 4 --model flan-t5-xl --config /tmp/flan_t5_xl_config.json
  
  # Start servers using configuration
  python server_startup.py start --config /tmp/flan_t5_xl_config.json --model flan-t5-xl
  
  # Stop all servers
  python server_startup.py stop --config /tmp/flan_t5_xl_config.json
  
  # Check server status
  python server_startup.py status --config /tmp/flan_t5_xl_config.json
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Commands")
    
    # Generate config command
    gen_parser = subparsers.add_parser("generate-config", help="Generate server configuration")
    gen_parser.add_argument("--gpu_ids", type=str, required=True,
                           help="Comma-separated GPU IDs (e.g., '0,1,2,3')")
    gen_parser.add_argument("--models_per_gpu", type=int, required=True,
                           help="Number of model instances per GPU")
    gen_parser.add_argument("--model", type=str, required=True,
                           choices=["flan-t5-base", "flan-t5-large", "flan-t5-xl", "flan-t5-xxl", "Qwen/Qwen2.5-7B-Instruct", "Qwen/Qwen2.5-3B-Instruct"],
                           help="Model name")
    gen_parser.add_argument("--config", type=str, required=True,
                           help="Output configuration file path")
    gen_parser.add_argument("--start_port", type=int, default=8010,
                           help="Starting port number (default: 8010)")
    gen_parser.add_argument("--timeout", type=int, default=500,
                           help="Request timeout in seconds (default: 500)")
    
    # Start servers command
    start_parser = subparsers.add_parser("start", help="Start servers from configuration")
    start_parser.add_argument("--config", type=str, required=True,
                             help="Configuration file path")
    start_parser.add_argument("--model", type=str,
                             help="Model name (overrides config)")
    start_parser.add_argument("--max_parallel", type=int, default=4,
                             help="Maximum parallel server starts (default: 4)")
    start_parser.add_argument("--no_wait", action="store_true",
                             help="Don't wait for servers to be ready")
    
    # Stop servers command
    stop_parser = subparsers.add_parser("stop", help="Stop servers from configuration")
    stop_parser.add_argument("--config", type=str, required=True,
                            help="Configuration file path")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Check server status")
    status_parser.add_argument("--config", type=str, required=True,
                              help="Configuration file path")
    status_parser.add_argument("--detailed", action="store_true",
                              help="Show detailed status for each server")
    
    # Start single server command (for health monitor auto-restart)
    single_parser = subparsers.add_parser("start_single", help="Start a single server on specific port")
    single_parser.add_argument("--port", type=int, required=True,
                              help="Port number for the server")
    single_parser.add_argument("--model", type=str, required=True,
                              help="Model name")
    single_parser.add_argument("--gpu-id", type=int, required=True,
                              help="GPU ID to use")
    single_parser.add_argument("--timeout", type=int, default=500,
                              help="Request timeout in seconds")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    starter = LLMServerStarter()
    
    try:
        if args.command == "generate-config":
            # Parse GPU IDs
            gpu_ids = parse_gpu_ids(args.gpu_ids)
            
            # Validate arguments
            if args.models_per_gpu <= 0:
                logger.error("models_per_gpu must be positive")
                sys.exit(1)
            
            total_servers = len(gpu_ids) * args.models_per_gpu
            logger.info(f"Generating config for {total_servers} servers across {len(gpu_ids)} GPUs")
            
            # Generate configuration
            config = starter.generate_config(
                gpu_ids=gpu_ids,
                models_per_gpu=args.models_per_gpu,
                model=args.model,
                start_port=args.start_port,
                timeout=args.timeout
            )
            
            # Save configuration
            starter.save_config(config, args.config)
            
            logger.info("✅ Configuration generated successfully")
            
        elif args.command == "start":
            # Load configuration
            config = starter.load_config(args.config)
            
            metadata = config.get('metadata', {})
            logger.info(f"Starting servers from configuration:")
            logger.info(f"  Total servers: {metadata.get('total_servers', 'unknown')}")
            logger.info(f"  Model: {args.model or metadata.get('model', 'unknown')}")
            logger.info(f"  Port range: {metadata.get('port_range', 'unknown')}")
            
            # Start servers
            success = starter.start_servers(
                config=config,
                model=args.model,
                max_parallel=args.max_parallel,
                wait_for_ready=not args.no_wait
            )
            
            if success:
                logger.info("✅ All servers started successfully")
                logger.info("Servers are running. Press Ctrl+C to stop all servers.")
                
                # Keep the script running to manage servers
                try:
                    while True:
                        time.sleep(10)
                except KeyboardInterrupt:
                    logger.info("Shutting down servers...")
                    starter.stop_all_servers()
            else:
                logger.error("❌ Some servers failed to start")
                sys.exit(1)
                
        elif args.command == "stop":
            # Load configuration
            config = starter.load_config(args.config)
            
            # Stop servers
            success = starter.stop_servers(config)
            
            if success:
                logger.info("✅ All servers stopped successfully")
            else:
                logger.warning("⚠️  Some servers may not have stopped cleanly")
                
        elif args.command == "status":
            # Load configuration
            config = starter.load_config(args.config)
            
            # Check status
            status_info = starter.check_servers_status(config)
            
            summary = status_info['summary']
            logger.info(f"📊 Server Status Summary:")
            logger.info(f"  Total servers: {summary['total']}")
            logger.info(f"  Running: {summary['running']}")
            logger.info(f"  Accessible: {summary['accessible']}")
            
            if args.detailed:
                logger.info("\n📋 Detailed Status:")
                for server in status_info['servers']:
                    port = server['port']
                    running = "✅" if server['running'] else "❌"
                    accessible = "✅" if server['accessible'] else "❌"
                    model = server.get('model', 'unknown')
                    error = server.get('error', '')
                    
                    status_line = f"  Port {port}: Running {running} | Accessible {accessible} | Model: {model}"
                    if error:
                        status_line += f" | Error: {error}"
                    
                    logger.info(status_line)
                    
        elif args.command == "start_single":
            # Start a single server (used by health monitor)
            success = starter.start_single_server(
                port=args.port,
                model=args.model,
                gpu_id=args.gpu_id,  # Python converts --gpu-id to gpu_id automatically
                timeout=args.timeout
            )
            
            if success:
                logger.info(f"✅ Server started successfully on port {args.port}")
                logger.info("Server is running. Press Ctrl+C to stop the server.")
                
                # Keep the script running to manage the server
                try:
                    while True:
                        time.sleep(10)
                except KeyboardInterrupt:
                    logger.info("Shutting down server...")
                    starter.stop_all_servers()
            else:
                logger.error(f"❌ Failed to start server on port {args.port}")
                sys.exit(1)
    
    except KeyboardInterrupt:
        logger.info("Operation interrupted by user")
        starter.stop_all_servers()
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()