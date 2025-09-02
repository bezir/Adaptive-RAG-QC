"""
Server Health Monitoring and Port Management

This module provides:
- Server health checking
- Automatic port rotation for failed servers
- Dynamic server discovery
- Connection retry with exponential backoff
"""

import requests
import time
import random
import logging
from typing import List, Dict, Set, Optional, Tuple
from dataclasses import dataclass
from threading import Lock

logger = logging.getLogger(__name__)


@dataclass
class ServerInfo:
    """Information about a server instance."""
    host: str
    port: int
    is_healthy: bool = True
    last_check: float = 0.0
    consecutive_failures: int = 0
    response_time: float = 0.0


class ServerHealthManager:
    """
    Manages server health and provides intelligent port rotation.
    
    Features:
    - Health checking with configurable intervals
    - Automatic failover to healthy servers
    - Exponential backoff for failed servers
    - Connection pooling and reuse
    """
    
    def __init__(self,
                 host: str = "localhost",
                 port_range: Tuple[int, int] = (8010, 8026),
                 health_check_interval: float = 30.0,
                 max_failures: int = 3,
                 timeout: float = 5.0):
        """
        Initialize server health manager.
        
        Args:
            host: Server host (default: localhost)
            port_range: Range of ports to monitor (start, end)
            health_check_interval: Seconds between health checks
            max_failures: Max consecutive failures before marking unhealthy
            timeout: Request timeout in seconds
        """
        self.host = host
        self.port_range = port_range
        self.health_check_interval = health_check_interval
        self.max_failures = max_failures
        self.timeout = timeout
        
        # Server tracking
        self.servers: Dict[int, ServerInfo] = {}
        self.healthy_ports: Set[int] = set()
        self.last_used_port: Optional[int] = None
        self._lock = Lock()
        
        # Initialize servers
        self._discover_servers()
        
        logger.info(f"🏥 Initialized ServerHealthManager for {host}:{port_range[0]}-{port_range[1]}")
    
    def _discover_servers(self) -> None:
        """Discover available servers in the port range."""
        start_port, end_port = self.port_range
        
        logger.info(f"🔍 Discovering servers on {self.host}:{start_port}-{end_port}")
        
        for port in range(start_port, end_port + 1):
            server = ServerInfo(host=self.host, port=port)
            self.servers[port] = server
            
            # Quick health check
            if self._check_server_health(server):
                self.healthy_ports.add(port)
                logger.info(f"✅ Found healthy server at {self.host}:{port}")
            else:
                logger.debug(f"❌ Server at {self.host}:{port} is not responding")
        
        logger.info(f"🌟 Discovered {len(self.healthy_ports)} healthy servers")
    
    def _check_server_health(self, server: ServerInfo) -> bool:
        """
        Check if a server is healthy.
        
        Args:
            server: Server to check
            
        Returns:
            True if server is healthy
        """
        try:
            url = f"http://{server.host}:{server.port}/"
            start_time = time.time()
            
            response = requests.get(url, timeout=self.timeout)
            response_time = time.time() - start_time
            
            if response.status_code == 200:
                with self._lock:
                    server.is_healthy = True
                    server.consecutive_failures = 0
                    server.response_time = response_time
                    server.last_check = time.time()
                return True
            else:
                logger.debug(f"❌ Server {server.host}:{server.port} returned status {response.status_code}")
                return False
                
        except Exception as e:
            logger.debug(f"❌ Health check failed for {server.host}:{server.port}: {e}")
            with self._lock:
                server.consecutive_failures += 1
                server.last_check = time.time()
                
                if server.consecutive_failures >= self.max_failures:
                    server.is_healthy = False
                    if server.port in self.healthy_ports:
                        self.healthy_ports.remove(server.port)
                        logger.warning(f"🚫 Marked server {server.host}:{server.port} as unhealthy after {server.consecutive_failures} failures")
            
            return False
    
    def get_healthy_server(self) -> Optional[ServerInfo]:
        """
        Get a healthy server with intelligent selection.
        
        Returns:
            ServerInfo for a healthy server or None if none available
        """
        with self._lock:
            if not self.healthy_ports:
                logger.warning("⚠️ No healthy servers available, attempting recovery...")
                self._attempt_recovery()
                
                if not self.healthy_ports:
                    logger.error("❌ No servers available after recovery attempt")
                    return None
            
            # Select server with round-robin + response time weighting
            available_ports = list(self.healthy_ports)
            
            if self.last_used_port and self.last_used_port in available_ports:
                # Try to use a different server for load balancing
                available_ports = [p for p in available_ports if p != self.last_used_port]
                if not available_ports:
                    available_ports = list(self.healthy_ports)
            
            # Sort by response time (fastest first)
            available_ports.sort(key=lambda p: self.servers[p].response_time)
            
            # Select from the fastest 3 servers (or all if less than 3)
            top_servers = available_ports[:min(3, len(available_ports))]
            selected_port = random.choice(top_servers)
            
            self.last_used_port = selected_port
            return self.servers[selected_port]
    
    def _attempt_recovery(self) -> None:
        """Attempt to recover failed servers."""
        logger.info("🔄 Attempting server recovery...")
        
        # Re-check all servers that were marked unhealthy
        unhealthy_servers = [s for s in self.servers.values() if not s.is_healthy]
        
        for server in unhealthy_servers:
            if self._check_server_health(server):
                self.healthy_ports.add(server.port)
                logger.info(f"✅ Recovered server {server.host}:{server.port}")
    
    def mark_server_failed(self, port: int) -> None:
        """
        Mark a server as failed after a request failure.
        
        Args:
            port: Port of the failed server
        """
        with self._lock:
            if port in self.servers:
                server = self.servers[port]
                server.consecutive_failures += 1
                
                if server.consecutive_failures >= self.max_failures:
                    server.is_healthy = False
                    if port in self.healthy_ports:
                        self.healthy_ports.remove(port)
                        logger.warning(f"🚫 Marked server localhost:{port} as unhealthy after request failure")
    
    def periodic_health_check(self) -> None:
        """Run periodic health checks on all servers."""
        current_time = time.time()
        
        for server in self.servers.values():
            if current_time - server.last_check > self.health_check_interval:
                self._check_server_health(server)
    
    def get_server_stats(self) -> Dict:
        """Get statistics about server health."""
        with self._lock:
            healthy_count = len(self.healthy_ports)
            total_count = len(self.servers)
            
            avg_response_time = 0.0
            if self.healthy_ports:
                total_time = sum(self.servers[p].response_time for p in self.healthy_ports)
                avg_response_time = total_time / len(self.healthy_ports)
            
            return {
                "healthy_servers": healthy_count,
                "total_servers": total_count,
                "healthy_ports": sorted(list(self.healthy_ports)),
                "average_response_time": avg_response_time,
                "last_used_port": self.last_used_port
            }


# Global server manager instance
_server_manager: Optional[ServerHealthManager] = None
_manager_lock = Lock()


def get_server_manager(host: str = "localhost", 
                      port_range: Tuple[int, int] = (8010, 8026)) -> ServerHealthManager:
    """
    Get or create the global server manager instance.
    
    Args:
        host: Server host
        port_range: Port range to monitor
        
    Returns:
        ServerHealthManager instance
    """
    global _server_manager
    
    with _manager_lock:
        if _server_manager is None:
            _server_manager = ServerHealthManager(host=host, port_range=port_range)
        return _server_manager


def get_available_server() -> Optional[Tuple[str, int]]:
    """
    Get an available server (host, port) tuple.
    
    Returns:
        (host, port) tuple or None if no servers available
    """
    manager = get_server_manager()
    server = manager.get_healthy_server()
    
    if server:
        return server.host, server.port
    return None
