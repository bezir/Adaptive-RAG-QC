#!/usr/bin/env python3
"""
Base Logger Module for Scaled Silver Labeling System

This module provides a comprehensive logging infrastructure for the labeling system.
It includes support for multiple log levels, structured logging, and specialized loggers for
different components of the system.
"""

import logging
import logging.handlers
import os
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from abc import ABC, abstractmethod
import threading


class LogFormatter(logging.Formatter):
    """Custom log formatter with timestamp and structured output"""
    
    def __init__(self, include_thread=True):
        self.include_thread = include_thread
        super().__init__()
    
    def format(self, record):
        """Format log record with structured information"""
        timestamp = datetime.fromtimestamp(record.created).isoformat()
        
        # Base log info
        log_data = {
            'timestamp': timestamp,
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
        }
        
        # Add thread information if requested
        if self.include_thread:
            log_data['thread'] = threading.current_thread().name
        
        # Add extra fields if present
        if hasattr(record, 'experiment_id'):
            log_data['experiment_id'] = record.experiment_id
        if hasattr(record, 'dataset_name'):
            log_data['dataset_name'] = record.dataset_name
        if hasattr(record, 'model_name'):
            log_data['model_name'] = record.model_name
        if hasattr(record, 'server_id'):
            log_data['server_id'] = record.server_id
        
        # Add exception info if present
        if record.exc_info:
            log_data['exception'] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class BaseLogger(ABC):
    """Abstract base class for all loggers in the system"""
    
    def __init__(self, 
                 name: str,
                 log_level: int = logging.INFO,
                 log_format: str = None,
                 log_file: str = None,
                 log_directory: str = "scaled_silver_labeling/logs"):
        """
        Initialize the base logger
        
        Args:
            name: Logger name
            log_level: Logging level (default: INFO)
            log_format: Log message format
            log_file: Log file name (if None, generates timestamp-based name)
            log_directory: Directory for log files (default: scaled_silver_labeling/logs)
        """
        self.name = name
        self.log_dir = Path(log_directory)
        # Handle log_level whether it's passed as string or integer
        if isinstance(log_level, str):
            self.log_level = getattr(logging, log_level.upper())
        else:
            self.log_level = log_level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(self.log_level)
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging handlers
        self._setup_handlers()
    
    def _setup_handlers(self):
        """Setup logging handlers for console and file output"""
        # Prevent duplicate handlers - only add if no handlers exist
        if not self.logger.handlers:
            # Console handler with human-readable format
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
            
            # File handler with JSON format
            log_file = self.log_dir / f"{self.name}.log"
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10*1024*1024, backupCount=5
            )
            file_handler.setLevel(self.log_level)
            file_formatter = LogFormatter()
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to parent loggers to avoid duplicate messages
        self.logger.propagate = False
    
    def log(self, level: str, message: str, extra: Optional[Dict[str, Any]] = None):
        """Log a message with optional extra fields"""
        log_method = getattr(self.logger, level.lower())
        log_method(message, extra=extra or {})
    
    def info(self, message: str, **kwargs):
        """Log an info message"""
        self.log("INFO", message, kwargs)
    
    def debug(self, message: str, **kwargs):
        """Log a debug message"""
        self.log("DEBUG", message, kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log a warning message"""
        self.log("WARNING", message, kwargs)
    
    def error(self, message: str, **kwargs):
        """Log an error message"""
        self.log("ERROR", message, kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log a critical message"""
        self.log("CRITICAL", message, kwargs)
    
    @abstractmethod
    def start_session(self, session_info: Dict[str, Any]) -> str:
        """Start a new logging session"""
        pass
    
    @abstractmethod
    def end_session(self, session_id: str, session_results: Dict[str, Any]):
        """End a logging session"""
        pass


class SystemLogger(BaseLogger):
    """System-wide logger for general system events"""
    
    def __init__(self, log_dir: str, log_level: str = "INFO"):
        super().__init__("system", log_dir, log_level)
        self.sessions = {}
    
    def start_session(self, session_info: Dict[str, Any]) -> str:
        """Start a new system session"""
        session_id = f"session_{int(time.time())}"
        self.sessions[session_id] = {
            'start_time': datetime.now().isoformat(),
            'info': session_info
        }
        
        self.info(f"Starting system session: {session_id}", 
                 session_id=session_id, **session_info)
        return session_id
    
    def end_session(self, session_id: str, session_results: Dict[str, Any]):
        """End a system session"""
        if session_id in self.sessions:
            self.sessions[session_id]['end_time'] = datetime.now().isoformat()
            self.sessions[session_id]['results'] = session_results
            
            self.info(f"Ending system session: {session_id}", 
                     session_id=session_id, **session_results)
        else:
            self.warning(f"Unknown session ID: {session_id}")
    
    def log_system_event(self, event_type: str, event_data: Dict[str, Any]):
        """Log a system event"""
        self.info(f"System event: {event_type}", 
                 event_type=event_type, **event_data)


class PerformanceLogger(BaseLogger):
    """Logger for performance metrics and monitoring"""
    
    def __init__(self, log_dir: str, log_level: str = "INFO"):
        super().__init__("performance", log_dir, log_level)
        self.metrics = []
    
    def start_session(self, session_info: Dict[str, Any]) -> str:
        """Start a new performance session"""
        session_id = f"perf_{int(time.time())}"
        self.info(f"Starting performance monitoring: {session_id}", 
                 session_id=session_id, **session_info)
        return session_id
    
    def end_session(self, session_id: str, session_results: Dict[str, Any]):
        """End a performance session"""
        self.info(f"Ending performance monitoring: {session_id}", 
                 session_id=session_id, **session_results)
    
    def log_metric(self, metric_name: str, value: float, unit: str = "", 
                   tags: Optional[Dict[str, str]] = None):
        """Log a performance metric"""
        metric_data = {
            'metric': metric_name,
            'value': value,
            'unit': unit,
            'tags': tags or {},
            'timestamp': datetime.now().isoformat()
        }
        
        self.metrics.append(metric_data)
        self.info(f"Metric: {metric_name} = {value} {unit}", **metric_data)
    
    def log_timing(self, operation: str, duration: float, context: Optional[Dict[str, Any]] = None):
        """Log timing information"""
        self.log_metric(f"{operation}_duration", duration, "seconds", 
                       tags={'operation': operation})
        
        if context:
            self.info(f"Timing: {operation} took {duration:.3f}s", 
                     operation=operation, duration=duration, **context)


class LoggerFactory:
    """Factory class for creating different types of loggers"""
    
    _instances = {}
    
    @classmethod
    def get_logger(cls, logger_type: str, log_dir: str, log_level: str = "INFO") -> BaseLogger:
        """Get or create a logger instance"""
        key = f"{logger_type}_{log_dir}_{log_level}"
        
        if key not in cls._instances:
            if logger_type == "system":
                cls._instances[key] = SystemLogger(log_dir, log_level)
            elif logger_type == "performance":
                cls._instances[key] = PerformanceLogger(log_dir, log_level)
            else:
                raise ValueError(f"Unknown logger type: {logger_type}")
        
        return cls._instances[key]
    
    @classmethod
    def create_llm_interaction_logger(cls, log_dir: str, 
                                     log_level: str = "INFO", 
                                     experiment_name: str = None) -> 'LLMInteractionLogger':
        """Create an LLM interaction logger"""
        return LLMInteractionLogger(log_dir, log_level, experiment_name)


class LLMInteractionLogger(BaseLogger):
    """Specialized logger for LLM interactions - uses JSONL format only"""
    
    def __init__(self, log_dir: str = "scaled_silver_labeling/logs", 
                 log_level: str = "INFO", experiment_name: str = None):
        """Initialize LLM interaction logger with optional experiment-specific naming"""
        if experiment_name:
            logger_name = f"llm_interactions_{experiment_name}"
        else:
            logger_name = "llm_interactions"
        
        # Initialize base without calling parent's _setup_handlers
        self.name = logger_name
        self.log_dir = Path(log_dir)
        if isinstance(log_level, str):
            self.log_level = getattr(logging, log_level.upper())
        else:
            self.log_level = log_level
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(self.log_level)
        
        # Create log directory if it doesn't exist
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup only console handler (no file duplication)
        self._setup_console_only()
        self.interactions = []  # Store for comprehensive JSONL export
        
        # Add time tracking and label counting
        self.start_time = time.time()
        self.processed_count = 0
        self.total_samples = 0
        self.label_counts = {'A': 0, 'B': 0, 'C': 0, 'DISCARDED': 0}
        self.completed_samples = set()  # Track unique sample IDs to avoid double counting

    def _setup_console_only(self):
        """Setup only console logging to avoid file duplication"""
        # Prevent duplicate handlers - only add if no handlers exist
        if not self.logger.handlers:
            # Console handler with human-readable format
            console_handler = logging.StreamHandler()
            console_handler.setLevel(self.log_level)
            console_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            console_handler.setFormatter(console_formatter)
            self.logger.addHandler(console_handler)
        
        # Prevent propagation to parent loggers to avoid duplicate messages
        self.logger.propagate = False
    
    def start_session(self, session_info: Dict[str, Any]) -> str:
        """Start a new LLM interaction session"""
        session_id = f"llm_{int(time.time())}"
        self.info(f"Starting LLM interaction session: {session_id}", 
                 session_id=session_id, **session_info)
        return session_id
    
    def end_session(self, session_id: str, session_results: Dict[str, Any]):
        """End an LLM interaction session"""
        self.info(f"Ending LLM interaction session: {session_id}", 
                 session_id=session_id, **session_results)
    
    def log_request(self, server_id: str, request_data: Dict[str, Any]):
        """Log an LLM request"""
        interaction_id = f"req_{int(time.time() * 1000)}"
        
        request_entry = {
            'interaction_id': interaction_id,
            'timestamp': datetime.now().isoformat(),
            'type': 'request',
            'server_id': server_id,
            'data': request_data
        }
        
        self.interactions.append(request_entry)
        self.info(f"LLM request to {server_id}", 
                 interaction_id=interaction_id, server_id=server_id, **request_data)
        
        return interaction_id
    
    def log_response(self, interaction_id: str, server_id: str, response_data: Dict[str, Any], 
                    latency: float):
        """Log an LLM response"""
        response_entry = {
            'interaction_id': interaction_id,
            'timestamp': datetime.now().isoformat(),
            'type': 'response',
            'server_id': server_id,
            'latency': latency,
            'data': response_data
        }
        
        self.interactions.append(response_entry)
        self.info(f"LLM response from {server_id} (latency: {latency:.3f}s)", 
                 interaction_id=interaction_id, server_id=server_id, 
                 latency=latency, **response_data)
    
    def log_error(self, interaction_id: str, server_id: str, error_data: Dict[str, Any]):
        """Log an LLM error"""
        error_entry = {
            'interaction_id': interaction_id,
            'timestamp': datetime.now().isoformat(),
            'type': 'error',
            'server_id': server_id,
            'data': error_data
        }
        
        self.interactions.append(error_entry)
        self.error(f"LLM error from {server_id}", 
                  interaction_id=interaction_id, server_id=server_id, **error_data)
    
    def set_total_samples(self, total: int):
        """Set the total number of samples for progress tracking"""
        self.total_samples = total
        self.start_time = time.time()
    
    def update_label_count(self, label: str):
        """Update the count for a specific label"""
        if label in self.label_counts:
            self.label_counts[label] += 1
    
    def update_sample_completion(self, sample_id: str, label: str = None):
        """Update sample completion progress and optionally label count"""
        if sample_id not in self.completed_samples:
            self.processed_count += 1
            self.completed_samples.add(sample_id)
            
            if label:
                self.update_label_count(label)
            
            # Log progress update after each sample completion
            time_estimate = self._get_time_estimate_string()
            label_distribution = self._get_label_distribution_string()
            self.info(f"Sample Complete: {time_estimate} | {label_distribution}")

    def _get_time_estimate_string(self) -> str:
        """Calculate and return estimated completion time string"""
        if self.processed_count == 0 or self.total_samples == 0:
            return "ETA: Calculating..."
        
        elapsed_time = time.time() - self.start_time
        samples_per_second = self.processed_count / elapsed_time
        
        if samples_per_second > 0 and self.processed_count < self.total_samples:
            remaining_samples = self.total_samples - self.processed_count
            estimated_seconds = remaining_samples / samples_per_second
            
            # Format time
            if estimated_seconds < 60:
                time_str = f"{estimated_seconds:.0f}s"
            elif estimated_seconds < 3600:
                time_str = f"{estimated_seconds/60:.0f}m {estimated_seconds%60:.0f}s"
            else:
                hours = estimated_seconds // 3600
                minutes = (estimated_seconds % 3600) // 60
                time_str = f"{hours:.0f}h {minutes:.0f}m"
            
            progress_percent = min((self.processed_count / self.total_samples) * 100, 100.0)
            return f"ETA: {time_str} | Progress: {self.processed_count}/{self.total_samples} ({progress_percent:.1f}%)"
        else:
            progress_percent = min((self.processed_count / self.total_samples) * 100, 100.0)
            if self.processed_count >= self.total_samples:
                return f"Completed | Progress: {self.total_samples}/{self.total_samples} (100.0%)"
            else:
                return f"ETA: Calculating... | Progress: {self.processed_count}/{self.total_samples} ({progress_percent:.1f}%)"

    def _get_label_distribution_string(self) -> str:
        """Get current label distribution as a string"""
        total_labels = sum(self.label_counts.values())
        if total_labels == 0:
            return "Labels: A:0 B:0 C:0"
        
        return (f"Labels: A:{self.label_counts['A']} "
                f"B:{self.label_counts['B']} "
                f"C:{self.label_counts['C']}")

    def log_qa_interaction(self, server_id: str, question: str, answer: str, 
                           system_type: str, dataset_name: str, sample_id: str,
                           latency: float, request_data: Dict[str, Any] = None,
                           ground_truth: Union[str, List[str]] = None, retrieved_documents: List[Dict[str, Any]] = None,
                           model_name: str = None, is_qwen_model: bool = None, raw_answer: str = None,
                           sample_label: str = None, **kwargs):
        """Log a comprehensive Q&A interaction with a structured JSONL format"""
        interaction_id = f"qa_{sample_id}_{int(time.time() * 1000)}"

        # Check for ground truth
        ground_truth_str = None
        ground_truth_for_log = ground_truth
        if ground_truth:
            if isinstance(ground_truth, list):
                ground_truth_str = ground_truth[0] if ground_truth else None
            else:
                ground_truth_str = ground_truth

            
        is_correct = self._check_answer_correctness(answer, ground_truth_str) if ground_truth_str else None

        # Label tracking is handled by update_sample_completion after final label is determined
        if sample_label:
            self.update_label_count(sample_label)

        # Reorder keys as requested
        qa_entry = {
            'system_type': system_type,
            'answer': answer,
            'ground_truth': ground_truth_for_log,
            'is_correct': is_correct,
            'question': question,
            'interaction_id': interaction_id,
            'timestamp': datetime.now().isoformat(),
            'type': 'qa_interaction',
            'server_id': server_id,
            'dataset_name': dataset_name,
            'sample_id': sample_id,
            'latency': latency,
            'request_data': request_data or {},
            'retrieved_documents': retrieved_documents or [],
            'retrieval_count': len(retrieved_documents) if retrieved_documents else 0,
            'model_name': model_name,
            'is_qwen_model': is_qwen_model,
            'raw_answer': raw_answer
        }
        qa_entry.update(kwargs) # Add any other dynamic data

        # Store for batch saving
        self.interactions.append(qa_entry)

        # Log retrieval information if documents were retrieved
        retrieval_info = ""
        if retrieved_documents and len(retrieved_documents) > 0:
            retrieval_info = f" | Retrieved: {len(retrieved_documents)} docs"
            # Log a summary of retrieved documents
            doc_titles = [doc.get('title', doc.get('passage_text', '')[:50] + '...' if doc.get('passage_text') else 'No title')[:50]
                         for doc in retrieved_documents[:3]]  # Show first 3 document titles
            retrieval_info += f" | Docs: {', '.join(doc_titles)}"
            if len(retrieved_documents) > 3:
                retrieval_info += f" (+{len(retrieved_documents)-3} more)"

        # Enhanced logging with time estimation and label counts instead of CORRECT/INCORRECT
        time_estimate = self._get_time_estimate_string()
        label_distribution = self._get_label_distribution_string()
        
        # Suppress Q&A interaction logs completely - only show sample completion progress
        pass

    def _check_answer_correctness(self, predicted: str, ground_truth: str) -> bool:
        """Check if predicted answer matches ground truth"""
        if not predicted or not ground_truth:
            return False
        
        # Robust normalization: remove quotes, extra whitespace, convert to lowercase
        def normalize_answer(text: str) -> str:
            # Strip whitespace and convert to lowercase
            normalized = text.strip().lower()
            # Remove surrounding quotes (single or double)
            if len(normalized) >= 2:
                if (normalized.startswith('"') and normalized.endswith('"')) or \
                   (normalized.startswith("'") and normalized.endswith("'")):
                    normalized = normalized[1:-1].strip()
            return normalized
        
        pred_norm = normalize_answer(predicted)
        gt_norm = normalize_answer(ground_truth)
        
        return pred_norm == gt_norm
    
    def save_interactions(self, output_file: Optional[str] = None, session_metadata: Dict[str, Any] = None):
        """Save comprehensive LLM interaction data to unified JSONL file"""
        if output_file is None:
            output_file = self.log_dir / f"{self.name}.jsonl"

        try:
            # Use 'a' to append if file exists, 'w' to create otherwise (but avoid duplicate sessions)
            mode = 'w' if not Path(output_file).exists() else 'a'
            with open(output_file, mode) as f:
                # If creating a new file, write session header
                if mode == 'w':
                    session_header = {
                        'record_type': 'session_start',
                        'timestamp': datetime.now().isoformat(),
                        'logger_name': self.name,
                        'session_metadata': session_metadata or {},
                    }
                    f.write(json.dumps(session_header) + '\n')

                # Write all Q&A interactions
                for interaction in self.interactions:
                    # Ensure each interaction has record_type
                    interaction['record_type'] = interaction.get('type', 'qa_interaction')
                    f.write(json.dumps(interaction) + '\n')

            self.info(f"📄 LLM interactions saved to: {output_file}")
            self.info(f"   📊 Total interactions saved in this session: {len(self.interactions)}")

        except Exception as e:
            self.error(f"Failed to save LLM interactions: {str(e)}",
                      error=str(e))
    
    def _summarize_interactions(self) -> Dict[str, int]:
        """Generate summary statistics of interaction types"""
        summary = {}
        for interaction in self.interactions:
            interaction_type = interaction.get('type', 'unknown')
            summary[interaction_type] = summary.get(interaction_type, 0) + 1
        return summary


# Utility functions for common logging patterns
def setup_logging(log_dir: str, log_level: str = "INFO"):
    """Setup logging for the entire system"""
    # Create log directory
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    
    # Create main system logger
    system_logger = LoggerFactory.get_logger("system", log_dir, log_level)
    system_logger.info("Logging system initialized", log_dir=log_dir, log_level=log_level)
    
    return system_logger


def get_logger(name: str, log_dir: str, log_level: str = "INFO") -> logging.Logger:
    """Get a configured logger instance"""
    logger = logging.getLogger(name)
    
    if not logger.handlers:
        logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create log directory
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        
        # File handler
        log_file = Path(log_dir) / f"{name}.log"
        file_handler = logging.handlers.RotatingFileHandler(
            log_file, maxBytes=10*1024*1024, backupCount=5
        )
        file_handler.setLevel(getattr(logging, log_level.upper()))
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
    
    return logger 