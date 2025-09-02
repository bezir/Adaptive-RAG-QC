#!/usr/bin/env python3
"""
Common Utility Functions for Scaled Silver Labeling System

This module contains utility functions and classes that are used across
the entire labeling system.
"""

import json
import os
import sys
import time
import hashlib
import random
import re
import string
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging

# Add config directory to path for importing environment configuration
_current_dir = Path(__file__).parent.absolute()
_config_dir = _current_dir.parent.parent / "config"
if str(_config_dir) not in sys.path:
    sys.path.insert(0, str(_config_dir))

try:
    from env_config import get_environment_config, get_adaptive_rag_path
    _env_config = get_environment_config()
except ImportError:
    # Fallback if config module is not available
    _env_config = None

# Module-level logger
logger = logging.getLogger(__name__)


def _get_project_root() -> str:
    """Get project root directory with fallback logic."""
    if _env_config:
        return _env_config['PROJECT_ROOT']
    
    # Fallback detection if config is not available
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == 'Adaptive-RAG':
            return str(parent.parent)
        # Check for project markers
        if any((parent / marker).exists() for marker in [
            'adaptive_rag_benchmark', 'classifier', 'scaled_silver_labeling'
        ]):
            return str(parent.parent) if parent.name == 'Adaptive-RAG' else str(parent)
    
    # Final fallback - use environment variable or current working directory
    project_root = os.environ.get('PROJECT_ROOT')
    if not project_root:
        raise RuntimeError("Could not auto-detect PROJECT_ROOT. Please set PROJECT_ROOT environment variable.")
    return project_root


def _get_adaptive_rag_root() -> str:
    """Get Adaptive-RAG directory path."""
    if _env_config:
        return _env_config['ADAPTIVE_RAG_ROOT']
    return f"{_get_project_root()}/Adaptive-RAG"


def _get_cache_dir() -> str:
    """Get cache directory path."""
    if _env_config:
        return _env_config['CACHE_DIR']
    return f"{_get_adaptive_rag_root()}/.cache"


def _get_processed_data_path(dataset_name: str = "") -> str:
    """Get processed data directory path."""
    base_path = f"{_get_adaptive_rag_root()}/processed_data"
    if dataset_name:
        return f"{base_path}/{dataset_name}"
    return base_path


def read_json(file_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Read JSON data from a file
    
    Args:
        file_path: Path to the JSON file
        
    Returns:
        Dictionary containing the JSON data
        
    Raises:
        FileNotFoundError: If the file doesn't exist
        json.JSONDecodeError: If the file contains invalid JSON
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"JSON file not found: {file_path}")
    except json.JSONDecodeError as e:
        raise json.JSONDecodeError(f"Invalid JSON in file {file_path}: {str(e)}")


def write_json(data: Dict[str, Any], file_path: Union[str, Path], indent: int = 2) -> None:
    """
    Write JSON data to a file
    
    Args:
        data: Dictionary to write as JSON
        file_path: Path to the output file
        indent: Number of spaces for indentation
    """
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)


def read_jsonl(file_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """
    Read JSONL (JSON Lines) data from a file
    
    Args:
        file_path: Path to the JSONL file
        
    Returns:
        List of dictionaries, one per line
    """
    data = []
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:  # Skip empty lines
                data.append(json.loads(line))
    return data


def write_jsonl(data: List[Dict[str, Any]], file_path: Union[str, Path]) -> None:
    """
    Write JSONL (JSON Lines) data to a file
    
    Args:
        data: List of dictionaries to write
        file_path: Path to the output file
    """
    # Create directory if it doesn't exist
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')


def ensure_dir(dir_path: Union[str, Path]) -> Path:
    """
    Ensure a directory exists, creating it if necessary
    
    Args:
        dir_path: Path to the directory
        
    Returns:
        Path object for the directory
    """
    path = Path(dir_path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_timestamp() -> str:
    """
    Get current timestamp as ISO format string
    
    Returns:
        Current timestamp in ISO format
    """
    return datetime.now().isoformat()


def get_timestamp_str() -> str:
    """
    Get current timestamp as string suitable for filenames
    
    Returns:
        Current timestamp as YYYYMMDD_HHMMSS string
    """
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def calculate_file_hash(file_path: Union[str, Path]) -> str:
    """
    Calculate SHA256 hash of a file
    
    Args:
        file_path: Path to the file
        
    Returns:
        SHA256 hash as hex string
    """
    sha256_hash = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256_hash.update(chunk)
    return sha256_hash.hexdigest()


def format_duration(seconds: float) -> str:
    """
    Format duration in seconds to human-readable string
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted duration string
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds // 60
        secs = seconds % 60
        return f"{int(minutes)}m {secs:.1f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        return f"{int(hours)}h {int(minutes)}m {secs:.1f}s"


def format_size(bytes_size: int) -> str:
    """
    Format size in bytes to human-readable string
    
    Args:
        bytes_size: Size in bytes
        
    Returns:
        Formatted size string
    """
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f}{unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f}PB"


def validate_dataset_name(dataset_name: str) -> bool:
    """
    Validate if a dataset name is supported
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        True if valid, False otherwise
    """
    valid_datasets = {
        'hotpotqa', '2wikimultihopqa', 'musique', 
        'nq', 'trivia', 'squad'
    }
    return dataset_name in valid_datasets


def validate_model_name(model_name: str) -> bool:
    """
    Validate if a model name is supported
    
    Args:
        model_name: Name of the model
        
    Returns:
        True if valid, False otherwise
    """
    valid_models = {
        'gemini-2.5-flash-lite', 'gemini-1.5-flash-8b'
    }
    return model_name in valid_models


def validate_strategy_name(strategy_name: str) -> bool:
    """
    Validate if a strategy name is supported
    
    Args:
        strategy_name: Name of the strategy
        
    Returns:
        True if valid, False otherwise
    """
    valid_strategies = {'original', 'optimized'}
    return strategy_name in valid_strategies


def get_dataset_type(dataset_name: str) -> str:
    """
    Get dataset type (single-hop or multi-hop)
    
    Args:
        dataset_name: Name of the dataset
        
    Returns:
        'single_hop' or 'multi_hop'
    """
    single_hop_datasets = {'nq', 'trivia', 'squad'}
    if dataset_name in single_hop_datasets:
        return 'single_hop'
    else:
        return 'multi_hop'


def safe_divide(numerator: float, denominator: float) -> float:
    """
    Safely divide two numbers, returning 0 if denominator is 0
    
    Args:
        numerator: Numerator
        denominator: Denominator
        
    Returns:
        Division result or 0 if denominator is 0
    """
    if denominator == 0:
        return 0.0
    return numerator / denominator


def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """
    Split a list into chunks of specified size
    
    Args:
        lst: List to chunk
        chunk_size: Size of each chunk
        
    Returns:
        List of chunks
    """
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def merge_dicts(dict1: Dict[str, Any], dict2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two dictionaries, with dict2 values taking precedence
    
    Args:
        dict1: First dictionary
        dict2: Second dictionary (takes precedence)
        
    Returns:
        Merged dictionary
    """
    result = dict1.copy()
    result.update(dict2)
    return result


def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """
    Flatten a nested dictionary
    
    Args:
        d: Dictionary to flatten
        parent_key: Parent key for recursion
        sep: Separator for nested keys
        
    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


class Timer:
    """Context manager for timing operations"""
    
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting {self.name}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        
        if self.logger:
            self.logger.info(f"{self.name} completed in {format_duration(duration)}")
        
        return False  # Don't suppress exceptions
    
    @property
    def duration(self) -> Optional[float]:
        """Get the duration of the timed operation"""
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None


class ProgressTracker:
    """Simple progress tracking utility"""
    
    def __init__(self, total: int, name: str = "Progress", logger: Optional[logging.Logger] = None):
        self.total = total
        self.name = name
        self.logger = logger
        self.current = 0
        self.start_time = time.time()
        self.last_update = 0
    
    def update(self, increment: int = 1):
        """Update progress by increment"""
        self.current += increment
        self.current = min(self.current, self.total)  # Don't exceed total
        
        # Log progress every 10% or every 100 items
        if (self.current - self.last_update >= 100 or 
            self.current / self.total >= (self.last_update / self.total) + 0.1):
            
            self._log_progress()
            self.last_update = self.current
    
    def _log_progress(self):
        """Log current progress"""
        if self.logger:
            percentage = (self.current / self.total) * 100
            elapsed = time.time() - self.start_time
            
            if self.current > 0:
                eta = (elapsed / self.current) * (self.total - self.current)
                self.logger.info(f"{self.name}: {self.current}/{self.total} ({percentage:.1f}%) - "
                               f"ETA: {format_duration(eta)}")
            else:
                self.logger.info(f"{self.name}: {self.current}/{self.total} ({percentage:.1f}%)")
    
    def finish(self):
        """Mark progress as finished"""
        self.current = self.total
        if self.logger:
            elapsed = time.time() - self.start_time
            self.logger.info(f"{self.name} completed: {self.total} items in {format_duration(elapsed)}")


class ConfigManager:
    """Configuration manager for loading and validating configs"""
    
    def __init__(self, config_dir: str):
        self.config_dir = Path(config_dir)
        self._configs = {}
    
    def load_config(self, config_name: str) -> Dict[str, Any]:
        """Load a configuration file"""
        if config_name not in self._configs:
            config_path = self.config_dir / f"{config_name}.json"
            self._configs[config_name] = read_json(config_path)
        
        return self._configs[config_name]
    
    def get_server_config(self) -> Dict[str, Any]:
        """Get server configuration"""
        return self.load_config("server_config")
    
    def get_experiment_config(self) -> Dict[str, Any]:
        """Get experiment configuration"""
        return self.load_config("experiment_config")
    
    def get_labeling_strategies_config(self) -> Dict[str, Any]:
        """Get labeling strategies configuration"""
        return self.load_config("labeling_strategies")
    
    def get_dataset_config(self, dataset_name: str) -> Dict[str, Any]:
        """Get configuration for a specific dataset"""
        exp_config = self.get_experiment_config()
        
        if dataset_name not in exp_config.get('datasets', []):
            raise ValueError(f"Dataset {dataset_name} not found in experiment configuration")
        
        # Return dataset-specific settings
        return {
            'dataset_name': dataset_name,
            'type': get_dataset_type(dataset_name),
            'retrieval_settings': exp_config.get('retrieval', {}),
            'processing_settings': exp_config.get('parallel_processing', {})
        }


def setup_environment() -> Dict[str, Any]:
    """
    Setup the environment for scaled silver labeling
    
    Returns:
        Dictionary with environment information
    """
    cache_dir = _get_cache_dir()
    
    env_info = {
        'timestamp': get_timestamp(),
        'working_directory': str(Path.cwd()),
        'python_version': f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
        'project_root': _get_project_root(),
        'adaptive_rag_root': _get_adaptive_rag_root(),
        'cache_directory': cache_dir,
        'processed_data_directory': _get_processed_data_path(),
        'conda_env': os.environ.get('CONDA_DEFAULT_ENV', 'adaptiverag')
    }
    
    # Ensure cache directory exists
    ensure_dir(cache_dir)
    
    # Ensure processed data directory exists
    ensure_dir(env_info['processed_data_directory'])
    
    return env_info


def validate_environment() -> List[str]:
    """
    Validate that the environment is properly set up
    
    Returns:
        List of validation errors (empty if all good)
    """
    errors = []
    
    # Check project root directory
    project_root = _get_project_root()
    if not os.path.exists(project_root):
        errors.append(f"Project root directory does not exist: {project_root}")
    
    # Check Adaptive-RAG directory
    adaptive_rag_root = _get_adaptive_rag_root()
    if not os.path.exists(adaptive_rag_root):
        errors.append(f"Adaptive-RAG directory does not exist: {adaptive_rag_root}")
    
    # Check cache directory
    cache_dir = _get_cache_dir()
    if not os.path.exists(cache_dir):
        errors.append(f"Cache directory does not exist: {cache_dir}")
    
    # Check if in correct conda environment
    conda_env = os.environ.get('CONDA_DEFAULT_ENV', '')
    expected_conda_env = 'adaptiverag'
    if conda_env != expected_conda_env:
        errors.append(f"Not in '{expected_conda_env}' conda environment (current: {conda_env})")
    
    # Check processed_data directory
    processed_data_dir = _get_processed_data_path()
    if not os.path.exists(processed_data_dir):
        errors.append(f"Processed data directory does not exist: {processed_data_dir}")
    
    return errors


# Constants for system configuration
SUPPORTED_DATASETS = {
    'hotpotqa', '2wikimultihopqa', 'musique', 
    'nq', 'trivia', 'squad'
}

SUPPORTED_MODELS = {
    'gemini-2.5-flash-lite', 'gemini-1.5-flash-8b'
}

SUPPORTED_STRATEGIES = {
    'original', 'optimized'
}

SUPPORTED_SYSTEMS = {
    'nor_qa', 'oner_qa', 'ircot_qa'
}

SINGLE_HOP_DATASETS = {'nq', 'trivia', 'squad'}
MULTI_HOP_DATASETS = {'hotpotqa', '2wikimultihopqa', 'musique'} 

def select_dynamic_few_shot_examples(
    dataset_name: str,
    test_question_ids: List[str],
    num_examples: int = 5,
    prompt_set: int = 1,
    train_data_path: str = None
) -> List[str]:
    """
    Dynamically select few-shot examples that don't overlap with test data.
    
    Args:
        dataset_name: Name of the dataset (e.g., "squad", "hotpotqa")
        test_question_ids: List of question IDs in the test set
        num_examples: Number of few-shot examples to select (default: 5)
        prompt_set: Prompt set number for reproducibility (1, 2, or 3)
        train_data_path: Path to training data file
        
    Returns:
        List of question IDs for few-shot examples that don't overlap with test set
    """
    # Set seed based on dataset and prompt set for reproducibility
    random.seed(13370 + hash(dataset_name) + prompt_set)
    
    # Default training data path if not provided
    if train_data_path is None:
        train_data_path = f"{_get_processed_data_path(dataset_name)}/train.jsonl"
    
    # Load training data
    if not os.path.exists(train_data_path):
        logger.warning(f"Training data not found at {train_data_path}")
        return []
    
    train_instances = read_jsonl(train_data_path)
    
    # Extract all available question IDs from training data
    available_question_ids = [instance["question_id"] for instance in train_instances]
    
    # Remove any IDs that appear in test set to prevent data leakage
    test_question_ids_set = set(test_question_ids)
    safe_question_ids = [
        qid for qid in available_question_ids 
        if qid not in test_question_ids_set
    ]
    
    # Select random subset for few-shot examples
    if len(safe_question_ids) < num_examples:
        logger.warning(
            f"Only {len(safe_question_ids)} safe examples available, "
            f"requested {num_examples}"
        )
        return safe_question_ids
    
    selected_ids = random.sample(safe_question_ids, num_examples)
    
    logger.info(
        f"Selected {len(selected_ids)} few-shot examples for {dataset_name} "
        f"prompt set {prompt_set} (avoiding {len(test_question_ids)} test IDs)"
    )
    
    return selected_ids


def validate_no_data_leakage(
    few_shot_ids: List[str],
    test_ids: List[str]
) -> bool:
    """
    Validate that few-shot examples don't overlap with test data.
    
    Args:
        few_shot_ids: Question IDs used in few-shot examples
        test_ids: Question IDs in test dataset
        
    Returns:
        True if no overlap, False if data leakage detected
    """
    overlap = set(few_shot_ids) & set(test_ids)
    
    if overlap:
        logger.error(f"DATA LEAKAGE DETECTED! Overlapping IDs: {overlap}")
        return False
    
    logger.info("✓ No data leakage detected - few-shot and test sets are disjoint")
    return True


def create_safe_dataset_splits(
    dataset_name: str,
    total_samples: int,
    few_shot_examples: int = 5,
    validation_split: float = 0.2
) -> Dict[str, List[str]]:
    """
    Create dataset splits ensuring no data leakage between few-shot examples and test data.
    
    Args:
        dataset_name: Name of the dataset
        total_samples: Total number of samples to use
        few_shot_examples: Number of few-shot examples needed
        validation_split: Fraction of data to use for validation
        
    Returns:
        Dictionary with 'train', 'test', 'validation', and 'few_shot' question IDs
    """
    # Load full dataset
    full_data_path = f"{_get_processed_data_path(dataset_name)}/test_subsampled.jsonl"
    full_instances = read_jsonl(full_data_path)
    
    # Extract all question IDs
    all_question_ids = [instance["question_id"] for instance in full_instances]
    
    # Randomly sample the requested number of samples
    if len(all_question_ids) < total_samples:
        logger.warning(
            f"Requested {total_samples} samples but only {len(all_question_ids)} available"
        )
        sampled_ids = all_question_ids
    else:
        random.seed(13370)  # For reproducibility
        sampled_ids = random.sample(all_question_ids, total_samples)
    
    # Split into validation and test
    validation_size = int(len(sampled_ids) * validation_split)
    validation_ids = sampled_ids[:validation_size]
    test_ids = sampled_ids[validation_size:]
    
    # Get few-shot examples from training data (ensuring no overlap)
    few_shot_ids = select_dynamic_few_shot_examples(
        dataset_name=dataset_name,
        test_question_ids=sampled_ids,  # Avoid overlap with ALL sampled data
        num_examples=few_shot_examples
    )
    
    # Validate no data leakage
    assert validate_no_data_leakage(few_shot_ids, sampled_ids)
    
    return {
        'test': test_ids,
        'validation': validation_ids,
        'few_shot': few_shot_ids,
        'total_samples': len(test_ids) + len(validation_ids)
    }


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    
    This normalization is crucial for fair evaluation, as it ensures that
    superficial differences in formatting don't penalize correct answers.
    Taken from squad_answer_em_f1.py for consistency with original evaluation.
    
    Args:
        s: Input string to normalize
        
    Returns:
        Normalized string with consistent formatting
    """
    def remove_articles(text):
        """Remove common articles (a, an, the) that don't affect answer correctness."""
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        """Normalize whitespace by splitting and rejoining."""
        return " ".join(text.split())

    def remove_punc(text):
        """Remove all punctuation marks."""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        """Convert to lowercase."""
        return text.lower()

    # Apply all normalization steps in sequence
    return white_space_fix(remove_articles(remove_punc(lower(s)))) 