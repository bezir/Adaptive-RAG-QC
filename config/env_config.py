#!/usr/bin/env python3
"""
Professional Environment Configuration for Adaptive RAG Project

This module provides centralized configuration for project paths and environment variables
using secure and configurable patterns.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Optional


def detect_project_root() -> str:
    """
    Automatically detect the project root directory using multiple strategies.
    
    Returns:
        str: Path to the project root directory
        
    Raises:
        RuntimeError: If project root cannot be determined
    """
    # Strategy 1: Use environment variable if set
    if os.environ.get('PROJECT_ROOT'):
        return os.environ['PROJECT_ROOT']
    
    # Strategy 2: Walk up from current file location
    current_path = Path(__file__).resolve()
    for parent in current_path.parents:
        if parent.name == 'Adaptive-RAG':
            return str(parent.parent)
        # Check for project markers
        if any((parent / marker).exists() for marker in [
            'adaptive_rag_benchmark', 'classifier', 'scaled_silver_labeling'
        ]):
            return str(parent.parent) if parent.name == 'Adaptive-RAG' else str(parent)
    
    # Strategy 3: Use current working directory if it looks right
    cwd = Path.cwd()
    if (cwd / 'Adaptive-RAG').exists():
        return str(cwd)
    if cwd.name == 'Adaptive-RAG' and cwd.parent.exists():
        return str(cwd.parent)
    
    raise RuntimeError(
        "Could not automatically detect PROJECT_ROOT. Please set the PROJECT_ROOT "
        "environment variable to point to your project root directory.\n"
        "Example: export PROJECT_ROOT=/path/to/your/project/root"
    )


def get_environment_config() -> Dict[str, Any]:
    """
    Get comprehensive environment configuration.
    
    Returns:
        Dictionary containing all environment settings
    """
    project_root = detect_project_root()
    adaptive_rag_root = f"{project_root}/Adaptive-RAG"
    cache_dir = f"{adaptive_rag_root}/.cache"
    
    return {
        'PROJECT_ROOT': project_root,
        'ADAPTIVE_RAG_ROOT': adaptive_rag_root,
        'CACHE_DIR': cache_dir,
        'HF_HOME': f"{cache_dir}/huggingface",
        'HF_DATASETS_CACHE': f"{cache_dir}/huggingface",
        'TRANSFORMERS_CACHE': f"{cache_dir}/transformers",
        'CONDA_ROOT': f"{project_root}/miniconda3",
        'PYTHONUNBUFFERED': '1',
        'TOKENIZERS_PARALLELISM': 'false',  # Avoid warnings in multi-processing
    }


def setup_environment(require_api_keys: bool = False) -> None:
    """
    Set up environment variables for the project.
    
    Args:
        require_api_keys: If True, will check for required API keys
    """
    config = get_environment_config()
    
    # Set all environment variables
    for key, value in config.items():
        os.environ[key] = value
    
    # Create cache directories
    cache_dirs = [
        config['CACHE_DIR'],
        config['HF_HOME'], 
        config['HF_DATASETS_CACHE'],
        config['TRANSFORMERS_CACHE']
    ]
    
    for cache_dir in cache_dirs:
        Path(cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Check for API keys if required
    if require_api_keys:
        missing_keys = []
        api_keys = ['GOOGLE_API_KEY', 'OPENAI_API_KEY']
        
        for key in api_keys:
            if not os.environ.get(key):
                missing_keys.append(key)
        
        if missing_keys:
            raise RuntimeError(
                f"Required API keys not found: {', '.join(missing_keys)}\n"
                "Please set these environment variables before running."
            )
    
    print("Environment configured successfully:")
    print(f"  PROJECT_ROOT: {config['PROJECT_ROOT']}")
    print(f"  ADAPTIVE_RAG_ROOT: {config['ADAPTIVE_RAG_ROOT']}")
    print(f"  CACHE_DIR: {config['CACHE_DIR']}")


def get_project_path(*paths: str) -> str:
    """Get a path relative to the project root."""
    config = get_environment_config()
    return os.path.join(config['PROJECT_ROOT'], *paths)


def get_adaptive_rag_path(*paths: str) -> str:
    """Get a path relative to the Adaptive RAG root."""
    config = get_environment_config()
    return os.path.join(config['ADAPTIVE_RAG_ROOT'], *paths)


if __name__ == "__main__":
    setup_environment()



