#!/usr/bin/env python3
"""
Configuration Validation Module for Scaled Silver Labeling

This module provides validation for experiment configurations to ensure
all required parameters are present and valid before running experiments.
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import jsonschema
from jsonschema import validate, ValidationError


class ConfigValidator:
    """Validates experiment configurations against schema"""
    
    def __init__(self):
        self.schema = self._get_config_schema()
    
    def _get_config_schema(self) -> Dict[str, Any]:
        """Define the JSON schema for experiment configuration"""
        return {
            "type": "object",
            "properties": {
                "experiment_name": {
                    "type": "string",
                    "minLength": 1
                },
                "description": {
                    "type": "string"
                },
                "sample_sizes": {
                    "type": "array",
                    "items": {
                        "type": "integer",
                        "minimum": 100,
                        "maximum": 10000
                    },
                    "minItems": 1
                },
                "datasets": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad"]
                    },
                    "minItems": 1
                },
                "models": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["flan-t5-xl", "flan-t5-xxl", "gemini", "gemini-2.5-flash-lite", "gemini-1.5-flash-8b", "qwen"]
                    },
                    "minItems": 1
                },
                "labeling_strategies": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["original", "optimized"]
                    },
                    "minItems": 1
                },
                "execution_config": {
                    "type": "object",
                    "properties": {
                        "max_parallel_tasks": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 20
                        },
                        "task_timeout": {
                            "type": "integer",
                            "minimum": 300,
                            "maximum": 7200
                        },
                        "retry_attempts": {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": 10
                        },
                        "retry_delay": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 300
                        }
                    },
                    "required": ["max_parallel_tasks", "task_timeout", "retry_attempts", "retry_delay"]
                },
                "output_config": {
                    "type": "object",
                    "properties": {
                        "output_directory": {
                            "type": "string",
                            "minLength": 1
                        },
                        "log_directory": {
                            "type": "string",
                            "minLength": 1
                        },
                        "save_intermediate_results": {
                            "type": "boolean"
                        },
                        "compress_results": {
                            "type": "boolean"
                        }
                    },
                    "required": ["output_directory", "log_directory"]
                },
                "cache_config": {
                    "type": "object",
                    "properties": {
                        "cache_directory": {
                            "type": "string",
                            "minLength": 1
                        },
                        "enable_caching": {
                            "type": "boolean"
                        },
                        "cache_expiry_hours": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 168
                        }
                    },
                    "required": ["cache_directory", "enable_caching"]
                },
                "validation_config": {
                    "type": "object",
                    "properties": {
                        "validate_no_overlap": {
                            "type": "boolean"
                        },
                        "validate_sample_sizes": {
                            "type": "boolean"
                        },
                        "validate_file_integrity": {
                            "type": "boolean"
                        }
                    }
                },
                "server_config": {
                    "type": "object",
                    "properties": {
                        "server_config_path": {
                            "type": ["string", "null"]
                        },
                        "health_check_interval": {
                            "type": "integer",
                            "minimum": 60,
                            "maximum": 3600
                        },
                        "max_retries": {
                            "type": "integer",
                            "minimum": 1,
                            "maximum": 10
                        }
                    }
                },
                "logging_config": {
                    "type": "object",
                    "properties": {
                        "log_level": {
                            "type": "string",
                            "enum": ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
                        },
                        "log_format": {
                            "type": "string"
                        },
                        "enable_llm_interaction_logging": {
                            "type": "boolean"
                        },
                        "enable_progress_tracking": {
                            "type": "boolean"
                        },
                        "save_logs_to_file": {
                            "type": "boolean"
                        }
                    }
                },
                "performance_config": {
                    "type": "object",
                    "properties": {
                        "monitor_throughput": {
                            "type": "boolean"
                        },
                        "calculate_efficiency_gains": {
                            "type": "boolean"
                        },
                        "track_resource_usage": {
                            "type": "boolean"
                        }
                    }
                }
            },
            "required": [
                "experiment_name",
                "sample_sizes",
                "datasets",
                "models",
                "labeling_strategies",
                "execution_config",
                "output_config",
                "cache_config"
            ]
        }
    
    def validate_config(self, config: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """Validate configuration against schema"""
        errors = []
        
        try:
            # Schema validation
            validate(instance=config, schema=self.schema)
            
            # Additional business logic validation
            business_errors = self._validate_business_logic(config)
            errors.extend(business_errors)
            
            # Path validation
            path_errors = self._validate_paths(config)
            errors.extend(path_errors)
            
            # Dependency validation
            dependency_errors = self._validate_dependencies(config)
            errors.extend(dependency_errors)
            
        except ValidationError as e:
            errors.append(f"Schema validation error: {e.message}")
        
        return len(errors) == 0, errors
    
    def _validate_business_logic(self, config: Dict[str, Any]) -> List[str]:
        """Validate business logic constraints"""
        errors = []
        
        # Check for valid dataset-strategy combinations
        datasets = config.get('datasets', [])
        strategies = config.get('labeling_strategies', [])
        
        single_hop_datasets = {'nq', 'trivia', 'squad'}
        multi_hop_datasets = {'hotpotqa', '2wikimultihopqa', 'musique'}
        
        if 'optimized' in strategies:
            for dataset in datasets:
                if dataset in single_hop_datasets:
                    # Single-hop datasets should only use NOR system
                    pass  # Valid
                elif dataset in multi_hop_datasets:
                    # Multi-hop datasets should use NOR + ONER
                    pass  # Valid
                else:
                    errors.append(f"Unknown dataset type: {dataset}")
        
        # Check sample size constraints
        sample_sizes = config.get('sample_sizes', [])
        for size in sample_sizes:
            if size < 100:
                errors.append(f"Sample size {size} is too small (minimum: 100)")
            elif size > 10000:
                errors.append(f"Sample size {size} is too large (maximum: 10000)")
        
        # Check model availability
        models = config.get('models', [])
        if any(m in models for m in ["gemini", "qwen"]) and "OPENAI_API_KEY" not in os.environ:
            errors.append("OpenAI API key (or equivalent) required for generative models")
        
        return errors
    
    def _validate_paths(self, config: Dict[str, Any]) -> List[str]:
        """Validate file and directory paths"""
        errors = []
        
        # Check cache directory
        cache_dir = config.get('cache_config', {}).get('cache_directory')
        if cache_dir and not Path(cache_dir).exists():
            errors.append(f"Cache directory does not exist: {cache_dir}")
        
        # Check server config path
        server_config_path = config.get('server_config', {}).get('server_config_path')
        if server_config_path and not Path(server_config_path).exists():
            errors.append(f"Server config file does not exist: {server_config_path}")
        
        # Check output directory can be created
        output_dir = config.get('output_config', {}).get('output_directory')
        if output_dir:
            try:
                Path(output_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create output directory {output_dir}: {str(e)}")
        
        # Check log directory can be created
        log_dir = config.get('output_config', {}).get('log_directory')
        if log_dir:
            try:
                Path(log_dir).mkdir(parents=True, exist_ok=True)
            except Exception as e:
                errors.append(f"Cannot create log directory {log_dir}: {str(e)}")
        
        return errors
    
    def _validate_dependencies(self, config: Dict[str, Any]) -> List[str]:
        """Validate system dependencies"""
        errors = []
        
        # Check required Python packages
        required_packages = ['jsonschema', 'requests', 'asyncio']
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                errors.append(f"Required package not installed: {package}")
        
        # Check for processed data directories
        datasets = config.get('datasets', [])
        for dataset in datasets:
            # Import here to avoid circular import
            from ..utils.common import _get_processed_data_path
            data_path = Path(_get_processed_data_path(dataset))
            if not data_path.exists():
                errors.append(f"Processed data directory not found: {data_path}")
        
        return errors
    
    def validate_config_file(self, config_path: str) -> Tuple[bool, List[str], Optional[Dict[str, Any]]]:
        """Validate configuration file"""
        errors = []
        config = None
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
        except FileNotFoundError:
            errors.append(f"Configuration file not found: {config_path}")
            return False, errors, None
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in configuration file: {str(e)}")
            return False, errors, None
        
        # Validate the loaded configuration
        is_valid, validation_errors = self.validate_config(config)
        errors.extend(validation_errors)
        
        return is_valid, errors, config
    
    def create_minimal_config(self, output_path: str) -> Dict[str, Any]:
        """Create a minimal valid configuration"""
        minimal_config = {
            "experiment_name": "minimal_experiment",
            "description": "Minimal configuration for testing",
            "sample_sizes": [1000],
            "datasets": ["hotpotqa"],
            "models": ["flan-t5-xl"],
            "labeling_strategies": ["original"],
            "execution_config": {
                "max_parallel_tasks": 1,
                "task_timeout": 3600,
                "retry_attempts": 3,
                "retry_delay": 60
            },
            "output_config": {
                "output_directory": "predictions",
                "log_directory": "logs",
                "save_intermediate_results": True,
                "compress_results": False
            },
            "cache_config": {
                "cache_directory": f"{os.environ.get('PROJECT_ROOT', os.path.expanduser('~'))}/Adaptive-RAG/.cache",
                "enable_caching": True,
                "cache_expiry_hours": 24
            },
            "validation_config": {
                "validate_no_overlap": True,
                "validate_sample_sizes": True,
                "validate_file_integrity": True
            },
            "server_config": {
                "server_config_path": None,
                "health_check_interval": 300,
                "max_retries": 3
            },
            "logging_config": {
                "log_level": "INFO",
                "enable_llm_interaction_logging": True,
                "enable_progress_tracking": True,
                "save_logs_to_file": True
            },
            "performance_config": {
                "monitor_throughput": True,
                "calculate_efficiency_gains": True,
                "track_resource_usage": False
            }
        }
        
        with open(output_path, 'w') as f:
            json.dump(minimal_config, f, indent=2)
        
        return minimal_config
    
    def get_config_template(self) -> Dict[str, Any]:
        """Get a template configuration with descriptions"""
        return {
            "_comment": "This is a configuration template for scaled silver labeling experiments",
            "experiment_name": "my_experiment",
            "description": "Description of the experiment",
            "sample_sizes": [1000, 2000, 5000],
            "datasets": ["hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad"],
            "models": ["flan-t5-xl", "flan-t5-xxl"],
            "labeling_strategies": ["original", "optimized"],
            "execution_config": {
                "_comment": "Configuration for experiment execution. max_parallel_tasks=1 is conservative since no health checking is implemented",
                "max_parallel_tasks": 1,
                "task_timeout": 3600,
                "retry_attempts": 3,
                "retry_delay": 60
            },
            "output_config": {
                "_comment": "Configuration for output handling",
                "output_directory": "predictions",
                "log_directory": "logs",
                "save_intermediate_results": True,
                "compress_results": False
            },
            "cache_config": {
                "_comment": "Configuration for caching",
                "cache_directory": f"{os.environ.get('PROJECT_ROOT', os.path.expanduser('~'))}/Adaptive-RAG/.cache",
                "enable_caching": True,
                "cache_expiry_hours": 24
            },
            "validation_config": {
                "_comment": "Configuration for data validation",
                "validate_no_overlap": True,
                "validate_sample_sizes": True,
                "validate_file_integrity": True
            },
            "server_config": {
                "_comment": "Configuration for LLM servers",
                "server_config_path": "config/server_config.json",
                "health_check_interval": 300,
                "max_retries": 3
            },
            "logging_config": {
                "_comment": "Configuration for logging",
                "log_level": "INFO",
                "enable_llm_interaction_logging": True,
                "enable_progress_tracking": True,
                "save_logs_to_file": True
            },
            "performance_config": {
                "_comment": "Configuration for performance monitoring",
                "monitor_throughput": True,
                "calculate_efficiency_gains": True,
                "track_resource_usage": False
            }
        }


def main():
    """Command line interface for configuration validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate experiment configuration")
    parser.add_argument('config_path', help='Path to configuration file')
    parser.add_argument('--create-template', action='store_true', 
                       help='Create a template configuration file')
    parser.add_argument('--create-minimal', action='store_true',
                       help='Create a minimal configuration file')
    
    args = parser.parse_args()
    
    validator = ConfigValidator()
    
    if args.create_template:
        template = validator.get_config_template()
        with open(args.config_path, 'w') as f:
            json.dump(template, f, indent=2)
        print(f"Template configuration created: {args.config_path}")
        return
    
    if args.create_minimal:
        minimal = validator.create_minimal_config(args.config_path)
        print(f"Minimal configuration created: {args.config_path}")
        return
    
    # Validate existing configuration
    is_valid, errors, config = validator.validate_config_file(args.config_path)
    
    if is_valid:
        print("Configuration is valid!")
    else:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
    
    return 0 if is_valid else 1


if __name__ == "__main__":
    import sys
    sys.exit(main()) 