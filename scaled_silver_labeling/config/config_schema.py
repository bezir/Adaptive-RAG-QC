#!/usr/bin/env python3
"""
Configuration Schema Validation Module

This module provides schema validation for the configuration files used in the 
scaled silver labeling system. It ensures that all configuration files are 
properly formatted and contain required fields.
"""

import json
import os
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from jsonschema import validate, ValidationError


@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: List[str]
    warnings: List[str]


class ConfigSchemaValidator:
    """Validates configuration files against JSON schemas"""
    
    def __init__(self):
        self.schemas = {
            'server_config': self._get_server_config_schema(),
            'experiment_config': self._get_experiment_config_schema(),
            'labeling_strategies': self._get_labeling_strategies_schema()
        }
    
    def _get_server_config_schema(self) -> Dict[str, Any]:
        """Get schema for server configuration"""
        return {
            "type": "object",
            "required": ["llm_servers", "server_pool", "load_balancer"],
            "properties": {
                "llm_servers": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "required": ["id", "model", "host", "port", "gpu_id"],
                        "properties": {
                            "id": {"type": "string"},
                            "model": {"type": "string"},
                            "host": {"type": "string"},
                            "port": {"type": "integer", "minimum": 1, "maximum": 65535},
                            "gpu_id": {"type": "integer", "minimum": 0},
                            "max_concurrent": {"type": "integer", "minimum": 1},
                            "timeout": {"type": "integer", "minimum": 1}
                        }
                    }
                },
                "server_pool": {
                    "type": "object",
                    "required": ["total_servers", "num_gpus", "models_per_gpu", "base_port"],
                    "properties": {
                        "total_servers": {"type": "integer", "minimum": 1},
                        "num_gpus": {"type": "integer", "minimum": 1},
                        "models_per_gpu": {"type": "integer", "minimum": 1},
                        "base_port": {"type": "integer", "minimum": 1, "maximum": 65535}
                    }
                },
                "load_balancer": {
                    "type": "object",
                    "required": ["strategy"],
                    "properties": {
                        "strategy": {"type": "string", "enum": ["round_robin", "least_loaded", "random"]},
                        "max_queue_size": {"type": "integer", "minimum": 1},
                        "request_timeout": {"type": "integer", "minimum": 1}
                    }
                }
            }
        }
    
    def _get_experiment_config_schema(self) -> Dict[str, Any]:
        """Get schema for experiment configuration"""
        return {
            "type": "object",
            "required": ["sample_sizes", "datasets", "models", "labeling_strategies", "paths"],
            "properties": {
                "sample_sizes": {
                    "type": "array",
                    "items": {"type": "integer", "minimum": 1}
                },
                "datasets": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad"]}
                },
                "models": {
                    "type": "array",
                    "items": {"type": "string"}
                },
                "labeling_strategies": {
                    "type": "array",
                    "items": {"type": "string", "enum": ["original", "optimized"]}
                },
                "paths": {
                    "type": "object",
                    "required": ["cache_directory", "output_directory", "processed_data_directory"],
                    "properties": {
                        "cache_directory": {"type": "string"},
                        "output_directory": {"type": "string"},
                        "processed_data_directory": {"type": "string"}
                    }
                }
            }
        }
    
    def _get_labeling_strategies_schema(self) -> Dict[str, Any]:
        """Get schema for labeling strategies configuration"""
        return {
            "type": "object",
            "required": ["original", "optimized"],
            "properties": {
                "original": {
                    "type": "object",
                    "required": ["systems", "priority", "label_assignment"],
                    "properties": {
                        "systems": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["nor_qa", "oner_qa", "ircot_qa"]}
                        },
                        "priority": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["nor_qa", "oner_qa", "ircot_qa"]}
                        },
                        "label_assignment": {
                            "type": "object",
                            "required": ["nor_qa_success", "oner_qa_success", "ircot_qa_success"],
                            "properties": {
                                "nor_qa_success": {"type": "string", "enum": ["A", "B", "C"]},
                                "oner_qa_success": {"type": "string", "enum": ["A", "B", "C"]},
                                "ircot_qa_success": {"type": "string", "enum": ["A", "B", "C"]}
                            }
                        }
                    }
                },
                "optimized": {
                    "type": "object",
                    "required": ["single_hop_datasets", "multi_hop_datasets", "system_selection"],
                    "properties": {
                        "single_hop_datasets": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["nq", "trivia", "squad"]}
                        },
                        "multi_hop_datasets": {
                            "type": "array",
                            "items": {"type": "string", "enum": ["hotpotqa", "2wikimultihopqa", "musique"]}
                        },
                        "system_selection": {
                            "type": "object",
                            "required": ["single_hop", "multi_hop"],
                            "properties": {
                                "single_hop": {
                                    "type": "object",
                                    "required": ["systems", "label_assignment"],
                                    "properties": {
                                        "systems": {
                                            "type": "array",
                                            "items": {"type": "string", "enum": ["nor_qa", "oner_qa", "ircot_qa"]}
                                        }
                                    }
                                },
                                "multi_hop": {
                                    "type": "object",
                                    "required": ["systems", "label_assignment"],
                                    "properties": {
                                        "systems": {
                                            "type": "array",
                                            "items": {"type": "string", "enum": ["nor_qa", "oner_qa", "ircot_qa"]}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    
    def validate_config(self, config_path: str, config_type: str) -> ConfigValidationResult:
        """
        Validate a configuration file against its schema
        
        Args:
            config_path: Path to the configuration file
            config_type: Type of configuration ('server_config', 'experiment_config', 'labeling_strategies')
            
        Returns:
            ConfigValidationResult object with validation results
        """
        errors = []
        warnings = []
        
        try:
            # Load configuration file
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            
            # Validate against schema
            if config_type not in self.schemas:
                errors.append(f"Unknown configuration type: {config_type}")
                return ConfigValidationResult(False, errors, warnings)
            
            validate(config_data, self.schemas[config_type])
            
            # Additional validation checks
            if config_type == 'server_config':
                errors.extend(self._validate_server_config(config_data))
            elif config_type == 'experiment_config':
                errors.extend(self._validate_experiment_config(config_data))
            elif config_type == 'labeling_strategies':
                errors.extend(self._validate_labeling_strategies(config_data))
            
            return ConfigValidationResult(len(errors) == 0, errors, warnings)
            
        except FileNotFoundError:
            errors.append(f"Configuration file not found: {config_path}")
        except json.JSONDecodeError as e:
            errors.append(f"Invalid JSON in configuration file: {str(e)}")
        except ValidationError as e:
            errors.append(f"Schema validation failed: {str(e)}")
        except Exception as e:
            errors.append(f"Unexpected error during validation: {str(e)}")
        
        return ConfigValidationResult(False, errors, warnings)
    
    def _validate_server_config(self, config: Dict[str, Any]) -> List[str]:
        """Additional validation for server configuration"""
        errors = []
        
        # Check for duplicate ports
        ports = [server['port'] for server in config['llm_servers']]
        if len(ports) != len(set(ports)):
            errors.append("Duplicate ports found in server configuration")
        
        # Check for duplicate server IDs
        server_ids = [server['id'] for server in config['llm_servers']]
        if len(server_ids) != len(set(server_ids)):
            errors.append("Duplicate server IDs found in server configuration")
        
        return errors
    
    def _validate_experiment_config(self, config: Dict[str, Any]) -> List[str]:
        """Additional validation for experiment configuration"""
        errors = []
        
        # Check if paths exist (cache directory should exist)
        cache_dir = config['paths']['cache_directory']
        if not os.path.exists(cache_dir):
            errors.append(f"Cache directory does not exist: {cache_dir}")
        
        return errors
    
    def _validate_labeling_strategies(self, config: Dict[str, Any]) -> List[str]:
        """Additional validation for labeling strategies configuration"""
        errors = []
        
        # Check that single_hop + multi_hop datasets cover all expected datasets
        expected_datasets = {"hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad"}
        single_hop = set(config['optimized']['single_hop_datasets'])
        multi_hop = set(config['optimized']['multi_hop_datasets'])
        
        if single_hop.union(multi_hop) != expected_datasets:
            errors.append("Single-hop and multi-hop datasets don't cover all expected datasets")
        
        if single_hop.intersection(multi_hop):
            errors.append("Dataset appears in both single-hop and multi-hop lists")
        
        return errors


def validate_all_configs(config_dir: str) -> Dict[str, ConfigValidationResult]:
    """
    Validate all configuration files in a directory
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Dictionary mapping config file names to validation results
    """
    validator = ConfigSchemaValidator()
    results = {}
    
    config_files = {
        'server_config.json': 'server_config',
        'experiment_config.json': 'experiment_config',
        'labeling_strategies.json': 'labeling_strategies'
    }
    
    for filename, config_type in config_files.items():
        config_path = os.path.join(config_dir, filename)
        results[filename] = validator.validate_config(config_path, config_type)
    
    return results


if __name__ == "__main__":
    """Command-line interface for configuration validation"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Validate configuration files")
    parser.add_argument("--config-dir", default="config", help="Configuration directory")
    parser.add_argument("--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    results = validate_all_configs(args.config_dir)
    
    all_valid = True
    for filename, result in results.items():
        if result.is_valid:
            print(f"✓ {filename}: Valid")
        else:
            print(f"✗ {filename}: Invalid")
            all_valid = False
            
        if args.verbose or not result.is_valid:
            for error in result.errors:
                print(f"  ERROR: {error}")
            for warning in result.warnings:
                print(f"  WARNING: {warning}")
    
    if all_valid:
        print("\nAll configuration files are valid!")
    else:
        print("\nSome configuration files have errors. Please fix them before proceeding.")
        exit(1) 