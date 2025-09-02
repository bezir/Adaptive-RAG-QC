#!/usr/bin/env python3
"""
Comprehensive Reporting System for Scaled Silver Labeling

This module provides comprehensive reporting capabilities for labeling
experiments, including performance analysis, efficiency comparisons, and detailed
experiment summaries.
"""

import json
import os
import statistics
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
# Visualization imports removed - no HTML generation
# import matplotlib.pyplot as plt
# import seaborn as sns  
# import pandas as pd
from dataclasses import dataclass
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class ExperimentSummary:
    """Summary of a single experiment"""
    experiment_id: str
    dataset: str
    model: str
    strategy: str
    sample_size: int
    execution_time: float
    success: bool
    efficiency_gain: Optional[float] = None
    error_message: Optional[str] = None


class ComprehensiveReporter:
    """Generates comprehensive reports for scaled silver labeling experiments"""
    
    def __init__(self, log_dir: str = "logs", output_dir: str = "reports"):
        self.log_dir = Path(log_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Report configuration - JSON/text output only
        self.report_config = {
            'include_charts': False,
            'include_statistics': True,
            'include_efficiency_analysis': True,
            'include_error_analysis': True,
            'output_format': 'json'
        }
    
    def generate_experiment_report(self, experiment_data: Dict[str, Any], 
                                 output_filename: Optional[str] = None) -> Path:
        """Generate comprehensive report for a single experiment"""
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"experiment_report_{timestamp}.json"
        
        report_path = self.output_dir / output_filename
        
        # Parse experiment data
        summaries = self._parse_experiment_data(experiment_data)
        
        # Generate efficiency analysis
        efficiency_analysis = self.generate_efficiency_analysis(experiment_data)
        
        # Create comprehensive report data
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'experiment_id': experiment_data.get('experiment_id', 'N/A'),
            'summary': {
                'total_experiments': len(summaries),
                'successful_experiments': sum(1 for s in summaries if s.success),
                'failed_experiments': sum(1 for s in summaries if not s.success),
                'success_rate': (sum(1 for s in summaries if s.success) / len(summaries)) if summaries else 0
            },
            'efficiency_analysis': efficiency_analysis,
            'experiment_details': [
                {
                    'experiment_id': s.experiment_id,
                    'dataset': s.dataset,
                    'model': s.model,
                    'strategy': s.strategy,
                    'sample_size': s.sample_size,
                    'execution_time': s.execution_time,
                    'success': s.success,
                    'efficiency_gain': s.efficiency_gain,
                    'error_message': s.error_message
                }
                for s in summaries
            ]
        }
        
        with open(report_path, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        logger.info(f"Experiment report generated: {report_path}")
        return report_path
    
    def generate_comparison_report(self, experiment_data_list: List[Dict[str, Any]], 
                                 output_filename: Optional[str] = None) -> Path:
        """Generate comparison report for multiple experiments"""
        
        if output_filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_filename = f"comparison_report_{timestamp}.json"
        
        report_path = self.output_dir / output_filename
        
        # Parse all experiment data
        all_summaries = []
        for exp_data in experiment_data_list:
            summaries = self._parse_experiment_data(exp_data)
            all_summaries.extend(summaries)
        
        # Group summaries by experiment
        experiments = {}
        for summary in all_summaries:
            exp_key = f"{summary.dataset}_{summary.model}_{summary.sample_size}"
            if exp_key not in experiments:
                experiments[exp_key] = []
            experiments[exp_key].append(summary)
        
        # Generate comparison data
        comparison_data = {
            'timestamp': datetime.now().isoformat(),
            'total_experiments': len(experiments),
            'comparisons': []
        }
        
        for exp_key, summaries in experiments.items():
            original_summary = None
            optimized_summary = None
            
            for summary in summaries:
                if summary.strategy == 'original':
                    original_summary = summary
                elif summary.strategy == 'optimized':
                    optimized_summary = summary
            
            if original_summary and optimized_summary:
                efficiency_gain = ((original_summary.execution_time - optimized_summary.execution_time) / original_summary.execution_time) * 100 if original_summary.execution_time > 0 else 0
                
                comparison_data['comparisons'].append({
                    'experiment': exp_key,
                    'original_strategy': {
                        'execution_time': original_summary.execution_time,
                        'success': original_summary.success
                    },
                    'optimized_strategy': {
                        'execution_time': optimized_summary.execution_time,
                        'success': optimized_summary.success
                    },
                    'efficiency_gain_percent': efficiency_gain
                })
        
        with open(report_path, 'w') as f:
            json.dump(comparison_data, f, indent=2)
        
        logger.info(f"Comparison report generated: {report_path}")
        return report_path
    
    def generate_efficiency_analysis(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate detailed efficiency analysis"""
        
        summaries = self._parse_experiment_data(experiment_data)
        
        # Group by dataset and strategy
        strategy_performance = {}
        dataset_performance = {}
        
        for summary in summaries:
            # Strategy performance
            if summary.strategy not in strategy_performance:
                strategy_performance[summary.strategy] = []
            strategy_performance[summary.strategy].append(summary.execution_time)
            
            # Dataset performance
            if summary.dataset not in dataset_performance:
                dataset_performance[summary.dataset] = {}
            if summary.strategy not in dataset_performance[summary.dataset]:
                dataset_performance[summary.dataset][summary.strategy] = []
            dataset_performance[summary.dataset][summary.strategy].append(summary.execution_time)
        
        # Calculate statistics
        efficiency_analysis = {
            'strategy_comparison': self._calculate_strategy_statistics(strategy_performance),
            'dataset_analysis': self._calculate_dataset_statistics(dataset_performance),
            'overall_efficiency': self._calculate_overall_efficiency(summaries)
        }
        
        return efficiency_analysis
    
    def generate_performance_summary(self, experiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate performance summary data (no charts)"""
        
        summaries = self._parse_experiment_data(experiment_data)
        
        # Calculate performance metrics
        strategy_performance = {}
        dataset_performance = {}
        success_rates = {}
        
        for summary in summaries:
            # Strategy performance
            if summary.strategy not in strategy_performance:
                strategy_performance[summary.strategy] = []
            strategy_performance[summary.strategy].append(summary.execution_time)
            
            # Dataset performance  
            if summary.dataset not in dataset_performance:
                dataset_performance[summary.dataset] = {}
            if summary.strategy not in dataset_performance[summary.dataset]:
                dataset_performance[summary.dataset][summary.strategy] = []
            dataset_performance[summary.dataset][summary.strategy].append(summary.execution_time)
            
            # Success rates
            if summary.strategy not in success_rates:
                success_rates[summary.strategy] = {'success': 0, 'total': 0}
            success_rates[summary.strategy]['total'] += 1
            if summary.success:
                success_rates[summary.strategy]['success'] += 1
        
        return {
            'strategy_performance': self._calculate_strategy_statistics(strategy_performance),
            'dataset_performance': self._calculate_dataset_statistics(dataset_performance),
            'success_rates': {
                strategy: (data['success'] / data['total']) * 100 if data['total'] > 0 else 0
                for strategy, data in success_rates.items()
            }
        }
    
    def _parse_experiment_data(self, experiment_data: Dict[str, Any]) -> List[ExperimentSummary]:
        """Parse experiment data into structured summaries"""
        
        summaries = []
        task_results = experiment_data.get('task_results', {})
        
        for task_id, task_result in task_results.items():
            # Parse task ID - handle both old and new formats
            parts = task_id.split('_')
            if len(parts) >= 4:
                dataset = parts[0]
                model = parts[1]
                strategy = parts[2]
                
                # Handle new format: dataset_model_strategy_full_annotation or dataset_model_strategy_subsampled_128
                if len(parts) >= 5 and parts[3] == 'full':
                    # Format: dataset_model_strategy_full_annotation
                    sample_size = 'all'
                elif len(parts) >= 5 and parts[3] == 'subsampled':
                    # Format: dataset_model_strategy_subsampled_128
                    try:
                        sample_size = int(parts[4])
                    except (ValueError, IndexError):
                        sample_size = 'unknown'
                else:
                    # Legacy format: dataset_model_strategy_128
                    try:
                        sample_size = int(parts[3])
                    except ValueError:
                        sample_size = parts[3]  # Keep as string if not numeric
            else:
                continue
            
            # Extract metadata
            metadata = task_result.get('metadata', {})
            execution_time = metadata.get('execution_time', 0)
            success = task_result.get('status') == 'completed'
            error_message = task_result.get('error') if not success else None
            
            summary = ExperimentSummary(
                experiment_id=task_id,
                dataset=dataset,
                model=model,
                strategy=strategy,
                sample_size=sample_size,
                execution_time=execution_time,
                success=success,
                error_message=error_message
            )
            
            summaries.append(summary)
        
        # Calculate efficiency gains
        self._calculate_efficiency_gains(summaries)
        
        return summaries
    
    def _calculate_efficiency_gains(self, summaries: List[ExperimentSummary]):
        """Calculate efficiency gains between strategies"""
        
        # Group by dataset and model
        grouped = {}
        for summary in summaries:
            key = f"{summary.dataset}_{summary.model}_{summary.sample_size}"
            if key not in grouped:
                grouped[key] = {}
            grouped[key][summary.strategy] = summary
        
        # Calculate efficiency gains
        for key, strategies in grouped.items():
            if 'original' in strategies and 'optimized' in strategies:
                original_time = strategies['original'].execution_time
                optimized_time = strategies['optimized'].execution_time
                
                if original_time > 0:
                    efficiency_gain = ((original_time - optimized_time) / original_time) * 100
                    strategies['optimized'].efficiency_gain = efficiency_gain
    
    def _calculate_strategy_statistics(self, strategy_performance: Dict[str, List[float]]) -> Dict[str, Any]:
        """Calculate statistics for strategy performance"""
        
        stats = {}
        for strategy, times in strategy_performance.items():
            if times:
                stats[strategy] = {
                    'avg_execution_time': statistics.mean(times),
                    'median_execution_time': statistics.median(times),
                    'min_execution_time': min(times),
                    'max_execution_time': max(times),
                    'std_execution_time': statistics.stdev(times) if len(times) > 1 else 0,
                    'total_experiments': len(times)
                }
        
        return stats
    
    def _calculate_dataset_statistics(self, dataset_performance: Dict[str, Dict[str, List[float]]]) -> Dict[str, Any]:
        """Calculate statistics for dataset performance"""
        
        stats = {}
        for dataset, strategies in dataset_performance.items():
            stats[dataset] = {}
            for strategy, times in strategies.items():
                if times:
                    stats[dataset][strategy] = {
                        'avg_execution_time': statistics.mean(times),
                        'median_execution_time': statistics.median(times),
                        'total_experiments': len(times)
                    }
        
        return stats
    
    def _calculate_overall_efficiency(self, summaries: List[ExperimentSummary]) -> Dict[str, Any]:
        """Calculate overall efficiency metrics"""
        
        total_experiments = len(summaries)
        successful_experiments = sum(1 for s in summaries if s.success)
        
        efficiency_gains = [s.efficiency_gain for s in summaries if s.efficiency_gain is not None]
        
        overall_stats = {
            'total_experiments': total_experiments,
            'successful_experiments': successful_experiments,
            'success_rate': successful_experiments / total_experiments if total_experiments > 0 else 0,
            'average_efficiency_gain': statistics.mean(efficiency_gains) if efficiency_gains else 0,
            'median_efficiency_gain': statistics.median(efficiency_gains) if efficiency_gains else 0,
            'total_time_saved': sum(efficiency_gains) if efficiency_gains else 0
        }
        
        return overall_stats
    