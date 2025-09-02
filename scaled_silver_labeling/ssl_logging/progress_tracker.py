#!/usr/bin/env python3
"""
Progress Tracking Module for Labeling System

This module provides comprehensive progress tracking for labeling experiments,
including time estimation, throughput monitoring, and detailed progress reporting.
"""

import time
import json
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from dataclasses import dataclass
from pathlib import Path
import statistics


@dataclass
class TaskProgress:
    """Represents progress for a single task"""
    task_id: str
    task_type: str
    dataset_name: str
    model_name: str
    strategy: str
    sample_size: int
    
    # Progress tracking
    total_samples: int = 0
    completed_samples: int = 0
    failed_samples: int = 0
    
    # Timing information
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    last_update: Optional[float] = None
    
    # Performance metrics
    samples_per_second: float = 0.0
    estimated_completion: Optional[float] = None
    
    def __post_init__(self):
        self.start_time = time.time()
        self.last_update = self.start_time


class ProgressTracker:
    """Main progress tracking class for scaled labeling experiments"""
    
    def __init__(self, log_dir: str = "logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Thread-safe storage for progress data
        self._lock = threading.Lock()
        self.tasks: Dict[str, TaskProgress] = {}
        self.global_stats = {
            'total_experiments': 0,
            'completed_experiments': 0,
            'failed_experiments': 0,
            'total_samples_processed': 0,
            'start_time': None,
            'last_update': None
        }
        
        # Performance tracking
        self.throughput_history: List[float] = []
        self.timing_history: List[Dict[str, Any]] = []
        
        # Session tracking
        self.sessions: Dict[str, Dict[str, Any]] = {}
        
    def start_experiment(self, experiment_config: Dict[str, Any]) -> str:
        """Start tracking a new experiment"""
        with self._lock:
            if self.global_stats['start_time'] is None:
                self.global_stats['start_time'] = time.time()
            
            self.global_stats['total_experiments'] += 1
            self.global_stats['last_update'] = time.time()
            
            # Calculate total expected tasks
            # Handle both new single sample_size and legacy sample_sizes array
            if 'sample_size' in experiment_config:
                sample_sizes = [experiment_config['sample_size']]
            else:
                sample_sizes = experiment_config.get('sample_sizes', [1000])
            
            datasets = experiment_config.get('datasets', [])
            models = experiment_config.get('models', [])
            strategies = experiment_config.get('strategies', ['original'])
            
            total_tasks = len(sample_sizes) * len(datasets) * len(models) * len(strategies)
            
            return f"experiment_{int(time.time())}"
    
    def start_session(self, session_info: Dict[str, Any]) -> str:
        """Start a new session for tracking"""
        with self._lock:
            session_id = f"session_{int(time.time())}"
            self.sessions[session_id] = {
                'start_time': datetime.now().isoformat(),
                'session_info': session_info,
                'end_time': None
            }
            return session_id
    
    def end_session(self, session_id: str, session_results: Dict[str, Any] = None):
        """End a session"""
        with self._lock:
            if session_id in self.sessions:
                self.sessions[session_id]['end_time'] = datetime.now().isoformat()
                if session_results:
                    self.sessions[session_id]['results'] = session_results
    
    def create_task(self, task_id: str, task_type: str, dataset_name: str, 
                   model_name: str, strategy: str, sample_size: int) -> TaskProgress:
        """Create a new task for tracking"""
        with self._lock:
            task = TaskProgress(
                task_id=task_id,
                task_type=task_type,
                dataset_name=dataset_name,
                model_name=model_name,
                strategy=strategy,
                sample_size=sample_size,
                total_samples=sample_size
            )
            self.tasks[task_id] = task
            return task
    
    def update_task_progress(self, task_id: str, completed: int, failed: int = 0):
        """Update progress for a specific task"""
        with self._lock:
            if task_id not in self.tasks:
                return
            
            task = self.tasks[task_id]
            task.completed_samples = completed
            task.failed_samples = failed
            task.last_update = time.time()
            
            # Calculate throughput
            if task.start_time and task.last_update:
                elapsed = task.last_update - task.start_time
                if elapsed > 0:
                    task.samples_per_second = completed / elapsed
                    
                    # Estimate completion time
                    remaining_samples = task.total_samples - completed
                    if task.samples_per_second > 0:
                        estimated_seconds = remaining_samples / task.samples_per_second
                        task.estimated_completion = task.last_update + estimated_seconds
            
            # Update global stats
            self.global_stats['last_update'] = time.time()
    
    def complete_task(self, task_id: str, success: bool = True):
        """Mark a task as completed"""
        with self._lock:
            if task_id not in self.tasks:
                return
            
            task = self.tasks[task_id]
            task.end_time = time.time()
            
            if success:
                self.global_stats['completed_experiments'] += 1
                task.completed_samples = task.total_samples
            else:
                self.global_stats['failed_experiments'] += 1
            
            self.global_stats['total_samples_processed'] += task.completed_samples
            
            # Record timing data
            if task.start_time and task.end_time:
                duration = task.end_time - task.start_time
                self.timing_history.append({
                    'task_id': task_id,
                    'dataset': task.dataset_name,
                    'model': task.model_name,
                    'strategy': task.strategy,
                    'sample_size': task.sample_size,
                    'duration': duration,
                    'throughput': task.samples_per_second,
                    'success': success
                })
            
            self.global_stats['last_update'] = time.time()
    
    def get_progress_summary(self) -> Dict[str, Any]:
        """Get comprehensive progress summary"""
        with self._lock:
            current_time = time.time()
            
            # Task summaries
            task_summaries = []
            total_progress = 0
            active_tasks = 0
            
            for task_id, task in self.tasks.items():
                if task.total_samples > 0:
                    progress_percent = (task.completed_samples / task.total_samples) * 100
                    total_progress += progress_percent
                    
                    if task.end_time is None:
                        active_tasks += 1
                    
                    task_summaries.append({
                        'task_id': task_id,
                        'dataset': task.dataset_name,
                        'model': task.model_name,
                        'strategy': task.strategy,
                        'sample_size': task.sample_size,
                        'progress_percent': progress_percent,
                        'completed_samples': task.completed_samples,
                        'failed_samples': task.failed_samples,
                        'throughput': task.samples_per_second,
                        'estimated_completion': task.estimated_completion,
                        'status': 'completed' if task.end_time else 'running'
                    })
            
            # Overall progress
            overall_progress = total_progress / len(self.tasks) if self.tasks else 0
            
            # Global timing
            global_duration = None
            if self.global_stats['start_time']:
                global_duration = current_time - self.global_stats['start_time']
            
            # Throughput statistics
            throughput_stats = {}
            if self.timing_history:
                throughputs = [t['throughput'] for t in self.timing_history if t['throughput'] > 0]
                if throughputs:
                    throughput_stats = {
                        'avg_throughput': statistics.mean(throughputs),
                        'median_throughput': statistics.median(throughputs),
                        'max_throughput': max(throughputs),
                        'min_throughput': min(throughputs)
                    }
            
            return {
                'timestamp': datetime.now().isoformat(),
                'overall_progress': overall_progress,
                'global_stats': self.global_stats.copy(),
                'active_tasks': active_tasks,
                'total_tasks': len(self.tasks),
                'global_duration': global_duration,
                'throughput_stats': throughput_stats,
                'task_summaries': task_summaries
            }
    
    def estimate_completion_time(self) -> Optional[datetime]:
        """Estimate overall completion time"""
        with self._lock:
            if not self.tasks:
                return None
            
            # Get latest estimated completion from all active tasks
            latest_completion = None
            for task in self.tasks.values():
                if task.estimated_completion and task.end_time is None:
                    if latest_completion is None or task.estimated_completion > latest_completion:
                        latest_completion = task.estimated_completion
            
            if latest_completion:
                return datetime.fromtimestamp(latest_completion)
            return None
    
    def get_performance_report(self) -> Dict[str, Any]:
        """Generate comprehensive performance report"""
        with self._lock:
            report = {
                'timestamp': datetime.now().isoformat(),
                'summary': self.get_progress_summary(),
                'timing_analysis': self._analyze_timing(),
                'throughput_analysis': self._analyze_throughput(),
                'efficiency_analysis': self._analyze_efficiency()
            }
            return report
    
    def _analyze_timing(self) -> Dict[str, Any]:
        """Analyze timing patterns"""
        if not self.timing_history:
            return {}
        
        # Group by dataset
        dataset_times = {}
        for record in self.timing_history:
            dataset = record['dataset']
            if dataset not in dataset_times:
                dataset_times[dataset] = []
            dataset_times[dataset].append(record['duration'])
        
        # Calculate statistics per dataset
        dataset_stats = {}
        for dataset, times in dataset_times.items():
            if times:
                dataset_stats[dataset] = {
                    'avg_duration': statistics.mean(times),
                    'median_duration': statistics.median(times),
                    'max_duration': max(times),
                    'min_duration': min(times),
                    'total_experiments': len(times)
                }
        
        return {
            'dataset_timing': dataset_stats,
            'total_experiments': len(self.timing_history),
            'avg_duration_all': statistics.mean([r['duration'] for r in self.timing_history]),
            'total_processing_time': sum([r['duration'] for r in self.timing_history])
        }
    
    def _analyze_throughput(self) -> Dict[str, Any]:
        """Analyze throughput patterns"""
        if not self.timing_history:
            return {}
        
        # Group by model and strategy
        model_throughput = {}
        strategy_throughput = {}
        
        for record in self.timing_history:
            model = record['model']
            strategy = record['strategy']
            throughput = record['throughput']
            
            if throughput > 0:
                if model not in model_throughput:
                    model_throughput[model] = []
                model_throughput[model].append(throughput)
                
                if strategy not in strategy_throughput:
                    strategy_throughput[strategy] = []
                strategy_throughput[strategy].append(throughput)
        
        # Calculate statistics
        model_stats = {}
        for model, throughputs in model_throughput.items():
            if throughputs:
                model_stats[model] = {
                    'avg_throughput': statistics.mean(throughputs),
                    'median_throughput': statistics.median(throughputs),
                    'max_throughput': max(throughputs),
                    'min_throughput': min(throughputs)
                }
        
        strategy_stats = {}
        for strategy, throughputs in strategy_throughput.items():
            if throughputs:
                strategy_stats[strategy] = {
                    'avg_throughput': statistics.mean(throughputs),
                    'median_throughput': statistics.median(throughputs),
                    'max_throughput': max(throughputs),
                    'min_throughput': min(throughputs)
                }
        
        return {
            'model_throughput': model_stats,
            'strategy_throughput': strategy_stats
        }
    
    def _analyze_efficiency(self) -> Dict[str, Any]:
        """Analyze efficiency comparisons between strategies"""
        if not self.timing_history:
            return {}
        
        # Compare original vs optimized strategy
        original_times = [r['duration'] for r in self.timing_history if r['strategy'] == 'original']
        optimized_times = [r['duration'] for r in self.timing_history if r['strategy'] == 'optimized']
        
        efficiency_report = {}
        
        if original_times and optimized_times:
            avg_original = statistics.mean(original_times)
            avg_optimized = statistics.mean(optimized_times)
            
            if avg_original > 0:
                efficiency_gain = ((avg_original - avg_optimized) / avg_original) * 100
                efficiency_report['strategy_comparison'] = {
                    'original_avg_duration': avg_original,
                    'optimized_avg_duration': avg_optimized,
                    'efficiency_gain_percent': efficiency_gain
                }
        
        # Dataset-specific efficiency
        dataset_efficiency = {}
        for dataset in set(r['dataset'] for r in self.timing_history):
            dataset_original = [r['duration'] for r in self.timing_history 
                              if r['dataset'] == dataset and r['strategy'] == 'original']
            dataset_optimized = [r['duration'] for r in self.timing_history 
                               if r['dataset'] == dataset and r['strategy'] == 'optimized']
            
            if dataset_original and dataset_optimized:
                avg_orig = statistics.mean(dataset_original)
                avg_opt = statistics.mean(dataset_optimized)
                
                if avg_orig > 0:
                    gain = ((avg_orig - avg_opt) / avg_orig) * 100
                    dataset_efficiency[dataset] = {
                        'original_avg': avg_orig,
                        'optimized_avg': avg_opt,
                        'efficiency_gain_percent': gain
                    }
        
        efficiency_report['dataset_efficiency'] = dataset_efficiency
        return efficiency_report
    
    def save_progress_report(self, filename: Optional[str] = None) -> Path:
        """Save progress report to file"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"progress_report_{timestamp}.json"
        
        output_path = self.log_dir / filename
        report = self.get_performance_report()
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        return output_path
    
    def clear_history(self):
        """Clear all tracking history"""
        with self._lock:
            self.tasks.clear()
            self.timing_history.clear()
            self.throughput_history.clear()
            self.global_stats = {
                'total_experiments': 0,
                'completed_experiments': 0,
                'failed_experiments': 0,
                'total_samples_processed': 0,
                'start_time': None,
                'last_update': None
            } 