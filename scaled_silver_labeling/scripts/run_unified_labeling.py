#!/usr/bin/env python3
"""
Unified Labeling Script

This script automatically detects whether to use Gemini API or traditional LLM servers
based on the model name provided.

Model Detection:
    - Models starting with "gemini" use Google's Gemini API
    - All other models use traditional LLM servers (require --port-range)

    *Models other than Gemini are in beta so needs more testing.
    
Labeling Strategies:
    - OPTIMIZED: Sequential pipeline approach (NOR -> ONER if needed, no IRCOT)
      * Faster execution, fewer retrieval calls
      * Uses early stopping when NOR/ONER succeeds
      * Optimized for efficiency
    
    - ORIGINAL: Full system approach (always runs NOR, ONER, IRCOT)
      * Follows original Adaptive-RAG paper methodology
      * All three systems are executed for comprehensive comparison
      * More retrieval calls but full coverage

Usage:
    # Gemini models (no port-range needed)
    python run_unified_labeling.py --model gemini-2.5-flash-lite --dataset hotpotqa --strategy optimized
    python run_unified_labeling.py --model gemini-2.5-flash-lite --dataset hotpotqa --strategy original --sample_size 500
    
    # Open Source LLMs (port-range required)
    python run_unified_labeling.py --model "Qwen/Qwen2.5-3B-Instruct" --port-range 8010-8026 --dataset trivia --strategy original (Beta Version, needs more testing)
"""

import argparse
import json
import os
import sys
import tempfile
import time
import logging
import warnings

# Suppress token length warnings immediately at script startup
warnings.filterwarnings("ignore")

# Redirect stderr to suppress low-level tokenizer warnings
class StderrFilter:
    def __init__(self, original_stderr):
        self.original_stderr = original_stderr
        
    def write(self, message):
        # Filter out token length warnings
        if "Token indices sequence length is longer than" not in message:
            self.original_stderr.write(message)
    
    def flush(self):
        self.original_stderr.flush()

# Apply the filter
sys.stderr = StderrFilter(sys.stderr)

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

# Add project root to path
project_root = Path(__file__).parent.parent.parent  # Go up to Adaptive-RAG directory
sys.path.insert(0, str(project_root))

# Import scaled labeling components
from scaled_silver_labeling.servers.gemini_server_adapter import GeminiAPIServerManager
from scaled_silver_labeling.servers.llm_server_manager import LLMServerManager
from scaled_silver_labeling.data.dataset_processor import DatasetProcessor
from scaled_silver_labeling.labeling.original_labeler import OriginalLabeler
from scaled_silver_labeling.labeling.optimized_labeler import OptimizedLabeler
from scaled_silver_labeling.ssl_logging.base_logger import LoggerFactory

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Suppress verbose HTTP request logs
logging.getLogger('httpx').setLevel(logging.WARNING)
logging.getLogger('google_genai').setLevel(logging.WARNING)
logging.getLogger('google.generativeai').setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

def is_gemini_model(model_name: str) -> bool:
    """Check if the model is a Gemini model based on name"""
    return model_name.lower().startswith('gemini')

def get_model_abbreviation(model_name: str) -> str:
    """Get model abbreviation for folder/file naming"""
    model_abbreviations = {
        'gemini-2.5-flash-lite': 'g2.5-lite',
        'gemini-1.5-flash-8b': 'g1.5-8b',
        # Add more model abbreviations as needed
    }
    return model_abbreviations.get(model_name, model_name)

def sanitize_model_name_for_path(model_name: str) -> str:
    """Sanitize model name for use in file paths"""
    return model_name.replace("/", "_").replace("-", "_").replace(".", "_")

def parse_port_range(port_range: str) -> List[int]:
    """
    Parse port range string into list of ports
    
    Args:
        port_range: Port range string like "8010-8039" or "8010,8011,8012"
        
    Returns:
        List of port numbers
    """
    if '-' in port_range:
        start_port, end_port = map(int, port_range.split('-'))
        return list(range(start_port, end_port + 1))
    elif ',' in port_range:
        return [int(port.strip()) for port in port_range.split(',')]
    else:
        return [int(port_range)]

def load_dataset_for_labeling(dataset_name: str, sample_size: Union[int, str], filter_yes_no: bool = True) -> tuple[List[Dict[str, Any]], str]:
    """
    Load and prepare dataset samples for labeling
    
    Args:
        dataset_name: Name of the dataset to load
        sample_size: Number of samples to process ('all' for entire dataset)
        filter_yes_no: Whether to filter out yes/no questions
        
    Returns:
        Tuple of (samples_list, data_source_description)
    """
    logger.info(f"Loading dataset: {dataset_name}")
    
    try:
        processor = DatasetProcessor(base_path="processed_data")
        
        # Convert 'all' to appropriate sample size
        if sample_size == 'all':
            target_sample_size = 100000  # Large number to get all samples
            size_str = "all"
        else:
            target_sample_size = int(sample_size)
            size_str = str(sample_size)
        
        logger.info(f"Requesting sample_size={target_sample_size}, filter_yes_no={filter_yes_no}")
        
        # Load samples using dataset processor (with filtering if requested)
        samples = processor.load_and_filter_samples(
            dataset_name=dataset_name,
            sample_size=target_sample_size,
            filter_yes_no=filter_yes_no
        )
        
        data_source = f"{dataset_name}_{size_str}_samples"
        if filter_yes_no:
            data_source += "_filtered"
        
        logger.info(f"Successfully loaded {len(samples)} samples from {data_source}")
        return samples, data_source
        
    except AttributeError as e:
        logger.error(f"Method not found in DatasetProcessor: {e}")
        logger.error("Available methods: load_dataset_with_subsampling, load_and_filter_samples")
        raise
    except FileNotFoundError as e:
        logger.error(f"Dataset file not found: {e}")
        logger.error(f"Check if dataset '{dataset_name}' exists in processed_data/ directory")
        logger.error("Available datasets: hotpotqa, 2wikimultihopqa, musique, trivia, nq, squad")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading dataset '{dataset_name}': {e}")
        logger.error(f"Args: sample_size={sample_size}, filter_yes_no={filter_yes_no}")
        raise

def determine_workers_from_ports(port_list: List[int]) -> tuple[int, List[int]]:
    """
    Determine number of workers based on available working ports
    
    Args:
        port_list: List of ports to test
        
    Returns:
        Tuple of (number of working ports, list of working ports)
    """
    working_ports = []
    for port in port_list:
        try:
            import requests
            response = requests.get(f"http://localhost:{port}/generate", 
                                  params={"prompt": "hello", "max_length": 5}, 
                                  timeout=15)  # Reasonable timeout for health check
            if response.status_code == 200:
                working_ports.append(port)
                logger.info(f"✅ Server on port {port} is responsive")
            else:
                logger.warning(f"⚠️ Server on port {port} returned status {response.status_code}")
        except Exception as e:
            logger.warning(f"⚠️ Server on port {port} is not responsive: {str(e)}")
    
    if not working_ports:
        logger.error("❌ No responsive servers found!")
        sys.exit(1)
    
    logger.info(f"📡 Found {len(working_ports)} working servers: {working_ports}")
    return len(working_ports), working_ports

def create_gemini_server_manager(model_name: str, parallel_calls: int, rate_limit_delay: float, timeout: int):
    """Create Gemini server manager"""
    return GeminiAPIServerManager(
        model_name=model_name,
        max_parallel_calls=parallel_calls,
        rate_limit_delay=rate_limit_delay,
        timeout=timeout
    )

def create_llm_server_manager(model_name: str, working_ports: List[int]):
    """Create traditional LLM server manager"""
    # Create a temporary server config for the working servers only
    config = {
        "llm_servers": [
            {
                "id": f"server_{port}",
                "model": model_name,
                "host": "localhost", 
                "port": port,
                "gpu_id": 0,  # Default value for temporary config (actual GPU allocation handled by server_startup.py)
                "timeout": 60  # Reasonable timeout for heavy model processing
            } for port in working_ports
        ]
    }
    
    # Write temporary config file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(config, f)
        config_path = f.name
    
    server_manager = LLMServerManager(config_path=config_path)
    
    # Test server connectivity
    available_server = server_manager.get_available_server()
    if not available_server:
        logger.error("❌ No LLM servers available. Please start servers first.")
        logger.error("   Use server_startup.py to start servers on the specified ports.")
        sys.exit(1)
    
    logger.info(f"✅ Found {len(server_manager.servers)} available servers")
    return server_manager

def main():
    """Main function to run unified labeling"""
    parser = argparse.ArgumentParser(
        description="Unified scaled silver labeling using Gemini API or LLM servers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Gemini models (no port-range needed)
  python run_unified_labeling.py --model gemini-2.5-flash-lite --dataset hotpotqa --sample_size 100 --strategy optimized
  python run_unified_labeling.py --model gemini-2.5-flash-lite --dataset hotpotqa --sample_size all --strategy original
  
  # Open Source LLMs (port-range required)
  python run_unified_labeling.py --model "Qwen/Qwen2.5-7B-Instruct" --port-range 8010-8026 --dataset trivia --sample_size all --strategy original
        """
    )
    
    # Core arguments
    parser.add_argument("--model", type=str, required=True,
                        help="Model to use (e.g., 'gemini-2.5-flash-lite', 'flan-t5-xl', 'Qwen/Qwen2.5-7B-Instruct')")
    parser.add_argument("--dataset", type=str, required=True,
                        choices=["hotpotqa", "2wikimultihopqa", "musique", "nq", "trivia", "squad"],
                        help="Dataset to process")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["original", "optimized"],
                        help="Labeling strategy: 'original' runs all systems (NOR/ONER/IRCOT), 'optimized' uses sequential pipeline (NOR->ONER, no IRCOT)")
    parser.add_argument("--sample_size", type=str, default="100",
                        help="Number of samples to process (or 'all' for entire dataset)")
    
    # Server configuration (conditional based on model type)
    parser.add_argument("--port-range", type=str, default=None,
                        help="Port range for LLM servers (e.g., '8010-8039'). Required for non-Gemini models.")
    
    # Performance tuning
    parser.add_argument("--workers", type=int, default=5,
                        help="Number of parallel workers/API calls (for Gemini: parallel API calls, for port-range: auto-set to number of working ports)")
    parser.add_argument("--rate_limit_delay", type=float, default=0.1,
                        help="Delay between API calls (seconds) - applies to Gemini models")
    parser.add_argument("--timeout", type=int, default=60,
                        help="Request timeout (seconds)")
    
    # System configuration  
    parser.add_argument("--oner_max_docs", type=int, default=None,
                        help="Max documents for ONER system")
    parser.add_argument("--ircot_max_docs", type=int, default=None,
                        help="Max documents for IRCOT system")
    
    # Output configuration
    parser.add_argument("--log_dir", type=str, default="scaled_silver_labeling/logs",
                        help="Directory for log files")
    parser.add_argument("--output_dir", type=str, default="scaled_silver_labeling/predictions",
                        help="Directory for output files")
    
    # Filtering options
    parser.add_argument("--no_filter_yes_no", action="store_true",
                        help="Disable filtering of yes/no questions")
    
    # Debug options
    parser.add_argument("--debug", action="store_true",
                        help="Enable debug logging (shows detailed API calls and processing info)")
    
    args = parser.parse_args()
    
    # Configure logging level based on debug flag
    if args.debug:
        # Enable debug logging for detailed output
        logging.getLogger().setLevel(logging.DEBUG)
        logging.getLogger('commaqa.models.gemini_generator').setLevel(logging.DEBUG)
        logger.info("🐛 Debug mode enabled - showing detailed logging")
    else:
        # Create a comprehensive filter to suppress debug-style messages
        class DebugFilter(logging.Filter):
            def filter(self, record):
                message = record.getMessage()
                # Filter out messages that look like debug output with broader patterns
                debug_patterns = [
                    '🎯 DEBUG:', '📝 DEBUG:', '🔍 DEBUG:', '🧠 DEBUG:', '✅ DEBUG:',
                    'worker_', 'processing question', 'Created FRESH', 'using adapter',
                    'instance ID:', 'created state instance', 'starting IRCoT for question'
                ]
                return not any(pattern in message for pattern in debug_patterns)
        
        # Apply the filter to the root logger and all existing loggers
        root_logger = logging.getLogger()
        debug_filter = DebugFilter()
        root_logger.addFilter(debug_filter)
        
        # Also apply to all existing loggers to catch dynamically created ones
        for logger_name in logging.root.manager.loggerDict:
            existing_logger = logging.getLogger(logger_name)
            existing_logger.addFilter(debug_filter)
        
        # Store the filter globally so we can apply it to new loggers
        args._debug_filter = debug_filter
        
        # Suppress verbose API logs
        logging.getLogger('commaqa.models.gemini_generator').setLevel(logging.WARNING)
        logger.info("🔇 Debug mode disabled - suppressing detailed processing logs")
    
    # Detect model type
    is_gemini = is_gemini_model(args.model)
    logger.info(f"🤖 Model: {args.model}")
    logger.info(f"🔍 Detected model type: {'Gemini API' if is_gemini else 'Traditional LLM Server'}")
    
    # Validate arguments based on model type
    if not is_gemini and not args.port_range:
        logger.error("❌ --port-range is required for non-Gemini models")
        logger.error("   Example: --port-range 8010-8039")
        sys.exit(1)
    
    if is_gemini and args.port_range:
        logger.warning("⚠️  --port-range is ignored for Gemini models")
    
    # Process arguments
    filter_yes_no = not args.no_filter_yes_no
    sample_size = args.sample_size
    if sample_size != 'all':
        try:
            sample_size = int(sample_size)
            if sample_size <= 0:
                raise ValueError("Sample size must be positive")
        except ValueError as e:
            logger.error(f"Invalid sample size: {args.sample_size}")
            sys.exit(1)
    
    # Validate number of workers
    if args.workers <= 0:
        logger.error("Number of workers must be positive")
        sys.exit(1)
    
    workers = args.workers
    
    # Determine size string for file naming
    size_str = "all" if sample_size == 'all' else str(sample_size)
    
    start_time = time.time()
    
    try:
        # Initialize components
        logger.info("🔧 Initializing components...")
        
        # Load dataset based on sample size
        if args.no_filter_yes_no:
            logger.info("⚠️  TRIVIAL ANSWER FILTERING DISABLED: All samples including trivial answers will be included")
        else:
            logger.info("🎯 TRIVIAL ANSWER FILTERING ENABLED: Samples with yes/no answers will be excluded")
        
        samples, data_source = load_dataset_for_labeling(args.dataset, sample_size, filter_yes_no)
        
        logger.info(f"✅ Loaded {len(samples)} samples for labeling")
        logger.info(f"   📁 Data source: {data_source}")
        
        # Check if we have any samples to process
        if len(samples) == 0:
            logger.warning("❌ No samples remain after filtering. Try:")
            logger.warning("   1. Use --no_filter_yes_no to disable filtering")
            logger.warning("   2. Increase --sample_size to get more samples")
            logger.warning("   3. Use a different dataset")
            sys.exit(1)
        
        # Get model abbreviation for consistent naming
        model_abbrev = get_model_abbreviation(args.model)
        
        # Initialize server manager based on model type
        if is_gemini:
            # For Gemini: workers = parallel calls (no complex calculation)
            parallel_calls = workers
            logger.info(f"   🔀 Parallel API calls: {parallel_calls}")
            
            server_manager = create_gemini_server_manager(
                args.model, parallel_calls, args.rate_limit_delay, args.timeout
            )
        else:
            # For port range: workers = number of available ports
            port_list = parse_port_range(args.port_range)
            logger.info(f"   📡 Testing ports: {port_list}")
            
            # Determine workers based on available ports
            workers, working_ports = determine_workers_from_ports(port_list)
            parallel_calls = workers  # workers = parallel calls = number of working ports
            logger.info(f"   🔀 Workers set to available ports: {workers}")
            
            server_manager = create_llm_server_manager(args.model, working_ports)
        
        # Create experiment-specific logger with model abbreviation
        experiment_name = f"{args.dataset}_{sanitize_model_name_for_path(model_abbrev)}_{args.strategy}_{size_str}"
        llm_logger = LoggerFactory.create_llm_interaction_logger(
            log_dir=args.log_dir,
            log_level="INFO",
            experiment_name=experiment_name
        )
        
        # Initialize labeler based on strategy
        if args.strategy == 'original':
            labeler = OriginalLabeler(server_manager, llm_logger)
        else:  # optimized
            labeler = OptimizedLabeler(server_manager, llm_logger)
        
        # Set parallelization configuration in labeler
        if hasattr(labeler, 'set_parallel_config'):
            labeler.set_parallel_config(max_workers=parallel_calls)
        
        # Apply debug filter to any new loggers created during initialization
        if not args.debug and hasattr(args, '_debug_filter'):
            for logger_name in logging.root.manager.loggerDict:
                existing_logger = logging.getLogger(logger_name)
                if not any(isinstance(f, type(args._debug_filter)) for f in existing_logger.filters):
                    existing_logger.addFilter(args._debug_filter)
        
        # Run labeling
        logger.info(f"🏷️  Running {args.strategy} labeling with {args.model} ({parallel_calls} parallel calls)...")
        
        # Monitor API call performance
        api_start_time = time.time()
        
        labeling_results = labeler.label_samples(
            dataset_name=args.dataset,
            sample_size=len(samples),  # Use actual sample count
            model_name=args.model,
            strategy=args.strategy,
            oner_max_docs=args.oner_max_docs,
            ircot_max_docs=args.ircot_max_docs,
            samples=samples
        )
        
        api_end_time = time.time()
        api_duration = api_end_time - api_start_time
        
        # Calculate throughput metrics
        total_samples = labeling_results.get('total_samples', 0)
        samples_per_second = total_samples / api_duration if api_duration > 0 else 0
        
        # Final check: apply debug filter to any loggers created during labeling
        if not args.debug and hasattr(args, '_debug_filter'):
            for logger_name in logging.root.manager.loggerDict:
                existing_logger = logging.getLogger(logger_name)
                if not any(isinstance(f, type(args._debug_filter)) for f in existing_logger.filters):
                    existing_logger.addFilter(args._debug_filter)
        
        # Calculate total steps and label distribution
        total_steps = 0
        label_counts = {}
        discarded_samples = 0
        
        if 'individual_results' in labeling_results:
            for sample in labeling_results['individual_results']:
                # Count steps
                total_steps += sample.get('steps', 0)
                
                # Count labels
                label = sample.get('label', 'Unknown')
                label_counts[label] = label_counts.get(label, 0) + 1
                
                # Count discarded samples separately
                if label == 'DISCARDED':
                    discarded_samples += 1
        
        # Add statistics to labeling results
        labeling_results['label_distribution'] = label_counts
        labeling_results['total_steps'] = total_steps
        labeling_results['avg_steps_per_sample'] = total_steps / total_samples if total_samples > 0 else 0
        labeling_results['discarded_samples'] = discarded_samples
        
        logger.info(f"📊 Performance Metrics:")
        logger.info(f"   ⏱️  Total processing time: {api_duration:.2f}s")
        logger.info(f"   🏃 Samples per second: {samples_per_second:.2f}")
        logger.info(f"   🔀 Parallel calls used: {parallel_calls}")
        logger.info(f"   🔍 Total steps: {total_steps}")
        logger.info(f"   📊 Avg steps/sample: {total_steps / total_samples:.2f}" if total_samples > 0 else "   📊 Avg steps/sample: 0.00")
        
        # Save results with model abbreviation
        output_file = f"{args.dataset}_{sanitize_model_name_for_path(model_abbrev)}_{args.strategy}_{size_str}_labeled_data.json"
        output_path = Path(args.output_dir) / f"dev_{size_str}" / f"{args.strategy}_strategy" / f"{model_abbrev}_predictions" / output_file
        
        # Create output directory
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save labeled data
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(labeling_results, f, indent=2, ensure_ascii=False)
        
        end_time = time.time()
        total_duration = end_time - start_time
        
        # Final summary
        logger.info("\n" + "="*80)
        logger.info("🎉 UNIFIED LABELING COMPLETED SUCCESSFULLY!")
        logger.info("="*80)
        logger.info(f"📊 Results Summary:")
        logger.info(f"   🤖 Model: {args.model} ({'Gemini API' if is_gemini else 'LLM Server'})")
        logger.info(f"   📁 Dataset: {args.dataset}")
        logger.info(f"   🎯 Strategy: {args.strategy}")
        logger.info(f"   📝 Samples processed: {total_samples}")
        if discarded_samples > 0:
            logger.info(f"   ⚠️  Samples discarded (resource exhausted): {discarded_samples}")
        logger.info(f"   ⏱️  Total time: {total_duration:.2f}s")
        logger.info(f"   🔍 Total steps: {labeling_results.get('total_steps', 0)}")
        logger.info(f"   📊 Avg steps/sample: {labeling_results.get('avg_steps_per_sample', 0):.2f}")
        logger.info(f"   💾 Output saved: {output_path}")
        
        # Display label distribution
        label_distribution = labeling_results.get('label_distribution', {})
        if label_distribution:
            logger.info(f"   🏷️  Label distribution:")
            for label in sorted(label_distribution.keys()):
                count = label_distribution[label]
                percentage = (count / total_samples) * 100 if total_samples > 0 else 0
                # Special handling for discarded samples
                if label == 'DISCARDED':
                    logger.info(f"      {label}: {count} ({percentage:.1f}%) - ⚠️ Resource exhausted samples")
                else:
                    logger.info(f"      {label}: {count} ({percentage:.1f}%)")
        
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.info("\n❌ Process interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Error during labeling: {e}")
        import traceback
        logger.error(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()