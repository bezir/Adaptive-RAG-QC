#!/usr/bin/env python3
"""
IRCoT Bridge Adapter

This module provides a bridge between the silver labeling system and the benchmark IRCoT
implementation. It ensures that the same IRCoT logic used in
benchmark evaluation is also used during silver label generation.

The adapter handles:
- Dynamic config generation based on model and dataset
- State management for the participant-based IRCoT system  
- Format translation between silver labeling and benchmark systems
- Transparent integration with existing silver labeling workflow
"""

import os
import sys
import json
import time
import tempfile
import logging
import copy  # Add this import for deep copying
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

# Add project root to path for benchmark imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


try:
    from ircot.core.engine import RobustIRCoT
    ROBUST_IRCOT_AVAILABLE = True
except ImportError as e:
    RobustIRCoT = None
    ROBUST_IRCOT_AVAILABLE = False

# Import benchmark IRCoT components (fallback)

from commaqa.inference.constants import MODEL_NAME_CLASS
from commaqa.inference.data_instances import StructuredDataInstance, QuestionAnsweringStep
from commaqa.inference.model_search import ModelController, BestFirstDecomposer
from commaqa.inference.utils import get_environment_variables
from commaqa.inference.dataset_readers import MultiParaRCReader
BENCHMARK_IRCOT_AVAILABLE = True

# Import silver labeling components  
from scaled_silver_labeling.utils.common import get_timestamp

class IRCoTBridgeAdapter:
    """
    Bridge adapter that integrates benchmark IRCoT implementation into silver labeling.
    
    This adapter provides a simple interface that matches the existing silver labeling
    system while using the actual benchmark IRCoT participants internally. It ensures
    consistency between silver label generation and benchmark evaluation.
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None, server_manager=None):
        """
        Initialize the IRCoT bridge adapter.
        
        Args:
            logger: Logger instance for recording operations
            server_manager: LLM server manager for load balancing (optional, for Qwen/local LLMs)
        """
        self.logger = logger or logging.getLogger(__name__)
        self.server_manager = server_manager  # Store server manager for load balancing

        self.retriever_host = os.getenv("RETRIEVER_HOST", "http://localhost")
        self.retriever_port = os.getenv("RETRIEVER_PORT", "8000")
        
        if server_manager:
            self.logger.debug("IRCoT Bridge Adapter initialized with server manager for load balancing")
        else:
            self.logger.debug("IRCoT Bridge Adapter initialized - no server manager (direct LLM connection)")
    
    def _get_corpus_name(self, dataset_name: str) -> str:
        """
        Get the appropriate corpus name for the dataset.
        Maps dataset names to their corresponding Elasticsearch index names.
        """
        # Map dataset names to their corresponding corpus names
        corpus_mapping = {
            'hotpotqa': 'hotpotqa',
            '2wikimultihopqa': '2wikimultihopqa', 
            'musique': 'musique',
            'squad': 'wiki',  # Single-hop datasets use wiki corpus
            'nq': 'wiki',
            'trivia': 'wiki'
        }
        
        # Default to dataset name if not in mapping
        return corpus_mapping.get(dataset_name.lower(), dataset_name.lower())

    def run_ircot_system(self, sample: Dict[str, Any], model_name: str, 
                        max_docs: int = 6, dataset_name: str = "hotpotqa") -> Dict[str, Any]:
        """
        Run the IRCoT system on a sample, preferring the robust implementation.
        
        This method first tries to use our robust IRCoT implementation for improved
        performance and reliability. Falls back to benchmark implementation if needed.
        
        Args:
            sample: Sample data containing question_text and other fields
            model_name: Name of the model (e.g., "flan-t5-xxl", "gemini", "qwen")
            max_docs: Maximum number of documents to retrieve per step
            dataset_name: Name of the dataset for config selection
            
        Returns:
            Dictionary containing:
            - answer: Final answer string
            - correct: Boolean indicating if answer matches ground truth
            - generated_steps: List of reasoning steps
            - retriever_calls: Number of retrieval operations performed
            - latency: Total processing time in seconds
            - steps: Number of reasoning iterations
        """
        start_time = time.time()
        
        # Clear any residual state to ensure query isolation
        import gc
        gc.collect()  # Force garbage collection to clear any lingering state
        
        try:
            # Try robust IRCOT implementation first
            if ROBUST_IRCOT_AVAILABLE:
                return self._run_robust_ircot_system(sample, model_name, max_docs, dataset_name, start_time)
            elif BENCHMARK_IRCOT_AVAILABLE:
                self.logger.warning("Robust IRCoT not available, falling back to benchmark implementation")
                return self._run_benchmark_ircot_system(sample, model_name, max_docs, dataset_name, start_time)
            else:
                raise RuntimeError("Neither robust nor benchmark IRCoT implementations are available")
        except Exception as e:
            self.logger.error(f"IRCoT system failed: {e}")
            # Return error result in expected format
            return {
                'answer': '',
                'correct': False,
                'generated_steps': [],
                'retriever_calls': 0,
                'latency': time.time() - start_time,
                'steps': 0,
                'error': str(e)
            }

    def _run_robust_ircot_system(self, sample: Dict[str, Any], model_name: str, 
                                max_docs: int, dataset_name: str, start_time: float) -> Dict[str, Any]:
        """
        Run our robust IRCoT implementation.
        """
        question_text = sample.get('question_text', sample.get('question', ''))
        ground_truths = self._extract_ground_truth(sample)
        
        self.logger.info(f"🚀 ROBUST IRCoT: Starting analysis")
        self.logger.info(f"📋 Question: {question_text}")
        self.logger.info(f"🎯 Dataset: {dataset_name}, Model: {model_name}")
        
        try:
            # Initialize robust IRCoT with optimized settings
            # Pass server_manager for load balancing (Qwen/local LLMs only, Gemini uses direct connection)
            ircot = RobustIRCoT(
                model=model_name,
                dataset=dataset_name,
                retriever_config={
                    "host": "http://localhost",
                    "port": 8000
                },
                logger=self.logger,
                server_manager=self.server_manager
            )
            
            if self.server_manager:
                self.logger.info(f"✅ IRCoT initialized with server manager for load balancing across multiple ports")
            else:
                self.logger.info(f"✅ IRCoT initialized with direct connection (Gemini)")
            
            # Extract pre-computed contexts if available (for fair comparison with ONER)
            raw_contexts = sample.get('contexts', [])
            initial_contexts = copy.deepcopy(raw_contexts) if raw_contexts else []
            
            # Run with configuration optimized for labeling
            result = ircot.run(
                question=question_text,
                ground_truth=ground_truths,
                initial_contexts=initial_contexts if initial_contexts else None,
                config_overrides={
                    "initial_retrieval_k": max_docs,
                    "iterative_retrieval_k": max(3, max_docs // 2),
                    "max_iterations": 5,
                    "max_total_docs": max_docs * 3,
                    "enable_final_reader": True,
                    "temperature": 0.0
                }
            )
            
            # Convert to bridge adapter format
            answer = result.get('answer', '')
            is_correct = result.get('correct', False)
            
            # Extract reasoning steps from the reasoning chain
            reasoning_chain = result.get('reasoning_chain', '')
            generated_steps = reasoning_chain.split('. ') if reasoning_chain else []
            
            # Count retrieval calls from retrieval history
            retrieval_history = result.get('retrieval_history', [])
            retriever_calls = len(retrieval_history)
            
            self.logger.info(f"✅ ROBUST IRCoT: Completed in {result['latency']:.2f}s")
            self.logger.info(f"📊 Results: {retriever_calls} retrievals, {result['iteration_count']} iterations")
            self.logger.info(f"🎯 Answer: {answer}")
            
            return {
                'answer': answer,
                'correct': is_correct,
                'generated_steps': generated_steps,
                'retriever_calls': retriever_calls,
                'latency': result['latency'],
                'steps': result['iteration_count'],
                'reasoning_chain': reasoning_chain,
                'cumulative_docs': len(result.get('cumulative_docs', [])),
                'robust_ircot': True  # Flag to indicate robust implementation was used
            }
            
        except Exception as e:
            self.logger.error(f"Robust IRCoT failed: {e}, falling back to benchmark implementation")
            if BENCHMARK_IRCOT_AVAILABLE:
                return self._run_benchmark_ircot_system(sample, model_name, max_docs, dataset_name, start_time)
            else:
                raise e

    def _run_benchmark_ircot_system(self, sample: Dict[str, Any], model_name: str, 
                                   max_docs: int, dataset_name: str, start_time: float) -> Dict[str, Any]:
        """
        Run the original benchmark IRCoT implementation (fallback).
        """
        try:
            question_text = sample.get('question_text', sample.get('question', ''))
            # FStore current question for enhanced detection method
            self._current_question = question_text
            self.logger.info(f"🔍 BENCHMARK IRCoT: Starting multi-hop reasoning")
            self.logger.info(f"📋 Question: {question_text}")
            self.logger.info(f"🎯 Dataset: {dataset_name}, Model: {model_name}")
            
            # Detect retriever availability for graceful fallback
            retriever_available = self._is_retriever_available()
            self.logger.info(f"🌐 Retriever available: {retriever_available}")
            if not retriever_available:
                self.logger.warning(
                    f"Retriever server not available at {self.retriever_host}:{self.retriever_port}. "
                    f"Running IRCoT in no-retrieval mode."
                )
            
            # Generate config for this model/dataset combination
            config = self._generate_ircot_config(model_name, dataset_name, max_docs, retriever_available)
            self.logger.info(f"⚙️ IRCoT config: start_state={config['start_state']}, models={list(config['models'].keys())}")
            
            # Check if benchmark components are available
            if not BENCHMARK_IRCOT_AVAILABLE:
                raise RuntimeError("Benchmark IRCoT components not available - using robust implementation")
            
            # Get or create decomposer for this configuration
            self.logger.info(f"🔄 IRCOT: Getting decomposer for config...")
            decomposer = self._get_decomposer(config)
            
            # Convert sample to benchmark format
            structured_instance = self._convert_to_structured_instance(sample)
            
            # Execute IRCoT decomposition
            try:
                self.logger.info("🔧 DEBUG: About to call find_answer_decomp...")
                self.logger.info(f"🔧 DEBUG: Question: {question_text}")
                self.logger.info(f"🔧 DEBUG: Structured instance keys: {list(structured_instance.keys())}")
                
                final_state, other_states = decomposer.find_answer_decomp(structured_instance, debug=True)
                
                self.logger.info("🔧 DEBUG: find_answer_decomp completed successfully")
                
                # COMPREHENSIVE IRCOT CYCLE ANALYSIS
                self._perform_comprehensive_cycle_analysis(final_state)
                
                # Add detailed analysis of each step in the reasoning chain
                if hasattr(final_state, 'data') and final_state.data:
                    reasoning_chain = ""
                    if 'step_by_step_cot_reasoning_sentences' in final_state.data:
                        sentences = final_state.data['step_by_step_cot_reasoning_sentences']
                        reasoning_chain = ' '.join(sentences) if sentences else ""
                        
                        self.logger.info(f"🔧 DEBUG: Final reasoning chain length: {len(sentences)} sentences")
                
            except Exception as e:
                self.logger.error(f"🔧 DEBUG: find_answer_decomp failed with error: {str(e)}")
                import traceback
                self.logger.error(f"🔧 DEBUG: Traceback: {traceback.format_exc()}")
                raise
            
            # Enhanced result extraction
            result = self._extract_result(final_state, sample, start_time)
            
            self.logger.info(f"✅ BENCHMARK IRCoT: Completed in {result.get('latency', 0):.2f}s")
            self.logger.info(f"🎯 Answer: {result.get('answer', '')}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ Benchmark IRCoT failed: {e}")
            import traceback
            self.logger.error(f"📋 Traceback: {traceback.format_exc()}")
            # Return fallback result to maintain compatibility
            return {
                'answer': "",
                'correct': False,
                'generated_steps': [],
                'retriever_calls': 0,
                'latency': time.time() - start_time,
                'steps': 0
            }
    
    def _generate_ircot_config(self, model_name: str, dataset_name: str, max_docs: int, 
                                retriever_available: bool) -> Dict[str, Any]:
        """
        Generate a benchmark-compatible config for the given model and dataset.
        
        This dynamically creates the same config structure used in benchmark evaluation
        but adapted for the silver labeling context.
        """
        # Normalize model name for config generation
        normalized_model = self._normalize_model_name(model_name)
        
        # Determine prompt paths and settings based on model
        # Choose prompt suffix to match the active model family
        if normalized_model == "gemini":
            prompt_suffix = "gemini"
        elif normalized_model == "qwen":
            prompt_suffix = "qwen"
        else:
            prompt_suffix = "flan_t5"
        
        if normalized_model == "gemini":
            llm_model_name = model_name
            question_prefix = ""
            max_length = 800
            model_tokens_limit = 128000
        elif normalized_model == "qwen":
            llm_model_name = self._get_full_model_name(model_name)
            question_prefix = ""
            max_length = 800
            model_tokens_limit = 32000
        else:  # flan_t5 and default
            llm_model_name = self._get_full_model_name(model_name)
            question_prefix = "Answer the following question by reasoning step-by-step.\n"
            max_length = 200
            model_tokens_limit = 6000
        
        # Determine context type - using paper specification
        context_type = "gold_with_1_distractors"
        
        # Build config following benchmark structure
        models_cfg: Dict[str, Any] = {}
        
        # Optional retriever + CoT path when retriever is available
        if retriever_available:
            retrieval_count = 10 # Was 15
            self.logger.info(f"🔍 IRCOT: ENHANCED RETRIEVAL - Setting retrieval_count to {retrieval_count} per iteration")
            self.logger.info(f"🔍 IRCOT: This matches ONER's 15 documents but allows iterative refinement")
            
            models_cfg["step_by_step_bm25_retriever"] = {
                "name": "retrieve_and_reset_paragraphs",
                "next_model": "step_by_step_cot_reasoning_gen",
                "retrieval_type": "bm25",
                "retriever_host": self.retriever_host,
                "retriever_port": int(self.retriever_port),
                "global_max_num_paras": 30, # Was 45
                "query_source": "question_or_last_generated_sentence",  # Key for iterative retrieval
                "source_corpus_name": self._get_corpus_name(dataset_name),
                "document_type": "title_paragraph_text",
                "return_pids": False,
                "cumulate_titles": True,
                "end_state": "[EOQ]",
            }
            start_state = "step_by_step_bm25_retriever"
            add_context_cot = True
        else:
            # No-retriever fallback: start directly with CoT without context
            start_state = "step_by_step_cot_reasoning_gen"
            add_context_cot = False
        
        # Build prompt file path with detailed logging
        # Use enhanced balanced grounding prompt for better multi-hop reasoning
        prompt_file_path = f"prompts/{dataset_name}/balanced_grounding_cot_qa_{prompt_suffix}_enhanced.txt"
        self.logger.info(f"🎯 IRCOT: Using ENHANCED BALANCED GROUNDING prompt: {prompt_file_path}")
        
        # Check if enhanced version exists, fallback to original if not
        import os
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        full_enhanced_path = os.path.join(project_root, prompt_file_path)
        if not os.path.exists(full_enhanced_path):
            self.logger.info(f"Enhanced prompt not found, falling back to original balanced grounding")
            prompt_file_path = f"prompts/{dataset_name}/balanced_grounding_cot_qa_{prompt_suffix}.txt"
        
        # Verify file existence and check prompt content
        import os
        full_prompt_path = os.path.join(project_root, prompt_file_path)
        if os.path.exists(full_prompt_path):
            self.logger.info(f"✅ IRCOT: Prompt file exists at {full_prompt_path}")
            
            # Check prompt content for IRCOT patterns
            try:
                with open(full_prompt_path, 'r') as f:
                    prompt_content = f.read()
                    self.logger.info(f"🔧 DEBUG: Prompt file size: {len(prompt_content)} chars")
                    
                    # Check for key IRCOT patterns
                    has_step_by_step = "I need to" in prompt_content or "first" in prompt_content.lower()
                    has_continue = "continue" in prompt_content.lower() or "now i need" in prompt_content.lower()
                    has_final_answer = "final answer" in prompt_content.lower() or "answer is" in prompt_content.lower()
                    
                    self.logger.info(f"🔧 DEBUG: Prompt analysis:")
                    self.logger.info(f"   - Has step-by-step patterns: {has_step_by_step}")
                    self.logger.info(f"   - Has continuation patterns: {has_continue}")
                    self.logger.info(f"   - Has final answer patterns: {has_final_answer}")
                    
                    # Show first few lines of prompt
                    lines = prompt_content.split('\n')[:5]
                    self.logger.info(f"🔧 DEBUG: First 5 lines of prompt:")
                    for i, line in enumerate(lines):
                        self.logger.info(f"   {i+1}: {line[:100]}{'...' if len(line) > 100 else ''}")
                        
            except Exception as e:
                self.logger.warning(f"⚠️  Could not read prompt file: {e}")
        else:
            self.logger.error(f"❌ IRCOT: Prompt file NOT found at {full_prompt_path}")
            
        answer_regex = r".*(?:So )?the answer is:?\s*(.*)"  # Paper + IRCoT compatible
        max_total_sentences = 20  # INCREASED: Allow enough cycles to complete reasoning (10 retrieve→generate cycles)
        
   
        # DEBUG: Comprehensive model configuration tracing
        self.logger.info("🔧 DEBUG: ================== MODEL CONFIGURATION START ==================")
        
        models_cfg["step_by_step_bm25_retriever"] = {
            "name": "retrieve_and_reset_paragraphs",
            "retrieval_type": "bm25",  # Specify BM25 retrieval 
            "next_model": "step_by_step_cot_reasoning_gen",
            "retrieval_count": 1,  # Use K=1 to force incremental reasoning
            "query_source": "question_or_last_generated_sentence",  # CRITICAL: Use last generated sentence for iterative retrieval
            "retriever_host": self.retriever_host,
            "retriever_port": self.retriever_port,
            "source_corpus_name": self._get_corpus_name(dataset_name),
            "global_max_num_paras": 15,  # Match IRCOT paper's 15 total paragraphs limit
        }
        self.logger.info(f"🔧 DEBUG: Configured step_by_step_bm25_retriever - retrieval_count: 1 (FORCE INCREMENTAL), next_model: step_by_step_cot_reasoning_gen")
        self.logger.info(f"🔧 DEBUG: Retriever query_source: question_or_last_generated_sentence (iterative retrieval based on reasoning)")
        
        models_cfg["step_by_step_cot_reasoning_gen"] = {
            "name": "step_by_step_cot_gen",
            "next_model": "step_by_step_exit_controller",
            "prompt_file": prompt_file_path,
            "question_prefix": question_prefix,
            "prompt_reader_args": self._get_prompt_reader_args(normalized_model),
            "generation_type": "sentences",
            "reset_queries_as_sentences": False,
            "add_context": add_context_cot,
            "shuffle_paras": False,
            "terminal_return_type": "answer",  # Return extracted answer, not titles
            "disable_exit": False,  #  Allow CoT to terminate when answer found
            "end_state": "[EOQ]",
            "gen_model": self._get_gen_model_type(normalized_model),
            "model_name": self._get_full_model_name(model_name),
            # Use newline stop token to force single-sentence generation
            "stop": ["\n"],  # RESTORE newline stop token for sentence-by-sentence generation
            "max_tokens": 150,  # Allow slightly longer reasoning steps
            "temperature": 0,  
            "disable_cache": True,
        }
        self.logger.info(f"🔧 DEBUG: Configured step_by_step_cot_reasoning_gen - prompt_file: {prompt_file_path}, next_model: step_by_step_exit_controller")
        self.logger.info(f"🔧 DEBUG: CoT config - disable_exit: False (ENABLED early termination), add_context: {add_context_cot}, generation_type: sentences")
        
        models_cfg["step_by_step_exit_controller"] = {
            "name": "step_by_step_exit_controller",  # FIXED: Add explicit name field
            "next_model": "step_by_step_bm25_retriever",  # Continue retrieval loop
            "answer_extractor_regex": answer_regex,  # CRITICAL: Use the FIXED regex pattern
            "answer_extractor_remove_last_fullstop": True,
            "terminal_state_next_model": "generate_main_question",  # When terminated, extract final answer
            "terminal_return_type": "answer",  # Return extracted answer, not titles
            "global_max_num_paras": 20,  # Match retriever settings
            "max_num_sentences": max_total_sentences,  # Use paper-specified max steps
        }
        
        self.logger.info(f"🔧 DEBUG: Configured step_by_step_exit_controller - answer_regex: {answer_regex}")
        self.logger.info(f"🔧 DEBUG: Exit controller - max_num_sentences: {max_total_sentences}, next_model: step_by_step_bm25_retriever")
        self.logger.info(f"🔧 DEBUG: Exit controller - terminal_state_next_model: generate_main_question")
        self.logger.info(f"🔧 CRITICAL: Exit controller will now properly detect 'So the answer is: X' AND 'the answer is X'")
        
        models_cfg["generate_main_question"] = {
            "name": "copy_question",
            "next_model": None,
            "eoq_after_n_calls": 0,  # Set to 0 for immediate termination after copying question
            "end_state": "[EOQ]"
        }
        self.logger.info(f"🔧 DEBUG: Configured generate_main_question as copy_question participant with terminal state [EOQ]")
        
        self.logger.info("🔧 DEBUG: ================== MODEL CONFIGURATION END ==================")
        self.logger.info(f"🔧 DEBUG: Expected IRCOT flow:")
        self.logger.info(f"🔧 DEBUG: 1. step_by_step_bm25_retriever (retrieves {min(max_docs, 3)} docs)")
        self.logger.info(f"🔧 DEBUG: 2. step_by_step_cot_reasoning_gen (generates reasoning)")
        self.logger.info(f"🔧 DEBUG: 3. step_by_step_exit_controller (checks termination: regex match OR {max_total_sentences} sentences)")
        self.logger.info(f"🔧 DEBUG: 4. If no termination: back to step_by_step_bm25_retriever")
        self.logger.info(f"🔧 DEBUG: 5. If terminated: generate_main_question (final answer extraction)")
        
        # Debug configuration completeness
        self.logger.info(f"🔧 DEBUG: Total models configured: {len(models_cfg)}")
        self.logger.info(f"🔧 DEBUG: Model names: {list(models_cfg.keys())}")
        
        for model_name, config in models_cfg.items():
            self.logger.info(f"🔧 DEBUG: {model_name} config keys: {list(config.keys())}")
            if "next_model" in config:
                self.logger.info(f"🔧 DEBUG: {model_name} -> {config['next_model']}")
            if "terminal_state_next_model" in config:
                self.logger.info(f"🔧 DEBUG: {model_name} (terminal) -> {config['terminal_state_next_model']}")
        
        config = {
            "start_state": start_state,
            "end_state": "[EOQ]",
            "models": models_cfg,
            "reader": {
                "name": "multi_para_rc",
                "add_paras": False,
                "add_gold_paras": False,
                "add_pinned_paras": False,
            },
            "prediction_type": "answer",
        }
        
        return config
    
    def _normalize_model_name(self, model_name: str) -> str:
        """Normalize model name to standard format."""
        model_lower = model_name.lower()
        if "flan" in model_lower or "t5" in model_lower:
            return "flan_t5"
        elif "gemini" in model_lower:
            return "gemini"
        elif "qwen" in model_lower:
            return "qwen"
        else:
            return "flan_t5"  # Default
    
    def _get_full_model_name(self, model_name: str) -> str:
        """Convert short model names to full names expected by LLM server."""
        model_mapping = {
            "flan-t5-xl": "google/flan-t5-xl",
            "flan-t5-xxl": "google/flan-t5-xxl", 
            "flan-t5-large": "google/flan-t5-large",
            "qwen": "Qwen/Qwen2.5-7B-Instruct",
        }
        return model_mapping.get(model_name, model_name)
    
    def _get_gen_model_type(self, normalized_model: str) -> str:
        """Get the generation model type for the participant."""
        if normalized_model == "gemini":
            return "gemini"
        elif normalized_model == "qwen":
            return "qwen"
        else:
            return "llm_api"  # For FLAN-T5 and others
    
    def _get_prompt_reader_args(self, normalized_model: str) -> Dict[str, Any]:
        """Get prompt reader arguments based on model type."""
        base_args = {
            "estimated_generation_length": 0,
            "shuffle": False,
            "model_length_limit": 1000000,
        }
        
        if normalized_model == "flan_t5":
            base_args["tokenizer_model_name"] = "google/flan-t5-xxl"
        
        return base_args
    
    def _get_decomposer(self, config: Dict[str, Any]):
        """Create a fresh decomposer for each run to avoid state persistence."""
        # Always create a fresh decomposer to avoid state persistence issues
        self.logger.info("🏗️  Creating fresh IRCoT decomposer (no caching to avoid state persistence)")
        
        # Instantiate models from config
        model_map = {}
        config_copy = config.copy()
        
        for key, value in config["models"].items():
            # Create a deep copy of value to avoid modifying the original config
            value = value.copy()
            
            # Handle models that may not have "name" field (like benchmark exit controller)
            if "name" in value:
                class_name = value.pop("name")
            else:
                # For exit controller without name field, infer the class name
                class_name = key  # Use the key as the class name
            
            self.logger.info(f"📦 Creating model '{key}' with class '{class_name}'")
            
            # Log prompt file if this is the CoT generation model
            if key == "step_by_step_cot_reasoning_gen" and "prompt_file" in value:
                self.logger.info(f"🎯 CoT model prompt file: {value['prompt_file']}")
                self.logger.info(f"🎯 CoT model add_context: {value.get('add_context', 'NOT_SET')}")
            
            if class_name not in MODEL_NAME_CLASS:
                raise ValueError(f"Unknown model class: {class_name}")
            
            model = MODEL_NAME_CLASS[class_name](**value)
            config_copy[key] = model.query
            model_map[key] = model
            
            # Ensure complete state isolation for each participant
            if hasattr(model, 'reset_state'):
                model.reset_state()
            
            # Clear any internal caches or state that might persist
            if hasattr(model, '_cache'):
                model._cache = {}
            if hasattr(model, 'num_calls'):
                model.num_calls = 0
            
            self.logger.debug(f"🔄 Reset and isolated state for participant '{key}'")
        
        # Create controller and decomposer
        controller = ModelController(config_copy, data_class=StructuredDataInstance)
        decomposer = BestFirstDecomposer(controller)
        
        self.logger.debug("✅ Fresh IRCoT decomposer created successfully")
        
        return decomposer
    
    def _convert_to_structured_instance(self, sample: Dict[str, Any]):
        """Convert silver labeling sample to benchmark StructuredDataInstance format."""
        # Extract core fields
        question_text = sample.get('question_text', sample.get('question', ''))
        
        # Create structured instance in benchmark format
        instance_data = {
            "question": question_text,
            "question_text": question_text,
            "titles": [],
            "paras": [],
            "metadata": {},
            "inference_seq": []
        }
        
        # Add ground truth if available
        if "answers_objects" in sample:
            instance_data["answers_objects"] = sample["answers_objects"]
        elif "answer" in sample:
            # Convert simple answer to benchmark format
            instance_data["answers_objects"] = [{
                "spans": [sample["answer"]]
            }]
        
        return StructuredDataInstance(instance_data)
    
    def _extract_result(self, final_state, sample: Dict[str, Any], start_time: float) -> Dict[str, Any]:
        """
        Extract final result from IRCoT decomposition with proper step counting.
        """
        latency = time.time() - start_time
        final_answer = ""
        reasoning_steps = 0
        retriever_calls = 0
        generated_steps = []
        reasoning_chain = ""
        
        # DEBUG: Log final_state structure for diagnosis
        self.logger.info("🔧 DEBUG: ================== FINAL STATE ANALYSIS START ==================")
        if final_state is not None:
            self.logger.info(f"🔧 DEBUG: final_state type: {type(final_state)}")
            self.logger.info(f"🔧 DEBUG: final_state attributes: {dir(final_state)}")
            if hasattr(final_state, 'result'):
                self.logger.info(f"🔧 DEBUG: final_state.result: {final_state.result}")
            if hasattr(final_state, '_data'):
                self.logger.info(f"🔧 DEBUG: final_state._data type: {type(final_state._data)}")
        
        if final_state is not None:
            # Check for answers in data instance FIRST - this is where IRCoT stores extracted answers
            if hasattr(final_state, '_data') and final_state._data:
                data_instance = final_state._data
                
                # Method 1: Try get_last_answer FIRST (this is where extracted answers are stored)
                if hasattr(data_instance, 'get_last_answer'):
                    try:
                        last_answer = data_instance.get_last_answer()
                        if last_answer:
                            last_answer_str = str(last_answer).strip().strip('"')
                            # Check if this looks like a clean answer (not raw retrieval data)
                            if len(last_answer_str) < 500 and not last_answer_str.startswith('["') and last_answer_str != "EMPTY":
                                final_answer = last_answer_str
                                self.logger.info(f"🔍 Found clean answer in get_last_answer: '{final_answer}'")
                            else:
                                self.logger.debug(f"🔍 Skipping get_last_answer - appears to be raw retrieval data or empty: '{last_answer_str[:100]}...'")
                    except Exception as e:
                        self.logger.debug(f"get_last_answer failed: {e}")
                
                # Method 1.5: If get_last_answer is EMPTY, check generated_sentences for extracted answers
                if not final_answer and hasattr(data_instance, 'get'):
                    generated_sentences = data_instance.get('generated_sentences', [])
                    if generated_sentences:
                        # Look for the most recent sentence with "So the answer is:" pattern
                        import re
                        answer_regex = re.compile(r'.*(?:So )?the answer is:?\s*(.*)', re.IGNORECASE)
                        
                        for sentence in reversed(generated_sentences):  # Check from most recent
                            if sentence and isinstance(sentence, str):
                                match = answer_regex.match(sentence.strip())
                                if match:
                                    extracted_answer = match.group(1).strip()
                                    if extracted_answer.endswith('.'):
                                        extracted_answer = extracted_answer[:-1]
                                    if extracted_answer and len(extracted_answer) < 200:
                                        final_answer = extracted_answer
                                        self.logger.info(f"🔍 Found extracted answer in generated_sentences: '{final_answer}'")
                                        break
                
                # Method 2: Parse reasoning chain for answers (fallback)
                if hasattr(data_instance, 'get_printable_reasoning_chain'):
                    reasoning_chain = data_instance.get_printable_reasoning_chain()
                    if reasoning_chain:
                        # Extract answer from reasoning chain if not found above
                        if not final_answer:
                            final_answer = self._extract_answer_from_reasoning_chain(reasoning_chain)
                            if final_answer:
                                self.logger.info(f"🔍 Found answer in reasoning chain: '{final_answer}'")
                        
                        # Extract question text from sample
                        question_text = sample.get('question_text', sample.get('question', ''))
                        
                        # Get correct IRCOT cycle count from detailed analysis
                        ircot_cycles, actual_retrievals = self._analyze_reasoning_chain(final_state, question_text)
                        reasoning_steps = ircot_cycles  # Use IRCOT cycles as the correct step count
                        retriever_calls = actual_retrievals  # Use the returned value directly
                        generated_steps = self._extract_reasoning_steps(reasoning_chain)
                        
                        self.logger.info(f"🔍 IRCoT Chain Analysis:")
                        self.logger.info(f"   Reasoning steps (retrieve→generate cycles): {reasoning_steps}")
                        self.logger.info(f"   Generated steps: {len(generated_steps)}")
                        self.logger.info(f"   Final answer from chain: '{final_answer}'")
                        self.logger.info(f"   Total retriever calls: {retriever_calls}")
            
            # Method 3: Extract answer from final state - check multiple sources (last resort)
            if not final_answer and hasattr(final_state, 'result') and final_state.result:
                final_answer = str(final_state.result).strip().strip('"')
                self.logger.info(f"🔍 Found answer in final_state.result: '{final_answer}'")
        
        # Ensure at least 1 step if we have any result
        if final_answer and reasoning_steps == 0:
            reasoning_steps = 1
            self.logger.info(f"   Defaulting to 1 step for non-empty answer")
        
        # Check correctness against ground truth
        correct = False
        ground_truths = self._extract_ground_truth(sample)
        
        if final_answer and ground_truths:
            # Simple correctness check
            final_answer_clean = final_answer.lower().strip()
            for truth in ground_truths:
                if truth.lower().strip() in final_answer_clean or final_answer_clean in truth.lower().strip():
                    correct = True
                    break
        
        result = {
            'final_answer': final_answer,  # Match comprehensive debugging script expectation
            'reasoning_chain': reasoning_chain,  # Add reasoning chain for analysis
            'reasoning_steps': reasoning_steps,  # Use consistent naming
            'retriever_calls': retriever_calls,
            'answer': final_answer,  # Keep for backward compatibility
            'correct': correct,
            'generated_steps': generated_steps,
            'latency': latency,
            'steps': reasoning_steps  # Now properly counted as retrieve→generate cycles
        }
        
        self.logger.info(f"📊 IRCoT Result Summary:")
        self.logger.info(f"   Final answer: '{final_answer}'")
        self.logger.info(f"   Reasoning steps: {reasoning_steps} (retrieve→generate cycles)")
        self.logger.info(f"   Retriever calls: {retriever_calls}")
        self.logger.info(f"   Correct: {correct}")
        self.logger.info(f"   Latency: {latency:.2f}s")
        
        return result
    
    def _count_reasoning_steps_from_chain(self, reasoning_chain: str) -> int:
        """
        Count actual reasoning steps from the reasoning chain.
        Each retrieve→generate cycle counts as 1 step.
        """
        if not reasoning_chain:
            return 0
        
        # Split into individual steps/actions
        steps = reasoning_chain.split('\n')
        
        # Look for patterns that indicate reasoning steps
        reasoning_step_count = 0
        retrieve_count = 0
        generate_count = 0
        
        for step in steps:
            step_lower = step.lower().strip()
            
            # Count retrieval operations
            if any(keyword in step_lower for keyword in ['retrieved', 'retrieval', 'retrieve']):
                retrieve_count += 1
            
            # Count generation operations (reasoning/thinking)
            elif any(keyword in step_lower for keyword in ['generated', 'reasoning', 'thinking', 'step']):
                generate_count += 1
            
            # Look for explicit step markers
            elif step_lower.startswith('step ') or 'reasoning step' in step_lower:
                reasoning_step_count += 1
        
        # IRCoT step calculation: each retrieve→generate cycle = 1 step
        # Method 1: If we have explicit step markers, use those
        if reasoning_step_count > 0:
            return reasoning_step_count
        
        # Method 2: Count retrieve→generate pairs
        # Each retrieval should be followed by generation/reasoning
        if retrieve_count > 0 and generate_count > 0:
            # Conservative estimate: min of retrievals and generations
            return min(retrieve_count, generate_count)
        
        # Method 3: Fallback - count major reasoning activities
        total_activities = retrieve_count + generate_count
        if total_activities > 0:
            # Assume half are retrievals, half are generations
            return max(1, total_activities // 2)
        
        # Ultimate fallback
        return 1 if reasoning_chain.strip() else 0
    
    def _extract_reasoning_steps(self, reasoning_chain: str) -> list:
        """
        Extract individual reasoning steps from the reasoning chain.
        """
        if not reasoning_chain:
            return []
        
        steps = reasoning_chain.split('\n')
        reasoning_steps = []
        
        for step in steps:
            step = step.strip()
            if step and len(step) > 10:  # Filter out very short lines
                # Clean up step text
                if step.startswith('Step '):
                    reasoning_steps.append(step)
                elif any(keyword in step.lower() for keyword in ['reasoning', 'thinking', 'generated', 'retrieved']):
                    reasoning_steps.append(step)
        
        return reasoning_steps
    
    def _detect_sufficient_information(self, reasoning_chain: str, question: str) -> tuple[bool, str]:
        """
        Paper-compliant detection of sufficient information based on IRCoT termination criteria.
        
        The IRCoT paper specifies termination occurs when:
        1. "answer is:" pattern is found in the generated text
        2. Maximum number of steps is reached
        
        This method checks for answer patterns in line with the paper's approach,
        avoiding hardcoded regex patterns that could leak test cases.
        """
        if not reasoning_chain:
            return False, ""
        
        import re
        
        # Primary check: Look for the prompt-consistent answer pattern
        answer_pattern = r'(?i)so\s+the\s+answer\s+is:?\s*(.+?)(?:\.|$)'
        match = re.search(answer_pattern, reasoning_chain)
        if match:
            answer = match.group(1).strip()
            if answer and len(answer) < 100:  # Reasonable answer length
                self.logger.info(f"🔍 SUFFICIENT INFO DETECTED: '{answer}' via prompt answer pattern")
                return True, answer
        
        # Secondary check: Look for conclusion-like statements in the final parts of reasoning
        # Only detect ACTUAL conclusions, not intermediate reasoning steps
        lines = reasoning_chain.strip().split('\n')
        if len(lines) > 2:
            # Check the last few lines for answer-like content
            for line in reversed(lines[-3:]):
                line = line.strip()
                if line and not line.startswith('A: ['):
                    # Remove common prefixes
                    clean_line = re.sub(r'^A:\s*', '', line)
                    
                    # Only consider statements that are ACTUAL CONCLUSIONS
                    # Exclude intermediate reasoning steps that indicate more work is needed
                    if any(phrase in clean_line.lower() for phrase in [
                        "i need to", "i should", "i must", "i have to", "let me", "first", "next", 
                        "then", "now i", "i will", "i'll", "i want to", "i'm going to", "step"
                    ]):
                        continue  # Skip intermediate reasoning steps
                    
                    # Only consider short, definitive statements as potential answers
                    if (clean_line and len(clean_line.split()) <= 6 and 
                        len(clean_line) < 30 and clean_line != question and
                        not clean_line.endswith('?')):  # Must not be a question
                        self.logger.info(f"🔍 SUFFICIENT INFO DETECTED: '{clean_line}' via reasoning conclusion")
                        return True, clean_line
        
        return False, ""

    def _extract_answer_from_reasoning_chain(self, reasoning_chain: str) -> str:
        """
        Answer extraction from IRCOT reasoning chain.
        
        Focus on the patterns that IRCOT already correctly identifies instead of 
        complex fallback mechanisms that override correct answers.
        """
        if not reasoning_chain:
            return ""
        
        import re
        
        self.logger.info("🔧 DEBUG: ================== ANSWER EXTRACTION START ==================")
        self.logger.info(f"🔧 DEBUG: Reasoning chain length: {len(reasoning_chain)} characters")
        
        # PRIORITY 1: Look for explicit "So the answer is: X" or "the answer is: X" patterns
        # These are what IRCOT correctly identifies, so use them directly
        answer_patterns = [
            r'(?i)so\s+the\s+answer\s+is:?\s*(.+?)\.?\s*$',      # "So the answer is: X"
            r'(?i)the\s+answer\s+is:?\s*(.+?)\.?\s*$',           # "the answer is: X"  
            r'(?i)therefore,?\s+the\s+answer\s+is:?\s*(.+?)\.?\s*$',  # "Therefore, the answer is: X"
            r'(?i)thus,?\s+the\s+answer\s+is:?\s*(.+?)\.?\s*$',       # "Thus, the answer is: X"
        ]
        
        # Search through each line for explicit answer patterns
        lines = reasoning_chain.split('\n')
        for line_idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Try each explicit answer pattern
            for pattern_idx, pattern in enumerate(answer_patterns):
                match = re.search(pattern, line)
                if match:
                    answer = match.group(1).strip()
                    # Clean up the answer - remove quotes, brackets, trailing periods
                    answer = re.sub(r'^["\'\[\]]+|["\'\[\]\.]+$', '', answer).strip()
                    
                    if answer and len(answer) < 200 and answer.lower() not in ['[eoq]', 'q:', 'q: [eoq]']:
                        self.logger.info(f"🔧 DEBUG: ✅ EXPLICIT ANSWER FOUND: '{answer}' (pattern {pattern_idx+1})")
                        self.logger.info("🔧 DEBUG: ================== ANSWER EXTRACTION END ==================")
                        return answer
        
        # PRIORITY 2: Look for generated sentences that contain factual information
        # Only if no explicit answer pattern found
        self.logger.info("🔧 DEBUG: No explicit answer patterns found, checking generated sentences...")
        
        generation_lines = []
        for line in lines:
            if line.startswith('A: ') and not line.startswith('A: ['):
                content = line[3:].strip()  # Remove 'A: ' prefix
                if len(content) > 10 and not content.startswith('['):  # Substantive content, not retrieval data
                    generation_lines.append(content)
        
        if generation_lines:
            # Take the last meaningful generated sentence as the answer
            last_generation = generation_lines[-1]
            self.logger.info(f"🔧 DEBUG: Using last generation as answer: '{last_generation}'")
            return last_generation
        
        self.logger.info("🔧 DEBUG: No answer patterns found in reasoning chain")
        self.logger.info("🔧 DEBUG: ================== ANSWER EXTRACTION END ==================")
        return ""
    
    def _extract_simple_answer_from_reasoning(self, reasoning_chain: str, question: str) -> str:
        """
        General answer extraction from reasoning chain - no hardcoded patterns.
        
        IRCoT paper approach: Extract the substantive factual information discovered
        through retrieve→generate cycles that directly addresses the question.
        
        This method avoids hardcoding and pattern matching - instead it looks for
        the most relevant factual content in the reasoning chain.
        """
        if not reasoning_chain or not question:
            return ""
        
        self.logger.info("🔍 GENERAL ANSWER EXTRACTION: Analyzing reasoning chain for substantive facts")
        
        # Parse reasoning chain into steps
        lines = [line.strip() for line in reasoning_chain.split('\n') if line.strip()]
        
        # Find generation steps (not retrieval steps)
        generation_steps = []
        for line in lines:
            if line.startswith('A: ') and not line.startswith('A: ['):
                # This is a generation step with reasoning content
                content = line[3:].strip()  # Remove 'A: ' prefix
                if len(content) > 15:  # Substantive content
                    generation_steps.append(content)
        
        if not generation_steps:
            self.logger.info("🔍 No generation steps found in reasoning chain")
            return ""
        
        self.logger.info(f"🔍 Found {len(generation_steps)} generation steps to analyze")
        
        # Look for the most informative step that contains factual content
        # Start from the end (latest reasoning) and work backwards
        for i, step in enumerate(reversed(generation_steps)):
            step_num = len(generation_steps) - i
            self.logger.debug(f"🔍 Analyzing step {step_num}: {step[:100]}...")
            
            # Check if this step contains substantive factual information
            factual_score = self._assess_factual_content(step, question)
            
            if factual_score > 0.5:  # Threshold for factual relevance
                # Extract the key factual element from this step
                extracted_fact = self._extract_key_fact_from_step(step)
                if extracted_fact:
                    self.logger.info(f"🔍 EXTRACTED FACT from step {step_num}: {extracted_fact}")
                    return extracted_fact
        
        # Fallback: if no high-scoring step found, take the last substantive step
        if generation_steps:
            last_step = generation_steps[-1]
            fallback_fact = self._extract_key_fact_from_step(last_step)
            if fallback_fact:
                self.logger.info(f"🔍 FALLBACK EXTRACTION: {fallback_fact}")
                return fallback_fact
        
        self.logger.info("🔍 No extractable facts found in reasoning chain")
        return ""
    
    def _assess_factual_content(self, step: str, question: str) -> float:
        """
        # TODO: This function is not used anywhere.
        Assess how much factual content a reasoning step contains.
        Returns a score between 0 and 1 based on factual indicators.
        
        This is general - no hardcoded patterns for specific question types.
        """
        step_lower = step.lower()
        question_lower = question.lower()
        
        factual_score = 0.0
        
        # General factual indicators (not domain-specific)
        factual_indicators = [
            'is', 'was', 'are', 'were',  # State of being
            'has', 'have', 'had',        # Possession/relationship
            'born', 'died', 'lived',     # Biographical
            'located', 'based', 'from',  # Location
            'during', 'in', 'at',        # Temporal/spatial
            'directed', 'written', 'created',  # Authorship
            'played', 'starred', 'performed'   # Performance
        ]
        
        # Score based on presence of factual indicators
        indicator_count = sum(1 for indicator in factual_indicators if indicator in step_lower)
        factual_score += min(indicator_count * 0.1, 0.5)  # Max 0.5 from indicators
        
        # Bonus for proper nouns (likely to be names, places, etc.)
        import re
        proper_nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', step)
        if proper_nouns:
            factual_score += min(len(proper_nouns) * 0.1, 0.3)  # Max 0.3 from proper nouns
        
        # Bonus for containing question keywords (relevance)
        question_words = [word for word in question_lower.split() if len(word) > 3]
        relevance = sum(1 for word in question_words if word in step_lower)
        if relevance > 0:
            factual_score += min(relevance * 0.1, 0.2)  # Max 0.2 from relevance
        
        return min(factual_score, 1.0)
    
    def _extract_key_fact_from_step(self, step: str) -> str:
        """
        Extract the key factual element from a reasoning step.
        
        Uses general linguistic patterns, not domain-specific hardcoding.
        """
        # Remove common prefixes and clean up
        cleaned = step.strip()
        
        # Look for the main clause containing factual information
        # Split on common sentence boundaries
        import re
        
        # Split into clauses
        clauses = re.split(r'[.!?;]', cleaned)
        
        # Find the most substantive clause
        best_clause = ""
        best_score = 0
        
        for clause in clauses:
            clause = clause.strip()
            if len(clause) < 10:  # Too short
                continue
            
            # Score based on information density
            score = 0
            
            # Prefer clauses with proper nouns
            proper_nouns = re.findall(r'\b[A-Z][a-z]+\b', clause)
            score += len(proper_nouns) * 2
            
            # Prefer clauses with factual verbs
            factual_verbs = ['is', 'was', 'are', 'were', 'has', 'have', 'had']
            for verb in factual_verbs:
                if f' {verb} ' in f' {clause.lower()} ':
                    score += 3
            
            # Prefer moderate length (not too short, not too long)
            if 20 <= len(clause) <= 100:
                score += 2
            
            if score > best_score:
                best_score = score
                best_clause = clause
        
        # Extract the most informative part of the best clause
        if best_clause:
            # Look for patterns like "X is Y" or "X has Y" etc.
            # But extract the Y part (the answer)
            for pattern in [r'.* (?:is|was|are|were) (.+)', r'.* (?:has|have|had) (.+)']:
                match = re.search(pattern, best_clause, re.IGNORECASE)
                if match:
                    extracted = match.group(1).strip()
                    # Clean up common endings
                    extracted = re.sub(r'[.,;!?]+$', '', extracted)
                    if len(extracted) > 3:
                        return extracted
            
            # Fallback: return the whole clause if no pattern matches
            return best_clause.strip()
        
        return ""
    
    def _extract_ground_truth(self, sample: Dict[str, Any]) -> List[str]:
        """Extract ground truth answers from sample."""
        ground_truths = []
        
        # Try different formats
        if "answers_objects" in sample:
            for ans_obj in sample["answers_objects"]:
                if "spans" in ans_obj:
                    ground_truths.extend(ans_obj["spans"])
        elif "answer" in sample:
            if isinstance(sample["answer"], list):
                ground_truths.extend(sample["answer"])
            else:
                ground_truths.append(sample["answer"])
        elif "answers" in sample:
            if isinstance(sample["answers"], list):
                ground_truths.extend(sample["answers"])
            else:
                ground_truths.append(sample["answers"])
        
        return [str(gt).strip() for gt in ground_truths if str(gt).strip()]
    
    def _perform_comprehensive_cycle_analysis(self, final_state):
        """
        Perform comprehensive analysis of IRCoT cycles to ensure proper paper compliance.
        
        This method analyzes the reasoning chain to verify:
        1. Proper retrieve → generate → retrieve → generate pattern
        2. Correct termination logic
        3. Answer extraction pipeline
        4. Multi-hop reasoning progression
        """
        if not hasattr(final_state, 'data') or not final_state.data:
            self.logger.warning("🔧 DEBUG: No final state data available for cycle analysis")
            return
        
        # Extract reasoning chain for analysis
        reasoning_chain = ""
        sentences = []
        if 'step_by_step_cot_reasoning_sentences' in final_state.data:
            sentences = final_state.data['step_by_step_cot_reasoning_sentences']
            reasoning_chain = '\n'.join(sentences) if sentences else ""
        
        if not reasoning_chain:
            self.logger.warning("🔧 DEBUG: No reasoning chain found for analysis")
            return
        
        self.logger.info("📚 IRCoT Reasoning Chain Analysis:")
        self.logger.info("=" * 60)
        
        # Analyze each step and identify retrieve/generate patterns
        retrieval_steps = []
        generation_steps = []
        cycles = []
        
        current_cycle = {"retrieve": None, "generate": None}
        
        for i, sentence in enumerate(sentences, 1):
            sentence = sentence.strip()
            step_info = f"Step {i}: {sentence[:100]}{'...' if len(sentence) > 100 else ''}"
            
            # Detect retrieval steps (typically start with A: ["...)
            if sentence.startswith('A: ['):
                retrieval_steps.append(i)
                current_cycle["retrieve"] = i
                self.logger.info(f"   🔍 RETRIEVAL #{len(retrieval_steps)} - Retrieved documents/snippets")
                self.logger.info(f"       Data: {sentence[:80]}...")
                
            # Detect generation steps (typically start with A: but not A: [)
            elif sentence.startswith('A: ') and not sentence.startswith('A: ['):
                generation_steps.append(i)
                current_cycle["generate"] = i
                self.logger.info(f"   🧠 GENERATION #{len(generation_steps)} - Reasoning sentence")
                
                # Complete cycle if we have both retrieve and generate
                if current_cycle["retrieve"] is not None:
                    cycles.append(current_cycle.copy())
                    self.logger.info(f"   ✅ COMPLETED IRCOT CYCLE #{len(cycles)} (retrieve→generate)")
                    
                    # Check if this looks like intermediate reasoning (should continue)
                    if "answer is" not in sentence.lower():
                        self.logger.info(f"   🔄 INTERMEDIATE reasoning - should continue iterating")
                    else:
                        self.logger.info(f"   🎯 FINAL answer detected - should terminate")
                    
                    # Reset for next cycle
                    current_cycle = {"retrieve": None, "generate": None}
                    
            self.logger.info(f"   ----------------------------------------")
        
        # Print comprehensive cycle summary
        self.logger.info("🔍 DETAILED IRCOT CYCLE ANALYSIS:")
        for i, cycle in enumerate(cycles, 1):
            retrieve_step = cycle["retrieve"]
            generate_step = cycle["generate"]
            self.logger.info(f"   CYCLE #{i}:")
            self.logger.info(f"     🔍 Retrieve: {sentences[retrieve_step-1][:80]}...")
            self.logger.info(f"     🧠 Generate: {sentences[generate_step-1][:80]}...")
        
        self.logger.info("=" * 60)
        self.logger.info("🔄 IRCOT ITERATIVE PATTERN SUMMARY:")
        self.logger.info(f"   - Total Retrievals: {len(retrieval_steps)}")
        self.logger.info(f"   - Total Generations: {len(generation_steps)}")
        self.logger.info(f"   - IRCOT Cycles (retrieve→generate): {len(cycles)}")
        self.logger.info(f"   - Total Steps: {len(sentences)}")
        self.logger.info(f"   - Expected Pattern: Retrieve→Generate→Retrieve→Generate...")
        
        # Validate IRCOT pattern compliance
        if len(cycles) > 0:
            gen_ret_ratio = len(generation_steps) / len(retrieval_steps) if len(retrieval_steps) > 0 else 0
            self.logger.info(f"✅ IRCOT SUCCESS: {len(cycles)} cycles detected (proper iterative reasoning)")
            self.logger.info(f"   - Gen/Ret Ratio: {gen_ret_ratio:.2f} (should be close to 1.0 for proper IRCOT)")
        else:
            self.logger.warning("❌ IRCOT PATTERN NOT DETECTED: No complete cycles found")
        
        self.logger.info("=" * 60)

    def _is_retriever_available(self) -> bool:
        """Quick health check for retriever server."""
        try:
            import requests
            url = f"{self.retriever_host}:{self.retriever_port}/"
            resp = requests.get(url, timeout=0.5)
            return resp.status_code == 200
        except Exception:
            return False

    def _analyze_reasoning_chain(self, final_state, question_text):
        """
        Analyzes the reasoning chain from the final state to provide detailed insights.
        This method is called when IRCoT completes successfully.
        """
        if final_state is None:
            self.logger.warning("IRCoT final state is None, cannot analyze reasoning chain.")
            return

        self.logger.info("📚 IRCoT Reasoning Chain Analysis:")
        self.logger.info("=" * 60)
        
        # Track IRCOT cycles (retrieve->generate pairs) in detail
        ircot_cycles = []
        current_cycle = {"retrieval": None, "generation": None, "cycle_num": 0}
        retrieval_count = 0
        generation_count = 0
        reasoning_steps = []
        
        # Extract the full reasoning chain from the final state
        if hasattr(final_state, '_data') and final_state._data:
            data_instance = final_state._data
            
            # Try to get the reasoning chain
            reasoning_chain = None
            if hasattr(data_instance, 'get_printable_reasoning_chain'):
                reasoning_chain = data_instance.get_printable_reasoning_chain()
            
            if reasoning_chain:
                self.logger.info("📖 Full reasoning chain found:")
                steps = reasoning_chain.split('\n')
                for i, step in enumerate(steps):
                    if step.strip():  # Skip empty lines
                        self.logger.info(f"Step {i+1}: {step}")
                        
                        # Track IRCOT retrieve->generate cycles based on actual output format
                        if step.startswith("A: [") and step.endswith("]"):
                            # This is a retrieval step (returns document titles/snippets in list format)
                            # Remove comma requirement - single documents like ["Hector Janse van Rensburg"] have no comma
                            retrieval_count += 1
                            if current_cycle["retrieval"] is None:
                                current_cycle["retrieval"] = step
                            self.logger.info(f"   🔍 RETRIEVAL #{retrieval_count} - Retrieved documents/snippets")
                            self.logger.info(f"       Data: {step[:100]}...")
                            
                        elif step.startswith("A: ") and not step.startswith("A: ["):
                            # This is a reasoning generation step
                            generation_count += 1
                            current_cycle["generation"] = step
                            self.logger.info(f"   🧠 GENERATION #{generation_count} - Reasoning sentence")
                            
                            # Complete a full IRCOT cycle if we have both retrieve and generate
                            if current_cycle["retrieval"] is not None:
                                cycle_num = len(ircot_cycles) + 1
                                current_cycle["cycle_num"] = cycle_num
                                ircot_cycles.append(current_cycle.copy())
                                self.logger.info(f"   ✅ COMPLETED IRCOT CYCLE #{cycle_num} (retrieve→generate)")
                                current_cycle = {"retrieval": None, "generation": None, "cycle_num": 0}
                            
                        elif step.startswith("Q: "):
                            # Question restatement
                            self.logger.info(f"   ❓ QUESTION: {step}")
                            
                        # Check for termination indicators
                        if "[EOQ]" in step:
                            self.logger.info(f"   🛑 TERMINATION: [EOQ] found - reasoning chain ended")
                        
                        # Check for final answer patterns
                        if any(phrase in step.lower() for phrase in ["the final answer is", "therefore, the answer is", "so the answer is"]):
                            self.logger.info(f"   ✅ FINAL ANSWER detected in step")
                        
                        # Check for intermediate reasoning patterns that should continue
                        if any(phrase in step.lower() for phrase in ["i need to find", "i need to identify", "i need to determine"]): # Removed "first", "next", "continue"
                            self.logger.info(f"   🔄 INTERMEDIATE reasoning - should continue iterating")
                            
                        reasoning_steps.append(step)
                        self.logger.info("-" * 40)
                
                # DETAILED CYCLE ANALYSIS
                self.logger.info("🔍 DETAILED IRCOT CYCLE ANALYSIS:")
                if ircot_cycles:
                    for cycle in ircot_cycles:
                        self.logger.info(f"   CYCLE #{cycle['cycle_num']}:")
                        self.logger.info(f"     🔍 Retrieve: {cycle['retrieval'][:80]}..." if cycle['retrieval'] else "     🔍 Retrieve: None")
                        self.logger.info(f"     🧠 Generate: {cycle['generation'][:80]}..." if cycle['generation'] else "     🧠 Generate: None")
                else:
                    self.logger.error("❌ CRITICAL: NO IRCOT CYCLES DETECTED!")
                    self.logger.error("   This indicates IRCOT is NOT performing retrieve→generate iterations!")
                    self.logger.error("   Expected: At least 2 cycles for multi-hop questions")
            
            # Try alternative methods to get reasoning steps
            else:
                self.logger.warning("No printable reasoning chain found, trying alternative methods...")
                if hasattr(data_instance, 'get_last_n_steps'):
                    try:
                        last_steps = data_instance.get_last_n_steps(20)  # Get last 20 steps
                        if last_steps:
                            self.logger.info(f"📝 Found {len(last_steps)} reasoning steps via get_last_n_steps:")
                            for i, step in enumerate(last_steps):
                                self.logger.info(f"Alt Step {i+1}: {step}")
                                reasoning_steps.append(str(step))
                    except Exception as e:
                        self.logger.debug(f"get_last_n_steps failed: {e}")
        
        # Summary of iterative pattern with IRCOT cycle analysis
        self.logger.info("=" * 60)
        self.logger.info(f"🔄 IRCOT ITERATIVE PATTERN SUMMARY:")
        self.logger.info(f"   - Total Retrievals: {retrieval_count}")
        self.logger.info(f"   - Total Generations: {generation_count}")
        self.logger.info(f"   - IRCOT Cycles (retrieve→generate): {len(ircot_cycles) if 'ircot_cycles' in locals() else 0}")
        self.logger.info(f"   - Total Steps: {len(reasoning_steps)}")
        self.logger.info(f"   - Expected Pattern: Retrieve→Generate→Retrieve→Generate...")
        
        # Diagnose IRCOT performance
        num_cycles = len(ircot_cycles) if 'ircot_cycles' in locals() else 0
        if num_cycles == 0:
            self.logger.error("❌ IRCOT FAILURE: No retrieve→generate cycles detected!")
            self.logger.error("   System is not performing iterative retrieval and reasoning")
        elif num_cycles == 1:
            self.logger.warning("⚠️  IRCOT INCOMPLETE: Only 1 cycle detected")
            self.logger.warning("   Multi-hop questions typically need 2+ cycles")
        else:
            self.logger.info(f"✅ IRCOT SUCCESS: {num_cycles} cycles detected (proper iterative reasoning)")
        
        if retrieval_count > 0 and generation_count > 0:
            ratio = generation_count / retrieval_count
            self.logger.info(f"   - Gen/Ret Ratio: {ratio:.2f} (should be close to 1.0 for proper IRCOT)")
        
        self.logger.info("=" * 60)
        
        # Return the cycle count and retrieval count for use in final reporting
        return num_cycles, retrieval_count
