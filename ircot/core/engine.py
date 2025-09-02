#!/usr/bin/env python3
"""
IRCOT Engine - Main implementation of the Interleaved Retrieval Chain-of-Thought system.

This module implements the core IRCOT algorithm following the paper's methodology while
incorporating improvements from both the bridge adapter and custom implementations.
"""

import time
import logging
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import sys

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from robust_ircot.core.state import IRCoTState
from robust_ircot.prompting.builder import PromptBuilder
from robust_ircot.retrieval.elasticsearch import ElasticsearchRetriever
from robust_ircot.reasoning.generator import ReasoningGenerator
from robust_ircot.reasoning.termination import TerminationChecker
from robust_ircot.reasoning.parser import AnswerParser


class RobustIRCoT:
    """
    Main IRCOT engine that implements the retrieve-reason-retrieve cycle.
    
    This class orchestrates the entire IRCOT process, managing:
    - Initial retrieval based on the question
    - Iterative reasoning and retrieval cycles
    - Termination detection
    - Final answer extraction
    """
    
    def __init__(self, 
                 model: str,
                 dataset: str,
                 retriever_config: Optional[Dict[str, Any]] = None,
                 prompt_config: Optional[Dict[str, Any]] = None,
                 logger: Optional[logging.Logger] = None,
                 server_manager=None,
                 assigned_port: Optional[int] = None):
        """
        Initialize the Robust IRCOT engine.
        
        Args:
            model: Model name (e.g., "gemini", "qwen", "flan-t5-xxl")
            dataset: Dataset name (e.g., "hotpotqa", "2wikimultihopqa", "musique")
            retriever_config: Configuration for the retriever
            prompt_config: Configuration for prompt building
            logger: Logger instance
            server_manager: LLM server manager for load balancing (optional, for Qwen/local LLMs)
            assigned_port: Pre-assigned port for parallel processing (overrides server_manager)
        """
        self.model = model
        self.dataset = dataset
        self.logger = logger or logging.getLogger(__name__)
        self.server_manager = server_manager  # Store server manager for load balancing
        self.assigned_port = assigned_port  # Store assigned port for parallel processing
        
        # Initialize components
        self.retriever = ElasticsearchRetriever(
            **(retriever_config or {}),
            logger=self.logger
        )
        
        self.prompt_builder = PromptBuilder(
            model=model,
            dataset=dataset,
            **(prompt_config or {}),
            logger=self.logger
        )
        
        # Pass both server_manager AND assigned_port to ReasoningGenerator
        # Server manager is needed for port recovery even with assigned port
        self.generator = ReasoningGenerator(
            model=model,
            logger=self.logger,
            server_manager=server_manager,
            assigned_port=assigned_port
        )
        
        self.termination_checker = TerminationChecker(logger=self.logger)
        self.answer_parser = AnswerParser(logger=self.logger)
        
        # Default configuration
        self.config = self._get_default_config()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration based on dataset and model."""
        config = {
            "initial_retrieval_k": 15,     # K documents for initial retrieval (increased for better foundation)
            "iterative_retrieval_k": 3,     # K documents per iteration (balanced for multi-step reasoning)
            "max_iterations": 10,           # Maximum reasoning iterations (stops early when answer found)
            "max_total_docs": 30,           # Maximum cumulative documents (balanced for context size)
            "enable_final_reader": True,    # Use final reader step
            "temperature": 0.0,             # Generation temperature
            "max_tokens": 150,              # Max tokens per generation
        }
        
        # Dataset-specific adjustments
        if self.dataset in ["musique"]:
            config["max_iterations"] = 7  # More complex reasoning
            config["max_total_docs"] = 35  # Increased from 20 to handle large docs
        elif self.dataset in ["2wikimultihopqa"]:
            config["max_total_docs"] = 30  # Multi-hop needs more docs
        
        # Model-specific adjustments
        if "gemini" in self.model.lower():
            config["max_tokens"] = 512  # Increased for fuller reasoning
            config["max_iterations"] = 5  # Match Gemini's limit in scaled_silver_labeling
        elif "qwen" in self.model.lower():
            config["max_length"] = 1024  # CRITICAL: Increased for complete IRCoT reasoning
            config["max_tokens"] = 1024  # Ensure complete multi-step answers
            config["max_iterations"] = 5  # Match Gemini's limit in scaled_silver_labeling
        elif "flan" in self.model.lower():
            config["max_length"] = 200  # Use max_length for LLMClientGenerator
            
        return config
    
    def run(self, 
            question: str,
            ground_truth: Optional[List[str]] = None,
            config_overrides: Optional[Dict[str, Any]] = None,
            initial_contexts: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        """
        Execute the IRCOT algorithm on a question.
        
        Args:
            question: The question to answer
            ground_truth: Optional ground truth answers for evaluation
            config_overrides: Optional config overrides
            initial_contexts: Optional pre-computed contexts to use for initial retrieval (like ONER)
            
        Returns:
            Dictionary containing:
            - answer: Final answer string
            - reasoning_chain: Complete reasoning chain
            - retrieval_history: List of retrieval queries and results
            - iteration_count: Number of iterations performed
            - cumulative_docs: All retrieved documents
            - correct: Boolean if ground truth provided
            - latency: Total execution time
        """
        start_time = time.time()
        
        # Merge configurations
        config = {**self.config, **(config_overrides or {})}
        
        # Initialize state with deep copy of config to prevent worker contamination
        import copy
        import threading
        worker_id = f"worker_{threading.current_thread().ident}"
        self.logger.info(f"🧠 DEBUG: {worker_id} starting IRCoT for question: {question[:100]}...")
        
        state = IRCoTState(question=question, config=copy.deepcopy(config))
        state_instance_id = id(state)
        self.logger.info(f"🧠 DEBUG: {worker_id} created state instance {state_instance_id}")
        
        try:
            # Validate input
            if not question or not question.strip():
                self.logger.warning("Empty or whitespace-only question provided")
                return {
                    "answer": "",
                    "reasoning_chain": "",
                    "retrieval_history": [],
                    "iteration_count": 0,
                    "cumulative_docs": [],
                    "latency": time.time() - start_time,
                    "error": "Empty question provided"
                }
            
            # Phase 1: Initial Retrieval
            self.logger.info(f"🔍 IRCOT: Starting with question: {question}")
            initial_docs = self._initial_retrieval(state, initial_contexts)
            
            # Phase 2: Retrieve-Reason-Retrieve Loop
            self._reasoning_loop(state)
            
            # Phase 3: Final Answer Extraction
            # Check if early termination already found an answer
            if state.final_answer:
                final_answer = state.final_answer
                self.logger.info(f"✅ Using early termination answer: {final_answer}")
            elif config["enable_final_reader"]:
                final_answer = self._final_reader(state)
            else:
                final_answer = self._extract_answer_from_chain(state)
            
            # Prepare results
            result = {
                "answer": final_answer,
                "reasoning_chain": state.get_full_reasoning_chain(),
                "retrieval_history": state.retrieval_history,
                "iteration_count": state.iteration_count,
                "cumulative_docs": state.get_all_documents(),
                "latency": time.time() - start_time,
            }
            
            # Check correctness if ground truth provided
            if ground_truth:
                result["correct"] = self._check_correctness(final_answer, ground_truth)
            
            self.logger.info(f"✅ IRCOT: Completed in {result['latency']:.2f}s with {state.iteration_count} iterations")
            self.logger.info(f"📝 Final answer: {final_answer}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"❌ IRCOT failed: {e}")
            import traceback
            self.logger.error(traceback.format_exc())
            
            # Return error result
            return {
                "answer": "",
                "reasoning_chain": state.get_full_reasoning_chain(),
                "retrieval_history": state.retrieval_history,
                "iteration_count": state.iteration_count,
                "cumulative_docs": state.get_all_documents(),
                "latency": time.time() - start_time,
                "error": str(e),
            }
    
    def _initial_retrieval(self, state: IRCoTState, initial_contexts: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """Perform initial retrieval based on the question or use pre-computed contexts."""
        
        if initial_contexts:
            # Use pre-computed contexts (like ONER does)
            self.logger.info(f"🔄 Using {len(initial_contexts)} pre-computed contexts (ONER-style)")
            docs = []
            for ctx in initial_contexts:
                # Create completely fresh document objects with deep isolation
                import copy
                from datetime import datetime
                import threading
                doc = {
                    "title": copy.deepcopy(ctx.get("title", "")),
                    "text": copy.deepcopy(ctx.get("paragraph_text", ctx.get("text", ""))),
                    "score": 1.0,
                    "metadata": {
                        "source": "pre_computed", 
                        "timestamp": datetime.now().isoformat(),
                        "worker_thread": threading.current_thread().ident,
                        "state_instance": id(state)
                    }
                }
                docs.append(doc)
            
            state.add_documents(docs, query=state.question, stage="initial_precomputed")
            self.logger.info(f"📚 Using {len(docs)} pre-computed documents")
        else:
            # Do fresh retrieval
            self.logger.info(f"🔍 Initial retrieval with K={state.config['initial_retrieval_k']}")
            
            docs = self.retriever.retrieve(
                query=state.question,
                k=state.config["initial_retrieval_k"],
                dataset=self.dataset
            )
            
            state.add_documents(docs, query=state.question, stage="initial")
            self.logger.info(f"📚 Retrieved {len(docs)} initial documents")
        
        return docs
    
    def _reasoning_loop(self, state: IRCoTState):
        """Execute the main retrieve-reason-retrieve loop."""
        max_iterations = state.config["max_iterations"]
        
        for iteration in range(max_iterations):
            state.iteration_count = iteration + 1
            self.logger.info(f"\n🔄 IRCOT Iteration {iteration + 1}/{max_iterations}")
            
            # Step 1: Generate reasoning
            reasoning_step = self._generate_reasoning(state)
            
            if not reasoning_step:
                self.logger.warning("Empty reasoning generated, terminating")
                break
                
            state.add_reasoning_step(reasoning_step)
            
            # Step 2: Check termination
            should_terminate, answer = self.termination_checker.should_terminate(
                reasoning_step=reasoning_step,
                state=state
            )
            
            if should_terminate:
                self.logger.info(f"🛑 Termination detected: {answer}")
                if answer:
                    state.set_answer(answer)
                break
            
            # Step 3: Iterative retrieval (always continue, will use sliding window if needed)
            self._iterative_retrieval(state, reasoning_step)
    
    def _generate_reasoning(self, state: IRCoTState) -> str:
        """Generate the next reasoning step."""
        # Build prompt based on current state
        if state.iteration_count == 1:
            prompt = self.prompt_builder.build_initial_prompt(state)
        else:
            prompt = self.prompt_builder.build_continuation_prompt(state)
        
        # Generate reasoning
        # Get the appropriate parameter name based on model type
        if "qwen" in self.model.lower() or "flan" in self.model.lower():
            max_param = state.config.get("max_length", state.config.get("max_tokens", 300))
        else:
            max_param = state.config.get("max_tokens", 300)
            
        response = self.generator.generate(
            prompt=prompt,
            temperature=state.config["temperature"],
            max_tokens=max_param,
            stop_sequences=self._get_stop_sequences()
        )
        
        # Extract first sentence if sentence-by-sentence generation
        reasoning_step = self._extract_reasoning_sentence(response)
        
        # Log successful reasoning generation
        import threading
        worker_id = f"worker_{threading.current_thread().ident}"
        self.logger.debug(f"✅ {worker_id} generated reasoning: {reasoning_step[:50]}...")
        
        self.logger.info(f"🧠 Generated: {reasoning_step}")
        return reasoning_step
    
    def _iterative_retrieval(self, state: IRCoTState, reasoning_step: str):
        """Perform retrieval based on the latest reasoning step."""
        # Extract query from reasoning
        query = self._extract_retrieval_query(reasoning_step, state.question)
        
        if not query:
            self.logger.warning("Could not extract retrieval query from reasoning")
            return
        
        self.logger.info(f"🔍 Iterative retrieval: '{query}' with K={state.config['iterative_retrieval_k']}")
        
        # CRITICAL FIX: Enhanced isolation with query-specific exclusion
        exclude_titles = state.get_retrieved_titles()
        # Add unique identifiers to prevent any cross-contamination
        import threading
        thread_id = threading.current_thread().ident
        state_id = id(state)
        exclude_titles.add(f"query_isolation_{thread_id}_{state_id}")  # Unique per query instance
        
        docs = self.retriever.retrieve(
            query=query,
            k=state.config["iterative_retrieval_k"],
            dataset=self.dataset,
            exclude_titles=exclude_titles  # Avoid duplicates with worker isolation
        )
        
        if docs:
            docs_before = len(state.get_all_documents())
            state.add_documents(docs, query=query, stage=f"iteration_{state.iteration_count}")
            docs_after = len(state.get_all_documents())
            
            if docs_after >= state.config["max_total_docs"]:
                self.logger.info(f"📚 Retrieved {len(docs)} new documents, total: {docs_after} (using sliding window)")
            else:
                self.logger.info(f"📚 Retrieved {len(docs)} new documents, total: {docs_after}")
        else:
            self.logger.info("📚 No new documents found")
    
    def _final_reader(self, state: IRCoTState) -> str:
        """Use a final reader prompt to synthesize the answer."""
        self.logger.info("📖 Running final reader step")
        
        prompt = self.prompt_builder.build_final_reader_prompt(state)
        
        # Use appropriate max_tokens value
        max_param = 200
        response = self.generator.generate(
            prompt=prompt,
            temperature=0.0,
            max_tokens=max_param
        )
        
        # Extract answer from reader response
        answer = self.answer_parser.extract_answer(response)
        
        return answer
    
    def _extract_answer_from_chain(self, state: IRCoTState) -> str:
        """Extract answer directly from the reasoning chain."""
        # Try to find answer in the last reasoning step
        if state.reasoning_chain:
            last_step = state.reasoning_chain[-1]
            answer = self.answer_parser.extract_answer(last_step)
            if answer:
                return answer
        
        # Try to find answer anywhere in the chain
        full_chain = state.get_full_reasoning_chain()
        answer = self.answer_parser.extract_answer(full_chain)
        
        return answer or ""
    
    def _extract_retrieval_query(self, reasoning_step: str, original_question: str) -> str:
        """Extract an effective retrieval query focusing on NEW information and avoiding repetition."""
        import re
        
        # Remove meta-commentary and focus on entities/facts
        meta_patterns = [
            r"^(I need to find|I need to identify|I need to determine|Let me find|Let me search for|Now I need to find|Next, I need to find)\s*",
            r"^(The user is asking|The question asks|The context states|The Wikipedia title)",
            r"^(Therefore|This directly|This confirms|This supports)",
            r"\.\s*This.*$",  # Remove trailing explanations
        ]
        
        query = reasoning_step
        for pattern in meta_patterns:
            query = re.sub(pattern, "", query, flags=re.IGNORECASE).strip()
        
        # Extract key entities and relationships (limit to 15 words max)
        words = query.split()
        if len(words) > 15:
            # Keep the most important parts - usually entities and relationships
            query = " ".join(words[:15])
        
        # Remove quotes around titles to make queries more searchable
        query = re.sub(r'"([^"]+)"', r'\1', query)
        
        # Remove trailing punctuation and clean up
        query = query.rstrip(".,!?").strip()
        
        # If query became too short or empty, extract key nouns from original reasoning
        if len(query.split()) < 3:
            # Extract proper nouns and key terms
            nouns = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', reasoning_step)
            if nouns:
                query = " ".join(nouns[:5])  # Max 5 key entities
            else:
                query = reasoning_step[:60]  # Fallback to first 60 chars
        
        return query
    
    def _extract_reasoning_sentence(self, response: str) -> str:
        """Extract the first logical reasoning step from the response."""
        response = response.strip()
        if not response:
            return ""
        
        # Handle numbered lists properly - extract the full first point
        import re
        if re.match(r'^\d+\.', response):
            # Find the end of the first numbered point
            lines = response.split('\n')
            first_line = lines[0].strip()
            
            # If first line ends mid-sentence, continue to next lines until complete thought
            if not re.search(r'[.!?]$', first_line):
                for i, line in enumerate(lines[1:], 1):
                    if line.strip() and not re.match(r'^\d+\.', line.strip()):
                        first_line += " " + line.strip()
                        if re.search(r'[.!?]$', line.strip()):
                            break
            
            return first_line
        
        # For non-numbered responses, take first complete sentence
        sentences = re.split(r'(?<=[.!?])\s+', response)
        if sentences:
            return sentences[0].strip()
        
        # Fallback: take first line
        return response.split('\n')[0].strip()
    
    def _get_stop_sequences(self) -> List[str]:
        """Get stop sequences based on the model."""
        if "gemini" in self.model.lower():
            return ["\n\n\n"]  # Only stop on multiple newlines, allow single newlines for numbered lists
        elif "qwen" in self.model.lower():
            return ["\n\n", "</s>"]
        else:  # flan-t5
            return ["\n\n"]
    
    def _check_correctness(self, answer: str, ground_truths: List[str]) -> bool:
        """Check if the answer matches any ground truth."""
        if not answer:
            return False
            
        answer_lower = answer.lower().strip()
        
        for truth in ground_truths:
            truth_lower = truth.lower().strip()
            # Exact match
            if answer_lower == truth_lower:
                return True
            # Containment check
            if truth_lower in answer_lower or answer_lower in truth_lower:
                return True
                
        return False 