
"""
Original Labeling Strategy

This module implements the original labeling strategy from the Adaptive-RAG paper:
NOR → ONER → IRCOT priority with all systems running on all samples.

Open Source Models are in beta so needs more testing.
"""

import os
import sys
import logging
import time
import hashlib
import requests
from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scaled_silver_labeling.labeling.base_labeler import BaseLabeler
from scaled_silver_labeling.utils.common import get_timestamp, validate_dataset_name, validate_model_name
from scaled_silver_labeling.utils.prompt_loader import PromptLoader
from scaled_silver_labeling.adapters.ircot_bridge_adapter import IRCoTBridgeAdapter
from data.dataset_processor import DatasetProcessor
from transformers import AutoTokenizer

# Add Qwen-specific constants
QWEN_MODELS = ["Qwen/Qwen2.5-7B-Instruct"]
FLAN_T5_MODELS = ["flan-t5-large", "flan-t5-xl", "flan-t5-xxl"]

class OriginalLabeler(BaseLabeler):
    """
    Original labeling strategy implementation following the Adaptive-RAG paper
    
    This strategy implements the original paper approach:
    1. Runs ALL THREE SYSTEMS (NOR, ONER, IRCOT) on every sample - no early stopping
    2. Labels samples based on priority: NOR → ONER → IRCOT (simplest working method wins)
    3. Uses dataset bias for samples where all systems fail
    """
    
    def __init__(self, server_manager, logger):
        """
        Initialize the original labeler
        
        Args:
            server_manager: LLM server manager for handling requests
            logger: Logger instance for recording operations
        """
        super().__init__(server_manager, logger)
        self.systems = ["nor_qa", "oner_qa", "ircot_qa"]
        self.dataset_processor = DatasetProcessor()
        
        # Initialize prompt loader for dynamic example loading
        try:
            self.prompt_loader = PromptLoader()
            self.logger.info("Initialized PromptLoader for dynamic example loading")
        except Exception as e:
            self.logger.warning(f"Failed to initialize PromptLoader: {e}, will use fallback examples")
            self.prompt_loader = None
        
        # Initialize IRCoT bridge adapter configuration
        try:
            # Test that IRCoT Bridge Adapter can be imported and instantiated
            test_adapter = IRCoTBridgeAdapter(logger=self.logger, server_manager=self.server_manager)
            self.logger.info("✅ IRCoT Bridge Adapter available with server manager - will create fresh instances per query")
            self.use_benchmark_ircot = True
            del test_adapter
        except Exception as e:
            self.logger.warning(f"Failed to initialize IRCoT Bridge Adapter: {e}, falling back to custom implementation")
            self.use_benchmark_ircot = False
        
        # Initialize IRCoT configuration
        self.ircot_max_docs = 6  # Default max documents for IRCoT system
        self.oner_max_docs = 10  # Default max documents for OneR system
        
        # Single-hop datasets → 'B', Multi-hop datasets → 'C'
        self.dataset_bias_labels = {
            "nq": "B",           # Single-hop dataset
            "trivia": "B",       # Single-hop dataset  
            "squad": "B",        # Single-hop dataset
            "hotpotqa": "C",     # Multi-hop dataset
            "2wikimultihopqa": "C",  # Multi-hop dataset
            "musique": "C"       # Multi-hop dataset
        }
    
    def _is_qwen_model(self, model_name: str) -> bool:
        """Check if the model is a Qwen model requiring chat template"""
        return any(qwen_model.lower() in model_name.lower() for qwen_model in QWEN_MODELS)
    
    def _is_flan_t5_model(self, model_name: str) -> bool:
        """Check if the model is a FLAN-T5 model"""
        return any(flan_model in model_name for flan_model in FLAN_T5_MODELS)
    
    def _is_gemini_model(self, model_name: str) -> bool:
        """Check if the model is a Gemini model"""
        return "gemini" in model_name.lower()
    
    def _get_fresh_ircot_adapter(self, sample_id: str = None) -> Optional[IRCoTBridgeAdapter]:
        """
        Create a fresh IRCoT adapter instance for each query to prevent contamination.
        
        This method creates a new adapter instance for each query, ensuring that
        state from previous queries doesn't contaminate the current query's processing.
        
        Args:
            sample_id: Identifier for the current sample/query
            
        Returns:
            Fresh IRCoTBridgeAdapter instance or None if creation fails
        """
        if not sample_id:
            import time
            import threading
            worker_id = f"worker_{threading.current_thread().ident}"
            sample_id = f"{worker_id}_{int(time.time() * 1000000)}"
        
        self.logger.info(f"🔍 DEBUG: Creating fresh IRCoT adapter for sample {sample_id}")
        
        try:
            adapter = IRCoTBridgeAdapter(logger=self.logger, server_manager=self.server_manager)
            adapter_instance_id = id(adapter)
            self.logger.info(f"✅ DEBUG: Created FRESH IRCoT adapter for sample {sample_id} (instance ID: {adapter_instance_id}) with server manager")
            return adapter
        except Exception as e:
            self.logger.error(f"Failed to create IRCoT adapter for sample {sample_id}: {e}")
            return None
    
    def _get_qwen_tokenizer(self):
        """Get the tokenizer for Qwen models."""
        return AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

    def _create_qwen_prompt(self, question: str, context: str = None) -> str:
        """Beta: Create a Qwen-specific prompt with chat template and few-shot examples"""
        if context:
            system_message = "You are a highly knowledgeable assistant. You MUST answer the question using both the provided context AND your extensive parametric knowledge. Never say you don't have enough information or cannot answer. If the context is incomplete, use your training knowledge to fill gaps, make educated inferences, and synthesize information creatively. Connect related concepts, make reasonable assumptions, and provide your best educated guess. Always deliver a substantive, specific answer. End your response with 'Answer is: ' followed by your concise answer."
            few_shot_examples = self._get_few_shot_examples("qwen", with_context=True)
            user_message = f"{few_shot_examples}\n\nContext: {context}\n\nQuestion: {question}"
        else:
            system_message = "You are a highly knowledgeable assistant. You MUST answer the question using your extensive parametric knowledge from training. Never say you don't know or lack information. Use your training knowledge to make educated inferences, connect related concepts, and think creatively. Use analogical reasoning and synthesize information from related topics. Always provide your best educated guess with confidence. End your response with 'Answer is: ' followed by your concise answer."
            few_shot_examples = self._get_few_shot_examples("qwen", with_context=False)
            user_message = f"{few_shot_examples}\n\nQuestion: {question}"
        
        # Format as chat template
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        # Truncate the prompt to fit within the model's context window
        tokenizer = self._get_qwen_tokenizer()
        max_length = 32000  
        tokens = tokenizer.encode(prompt, truncation=True, max_length=max_length)
        return tokenizer.decode(tokens)
    
    def _create_gemini_prompt(self, question: str, context: str = None) -> str:
        """
        Create a Gemini-specific prompt with structured answer format and few-shot examples
        
        Args:
            question: The question to answer
            context: Optional context from retrieval
            
        Returns:
            Formatted prompt string for Gemini models
        """
        if context:
            instruction = f"Answer the question using both the provided context documents AND your extensive parametric knowledge. You MUST provide a substantive answer - never say you don't have enough information. If the context is incomplete, use your training knowledge to fill gaps and make educated inferences. Synthesize information creatively, make reasonable assumptions, and provide your best educated guess. Think step-by-step and connect related concepts.\n\nProvide a clear concise explanation, then end your response with 'Answer is: [your specific answer]'. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples("gemini", with_context=True)
            prompt = f"{instruction}{few_shot_examples}Context: {context}\n\nQuestion: {question}\n\nAnswer is:"
        else:
            instruction = f"Answer the question using your extensive parametric knowledge from training. You MUST provide a substantive answer - never say you don't know or lack information. Use your training knowledge to make educated inferences, connect related concepts, and provide your best educated guess. Think about related topics, use analogical reasoning, and synthesize information creatively.\n\nProvide a clear concise explanation, then end your response with 'Answer is: [your specific answer]'. Here are some examples:"
            few_shot_examples = self._get_few_shot_examples("gemini", with_context=False)
            prompt = f"{instruction}{few_shot_examples}Question: {question}\n\nAnswer is:"
        return prompt
    
    def _create_flan_t5_prompt(self, question: str, context: str = None) -> str:
        """Create a FLAN-T5 specific prompt with few-shot examples"""
        if context:
            instruction = "You MUST answer this question. Use both the provided context and your extensive training knowledge. If context is incomplete, fill gaps with your parametric knowledge. Make educated inferences, connect related concepts, and provide your best educated guess. Never say you cannot answer or lack information. Always provide a specific, substantive answer."
            return f"{instruction}\n\nQuestion: {question}\nContext: {context}\nAnswer is:"
        else:
            instruction = "You MUST answer this question using your extensive training knowledge. Never say you don't know. Make educated inferences, connect related concepts, and use analogical reasoning. Think creatively and provide your best educated guess with confidence. Always provide a specific, substantive answer."
            return f"{instruction}\n\nQuestion: {question}\nAnswer is:"
    
    def _create_ircot_prompt(self, question: str, model_name: str, context: str, iteration: int, max_iterations: int) -> str:
        """Create IRCOT-specific prompts optimized for iterative reasoning"""
        if self._is_flan_t5_model(model_name):
            return self._create_flan_t5_ircot_prompt(question, context, iteration, max_iterations)
        elif self._is_gemini_model(model_name):
            return self._create_gemini_ircot_prompt(question, context, iteration, max_iterations)
        elif self._is_qwen_model(model_name):
            return self._create_qwen_ircot_prompt(question, context, iteration, max_iterations)
        else:
            # Default to FLAN-T5 style
            return self._create_flan_t5_ircot_prompt(question, context, iteration, max_iterations)
    
    def _create_flan_t5_ircot_prompt(self, question: str, context: str, iteration: int, max_iterations: int) -> str:
        """Create FLAN-T5 specific IRCOT prompt with clear instructions"""
        if iteration == max_iterations - 1:
            # Final iteration - request definitive answer with clear format
            return f"""You MUST answer this question based on the context and reasoning so far. Use all available information plus your training knowledge to provide a definitive answer. Never say you cannot answer. Make educated inferences and provide your best judgment.

{context}

Question: {question}

Instructions: Provide a direct, specific answer. Use the context and your knowledge to make educated inferences. Give your best educated guess with confidence.

Answer is:"""
        else:
            # Intermediate iteration - focused reasoning step
            return f"""Think about this question step by step. Use the context and your training knowledge to reason. Make connections and inferences beyond just the provided text.

{context}

Question: {question}

Instructions: Write one clear reasoning step. Be specific, make inferences, and connect related concepts. Use both context facts and your broader knowledge.

Step {iteration + 1}:"""
    
    def _create_gemini_ircot_prompt(self, question: str, context: str, iteration: int, max_iterations: int) -> str:
        """Create Gemini specific IRCOT prompt with structured format and clear instructions"""
        if iteration == max_iterations - 1:
            # Final iteration - explicitly request answer with strict format
            instruction = """You MUST answer this question using the provided context and your extensive knowledge. Never say you cannot answer or lack information. Use all available information to make educated inferences and provide your best judgment.

CRITICAL: End your response with exactly this format: "Answer is: [your specific final answer]"

Make educated guesses, connect related concepts, and provide a definitive answer. Use your training knowledge to fill any gaps."""
            
            few_shot_examples = self._get_gemini_ircot_examples()
            return f"""{instruction}

{few_shot_examples}

{context}

Question: {question}

Answer is:"""
        else:
            # Intermediate iteration - focused reasoning without repetition
            return f"""Based on the context below, think about the question step by step. Use both the context and your training knowledge to reason creatively. Make connections and inferences beyond just the provided text.

{context}

Question: {question}

Instructions: 
- Write one clear reasoning step that advances toward the answer
- Make inferences and connect related concepts from your knowledge
- Use both context facts and broader knowledge to reason
- Be specific and think creatively

Reasoning step {iteration + 1}:"""
    
    def _create_qwen_ircot_prompt(self, question: str, context: str, iteration: int, max_iterations: int) -> str:
        """Create Qwen specific IRCOT prompt with chat template format and structured instructions"""
        if iteration == max_iterations - 1:
            # Final iteration - explicitly request answer with strict format
            system_message = """You are a highly knowledgeable assistant. You MUST answer this question using the provided context and your extensive knowledge. Never say you cannot answer or lack information. Use all available information to make educated inferences and provide your best judgment.

CRITICAL: End your response with exactly this format: "Answer is: [your specific final answer]"

Make educated guesses, connect related concepts, and provide a definitive answer. Use your training knowledge to fill any gaps."""
            
            few_shot_examples = self._get_qwen_ircot_examples()
            user_message = f"""{few_shot_examples}

{context}

Question: {question}"""
        else:
            # Intermediate iteration - focused reasoning without repetition
            system_message = """You are a highly knowledgeable assistant. Based on the context below, think about the question step by step. Use both the context and your training knowledge to reason creatively. Make connections and inferences beyond just the provided text.

Instructions: 
- Write one clear reasoning step that advances toward the answer
- Make inferences and connect related concepts from your knowledge
- Use both context facts and broader knowledge to reason
- Be specific and think creatively"""
            
            user_message = f"""{context}

Question: {question}

Reasoning step {iteration + 1}:"""
        
        # Format as Qwen chat template
        prompt = f"<|im_start|>system\n{system_message}<|im_end|>\n<|im_start|>user\n{user_message}<|im_end|>\n<|im_start|>assistant\n"
        
        # Truncate the prompt to fit within the model's context window
        tokenizer = self._get_qwen_tokenizer()
        max_length = 32000  
        tokens = tokenizer.encode(prompt, truncation=True, max_length=max_length)
        return tokenizer.decode(tokens)
    
    def _create_original_ircot_prompt(self, question: str, context: str, model_name: str) -> str:
        """
        Create prompt using original IRCOT approach from the paper - safe examples NOT from evaluation datasets
        
        Args:
            question: The question to answer
            context: Retrieved documents formatted as context
            model_name: Name of the model being used
            
        Returns:
            Properly formatted prompt following original IRCOT structure
        """
        # Get few-shot examples based on model type
        if self._is_flan_t5_model(model_name):
            examples = self._get_safe_ircot_examples_flan_t5()
            prompt_prefix = "Answer the following question by reasoning step-by-step.\n"
        elif self._is_qwen_model(model_name):
            examples = self._get_safe_ircot_examples_qwen()
            prompt_prefix = ""
        else:  # Gemini and other models
            examples = self._get_safe_ircot_examples_gemini()
            prompt_prefix = ""
        
        # Format the test example
        if context.strip():
            test_example = f"{context}\n\nQ: {prompt_prefix}{question}\nA:"
        else:
            test_example = f"Q: {prompt_prefix}{question}\nA:"
        
        # Combine examples with test case using proper delimiter for the model
        if self._is_gemini_model(model_name):
            # Gemini expects \n\n\n delimiter between demonstrations
            full_prompt = f"{examples}\n\n\n{test_example}"
        else:
            # FLAN-T5 and other models use \n\n
            full_prompt = f"{examples}\n\n{test_example}"
        
        return full_prompt
    
    def _get_safe_ircot_examples_flan_t5(self) -> str:
        """Get few-shot examples for FLAN-T5 from existing prompt infrastructure"""
        if self.prompt_loader:
            try:
                # Use current dataset if available, with IRCOT paper-aligned settings
                current_dataset = getattr(self, 'current_dataset', None)
                return self.prompt_loader.get_few_shot_examples(
                    model_name="flan_t5",
                    dataset=current_dataset,
                    context_type="gold_with_1_distractors",  # Paper specification: distractor_count="1"
                    qa_type="cot"  # num_examples=None uses model default per IRCOT paper
                )
            except Exception as e:
                self.logger.error(f"Failed to load FLAN-T5 examples from prompt loader: {e}")
                raise RuntimeError(f"IRCOT prompt loading failed for FLAN-T5. Expected prompt files missing: {e}")

    def _get_safe_ircot_examples_gemini(self) -> str:
        """Get few-shot examples for Gemini from existing prompt infrastructure"""
        if self.prompt_loader:
            try:
                # Use current dataset if available, with IRCOT paper-aligned settings
                current_dataset = getattr(self, 'current_dataset', None)
                return self.prompt_loader.get_few_shot_examples(
                    model_name="gemini-2.5-flash-lite",  # Use stable model for prompt examples
                    dataset=current_dataset,
                    context_type="gold_with_1_distractors",  # Paper specification: distractor_count="1"
                    qa_type="cot"  # num_examples=None uses model default per IRCOT paper
                )
            except Exception as e:
                self.logger.error(f"Failed to load Gemini examples from prompt loader: {e}")
                raise RuntimeError(f"IRCOT prompt loading failed for Gemini. Expected prompt files missing: {e}")

    def _get_gemini_ircot_examples(self) -> str:
        """Get few-shot examples specifically for Gemini IRCOT final answer extraction from existing prompt infrastructure"""
        if self.prompt_loader:
            try:
                # Use current dataset if available, with IRCOT paper-aligned settings
                current_dataset = getattr(self, 'current_dataset', None)
                return self.prompt_loader.get_ircot_examples(
                    model_name="gemini-2.5-flash-lite",  # Use stable model for prompt examples
                    dataset=current_dataset
                    # num_examples=None uses model default per IRCOT paper
                )
            except Exception as e:
                self.logger.error(f"Failed to load Gemini IRCOT examples from prompt loader: {e}")
                raise RuntimeError(f"IRCOT prompt loading failed for Gemini. Expected prompt files missing: {e}")
    
    def _get_safe_ircot_examples_qwen(self) -> str:
        """Get few-shot examples for Qwen from existing prompt infrastructure"""
        if self.prompt_loader:
            try:
                # Use current dataset if available, with IRCOT paper-aligned settings
                current_dataset = getattr(self, 'current_dataset', None)
                return self.prompt_loader.get_few_shot_examples(
                    model_name="qwen",
                    dataset=current_dataset,
                    context_type="gold_with_1_distractors",  # Paper specification: distractor_count="1"
                    qa_type="cot"  # num_examples=None uses model default per IRCOT paper
                )
            except Exception as e:
                self.logger.error(f"Failed to load Qwen examples from prompt loader: {e}")
                raise RuntimeError(f"IRCOT prompt loading failed for Qwen. Expected prompt files missing: {e}")
    
    def _get_qwen_ircot_examples(self) -> str:
        """Get few-shot examples specifically for Qwen IRCOT final answer extraction from existing prompt infrastructure"""
        if self.prompt_loader:
            try:
                # Use current dataset if available, with IRCOT paper-aligned settings
                current_dataset = getattr(self, 'current_dataset', None)
                return self.prompt_loader.get_ircot_examples(
                    model_name="qwen",
                    dataset=current_dataset
                    # num_examples=None uses model default per IRCOT paper
                )
            except Exception as e:
                self.logger.error(f"Failed to load Qwen IRCOT examples from prompt loader: {e}")
                raise RuntimeError(f"IRCOT prompt loading failed for Qwen. Expected prompt files missing: {e}")
    
    def _extract_answer_from_qwen_response(self, response: str) -> str:
        """
        Extract answer from Qwen's response with improved handling for verbosity and repetition
        
        Args:
            response: Raw response from Qwen model
            
        Returns:
            Extracted answer string
        """
        import re
        response = response.strip()
        
        # Try structured extraction first
        structured = self._extract_structured_answer(response)
        if structured and structured != response and len(structured) < 100:
            return structured
        
        # Look for direct answer patterns (Qwen often uses these)
        patterns = [
            r'(?:the\s+)?(?:direct\s+)?answer\s+is:?\s*([^.\n]+)',
            r'Based on.*?,?\s*([^.\n]+)',
            r'Therefore,?\s*([^.\n]+)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                answer = match.group(1).strip()
                if answer and len(answer) < 80:
                    return answer
        
        # If still verbose, extract the shortest meaningful statement
        if len(response) > 200:
            sentences = response.split('.')
            candidates = []
            for sent in sentences:
                sent = sent.strip()
                if sent and 10 < len(sent) < 80:
                    # Prefer sentences without reasoning indicators
                    if not any(word in sent.lower() for word in ['step', 'need', 'find', 'first', 'second']):
                        candidates.append(sent)
            
            if candidates:
                # Return shortest candidate
                return min(candidates, key=len)
        
        # Final fallback: first sentence if reasonable
        first = response.split('.')[0].strip()
        if first and len(first) < 100:
            return first
        
        return response[:100].strip() + "..." if len(response) > 100 else response
    
    def _extract_answer_from_gemini_response(self, response: str) -> str:
        """
        Extract answer from Gemini's response with improved parsing for verbose responses
        
        Args:
            response: Raw response from Gemini model
            
        Returns:
            Extracted answer string
        """
        response = response.strip()
        
        # First try structured extraction
        structured_answer = self._extract_structured_answer(response)
        if structured_answer and structured_answer != response:
            return structured_answer
        
        # Handle cases where Gemini provides correct reasoning but poor final answer
        import re
        
        # Look for explicit answer patterns
        answer_patterns = [
            r'(?:the\s+)?(?:final\s+)?answer\s+is:?\s*(.+?)(?:\.|$)',
            r'(?:therefore|thus|so),?\s*(?:the\s+)?answer\s+is:?\s*(.+?)(?:\.|$)',
            r'(?:conclusion|result):\s*(.+?)(?:\.|$)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE | re.DOTALL)
            if matches:
                # Take the last (most specific) match
                answer = matches[-1].strip()
                # Clean up common artifacts
                answer = re.sub(r'\s+', ' ', answer)  # Normalize whitespace
                answer = answer.split('.')[0]  # Take only first sentence
                if answer and len(answer) < 200:  # Reasonable length check
                    return answer
        
        # If verbose response, try to extract the most confident statement
        sentences = re.split(r'[.!?]+', response)
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for definitive statements
            if any(indicator in sentence.lower() for indicator in ['is', 'was', 'are', 'were']) and len(sentence) < 100:
                return sentence
        
        # Fallback: return original response
        return response
    
    def _extract_ircot_final_answer(self, complete_reasoning: str, model_name: str) -> str:
        """
        Extract final answer from complete IRCOT reasoning with improved logic
        """
        import re
        
        # First, try standard extraction
        answer = self._extract_model_specific_answer(complete_reasoning, model_name)
        
        # If answer is still the entire reasoning, apply more aggressive extraction
        if len(answer) > 100 or answer == complete_reasoning or '[' in answer:
            # Special handling for answers with brackets or notes
            if 'A:' in complete_reasoning:
                # Find all "A:" patterns and take the cleanest one
                a_matches = re.findall(r'A:\s*([^.\n\[]+)', complete_reasoning)
                if a_matches:
                    # Filter out very short or long answers
                    valid_answers = [a.strip() for a in a_matches if 2 < len(a.strip()) < 80]
                    if valid_answers:
                        # Return the most common answer if repeated, otherwise the last one
                        from collections import Counter
                        answer_counts = Counter(valid_answers)
                        most_common = answer_counts.most_common(1)[0]
                        if most_common[1] > 1:  # Repeated answer
                            return most_common[0]
                        else:
                            return valid_answers[-1]
            
            # Look for the last occurrence of answer patterns
            patterns = [
                r'(?:the\s+)?(?:final\s+)?answer\s+is:?\s*([^.\[\n]+)',
                r'(?:therefore|thus|so),?\s*([^.\[\n]+)',
                r'Therefore,?\s*([^,\.\[\n]+)',
                r'The direct answer is:?\s*([^.\[\n]+)'
            ]
            
            for pattern in patterns:
                matches = list(re.finditer(pattern, complete_reasoning, re.IGNORECASE))
                if matches:
                    # Take the last match (most recent answer)
                    answer = matches[-1].group(1).strip()
                    if answer and 2 < len(answer) < 80:
                        return answer
            
            # If we have specific entities mentioned multiple times, they might be the answer
            # Extract proper nouns or key phrases
            words = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', complete_reasoning)
            if words:
                from collections import Counter
                word_counts = Counter(words)
                # Filter out common reasoning words
                filtered_counts = {w: c for w, c in word_counts.items() 
                                 if w not in ['The', 'This', 'Therefore', 'Based', 'Note', 'Reasoning']}
                if filtered_counts:
                    most_common = max(filtered_counts.items(), key=lambda x: x[1])
                    if most_common[1] >= 2:  # Mentioned at least twice
                        return most_common[0]
        
        # Clean up answer if it has artifacts
        if answer and len(answer) > 100:
            # Try to extract just the entity/noun phrase
            first_sentence = answer.split('.')[0].strip()
            if first_sentence and len(first_sentence) < 80:
                return first_sentence
        
        return answer if len(answer) < 100 else answer[:80] + "..."
    
    def _create_ircot_reasoning_prompt(self, question: str, context: str, previous_reasoning: str, model_name: str) -> str:
        """
        Create IRCOT-style prompt for generating the next reasoning step
        
        Args:
            question: The original question
            context: All accumulated retrieved documents  
            previous_reasoning: All previous reasoning steps joined
            model_name: Name of the model being used
            
        Returns:
            Formatted prompt for next reasoning step
        """
        # Format context documents
        formatted_context = context if context.strip() else ""
        
        # Create prompt following IRCOT paper format
        if previous_reasoning.strip():
            # Continuing reasoning chain
            if self._is_qwen_model(model_name):
                prompt = f"{formatted_context}\n\nQ: {question}\nA: {previous_reasoning.strip()}"
            elif self._is_flan_t5_model(model_name):
                prompt = f"{formatted_context}\n\nQ: {question}\nA: {previous_reasoning.strip()}"
            else:
                prompt = f"{formatted_context}\n\nQ: {question}\nA: {previous_reasoning.strip()}"
        else:
            # First reasoning step
            if self._is_qwen_model(model_name):
                prompt = f"{formatted_context}\n\nQ: {question}\nA:"
            elif self._is_flan_t5_model(model_name):
                prompt = f"{formatted_context}\n\nQ: {question}\nA:"
            else:
                prompt = f"{formatted_context}\n\nQ: {question}\nA:"
                
        return prompt
    
    def _extract_query_from_reasoning(self, reasoning_step: str, original_question: str) -> str:
        """
        Extract query terms from a reasoning step for next retrieval
        
        Args:
            reasoning_step: The reasoning sentence to extract query from
            original_question: Original question for context
            
        Returns:
            Query string for retrieval, or empty string if no good query found
        """
        import re
        
        reasoning_step = reasoning_step.strip()
        if not reasoning_step:
            return ""
        
        # Skip if this is already a final answer
        if re.search(r'(?:so the answer is:?|answer is:?)', reasoning_step, re.IGNORECASE):
            return ""
        
        # Priority patterns for better entity extraction
        
        # 1. Look for "X was/is manufactured/created/invented/made/... by Y" patterns
        by_patterns = [
            r'(?:was|is)\s+(?:manufactured|created|invented|made|produced|developed|founded|established)\s+by\s+([A-Z][a-zA-Z\s]+?)(?:\s*[.,]|$)',
            r'by\s+([A-Z][a-zA-Z\s]+?)(?:\s*[.,]|$)'
        ]
        
        for pattern in by_patterns:
            match = re.search(pattern, reasoning_step, re.IGNORECASE)
            if match:
                entity = match.group(1).strip()
                if len(entity) > 2:
                    return entity
        
        # 2. Look for subject-verb patterns "X invented/created/founded Y"
        subject_verb_patterns = [
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:invented|created|founded|established|discovered|made)\s+',
            r'([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)\s+(?:was born|died|lived)\s+',
        ]
        
        for pattern in subject_verb_patterns:
            match = re.search(pattern, reasoning_step)
            if match:
                entity = match.group(1).strip()
                if len(entity) > 2 and entity not in {'The', 'This', 'That'}:
                    return entity
        
        # 3. Extract meaningful proper nouns (multi-word entities preferred)
        entities = re.findall(r'\b[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*\b', reasoning_step)
        
        # Filter out common words
        common_words = {'The', 'This', 'That', 'They', 'It', 'He', 'She', 'We', 'You', 'I', 
                       'Therefore', 'Thus', 'So', 'But', 'And', 'Or', 'Not', 'Is', 'Was', 
                       'Are', 'Were', 'Have', 'Has', 'Had', 'Do', 'Does', 'Did', 'Will', 
                       'Would', 'Could', 'Should', 'May', 'Might', 'Can', 'A', 'An'}
        
        meaningful_entities = []
        for entity in entities:
            if entity not in common_words and len(entity) > 2:
                meaningful_entities.append(entity)
        
        # Prioritize multi-word entities, then longer single words
        if meaningful_entities:
            # First try multi-word entities
            multi_word = [e for e in meaningful_entities if ' ' in e]
            if multi_word:
                return max(multi_word, key=len)
            
            # Then single-word entities, prefer longer ones
            single_word = [e for e in meaningful_entities if ' ' not in e]
            if single_word:
                return max(single_word, key=len)
        
        # 5. Last resort patterns
        # Look for "the X" patterns  
        the_match = re.search(r'\bthe\s+([A-Z][a-zA-Z\s]+?)(?:\s+(?:is|was|of|in|at)|\.|$)', reasoning_step)
        if the_match:
            phrase = the_match.group(1).strip()
            if len(phrase) > 3:
                return phrase
        
        # Final fallback: return empty (use original question for next retrieval)
        return ""
    
    def _extract_answer_from_flan_t5_response(self, response: str) -> str:
        """
        Extract answer from FLAN-T5 response with improved parsing
        
        Args:
            response: Raw response from FLAN-T5 model
            
        Returns:
            Cleaned answer string
        """
        response = response.strip()
        
        # Handle common FLAN-T5 response patterns
        if response.startswith("A: "):
            return response[3:].strip()
        elif response.startswith("Answer: "):
            return response[8:].strip()
        elif "answer is:" in response.lower():
            # Extract after "answer is:"
            import re
            match = re.search(r'answer\s+is:?\s*(.+)', response, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        
        # Clean up repetitive responses (common FLAN-T5 issue)
        words = response.split()
        if len(words) > 1:
            # Check if all words are identical first
            if all(word == words[0] for word in words):
                return words[0]
            
            # Check if we have repeated phrases (like "Sierra-at-Tahoe Sierra-at-Tahoe")
            if len(words) % 2 == 0:  # Even number of words
                mid = len(words) // 2
                first_half = ' '.join(words[:mid])
                second_half = ' '.join(words[mid:])
                if first_half == second_half:
                    return first_half
        
        return response
    
    def _create_model_specific_prompt(self, question: str, model_name: str, context: str = None) -> str:
        """Create model-specific prompt based on model type"""
        if self._is_qwen_model(model_name):
            return self._create_qwen_prompt(question, context)
        elif self._is_gemini_model(model_name):
            return self._create_gemini_prompt(question, context)
        elif self._is_flan_t5_model(model_name):
            return self._create_flan_t5_prompt(question, context)
        else:
            # Default to FLAN-T5 style for unknown models
            return self._create_flan_t5_prompt(question, context)
    
    def _extract_model_specific_answer(self, response: str, model_name: str) -> str:
        """Extract answer using model-specific method"""
        if self._is_qwen_model(model_name):
            return self._extract_answer_from_qwen_response(response)
        elif self._is_gemini_model(model_name):
            return self._extract_answer_from_gemini_response(response)
        elif self._is_flan_t5_model(model_name):
            return self._extract_answer_from_flan_t5_response(response)
        else:
            # Default to FLAN-T5 extraction
            return self._extract_answer_from_flan_t5_response(response)
    
    def label_samples(self, dataset_name: str, sample_size: int, model_name: str, oner_max_docs: int = None, ircot_max_docs: int = None, samples: List[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
        """
        Label samples using the original Adaptive-RAG paper strategy.
        
        This method implements the paper's labeling approach:
        - Runs ALL THREE SYSTEMS (NOR, ONER, IRCOT) for EVERY sample
        - No early stopping - all systems always execute
        - Labels with priority: NOR → ONER → IRCOT (simplest working method wins)
        - Supports Qwen, FLAN-T5, and Gemini models with optimized prompting
        
        Args:
            dataset_name: Name of the dataset to process
            sample_size: Number of samples to label  
            model_name: Name of the model to use (qwen/flan-t5/gemini)
            oner_max_docs: Max documents for ONER system (default: 15 for T5, 6 for others)
            ircot_max_docs: Max documents for IRCOT system (default: 6 for T5, 3 for others)
            samples: Pre-filtered samples to use (optional)
            
        Returns:
            Dictionary containing aggregated results and metadata
        """
        # Set retrieval parameters based on paper specifications (run.py instantiation schemes)
        if oner_max_docs is None:
            if self._is_flan_t5_model(model_name):
                self.oner_max_docs = 15  # Encoder-decoder models can handle more documents
            else:
                self.oner_max_docs = 6   # Decoder-only models (Gemini/Qwen) use fewer documents
        else:
            self.oner_max_docs = oner_max_docs
            
        if ircot_max_docs is None:
            # Paper specification: IRCOT uses bm25_retrieval_count = ["6"] for all models  
            self.ircot_max_docs = 6
        else:
            self.ircot_max_docs = ircot_max_docs
            
        # Set current dataset for corpus mapping
        self.current_dataset = dataset_name
        
        # Start timing
        start_time = time.time()
        
        # Validate inputs
        validation_errors = self.validate_inputs(dataset_name, sample_size, model_name)
        if validation_errors:
            raise ValueError(f"Input validation failed: {validation_errors}")
        
        # Store current dataset for logging
        self.current_dataset = dataset_name
        
        self.log_labeling_start(dataset_name, sample_size, model_name)
        
        # Use provided filtered samples or load from dataset processor
        if samples is not None:
            self.logger.info(f"Using provided filtered samples ({len(samples)} samples)")
        else:
            # Fallback: Load dataset samples via dataset processor (legacy behavior)
            prediction_input = self.dataset_processor.prepare_prediction_input(dataset_name, sample_size)
            samples = prediction_input['samples']
        
        # Log start of Q&A interactions
        if hasattr(self.logger, 'log_qa_interaction'):
            self.logger.info(f"Starting Q&A logging for {dataset_name} with {sample_size} samples")
        
        # Process samples with all systems
        individual_results = []
        
        # Use configured max workers
        max_workers = min(self.parallel_config['max_workers'], len(samples))
        
        # Process samples in parallel across available servers
        self.logger.info(f"Processing {len(samples)} samples with {max_workers} parallel workers")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_sample = {
                executor.submit(self._process_sample, sample, dataset_name, model_name): sample
                for sample in samples
            }
            
            for future in as_completed(future_to_sample):
                sample = future_to_sample[future]
                try:
                    result = future.result()
                    individual_results.append(result)  # Always append results
                    
                    # Update statistics
                    if result and result.get('label') == 'DISCARDED':
                        self.update_stats(
                            total_samples_processed=1,
                            successful_labels=0,
                            failed_labels=0,
                            discarded_samples=1
                        )
                    else:
                        self.update_stats(
                            total_samples_processed=1,
                            successful_labels=1 if result and result['label'] in ['A', 'B', 'C'] else 0,
                            failed_labels=0  # Original labeler doesn't produce failed labels except for real errors
                        )
                    
                    # Log progress with timing info
                    if len(individual_results) % 10 == 0:
                        avg_steps = sum(r.get('steps', 0) for r in individual_results) / len(individual_results)
                        avg_time = sum(r.get('processing_time', 0) for r in individual_results) / len(individual_results)
                        self.logger.info(f"Progress: {len(individual_results)} samples processed, {len(individual_results)/len(samples)*100:.1f}% complete, avg steps: {avg_steps:.2f}, avg time: {avg_time:.2f}s")
                
                except Exception as e:
                    self.logger.error(f"Error processing sample {sample.get('id', 'unknown')}: {str(e)}")
                    individual_results.append(self._create_error_result(sample, str(e)))
                    self.update_stats(failed_labels=1)
            
            # Ensure all futures complete and executor shuts down properly
            executor.shutdown(wait=True)
        
        # Aggregate results
        processing_time = time.time() - start_time
        self.update_stats(processing_time=processing_time)
        
        aggregated_results = self.aggregate_results(individual_results, dataset_name, model_name)
        
        # Add strategy-specific metadata  
        discarded_count = 0  # Original labeler doesn't discard samples
        
        aggregated_results.update({
            'strategy_type': 'original',
            'systems_used': self.systems,
            'systems_skipped': [],
            'processing_time_seconds': processing_time,
            'samples_per_second': len(samples) / processing_time if processing_time > 0 else 0,
            'early_stopping_enabled': False,
            'run_all_systems': True,
            'discarded_samples_count': discarded_count
        })
        
        # Save LLM interactions if logger supports it
        if hasattr(self.logger, 'save_interactions'):
            session_metadata = {
                'strategy': 'original',
                'dataset': dataset_name,
                'model': model_name,
                'sample_size': len(samples),
                'systems_used': self.systems,
                'processing_time_seconds': processing_time,
                'run_all_systems': True,
                'early_stopping_enabled': False,
                'discarded_samples_count': discarded_count
            }
            self.logger.save_interactions(session_metadata=session_metadata)
        
        self.log_labeling_end(dataset_name, aggregated_results)
        return aggregated_results
    
    def get_required_systems(self, dataset_name: str) -> List[str]:
        """
        Get list of systems needed for this dataset
        
        Args:
            dataset_name: Name of the dataset
            
        Returns:
            List of required system names
        """
        return self.systems  # Always use all systems for original strategy
    
    def _process_sample(self, sample: Dict[str, Any], dataset_name: str, model_name: str) -> Dict[str, Any]:
        """
        Process a single sample by running ALL THREE SYSTEMS (NOR, ONER, IRCOT).
        
        This implements the original Adaptive-RAG paper approach where every sample
        is processed by all systems, then labeled with priority-based assignment.
        No early stopping - all systems always run.
        
        Args:
            sample: Sample to process
            dataset_name: Name of the dataset
            model_name: Name of the model
            
        Returns:
            Dictionary with labeling result
        """
        import time
        
        # Start timing annotation for this sample
        annotation_start_time = time.time()
        sample_steps = 0
        sample_processing_time = 0.0
        
        sample_id = self._get_sample_id(sample)
        ground_truths = self._extract_ground_truth(sample)
        
        # Run ALL THREE systems on this sample (NOR, ONER, IRCOT) - no early stopping
        system_results = {}
        system_steps_map = {}
        
        for system_name in self.systems:
            try:
                result = self._run_system(sample, system_name, model_name)
                system_results[system_name] = result
                
                # Check for resource exhausted error
                if result.get('resource_exhausted', False) or result.get('answer', '').strip() == '__RESOURCE_EXHAUSTED__':
                    self.logger.warning(f"Resource exhausted error for sample {sample_id} in {system_name} - discarding sample")
                    return self._create_discarded_result(sample, f"Resource exhausted during {system_name} system call")
                
                # Check for server unavailable error
                if result.get('server_unavailable', False) or result.get('answer', '').strip() == '__SERVER_UNAVAILABLE__':
                    error_detail = result.get('error', 'Server unavailable')
                    self.logger.warning(f"Server unavailable error for sample {sample_id} in {system_name} - discarding sample: {error_detail}")
                    return self._create_discarded_result(sample, f"Server unavailable during {system_name} system call: {error_detail}")
                
                # Track steps from this system
                system_step_count = result.get('steps', 0)
                system_steps_map[system_name] = system_step_count
                sample_processing_time += result.get('processing_time', 0.0)
                
            except Exception as e:
                self.logger.error(f"Error running {system_name} on sample {sample_id}: {str(e)}")
                system_results[system_name] = {
                    'answer': '',
                    'correct': False,
                    'error': str(e),
                    'processing_time': 0.0,
                    'steps': 0
                }
                system_steps_map[system_name] = 0
        
        # Apply original labeling logic
        label, reasoning = self._apply_original_labeling_logic(
            system_results, dataset_name, ground_truths, model_name, sample
        )
        
        # Check if LLM call failed during labeling logic
        if label is None and reasoning == "__LLM_FAILED__":
            self.logger.warning(f"LLM call failed during original labeling - discarding sample {self._get_sample_id(sample)}")
            return self._create_discarded_result(sample, "LLM call failed during original labeling synthetic check")
        
        # Determine primary system based on label
        label_to_system = {'A': 'nor_qa', 'B': 'oner_qa', 'C': 'ircot_qa'}
        primary_system = label_to_system.get(label, 'nor_qa')
        
        # Calculate total steps as sum of all systems used (nor + oner + ircot)
        total_steps = sum(system_steps_map.values())
        
        # Collect system answers for the result entry (both processed and raw)
        system_answers = {}
        for sys, res in system_results.items():
            system_answers[sys] = {
                'answer': res.get('answer', ''),
                'raw_answer': res.get('raw_answer', '')
            }
        
        # Calculate annotation time for this sample
        annotation_time = time.time() - annotation_start_time
        
        # Update logger with sample completion and label count
        if hasattr(self.logger, 'update_sample_completion'):
            sample_id = self._get_sample_id(sample)
            self.logger.update_sample_completion(sample_id, label)
        
        # Always create a result entry
        return self.create_result_entry(
            sample=sample,
            label=label,
            reasoning=reasoning,
            systems_used=list(system_results.keys()),
            systems_succeeded=[sys for sys, res in system_results.items() 
                             if self._evaluate_answer(res.get('answer', ''), ground_truths, model_name)[0]],
            system_answers=system_answers,
            annotation_time=annotation_time,
            steps=total_steps,
            system_steps=system_steps_map,
            processing_time=sample_processing_time,
            primary_system=primary_system,
            system_results=system_results
        )
    
    def _check_answer_with_semantic(self, question: str, answer: str, ground_truths: List[str], system_name: str) -> tuple[bool, str]:
        """
        Perform direct synthetic checking without subset matching
        
        Args:
            question: The question being asked
            answer: The system's answer to evaluate
            ground_truths: List of acceptable ground truth answers  
            system_name: Name of the system that produced the answer
            
        Returns:
            Tuple of (is_semantically_correct, best_ground_truth_matched)
            Special case: Returns (None, "__LLM_FAILED__") if LLM call fails and sample should be discarded
        """
        self.logger.debug(f"🔍 Starting synthetic check for {system_name}")
        
        if not self.synthetic_checker:
            self.logger.warning(f"❌ Synthetic checker not available for {system_name}")
            return False, ""
        
        if not answer or not question:
            self.logger.warning(f"❌ Missing answer or question for {system_name} synthetic check")
            return False, ""
        
        try:
            # Check against all ground truths (for multi-answer datasets like TriviaQA)
            for i, gt in enumerate(ground_truths):
                if not gt:
                    continue
                
                self.logger.debug(f"   🤖 Checking semantic equivalence:")
                self.logger.debug(f"      Answer: '{answer}'")
                self.logger.debug(f"      Ground Truth {i+1}: '{gt}'")
                    
                result = self.synthetic_checker.check_semantic_correctness(
                    question=question,
                    answer=answer,
                    ground_truth=gt,
                    source_pipeline=system_name.lower()
                )
                
                # Check if LLM call failed during synthetic checking
                if result.error:
                    self.logger.warning(f"❌ LLM call failed during semantic check for {system_name} - discarding sample: {result.error}")
                    return None, "__LLM_FAILED__"
                
                self.logger.debug(f"      Result: {'✅ SEMANTICALLY CORRECT' if result.is_semantically_correct else '❌ Not equivalent'}")
                
                if result.is_semantically_correct:
                    self.logger.debug(f"✅ LLM semantic match found for {system_name}: '{answer}' ≡ '{gt}'")
                    return True, gt
                    
            self.logger.debug(f"❌ No semantic match found for {system_name} after checking {len(ground_truths)} ground truths")
            return False, ""
                    
        except Exception as e:
            self.logger.error(f"❌ Synthetic checker failed for {system_name}: {e}")
            import traceback
            self.logger.error(f"   Traceback: {traceback.format_exc()}")
            # Treat exception as LLM failure and discard sample
            return None, "__LLM_FAILED__"

    def _apply_original_labeling_logic(self, 
                                    system_results: Dict[str, Any],
                                    dataset_name: str,
                                    ground_truths: List[str],
                                    model_name: str,
                                    sample: Dict[str, Any] = None) -> tuple[str, str]:
        """
        Apply the original Adaptive-RAG labeling logic with priority-based assignment.
        
        
        Priority order (simplest first):
        1. If NOR system answers correctly -> 'A' (no retrieval needed)
        2. If ONER system answers correctly -> 'B' (single-step retrieval needed)
        3. If IRCOT system answers correctly -> 'C' (multi-step retrieval needed)
        4. If none succeed -> use dataset bias
        
        Args:
            system_results: Results from ALL THREE QA systems (NOR, ONER, IRCOT)
            dataset_name: Name of the dataset for fallback logic
            ground_truths: List of ground truths for the sample
            model_name: Name of the model for evaluation
            sample: Original sample data
            
        Returns:
            Tuple of (label, reasoning) where label can be 'A', 'B', or 'C'
        """
        # Check NOR system first (priority labeling)
        nor_result = system_results.get('nor_qa', {})
        nor_answer = nor_result.get('answer', '')
        nor_raw_answer = nor_result.get('raw_answer', '')
        
        # Try exact match first
        is_correct, matched_gt = self._evaluate_answer(nor_answer, ground_truths, model_name)
        if is_correct:
            return 'A', "NOR system answered correctly (exact match)"
        
        # If exact match fails, try LLM semantic check for consistency with optimized strategy
        # Use raw_answer for semantic check as it contains the full reasoning and answer
        if nor_raw_answer and self.synthetic_checker:
            question = sample.get('question_text', '') if sample else ''
            is_semantic_correct, semantic_gt = self._check_answer_with_semantic(question, nor_raw_answer, ground_truths, 'nor_qa')
            
            # Check if LLM call failed during semantic checking
            if is_semantic_correct is None and semantic_gt == "__LLM_FAILED__":
                self.logger.warning(f"LLM call failed during NOR semantic check - sample needs to be discarded")
                return None, "__LLM_FAILED__"
            
            if is_semantic_correct:
                return 'A', "NOR system answered correctly (LLM semantic match)"
        
        # Check ONER system 
        oner_result = system_results.get('oner_qa', {})
        oner_answer = oner_result.get('answer', '')
        oner_raw_answer = oner_result.get('raw_answer', '')
        
        # Try exact match first
        is_correct, matched_gt = self._evaluate_answer(oner_answer, ground_truths, model_name)
        if is_correct:
            return 'B', "ONER system answered correctly (exact match)"
        
        # If exact match fails, try LLM semantic check for consistency with optimized strategy
        # Use raw_answer for semantic check as it contains the full reasoning and answer
        if oner_raw_answer and self.synthetic_checker:
            question = sample.get('question_text', '') if sample else ''
            is_semantic_correct, semantic_gt = self._check_answer_with_semantic(question, oner_raw_answer, ground_truths, 'oner_qa')
            
            # Check if LLM call failed during semantic checking
            if is_semantic_correct is None and semantic_gt == "__LLM_FAILED__":
                self.logger.warning(f"LLM call failed during ONER semantic check - sample needs to be discarded")
                return None, "__LLM_FAILED__"
            
            if is_semantic_correct:
                return 'B', "ONER system answered correctly (LLM semantic match)"
        
        # Check IRCOT system
        ircot_result = system_results.get('ircot_qa', {})
        ircot_answer = ircot_result.get('answer', '')
        ircot_raw_answer = ircot_result.get('raw_answer', '')
        
        # Try exact match first
        is_correct, matched_gt = self._evaluate_answer(ircot_answer, ground_truths, model_name)
        if is_correct:
            return 'C', "IRCOT system answered correctly (exact match)"
        
        # If exact match fails, try LLM semantic check for consistency with optimized strategy
        # Use raw_answer for semantic check as it contains the full reasoning and answer
        if ircot_raw_answer and self.synthetic_checker:
            question = sample.get('question_text', '') if sample else ''
            is_semantic_correct, semantic_gt = self._check_answer_with_semantic(question, ircot_raw_answer, ground_truths, 'ircot_qa')
            
            # Check if LLM call failed during semantic checking
            if is_semantic_correct is None and semantic_gt == "__LLM_FAILED__":
                self.logger.warning(f"LLM call failed during IRCOT semantic check - sample needs to be discarded")
                return None, "__LLM_FAILED__"
            
            if is_semantic_correct:
                return 'C', "IRCOT system answered correctly (LLM semantic match)"
        
        # All systems failed - use dataset bias (binary labeling fallback)
        dataset_key = dataset_name.lower()
        bias_label = self.dataset_bias_labels[dataset_key]
        return bias_label, f"All systems failed, using dataset bias: {dataset_name} -> {bias_label}"

    def _run_system(self, sample: Dict[str, Any], system_name: str, model_name: str) -> Dict[str, Any]:
        """
        Run a specific QA system on a sample
        
        Args:
            sample: Sample data with question and context
            system_name: Name of the system to run ('nor_qa', 'oner_qa', 'ircot_qa')
            model_name: Name of the model to use
            
        Returns:
            Dictionary with system response
        """
        try:
            if system_name == 'nor_qa':
                # NOR = No Retrieval, just LLM
                result = self._run_nor_system(sample, model_name)
            elif system_name == 'oner_qa':
                # ONER = One Retrieval step + LLM
                result = self._run_oner_system(sample, model_name)
            elif system_name == 'ircot_qa':
                # IRCOT = Iterative Retrieval + LLM
                result = self._run_ircot_system(sample, model_name)
            else:
                raise ValueError(f"Unknown system: {system_name}")
            
            self.logger.debug(f"System {system_name} response: {result}")
            return result
            
        except Exception as e:
            self.logger.error(f"Error running system {system_name}: {str(e)}")
            return {
                'answer': '',
                'system': system_name,
                'error': str(e),
                'success': False
            }

    def _construct_prompt(self, sample: Dict[str, Any], system_name: str) -> str:
        """
        Construct prompt for a specific system
        
        Args:
            sample: Sample data
            system_name: Name of the system
            
        Returns:
            Formatted prompt string
        """
        question = sample.get('question_text', '')
        context = sample.get('context', '')
        
        if system_name == 'nor_qa':
            # No retrieval QA - just question and context
            prompt = f"Question: {question}\nContext: {context}\nAnswer is:"
        elif system_name == 'oner_qa':
            # One-step retrieval QA
            prompt = f"Question: {question}\nContext: {context}\nProvide a detailed answer based on the context:\nAnswer is:"
        elif system_name == 'ircot_qa':
            # Iterative retrieval with chain of thought
            prompt = f"Question: {question}\nContext: {context}\nThink step by step and provide a comprehensive answer:\nAnswer is:"
        else:
            # Default prompt
            prompt = f"Question: {question}\nContext: {context}\nAnswer is:"
        
        return prompt

    def _run_nor_system(self, sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Run NOR system: No Retrieval, just LLM with model-specific prompting
        """
        question = sample.get('question_text', '') 
        sample_id = self._get_sample_id(sample)
        ground_truths = self._extract_ground_truth(sample)
        
        # Create model-specific prompt
        prompt = self._create_model_specific_prompt(question, model_name)
        
        request_data = {
            "prompt": prompt,
            "max_tokens": 100,
            "temperature": 0.0,
            "do_sample": True
        }
        
        start_time = time.time()
        response = self.server_manager.process_request(request_data, model_name)
        latency = time.time() - start_time
        
        # Check for resource exhausted error
        if response.get('resource_exhausted', False) or response.get('answer', '').strip() == '__RESOURCE_EXHAUSTED__':
            self.logger.warning(f"Resource exhausted error for sample {sample_id} in NOR system - returning special discard result")
            return {
                'answer': '__RESOURCE_EXHAUSTED__',
                'raw_answer': '__RESOURCE_EXHAUSTED__',
                'system': 'nor_qa',
                'server_id': response.get('server_id'),
                'processing_time': latency,
                'success': False,
                'resource_exhausted': True,
                'steps': 0,
                'error': 'Resource exhausted - sample discarded'
            }
        
        # Check for server unavailable error
        if response.get('server_unavailable', False) or response.get('answer', '').strip() == '__SERVER_UNAVAILABLE__':
            error_detail = response.get('error', 'Server unavailable')
            self.logger.warning(f"Server unavailable error for sample {sample_id} in NOR system - returning special discard result: {error_detail}")
            return {
                'answer': '__SERVER_UNAVAILABLE__',
                'raw_answer': '__SERVER_UNAVAILABLE__',
                'system': 'nor_qa',
                'server_id': response.get('server_id'),
                'processing_time': latency,
                'success': False,
                'server_unavailable': True,
                'steps': 0,
                'error': f'Server unavailable - sample discarded: {error_detail}'
            }
        
        raw_answer = response.get('answer', '').strip()
        
        # Retry once if we got an empty raw answer (additional safety net)
        if not raw_answer and response.get('retry_count', 0) == 0:
            self.logger.warning(f"Empty raw_answer received for NOR system, retrying once...")
            time.sleep(0.5)  # Brief delay
            retry_start = time.time()
            retry_response = self.server_manager.process_request(request_data, model_name)
            retry_latency = time.time() - retry_start
            retry_raw_answer = retry_response.get('answer', '').strip()
            
            # Use retry result if it's better
            if retry_raw_answer:
                response = retry_response
                latency = retry_latency
                raw_answer = retry_raw_answer
                self.logger.info(f"NOR retry successful: got non-empty response")
            else:
                self.logger.warning(f"NOR retry also returned empty response")
        
        # Extract answer using model-specific method
        answer = self._extract_model_specific_answer(raw_answer, model_name)
        
        # Evaluate answer correctness
        is_correct, matched_gt = self._evaluate_answer(answer, ground_truths, model_name)
        
        # Log Q&A interaction with ground truth
        if hasattr(self.logger, 'log_qa_interaction'):
            self.logger.log_qa_interaction(
                server_id=response.get('server_id', 'unknown'),
                question=question,
                answer=answer,
                system_type='nor_qa',
                dataset_name=getattr(self, 'current_dataset', 'unknown'),
                sample_id=sample_id,
                latency=latency,
                request_data=request_data,
                ground_truth=matched_gt if matched_gt else ground_truths[0] if ground_truths else "",
                is_correct=is_correct,
                # Add model-specific metadata
                model_name=model_name,
                is_qwen_model=self._is_qwen_model(model_name),
                raw_answer=raw_answer,  # Store raw answer for debugging
                # sample_label determined later by labeling logic
            )
        
        return {
            'answer': answer,
            'raw_answer': raw_answer,  # Store both for debugging
            'system': 'nor_qa',
            'server_id': response.get('server_id'),
            'processing_time': response.get('processing_time', latency),
            'success': True,
            'steps': 0  # NOR: 0 steps (no retrieval-generate cycle)
        }

    def _run_oner_system(self, sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Run ONER system: One Retrieval step + LLM with model-specific prompting
        """
        question = sample.get('question_text', '') 
        sample_id = self._get_sample_id(sample)
        ground_truths = self._extract_ground_truth(sample)
        
        # Step 1: Retrieve documents (15 documents for ONER - matches Adaptive-RAG paper for T5 models)
        retrieved_docs = self._retrieve_documents(question, max_docs=self.oner_max_docs, system_type="oner")
        
        # Step 2: Create context from retrieved documents
        context = self._format_retrieved_docs(retrieved_docs)
        
        # Step 3: Create model-specific prompt with context
        prompt = self._create_model_specific_prompt(question, model_name, context)
        
        request_data = {
            "prompt": prompt,
            "max_tokens": 500,
            "temperature": 0.0,
            "do_sample": True
        }
        
        start_time = time.time()
        response = self.server_manager.process_request(request_data, model_name)
        latency = time.time() - start_time
        
        # Check for resource exhausted error
        if response.get('resource_exhausted', False) or response.get('answer', '').strip() == '__RESOURCE_EXHAUSTED__':
            self.logger.warning(f"Resource exhausted error for sample {sample_id} in ONER system - returning special discard result")
            return {
                'answer': '__RESOURCE_EXHAUSTED__',
                'raw_answer': '__RESOURCE_EXHAUSTED__',
                'system': 'oner_qa',
                'server_id': response.get('server_id'),
                'processing_time': latency,
                'success': False,
                'resource_exhausted': True,
                'retrieved_docs': len(retrieved_docs),
                'steps': 1,
                'error': 'Resource exhausted - sample discarded'
            }
        
        raw_answer = response.get('answer', '').strip()
        
        # Retry once if we got an empty raw answer (additional safety net)
        if not raw_answer and response.get('retry_count', 0) == 0:
            self.logger.warning(f"Empty raw_answer received for ONER system, retrying once...")
            time.sleep(0.5)  # Brief delay
            retry_start = time.time()
            retry_response = self.server_manager.process_request(request_data, model_name)
            retry_latency = time.time() - retry_start
            retry_raw_answer = retry_response.get('answer', '').strip()
            
            # Use retry result if it's better
            if retry_raw_answer:
                response = retry_response
                latency = retry_latency
                raw_answer = retry_raw_answer
                self.logger.info(f"ONER retry successful: got non-empty response")
            else:
                self.logger.warning(f"ONER retry also returned empty response")
        
        # Extract answer using model-specific method
        answer = self._extract_model_specific_answer(raw_answer, model_name)
        
        # Evaluate answer correctness
        is_correct, matched_gt = self._evaluate_answer(answer, ground_truths, model_name)
        
        # Log Q&A interaction with ground truth
        if hasattr(self.logger, 'log_qa_interaction'):
            self.logger.log_qa_interaction(
                server_id=response.get('server_id', 'unknown'),
                question=question,
                answer=answer,
                system_type='oner_qa',
                dataset_name=getattr(self, 'current_dataset', 'unknown'),
                sample_id=sample_id,
                latency=latency,
                request_data=request_data,
                ground_truth=matched_gt if matched_gt else ground_truths[0] if ground_truths else "",
                is_correct=is_correct,
                retrieved_documents=retrieved_docs,
                # Add model-specific metadata
                model_name=model_name,
                is_qwen_model=self._is_qwen_model(model_name),
                raw_answer=raw_answer  # Store raw answer for debugging
            )
        
        return {
            'answer': answer,
            'raw_answer': raw_answer,  # Store both for debugging
            'system': 'oner_qa',
            'server_id': response.get('server_id'),
            'processing_time': response.get('processing_time', latency),
            'success': True,
            'retrieved_docs': len(retrieved_docs),
            'steps': 1  # ONER: 1 step (one retrieval-generate cycle)
        }

    def _run_ircot_system(self, sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Run IRCOT össystem using benchmark-consistent implementation.
        
        This method now uses the IRCoT Bridge Adapter which ensures consistency between
        labeling and benchmark evaluation by using the same IRCoT implementation
        from commaqa.inference that is used in the benchmark runs.
        
        Args:
            sample: Sample data containing question_text and other fields
            model_name: Name of the model (e.g., "flan-t5-xl", "gemini", "qwen")
            
        Returns:
            Dictionary containing answer, correctness, and metadata in labeling format
        """
        start_time = time.time()
        question = sample.get('question_text', '')
        sample_id = self._get_sample_id(sample)
        ground_truths = self._extract_ground_truth(sample)
        dataset_name = getattr(self, 'current_dataset', 'hotpotqa')
        
        self.logger.debug(f"Starting benchmark IRCoT for sample {sample_id}: {question[:100]}...")
        
        try:
            if self.use_benchmark_ircot:
                # DEBUG: Log worker and question mapping (similar to parallel version)
                import threading
                worker_id = f"worker_{threading.current_thread().ident}"
                self.logger.info(f"🎯 DEBUG: {worker_id} processing question {sample_id}")
                self.logger.info(f"📝 DEBUG: Question text: {question[:100]}...")
                
                # Create fresh IRCoT adapter to prevent contamination between queries
                fresh_adapter = self._get_fresh_ircot_adapter(sample_id)
                if fresh_adapter:
                    # Use robust IRCoT implementation via fresh bridge adapter
                    adapter_instance_id = id(fresh_adapter)
                    self.logger.info(f"🚀 Using robust IRCOT implementation via fresh bridge adapter (instance ID: {adapter_instance_id})")
                    self.logger.info(f"🧠 DEBUG: {worker_id} using adapter {adapter_instance_id} for question {sample_id}")
                    result = fresh_adapter.run_ircot_system(
                        sample=sample,
                        model_name=model_name,
                        max_docs=self.ircot_max_docs,
                        dataset_name=dataset_name
                    )
                    
                    # DEBUG: Check for contamination keywords (similar to parallel version)
                    answer = result.get('answer', '')
                    contamination_keywords = ['Emmanuelle Seigner', 'Roman Polanski', 'Am I Wrong', 'Étienne de Crécy']
                    for keyword in contamination_keywords:
                        if keyword in answer:
                            self.logger.error(f"🚨 CONTAMINATION DETECTED: got contaminated answer for {sample_id}")
                            self.logger.error(f"🚨 Question: {question[:100]}...")
                            self.logger.error(f"🚨 Contaminated answer contains: {keyword}")
                            self.logger.error(f"🚨 Adapter instance: {adapter_instance_id}")
                            break
                    
                    # Clean up the fresh adapter instance
                    del fresh_adapter
                else:
                    # Fresh adapter creation failed, fall back to custom implementation
                    self.logger.warning("Fresh IRCoT adapter creation failed, falling back to custom implementation")
                    return self._run_custom_ircot_system(sample, model_name)
                
                # Convert bridge adapter result to labeling format
                answer = result['answer']
                is_correct = result['correct']
                processing_time = result['latency']
                steps = result['steps']
                retriever_calls = result['retriever_calls']
                reasoning_steps = result.get('generated_steps', [])
                
                # Determine matched ground truth
                matched_gt = ""
                if is_correct and ground_truths:
                    from scaled_silver_labeling.utils.common import normalize_answer
                    normalized_answer = normalize_answer(answer)
                    for gt in ground_truths:
                        if normalize_answer(gt) == normalized_answer:
                            matched_gt = gt
                            break
                    if not matched_gt:
                        matched_gt = ground_truths[0]  # Fallback
                
                self.logger.debug(f"Benchmark IRCoT completed: {retriever_calls} retrieval calls, "
                                f"{steps} reasoning steps, answer: {answer}")
                
                # Find the last actual answer step (not EOQ)
                last_answer_step = answer
                if reasoning_steps:
                    # Look for the last "A:" step that contains actual content (not just documents or EOQ)
                    for step in reversed(reasoning_steps):
                        if step.startswith('A:') and not step.startswith('A: [') and step != 'A:':
                            last_answer_step = step
                            break
                
                # Log Q&A interaction with ground truth
                if hasattr(self.logger, 'log_qa_interaction'):
                    self.logger.log_qa_interaction(
                        server_id='benchmark_ircot',
                        question=question,
                        answer=answer,
                        system_type='ircot_qa',
                        dataset_name=getattr(self, 'current_dataset', 'unknown'),
                        sample_id=sample_id,
                        latency=processing_time,
                        request_data={'system': 'benchmark_ircot', 'max_docs': self.ircot_max_docs, 'steps': steps},
                        ground_truth=matched_gt if matched_gt else ground_truths[0] if ground_truths else "",
                        is_correct=is_correct,
                        retrieved_documents=[],  # IRCoT Bridge Adapter doesn't provide raw docs
                        # Add IRCOT-specific metadata
                        model_name=model_name,
                        is_qwen_model=self._is_qwen_model(model_name),
                        raw_answer=last_answer_step,
                        steps=steps,
                        retriever_calls=retriever_calls,
                        reasoning_steps=reasoning_steps,
                        benchmark_consistent=True
                    )
                
                return {
                    'answer': answer,
                    'raw_answer': last_answer_step,
                    'system': 'ircot_qa',
                    'server_id': 'benchmark_ircot',
                    'processing_time': processing_time,
                    'success': True,
                    'retrieved_docs': retriever_calls * self.ircot_max_docs,  # Estimate
                    'steps': steps,
                    'retriever_calls': retriever_calls,
                    'reasoning_steps': reasoning_steps,
                    'iterations_used': steps,
                    'benchmark_consistent': True  # Flag to indicate this used benchmark implementation
                }
                
            else:
                # Fallback to custom implementation if bridge adapter fails
                self.logger.warning("IRCoT Bridge Adapter not available, falling back to custom implementation")
                return self._run_custom_ircot_system(sample, model_name)
                
        except Exception as e:
            self.logger.error(f"Benchmark IRCoT failed: {e}, falling back to custom implementation")
            return self._run_custom_ircot_system(sample, model_name)
    
    def _run_custom_ircot_system(self, sample: Dict[str, Any], model_name: str) -> Dict[str, Any]:
        """
        Fallback custom IRCOT implementation (original labeling approach).
        
        This is kept as a backup in case the benchmark IRCoT adapter fails.
        It implements a simplified version of the IRCOT algorithm.
        """
        question = sample.get('question_text', '')
        sample_id = self._get_sample_id(sample)
        ground_truths = self._extract_ground_truth(sample)
        
        # Simplified IRCOT with reduced complexity
        max_iterations = 3  # Reduced for fallback
        max_tokens_per_step = 200 if self._is_flan_t5_model(model_name) else 400
        retriever_calls = 0
        all_retrieved_docs = []
        reasoning_steps = []
        total_latency = 0.0
        
        self.logger.debug(f"Starting fallback IRCoT for sample {sample_id}")
        
        # Initial retrieval
        retrieved_docs = self._retrieve_documents(question, max_docs=self.ircot_max_docs, system_type="ircot")
        all_retrieved_docs.extend(retrieved_docs)
        retriever_calls += 1
        
        # True IRCOT loop: Interleave Reason and Retrieve steps
        max_total_docs = 15  # As per IRCOT paper
        answer_found = False
        
        for iteration in range(max_iterations):
            self.logger.info(f"IRCOT Iteration {iteration + 1}/{max_iterations}")
            
            # STEP 1: REASON - Generate next CoT sentence
            context = self._format_retrieved_docs(all_retrieved_docs[:max_total_docs])
            
            # Create IRCOT-style prompt (intermediate reasoning, not final answer)
            previous_reasoning = " ".join(reasoning_steps)
            prompt = self._create_ircot_reasoning_prompt(question, context, previous_reasoning, model_name)
            
            request_data = {
                "prompt": prompt,
                "max_tokens": max_tokens_per_step,
                "temperature": 0.0,
                "do_sample": True
            }
            
            start_time = time.time()
            response = self.server_manager.process_request(request_data, model_name)
            latency = time.time() - start_time
            
            # Check for resource exhausted error
            if response.get('resource_exhausted', False) or response.get('answer', '').strip() == '__RESOURCE_EXHAUSTED__':
                self.logger.warning(f"Resource exhausted error for sample {sample_id} in IRCOT system - returning special discard result")
                return {
                    'answer': '__RESOURCE_EXHAUSTED__',
                    'raw_answer': '__RESOURCE_EXHAUSTED__',
                    'system': 'ircot_qa',
                    'server_id': response.get('server_id'),
                    'processing_time': total_latency + latency,
                    'success': False,
                    'resource_exhausted': True,
                    'iterations': iteration + 1,
                    'retrieved_docs': len(all_retrieved_docs),
                    'retriever_calls': retriever_calls,
                    'steps': iteration + 1,
                    'error': 'Resource exhausted - sample discarded'
                }
            
            # Check for server unavailable error
            if response.get('server_unavailable', False) or response.get('answer', '').strip() == '__SERVER_UNAVAILABLE__':
                error_detail = response.get('error', 'Server unavailable')
                self.logger.warning(f"Server unavailable error for sample {sample_id} in IRCOT system - returning special discard result: {error_detail}")
                return {
                    'answer': '__SERVER_UNAVAILABLE__',
                    'raw_answer': '__SERVER_UNAVAILABLE__',
                    'system': 'ircot_qa',
                    'server_id': response.get('server_id'),
                    'processing_time': total_latency + latency,
                    'success': False,
                    'server_unavailable': True,
                    'iterations': iteration + 1,
                    'retrieved_docs': len(all_retrieved_docs),
                    'retriever_calls': retriever_calls,
                    'steps': iteration + 1,
                    'error': f'Server unavailable - sample discarded: {error_detail}'
                }
            
            total_latency += latency
            
            raw_reasoning = response.get('answer', '').strip()
            if raw_reasoning:
                # Take only first sentence as per IRCOT paper
                first_sentence = raw_reasoning.split('.')[0] + '.'
                reasoning_steps.append(first_sentence)
                
                self.logger.info(f"  Reasoning Step: {first_sentence}")
                
                # Check for final answer (termination condition)
                import re
                if re.search(r'(?:so the answer is:?|answer is:?)', first_sentence, re.IGNORECASE):
                    answer_found = True
                    self.logger.info(f"  Final answer detected!")
                
                # STEP 2: RETRIEVE - Extract query from reasoning and get more docs
                if len(all_retrieved_docs) < max_total_docs and not answer_found:
                    # Extract query from the reasoning step
                    retrieval_query = self._extract_query_from_reasoning(first_sentence, question)
                    
                    if retrieval_query:
                        self.logger.info(f"  Query extracted: '{retrieval_query}'")
                        
                        # Retrieve more documents using reasoning-based query
                        additional_docs = self._retrieve_documents(
                            retrieval_query, 
                            max_docs=min(self.ircot_max_docs, max_total_docs - len(all_retrieved_docs)), 
                            system_type="ircot"
                        )
                        
                        if additional_docs:
                            all_retrieved_docs.extend(additional_docs)
                            retriever_calls += 1
                            self.logger.info(f"  Retrieved {len(additional_docs)} more docs (total: {len(all_retrieved_docs)})")
                
                # Terminate if we found answer or hit doc limit
                if answer_found:
                    self.logger.info(f"  IRCOT terminating: Answer found")
                    break
            else:
                # Empty response, continue to next iteration
                reasoning_steps.append("")
        
        # Extract final answer with improved logic for IRCOT
        if reasoning_steps:
            # Find the last actual reasoning step (not empty)
            last_step = ""
            for step in reversed(reasoning_steps):
                if step.strip():
                    last_step = step
                    break
            
            answer = self._extract_model_specific_answer(last_step, model_name)
            
            # If answer is too long or unclear, try extracting from complete reasoning
            if len(answer) > 100 or answer == last_step:
                complete_reasoning = ' '.join(reasoning_steps)
                answer = self._extract_ircot_final_answer(complete_reasoning, model_name)
            else:
                complete_reasoning = ' '.join(reasoning_steps)
        else:
            # No reasoning steps
            complete_reasoning = ""
            last_step = ""
            answer = ""
        
        is_correct, matched_gt = self._evaluate_answer(answer, ground_truths, model_name)
        
        # Log Q&A interaction with ground truth
        if hasattr(self.logger, 'log_qa_interaction'):
            self.logger.log_qa_interaction(
                server_id=response.get('server_id', 'unknown'),
                question=question,
                answer=answer,
                system_type='ircot_qa',
                dataset_name=getattr(self, 'current_dataset', 'unknown'),
                sample_id=sample_id,
                latency=total_latency,
                request_data={'system': 'custom_ircot', 'max_docs': self.ircot_max_docs, 'max_iterations': max_iterations},
                ground_truth=matched_gt if matched_gt else ground_truths[0] if ground_truths else "",
                is_correct=is_correct,
                retrieved_documents=all_retrieved_docs if len(all_retrieved_docs) <= 10 else all_retrieved_docs[:10],  # Limit docs in logs
                # Add IRCOT-specific metadata
                model_name=model_name,
                is_qwen_model=self._is_qwen_model(model_name),
                raw_answer=last_step if reasoning_steps else "",
                steps=len(reasoning_steps),
                retriever_calls=retriever_calls,
                reasoning_steps=reasoning_steps,
                benchmark_consistent=False
            )
        
        return {
            'answer': answer,
            'raw_answer': last_step if reasoning_steps else "",
            'system': 'ircot_qa',
            'server_id': response.get('server_id', 'unknown'),
            'processing_time': total_latency,
            'success': True,
            'retrieved_docs': len(all_retrieved_docs),
            'steps': len(reasoning_steps),
            'retriever_calls': retriever_calls,
            'reasoning_steps': reasoning_steps,
            'iterations_used': len(reasoning_steps),
            'benchmark_consistent': False  # Flag to indicate this used custom implementation
        }

    def _extract_retrieval_query_from_reasoning(self, reasoning_text: str, original_question: str) -> str:
        """
        Extract key entities or concepts from reasoning text to use as a query for next retrieval.
        
        This method identifies important entities, concepts, or questions raised in the reasoning
        that could benefit from additional document retrieval.
        
        Args:
            reasoning_text: The latest reasoning step generated by the model
            original_question: The original question being answered
            
        Returns:
            A query string for the next retrieval, or empty string if no good query found
        """
        if not reasoning_text or not reasoning_text.strip():
            return ""
            
        reasoning_text = reasoning_text.strip()
        
        # Strategy 1: Look for questions raised in the reasoning
        question_patterns = [
            r'(?:what|who|when|where|why|how|which)[^?]*\?',
            r'(?:need to (?:know|find|understand))[^.]*',
            r'(?:unclear|unsure|uncertain)[^.]*about[^.]*',
            r'(?:more information)[^.]*(?:about|on)[^.]*'
        ]
        
        import re
        for pattern in question_patterns:
            matches = re.findall(pattern, reasoning_text.lower())
            if matches:
                # Clean up the match and use as query
                query = matches[0].strip().rstrip('?.,')
                if len(query) > 10 and len(query) < 100:  # Reasonable length
                    return query
        
        # Strategy 2: Extract mentioned entities/proper nouns
        # Look for capitalized words that might be entities
        entity_pattern = r'\b[A-Z][a-zA-Z]{2,}(?:\s+[A-Z][a-zA-Z]*)*\b'
        entities = re.findall(entity_pattern, reasoning_text)
        
        # Filter out common words and keep meaningful entities
        common_words = {'The', 'This', 'That', 'These', 'Those', 'But', 'And', 'Or', 'So', 'However', 'Therefore', 'Thus', 'Hence'}
        meaningful_entities = [e for e in entities if e not in common_words and len(e) > 2]
        
        if meaningful_entities:
            # Use the first meaningful entity with some context
            entity = meaningful_entities[0]
            # Try to add some context words around the entity
            entity_context_pattern = r'\b\w+\s+' + re.escape(entity) + r'\s+\w+\b'
            context_match = re.search(entity_context_pattern, reasoning_text)
            if context_match:
                return context_match.group(0)
            else:
                return entity
        
        # Strategy 3: Extract key phrases that suggest need for more information
        key_phrases = [
            r'(?:about|regarding|concerning)\s+([^.]{5,30})',
            r'(?:related to|connected to|associated with)\s+([^.]{5,30})',
            r'(?:details about|information on|facts about)\s+([^.]{5,30})'
        ]
        
        for pattern in key_phrases:
            match = re.search(pattern, reasoning_text.lower())
            if match:
                phrase = match.group(1).strip()
                if len(phrase) > 5 and len(phrase) < 50:
                    return phrase
        
        # Strategy 4: Extract the most informative sentence
        sentences = re.split(r'[.!?]+', reasoning_text)
        for sentence in sentences:
            sentence = sentence.strip()
            # Look for sentences with question words or uncertainty indicators
            if any(word in sentence.lower() for word in ['what', 'who', 'when', 'where', 'why', 'how', 'need', 'unclear', 'more']):
                if 20 < len(sentence) < 80:  # Reasonable length for a query
                    return sentence
        
        # Fallback: return empty string if no good query found
        return ""

    def _get_corpus_name(self, dataset_name: str = None) -> str:
        """
        Get the appropriate corpus name for the dataset
        """
        if dataset_name:
            # Use dataset-specific corpus when available
            corpus_mapping = {
                'hotpotqa': 'hotpotqa',
                '2wikimultihopqa': '2wikimultihopqa', 
                'musique': 'musique',
                'squad': 'wiki',  # Single-hop datasets use wiki corpus
                'nq': 'wiki',
                'trivia': 'wiki'
            }
            return corpus_mapping.get(dataset_name.lower(), 'wiki')
        else:
            # Default to wiki corpus
            return 'wiki'

    def _retrieve_documents(self, question: str, max_docs: int = 15, system_type: str = "oner") -> List[Dict[str, Any]]:
        """
        Retrieve documents using the retriever server
        
        Args:
            question: The question to retrieve documents for
            max_docs: Maximum number of documents to retrieve
            system_type: Type of system ("oner", "ircot") to determine optimal retrieval count
            
        Returns:
            List of retrieved documents
            
        Note:
            Based on Adaptive-RAG paper findings for T5 models:
            - ONER (Single-step): 15 documents (optimal for single-hop questions)
            - IRCOT (Multi-step): 6 documents (optimal for multi-hop reasoning)
            - T5 encoder-decoder architecture can handle more documents efficiently
        """
        try:

            # Get appropriate corpus name for current dataset
            corpus_name = self._get_corpus_name(getattr(self, 'current_dataset', None))
            
            # Call retriever server
            retriever_url = "http://localhost:8000/retrieve"
            payload = {
                "retrieval_method": "retrieve_from_elasticsearch",
                "query_text": question,
                "max_hits_count": max_docs,
                "corpus_name": corpus_name
            }
            
            response = requests.post(retriever_url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                retrieval = result.get('retrieval', [])
                self.logger.debug(f"Retrieved {len(retrieval)} documents for question: {question[:50]}... (system: {system_type}, max_docs: {max_docs})")
                return retrieval
            else:
                self.logger.error(f"Retriever server error: {response.status_code}")
                return []
                
        except Exception as e:
            self.logger.debug(f"Error calling retriever server: {e}")  # Non-critical, system continues without retrieval
            return []

    def _create_discarded_result(self, sample: Dict[str, Any], reason: str) -> Dict[str, Any]:
        """
        Create a result entry for discarded samples (e.g., due to resource exhausted or server unavailable errors)
        
        Args:
            sample: Original sample data
            reason: Reason for discarding the sample
            
        Returns:
            Discarded result dictionary
        """
        ground_truths = self._extract_ground_truth(sample)
        
        # Determine the type of failure based on the reason
        is_resource_exhausted = 'resource exhausted' in reason.lower() or 'resource_exhausted' in reason.lower()
        is_server_unavailable = 'server unavailable' in reason.lower() or 'server_unavailable' in reason.lower()
        
        result = {
            'sample_id': self._get_sample_id(sample),
            'question': sample.get('question_text', ''),
            'ground_truth': ground_truths[0] if ground_truths else "",
            'answer': '',
            'raw_answer': '',
            'label': 'DISCARDED',
            'reasoning': f"Sample discarded: {reason}",
            'systems_used': [],
            'systems_succeeded': [],
            'discarded': True,
            'error': False,  # This is not an error, it's an intentional discard
            'timestamp': datetime.now().isoformat()
        }
        
        # Add specific failure type flags
        if is_resource_exhausted:
            result['resource_exhausted'] = True
        if is_server_unavailable:
            result['server_unavailable'] = True
            
        return result

    def _format_retrieved_docs(self, docs: List[Dict[str, Any]]) -> str:
        """
        Format retrieved documents into a context string
        
        Args:
            docs: List of retrieved document dictionaries
            
        Returns:
            Formatted context string
        """
        if not docs:
            return ""
        
        context_parts = []
        for i, doc in enumerate(docs):  # Use all retrieved documents
            title = doc.get('title', f'Document {i+1}')
            text = doc.get('text', doc.get('paragraph_text', ''))
            if text:
                # Truncate each document to prevent context overflow
                truncated_text = text[:3000] + "..." if len(text) > 3000 else text
                context_parts.append(f"[{title}] {truncated_text}")
        
        return "\n\n".join(context_parts)

    def _create_error_result(self, sample: Dict[str, Any], error_message: str) -> Dict[str, Any]:
        """
        Create a result entry for failed processing
        
        Args:
            sample: Original sample data
            error_message: Description of the error
            
        Returns:
            Error result dictionary
        """
        return {
            'sample_id': self._get_sample_id(sample),
            'question': sample.get('question', ''),
            'ground_truth': sample.get('answer', ''),
            'label': 'ERROR',
            'reasoning': f"Processing failed: {error_message}",
            'systems_used': [],
            'systems_succeeded': [],
            'error': True,
            'timestamp': datetime.now().isoformat()
        } 