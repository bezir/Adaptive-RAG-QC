#!/usr/bin/env python3
"""
Prompt Builder - Constructs prompts for different stages of IRCOT reasoning.

This module handles:
- Dynamic prompt construction based on model and dataset
- Few-shot example selection
- Context formatting
- Model-specific adaptations
"""

import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
import random

from ircot.core.state import IRCoTState
from ircot.prompting.examples import IRCoTExampleBank


class PromptBuilder:
    """
    Builds prompts for different stages of the IRCOT process.
    
    Handles:
    - Initial reasoning prompts
    - Continuation prompts  
    - Final reader prompts
    - Model-specific formatting
    """
    
    def __init__(self,
                 model: str,
                 dataset: str,
                 example_bank: Optional[IRCoTExampleBank] = None,
                 logger: Optional[logging.Logger] = None,
                 **kwargs):
        """
        Initialize the prompt builder.
        
        Args:
            model: Model name (e.g., "gemini", "qwen", "flan-t5-xxl")
            dataset: Dataset name
            example_bank: Optional custom example bank
            logger: Logger instance
        """
        self.model = model.lower()
        self.dataset = dataset.lower()
        self.logger = logger or logging.getLogger(__name__)
        
        # Initialize example bank
        self.example_bank = example_bank or IRCoTExampleBank(dataset=dataset)
        
        # Model-specific settings
        self.model_config = self._get_model_config()
        
        # Prompt templates
        self.templates = self._load_templates()
    
    def _get_model_config(self) -> Dict[str, Any]:
        """Get model-specific configuration."""
        configs = {
            "gemini": {
                "max_examples": 5,
                "example_format": "detailed",
                "reasoning_style": "natural",
                "include_metadata": True,
            },
            "qwen": {
                "max_examples": 4,
                "example_format": "concise",
                "reasoning_style": "structured",
                "include_metadata": False,
            },
            "flan-t5": {
                "max_examples": 3,
                "example_format": "minimal",
                "reasoning_style": "direct",
                "include_metadata": False,
            }
        }
        
        # Determine model family
        model_lower = self.model.lower()
        if "gemini" in model_lower:
            model_family = "gemini"
        elif "qwen" in model_lower:
            model_family = "qwen"
        else:
            model_family = "flan-t5"
            
        return configs.get(model_family, configs["flan-t5"])
    
    def _load_templates(self) -> Dict[str, str]:
        """Load prompt templates."""
        return {
            "initial": {
                "gemini": self._get_gemini_initial_template(),
                "qwen": self._get_qwen_initial_template(),
                "flan-t5": self._get_flan_initial_template(),
            },
            "continuation": {
                "gemini": self._get_gemini_continuation_template(),
                "qwen": self._get_qwen_continuation_template(),
                "flan-t5": self._get_flan_continuation_template(),
            },
            "reader": {
                "gemini": self._get_gemini_reader_template(),
                "qwen": self._get_qwen_reader_template(), 
                "flan-t5": self._get_flan_reader_template(),
            }
        }
    
    def build_initial_prompt(self, state: IRCoTState) -> str:
        """
        Build the initial reasoning prompt.
        
        Args:
            state: Current IRCOT state
            
        Returns:
            Formatted prompt string
        """
        # CRITICAL FIX: Add worker isolation to prevent prompt contamination
        import copy
        
        # Get appropriate examples
        raw_examples = self.example_bank.get_examples(
            num_hops=self._estimate_hop_count(state.question),
            num_examples=self.model_config["max_examples"],
            include_unanswerable=True
        )
        
        # Deep copy examples to prevent cross-worker contamination
        examples = copy.deepcopy(raw_examples)
        
        # Format examples
        formatted_examples = self._format_examples(examples, stage="initial")
        
        # Get context
        context = state.get_context_for_prompt()
        
        # Select template
        model_family = self._get_model_family()
        template = self.templates["initial"][model_family]
        
        # Build prompt
        prompt = template.format(
            examples=formatted_examples,
            context=context,
            question=state.question
        )
        
        return prompt
    
    def build_continuation_prompt(self, state: IRCoTState) -> str:
        """
        Build a continuation prompt for ongoing reasoning.
        
        Args:
            state: Current IRCOT state
            
        Returns:
            Formatted prompt string
        """
        # Get recent context
        context = state.get_context_for_prompt()
        
        # Get reasoning so far
        reasoning_so_far = " ".join(state.get_last_reasoning_steps(3))
        
        # Select template
        model_family = self._get_model_family()
        template = self.templates["continuation"][model_family]
        
        # Build prompt
        prompt = template.format(
            context=context,
            reasoning_so_far=reasoning_so_far,
            question=state.question
        )
        
        return prompt
    
    def build_final_reader_prompt(self, state: IRCoTState) -> str:
        """
        Build the final reader prompt for answer synthesis.
        
        Args:
            state: Current IRCOT state
            
        Returns:
            Formatted prompt string
        """
        # Get full context
        context = state.get_context_for_prompt()
        
        # Get full reasoning chain
        reasoning_chain = state.get_full_reasoning_chain()
        
        # Select template
        model_family = self._get_model_family()
        template = self.templates["reader"][model_family]
        
        # Build prompt
        prompt = template.format(
            context=context,
            reasoning_chain=reasoning_chain,
            question=state.question
        )
        
        return prompt
    
    def _format_examples(self, examples: List[Dict[str, Any]], stage: str) -> str:
        """Format examples based on model requirements."""
        formatted_parts = []
        
        for i, example in enumerate(examples):
            if self.model_config["example_format"] == "detailed":
                formatted = self._format_detailed_example(example, i + 1)
            elif self.model_config["example_format"] == "concise":
                formatted = self._format_concise_example(example)
            else:  # minimal
                formatted = self._format_minimal_example(example)
                
            formatted_parts.append(formatted)
        
        return "\n\n".join(formatted_parts)
    
    def _format_detailed_example(self, example: Dict[str, Any], example_num: int) -> str:
        """Format example with full details (for Gemini)."""
        parts = []
        
        # Add metadata if enabled
        if self.model_config["include_metadata"]:
            parts.append(f'# METADATA: {{"qid": "example_{example_num}"}}')
        
        # Add context documents
        for doc in example.get("documents", []):
            parts.append(f"Wikipedia Title: {doc['title']}")
            parts.append(doc['text'])
            parts.append("")
        
        # Add question
        parts.append("Q: Answer the following question by reasoning step-by-step.")
        parts.append(example["question"])
        
        # Add reasoning steps
        parts.append("A: " + example["reasoning_steps"][0])
        
        # Add continuation steps if multi-hop
        if len(example["reasoning_steps"]) > 1:
            for step in example["reasoning_steps"][1:]:
                parts.append("")
                parts.append("Q: Continue reasoning step-by-step.")
                parts.append("A: " + step)
        
        return "\n".join(parts)
    
    def _format_concise_example(self, example: Dict[str, Any]) -> str:
        """Format example concisely (for Qwen)."""
        parts = []
        
        # Condensed context
        context_summary = self._summarize_context(example.get("documents", []))
        if context_summary:
            parts.append(f"Context: {context_summary}")
        
        # Question and answer
        parts.append(f"Question: {example['question']}")
        parts.append(f"Reasoning: {' '.join(example['reasoning_steps'])}")
        parts.append(f"Answer: {example['answer']}")
        
        return "\n".join(parts)
    
    def _format_minimal_example(self, example: Dict[str, Any]) -> str:
        """Format example minimally (for FLAN-T5)."""
        # Just question and final answer for FLAN-T5
        return f"Q: {example['question']}\nA: {example['answer']}"
    
    def _summarize_context(self, documents: List[Dict[str, str]]) -> str:
        """Create a brief summary of document titles."""
        if not documents:
            return ""
        titles = [doc["title"] for doc in documents[:3]]  # First 3 titles
        return "Documents about: " + ", ".join(titles)
    
    def _estimate_hop_count(self, question: str) -> int:
        """Estimate the number of hops needed for a question."""
        # Simple heuristic based on question complexity
        question_lower = question.lower()
        
        # Look for multi-hop indicators
        if any(phrase in question_lower for phrase in [
            "of the", "that", "which", "who was", "when was the"
        ]):
            if question_lower.count("of") >= 2 or question_lower.count("that") >= 2:
                return 3
            return 2
        
        return 1
    
    def _get_model_family(self) -> str:
        """Determine the model family."""
        if "gemini" in self.model:
            return "gemini"
        elif "qwen" in self.model:
            return "qwen"
        else:
            return "flan-t5"
    
    # Template methods for each model
    def _get_gemini_initial_template(self) -> str:
        """Get Gemini initial prompt template."""
        return """{examples}

# METADATA: {{"qid": "current_question"}}
{context}

Q: Answer this question step-by-step using the provided context: {question}

Think step-by-step:
- Can you answer the question directly from the context?
- If YES → Write "The answer is: " followed by your complete answer
- If NO → Describe what specific information you need to find next

Important: Always provide complete sentences and specific details in your reasoning.

A: """

    def _get_gemini_continuation_template(self) -> str:
        """Get Gemini continuation template."""
        return """{context}

Q: Continue reasoning step-by-step to answer this question: {question}

Current reasoning: {reasoning_so_far}

Continue your reasoning with a complete sentence:
- If you can answer now: Write "The answer is: " followed by your complete final answer
- If you need more information: Describe exactly what information you still need to find
- Build upon your previous reasoning without repeating it

Provide clear, specific reasoning in complete sentences.

A: """

    def _get_gemini_reader_template(self) -> str:
        """Get Gemini reader template."""
        return """Question: {question}

Retrieved documents:
{context}

Reasoning chain:
{reasoning_chain}

Based on the reasoning chain above, provide your final answer using the exact format below.

You must start your response with "So the answer is:" followed by your answer.

Examples:
- "So the answer is: X was released more recently than Y"
- "So the answer is: No, they are from different countries" 
- "So the answer is: John Smith"
- "So the answer is: July 15, 1985"""

    def _get_qwen_initial_template(self) -> str:
        """Get Qwen initial prompt template with enhanced multi-hop reasoning."""
        return """{examples}

# METADATA: {{"qid": "current_question"}}
{context}

Q: Answer this question step-by-step using the provided context: {question}

ANALYSIS FRAMEWORK:
1. Break down the question into sub-questions if it has multiple parts
2. For each sub-question, identify the specific entities, dates, or facts needed
3. Search the context systematically for each piece of information
4. Connect the information logically to build the complete answer

Think step-by-step:
- Is this a multi-part question? If yes, what are the individual steps I need to solve?
- What specific entities, names, dates, or facts do I need to find?
- Can I find all required information in the current context?
- If YES → Write "The answer is: " followed by your complete answer with specific details
- If NO → Write "I need to find: " followed by the exact missing information needed

Important: For multi-hop questions, trace through each logical step and provide specific entity names, dates, and details.

A: """

    def _get_qwen_continuation_template(self) -> str:
        """Get Qwen continuation template with enhanced multi-hop reasoning."""
        return """{context}

Q: Continue reasoning step-by-step to answer this question: {question}

Previous reasoning: {reasoning_so_far}

CONTINUE MULTI-HOP ANALYSIS:
1. Review what I found in my previous reasoning
2. Identify what specific information I still need
3. Search the new context for the missing pieces
4. Connect all information to form the complete answer

Continue your reasoning:
- Have I found all the entities, dates, and facts needed for each part of the question?
- Can I now trace through the complete logical chain to the final answer?
- If YES → Write "The answer is: " followed by your complete final answer with specific details
- If NO → Write "I need to find: " followed by the exact missing information needed

Important: For multi-hop questions, ensure you've followed the complete logical chain. Don't repeat previous reasoning - build upon it.

A: """

    def _get_qwen_reader_template(self) -> str:
        """Get Qwen reader template."""
        return """Question: {question}

Retrieved documents:
{context}

Reasoning chain:
{reasoning_chain}

Based on the reasoning chain above, provide your final answer using the exact format below.

You must start your response with "So the answer is:" followed by your answer.

Examples:
- "So the answer is: X was released more recently than Y"
- "So the answer is: No, they are from different countries" 
- "So the answer is: John Smith"
- "So the answer is: July 15, 1985"""

    def _get_flan_initial_template(self) -> str:
        """Get FLAN-T5 initial prompt template."""
        return """{examples}

Context: {context}
Q: {question}
A: """

    def _get_flan_continuation_template(self) -> str:
        """Get FLAN-T5 continuation template."""
        return """Context: {context}
Previous: {reasoning_so_far}
Continue: """

    def _get_flan_reader_template(self) -> str:
        """Get FLAN-T5 reader template."""
        return """Context: {context}
Question: {question}
Reasoning: {reasoning_chain}
Answer: """ 