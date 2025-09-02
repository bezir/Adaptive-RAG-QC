#!/usr/bin/env python3
"""
Termination Checker - Determines when to stop the IRCOT reasoning loop.

This module provides:
- Answer pattern detection
- Maximum iteration checking
- Sufficient information detection
- Cycle detection
"""

import re
import logging
from typing import Tuple, Optional
from robust_ircot.core.state import IRCoTState


class TerminationChecker:
    """
    Checks for termination conditions in the IRCOT loop.
    
    Termination conditions:
    1. Answer pattern found in reasoning
    2. Maximum iterations reached
    3. Sufficient information gathered
    4. Reasoning cycle detected
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the termination checker.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # IRCoT paper standard answer patterns ONLY
        self.answer_patterns = [
            re.compile(r"(?i)so\s+the\s+answer\s+is:?\s*(.+?)(?:\.|$)"),
            re.compile(r"(?i)the\s+answer\s+is:?\s*(.+?)(?:\.|$)"),
        ]
        
        # Insufficient information patterns
        self.insufficient_patterns = [
            re.compile(r"(?i)cannot\s+find\s+(?:sufficient\s+)?information"),
            re.compile(r"(?i)(?:not|no)\s+(?:enough|sufficient)\s+information"),
            re.compile(r"(?i)unable\s+to\s+(?:find|determine|answer)"),
            re.compile(r"(?i)information\s+is\s+(?:not\s+)?available"),
        ]
    
    def should_terminate(self, 
                        reasoning_step: str,
                        state: IRCoTState) -> Tuple[bool, Optional[str]]:
        """
        Check if the reasoning should terminate.
        
        Args:
            reasoning_step: The latest reasoning step
            state: Current IRCOT state
            
        Returns:
            Tuple of (should_terminate, extracted_answer)
        """
        # Check for answer patterns
        answer = self._check_answer_patterns(reasoning_step)
        if answer:
            self.logger.info(f"✅ Answer pattern detected: '{answer}'")
            return True, answer
        
        # Check for insufficient information
        if self._check_insufficient_info(reasoning_step):
            self.logger.info("❌ Insufficient information detected")
            return True, "I cannot find sufficient information to answer this question."
        
        # Check maximum iterations
        if state.iteration_count >= state.config.get("max_iterations", 5):
            self.logger.info(f"🛑 Maximum iterations reached ({state.iteration_count})")
            return True, None
        
        # Check for reasoning cycles
        if self._check_reasoning_cycle(state):
            self.logger.info("🔄 Reasoning cycle detected")
            return True, None
        
        # Check if we have enough information
        if self._check_sufficient_information(state):
            self.logger.info("✅ Sufficient information gathered")
            return True, None
        
        return False, None
    
    def _check_answer_patterns(self, text: str) -> Optional[str]:
        """Check if text contains an answer pattern."""
        for pattern in self.answer_patterns:
            match = pattern.search(text)
            if match:
                answer = match.group(1).strip()
                # Clean up the answer
                answer = self._clean_answer(answer)
                if answer:
                    return answer
        return None
    
    def _check_insufficient_info(self, text: str) -> bool:
        """Check if text indicates insufficient information."""
        for pattern in self.insufficient_patterns:
            if pattern.search(text):
                return True
        return False
    
    def _check_reasoning_cycle(self, state: IRCoTState) -> bool:
        """Check if reasoning is stuck in a cycle."""
        if len(state.reasoning_chain) < 3:
            return False
        
        # Check if recent steps are very similar
        recent_steps = state.get_last_reasoning_steps(3)
        
        # Filter out low-quality steps before checking cycles
        filtered_steps = []
        for step in recent_steps:
            step_clean = step.strip()
            # Skip very short steps that are likely generation errors
            if len(step_clean) > 3 and not self._is_low_quality_step(step_clean):
                filtered_steps.append(step_clean)
        
        # Need at least 2 valid steps to detect a cycle
        if len(filtered_steps) < 2:
            return False
        
        # Simple similarity check - if last 3 steps are very similar
        if len(filtered_steps) >= 3 and len(set(filtered_steps)) == 1:
            return True
        
        # Check for repeating patterns
        if len(state.reasoning_chain) >= 6:
            # Check if pattern repeats every 2 steps
            last_6 = state.reasoning_chain[-6:]
            if (last_6[0] == last_6[2] == last_6[4] and 
                last_6[1] == last_6[3] == last_6[5]):
                return True
        
        return False
    
    def _is_low_quality_step(self, step: str) -> bool:
        """Check if a reasoning step is low quality (likely generation error)."""
        step_clean = step.strip()
        
        # Check for common low-quality patterns
        low_quality_patterns = [
            r'^\d+\.$',           # Just "1." or "2." etc
            r'^[A-Za-z]\.$',      # Just "A." or "B." etc  
            r'^\w{1,2}$',         # Single character or two characters
            r'^[\.\,\;\:\!]*$',   # Just punctuation
            r'^The$',             # Just "The"
            r'^I$',               # Just "I"
            r'^Check$',           # Just "Check"
        ]
        
        import re
        for pattern in low_quality_patterns:
            if re.match(pattern, step_clean):
                return True
        
        return False
    
    def _check_sufficient_information(self, state: IRCoTState) -> bool:
        """
        Check if sufficient information has been gathered for answering the question.
        
        Uses a principled approach based on reasoning progression and information completeness.
        """
        num_docs = len(state.get_all_documents())
        num_steps = len(state.reasoning_chain)
        
        # Need at least 2 reasoning steps to have any meaningful chain
        if num_steps < 2:
            return False
        
        # Analyze the reasoning chain for completion patterns
        if num_steps >= 2:
            # Get the last few steps to analyze progression
            recent_steps = state.get_last_reasoning_steps(min(3, num_steps))
            combined_reasoning = " ".join(recent_steps).lower()
            
            # Check for logical completion indicators (not hardcoded answers)
            completion_patterns = [
                # Logical conclusion words
                "therefore", "thus", "so", "hence", "consequently",
                # Completion phrases  
                "this means", "this shows", "this indicates",
                # Information sufficiency
                "have identified", "have found", "have determined",
                "now know", "confirmed that", "established that"
            ]
            
            has_conclusion = any(pattern in combined_reasoning for pattern in completion_patterns)
            
            # Check if we're still in information gathering mode
            gathering_patterns = [
                "need to find", "need to identify", "need to determine", 
                "should find", "should identify", "must find",
                "first", "next", "then", "let me", "i will"
            ]
            
            still_gathering = any(pattern in combined_reasoning for pattern in gathering_patterns)
            
            # Check for circular reasoning (repeating the same facts)
            if num_steps >= 3:
                last_two = [step.strip().lower() for step in state.reasoning_chain[-2:]]
                if len(set(last_two)) == 1:  # Exact repetition
                    return True
            
            # If we have conclusion language and aren't still gathering, consider complete
            if has_conclusion and not still_gathering:
                self.logger.info(f"🏁 TERMINATION: Conclusion detected: '{combined_reasoning[:100]}...'")
                return True
        
        # For multi-hop questions, ensure we have sufficient evidence
        # Heuristic: if we have good document coverage and reasoning depth
        if num_docs >= 8 and num_steps >= 3:
            # Check if the reasoning chain addresses multiple aspects of the question
            full_chain = state.get_full_reasoning_chain().lower()
            
            # Count information-rich statements (not just "I need to find")
            factual_statements = 0
            for step in state.reasoning_chain:
                step_clean = step.strip().lower()
                if (len(step_clean) > 20 and  # Substantial content
                    not any(phrase in step_clean for phrase in ["need to", "should", "let me", "first"]) and
                    any(word in step_clean for word in ["is", "was", "were", "are", "born", "died", "directed", "released"])):
                    factual_statements += 1
            
            # If we have multiple factual statements and good coverage, likely sufficient
            if factual_statements >= 2:
                return True
        
        return False
    
    def _clean_answer(self, answer: str) -> str:
        """Clean extracted answer."""
        # Remove trailing punctuation
        answer = answer.rstrip(".,;!?")
        
        # Remove quotes if present
        if len(answer) > 2 and answer[0] == answer[-1] and answer[0] in ['"', "'"]:
            answer = answer[1:-1]
        
        # Remove common suffixes
        suffixes_to_remove = [
            "is the answer",
            "is my answer",
            "is the final answer",
        ]
        
        answer_lower = answer.lower()
        for suffix in suffixes_to_remove:
            if answer_lower.endswith(suffix):
                answer = answer[:-len(suffix)].strip()
        
        return answer.strip() 