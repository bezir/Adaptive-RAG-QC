#!/usr/bin/env python3
"""
Answer Parser - Extracts final answers from reasoning chains.

This module provides:
- Answer extraction from various formats
- Answer cleaning and normalization
- Fallback strategies
"""

import re
import logging
from typing import Optional, List


class AnswerParser:
    """
    Parses and extracts answers from IRCOT reasoning chains.
    
    Handles:
    - Multiple answer formats
    - Answer cleaning
    - Fallback extraction
    """
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize the answer parser.
        
        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
        
        # IRCoT paper standard answer patterns ONLY
        self.answer_patterns = [
            # "So the answer is: X" - PRIMARY IRCoT format
            re.compile(r"(?i)so\s+the\s+answer\s+is:?\s*(.+?)(?:\.|$)"),
            # "The answer is: X" - SECONDARY IRCoT format  
            re.compile(r"(?i)the\s+answer\s+is:?\s*(.+?)(?:\.|$)"),
        ]
        
        # Fallback patterns for less explicit answers
        self.fallback_patterns = [
            # "X is the answer"
            re.compile(r"(.+?)\s+is\s+the\s+answer", re.IGNORECASE),
            # "It is X"
            re.compile(r"(?i)it\s+is\s+(.+?)(?:\.|$)"),
            # "The result is X"
            re.compile(r"(?i)the\s+result\s+is\s+(.+?)(?:\.|$)"),
        ]
    
    def extract_answer(self, text: str) -> str:
        """
        Extract answer from text.
        
        Args:
            text: Text containing potential answer
            
        Returns:
            Extracted answer or empty string
        """
        if not text:
            return ""
        
        # Try primary patterns first
        answer = self._try_patterns(text, self.answer_patterns)
        if answer:
            self.logger.debug(f"Found answer with primary pattern: '{answer}'")
            return answer
        
        # Try fallback patterns
        answer = self._try_patterns(text, self.fallback_patterns)
        if answer:
            self.logger.debug(f"Found answer with fallback pattern: '{answer}'")
            return answer
        
        # Last resort: extract from final sentence
        answer = self._extract_from_final_sentence(text)
        if answer:
            self.logger.debug(f"Extracted answer from final sentence: '{answer}'")
            return answer
        
        self.logger.debug("No answer found in text")
        return ""
    
    def _try_patterns(self, text: str, patterns: List[re.Pattern]) -> Optional[str]:
        """Try multiple regex patterns to extract answer."""
        # Search in reverse order (last occurrence often most relevant)
        lines = text.split('\n')
        
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
                
            for pattern in patterns:
                match = pattern.search(line)
                if match:
                    answer = match.group(1).strip()
                    answer = self._clean_answer(answer)
                    if answer and self._is_valid_answer(answer):
                        return answer
        
        return None
    
    def _extract_from_final_sentence(self, text: str) -> Optional[str]:
        """Extract answer from the final sentence as last resort."""
        # Split into sentences
        sentences = re.split(r'[.!?]+', text.strip())
        
        # Work backwards through sentences
        for sentence in reversed(sentences):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Skip meta-statements
            skip_phrases = [
                "i need to", "i should", "let me", "i will",
                "i must", "i have to", "first", "next",
                "now i", "then i", "cannot find", "no information"
            ]
            
            sentence_lower = sentence.lower()
            if any(phrase in sentence_lower for phrase in skip_phrases):
                continue
            
            # Check if this looks like a factual statement
            if self._looks_like_answer(sentence):
                # Clean and return
                answer = self._clean_answer(sentence)
                if answer and self._is_valid_answer(answer):
                    return answer
        
        return None
    
    def _looks_like_answer(self, text: str) -> bool:
        """Check if text looks like an answer."""
        text_lower = text.lower()
        
        # Positive indicators
        positive_indicators = [
            # Dates/years
            re.compile(r'\b(19|20)\d{2}\b'),
            # Names (capitalized words)
            re.compile(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b'),
            # Numbers
            re.compile(r'\b\d+\b'),
            # Common answer words
            re.compile(r'\b(yes|no|true|false)\b', re.IGNORECASE),
        ]
        
        for pattern in positive_indicators:
            if pattern.search(text):
                return True
        
        # Check length - very short statements often are answers
        if len(text.split()) <= 5:
            return True
        
        return False
    
    def _clean_answer(self, answer: str) -> str:
        """Clean and normalize answer."""
        # Remove leading/trailing whitespace
        answer = answer.strip()
        
        # Remove trailing punctuation
        answer = answer.rstrip(".,;!?:")
        
        # Remove quotes if the entire answer is quoted
        if len(answer) > 2 and answer[0] == answer[-1] and answer[0] in ['"', "'"]:
            answer = answer[1:-1]
        
        # Remove common prefixes
        prefixes_to_remove = [
            "the answer is",
            "it is",
            "that is",
            "which is",
        ]
        
        answer_lower = answer.lower()
        for prefix in prefixes_to_remove:
            if answer_lower.startswith(prefix):
                answer = answer[len(prefix):].strip()
        
        # Remove common suffixes
        suffixes_to_remove = [
            "is the answer",
            "is correct",
            "is right",
        ]
        
        for suffix in suffixes_to_remove:
            if answer_lower.endswith(suffix):
                answer = answer[:-len(suffix)].strip()
        
        return answer
    
    def _is_valid_answer(self, answer: str) -> bool:
        """Check if answer is valid."""
        # Too short
        if len(answer) < 1:
            return False
        
        # Too long (likely not an answer)
        if len(answer) > 200:
            return False
        
        # Just punctuation or special characters
        if not any(c.isalnum() for c in answer):
            return False
        
        # Common non-answers
        non_answers = [
            "[eoq]", "q:", "a:", "none", "n/a", "null",
            "not found", "unknown", "unclear"
        ]
        
        if answer.lower() in non_answers:
            return False
        
        return True 