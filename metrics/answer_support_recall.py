"""
Answer support recall as a measure of retrieval performance.

This metric evaluates whether the retriever component of a RAG system successfully
fetches paragraphs that contain the correct answer. It's a critical diagnostic tool
for understanding retrieval failures in the Adaptive RAG pipeline.
"""
from typing import Tuple, List
import re

from metrics.metric import Metric
from metrics.squad_answer_em_f1 import normalize_answer


class AnswerSupportRecallMetric(Metric):
    """
    AnswerSupportRecall: Recall of the presence of the answer/s in the retrieved paragraphs.
    
    This metric directly measures retrieval quality by checking if the gold answer text
    appears in any of the retrieved paragraphs. It answers the question: "Did we retrieve
    the right information to potentially answer the question correctly?"
    
    Key insights this metric provides:
    - If answer support recall is low, the retriever is the bottleneck
    - If answer support recall is high but final answer accuracy is low, 
      the generator/reader is the bottleneck
    - Essential for diagnosing and improving Adaptive RAG strategies
    """

    def __init__(self) -> None:
        """Initialize counters for tracking answer support statistics."""
        self._total_count = 0  # Total number of examples evaluated
        self._total_num_retrieved_paras = 0  # Total paragraphs retrieved across all examples
        self._total_answer_support_recall = 0  # Sum of answer support recall scores

    def __call__(self, predicted_paragraph_texts: List[str], gold_answers: List[str]):
        """
        Update answer support recall statistics for one example.
        
        Args:
            predicted_paragraph_texts: List of retrieved paragraph texts
            gold_answers: List of acceptable ground truth answers
            
        The method checks if any gold answer appears in any retrieved paragraph,
        using both exact string matching and normalized matching for robustness.
        """
        answer_covered_count = 0
        
        # Check each gold answer against all retrieved paragraphs
        for gold_answer in gold_answers:
            for predicted_paragraph_text in predicted_paragraph_texts:

                def lower_clean_ws(e):
                    """Helper function to normalize text by lowercasing and cleaning whitespace."""
                    return re.sub(" +", " ", e.lower().strip())

                # Two matching strategies for robustness:
                # 1. Simple lowercase + whitespace normalization
                condition_1 = lower_clean_ws(gold_answer) in lower_clean_ws(predicted_paragraph_text)
                # 2. Full normalization (removes punctuation, articles, etc.)
                condition_2 = normalize_answer(gold_answer) in normalize_answer(predicted_paragraph_text)
                
                if condition_1 or condition_2:
                    answer_covered_count += 1
                    break  # Found this answer, move to next gold answer

        # Calculate recall: fraction of gold answers found in retrieved paragraphs
        answer_support_recall = answer_covered_count / len(gold_answers)
        
        # Update running totals
        self._total_answer_support_recall += answer_support_recall
        self._total_num_retrieved_paras += len(predicted_paragraph_texts)
        self._total_count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Compute and return average answer support recall and retrieval statistics.
        
        Args:
            reset: Whether to reset internal counters after computing metrics
            
        Returns:
            Dictionary containing:
            - answer_support_recall: Average fraction of answers found in retrieved paragraphs
            - avg_predicted_paras: Average number of paragraphs retrieved per example
            - count: Total number of examples evaluated
        """
        # Calculate averages, handling division by zero
        avg_answer_support_recall = (
            self._total_answer_support_recall / self._total_count if self._total_count > 0 else 0
        )
        avg_retrieved_paras = self._total_num_retrieved_paras / self._total_count if self._total_count > 0 else 0

        # Round for cleaner reporting
        avg_answer_support_recall = round(avg_answer_support_recall, 3)
        avg_retrieved_paras = round(avg_retrieved_paras, 3)

        if reset:
            self.reset()

        return {
            "answer_support_recall": avg_answer_support_recall,
            "avg_predicted_paras": avg_retrieved_paras,
            "count": self._total_count,
        }

    def reset(self):
        """Reset all internal counters to prepare for new evaluation."""
        self._total_count = 0
        self._total_num_retrieved_paras = 0
        self._total_answer_support_recall = 0
