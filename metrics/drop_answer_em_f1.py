"""
DROP (Discrete Reasoning Over Paragraphs) answer evaluation metrics.

This module implements evaluation metrics specifically designed for the DROP dataset,
which requires numerical reasoning and handling of complex answer types. Unlike SQuAD,
DROP answers can be numbers, dates, or multiple text spans, requiring specialized
evaluation logic for fair comparison in RAG systems.
"""
from typing import Tuple, List

import ftfy
from metrics.metric import Metric
from metrics.drop_eval import (
    get_metrics as drop_em_and_f1,
)
from metrics.squad_answer_em_f1 import metric_max_over_ground_truths


class DropAnswerEmAndF1(Metric):
    """
    This :class:`Metric` takes the best span string computed by a model, along with the answer
    strings labeled in the data, and computes exact match and F1 score using the official DROP
    evaluator (which has special handling for numbers and for questions with multiple answer spans,
    among other things).
    
    Key features of DROP evaluation:
    - Handles numerical answers with normalization (e.g., "2.0" == "2")
    - Supports multiple answer spans that must be aligned optimally
    - Provides precision and recall in addition to F1 and EM
    - Essential for evaluating RAG systems on numerical reasoning tasks
    
    This metric is crucial for Adaptive RAG systems that need to handle diverse
    question types requiring mathematical reasoning and multi-step inference.
    """

    def __init__(self) -> None:
        """Initialize counters for tracking DROP answer quality statistics."""
        self._total_em = 0.0      # Sum of exact match scores
        self._total_f1 = 0.0      # Sum of F1 scores  
        self._total_prec = 0.0    # Sum of precision scores
        self._total_recall = 0.0  # Sum of recall scores
        self._count = 0           # Number of examples evaluated

    def __call__(
        self,
        predicted_answer_list: List[str],
        list_of_ground_truth_answer_list: List[List[str]],
    ):
        """
        Update DROP answer quality statistics for one example.
        
        Args:
            predicted_answer_list: List of predicted answer strings (can be multiple spans)
            list_of_ground_truth_answer_list: List of lists, where each inner list contains
                                             alternative valid answers for that position
        
        The method handles the complex answer formats in DROP and uses optimal alignment
        to fairly compare predicted and ground truth answer spans.
        """
        # Validate input types
        assert isinstance(predicted_answer_list, (list, tuple))
        assert isinstance(list_of_ground_truth_answer_list, (list, tuple))

        # Handle empty predictions
        if not predicted_answer_list:
            predicted_answer_list = [""]

        # Further input validation
        assert isinstance(predicted_answer_list[0], str)
        assert isinstance(list_of_ground_truth_answer_list[0], (list, tuple))
        assert isinstance(list_of_ground_truth_answer_list[0][0], str)

        # Clean up text encoding issues in both predictions and ground truth
        predicted_answer_list = [ftfy.fix_text(e) for e in predicted_answer_list]
        list_of_ground_truth_answer_list = [
            [ftfy.fix_text(e) for e in ground_truth_answer_list]
            for ground_truth_answer_list in list_of_ground_truth_answer_list
        ]

        # Use DROP's specialized evaluation logic that handles:
        # - Number normalization and comparison
        # - Optimal alignment of multiple answer spans
        # - Special handling for different answer types (numbers, dates, spans)
        exact_match, f1_score, prec_score, recall_score = metric_max_over_ground_truths(
            drop_em_and_f1, predicted_answer_list, list_of_ground_truth_answer_list
        )

        # Update running totals
        # Converting EM to int since we want to count exact matches
        self._total_em += int(exact_match)
        self._total_f1 += f1_score
        self._total_prec += prec_score
        self._total_recall += recall_score
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Compute and return average DROP metrics.
        
        Args:
            reset: Whether to reset internal counters after computing metrics
            
        Returns:
            Dictionary containing:
            - em: Average exact match score (0.0 to 1.0)
            - f1: Average F1 score (0.0 to 1.0)
            - precision: Average precision score (0.0 to 1.0)
            - recall: Average recall score (0.0 to 1.0)
            - count: Total number of examples evaluated
        
        These metrics provide a comprehensive view of model performance on
        numerical reasoning tasks, with precision/recall helping diagnose
        whether the model is over-generating or under-generating answer spans.
        """
        # Calculate averages, handling division by zero
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        prec_score = self._total_prec / self._count if self._count > 0 else 0
        recall_score = self._total_recall / self._count if self._count > 0 else 0
        
        if reset:
            self.reset()
            
        return {
            "em": round(exact_match, 3),
            "f1": round(f1_score, 3),
            "precision": round(prec_score, 3),
            "recall": round(recall_score, 3),
            "count": self._count,
        }

    def reset(self):
        """Reset all internal counters to prepare for new evaluation."""
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._total_prec = 0.0
        self._total_recall = 0.0
        self._count = 0
