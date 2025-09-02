"""
Support metric -- mostly taken directly from hotpotqa

This module implements evaluation metrics for supporting evidence retrieval,
particularly designed for multi-hop reasoning tasks like HotpotQA. These metrics
evaluate whether the RAG system retrieves the correct supporting documents/paragraphs
needed to answer complex questions requiring multiple reasoning steps.
"""
from typing import Tuple, List, Dict
import ftfy
import re

from metrics.metric import Metric
from metrics.squad_answer_em_f1 import normalize_answer


def compute_metrics(predicted_support: List[str], gold_support: List[str]) -> Dict:
    """
    Compute precision, recall, F1, and exact match for supporting evidence.
    
    This function is adapted from HotpotQA evaluation and measures how well
    the model identifies the correct supporting documents/paragraphs.
    
    Args:
        predicted_support: List of predicted supporting evidence identifiers
        gold_support: List of ground truth supporting evidence identifiers
        
    Returns:
        Dictionary with precision, recall, F1, and exact match scores
        
    The evaluation treats support identification as a set matching problem,
    where we want to measure how many of the predicted supports are correct
    (precision) and how many of the required supports were found (recall).
    """
    # Normalize support identifiers by removing spaces and fixing encoding
    predicted_support = set([re.sub(r" +", "", ftfy.fix_text(str(e)).lower()) for e in predicted_support])
    gold_support = set([re.sub(r" +", "", ftfy.fix_text(str(e)).lower()) for e in gold_support])

    # Calculate true positives, false positives, and false negatives
    tp, fp, fn = 0, 0, 0
    
    # Count correct predictions (true positives) and incorrect predictions (false positives)
    for e in predicted_support:
        if e in gold_support:
            tp += 1
        else:
            fp += 1
            
    # Count missed gold supports (false negatives)
    for e in gold_support:
        if e not in predicted_support:
            fn += 1
            
    # Calculate standard metrics
    prec = 1.0 * tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = 1.0 * tp / (tp + fn) if tp + fn > 0 else 0.0
    f1 = 2 * prec * recall / (prec + recall) if prec + recall > 0 else 0.0
    em = 1.0 if fp + fn == 0 else 0.0

    # Special case: if both predicted and gold are empty, consider it perfect
    # This handles questions that don't require supporting evidence
    if not predicted_support and not gold_support:
        f1, em = 1.0, 1.0

    return {"prec": prec, "recall": recall, "f1": f1, "em": em}


class SupportEmF1Metric(Metric):
    """
    SupportMetric: EM and F1 (Similar to HotpotQA Sp metric)
    
    This metric evaluates the quality of supporting evidence retrieval in multi-hop
    reasoning tasks. It's crucial for Adaptive RAG systems that need to identify
    and retrieve multiple relevant documents to answer complex questions.
    
    The metric handles two levels of granularity:
    1. Title-level: Evaluates whether the correct document titles were identified
    2. Paragraph-level: Evaluates whether the correct specific paragraphs were identified
    
    This dual evaluation helps diagnose whether retrieval failures occur at the
    document selection level or the paragraph selection level within documents.
    """

    def __init__(self, do_normalize_answer: bool = False) -> None:
        """
        Initialize counters for tracking support evidence quality statistics.
        
        Args:
            do_normalize_answer: Whether to apply answer normalization to support identifiers
        """
        # Title-level metrics (document-level retrieval quality)
        self._titles_total_em = 0.0
        self._titles_total_f1 = 0.0
        self._titles_total_precision = 0.0
        self._titles_total_recall = 0.0

        # Paragraph-level metrics (fine-grained retrieval quality)
        self._paras_total_em = 0.0
        self._paras_total_f1 = 0.0
        self._paras_total_precision = 0.0
        self._paras_total_recall = 0.0

        # Statistics for predicted support counts
        self._total_predicted_titles = 0
        self._max_predicted_titles = -float("inf")
        self._min_predicted_titles = float("inf")

        self._total_predicted_paras = 0
        self._max_predicted_paras = -float("inf")
        self._min_predicted_paras = float("inf")

        self._do_normalize_answer = do_normalize_answer
        self._count = 0

    def __call__(self, predicted_support: List[str], gold_support: List[str]):
        """
        Update support evidence quality statistics for one example.
        
        Args:
            predicted_support: List of predicted supporting evidence identifiers
            gold_support: List of ground truth supporting evidence identifiers
            
        The method handles different support identifier formats:
        - Plain text: Direct comparison of support identifiers
        - PID format: Structured identifiers like "pid123___DocumentTitle"
        
        It evaluates both title-level and paragraph-level retrieval quality.
        """
        # Handle empty predictions
        predicted_support = predicted_support or []

        # Apply normalization if requested
        if self._do_normalize_answer:
            predicted_support = [normalize_answer(e) for e in predicted_support]
            gold_support = [normalize_answer(e) for e in gold_support]

        # Determine support identifier format and extract titles/paragraphs accordingly
        if not gold_support:
            # No gold support required - treat as simple comparison
            gold_support_titles = []
            gold_support_paras = []
            predicted_support_titles = predicted_support_paras = predicted_support

        elif gold_support[0].startswith("pid"):
            # PID format: "pid123___DocumentTitle" - extract titles and keep full IDs
            for e in gold_support + predicted_support:
                assert e.startswith("pid"), f"Expected PID format, got: {e}"
            
            # Extract document titles from PID format
            predicted_support_titles = [e.split("___")[1] for e in predicted_support]
            predicted_support_paras = predicted_support
            gold_support_titles = [e.split("___")[1] for e in gold_support]
            gold_support_paras = gold_support

        else:
            # Plain text format - use identifiers as both titles and paragraphs
            for e in gold_support + predicted_support:
                assert not e.startswith("pid"), f"Mixed formats not supported: {e}"
            
            predicted_support_titles = predicted_support_paras = predicted_support
            gold_support_titles = gold_support_paras = gold_support

        # Convert to sets for set-based evaluation
        predicted_support_titles = set(map(str, predicted_support_titles))
        predicted_support_paras = set(map(str, predicted_support_paras))
        gold_support_titles = set(map(str, gold_support_titles))
        gold_support_paras = set(map(str, gold_support_paras))

        # Compute metrics for both title and paragraph levels
        titles_metrics = compute_metrics(predicted_support_titles, gold_support_titles)
        paras_metrics = compute_metrics(predicted_support_paras, gold_support_paras)

        # Update statistics for predicted support counts
        self._total_predicted_titles += len(predicted_support_titles)
        self._max_predicted_titles = max(self._max_predicted_titles, len(predicted_support_titles))
        self._min_predicted_titles = min(self._min_predicted_titles, len(predicted_support_titles))

        self._total_predicted_paras += len(predicted_support_paras)
        self._max_predicted_paras = max(self._max_predicted_titles, len(predicted_support_paras))
        self._min_predicted_paras = min(self._min_predicted_titles, len(predicted_support_paras))

        # Update running totals for title-level metrics
        self._titles_total_em += float(titles_metrics["em"])
        self._titles_total_f1 += titles_metrics["f1"]
        self._titles_total_precision += titles_metrics["prec"]
        self._titles_total_recall += titles_metrics["recall"]

        # Update running totals for paragraph-level metrics
        self._paras_total_em += float(paras_metrics["em"])
        self._paras_total_f1 += paras_metrics["f1"]
        self._paras_total_precision += paras_metrics["prec"]
        self._paras_total_recall += paras_metrics["recall"]

        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Compute and return average support evidence metrics.
        
        Args:
            reset: Whether to reset internal counters after computing metrics
            
        Returns:
            Dictionary containing comprehensive support evidence metrics:
            
            Title-level metrics (document retrieval quality):
            - title_em: Exact match for document titles
            - title_f1: F1 score for document titles  
            - title_precision: Precision for document titles
            - title_recall: Recall for document titles
            
            Paragraph-level metrics (fine-grained retrieval quality):
            - para_em: Exact match for specific paragraphs
            - para_f1: F1 score for specific paragraphs
            - para_precision: Precision for specific paragraphs
            - para_recall: Recall for specific paragraphs
            
            Statistics:
            - avg_predicted_titles/paras: Average number of supports predicted
            - max/min_predicted_titles/paras: Range of supports predicted
            - count: Total number of examples evaluated
        """
        # Calculate title-level averages
        titles_exact_match = self._titles_total_em / self._count if self._count > 0 else 0
        titles_f1_score = self._titles_total_f1 / self._count if self._count > 0 else 0
        titles_precision_score = self._titles_total_precision / self._count if self._count > 0 else 0
        titles_recall_score = self._titles_total_recall / self._count if self._count > 0 else 0

        # Calculate paragraph-level averages
        paras_exact_match = self._paras_total_em / self._count if self._count > 0 else 0
        paras_f1_score = self._paras_total_f1 / self._count if self._count > 0 else 0
        paras_precision_score = self._paras_total_precision / self._count if self._count > 0 else 0
        paras_recall_score = self._paras_total_recall / self._count if self._count > 0 else 0

        # Calculate prediction statistics
        avg_predicted_titles = self._total_predicted_titles / self._count if self._count > 0 else 0
        avg_predicted_paras = self._total_predicted_paras / self._count if self._count > 0 else 0

        if reset:
            self.reset()

        return {
            # Title-level metrics
            "title_em": round(titles_exact_match, 3),
            "title_f1": round(titles_f1_score, 3),
            "title_precision": round(titles_precision_score, 3),
            "title_recall": round(titles_recall_score, 3),
            
            # Paragraph-level metrics
            "para_em": round(paras_exact_match, 3),
            "para_f1": round(paras_f1_score, 3),
            "para_precision": round(paras_precision_score, 3),
            "para_recall": round(paras_recall_score, 3),
            
            # Prediction statistics
            "avg_predicted_titles": avg_predicted_titles,
            "max_predicted_titles": self._max_predicted_titles,
            "min_predicted_titles": self._min_predicted_titles,
            "avg_predicted_paras": avg_predicted_paras,
            "max_predicted_paras": self._max_predicted_paras,
            "min_predicted_paras": self._min_predicted_paras,
            "count": self._count,
        }

    def reset(self):
        """Reset all internal counters to prepare for new evaluation."""
        # Reset title-level counters
        self._titles_total_em = 0.0
        self._titles_total_f1 = 0.0
        self._titles_total_precision = 0.0
        self._titles_total_recall = 0.0

        # Reset paragraph-level counters
        self._paras_total_em = 0.0
        self._paras_total_f1 = 0.0
        self._paras_total_precision = 0.0
        self._paras_total_recall = 0.0

        # Reset prediction statistics
        self._total_predicted_titles = 0
        self._max_predicted_titles = -float("inf")
        self._min_predicted_titles = float("inf")

        self._total_predicted_paras = 0
        self._max_predicted_paras = -float("inf")
        self._min_predicted_paras = float("inf")

        self._count = 0
