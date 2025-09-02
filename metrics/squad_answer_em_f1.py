"""
Answer metric -- mostly taken directly from squad_tools of allennlp.

This module implements the standard evaluation metrics for question answering tasks,
particularly those used in the SQuAD (Stanford Question Answering Dataset) benchmark.
These metrics are fundamental for evaluating the answer generation quality in RAG systems.
"""
import re
import string
import collections
from typing import Tuple, List
import ftfy

from metrics.metric import Metric


def normalize_answer(s):
    """
    Lower text and remove punctuation, articles and extra whitespace.
    
    This normalization is crucial for fair evaluation, as it ensures that
    superficial differences in formatting don't penalize correct answers.
    For example, "The Beatles" and "beatles" should be considered equivalent.
    
    Args:
        s: Input string to normalize
        
    Returns:
        Normalized string with consistent formatting
    """

    def remove_articles(text):
        """Remove common articles (a, an, the) that don't affect answer correctness."""
        regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
        return re.sub(regex, " ", text)

    def white_space_fix(text):
        """Normalize whitespace by splitting and rejoining."""
        return " ".join(text.split())

    def remove_punc(text):
        """Remove all punctuation marks."""
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        """Convert to lowercase."""
        return text.lower()

    # Apply all normalization steps in sequence
    return white_space_fix(remove_articles(remove_punc(lower(s))))


def get_tokens(s):
    """
    Tokenize a string after normalization.
    
    Args:
        s: Input string to tokenize
        
    Returns:
        List of tokens, or empty list if input is empty
    """
    if not s:
        return []
    return normalize_answer(s).split()


def compute_exact(a_gold, a_pred):
    """
    Compute exact match score between gold and predicted answers.
    
    Args:
        a_gold: Ground truth answer
        a_pred: Predicted answer
        
    Returns:
        1 if answers match exactly (after normalization), 0 otherwise
    """
    return int(normalize_answer(a_gold) == normalize_answer(a_pred))


def compute_f1(a_gold, a_pred):
    """
    Compute F1 score between gold and predicted answers.
    
    F1 score measures the overlap between the predicted and gold answers
    at the token level. It's more lenient than exact match and gives
    partial credit for partially correct answers.
    
    Args:
        a_gold: Ground truth answer
        a_pred: Predicted answer
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    gold_toks = get_tokens(a_gold)
    pred_toks = get_tokens(a_pred)
    
    # Count common tokens between gold and predicted
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    
    # Handle edge cases
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    
    # Calculate precision, recall, and F1
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    """
    Compute the maximum metric score over multiple ground truth answers.
    
    Many questions have multiple valid answers. This function computes
    the metric against each ground truth and returns the maximum score,
    giving the model credit for matching any valid answer.
    
    Args:
        metric_fn: Function to compute metric (e.g., compute_f1)
        prediction: Single predicted answer
        ground_truths: List of valid ground truth answers
        
    Returns:
        Maximum metric score across all ground truths
    """
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


class SquadAnswerEmF1Metric(Metric):
    """
    Metric for evaluating answer quality using SQuAD-style Exact Match and F1 scores.
    
    This is one of the most important metrics for evaluating the answer generation
    component of a RAG system. It measures how well the model generates correct
    answers compared to human-annotated ground truth.
    
    Metrics computed:
    - Exact Match (EM): Percentage of predictions that match ground truth exactly
    - F1 Score: Average token-level F1 score, giving partial credit
    """
    
    def __init__(self) -> None:
        """Initialize counters for tracking answer quality statistics."""
        self._total_em = 0.0  # Sum of exact match scores
        self._total_f1 = 0.0  # Sum of F1 scores
        self._count = 0  # Number of examples evaluated

    def __call__(
        self,
        predicted_answer: str,
        ground_truth_answers: List[str],
    ):
        """
        Update answer quality statistics for one example.
        
        Args:
            predicted_answer: Single predicted answer string (or list with one element)
            ground_truth_answers: List of acceptable ground truth answers
            
        The method handles various input formats and computes both EM and F1 scores
        against the best matching ground truth answer.
        """
        # Handle input format variations
        if isinstance(predicted_answer, list): 
            predicted_answer = predicted_answer[0]
        if isinstance(ground_truth_answers[0], tuple): 
            ground_truth_answers = [i for i in ground_truth_answers[0]]
        
        # Clean up text encoding issues
        predicted_answer = ftfy.fix_text(predicted_answer)
        ground_truth_answers = [ftfy.fix_text(e) for e in ground_truth_answers]
        
        # Validate input types
        assert isinstance(predicted_answer, str)
        assert isinstance(ground_truth_answers, (Tuple, List))

        # Compute metrics against the best matching ground truth
        exact_scores = metric_max_over_ground_truths(compute_exact, predicted_answer, ground_truth_answers)
        f1_scores = metric_max_over_ground_truths(compute_f1, predicted_answer, ground_truth_answers)

        # Update running totals
        self._total_em += int(exact_scores)
        self._total_f1 += f1_scores
        self._count += 1

    def get_metric(self, reset: bool = False) -> Tuple[float, float]:
        """
        Compute and return average exact match and F1 scores.
        
        Args:
            reset: Whether to reset internal counters after computing metrics
            
        Returns:
            Dictionary containing:
            - em: Average exact match score (0.0 to 1.0)
            - f1: Average F1 score (0.0 to 1.0)  
            - count: Total number of examples evaluated
        """
        # Calculate averages, handling division by zero
        exact_match = self._total_em / self._count if self._count > 0 else 0
        f1_score = self._total_f1 / self._count if self._count > 0 else 0
        
        if reset:
            self.reset()
            
        return {
            "em": round(exact_match, 3), 
            "f1": round(f1_score, 3), 
            "count": self._count
        }

    def reset(self):
        """Reset all internal counters to prepare for new evaluation."""
        self._total_em = 0.0
        self._total_f1 = 0.0
        self._count = 0
