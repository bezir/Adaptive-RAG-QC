#!/usr/bin/python
"""
Official DROP (Discrete Reasoning Over Paragraphs) evaluation script.

This module implements the official evaluation logic for the DROP dataset,
which requires numerical reasoning and complex answer matching. Unlike simple
text-based QA, DROP answers can be numbers, dates, or multiple text spans,
requiring sophisticated normalization and alignment algorithms.

Key features:
- Number normalization and comparison (e.g., "2.0" == "2")
- Optimal alignment of multiple answer spans using Hungarian algorithm
- Special handling for different answer types (numbers, dates, spans)
- Robust text normalization for fair comparison

This is essential for evaluating Adaptive RAG systems on numerical reasoning tasks.
"""

from collections import defaultdict
from typing import Any, Dict, List, Set, Tuple, Union, Optional
import json
import argparse
import string
import re

import numpy as np
from scipy.optimize import linear_sum_assignment


# From here through _normalize_answer was originally copied from:
# https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/
# Then cleaned up and modified a bit.
def _remove_articles(text: str) -> str:
    """Remove articles (a, an, the) that don't affect answer correctness."""
    regex = re.compile(r"\b(a|an|the)\b", re.UNICODE)
    return re.sub(regex, " ", text)


def _white_space_fix(text: str) -> str:
    """Normalize whitespace by splitting and rejoining."""
    return " ".join(text.split())


EXCLUDE = set(string.punctuation)


def _remove_punc(text: str) -> str:
    """
    Remove punctuation from text, but preserve numbers.
    
    This is crucial for DROP evaluation since numerical answers
    should not be affected by punctuation removal.
    """
    if not _is_number(text):
        return "".join(ch for ch in text if ch not in EXCLUDE)
    else:
        return text


def _lower(text: str) -> str:
    """Convert text to lowercase."""
    return text.lower()


def _tokenize(text: str) -> List[str]:
    """Tokenize text by splitting on spaces and hyphens."""
    return re.split(" |-", text)


def _normalize_answer(text: str) -> str:
    """
    Comprehensive answer normalization for DROP evaluation.
    
    This function applies multiple normalization steps to ensure fair
    comparison between predicted and gold answers:
    1. Tokenization
    2. Lowercasing
    3. Punctuation removal (preserving numbers)
    4. Article removal
    5. Number normalization
    6. Whitespace normalization
    
    Args:
        text: Raw answer text
        
    Returns:
        Normalized answer text ready for comparison
    """
    parts = [
        _white_space_fix(_remove_articles(_normalize_number(_remove_punc(_lower(token))))) for token in _tokenize(text)
    ]
    parts = [part for part in parts if part.strip()]
    normalized = " ".join(parts).strip()
    return normalized


def _is_number(text: str) -> bool:
    """
    Check if text represents a number.
    
    This is essential for DROP's numerical reasoning evaluation,
    as numbers need special handling during normalization.
    """
    try:
        float(text)  # Could be extended to handle comma-separated numbers
        return True
    except ValueError:
        return False


def _normalize_number(text: str) -> str:
    """
    Normalize numerical representations for consistent comparison.
    
    This ensures that different representations of the same number
    (e.g., "2", "2.0", "2.00") are treated as equivalent.
    """
    if _is_number(text):
        return str(float(text))  # Could handle comma-separated numbers
    else:
        return text


def _answer_to_bags(answer: Union[str, List[str], Tuple[str, ...]]) -> Tuple[List[str], List[Set[str]]]:
    """
    Convert answer(s) to normalized spans and token bags.
    
    This function handles the conversion of answers (which can be single strings
    or lists of strings) into two representations:
    1. Normalized spans: For exact matching
    2. Token bags: For F1 calculation
    
    Args:
        answer: Answer in various formats (string, list, or tuple)
        
    Returns:
        Tuple of (normalized_spans, token_bags) for evaluation
    """
    if isinstance(answer, (list, tuple)):
        raw_spans = answer
    else:
        raw_spans = [answer]
        
    normalized_spans: List[str] = []
    token_bags = []
    
    for raw_span in raw_spans:
        normalized_span = _normalize_answer(raw_span)
        normalized_spans.append(normalized_span)
        token_bags.append(set(normalized_span.split()))
        
    return normalized_spans, token_bags


def _align_bags(predicted: List[Set[str]], gold: List[Set[str]]) -> Tuple[List[float], List[float], List[float]]:
    """
    Optimally align predicted and gold answer bags using Hungarian algorithm.
    
    This is one of the most sophisticated parts of DROP evaluation. When answers
    consist of multiple spans, we need to find the optimal 1-1 alignment between
    predicted and gold spans to maximize the overall F1 score.
    
    Args:
        predicted: List of predicted answer token sets
        gold: List of gold answer token sets
        
    Returns:
        Tuple of (max_f1s, max_precisions, max_recalls) for optimal alignment
        
    The Hungarian algorithm ensures we get the maximum possible score by
    finding the best possible pairing between predicted and gold spans.
    """
    f1s = np.zeros([len(gold), len(predicted)])
    precs = np.zeros([len(gold), len(predicted)])
    recalls = np.zeros([len(gold), len(predicted)])
    
    # Compute all pairwise F1 scores
    for gold_index, gold_item in enumerate(gold):
        for pred_index, pred_item in enumerate(predicted):
            # Only compute F1 if numbers match (if present)
            if _match_numbers_if_present(gold_item, pred_item):
                f1_prec_recall = _compute_f1(pred_item, gold_item)
                f1s[gold_index, pred_index] = f1_prec_recall[0]
                precs[gold_index, pred_index] = f1_prec_recall[1]
                recalls[gold_index, pred_index] = f1_prec_recall[2]
                
    # Use Hungarian algorithm to find optimal alignment
    row_ind, col_ind = linear_sum_assignment(-f1s)

    # Extract maximum scores for optimal alignment
    max_f1s = np.zeros([max(len(gold), len(predicted))])
    max_precs = np.zeros([max(len(gold), len(predicted))])
    max_recalls = np.zeros([max(len(gold), len(predicted))])
    
    for row, column in zip(row_ind, col_ind):
        max_f1s[row] = max(max_f1s[row], f1s[row, column])
        max_precs[row] = max(max_precs[row], precs[row, column])
        max_recalls[row] = max(max_recalls[row], recalls[row, column])
        
    return (max_f1s, max_precs, max_recalls)


def _compute_f1(predicted_bag: Set[str], gold_bag: Set[str]) -> Tuple[float, float, float]:
    """
    Compute F1, precision, and recall between two token sets.
    
    This is the core metric computation for DROP evaluation, measuring
    the overlap between predicted and gold answer tokens.
    """
    intersection = len(gold_bag.intersection(predicted_bag))
    
    # Handle edge cases
    if not predicted_bag:
        precision = 1.0
    else:
        precision = intersection / float(len(predicted_bag))
        
    if not gold_bag:
        recall = 1.0
    else:
        recall = intersection / float(len(gold_bag))
        
    # Compute F1 score
    f1 = (2 * precision * recall) / (precision + recall) if not (precision == 0.0 and recall == 0.0) else 0.0
    
    return (f1, precision, recall)


def _match_numbers_if_present(gold_bag: Set[str], predicted_bag: Set[str]) -> bool:
    """
    Check if numerical values match between gold and predicted bags.
    
    This is crucial for DROP evaluation: if both bags contain numbers,
    they must have at least one number in common for the comparison to be valid.
    This prevents false matches between answers that differ in their numerical content.
    
    Args:
        gold_bag: Set of gold answer tokens
        predicted_bag: Set of predicted answer tokens
        
    Returns:
        True if numerical constraints are satisfied, False otherwise
    """
    gold_numbers = set()
    predicted_numbers = set()
    
    # Extract all numbers from both bags
    for word in gold_bag:
        if _is_number(word):
            gold_numbers.add(word)
    for word in predicted_bag:
        if _is_number(word):
            predicted_numbers.add(word)
    
    # If no numbers in gold, or if there's overlap in numbers, allow comparison
    if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
        return True
    return False


def get_metrics(
    predicted: Union[str, List[str], Tuple[str, ...]], gold: Union[str, List[str], Tuple[str, ...]]
) -> Tuple[float, float, float, float]:
    """
    Compute DROP evaluation metrics for a single prediction-gold pair.
    
    This is the main evaluation function that computes exact match and F1 scores
    using DROP's sophisticated evaluation logic.
    
    Args:
        predicted: Predicted answer (string, list, or tuple)
        gold: Gold answer (string, list, or tuple)
        
    Returns:
        Tuple of (exact_match, f1, precision, recall) scores
        
    This function handles the full complexity of DROP evaluation:
    - Multiple answer spans
    - Numerical reasoning
    - Optimal alignment
    - Robust normalization
    """
    # Convert answers to normalized bags
    predicted_bags = _answer_to_bags(predicted)
    gold_bags = _answer_to_bags(gold)

    # Check for exact match (normalized spans must be identical)
    if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
        exact_match = 1.0
    else:
        exact_match = 0.0

    # Compute F1 using optimal alignment
    f1_per_bag, prec_per_bag, recall_per_bag = _align_bags(predicted_bags[1], gold_bags[1])

    # Average across all aligned bags
    f1 = np.mean(f1_per_bag)
    f1 = round(f1, 2)

    prec = np.mean(prec_per_bag)
    prec = round(prec, 2)

    recall = np.mean(recall_per_bag)
    recall = round(recall, 2)

    return exact_match, f1, prec, recall


def answer_json_to_strings(answer: Dict[str, Any]) -> Tuple[Tuple[str, ...], str]:
    """
    Convert DROP answer JSON to string representation for evaluation.
    
    DROP answers come in different formats:
    - Numbers: {"number": 42}
    - Text spans: {"spans": ["span1", "span2"]}
    - Dates: {"date": {"day": 1, "month": "January", "year": 2020}}
    
    This function normalizes all formats to strings for consistent evaluation.
    
    Args:
        answer: Answer dictionary from DROP data
        
    Returns:
        Tuple of (answer_strings, answer_type) for evaluation
    """
    if "number" in answer and answer["number"]:
        return tuple([str(answer["number"])]), "number"
    elif "spans" in answer and answer["spans"]:
        return tuple(answer["spans"]), "span" if len(answer["spans"]) == 1 else "spans"
    elif "date" in answer:
        return (
            tuple(["{0} {1} {2}".format(answer["date"]["day"], answer["date"]["month"], answer["date"]["year"])]),
            "date",
        )
    else:
        raise ValueError(f"Answer type not found, should be one of number, spans or date at: {json.dumps(answer)}")


def evaluate_json(annotations: Dict[str, Any], predicted_answers: Dict[str, Any]) -> Tuple[float, float]:
    """
    Evaluate predictions against DROP annotations.
    
    This function processes the full DROP evaluation format, handling multiple
    answer candidates and computing comprehensive statistics by answer type.
    
    Args:
        annotations: DROP annotations in official format
        predicted_answers: Dictionary mapping query_id to predicted answers
        
    Returns:
        Tuple of (global_em, global_f1) scores
        
    The function also prints detailed statistics broken down by answer type
    (number, span, date) to help analyze model performance patterns.
    """
    instance_exact_match = []
    instance_f1 = []
    
    # Track performance by answer type
    type_to_em: Dict[str, List[float]] = defaultdict(list)
    type_to_f1: Dict[str, List[float]] = defaultdict(list)
    
    # Process each passage and its QA pairs
    for _, annotation in annotations.items():
        for qa_pair in annotation["qa_pairs"]:
            query_id = qa_pair["query_id"]
            max_em_score = 0.0
            max_f1_score = 0.0
            max_type = None
            
            if query_id in predicted_answers:
                predicted = predicted_answers[query_id]
                
                # Collect all valid answers (main + validated)
                candidate_answers = [qa_pair["answer"]]
                if "validated_answers" in qa_pair and qa_pair["validated_answers"]:
                    candidate_answers += qa_pair["validated_answers"]
                
                # Evaluate against all candidate answers, take maximum score
                for answer in candidate_answers:
                    gold_answer, gold_type = answer_json_to_strings(answer)
                    em_score, f1_score, _, _ = get_metrics(predicted, gold_answer)
                    
                    # Only consider non-empty answers
                    if gold_answer[0].strip() != "":
                        max_em_score = max(max_em_score, em_score)
                        max_f1_score = max(max_f1_score, f1_score)
                        if max_em_score == em_score and max_f1_score == f1_score:
                            max_type = gold_type
            else:
                # Missing prediction
                print("Missing prediction for question: {}".format(query_id))
                if qa_pair and qa_pair["answer"]:
                    max_type = answer_json_to_strings(qa_pair["answer"])[1]
                else:
                    max_type = "number"
                max_em_score = 0.0
                max_f1_score = 0.0
                
            # Record scores
            instance_exact_match.append(max_em_score)
            instance_f1.append(max_f1_score)
            type_to_em[max_type].append(max_em_score)
            type_to_f1[max_type].append(max_f1_score)

    # Compute and display overall metrics
    global_em = np.mean(instance_exact_match)
    global_f1 = np.mean(instance_f1)
    
    print("Exact-match accuracy {0:.2f}".format(global_em * 100))
    print("F1 score {0:.2f}".format(global_f1 * 100))
    print("{0:.2f}   &   {1:.2f}".format(global_em * 100, global_f1 * 100))
    print("----")
    
    # Display breakdown by answer type
    total = np.sum([len(v) for v in type_to_em.values()])
    for typ in sorted(type_to_em.keys()):
        print("{0}: {1} ({2:.2f}%)".format(typ, len(type_to_em[typ]), 100.0 * len(type_to_em[typ]) / total))
        print("  Exact-match accuracy {0:.3f}".format(100.0 * np.mean(type_to_em[typ])))
        print("  F1 score {0:.3f}".format(100.0 * np.mean(type_to_f1[typ])))
        
    return global_em, global_f1


def evaluate_prediction_file(
    prediction_path: str, gold_path: str, output_path: Optional[str] = None
) -> Tuple[float, float]:
    """
    Evaluate a prediction file against gold annotations.
    
    This is the main entry point for batch evaluation of DROP predictions.
    
    Args:
        prediction_path: Path to JSON file with predictions
        gold_path: Path to JSON file with gold annotations
        output_path: Optional path to save evaluation results
        
    Returns:
        Tuple of (global_em, global_f1) scores
        
    The prediction file should be a JSON dictionary mapping query_id to answers.
    The gold file should be in the official DROP format.
    """
    predicted_answers = json.load(open(prediction_path, encoding="utf-8"))
    annotations = json.load(open(gold_path, encoding="utf-8"))
    global_em, global_f1 = evaluate_json(annotations, predicted_answers)

    # Save results if output path provided
    if output_path is not None:
        output_dict = {"global_em": global_em, "global_f1": global_f1}
        with open(output_path, "w", encoding="utf8") as outfile:
            json.dump(output_dict, outfile)

    return (global_em, global_f1)


if __name__ == "__main__":
    # Command-line interface for DROP evaluation
    parser = argparse.ArgumentParser(description="evaluate on drop dataset")
    parser.add_argument(
        "--gold_path",
        type=str,
        required=False,
        default="drop_dataset_test.gold.json",
        help="location of the gold file",
    )
    parser.add_argument(
        "--prediction_path",
        type=str,
        required=False,
        default="sample_predictions.json",
        help="location of the prediction file",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=False,
        default=None,
        help="location of the output metrics file",
    )

    args = parser.parse_args()
    evaluate_prediction_file(args.prediction_path, args.gold_path, args.output_path)
