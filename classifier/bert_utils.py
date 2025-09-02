#!/usr/bin/env python
# coding=utf-8
"""
Utility functions for BERT-based query complexity classification.
"""

import torch
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from typing import Dict, List, Tuple
import json
import os
from transformers import AutoTokenizer, BertForSequenceClassification
import datetime


def load_bert_model_and_tokenizer(model_path: str, num_labels: int = 3):
    """
    Load BERT model and tokenizer from path.
    
    Args:
        model_path: Path to the model directory
        num_labels: Number of classification labels (default: 3 for A, B, C)
    
    Returns:
        Tuple of (model, tokenizer)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = BertForSequenceClassification.from_pretrained(model_path, num_labels=num_labels)
    
    return model, tokenizer


def preprocess_data_for_bert(examples: Dict, tokenizer, max_length: int = 384) -> Dict:
    """
    Preprocess data for BERT classification.
    
    Args:
        examples: Dictionary with 'question' and 'answer' keys
        tokenizer: BERT tokenizer
        max_length: Maximum sequence length
    
    Returns:
        Dictionary with tokenized inputs and labels
    """
    # Label mapping
    option_to_label = {
        'A': 0,  # No retrieval
        'B': 1,  # Single-step retrieval
        'C': 2,  # Multi-step retrieval
    }
    
    # Tokenize questions
    tokenized = tokenizer(
        examples["question"],
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    
    # Convert string labels to integers
    labels = [option_to_label[label] for label in examples["answer"]]
    tokenized["labels"] = labels
    
    return tokenized


def calculate_bert_metrics(predictions: List[int], true_labels: List[int]) -> Dict:
    """
    Calculate comprehensive metrics for BERT classification.
    
    Args:
        predictions: List of predicted labels (0, 1, 2)
        true_labels: List of true labels (0, 1, 2)
    
    Returns:
        Dictionary with various metrics
    """
    # Overall accuracy
    accuracy = accuracy_score(true_labels, predictions)
    
    # Classification report
    target_names = ['A (No Retrieval)', 'B (Single-Step)', 'C (Multi-Step)']
    class_report = classification_report(
        true_labels, predictions, 
        target_names=target_names,
        output_dict=True
    )
    
    # Confusion matrix
    cm = confusion_matrix(true_labels, predictions)
    
    # Per-class accuracy
    class_accuracies = {}
    for class_idx, class_name in enumerate(['A', 'B', 'C']):
        # Find indices where true label is this class
        class_indices = [i for i, label in enumerate(true_labels) if label == class_idx]
        
        if len(class_indices) > 0:
            # Calculate accuracy for this class
            class_predictions = [predictions[i] for i in class_indices]
            class_true = [true_labels[i] for i in class_indices]
            class_accuracy = accuracy_score(class_true, class_predictions)
            
            class_accuracies[f'{class_name}_accuracy'] = class_accuracy
            class_accuracies[f'{class_name}_count'] = len(class_indices)
            class_accuracies[f'{class_name}_predicted_count'] = len([p for p in predictions if p == class_idx])
        else:
            class_accuracies[f'{class_name}_accuracy'] = 0.0
            class_accuracies[f'{class_name}_count'] = 0
            class_accuracies[f'{class_name}_predicted_count'] = len([p for p in predictions if p == class_idx])
    
    return {
        'accuracy': accuracy,
        'classification_report': class_report,
        'confusion_matrix': cm.tolist(),
        'per_class_metrics': class_accuracies
    }


def predict_with_bert(model, tokenizer, questions: List[str], device: str = "cuda", max_length: int = 384) -> List[str]:
    """
    Make predictions using trained BERT model.
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        questions: List of questions to classify
        device: Device to run inference on
        max_length: Maximum sequence length
    
    Returns:
        List of predicted labels as strings ('A', 'B', 'C')
    """
    label_to_option = {0: 'A', 1: 'B', 2: 'C'}
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for question in questions:
            # Tokenize single question
            inputs = tokenizer(
                question,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt"
            ).to(device)
            
            # Get prediction
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class = torch.argmax(logits, dim=-1).cpu().numpy()[0]
            
            predictions.append(label_to_option[predicted_class])
    
    return predictions


def save_bert_results(results: Dict, output_dir: str):
    """
    Save BERT classification results to files.
    
    Args:
        results: Dictionary with results to save
        output_dir: Directory to save results
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Save main results
    with open(os.path.join(output_dir, "bert_results.json"), "w") as f:
        json.dump(results, f, indent=2)
    
    # Save confusion matrix as numpy array
    if 'confusion_matrix' in results:
        np.save(os.path.join(output_dir, "confusion_matrix.npy"), np.array(results['confusion_matrix']))
    
    # Save per-class metrics
    if 'per_class_metrics' in results:
        with open(os.path.join(output_dir, "per_class_metrics.json"), "w") as f:
            json.dump(results['per_class_metrics'], f, indent=2)


def load_classification_data(file_path: str) -> Tuple[List[str], List[str], List[str]]:
    """
    Load classification data from JSON file.
    
    Args:
        file_path: Path to JSON file
    
    Returns:
        Tuple of (questions, labels, dataset_names)
    """
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    questions = [item['question'] for item in data]
    labels = [item['answer'] for item in data]
    dataset_names = [item.get('dataset_name', 'unknown') for item in data]
    
    return questions, labels, dataset_names


def create_bert_dataset_summary(data_path: str) -> Dict:
    """
    Create a summary of the dataset for BERT classification.
    
    Args:
        data_path: Path to the dataset JSON file
    
    Returns:
        Dictionary with dataset statistics
    """
    questions, labels, dataset_names = load_classification_data(data_path)
    
    # Count labels
    label_counts = {}
    for label in labels:
        label_counts[label] = label_counts.get(label, 0) + 1
    
    # Count by dataset
    dataset_counts = {}
    for dataset in dataset_names:
        dataset_counts[dataset] = dataset_counts.get(dataset, 0) + 1
    
    # Count by dataset and label
    dataset_label_counts = {}
    for dataset, label in zip(dataset_names, labels):
        if dataset not in dataset_label_counts:
            dataset_label_counts[dataset] = {}
        dataset_label_counts[dataset][label] = dataset_label_counts[dataset].get(label, 0) + 1
    
    summary = {
        'total_examples': len(questions),
        'label_distribution': label_counts,
        'dataset_distribution': dataset_counts,
        'dataset_label_distribution': dataset_label_counts,
        'average_question_length': sum(len(q.split()) for q in questions) / len(questions),
        'max_question_length': max(len(q.split()) for q in questions),
        'min_question_length': min(len(q.split()) for q in questions),
    }
    
    return summary


def evaluate_bert_classifier(model, tokenizer, test_data_path: str, device: str = "cuda", batch_size: int = 32) -> Dict:
    """
    Evaluate BERT classifier on test data.
    
    Args:
        model: Trained BERT model
        tokenizer: BERT tokenizer
        test_data_path: Path to test data JSON file
        device: Device to run on
        batch_size: Batch size for evaluation
    
    Returns:
        Dictionary with evaluation results
    """
    questions, true_labels, dataset_names = load_classification_data(test_data_path)
    
    # Convert string labels to integers
    option_to_label = {'A': 0, 'B': 1, 'C': 2}
    true_labels_int = [option_to_label[label] for label in true_labels]
    
    # Make predictions
    predicted_labels = predict_with_bert(model, tokenizer, questions, device)
    predicted_labels_int = [option_to_label[label] for label in predicted_labels]
    
    # Calculate metrics
    metrics = calculate_bert_metrics(predicted_labels_int, true_labels_int)
    
    # Add dataset-wise metrics
    dataset_metrics = {}
    for dataset in set(dataset_names):
        dataset_indices = [i for i, d in enumerate(dataset_names) if d == dataset]
        if len(dataset_indices) > 0:
            dataset_true = [true_labels_int[i] for i in dataset_indices]
            dataset_pred = [predicted_labels_int[i] for i in dataset_indices]
            dataset_metrics[dataset] = calculate_bert_metrics(dataset_pred, dataset_true)
    
    metrics['dataset_metrics'] = dataset_metrics
    
    return metrics


def format_time(elapsed):
    """
    Takes a time in seconds and returns a string hh:mm:ss
    """
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def create_training_summary(train_results: Dict, eval_results: Dict) -> Dict:
    """
    Create a comprehensive training summary.
    
    Args:
        train_results: Training results dictionary
        eval_results: Evaluation results dictionary
    
    Returns:
        Dictionary with training summary
    """
    summary = {
        'training_completed': True,
        'final_train_loss': train_results.get('final_loss', 'N/A'),
        'final_eval_accuracy': eval_results.get('accuracy', 'N/A'),
        'best_eval_accuracy': train_results.get('best_eval_accuracy', 'N/A'),
        'total_training_time': train_results.get('total_time', 'N/A'),
        'num_training_steps': train_results.get('num_steps', 'N/A'),
        'num_epochs': train_results.get('num_epochs', 'N/A'),
        'per_class_performance': eval_results.get('per_class_metrics', {}),
    }
    
    return summary 