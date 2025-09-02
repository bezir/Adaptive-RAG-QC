"""
An abstract class representing a metric which can be accumulated.

This serves as the base class for all evaluation metrics in the Adaptive RAG system,
providing a consistent interface for measuring both retrieval and generation performance.
"""
from typing import Any, Dict


class Metric:
    """
    An abstract class representing a metric which can be accumulated.
    
    This base class ensures all metrics follow a consistent pattern:
    1. Accumulate results over multiple examples via __call__
    2. Compute final metrics via get_metric
    3. Reset state for new evaluation rounds via reset
    
    This design allows for efficient batch evaluation and consistent reporting
    across different types of metrics (answer quality, retrieval quality, etc.)
    """

    def __call__(self, predictions: Any, gold_labels: Any):
        """
        Update the metric's internal state with a new prediction/gold label pair.
        
        This method is called for each example during evaluation to accumulate
        the necessary statistics for computing the final metric values.
        
        Args:
            predictions: Model predictions (format varies by metric type)
            gold_labels: Ground truth labels (format varies by metric type)
        """
        raise NotImplementedError

    def get_metric(self, reset: bool) -> Dict[str, Any]:
        """
        Compute and return the metric. Optionally also call `self.reset`.
        
        This method calculates the final metric values based on all accumulated
        examples and returns them as a dictionary for easy reporting.
        
        Args:
            reset: Whether to reset the internal state after computing metrics
            
        Returns:
            Dictionary containing metric names and their computed values
        """
        raise NotImplementedError

    def reset(self) -> None:
        """
        Reset any accumulators or internal state.
        
        This method clears all accumulated statistics, preparing the metric
        for a new evaluation round. Essential for proper batch evaluation.
        """
        raise NotImplementedError
