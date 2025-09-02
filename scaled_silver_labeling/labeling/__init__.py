"""
Labeling Package for Scaled Silver Labeling

This package contains the implementation of different labeling strategies
for generating silver labels at scale.
"""

from .base_labeler import BaseLabeler
from .original_labeler import OriginalLabeler
from .optimized_labeler import OptimizedLabeler

__all__ = ['BaseLabeler', 'OriginalLabeler', 'OptimizedLabeler']
