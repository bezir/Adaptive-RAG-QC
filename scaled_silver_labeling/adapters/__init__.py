#!/usr/bin/env python3
"""
IRCoT Adapters Package

This package contains adapters that bridge the gap between the silver labeling system
and the benchmark IRCoT implementation, ensuring consistency between training and evaluation.
"""

from .ircot_bridge_adapter import IRCoTBridgeAdapter

__all__ = ['IRCoTBridgeAdapter']