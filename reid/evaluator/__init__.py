from __future__ import absolute_import
from .eva_functions import accuracy, cmc, mean_ap
from .evaluator import CNNEvaluator
from .attevaluator import ATTEvaluator


__all__ = [
    'accuracy',
    'cmc',
    'mean_ap',
]


