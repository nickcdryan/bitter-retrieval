"""
Bitter Retrieval: A modular training framework for retrieval models
"""

__version__ = "0.1.0"

from . import config
from . import models
from . import training
from . import data
from . import evaluation
from . import utils

__all__ = [
    "config",
    "models", 
    "training",
    "data",
    "evaluation",
    "utils"
] 