"""
Core Configuration Module
Contains all configuration settings and model definitions
"""

from .config import config
from .model_config import (
    MODELS, 
    get_model_config, 
    get_model_paths, 
    list_available_models, 
    create_model_directories
)

__all__ = [
    'config',
    'MODELS',
    'get_model_config',
    'get_model_paths', 
    'list_available_models',
    'create_model_directories'
] 