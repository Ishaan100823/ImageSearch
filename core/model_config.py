"""
Model Configuration for Multi-Model Image Search System
Centralizes all model settings for easy management
"""
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent  # Go up one level since we're in core/
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Model configurations
MODELS = {
    "general": {
        "name": "General Product Search",
        "description": "Multi-category product search (175 images, 5 categories)",
        "data_dir": str(DATA_DIR / "general"),
        "models_dir": str(MODELS_DIR / "general"),
        "csv_file": "products.csv",
        "categories": ["Hoodies", "Jeans", "Shirts", "Sweaters", "T-Shirts", "Trousers"],
        "api_path": "/general",
        "port_offset": 0  # Will use base port (8000)
    },
    "shirts": {
        "name": "Specialized Shirts Search", 
        "description": "Shirts-only specialized search (3-4k images)",
        "data_dir": str(DATA_DIR / "shirts"),
        "models_dir": str(MODELS_DIR / "shirts"), 
        "csv_file": "products.csv",  # Your shirts CSV goes in data/shirts/
        "categories": ["Casual Shirts", "Formal Shirts", "Polo Shirts", "Button-down", "T-Shirts"],
        "api_path": "/shirts",
        "port_offset": 1  # Optional: separate port (8001)
    }
}

def get_model_config(model_name: str) -> dict:
    """Get configuration for a specific model"""
    if model_name not in MODELS:
        raise ValueError(f"Model '{model_name}' not found. Available: {list(MODELS.keys())}")
    return MODELS[model_name]

def get_model_paths(model_name: str) -> dict:
    """Get all file paths for a specific model"""
    config = get_model_config(model_name)
    base_data = Path(config["data_dir"])
    base_models = Path(config["models_dir"])
    
    return {
        "csv_file": str(base_data / config["csv_file"]),
        "image_cache_dir": str(base_data / "image_cache"),
        "processed_products_file": str(base_data / "processed_products.json"),
        "index_file": str(base_models / "index.faiss"),
        "metadata_file": str(base_models / "metadata.pkl"),
        "embeddings_file": str(base_models / "embeddings.npy")
    }

def list_available_models() -> list:
    """List all available model configurations"""
    return list(MODELS.keys())

def create_model_directories(model_name: str):
    """Create necessary directories for a model"""
    config = get_model_config(model_name)
    Path(config["data_dir"]).mkdir(parents=True, exist_ok=True)
    Path(config["models_dir"]).mkdir(parents=True, exist_ok=True)
    (Path(config["data_dir"]) / "image_cache").mkdir(exist_ok=True) 