"""
Configuration file for Image Search System
Handles all settings, paths, and process control
"""
import os
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass, field

@dataclass
class ProcessConfig:
    """Configuration for individual processes with resume capability"""
    enabled: bool = True
    force_rebuild: bool = False
    checkpoint_file: str = ""
    batch_size: int = 32
    max_workers: int = 8

@dataclass
class DataConfig:
    """Data processing configuration"""
    csv_file: str = "products.csv"
    image_cache_dir: str = "image_cache"
    processed_products_file: str = "processed_products.json"
    max_images_per_product: int = 6
    min_image_size: tuple = (100, 100)
    image_formats: List[str] = field(default_factory=lambda: ['jpg', 'jpeg', 'png', 'webp'])
    download_timeout: int = 15
    
    # Multi-category dataset configuration
    expected_categories: List[str] = field(default_factory=lambda: [
        'Hoodies', 'Jeans', 'Shirts', 'Sweaters', 'T-Shirts', 'Trousers'
    ])
    products_per_category: int = 30
    total_expected_products: int = 180
    
@dataclass
class ModelConfig:
    """Model and embedding configuration"""
    clip_model: str = "openai/clip-vit-base-patch32"
    embedding_dim: int = 512
    device: str = "auto"  # auto, cpu, cuda
    batch_size: int = 32
    
@dataclass
class IndexConfig:
    """FAISS index configuration"""
    index_file: str = "product_index.faiss"
    metadata_file: str = "product_metadata.pkl"
    embeddings_file: str = "product_embeddings.npy"
    index_type: str = "flat_ip"
    similarity_metric: str = "cosine"
    build_on_gpu: bool = False
    
@dataclass
class APIConfig:
    """API server configuration"""
    host: str = "127.0.0.1"
    port: int = 8000
    reload: bool = True
    
@dataclass
class UIConfig:
    """Frontend UI configuration"""
    host: str = "127.0.0.1"
    port: int = 7860
    share: bool = False
    top_k_results: int = 5
    

class Config:
    """Main configuration class"""
    
    def __init__(self):
        # Base paths
        self.project_root = Path(__file__).parent
        self.data_dir = self.project_root / "data"
        self.models_dir = self.project_root / "models"
        self.logs_dir = self.project_root / "logs"
        self.checkpoints_dir = self.project_root / "checkpoints"
        
        # Create directories
        for dir_path in [self.data_dir, self.models_dir, self.logs_dir, self.checkpoints_dir]:
            dir_path.mkdir(exist_ok=True)
        
        # Process control - Easy to enable/disable stages
        self.processes = {
            "data_validation": ProcessConfig(
                enabled=True,
                checkpoint_file=str(self.checkpoints_dir / "data_validation.json")
            ),
            "image_download": ProcessConfig(
                enabled=True,
                checkpoint_file=str(self.checkpoints_dir / "image_download.json"),
                batch_size=50,
                max_workers=8
            ),
            "feature_extraction": ProcessConfig(
                enabled=True,
                checkpoint_file=str(self.checkpoints_dir / "feature_extraction.json"),
                batch_size=16,
                max_workers=4
            ),
            "index_building": ProcessConfig(
                enabled=True,
                checkpoint_file=str(self.checkpoints_dir / "index_building.json")
            ),
            "api_server": ProcessConfig(
                enabled=True
            ),
            "ui_frontend": ProcessConfig(
                enabled=True
            )
        }
        
        # Component configurations
        self.data = DataConfig()
        self.model = ModelConfig()
        self.index = IndexConfig()
        self.api = APIConfig()
        self.ui = UIConfig()
        
        # Update paths to be relative to project
        self.data.image_cache_dir = str(self.data_dir / "image_cache")
        self.data.processed_products_file = str(self.data_dir / "processed_products.json")
        self.index.index_file = str(self.models_dir / "product_index.faiss")
        self.index.metadata_file = str(self.models_dir / "product_metadata.pkl")
    
    def skip_process(self, process_name: str):
        """Skip a specific process"""
        if process_name in self.processes:
            self.processes[process_name].enabled = False
            print(f"⏭️  Skipped: {process_name}")
    
    def force_rebuild_process(self, process_name: str):
        """Force rebuild a specific process"""
        if process_name in self.processes:
            self.processes[process_name].force_rebuild = True
            print(f"🔄 Force rebuild: {process_name}")
    
    def enable_only(self, process_names: List[str]):
        """Enable only specific processes"""
        for name in self.processes:
            self.processes[name].enabled = name in process_names
        print(f"✅ Enabled only: {', '.join(process_names)}")
    
    def get_status(self) -> Dict:
        """Get current configuration status"""
        return {
            "enabled_processes": [name for name, config in self.processes.items() if config.enabled],
            "force_rebuild": [name for name, config in self.processes.items() if config.force_rebuild],
            "data_dir": str(self.data_dir),
            "models_dir": str(self.models_dir),
            "device": self.model.device
        }

# Global configuration instance
config = Config()

# Quick configuration presets
def config_full_rebuild():
    """Configure for full rebuild of everything"""
    for process in config.processes.values():
        process.force_rebuild = True
    print("🔄 Full rebuild mode enabled")

def config_quick_test():
    """Configure for quick testing (skip heavy processes)"""
    config.skip_process("image_download")
    config.skip_process("feature_extraction") 
    config.skip_process("index_building")
    print("⚡ Quick test mode enabled")

def config_resume_from_images():
    """Resume from after image download"""
    config.skip_process("data_validation")
    config.skip_process("image_download")
    print("📤 Resuming from feature extraction")

def config_resume_from_features():
    """Resume from after feature extraction"""
    config.skip_process("data_validation")
    config.skip_process("image_download") 
    config.skip_process("feature_extraction")
    print("🔍 Resuming from index building")

def config_api_only():
    """Run only API and UI"""
    config.enable_only(["api_server", "ui_frontend"])
    print("🚀 API-only mode enabled") 
    