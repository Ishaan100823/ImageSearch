"""
Universal Model Training Script
Trains any model by dynamically setting paths and configurations
"""
import sys
import os
import time
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(str(Path(__file__).parent.parent))

from core.model_config import get_model_config, get_model_paths, create_model_directories, list_available_models
from core.config import config
from pipeline.main import ImageSearchPipeline
from utils.logging_utils import logger


class ModelTrainer:
    """Universal model trainer that works with any model configuration"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model_config = get_model_config(model_name)
        self.model_paths = get_model_paths(model_name)
        self.pipeline = ImageSearchPipeline()
    
    def setup_environment(self):
        """Setup environment for training the specific model"""
        logger.info(f"🔧 Setting up environment for model: {self.model_name}")
        
        # Create necessary directories
        create_model_directories(self.model_name)
        
        # Temporarily update config to point to this model's paths
        self._update_config_paths()
        
        # Verify CSV file exists
        if not Path(self.model_paths["csv_file"]).exists():
            raise FileNotFoundError(
                f"CSV file not found: {self.model_paths['csv_file']}\n"
                f"Please place your CSV file at this location."
            )
        
        logger.success(f"✅ Environment ready for {self.model_config['name']}")
    
    def _update_config_paths(self):
        """Update global config to use this model's paths"""
        config.data.csv_file = self.model_paths["csv_file"]
        config.data.image_cache_dir = self.model_paths["image_cache_dir"]
        config.data.processed_products_file = self.model_paths["processed_products_file"]
        config.index.index_file = self.model_paths["index_file"]
        config.index.metadata_file = self.model_paths["metadata_file"]
        config.index.embeddings_file = self.model_paths["embeddings_file"]
        config.data.expected_categories = self.model_config["categories"]
        
        logger.info(f"📁 Updated paths for {self.model_name}:")
        logger.info(f"   CSV: {self.model_paths['csv_file']}")
        logger.info(f"   Cache: {self.model_paths['image_cache_dir']}")
        logger.info(f"   Index: {self.model_paths['index_file']}")
    
    def train(self, force_rebuild: bool = False):
        """Train the model"""
        logger.info(f"🚀 Starting training for {self.model_config['name']}")
        logger.info(f"📊 Categories: {', '.join(self.model_config['categories'])}")
        
        # Setup environment
        self.setup_environment()
        
        # Configure force rebuild if requested
        if force_rebuild:
            for process in config.processes.values():
                process.force_rebuild = True
            logger.info("🔄 Force rebuild enabled")
        
        # Run the training pipeline
        start_time = time.time()
        success = self.pipeline.run_complete_pipeline()
        training_time = time.time() - start_time
        
        if success:
            logger.success(f"✅ Model '{self.model_name}' trained successfully!")
            logger.info(f"⏱️  Training time: {training_time:.1f}s")
            self._show_model_info()
        else:
            logger.error(f"❌ Training failed for model '{self.model_name}'")
            return False
        
        return True
    
    def _show_model_info(self):
        """Show information about the trained model"""
        logger.info(f"📊 Model Information:")
        logger.info(f"   Name: {self.model_config['name']}")
        logger.info(f"   Description: {self.model_config['description']}")
        logger.info(f"   API Path: {self.model_config['api_path']}")
        logger.info(f"   Categories: {len(self.model_config['categories'])}")
        
        # Show file sizes
        index_path = Path(self.model_paths["index_file"])
        if index_path.exists():
            size_mb = index_path.stat().st_size / 1024 / 1024
            logger.info(f"   Index size: {size_mb:.1f} MB")


def main():
    """Main training entry point"""
    parser = argparse.ArgumentParser(description="Universal Model Training")
    parser.add_argument("model", choices=list_available_models(), help="Model to train")
    parser.add_argument("--force", action="store_true", help="Force rebuild all components")
    parser.add_argument("--list", action="store_true", help="List available models")
    
    args = parser.parse_args()
    
    if args.list:
        logger.info("📋 Available Models:")
        for model_name in list_available_models():
            model_config = get_model_config(model_name)
            logger.info(f"  • {model_name}: {model_config['description']}")
        return
    
    try:
        trainer = ModelTrainer(args.model)
        success = trainer.train(force_rebuild=args.force)
        
        if success:
            logger.info("\n🎉 Next steps:")
            logger.info(f"  1. Start API: python api/enhanced_api.py")
            logger.info(f"  2. Test model: POST http://localhost:8000{get_model_config(args.model)['api_path']}/search")
            logger.info(f"  3. View docs: http://localhost:8000/docs")
        
        sys.exit(0 if success else 1)
        
    except FileNotFoundError as e:
        logger.error(f"❌ {str(e)}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Training failed: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 