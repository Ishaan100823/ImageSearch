"""
Main Pipeline Orchestrator for Image Search System
Clean, simple workflow with excellent resume capability
"""
import time
import sys
from pathlib import Path

from core.config import config
from utils.logging_utils import logger, log_system_requirements
from .process_csv_data import CSVProcessor
from .build_engine import SearchEngineBuilder


class ImageSearchPipeline:
    """Clean main pipeline orchestrator"""
    
    def __init__(self):
        self.csv_processor = CSVProcessor()
        self.engine_builder = SearchEngineBuilder()
    
    def run_complete_pipeline(self) -> bool:
        """Run the complete image search pipeline with resume capability"""
        logger.info("🚀 Starting Image Search Pipeline")
        logger.info("=" * 50)
        
        # Log system requirements
        log_system_requirements()
        
        # Create required directories
        self._ensure_directories()
        
        # Phase 1: Data Processing
        logger.info("📊 Phase 1: Data Processing")
        if not self.csv_processor.process_all():
            logger.error("❌ Data processing failed")
            return False
        
        # Phase 2: ML Pipeline  
        logger.info("🧠 Phase 2: ML Pipeline")
        if not self.engine_builder.build_all():
            logger.error("❌ ML pipeline failed")
            return False
        
        # Pipeline completed successfully
        logger.success("✅ Complete pipeline finished successfully!")
        logger.info("🎉 Your image search system is ready!")
        logger.info(f"   📁 Index: {config.index.index_file}")
        logger.info(f"   📊 Metadata: {config.index.metadata_file}")
        logger.info(f"   🖼️ Images: {config.data.image_cache_dir}")
        
        return True
    
    def run_data_only(self) -> bool:
        """Run only data processing phases"""
        logger.info("📊 Running Data Processing Only")
        
        self._ensure_directories()
        
        if not self.csv_processor.process_all():
            logger.error("❌ Data processing failed")
            return False
        
        logger.success("✅ Data processing completed!")
        return True
    
    def run_ml_only(self) -> bool:
        """Run only ML phases (assumes data is already processed)"""
        logger.info("🧠 Running ML Pipeline Only")
        
        self._ensure_directories()
        
        if not self.engine_builder.build_all():
            logger.error("❌ ML pipeline failed")
            return False
        
        logger.success("✅ ML pipeline completed!")
        return True
    
    def _ensure_directories(self):
        """Create required directories"""
        dirs_to_create = [
            config.data.image_cache_dir,
            config.models_dir,
            config.checkpoints_dir,
            config.logs_dir,
            Path(config.index.index_file).parent,
            Path(config.index.metadata_file).parent
        ]
        
        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)


def main():
    """Main entry point with command line argument support"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Image Search System Pipeline")
    parser.add_argument(
        "--mode", 
        choices=["all", "data", "ml"], 
        default="all",
        help="Pipeline mode: 'all' (complete), 'data' (data processing only), 'ml' (ML only)"
    )
    
    args = parser.parse_args()
    
    # Initialize pipeline
    pipeline = ImageSearchPipeline()
    
    # Run based on mode
    start_time = time.time()
    
    try:
        if args.mode == "all":
            success = pipeline.run_complete_pipeline()
        elif args.mode == "data":
            success = pipeline.run_data_only()
        elif args.mode == "ml":
            success = pipeline.run_ml_only()
        else:
            logger.error(f"Unknown mode: {args.mode}")
            sys.exit(1)
        
        # Report results
        total_time = time.time() - start_time
        
        if success:
            logger.success(f"🎉 Pipeline completed successfully in {total_time:.1f}s")
            
            # Show next steps
            if args.mode in ["all", "ml"]:
                logger.info("Next steps:")
                logger.info("  1. Start API: python main_api.py")
                logger.info("  2. Test search: upload images via API")
                logger.info("  3. View docs: http://localhost:8000/docs")
        else:
            logger.error(f"❌ Pipeline failed after {total_time:.1f}s")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.error("Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Pipeline failed with unexpected error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 