"""
Main Pipeline Orchestrator for Image Search System
Handles the complete end-to-end process with configurable stages
"""
import sys
import time
import argparse
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from config import config, config_full_rebuild, config_quick_test, config_resume_from_images, config_resume_from_features, config_api_only
from utils.logging_utils import process_logger, log_system_requirements
from process_csv_data import CSVProcessor
from build_engine import SearchEngineBuilder


class ImageSearchPipeline:
    """Main pipeline orchestrator with configurable stages"""
    
    def __init__(self):
        self.logger = process_logger.logger
        self.csv_processor = CSVProcessor()
        self.engine_builder = SearchEngineBuilder()
    
    def run_data_pipeline(self) -> bool:
        """Run data processing stages (Phase 1-2)"""
        self.logger.info("🚀 Starting Data Pipeline...")
        
        pipeline_start = time.time()
        
        # Phase 1: Data validation
        if config.processes["data_validation"].enabled:
            self.logger.info("📋 Phase 1: Data Validation")
            if not self.csv_processor.validate_data():
                self.logger.error("❌ Data validation failed")
                return False
        
        # Phase 2: Image download
        if config.processes["image_download"].enabled:
            self.logger.info("📥 Phase 2: Image Download")
            if not self.csv_processor.download_images():
                self.logger.error("❌ Image download failed")
                return False
        
        pipeline_duration = time.time() - pipeline_start
        self.logger.info(f"✅ Data Pipeline completed in {pipeline_duration:.1f}s")
        return True
    
    def run_ml_pipeline(self) -> bool:
        """Run ML processing stages (Phase 3-4)"""
        self.logger.info("🧠 Starting ML Pipeline...")
        
        pipeline_start = time.time()
        
        # Phase 3: Feature extraction
        if config.processes["feature_extraction"].enabled:
            self.logger.info("🔬 Phase 3: Feature Extraction")
            if not self.engine_builder.extract_features():
                self.logger.error("❌ Feature extraction failed")
                return False
        
        # Phase 4: Index building
        if config.processes["index_building"].enabled:
            self.logger.info("🔍 Phase 4: Index Building")
            if not self.engine_builder.build_index():
                self.logger.error("❌ Index building failed")
                return False
        
        pipeline_duration = time.time() - pipeline_start
        self.logger.info(f"✅ ML Pipeline completed in {pipeline_duration:.1f}s")
        return True
    
    def run_full_pipeline(self) -> bool:
        """Run the complete pipeline from start to finish"""
        self.logger.info("🚀 Starting Full Image Search Pipeline...")
        
        total_start = time.time()
        
        # Log configuration status
        status = config.get_status()
        self.logger.info(f"📋 Pipeline Configuration: {status}")
        
        # Data Pipeline (Phases 1-2)
        if not self.run_data_pipeline():
            return False
        
        # ML Pipeline (Phases 3-4)
        if not self.run_ml_pipeline():
            return False
        
        total_duration = time.time() - total_start
        self.logger.info(f"🎉 Full Pipeline completed successfully in {total_duration:.1f}s!")
        
        # Print final summary
        self._print_completion_summary()
        return True
    
    def run_servers(self):
        """Launch API and UI servers"""
        self.logger.info("🚀 Starting Servers...")
        
        import subprocess
        import threading
        
        def start_api():
            """Start API server in separate process"""
            if config.processes["api_server"].enabled:
                self.logger.info("🔧 Starting API server...")
                subprocess.run([
                    sys.executable, "main_api.py"
                ])
        
        def start_ui():
            """Start UI server in separate process"""
            if config.processes["ui_frontend"].enabled:
                # Wait a bit for API to start
                time.sleep(3)
                self.logger.info("🖥️ Starting UI frontend...")
                subprocess.run([
                    sys.executable, "app_ui.py"
                ])
        
        # Start servers in threads
        threads = []
        
        if config.processes["api_server"].enabled:
            api_thread = threading.Thread(target=start_api, daemon=True)
            api_thread.start()
            threads.append(api_thread)
        
        if config.processes["ui_frontend"].enabled:
            ui_thread = threading.Thread(target=start_ui, daemon=True)
            ui_thread.start()
            threads.append(ui_thread)
        
        if threads:
            self.logger.info("🌟 Servers started! Press Ctrl+C to stop.")
            try:
                # Keep main thread alive
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                self.logger.info("🛑 Shutting down servers...")
        else:
            self.logger.info("ℹ️ No servers enabled in configuration")
    
    def _print_completion_summary(self):
        """Print completion summary with file locations"""
        print("\n" + "="*60)
        print("🎉 IMAGE SEARCH SYSTEM - PIPELINE COMPLETED")
        print("="*60)
        
        # Data files
        if Path(config.data.processed_products_file).exists():
            print(f"📊 Processed Products: {config.data.processed_products_file}")
        
        if Path(config.data.image_cache_dir).exists():
            cache_count = len(list(Path(config.data.image_cache_dir).glob("*")))
            print(f"🖼️ Image Cache: {config.data.image_cache_dir} ({cache_count} products)")
        
        # Model files
        if Path(config.index.index_file).exists():
            print(f"🔍 FAISS Index: {config.index.index_file}")
        
        if Path(config.index.metadata_file).exists():
            print(f"📋 Metadata: {config.index.metadata_file}")
        
        print("\n🚀 READY TO USE:")
        print(f"   API Server: python main_api.py")
        print(f"   UI Frontend: python app_ui.py")
        print(f"   Or run both: python main.py --servers")
        print("="*60)


def setup_argument_parser():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Image Search System Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Run full pipeline
  python main.py --servers          # Start API and UI servers
  python main.py --data-only        # Run only data processing
  python main.py --ml-only          # Run only ML pipeline
  python main.py --resume-images    # Resume from feature extraction
  python main.py --resume-features  # Resume from index building
  python main.py --full-rebuild     # Force rebuild everything
  python main.py --quick-test       # Quick test mode (skip heavy processes)
        """
    )
    
    # Pipeline modes
    parser.add_argument(
        "--servers", action="store_true",
        help="Start API and UI servers (skips pipeline)"
    )
    parser.add_argument(
        "--data-only", action="store_true",
        help="Run only data processing pipeline"
    )
    parser.add_argument(
        "--ml-only", action="store_true",
        help="Run only ML pipeline"
    )
    
    # Resume options
    parser.add_argument(
        "--resume-images", action="store_true",
        help="Resume from feature extraction (skip data validation and image download)"
    )
    parser.add_argument(
        "--resume-features", action="store_true",
        help="Resume from index building (skip all previous stages)"
    )
    
    # Configuration presets
    parser.add_argument(
        "--full-rebuild", action="store_true",
        help="Force rebuild everything (ignore checkpoints)"
    )
    parser.add_argument(
        "--quick-test", action="store_true",
        help="Quick test mode (skip heavy processes)"
    )
    
    # Process control
    parser.add_argument(
        "--skip", nargs="+", 
        choices=["data_validation", "image_download", "feature_extraction", "index_building"],
        help="Skip specific processes"
    )
    parser.add_argument(
        "--force", nargs="+",
        choices=["data_validation", "image_download", "feature_extraction", "index_building"],
        help="Force rebuild specific processes"
    )
    
    return parser


def main():
    """Main entry point with argument parsing"""
    
    # Parse arguments
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Log system requirements
    log_system_requirements()
    
    # Apply configuration presets
    if args.full_rebuild:
        config_full_rebuild()
    elif args.quick_test:
        config_quick_test()
    elif args.resume_images:
        config_resume_from_images()
    elif args.resume_features:
        config_resume_from_features()
    elif args.servers:
        config_api_only()
    
    # Apply manual process control
    if args.skip:
        for process in args.skip:
            config.skip_process(process)
    
    if args.force:
        for process in args.force:
            config.force_rebuild_process(process)
    
    # Initialize pipeline
    pipeline = ImageSearchPipeline()
    
    try:
        # Run based on mode
        if args.servers:
            pipeline.run_servers()
        elif args.data_only:
            success = pipeline.run_data_pipeline()
        elif args.ml_only:
            success = pipeline.run_ml_pipeline()
        else:
            # Full pipeline
            success = pipeline.run_full_pipeline()
        
        # Exit with appropriate code
        if not args.servers:  # Servers run indefinitely
            if success:
                print("\n✅ Pipeline completed successfully!")
                sys.exit(0)
            else:
                print("\n❌ Pipeline failed!")
                sys.exit(1)
                
    except KeyboardInterrupt:
        print("\n🛑 Pipeline interrupted by user")
        sys.exit(1)
    except Exception as e:
        process_logger.log_process_error("main_pipeline", e)
        print(f"\n❌ Pipeline failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main() 