#!/usr/bin/env python3
"""
Quick Setup for 180-Product Image Search
Simplified workflow optimized for your dataset
"""

import sys
import time
from pathlib import Path

# Add utils to path
sys.path.append(str(Path(__file__).parent))

from config import config
from utils.logging_utils import process_logger
from process_csv_data import CSVProcessor
from build_engine import SearchEngineBuilder


def clean_previous_data():
    """Clean old dataset artifacts"""
    print("🧹 Cleaning previous dataset...")
    
    # Clean old cached images
    cache_dir = Path(config.data.image_cache_dir)
    if cache_dir.exists():
        import shutil
        shutil.rmtree(cache_dir)
        cache_dir.mkdir(exist_ok=True)
    
    # Clean old models
    models_dir = Path(config.models_dir)
    if models_dir.exists():
        for file in models_dir.glob("*"):
            file.unlink()
    
    # Clean old checkpoints
    checkpoints_dir = Path(config.checkpoints_dir)
    if checkpoints_dir.exists():
        for file in checkpoints_dir.glob("*"):
            file.unlink()
    
    print("✅ Cleanup completed!")


def validate_dataset():
    """Quick dataset validation"""
    print("📊 Validating products.csv...")
    
    import pandas as pd
    
    csv_path = Path(config.data.csv_file)
    if not csv_path.exists():
        print(f"❌ Dataset not found: {csv_path}")
        return False
    
    df = pd.read_csv(csv_path)
    print(f"✅ Found {len(df)} products")
    
    # Check categories
    categories = df['category'].value_counts()
    print("📋 Category distribution:")
    for cat, count in categories.items():
        print(f"   {cat}: {count} products")
    
    return True


def run_streamlined_pipeline():
    """Run optimized pipeline for 180 products"""
    print("🚀 Starting Streamlined Pipeline for 180 Products...")
    
    start_time = time.time()
    
    # Initialize processors
    csv_processor = CSVProcessor()
    engine_builder = SearchEngineBuilder()
    
    # Phase 1: Data validation
    print("\n📋 Phase 1: Data Validation")
    if not csv_processor.validate_data():
        print("❌ Data validation failed")
        return False
    
    # Phase 2: Image download (optimized for 6 images max)
    print("\n📥 Phase 2: Image Download (max 6 per product)")
    if not csv_processor.download_images():
        print("❌ Image download failed")
        return False
    
    # Phase 3: Feature extraction
    print("\n🧠 Phase 3: CLIP Feature Extraction")
    if not engine_builder.extract_features():
        print("❌ Feature extraction failed")
        return False
    
    # Phase 4: Index building (Flat IP for exact search)
    print("\n🔍 Phase 4: Building Flat Index (Exact Search)")
    if not engine_builder.build_index():
        print("❌ Index building failed")
        return False
    
    duration = time.time() - start_time
    print(f"\n🎉 Pipeline completed in {duration:.1f}s!")
    
    # Print summary
    print_summary()
    return True


def print_summary():
    """Print completion summary"""
    print("\n" + "="*60)
    print("🎉 180-PRODUCT IMAGE SEARCH SYSTEM READY")
    print("="*60)
    
    # Check files
    models_dir = Path(config.models_dir)
    if (models_dir / "product_index.faiss").exists():
        print(f"✅ FAISS Index: {models_dir}/product_index.faiss")
    
    if (models_dir / "product_metadata.pkl").exists():
        print(f"✅ Metadata: {models_dir}/product_metadata.pkl")
    
    cache_dir = Path(config.data.image_cache_dir)
    if cache_dir.exists():
        product_count = len(list(cache_dir.glob("product_*")))
        print(f"✅ Cached Images: {product_count} products")
    
    print("\n🚀 NEXT STEPS:")
    print("1. Start API: python main_api.py")
    print("2. Start UI: python app_streamlit.py") 
    print("3. Test search: Upload an image!")
    print("="*60)


def main():
    """Main execution"""
    print("🔍 180-PRODUCT IMAGE SEARCH - QUICK SETUP")
    print("=" * 50)
    
    # Step 1: Validate dataset
    if not validate_dataset():
        return
    
    # Step 2: Clean previous data
    clean_previous_data()
    
    # Step 3: Run pipeline
    if run_streamlined_pipeline():
        print("\n✅ Setup completed successfully!")
    else:
        print("\n❌ Setup failed!")


if __name__ == "__main__":
    main() 