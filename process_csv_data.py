"""
CSV Data Processing Module for Image Search System
Handles data validation, image downloading, and caching with resume capability
"""
import json
import os
import ast
import requests
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
import pandas as pd
from PIL import Image
import time
import re

from config import config
from utils.logging_utils import process_logger, create_progress_tracker


class DataValidator:
    """Validates CSV data and structure"""
    
    def __init__(self):
        self.required_columns = ['shopify_product_id', 'title', 'images', 'preview_image']
        self.logger = process_logger.logger
    
    def validate_csv_structure(self, csv_path: str) -> Tuple[bool, Dict]:
        """Validate CSV file structure and contents"""
        try:
            # Check file exists
            if not os.path.exists(csv_path):
                return False, {"error": "CSV file not found", "path": csv_path}
            
            # Load CSV
            df = pd.read_csv(csv_path, encoding='utf-8')
            
            # Check required columns
            missing_cols = [col for col in self.required_columns if col not in df.columns]
            if missing_cols:
                return False, {"error": "Missing required columns", "missing": missing_cols}
            
            # Check data quality
            stats = {
                "total_rows": len(df),
                "columns": list(df.columns),
                "missing_data": {}
            }
            
            for col in self.required_columns:
                missing_count = df[col].isna().sum()
                stats["missing_data"][col] = int(missing_count)  # Convert numpy.int64 to Python int
            
            # Sample validation
            sample_errors = []
            for idx, row in df.head(5).iterrows():
                try:
                    self._validate_images_array(row['images'])
                except Exception as e:
                    sample_errors.append(f"Row {idx}: {str(e)}")
            
            if sample_errors:
                stats["sample_errors"] = sample_errors
            
            self.logger.info(f"✅ CSV validation passed: {stats['total_rows']} products found")
            return True, stats
            
        except Exception as e:
            return False, {"error": str(e)}
    
    def _validate_images_array(self, images_str: str) -> List[str]:
        """Parse and validate images JSON array"""
        try:
            # First try normal JSON parsing
            if images_str.startswith('['):
                try:
                    images = json.loads(images_str)
                    # If successful, validate and return
                    valid_urls = []
                    for url in images:
                        if isinstance(url, str) and url.strip():
                            valid_urls.append(url.strip())
                    return valid_urls
                except json.JSONDecodeError:
                    # Fall back to fixing malformed JSON
                    pass
            
            # Handle malformed JSON arrays with missing commas
            if images_str.startswith('[') and images_str.endswith(']'):
                # Fix the malformed JSON by adding commas between quoted strings
                fixed_str = re.sub(r'"\s+"', '", "', images_str)
                try:
                    images = json.loads(fixed_str)
                    valid_urls = []
                    for url in images:
                        if isinstance(url, str) and url.strip():
                            valid_urls.append(url.strip())
                    return valid_urls
                except json.JSONDecodeError:
                    pass
            
            # Try ast.literal_eval as fallback
            images = ast.literal_eval(images_str)
            
            # Ensure all items are valid URLs
            valid_urls = []
            for url in images:
                if isinstance(url, str) and url.strip():
                    valid_urls.append(url.strip())
            
            return valid_urls
        except (json.JSONDecodeError, ValueError, SyntaxError) as e:
            raise ValueError(f"Failed to parse images array: {str(e)}")


class ImageDownloader:
    """Downloads and caches product images with resume capability"""
    
    def __init__(self):
        self.logger = process_logger.logger
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def download_image(self, url: str, save_path: str, timeout: int = 10) -> bool:
        """Download single image with validation"""
        try:
            response = self.session.get(url, timeout=timeout, stream=True)
            response.raise_for_status()
            
            # Validate image before saving
            img_data = BytesIO(response.content)
            img = Image.open(img_data)
            img.verify()  # Verify it's a valid image
            
            # Reset stream and save
            img_data.seek(0)
            img = Image.open(img_data)
            img = img.convert('RGB')  # Ensure consistent format
            
            # Create directory if needed
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            
            # Save image
            img.save(save_path, 'JPEG', quality=85)
            
            # Validate saved image
            return self._validate_saved_image(save_path)
            
        except Exception as e:
            self.logger.debug(f"Failed to download {url}: {str(e)}")
            return False
    
    def _validate_saved_image(self, image_path: str) -> bool:
        """Validate downloaded image quality"""
        try:
            with Image.open(image_path) as img:
                # Check minimum size requirements
                if img.size[0] < config.data.min_image_size[0] or img.size[1] < config.data.min_image_size[1]:
                    os.remove(image_path)  # Remove undersized image
                    return False
                
                # Verify image integrity
                img.verify()
                return True
                
        except Exception:
            # Remove corrupted image
            if os.path.exists(image_path):
                os.remove(image_path)
            return False
    
    def download_product_images(self, product_id: str, images_urls: List[str], 
                               preview_url: Optional[str] = None) -> Dict:
        """Download all images for a single product"""
        # Clean product ID for safe filename
        clean_id = str(product_id).replace(',', '').replace('/', '_')
        product_dir = os.path.join(config.data.image_cache_dir, f'product_{clean_id}')
        os.makedirs(product_dir, exist_ok=True)
        
        downloaded_images = []
        
        # Download main product images (limit to max_images_per_product)
        for i, url in enumerate(images_urls[:config.data.max_images_per_product]):
            # Generate safe filename
            ext = self._get_image_extension(url)
            local_path = os.path.join(product_dir, f'view_{i+1}.{ext}')
            
            if self.download_image(url, local_path, config.data.download_timeout):
                downloaded_images.append({
                    'path': local_path,
                    'url': url,
                    'view_index': i+1
                })
        
        # Download preview image separately
        preview_path = None
        if preview_url:
            ext = self._get_image_extension(preview_url)
            preview_path = os.path.join(product_dir, f'preview.{ext}')
            if not self.download_image(preview_url, preview_path, config.data.download_timeout):
                preview_path = None
        
        return {
            'id': clean_id,
            'original_id': product_id,
            'images': downloaded_images,
            'preview_path': preview_path,
            'num_views': len(downloaded_images),
            'preview_url': preview_url
        }
    
    def _get_image_extension(self, url: str) -> str:
        """Extract image extension from URL"""
        ext = url.split('.')[-1].split('?')[0][:4].lower()
        if ext not in config.data.image_formats:
            ext = 'jpg'
        return ext


class CSVProcessor:
    """Main CSV processing orchestrator"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.downloader = ImageDownloader()
        self.logger = process_logger.logger
    
    def validate_data(self) -> bool:
        """Phase 1: Validate CSV data"""
        if not config.processes["data_validation"].enabled:
            self.logger.info("⏭️ Skipping data validation")
            return True
        
        process_config = config.processes["data_validation"]
        start_time = time.time()
        
        try:
            self.logger.info("🔍 Starting data validation...")
            
            # Validate CSV structure
            is_valid, validation_result = self.validator.validate_csv_structure(config.data.csv_file)
            
            if not is_valid:
                self.logger.error(f"❌ Data validation failed: {validation_result}")
                return False
            
            # Save validation results
            checkpoint_data = {
                "validation_passed": True,
                "validation_stats": validation_result,
                "timestamp": time.time()
            }
            
            with open(process_config.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            duration = time.time() - start_time
            process_logger.log_process_complete("data_validation", duration, validation_result)
            
            return True
            
        except Exception as e:
            process_logger.log_process_error("data_validation", e)
            return False
    
    def download_images(self) -> bool:
        """Phase 2: Download product images with resume capability"""
        if not config.processes["image_download"].enabled:
            self.logger.info("⏭️ Skipping image download")
            return True
        
        process_config = config.processes["image_download"]
        start_time = time.time()
        
        try:
            # Load CSV data
            df = pd.read_csv(config.data.csv_file, encoding='utf-8')
            
            # Create progress tracker
            tracker = create_progress_tracker(
                "image_download",
                len(df),
                process_config.checkpoint_file,
                "Downloading product images"
            )
            
            successful_downloads = []
            failed_downloads = []
            
            def process_product_row(row_data):
                """Process single product row"""
                idx, row = row_data
                product_id = str(row['shopify_product_id'])
                
                # Check if already completed
                if tracker.is_completed(product_id):
                    return None
                
                try:
                    # Parse images
                    images_urls = self.validator._validate_images_array(row['images'])
                    preview_url = row.get('preview_image')
                    
                    # Download images
                    result = self.downloader.download_product_images(
                        product_id, images_urls, preview_url
                    )
                    
                    if result['images']:  # At least one image downloaded successfully
                        result['title'] = row['title']
                        tracker.update(product_id, {"downloaded_images": len(result['images'])})
                        return ("success", result)
                    else:
                        tracker.update(product_id, {"error": "no_images_downloaded"})
                        return ("failed", {"id": product_id, "reason": "no_images_downloaded"})
                        
                except Exception as e:
                    tracker.update(product_id, {"error": str(e)})
                    return ("failed", {"id": product_id, "reason": str(e)})
            
            # Process with ThreadPoolExecutor
            with ThreadPoolExecutor(max_workers=process_config.max_workers) as executor:
                # Submit all tasks
                future_to_product = {
                    executor.submit(process_product_row, (idx, row)): (idx, row)
                    for idx, row in df.iterrows()
                    if not tracker.is_completed(str(row['shopify_product_id']))
                }
                
                # Process results as they complete
                for future in as_completed(future_to_product):
                    result = future.result()
                    if result:
                        status, data = result
                        if status == "success":
                            successful_downloads.append(data)
                            tracker.set_postfix(
                                Success=len(successful_downloads),
                                Failed=len(failed_downloads)
                            )
                        else:
                            failed_downloads.append(data)
            
            # Close tracker and save results
            tracker.close()
            
            # Save processed products
            if successful_downloads:
                with open(config.data.processed_products_file, 'w') as f:
                    json.dump(successful_downloads, f, indent=2)
            
            duration = time.time() - start_time
            stats = {
                "total_products": len(df),
                "successful_downloads": len(successful_downloads),
                "failed_downloads": len(failed_downloads),
                "success_rate": len(successful_downloads) / len(df) * 100
            }
            
            process_logger.log_process_complete("image_download", duration, stats)
            
            if len(successful_downloads) == 0:
                self.logger.error("❌ No products were successfully processed")
                return False
            
            return True
            
        except Exception as e:
            process_logger.log_process_error("image_download", e)
            return False
    
    def process_all(self) -> bool:
        """Run complete CSV processing pipeline"""
        self.logger.info("🚀 Starting CSV data processing pipeline...")
        
        # Phase 1: Data validation
        if not self.validate_data():
            return False
        
        # Phase 2: Image download
        if not self.download_images():
            return False
        
        self.logger.info("✅ CSV processing pipeline completed successfully!")
        return True


def main():
    """Main function for standalone execution"""
    from utils.logging_utils import log_system_requirements
    
    # Log system info
    log_system_requirements()
    
    # Initialize processor
    processor = CSVProcessor()
    
    # Process data
    success = processor.process_all()
    
    if success:
        print("\n✅ CSV processing completed successfully!")
        print(f"📁 Processed products saved to: {config.data.processed_products_file}")
        print(f"🖼️ Images cached in: {config.data.image_cache_dir}")
    else:
        print("\n❌ CSV processing failed!")
        exit(1)


if __name__ == "__main__":
    main() 