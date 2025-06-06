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

from core.config import config
from utils.logging_utils import logger, create_progress


class DataValidator:
    """Validates CSV data and structure"""
    
    def __init__(self):
        self.required_columns = ['shopify_product_id', 'title', 'images', 'preview_image']
    
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
            
            logger.success(f"CSV validation passed: {stats['total_rows']} products found")
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
            
        except Exception:
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
            'product_id': product_id,
            'product_dir': product_dir,
            'downloaded_images': downloaded_images,
            'preview_path': preview_path,
            'success': len(downloaded_images) > 0
        }
    
    def _get_image_extension(self, url: str) -> str:
        """Get safe image extension from URL"""
        if '.webp' in url.lower():
            return 'jpg'  # Convert webp to jpg for consistency
        elif '.png' in url.lower():
            return 'jpg'  # Convert to jpg for consistency
        else:
            return 'jpg'  # Default to jpg


class CSVProcessor:
    """Main CSV processing orchestrator"""
    
    def __init__(self):
        self.validator = DataValidator()
        self.downloader = ImageDownloader()
    
    def validate_data(self) -> bool:
        """Phase 1: Data validation with NO WASTE"""
        if not config.processes["data_validation"].enabled:
            logger.info("⏭️ Skipping data validation")
            return True
        
        # Check if already completed
        checkpoint_file = config.processes["data_validation"].checkpoint_file
        progress = create_progress("Data Validation", 1, checkpoint_file)
        
        if progress.skip_if_done("validation_complete"):
            progress.finish()
            return True
        
        logger.stage_start("Data Validation")
        start_time = time.time()
        
        try:
            # Validate CSV structure
            success, stats = self.validator.validate_csv_structure(config.data.csv_file)
            
            if not success:
                logger.error(f"Data validation failed: {stats}")
                return False
            
            # Save validation results
            validation_data = {
                'validation_passed': True,
                'stats': stats,
                'timestamp': time.time()
            }
            
            with open(config.data.processed_products_file, 'w') as f:
                json.dump(validation_data, f, indent=2)
            
            progress.mark_complete("validation_complete")
            progress.finish()
            
            duration = time.time() - start_time
            logger.stage_complete("Data Validation", duration)
            return True
            
        except Exception as e:
            logger.error(f"Data validation error: {str(e)}")
            return False
    
    def download_images(self) -> bool:
        """Phase 2: Image download with smart resume"""
        if not config.processes["image_download"].enabled:
            logger.info("⏭️ Skipping image download")
            return True
        
        logger.stage_start("Image Download")
        start_time = time.time()
        
        try:
            # Load product data
            df = pd.read_csv(config.data.csv_file, encoding='utf-8')
            
            # Create progress tracker
            checkpoint_file = config.processes["image_download"].checkpoint_file
            progress = create_progress("Image Download", len(df), checkpoint_file)
            
            max_workers = config.processes["image_download"].max_workers
            
            def process_product_row(row_data):
                """Process single product with WASTE PREVENTION"""
                idx, row = row_data
                product_id = str(row['shopify_product_id']).replace(',', '')
                
                # SKIP IF ALREADY DONE - KEY RESUME FEATURE
                if progress.skip_if_done(product_id):
                    return None
                
                try:
                    # Parse image URLs
                    images_urls = self.validator._validate_images_array(row['images'])
                    preview_url = row.get('preview_image')
                    
                    # Download images
                    result = self.downloader.download_product_images(
                        product_id, images_urls, preview_url
                    )
                    
                    # Mark complete
                    progress.mark_complete(product_id)
                    return result
                    
                except Exception as e:
                    logger.error(f"Failed to process product {product_id}: {str(e)}")
                    # Still mark as complete to prevent infinite retries
                    progress.mark_complete(product_id)
                    return None
            
            # Process with thread pool
            successful_downloads = 0
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit all tasks
                future_to_product = {
                    executor.submit(process_product_row, (idx, row)): idx 
                    for idx, row in df.iterrows()
                }
                
                # Process completed tasks
                for future in as_completed(future_to_product):
                    result = future.result()
                    if result and result.get('success'):
                        successful_downloads += 1
                    
                    # Save progress periodically
                    if len(progress.checkpoint.completed_ids) % 10 == 0:
                        progress.save()
            
            progress.finish()
            
            duration = time.time() - start_time
            logger.stage_complete("Image Download", duration)
            logger.info(f"Successfully downloaded images for {successful_downloads} products")
            return True
            
        except Exception as e:
            logger.error(f"Image download error: {str(e)}")
            return False
    
    def process_all(self) -> bool:
        """Run both validation and download phases"""
        logger.info("🚀 Starting CSV data processing")
        
        # Phase 1: Validation
        if not self.validate_data():
            logger.error("❌ Data validation failed")
            return False
        
        # Phase 2: Image download
        if not self.download_images():
            logger.error("❌ Image download failed")
            return False
        
        logger.success("✅ CSV processing completed successfully")
        return True


def main():
    """Main entry point for CSV processing"""
    processor = CSVProcessor()
    success = processor.process_all()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main() 