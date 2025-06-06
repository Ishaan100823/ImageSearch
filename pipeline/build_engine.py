"""
Engine Building Module for Image Search System
Handles CLIP feature extraction and FAISS index creation with resume capability
"""
import json
import os
import pickle
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
import faiss
import pandas as pd
from transformers import CLIPModel, CLIPProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

from core.config import config
from utils.logging_utils import logger, create_progress


class CLIPFeatureExtractor:
    """CLIP-based feature extraction with batch processing"""
    
    def __init__(self):
        self.model = None
        self.processor = None
        self.device = None
        
    def initialize_model(self):
        """Initialize CLIP model and processor"""
        try:
            logger.info(f"🧠 Loading CLIP model: {config.model.clip_model}")
            
            # Determine device
            if config.model.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.model.device
            
            logger.info(f"🔧 Using device: {self.device}")
            
            # Load model and processor
            self.model = CLIPModel.from_pretrained(config.model.clip_model)
            self.processor = CLIPProcessor.from_pretrained(config.model.clip_model)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            logger.success("CLIP model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load CLIP model: {str(e)}")
            return False
    
    def extract_image_features(self, image_path: str) -> Optional[np.ndarray]:
        """Extract features from a single image"""
        try:
            # Load and preprocess image
            image = Image.open(image_path).convert("RGB")
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self.model.get_image_features(**inputs)
                # L2 normalize for cosine similarity
                image_features = F.normalize(image_features, p=2, dim=-1)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception:
            return None
    
    def extract_batch_features(self, image_paths: List[str]) -> List[Optional[np.ndarray]]:
        """Extract features from a batch of images"""
        try:
            # Load and preprocess all images
            images = []
            valid_indices = []
            
            for i, path in enumerate(image_paths):
                try:
                    image = Image.open(path).convert("RGB")
                    images.append(image)
                    valid_indices.append(i)
                except Exception:
                    pass
            
            if not images:
                return [None] * len(image_paths)
            
            # Process batch
            inputs = self.processor(images=images, return_tensors="pt", padding=True).to(self.device)
            
            # Extract features
            with torch.no_grad():
                batch_features = self.model.get_image_features(**inputs)
                # L2 normalize for cosine similarity
                batch_features = F.normalize(batch_features, p=2, dim=-1)
            
            # Map results back to original order
            results = [None] * len(image_paths)
            for i, valid_idx in enumerate(valid_indices):
                results[valid_idx] = batch_features[i].cpu().numpy()
            
            return results
            
        except Exception:
            return [None] * len(image_paths)
    
    def aggregate_product_features(self, image_features: List[np.ndarray]) -> np.ndarray:
        """Aggregate multiple image features for a product"""
        if not image_features:
            return np.zeros(config.model.embedding_dim, dtype=np.float32)
        
        # Simple average aggregation (can be enhanced with attention)
        stacked_features = np.stack(image_features)
        aggregated = np.mean(stacked_features, axis=0)
        
        # Renormalize
        norm = np.linalg.norm(aggregated)
        if norm > 0:
            aggregated = aggregated / norm
        
        return aggregated.astype(np.float32)


class FAISSIndexBuilder:
    """FAISS index builder with different index types"""
    
    def __init__(self):
        pass
    
    def create_index(self, dimension: int, index_type: str = "flat") -> faiss.Index:
        """Create FAISS index based on configuration"""
        logger.info(f"🔍 Creating FAISS index: {index_type} (dim={dimension})")
        
        if index_type == "flat" or index_type == "flat_ip":
            # Simple flat index for exact search
            index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        elif index_type == "ivf":
            # IVF index for approximate search
            n_centroids = min(100, dimension // 4)
            quantizer = faiss.IndexFlatIP(dimension)
            index = faiss.IndexIVFFlat(quantizer, dimension, n_centroids)
        elif index_type == "hnsw":
            # HNSW index for fast approximate search
            index = faiss.IndexHNSWFlat(dimension, 32)
            index.hnsw.efConstruction = 200
            index.hnsw.efSearch = 100
        else:
            raise ValueError(f"Unsupported index type: {index_type}")
        
        return index
    
    def build_index(self, embeddings: np.ndarray, index_type: str = "flat") -> faiss.Index:
        """Build and train FAISS index"""
        try:
            dimension = embeddings.shape[1]
            index = self.create_index(dimension, index_type)
            
            # Train index if needed
            if hasattr(index, 'train'):
                logger.info("🏋️ Training FAISS index...")
                index.train(embeddings)
            
            # Add embeddings to index
            logger.info(f"📚 Adding {len(embeddings)} embeddings to index...")
            index.add(embeddings)
            
            logger.success(f"FAISS index built successfully: {index.ntotal} vectors")
            return index
            
        except Exception as e:
            logger.error(f"Failed to build FAISS index: {str(e)}")
            raise


class SearchEngineBuilder:
    """Main search engine builder orchestrator"""
    
    def __init__(self):
        self.feature_extractor = CLIPFeatureExtractor()
        self.index_builder = FAISSIndexBuilder()
    
    def load_csv_metadata(self) -> Dict[str, Dict]:
        """Load product metadata from CSV file"""
        try:
            df = pd.read_csv(config.data.csv_file, encoding='utf-8')
            metadata_dict = {}
            
            for _, row in df.iterrows():
                product_id = str(row['shopify_product_id']).replace(',', '')
                metadata_dict[product_id] = {
                    'id': product_id,
                    'title': row.get('title', 'Unknown Product'),
                    'category': row.get('category', 'Unknown'),
                    'preview_image': row.get('preview_image', ''),
                    'images': row.get('images', '[]')
                }
            
            logger.info(f"Loaded metadata for {len(metadata_dict)} products from CSV")
            return metadata_dict
            
        except Exception as e:
            logger.error(f"Failed to load CSV metadata: {str(e)}")
            return {}
    
    def extract_features(self) -> bool:
        """Phase 3: Extract CLIP features with smart resume"""
        if not config.processes["feature_extraction"].enabled:
            logger.info("⏭️ Skipping feature extraction")
            return True
        
        # Check if model initialization needed
        if self.feature_extractor.model is None:
            if not self.feature_extractor.initialize_model():
                return False
        
        logger.stage_start("Feature Extraction")
        start_time = time.time()
        
        try:
            # Load CSV metadata for enriching product information
            csv_metadata = self.load_csv_metadata()
            
            # Get all product directories
            cache_dir = Path(config.data.image_cache_dir)
            if not cache_dir.exists():
                logger.error("Image cache directory not found")
                return False
            
            product_dirs = [d for d in cache_dir.iterdir() if d.is_dir() and d.name.startswith('product_')]
            
            # Create progress tracker
            checkpoint_file = config.processes["feature_extraction"].checkpoint_file
            progress = create_progress("Feature Extraction", len(product_dirs), checkpoint_file)
            
            embeddings = []
            metadata = []
            
            for product_dir in product_dirs:
                product_id = product_dir.name.replace('product_', '')
                
                # SKIP IF ALREADY DONE - NO WASTE
                if progress.skip_if_done(product_id):
                    continue
                
                try:
                    # Get all image files for this product
                    image_files = []
                    for ext in ['jpg', 'jpeg', 'png']:
                        image_files.extend(product_dir.glob(f'*.{ext}'))
                    
                    if not image_files:
                        progress.mark_complete(product_id)
                        continue
                    
                    # Extract features from each image
                    image_features = []
                    for img_path in image_files:
                        features = self.feature_extractor.extract_image_features(str(img_path))
                        if features is not None:
                            image_features.append(features)
                    
                    if image_features:
                        # Aggregate multiple images into single product embedding
                        product_embedding = self.feature_extractor.aggregate_product_features(image_features)
                        embeddings.append(product_embedding)
                        
                        # Get CSV metadata for this product
                        csv_data = csv_metadata.get(product_id, {})
                        
                        # Store complete metadata
                        metadata.append({
                            'id': product_id,
                            'title': csv_data.get('title', 'Unknown Product'),
                            'category': csv_data.get('category', 'Unknown'),
                            'preview_image': csv_data.get('preview_image', ''),
                            'num_images': len(image_features),
                            'local_images_dir': str(product_dir)
                        })
                    
                    progress.mark_complete(product_id)
                    
                    # Save progress periodically
                    if len(progress.checkpoint.completed_ids) % 10 == 0:
                        progress.save()
                
                except Exception as e:
                    logger.error(f"Failed to process product {product_id}: {str(e)}")
                    progress.mark_complete(product_id)  # Skip on error
            
            progress.finish()
            
            if not embeddings:
                logger.error("No features extracted")
                return False
            
            # Save embeddings and metadata
            embeddings_array = np.stack(embeddings)
            
            # Save to files
            np.save(config.index.embeddings_file, embeddings_array)
            
            with open(config.index.metadata_file, 'wb') as f:
                pickle.dump(metadata, f)
            
            duration = time.time() - start_time
            logger.stage_complete("Feature Extraction", duration)
            logger.info(f"Extracted features for {len(embeddings)} products")
            return True
            
        except Exception as e:
            logger.error(f"Feature extraction error: {str(e)}")
            return False
    
    def build_index(self) -> bool:
        """Phase 4: Build FAISS index with resume capability"""
        if not config.processes["index_building"].enabled:
            logger.info("⏭️ Skipping index building")
            return True
        
        # Check if already completed
        checkpoint_file = config.processes["index_building"].checkpoint_file
        progress = create_progress("Index Building", 1, checkpoint_file)
        
        if progress.skip_if_done("index_complete"):
            progress.finish()
            return True
        
        logger.stage_start("Index Building")
        start_time = time.time()
        
        try:
            # Load embeddings
            if not os.path.exists(config.index.embeddings_file):
                logger.error("Embeddings file not found")
                return False
            
            embeddings = np.load(config.index.embeddings_file)
            logger.info(f"Loaded {len(embeddings)} embeddings")
            
            # Build FAISS index
            index = self.index_builder.build_index(embeddings, config.index.index_type)
            
            # Save index
            faiss.write_index(index, config.index.index_file)
            
            progress.mark_complete("index_complete")
            progress.finish()
            
            duration = time.time() - start_time
            logger.stage_complete("Index Building", duration)
            return True
            
        except Exception as e:
            logger.error(f"Index building error: {str(e)}")
            return False
    
    def build_all(self) -> bool:
        """Run complete ML pipeline"""
        logger.info("🚀 Starting ML pipeline")
        
        # Phase 3: Feature extraction
        if not self.extract_features():
            logger.error("❌ Feature extraction failed")
            return False
        
        # Phase 4: Index building
        if not self.build_index():
            logger.error("❌ Index building failed")
            return False
        
        logger.success("✅ ML pipeline completed successfully")
        return True


def main():
    """Main entry point for engine building"""
    builder = SearchEngineBuilder()
    success = builder.build_all()
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main() 