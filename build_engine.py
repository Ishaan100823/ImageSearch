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
from transformers import CLIPModel, CLIPProcessor
from concurrent.futures import ThreadPoolExecutor, as_completed

from config import config
from utils.logging_utils import process_logger, create_progress_tracker


class CLIPFeatureExtractor:
    """CLIP-based feature extraction with batch processing"""
    
    def __init__(self):
        self.logger = process_logger.logger
        self.model = None
        self.processor = None
        self.device = None
        
    def initialize_model(self):
        """Initialize CLIP model and processor"""
        try:
            self.logger.info(f"🧠 Loading CLIP model: {config.model.clip_model}")
            
            # Determine device
            if config.model.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.model.device
            
            self.logger.info(f"🔧 Using device: {self.device}")
            
            # Load model and processor
            self.model = CLIPModel.from_pretrained(config.model.clip_model)
            self.processor = CLIPProcessor.from_pretrained(config.model.clip_model)
            
            # Move model to device
            self.model = self.model.to(self.device)
            self.model.eval()
            
            self.logger.info("✅ CLIP model loaded successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Failed to load CLIP model: {str(e)}")
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
            
        except Exception as e:
            self.logger.debug(f"Failed to extract features from {image_path}: {str(e)}")
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
                except Exception as e:
                    self.logger.debug(f"Failed to load image {path}: {str(e)}")
            
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
            
        except Exception as e:
            self.logger.debug(f"Failed to extract batch features: {str(e)}")
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
        self.logger = process_logger.logger
    
    def create_index(self, dimension: int, index_type: str = "flat") -> faiss.Index:
        """Create FAISS index based on configuration"""
        self.logger.info(f"🔍 Creating FAISS index: {index_type} (dim={dimension})")
        
        if index_type == "flat":
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
                self.logger.info("🏋️ Training FAISS index...")
                index.train(embeddings)
            
            # Add embeddings to index
            self.logger.info(f"📚 Adding {len(embeddings)} embeddings to index...")
            index.add(embeddings)
            
            self.logger.info(f"✅ FAISS index built successfully: {index.ntotal} vectors")
            return index
            
        except Exception as e:
            self.logger.error(f"❌ Failed to build FAISS index: {str(e)}")
            raise


class SearchEngineBuilder:
    """Main search engine builder orchestrator"""
    
    def __init__(self):
        self.feature_extractor = CLIPFeatureExtractor()
        self.index_builder = FAISSIndexBuilder()
        self.logger = process_logger.logger
    
    def extract_features(self) -> bool:
        """Phase 3: Extract CLIP features from images with resume capability"""
        if not config.processes["feature_extraction"].enabled:
            self.logger.info("⏭️ Skipping feature extraction")
            return True
        
        # Check if model initialization needed
        if self.feature_extractor.model is None:
            if not self.feature_extractor.initialize_model():
                return False
        
        process_config = config.processes["feature_extraction"]
        start_time = time.time()
        
        try:
            # Load processed products
            if not os.path.exists(config.data.processed_products_file):
                self.logger.error(f"❌ Processed products file not found: {config.data.processed_products_file}")
                return False
            
            with open(config.data.processed_products_file, 'r') as f:
                products = json.load(f)
            
            self.logger.info(f"📊 Processing features for {len(products)} products")
            
            # Create progress tracker
            tracker = create_progress_tracker(
                "feature_extraction",
                len(products),
                process_config.checkpoint_file,
                "Extracting CLIP features"
            )
            
            # Storage for results
            product_embeddings = []
            product_metadata = []
            
            # Process products in batches
            batch_size = process_config.batch_size
            
            for i in range(0, len(products), batch_size):
                batch_products = products[i:i + batch_size]
                batch_results = self._process_product_batch(batch_products, tracker)
                
                for result in batch_results:
                    if result is not None:
                        embedding, metadata = result
                        product_embeddings.append(embedding)
                        product_metadata.append(metadata)
                
                # Update progress
                tracker.set_postfix(
                    Processed=len(product_embeddings),
                    Batch=f"{i//batch_size + 1}/{(len(products) + batch_size - 1)//batch_size}"
                )
            
            tracker.close()
            
            if not product_embeddings:
                self.logger.error("❌ No valid embeddings extracted")
                return False
            
            # Convert to numpy array
            embeddings_array = np.array(product_embeddings, dtype=np.float32)
            
            # Save embeddings and metadata
            embeddings_file = config.models_dir / "product_embeddings.npy"
            metadata_file = config.index.metadata_file
            
            np.save(embeddings_file, embeddings_array)
            with open(metadata_file, 'wb') as f:
                pickle.dump(product_metadata, f)
            
            duration = time.time() - start_time
            stats = {
                "total_products": len(products),
                "valid_embeddings": len(product_embeddings),
                "embedding_dimension": embeddings_array.shape[1],
                "success_rate": len(product_embeddings) / len(products) * 100
            }
            
            process_logger.log_process_complete("feature_extraction", duration, stats)
            
            return True
            
        except Exception as e:
            process_logger.log_process_error("feature_extraction", e)
            return False
    
    def _process_product_batch(self, batch_products: List[Dict], tracker) -> List[Optional[tuple]]:
        """Process a batch of products for feature extraction"""
        results = []
        
        for product in batch_products:
            product_id = product['id']
            
            # Check if already completed
            if tracker.is_completed(product_id):
                # Load from checkpoint if needed
                results.append(None)
                continue
            
            try:
                # Extract features from all product images
                image_features = []
                for img_info in product['images']:
                    if os.path.exists(img_info['path']):
                        features = self.feature_extractor.extract_image_features(img_info['path'])
                        if features is not None:
                            image_features.append(features)
                
                if image_features:
                    # Aggregate features
                    aggregated_features = self.feature_extractor.aggregate_product_features(image_features)
                    
                    # Create metadata
                    metadata = {
                        'id': product_id,
                        'original_id': product['original_id'],
                        'title': product['title'],
                        'num_views': len(image_features),
                        'preview_path': product.get('preview_path'),
                        'preview_url': product.get('preview_url')
                    }
                    
                    tracker.update(product_id, {"num_features": len(image_features)})
                    results.append((aggregated_features, metadata))
                else:
                    tracker.update(product_id, {"error": "no_valid_features"})
                    results.append(None)
                    
            except Exception as e:
                tracker.update(product_id, {"error": str(e)})
                results.append(None)
        
        return results
    
    def build_index(self) -> bool:
        """Phase 4: Build FAISS search index with resume capability"""
        if not config.processes["index_building"].enabled:
            self.logger.info("⏭️ Skipping index building")
            return True
        
        process_config = config.processes["index_building"]
        start_time = time.time()
        
        try:
            # Load embeddings
            embeddings_file = config.models_dir / "product_embeddings.npy"
            if not embeddings_file.exists():
                self.logger.error(f"❌ Embeddings file not found: {embeddings_file}")
                return False
            
            embeddings = np.load(embeddings_file)
            self.logger.info(f"📊 Loaded {len(embeddings)} embeddings (dim={embeddings.shape[1]})")
            
            # Build index
            index = self.index_builder.build_index(embeddings, config.index.index_type)
            
            # Save index
            faiss.write_index(index, config.index.index_file)
            self.logger.info(f"💾 FAISS index saved to: {config.index.index_file}")
            
            # Save checkpoint
            checkpoint_data = {
                "index_built": True,
                "num_vectors": int(index.ntotal),
                "index_type": config.index.index_type,
                "timestamp": time.time()
            }
            
            with open(process_config.checkpoint_file, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            duration = time.time() - start_time
            stats = {
                "num_vectors": int(index.ntotal),
                "index_type": config.index.index_type,
                "dimension": embeddings.shape[1]
            }
            
            process_logger.log_process_complete("index_building", duration, stats)
            
            return True
            
        except Exception as e:
            process_logger.log_process_error("index_building", e)
            return False
    
    def build_all(self) -> bool:
        """Run complete engine building pipeline"""
        self.logger.info("🚀 Starting search engine building pipeline...")
        
        # Phase 3: Feature extraction
        if not self.extract_features():
            return False
        
        # Phase 4: Index building
        if not self.build_index():
            return False
        
        self.logger.info("✅ Search engine building completed successfully!")
        return True


def main():
    """Main function for standalone execution"""
    from utils.logging_utils import log_system_requirements
    
    # Log system info
    log_system_requirements()
    
    # Initialize builder
    builder = SearchEngineBuilder()
    
    # Build engine
    success = builder.build_all()
    
    if success:
        print("\n✅ Search engine building completed successfully!")
        print(f"🔍 FAISS index saved to: {config.index.index_file}")
        print(f"📊 Metadata saved to: {config.index.metadata_file}")
    else:
        print("\n❌ Search engine building failed!")
        exit(1)


if __name__ == "__main__":
    main() 