"""
FastAPI Main Application for Image Search System
Clean, simple, focused on core functionality
"""
import os
import pickle
import numpy as np
import torch
import faiss
from pathlib import Path
from typing import Dict, List, Optional
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPModel, CLIPProcessor
import torch.nn.functional as F
from io import BytesIO

from config import config
from utils.logging_utils import logger


class ImageSearchEngine:
    """Clean image search engine with CLIP + FAISS"""
    
    def __init__(self):
        self.clip_model = None
        self.clip_processor = None
        self.faiss_index = None
        self.product_metadata = None
        self.device = None
        self.is_initialized = False
    
    def initialize(self) -> bool:
        """Initialize all models and data - called once at startup"""
        try:
            logger.info("🚀 Initializing search engine...")
            
            # Setup device
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            logger.info(f"Using device: {self.device}")
            
            # Load CLIP model
            logger.info("Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained(config.model.clip_model)
            self.clip_processor = CLIPProcessor.from_pretrained(config.model.clip_model)
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            
            # Load FAISS index
            logger.info("Loading FAISS index...")
            if not os.path.exists(config.index.index_file):
                raise FileNotFoundError("FAISS index not found")
            self.faiss_index = faiss.read_index(config.index.index_file)
            
            # Load metadata
            logger.info("Loading product metadata...")
            if not os.path.exists(config.index.metadata_file):
                raise FileNotFoundError("Product metadata not found")
            
            with open(config.index.metadata_file, 'rb') as f:
                self.product_metadata = pickle.load(f)
            
            self.is_initialized = True
            logger.success(f"✅ Search engine initialized successfully!")
            logger.info(f"  - Products: {len(self.product_metadata)}")
            logger.info(f"  - Index vectors: {self.faiss_index.ntotal}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize search engine: {str(e)}")
            return False
    
    def extract_query_features(self, image_bytes: bytes) -> Optional[np.ndarray]:
        """Extract features from query image"""
        try:
            # Load and preprocess image
            image = Image.open(BytesIO(image_bytes)).convert("RGB")
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # Extract features
            with torch.no_grad():
                features = self.clip_model.get_image_features(**inputs)
                features = F.normalize(features, p=2, dim=-1)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            logger.error(f"Feature extraction failed: {str(e)}")
            return None
    
    def search_similar_products(self, query_features: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar products using FAISS"""
        try:
            # Ensure query is 2D array for FAISS
            query_vector = query_features.reshape(1, -1).astype(np.float32)
            
            # Search in FAISS index
            similarities, indices = self.faiss_index.search(query_vector, top_k)
            
            # Format results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.product_metadata):
                    metadata = self.product_metadata[idx]
                    results.append({
                        'product_id': str(metadata['id']),
                        'title': metadata.get('title', 'Unknown Product'),
                        'category_from_db': metadata.get('category', 'Unknown'),
                        'similarity_score': float(similarity),
                        'rank': i + 1,
                        'preview_url': metadata.get('preview_image', ''),
                        'num_views': 1000 + (i * 100)  # Fake view count for demo
                    })
            
            return results
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []


# Global search engine instance - loaded once at startup
search_engine = ImageSearchEngine()

# FastAPI app
app = FastAPI(
    title="Image Search API",
    description="Clean image similarity search using CLIP + FAISS",
    version="1.0.0"
)

# Enable CORS for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup - NO WASTE"""
    if not search_engine.initialize():
        logger.error("❌ Failed to initialize search engine")
        raise RuntimeError("Search engine initialization failed")


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if not search_engine.is_initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "status": "healthy",
        "products": len(search_engine.product_metadata),
        "index_vectors": search_engine.faiss_index.ntotal
    }


@app.post("/search")
async def search_products(
    file: UploadFile = File(...),
    top_k: int = 5
):
    """
    Search for similar products using uploaded image
    Returns ranked list of similar products
    """
    # Check if service is ready
    if not search_engine.is_initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    # Validate file
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Invalid file type. Please upload an image.")
    
    # Validate top_k
    if top_k < 1 or top_k > 50:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 50")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        
        # Extract features from query image
        query_features = search_engine.extract_query_features(image_bytes)
        if query_features is None:
            raise HTTPException(status_code=400, detail="Failed to process image")
        
        # Search for similar products
        results = search_engine.search_similar_products(query_features, top_k)
        
        if not results:
            return {
                "message": "No similar products found",
                "total_results": 0,
                "query_time_ms": 0,
                "results": []
            }
        
        return {
            "message": f"Found {len(results)} similar products",
            "total_results": len(results),
            "query_time_ms": 0,  # We'll calculate this properly later
            "results": results
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search request failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")


@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    if not search_engine.is_initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "total_products": len(search_engine.product_metadata),
        "index_vectors": search_engine.faiss_index.ntotal,
        "model_device": search_engine.device,
        "index_type": "FAISS"
    }


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Starting FastAPI server...")
    uvicorn.run(
        "main_api:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info"
    ) 