"""
FastAPI Backend for Image Search System
Provides REST API for image similarity search
"""
import os
import pickle
import time
from typing import List, Dict, Any, Optional
from io import BytesIO
import numpy as np
import faiss
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import CLIPModel, CLIPProcessor

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import config
from utils.logging_utils import process_logger


# Response models
class SearchResult(BaseModel):
    product_id: str
    title: str
    similarity_score: float
    preview_url: Optional[str] = None
    num_views: int

class SearchResponse(BaseModel):
    query_time_ms: float
    total_results: int
    results: List[SearchResult]


class ImageSearchEngine:
    """Main search engine class"""
    
    def __init__(self):
        self.logger = process_logger.logger
        self.clip_model = None
        self.clip_processor = None
        self.faiss_index = None
        self.product_metadata = None
        self.device = None
        self.initialized = False
    
    async def initialize(self):
        """Initialize all models and indices"""
        try:
            self.logger.info("🚀 Initializing Image Search Engine...")
            
            # Determine device
            if config.model.device == "auto":
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = config.model.device
            
            self.logger.info(f"🔧 Using device: {self.device}")
            
            # Load CLIP model
            self.logger.info("🧠 Loading CLIP model...")
            self.clip_model = CLIPModel.from_pretrained(config.model.clip_model)
            self.clip_processor = CLIPProcessor.from_pretrained(config.model.clip_model)
            self.clip_model = self.clip_model.to(self.device)
            self.clip_model.eval()
            
            # Load FAISS index
            self.logger.info("🔍 Loading FAISS index...")
            if not os.path.exists(config.index.index_file):
                raise FileNotFoundError(f"FAISS index not found: {config.index.index_file}")
            
            self.faiss_index = faiss.read_index(config.index.index_file)
            self.logger.info(f"📚 Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            
            # Load product metadata
            self.logger.info("📊 Loading product metadata...")
            if not os.path.exists(config.index.metadata_file):
                raise FileNotFoundError(f"Metadata file not found: {config.index.metadata_file}")
            
            with open(config.index.metadata_file, 'rb') as f:
                self.product_metadata = pickle.load(f)
            
            self.logger.info(f"✅ Loaded metadata for {len(self.product_metadata)} products")
            
            self.initialized = True
            self.logger.info("🎉 Image Search Engine initialized successfully!")
            
        except Exception as e:
            self.logger.error(f"❌ Failed to initialize search engine: {str(e)}")
            raise
    
    def extract_query_features(self, image: Image.Image) -> np.ndarray:
        """Extract features from query image"""
        try:
            # Preprocess image
            inputs = self.clip_processor(images=image, return_tensors="pt").to(self.device)
            
            # Extract features
            with torch.no_grad():
                image_features = self.clip_model.get_image_features(**inputs)
                # L2 normalize for cosine similarity
                image_features = F.normalize(image_features, p=2, dim=-1)
            
            return image_features.cpu().numpy().flatten()
            
        except Exception as e:
            self.logger.error(f"Failed to extract query features: {str(e)}")
            raise
    
    def search_similar_products(self, query_features: np.ndarray, top_k: int = 5) -> List[Dict]:
        """Search for similar products using FAISS"""
        try:
            # Ensure query features are normalized and in correct format
            query_features = query_features.reshape(1, -1).astype(np.float32)
            
            # Search FAISS index
            similarities, indices = self.faiss_index.search(query_features, top_k)
            
            # Convert results
            results = []
            for i, (similarity, idx) in enumerate(zip(similarities[0], indices[0])):
                if idx < len(self.product_metadata):
                    metadata = self.product_metadata[idx]
                    
                    result = {
                        "product_id": metadata['id'],
                        "title": metadata['title'],
                        "similarity_score": float(similarity),  # Convert from float32
                        "preview_url": metadata.get('preview_url'),
                        "num_views": metadata.get('num_views', 0),
                        "rank": i + 1
                    }
                    results.append(result)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Failed to search similar products: {str(e)}")
            raise
    
    async def search_by_image(self, image: Image.Image, top_k: int = 5) -> List[Dict]:
        """Complete search pipeline for an uploaded image"""
        if not self.initialized:
            raise HTTPException(status_code=503, detail="Search engine not initialized")
        
        start_time = time.time()
        
        try:
            # Extract features from query image
            query_features = self.extract_query_features(image)
            
            # Search for similar products
            results = self.search_similar_products(query_features, top_k)
            
            query_time = (time.time() - start_time) * 1000  # Convert to ms
            
            self.logger.info(f"🔍 Search completed in {query_time:.1f}ms, found {len(results)} results")
            
            return {
                "query_time_ms": query_time,
                "total_results": len(results),
                "results": results
            }
            
        except Exception as e:
            self.logger.error(f"Search failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


# Global search engine instance
search_engine = ImageSearchEngine()

# FastAPI app
app = FastAPI(
    title="Image Search API",
    description="AI-powered image similarity search for products",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup"""
    await search_engine.initialize()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "Image Search API",
        "initialized": search_engine.initialized,
        "timestamp": time.time()
    }


@app.get("/health")
async def health_check():
    """Detailed health check"""
    if not search_engine.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "status": "healthy",
        "faiss_index_size": search_engine.faiss_index.ntotal,
        "metadata_count": len(search_engine.product_metadata),
        "device": search_engine.device,
        "model": config.model.clip_model
    }


@app.post("/search", response_model=SearchResponse)
async def search_products(
    file: UploadFile = File(..., description="Query image file"),
    top_k: int = Query(default=5, ge=1, le=20, description="Number of results to return")
):
    """
    Search for similar products using an uploaded image
    
    - **file**: Upload an image file (JPEG, PNG, WebP)
    - **top_k**: Number of similar products to return (1-20)
    """
    if not search_engine.initialized:
        raise HTTPException(status_code=503, detail="Search engine not initialized")
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and validate image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Check image size
        if image.size[0] < 50 or image.size[1] < 50:
            raise HTTPException(status_code=400, detail="Image too small (minimum 50x50)")
        
        # Perform search
        search_results = await search_engine.search_by_image(image, top_k)
        
        # Convert to response model
        results = [
            SearchResult(
                product_id=result["product_id"],
                title=result["title"],
                similarity_score=result["similarity_score"],
                preview_url=result.get("preview_url"),
                num_views=result.get("num_views", 0)
            )
            for result in search_results["results"]
        ]
        
        return SearchResponse(
            query_time_ms=search_results["query_time_ms"],
            total_results=search_results["total_results"],
            results=results
        )
        
    except HTTPException:
        raise
    except Exception as e:
        process_logger.log_process_error("image_search", e, {"filename": file.filename})
        raise HTTPException(status_code=500, detail=f"Processing error: {str(e)}")


@app.get("/stats")
async def get_stats():
    """Get search engine statistics"""
    if not search_engine.initialized:
        raise HTTPException(status_code=503, detail="Service not initialized")
    
    return {
        "total_products": len(search_engine.product_metadata),
        "index_size": search_engine.faiss_index.ntotal,
        "embedding_dimension": search_engine.faiss_index.d,
        "device": search_engine.device,
        "model": config.model.clip_model,
        "index_type": config.index.index_type
    }


if __name__ == "__main__":
    import uvicorn
    
    print("🚀 Starting Image Search API...")
    print(f"📍 API will be available at: http://{config.api.host}:{config.api.port}")
    print(f"📖 API docs will be available at: http://{config.api.host}:{config.api.port}/docs")
    
    uvicorn.run(
        "main_api:app",
        host=config.api.host,
        port=config.api.port,
        reload=config.api.reload,
        log_level="info"
    ) 