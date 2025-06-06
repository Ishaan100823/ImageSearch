"""
Enhanced Multi-Model API for Image Search System
Serves multiple specialized models with clean API paths
"""
import os
import io
import pickle
import time
import numpy as np
from typing import Dict, List, Optional
from pathlib import Path

import faiss
import torch
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from core.model_config import MODELS, get_model_config, get_model_paths
from pipeline.build_engine import CLIPFeatureExtractor
from utils.logging_utils import logger


class SearchResult(BaseModel):
    """Single search result"""
    product_id: str
    title: str
    category_from_db: str
    similarity_score: float
    image_url: str


class SearchResponse(BaseModel):
    """API response for search results"""
    model_used: str
    model_name: str
    total_results: int
    query_time_ms: float
    results: List[SearchResult]


class ModelInfo(BaseModel):
    """Information about a model"""
    key: str
    name: str
    description: str
    status: str
    categories: List[str]
    total_products: int
    api_path: str


class MultiModelSearchEngine:
    """Search engine that manages multiple specialized models"""
    
    def __init__(self):
        self.models: Dict[str, Dict] = {}
        self.feature_extractor = CLIPFeatureExtractor()
        self.initialized = False
    
    async def initialize(self):
        """Initialize the feature extractor and load all available models"""
        try:
            logger.info("🚀 Initializing Enhanced Multi-Model Search Engine")
            
            # Initialize CLIP model
            success = self.feature_extractor.initialize_model()
            if not success:
                raise Exception("Failed to initialize CLIP model")
            
            # Load all available trained models
            await self._load_all_models()
            
            self.initialized = True
            logger.success("✅ Enhanced Search Engine initialized successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize search engine: {str(e)}")
            raise
    
    async def _load_all_models(self):
        """Load all trained models into memory"""
        for model_key in MODELS.keys():
            try:
                await self._load_single_model(model_key)
            except Exception as e:
                logger.warning(f"⚠️ Failed to load model '{model_key}': {str(e)}")
    
    async def _load_single_model(self, model_key: str):
        """Load a single model's index and metadata"""
        model_config = get_model_config(model_key)
        model_paths = get_model_paths(model_key)
        
        index_path = Path(model_paths["index_file"])
        metadata_path = Path(model_paths["metadata_file"])
        
        if not (index_path.exists() and metadata_path.exists()):
            logger.info(f"⏭️ Model '{model_key}' not trained yet, skipping")
            return
        
        logger.info(f"📚 Loading model: {model_config['name']}")
        
        # Load FAISS index
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        with open(metadata_path, 'rb') as f:
            metadata = pickle.load(f)
        
        self.models[model_key] = {
            'index': index,
            'metadata': metadata,
            'config': model_config,
            'paths': model_paths,
            'total_products': len(metadata)
        }
        
        logger.success(f"✅ Loaded model '{model_key}': {len(metadata)} products")
    
    def get_available_models(self) -> List[ModelInfo]:
        """Get information about all available models"""
        model_info = []
        
        for model_key, model_config in MODELS.items():
            if model_key in self.models:
                status = "trained"
                total_products = self.models[model_key]['total_products']
            else:
                status = "not_trained"
                total_products = 0
            
            model_info.append(ModelInfo(
                key=model_key,
                name=model_config['name'],
                description=model_config['description'],
                status=status,
                categories=model_config['categories'],
                total_products=total_products,
                api_path=model_config['api_path']
            ))
        
        return model_info
    
    async def search(self, image_bytes: bytes, model_key: str, top_k: int = 5) -> SearchResponse:
        """Search for similar products using specified model"""
        if not self.initialized:
            raise HTTPException(status_code=503, detail="Search engine not initialized")
        
        if model_key not in self.models:
            available_models = list(self.models.keys())
            raise HTTPException(
                status_code=400, 
                detail=f"Model '{model_key}' not available. Available models: {available_models}"
            )
        
        start_time = time.time()
        
        try:
            # Extract features from query image
            query_features = await self._extract_query_features(image_bytes)
            
            # Search in specified model
            model_data = self.models[model_key]
            distances, indices = model_data['index'].search(query_features, top_k)
            
            # Format results
            results = []
            for i, (distance, idx) in enumerate(zip(distances[0], indices[0])):
                if idx < len(model_data['metadata']):
                    product = model_data['metadata'][idx]
                    results.append(SearchResult(
                        product_id=product['id'],
                        title=product['title'],
                        category_from_db=product['category'],
                        similarity_score=float(distance),
                        image_url=product['original_image_url']
                    ))
            
            query_time = (time.time() - start_time) * 1000
            
            return SearchResponse(
                model_used=model_key,
                model_name=model_data['config']['name'],
                total_results=len(results),
                query_time_ms=query_time,
                results=results
            )
            
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")
    
    async def search_all_models(self, image_bytes: bytes, top_k: int = 3) -> Dict[str, SearchResponse]:
        """Search across all available models"""
        results = {}
        
        for model_key in self.models.keys():
            try:
                result = await self.search(image_bytes, model_key, top_k)
                results[model_key] = result
            except Exception as e:
                logger.warning(f"Search failed for model '{model_key}': {str(e)}")
        
        return results
    
    async def _extract_query_features(self, image_bytes: bytes) -> np.ndarray:
        """Extract features from query image bytes"""
        try:
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            
            # Process with CLIP
            inputs = self.feature_extractor.processor(images=image, return_tensors="pt")
            inputs = {k: v.to(self.feature_extractor.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                features = self.feature_extractor.model.get_image_features(**inputs)
                features = torch.nn.functional.normalize(features, p=2, dim=-1)
            
            return features.cpu().numpy()
            
        except Exception as e:
            raise Exception(f"Failed to extract features: {str(e)}")


# Initialize the search engine
search_engine = MultiModelSearchEngine()

# Create FastAPI app
app = FastAPI(
    title="Enhanced Multi-Model Image Search API",
    description="Advanced image search with multiple specialized models and clean API paths",
    version="2.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
async def startup_event():
    """Initialize search engine on startup"""
    await search_engine.initialize()


@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about all available models"""
    return search_engine.get_available_models()


@app.post("/general/search", response_model=SearchResponse)
async def search_general_model(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20)
):
    """Search using the general product model"""
    contents = await file.read()
    return await search_engine.search(contents, "general", top_k)


@app.post("/shirts/search", response_model=SearchResponse)
async def search_shirts_model(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20)
):
    """Search using the specialized shirts model"""
    contents = await file.read()
    return await search_engine.search(contents, "shirts", top_k)


@app.post("/all/search", response_model=Dict[str, SearchResponse])
async def search_all_models(
    file: UploadFile = File(...),
    top_k: int = Query(3, ge=1, le=10)
):
    """Search across all available models"""
    contents = await file.read()
    return await search_engine.search_all_models(contents, top_k)


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if search_engine.initialized else "initializing",
        "available_models": len(search_engine.models),
        "total_products": sum(model['total_products'] for model in search_engine.models.values()),
        "models": {
            model_key: {
                "name": model_data['config']['name'],
                "products": model_data['total_products'],
                "api_path": model_data['config']['api_path']
            }
            for model_key, model_data in search_engine.models.items()
        }
    }


# Backward compatibility: support original API path
@app.post("/search_products/", response_model=SearchResponse)
async def search_products_legacy(
    file: UploadFile = File(...),
    top_k: int = Query(5, ge=1, le=20)
):
    """Legacy endpoint - uses general model for backward compatibility"""
    contents = await file.read()
    return await search_engine.search(contents, "general", top_k)


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        "enhanced_api:app",
        host="127.0.0.1",
        port=8000,
        reload=True
    ) 