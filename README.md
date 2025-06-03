# 🔍 Image Search System

An advanced AI-powered image similarity search system built with CLIP and FAISS. Upload an image and find visually similar products from a database with sophisticated multi-image aggregation and robust pipeline architecture.

## ✨ Features

- **🧠 AI-Powered Search**: Uses OpenAI CLIP for deep visual understanding and feature extraction
- **⚡ Fast Similarity Search**: FAISS indexing for sub-second search results (< 200ms)
- **📊 Advanced Pipeline**: 4-phase processing with comprehensive progress tracking
- **🔄 Resume Capability**: Sophisticated checkpoint system to resume from any interruption
- **🛠️ Configurable Architecture**: Easy process control and stage skipping
- **🖼️ Multi-Image Support**: Aggregates features from multiple product views (up to 10 images)
- **🖥️ Modern UI**: Clean Streamlit interface with visual product cards
- **🚀 Production Ready**: FastAPI backend with comprehensive logging and error handling
- **📈 Scalable**: Supports large product catalogs with configurable index types

## 🏗️ Complete Pipeline Architecture

```
📁 ImageSearch/
├── 📄 check-shirts.csv             # Product data with multi-image URLs (your data file)
├── 🔧 config.py                    # Configuration and process control
├── 📊 main.py                      # Main pipeline orchestrator with argument parsing
├── 🔍 process_csv_data.py          # Phase 1-2: Data validation and image downloading
├── 🧠 build_engine.py              # Phase 3-4: CLIP feature extraction and FAISS indexing
├── 🚀 main_api.py                  # FastAPI backend server with search endpoints
├── 🖥️ app_streamlit.py             # Streamlit frontend interface
├── 🛠️ setup.py                     # Automated setup script with PyTorch handling
├── 🧪 test_setup.py                # Installation validation script
├── 📁 utils/
│   └── 📝 logging_utils.py         # Advanced progress tracking and logging
├── 📁 data/                        # Generated during pipeline
│   ├── 📄 processed_products.json  # Cleaned and validated product data
│   └── 📁 image_cache/             # Downloaded and cached product images
│       └── 📁 product_*/           # Organized by product ID
├── 📁 models/                      # Generated ML artifacts
│   ├── 🧠 product_embeddings.npy   # CLIP feature vectors (not used in current version)
│   ├── 🔍 product_index.faiss      # FAISS similarity index
│   └── 📊 product_metadata.pkl     # Product metadata with IDs and titles
├── 📁 checkpoints/                 # Resume points for each phase
│   ├── 📄 data_validation.json
│   ├── 📄 image_download.json
│   ├── 📄 feature_extraction.json
│   └── 📄 index_building.json
└── 📁 logs/                        # Comprehensive system logs
```

## 🔄 4-Phase Pipeline Detailed Flow

### **Phase 1: Data Validation** 📋
```python
# Located in: process_csv_data.py:DataValidator
```
- **Input**: Raw CSV file (`check-shirts.csv`)
- **Process**: 
  - Validates CSV structure and required columns: `shopify_product_id`, `title`, `images`, `preview_image`
  - Handles malformed JSON arrays in `images` column (common in Shopify exports)
  - Uses robust parsing with `json.loads()` and `ast.literal_eval()` fallbacks
  - Validates sample rows and reports data quality issues
- **Output**: Validation statistics and cleaned data structure
- **Resume**: Saves validation results to `checkpoints/data_validation.json`

### **Phase 2: Image Download & Caching** 📥
```python
# Located in: process_csv_data.py:ImageDownloader
```
- **Input**: Validated product data with image URLs
- **Process**:
  - Downloads images with concurrent processing (8 workers by default)
  - Validates each image: format, size (min 100x100), integrity
  - Converts all images to RGB format for consistency
  - Organizes cache: `data/image_cache/product_{id}/view_{n}.jpg`
  - Implements timeout (10s) and retry logic
- **Output**: Local image cache organized by product ID
- **Resume**: Tracks downloaded images in `checkpoints/image_download.json`

### **Phase 3: CLIP Feature Extraction** 🧠
```python
# Located in: build_engine.py:CLIPFeatureExtractor
```
- **Input**: Cached product images
- **Process**:
  - Loads OpenAI CLIP model: `openai/clip-vit-base-patch32`
  - Processes images in batches (16 by default) for efficiency
  - Extracts 512-dimensional feature vectors
  - L2 normalizes features for cosine similarity
  - **Multi-Image Aggregation**: Averages features from multiple product views
  - Auto-detects device (CUDA/CPU) and optimizes accordingly
- **Output**: Normalized feature embeddings (512-dim vectors per product)
- **Resume**: Saves progress in `checkpoints/feature_extraction.json`

### **Phase 4: FAISS Index Building** 🔍
```python
# Located in: build_engine.py:FAISSIndexBuilder
```
- **Input**: Product embeddings and metadata
- **Process**:
  - Creates FAISS index (Flat IP for exact cosine similarity by default)
  - Supports multiple index types: Flat, IVF, HNSW
  - Trains index if required (for approximate search methods)
  - Adds all product embeddings to index for fast similarity search
- **Output**: 
  - `models/product_index.faiss`: Searchable FAISS index
  - `models/product_metadata.pkl`: Product metadata for result mapping
- **Resume**: Saves completion status in `checkpoints/index_building.json`

## ⚡ Quick Start

### Method 1: Automated Setup (Recommended)

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Run automated setup (handles PyTorch installation based on your system)
python setup.py

# 3. Validate installation
python test_setup.py

# 4. Run complete pipeline (processes your check-shirts.csv file)
python main.py
```

### Method 2: Manual Installation

```bash
# 1. Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# 2. Install PyTorch (choose based on your system)
# For macOS/CPU
pip install torch torchvision torchaudio

# For Linux with CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only systems
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# 3. Install remaining dependencies
pip install -r requirements.txt

# 4. Validate setup
python test_setup.py

# 5. Run pipeline
python main.py
```

### 3. Start Search System

```bash
# Option 1: Start both servers automatically
python main.py --servers

# Option 2: Start manually (recommended for production)
# Terminal 1: API Backend
python main_api.py

# Terminal 2: Frontend UI  
python app_streamlit.py
```

**Access Points:**
- **🖥️ UI Interface**: http://localhost:7860 (Streamlit with visual cards)
- **📖 API Documentation**: http://localhost:8000/docs (FastAPI auto-generated docs)
- **🔍 API Health Check**: http://localhost:8000/health

## 📊 Data Format & Requirements

Your CSV file should follow this exact structure:

```csv
shopify_product_id,title,images,preview_image
"8,526,513,733,794","Grey Checks Slim Fit Shirt","[""https://img1.jpg"" ""https://img2.jpg"" ""https://img3.jpg""]","https://preview.jpg"
```

**Column Details:**
- **`shopify_product_id`**: Unique product identifier (can contain commas)
- **`title`**: Product name/description (used in search results)
- **`images`**: JSON array of image URLs (multiple product views, up to 10)
- **`preview_image`**: Primary product image URL (used for display)

**Important Notes:**
- The system handles malformed JSON arrays (missing commas between URLs)
- Images are validated for format, size (min 100x100px), and integrity
- All images are converted to RGB format for consistency
- Products with invalid/inaccessible images are logged but processing continues

## 🎛️ Advanced Configuration & Control

### Quick Configuration Presets

```bash
# Resume from different stages (skip completed work)
python main.py --resume-images     # Skip data processing, start from feature extraction
python main.py --resume-features   # Skip to index building only
python main.py --servers           # Just run API/UI servers

# Rebuild modes
python main.py --full-rebuild      # Force rebuild everything from scratch
python main.py --quick-test        # Skip heavy processes for testing

# Selective processing
python main.py --data-only         # Only data validation and image download
python main.py --ml-only          # Only feature extraction and index building
```

### Manual Process Control

```bash
# Skip specific processes
python main.py --skip image_download feature_extraction

# Force rebuild specific processes
python main.py --force feature_extraction index_building

# Process individual stages
python process_csv_data.py  # Data processing only
python build_engine.py      # ML pipeline only
```

### Configuration Customization

Edit `config.py` for fine-tuned control:

```python
# Process Control
config.skip_process("image_download")           # Skip if images already cached
config.force_rebuild_process("feature_extraction")  # Force rebuild features

# Performance Tuning
config.processes["feature_extraction"].batch_size = 32     # Larger batches for GPU
config.processes["image_download"].max_workers = 16        # More concurrent downloads
config.data.max_images_per_product = 5                     # Limit images per product

# Model Configuration
config.model.clip_model = "openai/clip-vit-base-patch32"   # CLIP model variant
config.model.device = "cuda"                               # Force GPU usage
config.index.index_type = "ivf"                           # Use approximate search
```

## 🔧 Troubleshooting & Performance

### Common Issues & Solutions

**❌ "FAISS index not found"**
```bash
# Build the search index first
python main.py --ml-only
# Or run just the index building phase
python build_engine.py
```

**❌ "API not available" / Connection errors**
```bash
# Check if API server is running
curl http://localhost:8000/health

# Start API server manually
python main_api.py

# Check for port conflicts
lsof -i :8000  # macOS/Linux
netstat -an | findstr :8000  # Windows
```

**❌ "No images downloaded" / Image download failures**
```bash
# Check internet connection and CSV URLs
python main.py --force image_download

# Increase timeout for slow connections
# Edit config.py: config.data.download_timeout = 30
```

**❌ "CUDA out of memory" / GPU issues**
```python
# In config.py, reduce batch size or force CPU
config.processes["feature_extraction"].batch_size = 8
config.model.device = "cpu"  # Force CPU processing
```

**❌ CSV parsing errors**
- The system handles malformed JSON arrays automatically
- Check that your CSV has all required columns
- Ensure image URLs are accessible (test a few manually)

### Performance Optimization

**For faster processing:**
```python
# Increase batch sizes (if you have sufficient GPU memory)
config.processes["feature_extraction"].batch_size = 32
config.processes["image_download"].max_workers = 16

# Use GPU acceleration
config.model.device = "cuda"

# Use approximate search for large catalogs
config.index.index_type = "ivf"  # or "hnsw"
```

**For lower memory usage:**
```python
# Reduce batch sizes
config.processes["feature_extraction"].batch_size = 8
config.data.max_images_per_product = 3

# Force CPU usage
config.model.device = "cpu"
```

**Resume from interruptions:**
The system automatically saves checkpoints. If interrupted:
```bash
# Check what was completed
ls checkpoints/

# Resume automatically
python main.py  # Will continue from last checkpoint

# Force restart specific phase
python main.py --force feature_extraction
```

## 📈 API Usage & Integration

### Search Endpoint

```python
import requests
from PIL import Image
import io

# Search with uploaded image
with open('query_image.jpg', 'rb') as f:
    response = requests.post(
        'http://localhost:8000/search',
        files={'file': ('query.jpg', f, 'image/jpeg')},
        params={'top_k': 10}
    )

results = response.json()
print(f"Found {results['total_results']} similar products in {results['query_time_ms']:.1f}ms")

# Process results
for i, product in enumerate(results['results']):
    print(f"{i+1}. {product['title']} (similarity: {product['similarity_score']:.3f})")
```

### Response Format

```json
{
  "query_time_ms": 156.7,
  "total_results": 5,
  "results": [
    {
      "product_id": "8526513733794",
      "title": "Grey Checks Slim Fit Shirt",
      "similarity_score": 0.892,
      "preview_url": "https://cdn.shopify.com/...",
      "num_views": 10,
      "rank": 1
    }
  ]
}
```

### Health & Statistics Endpoints

```python
# Check system health
health = requests.get('http://localhost:8000/health').json()
print(f"Status: {health['status']}, Index size: {health['index_size']}")

# Get system statistics  
stats = requests.get('http://localhost:8000/stats').json()
print(f"Total products: {stats['total_products']}")
```

## 📊 System Performance & Specifications

### Performance Metrics
- **Search Speed**: < 200ms per query (CPU), < 100ms (GPU)
- **Index Building**: ~2-3 minutes per 1000 products (GPU)
- **Memory Usage**: ~4-6GB during processing, ~2GB during serving
- **Storage**: ~50-100MB per 1000 products (images + index)
- **Throughput**: 10-50 searches/second (depending on hardware)

### System Requirements
- **Python**: 3.8-3.11 (3.10+ recommended for best PyTorch compatibility)
- **Memory**: 8GB minimum, 16GB recommended for large catalogs
- **Storage**: 5GB for models + variable for image cache
- **GPU**: Optional but recommended (RTX 3060+ or equivalent)
- **Network**: Stable internet for model downloads and image URLs

### Hardware Recommendations

**Development Setup:**
- CPU: 4+ cores, 8GB RAM
- Storage: 10GB available space
- Network: Broadband for image downloads

**Production Setup:**
- CPU: 8+ cores, 16GB RAM
- GPU: RTX 3060+ with 8GB+ VRAM
- Storage: SSD with 50GB+ space
- Network: High-bandwidth for large image catalogs

## 📁 Generated Files Structure

After successful pipeline execution:

```
📁 data/
├── 📄 processed_products.json      # Validated and cleaned product data
└── 📁 image_cache/                 # Downloaded and cached images
    ├── 📁 product_8526513733794/   # Organized by product ID
    │   ├── 🖼️ view_1.jpg          # Multiple product views
    │   ├── 🖼️ view_2.jpg
    │   └── 🖼️ preview.jpg         # Primary image
    └── 📁 product_*/               # Additional products...

📁 models/
├── 🔍 product_index.faiss          # FAISS similarity search index
└── 📊 product_metadata.pkl         # Product metadata for results

📁 checkpoints/                     # Resume points for pipeline stages
├── 📄 data_validation.json         # Data validation completion
├── 📄 image_download.json          # Image download progress
├── 📄 feature_extraction.json      # Feature extraction progress
└── 📄 index_building.json          # Index building completion

📁 logs/                            # Comprehensive system logs
├── 📄 image_search_YYYYMMDD_HHMMSS.log  # Detailed debug logs
└── 📄 errors.log                   # Error-only logs
```

## 🔐 Production Deployment Notes

### Security Considerations
```python
# Update CORS settings in main_api.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://yourdomain.com"],  # Replace ["*"]
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)
```

### Environment Variables
```bash
# Set production configurations
export CLIP_MODEL="openai/clip-vit-base-patch32"
export API_HOST="0.0.0.0"
export API_PORT="8000"
export DEVICE="cuda"  # or "cpu"
```

### Monitoring & Health Checks
- Use `/health` endpoint for load balancer health checks
- Monitor logs in `logs/` directory
- Set up alerts for API response times > 500ms
- Monitor GPU memory usage if using CUDA

### Scaling Considerations
- Consider Redis for caching frequent searches
- Implement rate limiting for API endpoints
- Use CDN for serving product images
- Consider database backend for metadata instead of pickle files

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**🚀 Ready to build your AI-powered image search system?**

```bash
python setup.py && python main.py
```

**Need help?** Check the troubleshooting section or examine the generated logs in the `logs/` directory for detailed debugging information. 