# 🔍 Multi-Model Image Search System

A sophisticated AI-powered visual similarity search platform with specialized model support. Built with CLIP and FAISS, featuring modular architecture, multi-model training capabilities, and modern web interfaces.

## ✨ Key Features

- **🎯 Multi-Model Architecture**: Train and deploy specialized models for different product categories
- **🧠 AI-Powered Search**: OpenAI CLIP for deep visual understanding and feature extraction
- **⚡ Lightning Fast**: FAISS indexing delivers sub-second search results (< 200ms)
- **🔄 Smart Resume**: Comprehensive checkpoint system prevents work duplication
- **🏗️ Modular Design**: Clean separation of concerns with organized codebase
- **📊 Advanced Pipeline**: 4-phase processing with intelligent progress tracking
- **🖥️ Modern UI**: Enhanced Streamlit interface with model selection
- **🚀 Production Ready**: FastAPI backend with multiple endpoints
- **📈 Horizontally Scalable**: Support for unlimited specialized models

## 🏗️ Architecture Overview

```
📁 ImageSearch/
├── 📂 core/                        # Configuration & Model Management
│   ├── config.py                   # System configuration
│   ├── model_config.py             # Multi-model definitions
│   └── __init__.py
├── 📂 pipeline/                    # Data Processing & ML Pipeline
│   ├── process_csv_data.py         # Data validation & image downloading
│   ├── build_engine.py             # CLIP extraction & FAISS indexing
│   ├── main.py                     # Pipeline orchestrator
│   └── __init__.py
├── 📂 api/                         # Backend Services
│   ├── main_api.py                 # Original single-model API
│   ├── enhanced_api.py             # Multi-model API with routing
│   └── __init__.py
├── 📂 training/                    # Model Training
│   ├── train_model.py              # Universal training script
│   └── __init__.py
├── 📂 ui/                          # Frontend Interfaces
│   ├── app_streamlit.py            # Enhanced multi-model UI
│   └── __init__.py
├── 📂 utils/                       # Shared Utilities
│   ├── logging_utils.py            # Advanced logging & progress tracking
│   └── __init__.py
├── 📂 data/                        # Organized by Model
│   ├── general/                    # General product model data
│   │   ├── products.csv
│   │   ├── image_cache/
│   │   └── processed_products.json
│   └── shirts/                     # Specialized shirts model data
│       ├── products.csv
│       └── image_cache/
├── 📂 models/                      # Trained Models
│   ├── general/                    # General model artifacts
│   │   ├── index.faiss
│   │   ├── metadata.pkl
│   │   └── embeddings.npy
│   └── shirts/                     # Shirts model artifacts
├── 📂 checkpoints/                 # Resume Points
├── 📂 logs/                        # System Logs
├── run_pipeline.py                 # 🚀 Pipeline launcher
├── run_training.py                 # 🎯 Training launcher
├── run_api.py                      # 🌐 API launcher
├── run_streamlit.py                # 🖥️ UI launcher
├── products.csv                    # Original data file
└── requirements.txt                # Dependencies
```

## 🎯 Multi-Model System

### Available Models

- **🏪 General Product Search**: Multi-category product search (your current 175 products)
- **👔 Specialized Shirts Search**: Dedicated shirts model (ready for 3-4k images)
- **➕ Extensible**: Easy to add new specialized models

### API Endpoints

```bash
# Model Information
GET  /models          # List all available models
GET  /health          # System status & model info

# Specialized Search Endpoints
POST /general/search  # Search general model
POST /shirts/search   # Search shirts model
POST /all/search      # Search across all models

# Legacy Support
POST /search_products/ # Backward compatibility (uses general model)
```

## ⚡ Quick Start

### 1. Installation

```bash
# Clone and setup environment
git clone <repository>
cd ImageSearch
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Validate installation
python test_setup.py
```

### 2. Train Your First Model

```bash
# Your current general model (already trained)
python run_training.py general

# Train a new shirts model (place CSV in data/shirts/products.csv)
python run_training.py shirts --force

# List available models
python run_training.py --list
```

### 3. Start the System

```bash
# Start multi-model API backend
python run_api.py

# Start enhanced Streamlit frontend
python run_streamlit.py
```

**Access Points:**
- **🖥️ Enhanced UI**: http://localhost:8501 (Multi-model interface)
- **📖 API Docs**: http://localhost:8000/docs (Interactive documentation)
- **🔍 Models Info**: http://localhost:8000/models (Available models)

## 🔧 Configuration & Customization

### Adding New Models

1. **Define Model** in `core/model_config.py`:
```python
MODELS = {
    "your_model": {
        "name": "Your Model Name",
        "description": "Model description",
        "data_dir": str(DATA_DIR / "your_model"),
        "models_dir": str(MODELS_DIR / "your_model"),
        "csv_file": "products.csv",
        "categories": ["Category1", "Category2"],
        "api_path": "/your_model",
        "port_offset": 2
    }
}
```

2. **Prepare Data**:
```bash
mkdir data/your_model
# Place your products.csv in data/your_model/
```

3. **Train Model**:
```bash
python run_training.py your_model --force
```

### Data Format

Your CSV files should follow this structure:

```csv
shopify_product_id,title,images,preview_image,category
123456789,"Product Title","[""url1.jpg"", ""url2.jpg""]","preview.jpg","Category"
```

**Required Columns:**
- `shopify_product_id`: Unique identifier
- `title`: Product name/description
- `images`: JSON array of image URLs (up to 10 images)
- `preview_image`: Primary display image

**Optional Columns:**
- `category`: Product category for filtering

## 🚀 Usage Examples

### Python API

```python
import requests

# Search general model
response = requests.post(
    "http://localhost:8000/general/search",
    files={"file": open("query_image.jpg", "rb")},
    params={"top_k": 5}
)
results = response.json()

# Search all models
response = requests.post(
    "http://localhost:8000/all/search",
    files={"file": open("query_image.jpg", "rb")},
    params={"top_k": 3}
)
all_results = response.json()
```

### Command Line

```bash
# Train models
python run_training.py general --force
python run_training.py shirts --force

# Run pipeline only (no API)
python run_pipeline.py --mode data    # Data processing only
python run_pipeline.py --mode ml      # ML pipeline only
python run_pipeline.py --mode all     # Complete pipeline
```

## 📊 Pipeline Phases

### Phase 1: Data Validation
- Validates CSV structure and required columns
- Handles malformed JSON arrays in image URLs
- Reports data quality statistics
- **Checkpoint**: `checkpoints/data_validation.json`

### Phase 2: Image Download & Caching
- Concurrent download with 8 workers
- Image validation (format, size, integrity)
- Organized cache by product ID
- **Checkpoint**: `checkpoints/image_download.json`

### Phase 3: CLIP Feature Extraction
- OpenAI CLIP model: `openai/clip-vit-base-patch32`
- Batch processing for efficiency
- Multi-image aggregation (averages multiple views)
- L2 normalization for cosine similarity
- **Checkpoint**: `checkpoints/feature_extraction.json`

### Phase 4: FAISS Index Building
- Creates searchable FAISS index
- Supports multiple index types (Flat, IVF, HNSW)
- Exact cosine similarity by default
- **Checkpoint**: `checkpoints/index_building.json`

## 🔍 Advanced Features

### Smart Resume System
```bash
# Skip completed phases automatically
python run_training.py shirts  # Resumes from last checkpoint

# Force complete rebuild
python run_training.py shirts --force
```

### Performance Configuration
Modify `core/config.py`:
```python
# Download settings
max_images_per_product = 10      # Images per product
download_timeout = 10            # Download timeout (seconds)
max_workers = 8                  # Concurrent downloads

# Model settings
device = "auto"                  # auto/cuda/cpu
batch_size = 16                  # CLIP batch size
embedding_dim = 512              # Feature dimensions

# Index settings
index_type = "flat"              # flat/ivf/hnsw
similarity_metric = "cosine"     # cosine/euclidean
```

### Custom Index Types
```python
# For large datasets (>100k products)
config.index.index_type = "ivf"

# For maximum speed (approximate search)
config.index.index_type = "hnsw"
```

## 🧪 Testing & Validation

```bash
# System validation
python test_setup.py

# Health check
curl http://localhost:8000/health

# Model information
curl http://localhost:8000/models

# Test search (with image file)
curl -X POST \
  -F "file=@test_image.jpg" \
  "http://localhost:8000/general/search?top_k=5"
```

## 📈 Monitoring & Logging

### Log Files
- **Pipeline logs**: `logs/pipeline.log`
- **API logs**: `logs/api.log`
- **Training logs**: `logs/training.log`

### Progress Tracking
- Real-time progress bars with ETA
- Detailed checkpoint information
- Performance metrics and timing

### Health Monitoring
```python
# API health endpoint returns:
{
    "status": "healthy",
    "available_models": 2,
    "total_products": 3500,
    "models": {
        "general": {"products": 175, "name": "General Product Search"},
        "shirts": {"products": 3325, "name": "Specialized Shirts Search"}
    }
}
```

## 🔧 Troubleshooting

### Common Issues

**Model Not Training**
```bash
# Check CSV file location
ls data/your_model/products.csv

# Check logs
tail -f logs/training.log

# Force rebuild
python run_training.py your_model --force
```

**API Connection Issues**
```bash
# Check if API is running
curl http://localhost:8000/health

# Restart API
python run_api.py
```

**Memory Issues**
```python
# Reduce batch size in core/config.py
config.model.batch_size = 8

# Use CPU-only mode
config.model.device = "cpu"
```

### Performance Optimization

**For Large Datasets (>10k products)**:
- Use `index_type = "ivf"` for faster approximate search
- Increase `batch_size` if you have GPU memory
- Consider distributed processing for huge datasets

**For Production Deployment**:
- Use GPU acceleration for faster feature extraction
- Enable CORS properly for web deployment
- Set up proper logging and monitoring
- Use load balancing for high-traffic scenarios

## 📄 License

[Your License Here]

## 🤝 Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🆘 Support

For issues, questions, or feature requests:
- 📧 Email: [your-email]
- 🐛 Issues: [GitHub Issues URL]
- 📖 Docs: [Documentation URL]

---

**Built with ❤️ using OpenAI CLIP, FAISS, FastAPI, and Streamlit** 