"""
Simple and Reliable Streamlit Frontend for Image Search System
Clean, minimalistic design with visual product cards
"""
import streamlit as st
import requests
import io
import time
from PIL import Image
from typing import Dict, Any
from config import config

# Page configuration
st.set_page_config(
    page_title="Image Search System",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 600;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .product-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
    }
    .metric-container {
        text-align: center;
        padding: 0.5rem;
    }
    .similarity-badge {
        background: linear-gradient(90deg, #10b981, #34d399);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
        text-align: center;
    }
    .product-title {
        font-weight: 600;
        color: #1f2937;
        margin-bottom: 0.5rem;
    }
    .product-meta {
        color: #6b7280;
        font-size: 0.875rem;
    }
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
    }
    .card-layout {
        display: flex;
        align-items: center;
        gap: 1rem;
        padding: 0.5rem;
        border-radius: 8px;
        background: #f9fafb;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = f"http://{config.api.host}:{config.api.port}"
SEARCH_ENDPOINT = f"{API_URL}/search"
HEALTH_ENDPOINT = f"{API_URL}/health"
STATS_ENDPOINT = f"{API_URL}/stats"

def check_api_health():
    """Check if the API backend is healthy"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)

def get_api_stats():
    """Get API statistics"""
    try:
        response = requests.get(STATS_ENDPOINT, timeout=5)
        return response.json() if response.status_code == 200 else None
    except:
        return None

def load_image_from_url(url, max_size=(200, 200)):
    """Load and resize image from URL"""
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            image = Image.open(io.BytesIO(response.content))
            image.thumbnail(max_size, Image.Resampling.LANCZOS)
            return image
        return None
    except Exception:
        return None

def search_similar_products(image: Image.Image, top_k: int):
    """Search for similar products"""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {'file': ('query_image.png', img_byte_arr, 'image/png')}
        params = {'top_k': top_k}
        
        start_time = time.time()
        response = requests.post(SEARCH_ENDPOINT, files=files, params=params, timeout=30)
        query_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            return True, data, query_time
        else:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'API Error'
            return False, error_detail, query_time
            
    except requests.exceptions.ConnectionError:
        return False, "Cannot connect to API. Please ensure the backend server is running.", 0
    except requests.exceptions.Timeout:
        return False, "Search request timed out. Please try again.", 0
    except Exception as e:
        return False, f"Error: {str(e)}", 0

def display_compact_card(result, rank):
    """Display a product result as a compact, beautiful card - using single row layout"""
    # Use a single row with 4 columns to avoid nesting
    col_img, col_title, col_score, col_meta = st.columns([1, 3, 1, 2])
    
    with col_img:
        if result.get('preview_url'):
            product_image = load_image_from_url(result['preview_url'], (120, 120))
            if product_image:
                st.image(product_image, use_container_width=True)
            else:
                st.markdown("```\n📷\nNo Image\n```")
        else:
            st.markdown("```\n📷\nNo Image\n```")
    
    with col_title:
        title = result['title'][:45] + '...' if len(result['title']) > 45 else result['title']
        st.markdown(f"**#{rank} {title}**")
        st.caption(f"ID: {result['product_id']}")
    
    with col_score:
        score_percent = result['similarity_score'] * 100
        st.markdown(f"<div class='similarity-badge'>{score_percent:.1f}%</div>", 
                   unsafe_allow_html=True)
    
    with col_meta:
        st.caption(f"👁 {result['num_views']:,} views")
        st.progress(result['similarity_score'])
    
    st.divider()

def display_grid_card(result, rank):
    """Display a product result in grid format"""
    with st.container():
        # Image
        if result.get('preview_url'):
            product_image = load_image_from_url(result['preview_url'], (150, 150))
            if product_image:
                st.image(product_image, use_container_width=True)
            else:
                st.markdown("```\n📷\n```")
        else:
            st.markdown("```\n📷\n```")
        
        # Content
        score_percent = result['similarity_score'] * 100
        st.markdown(f"**#{rank}** • {score_percent:.1f}%")
        
        title = result['title'][:25] + '...' if len(result['title']) > 25 else result['title']
        st.caption(title)
        
        st.caption(f"👁 {result['num_views']:,}")
        st.progress(result['similarity_score'])

# Main app
def main():
    # Header
    st.markdown("<h1 class='main-header'>🔍 Image Search</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Find similar products using AI-powered visual search</p>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        
        # API Health Check
        health_status, health_data = check_api_health()
        if health_status:
            st.success("✅ Connected")
            with st.expander("Details", expanded=False):
                st.json(health_data)
        else:
            st.error(f"❌ Disconnected")
            st.caption(f"{health_data}")
            st.stop()
        
        # API Stats
        stats_data = get_api_stats()
        if stats_data:
            st.markdown("### 📊 Database")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Products", stats_data['total_products'])
            with col2:
                st.metric("Dimension", stats_data['embedding_dimension'])
        
        st.markdown("### 🎛️ Settings")
        top_k = st.slider("Results", 1, 15, 5)
        view_mode = st.radio("Layout", ["Cards", "Grid"], horizontal=True)
    
    # Main content
    col_upload, col_results = st.columns([1, 2] if view_mode == "Cards" else [1, 2.5])
    
    with col_upload:
        st.markdown("### 📤 Upload Image")
        uploaded_file = st.file_uploader(
            "",
            type=['png', 'jpg', 'jpeg', 'webp'],
            help="Upload an image to search for similar products"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption=f"{image.size[0]}×{image.size[1]}px", use_container_width=True)
            
            if st.button("🔍 Search", type="primary", use_container_width=True):
                with st.spinner("Searching..."):
                    success, result, query_time = search_similar_products(image, top_k)
                
                if success:
                    st.session_state['search_results'] = result
                    st.session_state['search_time'] = query_time
                    st.success(f"Found {result['total_results']} results!")
                    st.rerun()
                else:
                    st.error(f"Search failed: {result}")
    
    with col_results:
        st.markdown("### 📊 Results")
        
        if 'search_results' in st.session_state:
            data = st.session_state['search_results']
            client_time = st.session_state.get('search_time', 0)
            
            # Metrics
            met1, met2, met3 = st.columns(3)
            with met1:
                st.metric("Found", data['total_results'])
            with met2:
                st.metric("Client", f"{client_time:.0f}ms")
            with met3:
                st.metric("API", f"{data['query_time_ms']:.0f}ms")
            
            st.markdown("---")
            
            # Results
            if data['results']:
                if view_mode == "Grid":
                    cols = st.columns(3)
                    for i, result in enumerate(data['results']):
                        with cols[i % 3]:
                            display_grid_card(result, i + 1)
                else:
                    for i, result in enumerate(data['results']):
                        display_compact_card(result, i + 1)
            else:
                st.info("No similar products found")
        else:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; color: #6b7280;'>
                <h3>Ready to search</h3>
                <p>Upload an image to find visually similar products</p>
                <p>🖼️ → 🔍 → 📊</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>"
        f"Powered by CLIP & FAISS • <a href='{API_URL}/docs' target='_blank'>API Docs</a>"
        f"</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 