"""
Enhanced Multi-Model Streamlit Frontend for Image Search System
Support for multiple specialized models with clean model selection
"""
import streamlit as st
import requests
import io
import time
from PIL import Image
from typing import Dict, Any, List

# Page configuration
st.set_page_config(
    page_title="Multi-Model Image Search",
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
    .model-badge {
        background: linear-gradient(90deg, #3b82f6, #06b6d4);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.875rem;
        display: inline-block;
        margin-right: 0.5rem;
    }
    .model-card {
        background: white;
        border-radius: 12px;
        padding: 1rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
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
    .status-trained {
        color: #059669;
        font-weight: 600;
    }
    .status-not-trained {
        color: #dc2626;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# API configuration
API_URL = "http://127.0.0.1:8000"
MODELS_ENDPOINT = f"{API_URL}/models"
HEALTH_ENDPOINT = f"{API_URL}/health"

def get_available_models():
    """Get available models from the API"""
    try:
        response = requests.get(MODELS_ENDPOINT, timeout=5)
        if response.status_code == 200:
            return True, response.json()
        return False, []
    except Exception as e:
        return False, str(e)

def check_api_health():
    """Check if the API backend is healthy"""
    try:
        response = requests.get(HEALTH_ENDPOINT, timeout=5)
        return response.status_code == 200, response.json() if response.status_code == 200 else None
    except Exception as e:
        return False, str(e)

def search_with_model(image: Image.Image, model_key: str, top_k: int):
    """Search using a specific model"""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {'file': ('query_image.png', img_byte_arr, 'image/png')}
        params = {'top_k': top_k}
        
        # Use model-specific endpoint
        search_url = f"{API_URL}/{model_key}/search"
        
        start_time = time.time()
        response = requests.post(search_url, files=files, params=params, timeout=30)
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

def search_all_models(image: Image.Image, top_k: int):
    """Search across all available models"""
    try:
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_byte_arr.seek(0)
        
        files = {'file': ('query_image.png', img_byte_arr, 'image/png')}
        params = {'top_k': top_k}
        
        search_url = f"{API_URL}/all/search"
        
        start_time = time.time()
        response = requests.post(search_url, files=files, params=params, timeout=30)
        query_time = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            data = response.json()
            return True, data, query_time
        else:
            error_detail = response.json().get('detail', 'Unknown error') if response.content else 'API Error'
            return False, error_detail, query_time
            
    except Exception as e:
        return False, f"Error: {str(e)}", 0

def display_model_results(model_key: str, results_data: Dict, rank_offset: int = 0):
    """Display results for a specific model"""
    if not results_data or not results_data.get('results'):
        st.info(f"No results from {results_data.get('model_name', model_key)} model")
        return
    
    # Model header
    st.markdown(f"""
    <div class='model-card'>
        <div class='model-badge'>{results_data['model_name']}</div>
        <span style='color: #6b7280;'>• {results_data['total_results']} results • {results_data['query_time_ms']:.0f}ms</span>
    </div>
    """, unsafe_allow_html=True)
    
    # Results
    for i, result in enumerate(results_data['results']):
        col_rank, col_title, col_score, col_meta = st.columns([0.5, 3, 1, 1.5])
        
        with col_rank:
            st.markdown(f"**#{rank_offset + i + 1}**")
        
        with col_title:
            title = result['title'][:50] + '...' if len(result['title']) > 50 else result['title']
            st.markdown(f"**{title}**")
            st.caption(f"ID: {result['product_id']} • {result['category_from_db']}")
        
        with col_score:
            score_percent = result['similarity_score'] * 100
            st.markdown(f"<div class='similarity-badge'>{score_percent:.1f}%</div>", 
                       unsafe_allow_html=True)
        
        with col_meta:
            st.progress(result['similarity_score'])
        
        st.divider()

def main():
    # Header
    st.markdown("<h1 class='main-header'>🔍 Multi-Model Image Search</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>AI-powered visual search across specialized models</p>", 
                unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ System Status")
        
        # API Health Check
        health_status, health_data = check_api_health()
        if health_status:
            st.success("✅ API Connected")
            if health_data and 'models' in health_data:
                for model_key, model_info in health_data['models'].items():
                    st.metric(
                        f"{model_info['name']}", 
                        f"{model_info['products']} products"
                    )
        else:
            st.error(f"❌ API Disconnected")
            st.caption(f"{health_data}")
            st.stop()
        
        st.markdown("### 🎯 Model Selection")
        
        # Get available models
        models_status, models_data = get_available_models()
        if models_status and models_data:
            # Filter only trained models
            trained_models = [m for m in models_data if m['status'] == 'trained']
            
            if trained_models:
                model_options = {
                    "🌟 Search All Models": "all"
                }
                for model in trained_models:
                    icon = "👔" if model['key'] == 'shirts' else "🏪"
                    model_options[f"{icon} {model['name']}"] = model['key']
                
                selected_model_display = st.selectbox(
                    "Choose Model",
                    options=list(model_options.keys()),
                    help="Select which model to use for search"
                )
                selected_model = model_options[selected_model_display]
                
                # Show model info
                if selected_model != "all":
                    model_info = next(m for m in trained_models if m['key'] == selected_model)
                    st.info(f"**{model_info['name']}**\n\n{model_info['description']}\n\n**Categories:** {', '.join(model_info['categories'][:3])}...")
                
            else:
                st.warning("No trained models available")
                st.stop()
        else:
            st.error("Cannot load models list")
            st.stop()
        
        st.markdown("### 🎛️ Settings")
        top_k = st.slider("Results per model", 1, 10, 3)
        
        # Show all available models
        st.markdown("### 📋 Available Models")
        for model in models_data:
            status_class = "status-trained" if model['status'] == 'trained' else "status-not-trained"
            status_icon = "✅" if model['status'] == 'trained' else "❌"
            st.markdown(f"""
            <div style='margin-bottom: 0.5rem;'>
                <strong>{model['name']}</strong> {status_icon}<br>
                <small class='{status_class}'>{model['status'].replace('_', ' ').title()}</small><br>
                <small style='color: #6b7280;'>{model['total_products']} products</small>
            </div>
            """, unsafe_allow_html=True)
    
    # Main content
    col_upload, col_results = st.columns([1, 2])
    
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
            
            search_button_text = "🔍 Search All Models" if selected_model == "all" else f"🔍 Search with {selected_model_display.split(' ', 1)[1]}"
            
            if st.button(search_button_text, type="primary", use_container_width=True):
                with st.spinner("Searching..."):
                    if selected_model == "all":
                        success, result, query_time = search_all_models(image, top_k)
                    else:
                        success, result, query_time = search_with_model(image, selected_model, top_k)
                
                if success:
                    st.session_state['search_results'] = result
                    st.session_state['search_time'] = query_time
                    st.session_state['search_mode'] = selected_model
                    
                    if selected_model == "all":
                        total_results = sum(model_results['total_results'] for model_results in result.values())
                        st.success(f"Found {total_results} results across {len(result)} models!")
                    else:
                        st.success(f"Found {result['total_results']} results!")
                    st.rerun()
                else:
                    st.error(f"Search failed: {result}")
    
    with col_results:
        st.markdown("### 📊 Search Results")
        
        if 'search_results' in st.session_state:
            results = st.session_state['search_results']
            search_mode = st.session_state.get('search_mode', 'single')
            client_time = st.session_state.get('search_time', 0)
            
            if search_mode == "all":
                # Multi-model results
                st.markdown(f"**Search completed in {client_time:.0f}ms**")
                st.markdown("---")
                
                rank_offset = 0
                for model_key, model_results in results.items():
                    display_model_results(model_key, model_results, rank_offset)
                    rank_offset += len(model_results.get('results', []))
                    
            else:
                # Single model results
                met1, met2, met3 = st.columns(3)
                with met1:
                    st.metric("Found", results['total_results'])
                with met2:
                    st.metric("Client", f"{client_time:.0f}ms")
                with met3:
                    st.metric("API", f"{results['query_time_ms']:.0f}ms")
                
                st.markdown("---")
                display_model_results(search_mode, results)
                
        else:
            st.markdown("""
            <div style='text-align: center; padding: 2rem; color: #6b7280;'>
                <h3>Ready to search</h3>
                <p>Upload an image and select a model to find similar products</p>
                <p>🖼️ → 🎯 → 🔍 → 📊</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Footer
    st.markdown("---")
    st.markdown(
        f"<div style='text-align: center; color: #6b7280; font-size: 0.875rem;'>"
        f"Enhanced Multi-Model Search • <a href='{API_URL}/docs' target='_blank'>API Docs</a> • "
        f"<a href='{API_URL}/models' target='_blank'>Models Info</a>"
        f"</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main() 