import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
import clip
import numpy as np
from DFAD_model_base import DFADModel
from transformers import BlipProcessor, BlipForConditionalGeneration


st.set_page_config(
    page_title="Model Comparison - Real vs Fake Detector",
    page_icon="⚖️",
    layout="wide"
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

@st.cache_resource
def load_model(checkpoint_path, model_name):
    """Load model with a unique name for caching"""
    model = DFADModel()
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Extract model state dict from checkpoint dictionary
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    # Handle DataParallel weights
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_clip_model():
    clip_model, preprocess = clip.load("ViT-L/14", device=device)
    return clip_model, preprocess

@st.cache_resource
def load_blip_model(model_dir:str = None):
    """Load BLIP model for caption generation"""
    if model_dir is None:
        processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip-image-captioning-large"
        ).to(device)
    else:
        # Load from local directory (works offline)
        processor = BlipProcessor.from_pretrained(model_dir, local_files_only=True)
        blip_model = BlipForConditionalGeneration.from_pretrained(
            model_dir,
            local_files_only=True
        ).to(device)
    
    return processor, blip_model

def generate_caption(image, processor, blip_model):
    """Generate caption using BLIP"""
    # Process image
    inputs = processor(image, return_tensors="pt").to(device)
    
    # Generate caption
    with torch.no_grad():
        out = blip_model.generate(**inputs, max_length=50)
    
    caption = processor.decode(out[0], skip_special_tokens=True)
    return caption

def predict(image, caption, model, clip_model, preprocess):
    """Make prediction"""
    # Extract image features
    image_input = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        image_features = clip_model.encode_image(image_input).cpu().numpy().squeeze()
    
    # Extract text features
    text_input = clip.tokenize([caption], truncate=True).to(device)
    with torch.no_grad():
        text_features = clip_model.encode_text(text_input).cpu().numpy().squeeze()
    
    # Predict
    image_features = torch.tensor(image_features, dtype=torch.float32).unsqueeze(0).to(device)
    text_features = torch.tensor(text_features, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(image_features, text_features).squeeze()
        probability = torch.sigmoid(output).item()
    
    return probability

def display_prediction(probability, threshold, model_name, col):
    """Display prediction results in a column"""
    with col:
        st.markdown(f"### {model_name}")
        
        # Prediction
        prediction = "FAKE" if probability > threshold else "REAL"
        color = "red" if prediction == "FAKE" else "green"
        st.markdown(f"#### Prediction: :{color}[**{prediction}**]")
        
        # Progress bars
        st.progress(probability, text=f"Fake: {probability*100:.1f}%")
        st.progress(1-probability, text=f"Real: {(1-probability)*100:.1f}%")
        
        # Confidence interpretation
        if probability > 0.8:
            st.error("🚨 Very likely FAKE")
        elif probability > threshold:
            st.warning("⚠️ Likely FAKE")
        elif probability > 0.3:
            st.info("🤷 Uncertain")
        elif probability > 0.2:
            st.success("✅ Likely REAL")
        else:
            st.success("✅ Very likely REAL")

def main():
    st.title("⚖️ Model Comparison - Real vs Fake Image Detector")
    st.markdown("Compare predictions from two different model checkpoints side-by-side")
    
    # Sidebar - Model Settings
    st.sidebar.header("⚙️ Model Settings")
    
    # Model 1 settings
    st.sidebar.subheader("🔵 Model A")
    model1_name = st.sidebar.text_input("Model A Name", value="RELU")
    model1_path = st.sidebar.text_input(
        "Model A Checkpoint",
        value="relu_checkpoints_from_images_with_blip/checkpoint_epoch_0049.pt"
        # value="relu_af_checkpoints_auc_gamma_0.8_approx/checkpoint_epoch_0031.pt"
    )
    
    st.sidebar.markdown("---")
    
    # Model 2 settings
    st.sidebar.subheader("🟢 Model B")
    model2_name = st.sidebar.text_input("Model B Name", value="GELU")
    model2_path = st.sidebar.text_input(
        "Model B Checkpoint",
        value="gelu_checkpoints_from_images_with_blip/checkpoint_epoch_0049.pt"
        # value="gelu_af_checkpoints_auc_gamma_0.8_approx/checkpoint_epoch_0031.pt"
    )
    
    st.sidebar.markdown("---")
    
    # Threshold
    threshold = st.sidebar.slider("Classification Threshold", 0.0, 1.0, 0.5, 0.01)
    
    # Load models
    try:
        with st.spinner("Loading models... This may take a minute..."):
            model1 = load_model(model1_path, "model1")
            model2 = load_model(model2_path, "model2")
            clip_model, preprocess = load_clip_model()
            blip_processor, blip_model = load_blip_model(model_dir='./models/blip-large')
        st.sidebar.success(f"✓ All models loaded on {device}")
    except Exception as e:
        st.sidebar.error(f"Error loading models: {e}")
        st.stop()
    
    # Main content
    st.markdown("---")
    
    # Upload section
    upload_col1, upload_col2 = st.columns([1, 1])
    
    with upload_col1:
        st.subheader("📷 Upload Image")
        uploaded_file = st.file_uploader(
            "Choose an image",
            type=['png', 'jpg', 'jpeg']
        )
        
        if uploaded_file:
            image = Image.open(uploaded_file).convert('RGB')
            # Create centered thumbnail
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                st.image(image, width=300)
    
    with upload_col2:
        st.subheader("📝 Caption")
        
        # Initialize session state for caption if not exists
        if 'generated_caption' not in st.session_state:
            st.session_state.generated_caption = ""
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None
        
        # Reset caption when a new image is uploaded
        if uploaded_file:
            current_file_id = uploaded_file.file_id if hasattr(uploaded_file, 'file_id') else uploaded_file.name
            if st.session_state.last_uploaded_file != current_file_id:
                st.session_state.generated_caption = ""
                st.session_state.last_uploaded_file = current_file_id
        
        # Auto-generate caption button
        if uploaded_file:
            if st.button("🤖 Auto-Generate Caption with BLIP", type="secondary"):
                with st.spinner("Generating caption..."):
                    st.session_state.generated_caption = generate_caption(
                        image, blip_processor, blip_model
                    )
                st.success("✓ Caption generated!")
        
        # Caption input (can be edited or left empty for auto-generation)
        caption = st.text_area(
            "Image Caption/Description",
            value=st.session_state.generated_caption,
            height=150,
            placeholder="Leave empty for auto-generation, or enter your own caption...",
            help="Leave empty to auto-generate with BLIP, or provide your own caption"
        )
        
        st.markdown("")
        st.markdown("")
        analyze_button = st.button(
            "🔍 Analyze with Both Models",
            type="primary",
            use_container_width=True
        )
    
    # Prediction section
    if analyze_button:
        if not uploaded_file:
            st.error("⚠️ Please upload an image first!")
        else:
            # Auto-generate caption if empty
            final_caption = caption.strip()
            caption_source = "Manual"
            
            if not final_caption:
                with st.spinner("Auto-generating caption with BLIP..."):
                    final_caption = generate_caption(image, blip_processor, blip_model)
                    st.session_state.generated_caption = final_caption
                    caption_source = "Auto-generated"
                st.info(f"🤖 **Auto-generated Caption:** {final_caption}")
            
            st.markdown("---")
            st.subheader("📊 Comparison Results")
            
            # Show caption used
            st.info(f"**Caption used ({caption_source}):** {final_caption}")
            
            # Create comparison columns
            comp_col1, comp_col2 = st.columns(2)
            
            # Get predictions
            with st.spinner("Analyzing with both models..."):
                prob1 = predict(image, final_caption, model1, clip_model, preprocess)
                prob2 = predict(image, final_caption, model2, clip_model, preprocess)
            
            # Display results side by side
            display_prediction(prob1, threshold, model1_name, comp_col1)
            display_prediction(prob2, threshold, model2_name, comp_col2)
    
    # Info section
    with st.expander("💡 How to Use"):
        st.markdown("""
        ### Using the Comparison Tool:
           
        1. **Upload an image**
        
        2. **Caption Options:**
           - **Auto-generate:** Click "Auto-Generate Caption with BLIP" or leave caption empty
           - **Manual:** Type your own caption in the text area
           - **Edit:** Generate a caption then edit it before analyzing
           
        3. **Click "Analyze with Both Models"** to get predictions
           - If caption is empty, it will be auto-generated automatically
        
        4. **Compare Results:**
           - View predictions side-by-side
           - Check if models agree or disagree
           - See which model is more confident
           
        ### Features:
        - **Automatic caption generation** using BLIP when no caption is provided
        - **Manual caption override** - you can always provide your own caption
        - **Side-by-side comparison** of two different models
        - **Detailed metrics** and visualizations
        
        ### Use Cases:
        - Compare different training epochs
        - Evaluate different architectures (RELU vs GELU)
        - Test with and without custom captions
        - Validate model improvements
        """)
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center'>
            <p>Model Comparison Tool with Auto-Captioning • Built with Streamlit • Powered by BLIP, CLIP, and PyTorch</p>
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()