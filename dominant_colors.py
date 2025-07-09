#!/usr/bin/env python3
"""
Dominant Color Finder - Streamlit Web App
A web application to extract dominant colors from images using K-Means clustering.
"""

import streamlit as st
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import io
import base64


def load_and_preprocess_image(uploaded_file, max_size=300):
    """
    Load an image and convert it to RGB format.
    Resize if the image is too large for faster processing.
    
    Args:
        uploaded_file: Streamlit uploaded file object
        max_size (int): Maximum dimension for resizing (default: 300)
    
    Returns:
        PIL.Image: Preprocessed image
        numpy.ndarray: Array of RGB pixels
    """
    try:
        # Load the image
        image = Image.open(uploaded_file)
        
        # Convert to RGB if not already (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize image if it's too large (for faster processing)
        width, height = image.size
        if max(width, height) > max_size:
            if width > height:
                new_width = max_size
                new_height = int(height * max_size / width)
            else:
                new_height = max_size
                new_width = int(width * max_size / height)
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # Convert to numpy array
        image_array = np.array(image)
        
        # Reshape to 2D array: (num_pixels, 3) for RGB values
        pixels = image_array.reshape(-1, 3)
        
        return image, pixels
        
    except Exception as e:
        st.error(f"Error loading image: {e}")
        return None, None


def find_dominant_colors(pixels, k=5):
    """
    Use K-Means clustering to find k dominant colors in the image.
    
    Args:
        pixels (numpy.ndarray): Array of RGB pixel values
        k (int): Number of clusters (dominant colors) to find
    
    Returns:
        numpy.ndarray: Array of dominant colors (RGB values)
        numpy.ndarray: Labels for each pixel
    """
    # Apply K-Means clustering
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    
    # Get the dominant colors (cluster centers)
    dominant_colors = kmeans.cluster_centers_
    
    # Ensure values are in valid RGB range [0, 255]
    dominant_colors = np.clip(dominant_colors, 0, 255).astype(int)
    
    return dominant_colors, labels


def rgb_to_hex(rgb):
    """
    Convert RGB values to hexadecimal color code.
    
    Args:
        rgb (tuple or list): RGB values (0-255)
    
    Returns:
        str: Hexadecimal color code
    """
    return "#{:02x}{:02x}{:02x}".format(int(rgb[0]), int(rgb[1]), int(rgb[2]))


def create_color_palette(dominant_colors):
    """
    Create a color palette visualization.
    
    Args:
        dominant_colors (numpy.ndarray): Array of dominant colors
    
    Returns:
        matplotlib.figure.Figure: Figure object with the palette
    """
    # Create figure for color palette
    fig, ax = plt.subplots(1, 1, figsize=(10, 2))
    
    # Create color palette
    palette_height = 100
    palette_width = 400
    color_width = palette_width // len(dominant_colors)
    
    palette = np.zeros((palette_height, palette_width, 3), dtype=np.uint8)
    
    for i, color in enumerate(dominant_colors):
        start_x = i * color_width
        end_x = (i + 1) * color_width
        palette[:, start_x:end_x] = color
    
    # Display color palette
    ax.imshow(palette)
    ax.set_title("Dominant Colors Palette", fontsize=16, fontweight='bold')
    ax.axis('off')
    
    plt.tight_layout()
    return fig


def create_download_link(fig, filename="palette.png"):
    """
    Create a download link for the color palette.
    
    Args:
        fig: matplotlib figure object
        filename (str): Name for the downloaded file
    
    Returns:
        str: HTML download link
    """
    # Save figure to bytes
    img_buffer = io.BytesIO()
    fig.savefig(img_buffer, format='png', dpi=150, bbox_inches='tight')
    img_buffer.seek(0)
    
    # Encode to base64
    img_b64 = base64.b64encode(img_buffer.read()).decode()
    
    # Create download link
    href = f'<a href="data:image/png;base64,{img_b64}" download="{filename}">Download Palette PNG</a>'
    return href


def display_color_info(dominant_colors):
    """
    Display color information in a formatted way.
    
    Args:
        dominant_colors (numpy.ndarray): Array of dominant colors
    """
    st.subheader("🎨 Color Information")
    
    # Create columns for better layout
    cols = st.columns(min(len(dominant_colors), 3))
    
    for i, color in enumerate(dominant_colors):
        col_idx = i % len(cols)
        
        with cols[col_idx]:
            rgb = tuple(color)
            hex_code = rgb_to_hex(color)
            
            # Create a colored box using HTML/CSS
            color_box = f"""
            <div style="
                width: 100px;
                height: 50px;
                background-color: {hex_code};
                border: 1px solid #ccc;
                border-radius: 5px;
                margin: 5px 0;
            "></div>
            """
            
            st.markdown(f"**Color {i+1}**")
            st.markdown(color_box, unsafe_allow_html=True)
            st.code(f"RGB: {rgb}")
            st.code(f"HEX: {hex_code}")
            st.markdown("---")


def main():
    """Main Streamlit application."""
    
    # Page configuration
    st.set_page_config(
        page_title="Dominant Color Finder",
        page_icon="🎨",
        layout="wide"
    )
    
    # Header
    st.title("🎨 Dominant Color Finder")
    st.markdown("Extract dominant colors from images using K-Means clustering")
    
    # Sidebar for controls
    st.sidebar.header("⚙️ Controls")
    
    # File upload
    uploaded_file = st.sidebar.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'gif', 'bmp', 'tiff'],
        help="Upload an image to analyze its dominant colors"
    )
    
    # Number of clusters slider
    num_clusters = st.sidebar.slider(
        "Number of Colors",
        min_value=2,
        max_value=15,
        value=5,
        help="Number of dominant colors to extract"
    )
    
    # Processing section
    if uploaded_file is not None:
        # Create two columns for layout
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("📸 Original Image")
            st.image(uploaded_file, use_column_width=True)
        
        # Process the image
        with st.spinner("🔍 Analyzing image colors..."):
            image, pixels = load_and_preprocess_image(uploaded_file)
            
            if image is not None and pixels is not None:
                # Find dominant colors
                dominant_colors, labels = find_dominant_colors(pixels, num_clusters)
                
                with col2:
                    st.subheader("🎯 Analysis Results")
                    
                    # Display image info
                    st.info(f"**Image Size:** {image.size[0]} x {image.size[1]} pixels")
                    st.info(f"**Colors Found:** {len(dominant_colors)}")
                    
                    # Create and display color palette
                    palette_fig = create_color_palette(dominant_colors)
                    st.pyplot(palette_fig)
                    
                    # Download button
                    download_link = create_download_link(palette_fig)
                    st.markdown(download_link, unsafe_allow_html=True)
                
                # Display color information
                display_color_info(dominant_colors)
                
                # Additional statistics
                st.subheader("📊 Statistics")
                col_stats1, col_stats2, col_stats3 = st.columns(3)
                
                with col_stats1:
                    st.metric("Total Pixels", f"{len(pixels):,}")
                
                with col_stats2:
                    st.metric("Clusters", num_clusters)
                
                with col_stats3:
                    avg_color = np.mean(dominant_colors, axis=0).astype(int)
                    st.metric("Avg Color", rgb_to_hex(avg_color))
                
                # Show raw data option
                if st.expander("🔍 Show Raw Color Data"):
                    st.subheader("RGB Values")
                    st.dataframe({
                        'Color': [f"Color {i+1}" for i in range(len(dominant_colors))],
                        'Red': dominant_colors[:, 0],
                        'Green': dominant_colors[:, 1],
                        'Blue': dominant_colors[:, 2],
                        'HEX': [rgb_to_hex(color) for color in dominant_colors]
                    })
    
    else:
        # Instructions when no file is uploaded
        st.info("👆 Please upload an image file using the sidebar to get started!")
        
        # Example section
        st.subheader("📋 How to Use")
        st.markdown("""
        1. **Upload an Image**: Use the file uploader in the sidebar
        2. **Choose Number of Colors**: Adjust the slider to set how many dominant colors to find
        3. **View Results**: See the original image, color palette, and detailed color information
        4. **Download**: Save the color palette as a PNG file
        
        **Supported formats:** PNG, JPG, JPEG, GIF, BMP, TIFF
        """)
        
        st.subheader("🎯 Features")
        st.markdown("""
        - **K-Means Clustering**: Advanced algorithm for accurate color analysis
        - **Visual Results**: Side-by-side comparison with color palette
        - **Multiple Formats**: RGB and HEX color codes
        - **Download Option**: Save your color palette
        - **Real-time Analysis**: Instant results as you adjust settings
        """)


if __name__ == "__main__":
    main()
