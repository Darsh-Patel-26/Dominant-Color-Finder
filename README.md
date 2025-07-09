# Dominant Color Finder - Streamlit Web App

A beautiful web application that extracts dominant colors from images using K-Means clustering. This interactive tool analyzes any image and identifies the most prominent colors, displaying them in an intuitive web interface.

## Features

- 🎨 Extract dominant colors from any image format (JPG, PNG, etc.)
- 🔍 Uses K-Means clustering for accurate color analysis
- 🖥️ Interactive web interface built with Streamlit
- 📊 Visual output showing original image and color palette
- 🎯 Real-time adjustment of color count (2-15 colors)
- 💾 Download color palette as PNG file
- 📋 Display RGB and HEX values with color swatches
- 📊 Statistics and raw data viewing options

## Requirements

Make sure you have Python 3.6+ installed, then install the required packages:

```bash
pip install streamlit numpy scikit-learn matplotlib pillow
```

### Dependencies

- **streamlit**: For the web application interface
- **numpy**: For numerical operations and array handling
- **scikit-learn**: For K-Means clustering algorithm
- **matplotlib**: For creating visualizations and plots
- **PIL (Pillow)**: For image loading and processing

## Installation

1. Clone or download the project files
2. Install the required dependencies (see Requirements section)
3. You're ready to go!

## Usage

### Running the Web App

```bash
streamlit run dominant_colors.py
```

This will open your default web browser and launch the application at `http://localhost:8501`

### Using the Web Interface

1. **Upload an Image**: Use the file uploader in the sidebar
2. **Adjust Settings**: Use the slider to choose how many colors to extract (2-15)
3. **View Results**: See the original image, color palette, and detailed information
4. **Download**: Save the color palette as a PNG file
5. **Explore**: Check statistics and raw data in the expandable sections

### Supported File Formats

- PNG, JPG, JPEG, GIF, BMP, TIFF

## How It Works

1. **Image Upload**: Upload any image through the web interface
2. **Preprocessing**: Large images are automatically resized for faster processing
3. **K-Means Clustering**: The algorithm groups similar colors together
4. **Color Extraction**: Cluster centers represent the dominant colors
5. **Visualization**: Interactive display of original image and color palette
6. **Download**: Save the color palette as a PNG file

## Web Interface Features

The Streamlit app provides:

1. **Interactive Controls**: 
   - File uploader for images
   - Slider to adjust number of colors (2-15)
   
2. **Visual Results**:
   - Original image display
   - Color palette visualization
   - Individual color swatches with RGB/HEX values
   
3. **Data Export**:
   - Download color palette as PNG
   - View raw color data in table format
   
4. **Statistics**:
   - Total pixels analyzed
   - Number of clusters used
   - Average color calculation

## Tips for Best Results

- **Image Quality**: Higher quality images typically yield better color analysis
- **Cluster Count**: 
  - 3-5 clusters work well for simple images
  - 5-8 clusters are good for complex images
  - More than 10 clusters may produce too many similar colors
- **Image Size**: The program automatically resizes large images for faster processing
- **File Formats**: Supports JPG, PNG, and most common image formats

## Troubleshooting

### Common Issues

**"Error loading image"**
- Ensure the image file isn't corrupted
- Try a different image format
- Check that the file size isn't too large

**Slow processing**
- Large images are automatically resized for faster processing
- Try reducing the number of clusters if processing is slow

**Poor color results**
- Try adjusting the number of clusters
- Ensure the image has good color variation
- Consider using a higher quality image

### Performance Tips

- The app automatically resizes large images for optimal performance
- For best results, use images with clear color distinctions
- 3-8 clusters typically work well for most images

## Project Structure

```
dominant-color-finder/
├── dominant_colors.py    # Main Streamlit application
├── README.md            # This file
└── requirements.txt     # Dependencies (optional)
```

## Running in Production

To deploy the Streamlit app:

1. **Local Development**:
   ```bash
   streamlit run dominant_colors.py
   ```

2. **Streamlit Cloud**: 
   - Push to GitHub
   - Connect to Streamlit Cloud
   - Deploy automatically

3. **Docker**: 
   ```dockerfile
   FROM python:3.9-slim
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   COPY dominant_colors.py .
   EXPOSE 8501
   CMD ["streamlit", "run", "dominant_colors.py"]
   ```

## License

This project is open source and available under the MIT License.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve this project!
