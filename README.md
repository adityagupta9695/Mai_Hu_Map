# Anime Scene to Mapillary Matcher

Find real-world street-level images from Mapillary that match scenes from anime or other images using AI-powered panoptic segmentation.

## 🎯 What It Does

This tool analyzes an input image (like an anime scene) using Meta's Mask2Former panoptic segmentation model, detects objects and scene elements, then searches Mapillary's database to find real-world street-view images with similar compositions.

**Example Use Case**: Input an anime screenshot of a street with a bicycle, utility pole, and buildings → Get real-world Mapillary images with the same elements.

## 🔍 How It Works

The pipeline runs in 3 stages:

1. **Stage 1: Panoptic Segmentation**
   - Analyzes your input image using AI
   - Detects objects (bicycles, poles, signs) and areas (buildings, vegetation, sky)
   - Maps detections to Mapillary's tag system

2. **Stage 2: Point Feature Search**
   - Searches Mapillary for images containing all detected point features
   - Uses grid-based spatial search for efficiency
   - Returns candidate images

3. **Stage 3: Area Feature Filtering**
   - Filters candidates by area/segmentation tags
   - Returns final matches with all required elements

## 📋 Prerequisites

- Python 3.8 or higher
- CUDA-capable GPU (optional, but recommended for faster processing)
- Mapillary API access token ([Get one here](https://www.mapillary.com/developer))

## 🚀 Installation

1. **Clone the repository**
```bash
git clone https://github.com/adityagupta9695/Mai_Hu_Map.git
cd Mai_Hu_Map
```

2. **Create a virtual environment** (recommended)
```bash
python -m venv mapillary_env
source mapillary_env/bin/activate  # On Windows: mapillary_env\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up your Mapillary API token**
```bash
export MAPILLARY_API_KEY='your_token_here'  # On Windows: set MAPILLARY_API_KEY=your_token_here
```

## 💻 Usage

### Basic Command

```bash
python anime_to_mapillary_pipeline.py <image_path> --bbox <min_lon> <min_lat> <max_lon> <max_lat>
```

### Examples

**Search for matches in Delhi area:**
```bash
python anime_to_mapillary_pipeline.py scene.jpg --bbox 77.0 28.5 77.2 28.7
```

**Custom output file:**
```bash
python anime_to_mapillary_pipeline.py scene.jpg --bbox 77.0 28.5 77.2 28.7 -o my_results.json
```

**Skip visualization generation:**
```bash
python anime_to_mapillary_pipeline.py scene.jpg --bbox 77.0 28.5 77.2 28.7 --no-visualization
```

**Pass token directly (not recommended for security):**
```bash
python anime_to_mapillary_pipeline.py scene.jpg --bbox 77.0 28.5 77.2 28.7 --token YOUR_TOKEN
```

### Command-Line Arguments

| Argument | Required | Description |
|----------|----------|-------------|
| `image` | Yes | Path to input image |
| `--bbox` | Yes | Bounding box coordinates (min_lon min_lat max_lon max_lat) |
| `--token` | No | Mapillary API token (or use `MAPILLARY_API_KEY` env var) |
| `--output`, `-o` | No | Output JSON filename (default: `matched_images.json`) |
| `--no-visualization` | No | Skip saving segmentation visualization |

### Finding Bounding Box Coordinates

Use [bboxfinder.com](http://bboxfinder.com/) to easily find coordinates for your area of interest.

## 📁 Project Structure

```
Mai_Hu_Map/
├── anime_to_mapillary_pipeline.py   # Main script
├── pretty_to_api_tags.json          # Model labels → Mapillary tags mapping
├── requirements.txt                 # Python dependencies
├── README.md                        # This file
├── .gitignore                       # Git ignore rules
```

## 📊 Output

The script generates:

1. **Segmentation Visualization**: `<input_filename>_segmented.jpg`
   - Visual overlay showing detected objects and areas

2. **Results JSON**: `matched_images.json` (or custom name)
   - List of Mapillary image IDs that match your criteria
   - Direct links to view images on Mapillary

3. **Console Output**: Detailed progress and results
   - Detected objects in input image
   - Tag mapping results
   - Search progress
   - Final match count and links

## 🔧 Configuration

### Adjusting Search Parameters

Edit these constants in `anime_to_mapillary_pipeline.py`:

```python
LON_DIVISIONS = 10          # Grid divisions (longitude)
LAT_DIVISIONS = 10          # Grid divisions (latitude)
CONCURRENCY_LIMIT = 20      # Parallel API requests
MAX_IMAGES_TO_ANALYZE = 50  # Max images to check in Stage 3
```

### Model Configuration

The script uses `facebook/mask2former-swin-large-mapillary-vistas-panoptic` by default. This model is optimized for street-level scene understanding.

## ⚠️ Limitations

- API rate limits apply (respect Mapillary's terms of service)
- Large bounding boxes may require many API calls
- Specific object combinations may yield few results
- Best results with street-level scene images
- GPU recommended for faster segmentation (CPU works but is slower)

## 📝 License

MIT License - feel free to use this project for any purpose.

## 🙏 Acknowledgments

- [Meta AI's Mask2Former](https://github.com/facebookresearch/Mask2Former) for panoptic segmentation
- [Mapillary](https://www.mapillary.com/) for street-level imagery and API
- [Hugging Face Transformers](https://huggingface.co/transformers/) for model implementation

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📧 Contact

Your Name - Aditya Gupta

Project Link: [https://github.com/adityagupta9695/Mai_Hu_Map.git](https://github.com/adityagupta9695/Mai_Hu_Map.git)

## 🔮 Future Improvements

- [ ] Add similarity scoring for ranked results
- [ ] Support for custom segmentation models
- [ ] Web interface for easier usage
- [ ] Batch processing multiple images
- [ ] Caching to avoid redundant API calls
- [ ] Visual comparison tool for results
