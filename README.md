# TinyTrash

Waste classification system using computer vision and PyTorch. Classifies trash into 4 categories: glass, metal, paper, and plastic.

---

## Dataset Collection - "Others" Category

The "others" category helps the model distinguish waste from non-waste items, reducing false positives.

### What to Include

Capture ~150 diverse photos of:
- Random household items (books, phones, remote controls, cups)
- Empty backgrounds (tables, walls, floors)
- Hands and arms (they appear during real usage)
- Food items (fruits, vegetables, packaged food)
- Furniture and electronics

### Using the Photo Capture Tool
```bash
# Install requirements
pip install opencv-python

# Run the capture script
python capture_photos.py
```

**Controls:**
- `SPACE` - Take photo
- `Q` - Quit

Photos are saved to `others` folder. After capturing, zip the folder and upload to your google drive

---
## Features

- Real-time waste classification using webcam
- MobileNetV2-based model for efficient inference
- Visual feedback with confidence scores
- Cross-platform support (Linux, Windows, macOS)

## Requirements

- Python 3.8+
- Webcam

## Installation

> *Note:* The `requirements.txt` and `uv.lock` dependecies are only for the `inference.py` script.

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/yourusername/tinytrash.git
cd tinytrash

# Install dependencies
uv sync

# Run inference
uv run python inference.py
```

## Usage

1. Place your trained model file (`tinytrash_model.pth`) in the project directory
2. Run the inference script:

```bash
python inference.py
```

3. Point your webcam at waste items to classify them

### Controls

- `q` - Quit
- `p` - Pause/unpause
- `s` - Save screenshot

## Model Training

The model uses MobileNetV2 architecture fine-tuned on a custom dataset of waste images categorized into 4 classes.
Refer to `tinytrash.ipynb` for more information.

## Categories

- 🔵 **Glass** - Glass bottles, jars
- ⚪ **Metal** - Aluminum cans, metal containers
- 🟡 **Paper** - Cardboard, paper packaging
- 🔴 **Plastic** - Plastic bottles, containers

---

## Acknowledgments

Developed as part of Embedded Systems course at National Taiwan University of Science and Technology (NTUST).
