# TinyTrash

Waste classification system using computer vision and PyTorch. Classifies trash into 4 categories: glass, metal, paper, and plastic.

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

### Using uv (recommended)

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

### Using pip

```bash
pip install torch torchvision opencv-python pillow
python inference.py
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

## Categories

- 🔵 **Glass** - Glass bottles, jars
- ⚪ **Metal** - Aluminum cans, metal containers
- 🟡 **Paper** - Cardboard, paper packaging
- 🔴 **Plastic** - Plastic bottles, containers

## Acknowledgments

Developed as part of Embedded Systems course at National Taiwan University of Science and Technology (NTUST).
