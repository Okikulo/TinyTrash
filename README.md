# TinyTrash

Waste classification system using computer vision and PyTorch. Classifies trash into 5 categories: glass, metal, paper, plastic and others.

---

## Model Training

The model uses MobileNetV2 architecture fine-tuned on a custom dataset of waste images categorized into 4 or 5 classes.
Refer to `tinytrash.ipynb` for more information.

## Categories

- 🔵 **Glass** - Glass bottles, jars
- ⚪ **Metal** - Aluminum cans, metal containers
- 🟡 **Paper** - Cardboard, paper packaging
- 🔴 **Plastic** - Plastic bottles, containers
- 🟤 **Others** - Anything that does not fall in the previous four categories (*work in progress*)

---

## Features

- Real-time waste classification using webcam
- MobileNetV2-based model for efficient inference
- Visual feedback with confidence scores
- Cross-platform support (Linux, Windows, macOS)

## Requirements

- Python 3.8+
- Webcam
- `requirements.txt`

## Installation

```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone repository
git clone https://github.com/Okikulo/TinyTrash.git
cd TinyTrash

# Create virtual env (optional)
uv venv

# Enter the virtual venv (if created)
source .venv/bin/activate

# Install dependencies
uv sync
```
## Usage

#### Live Mode (`inference_live.py`)
Continuous real-time classification with visual overlay.
```bash
python inference_live.py

# With optional Wio Terminal display
python inference_live.py --serial --port COM3  # Windows
python inference_live.py --serial --port /dev/ttyACM0  # Linux
```

**Features:**
- Real-time classification
- FPS counter
- All class probabilities displayed
- Screenshots saved to `screenshots/` folder

#### Capture Mode (`inference_capture.py`)
Capture and classify individual photos on demand.
```bash
python inference_capture.py

# With optional Wio Terminal display
python inference_capture.py --serial --port COM3
```

**Features:**
- Clean camera preview
- Capture on keypress
- Selective saving by category
- Images organized in `captures/{category}/` folders
- Results in separate window

### Controls

**Live Mode (`inference_live.py`):**
- `p` - Pause/unpause inference
- `s` - Save screenshot
- `q` - Quit

**Capture Mode (`inference_capture.py`):**
- `c` - Capture photo and classify
- `s` - Save current classification
- `q` - Quit

### Command-Line Options

Both scripts support optional Wio Terminal communication:
```bash
--serial              # Enable serial communication
--port PORT           # Serial port (e.g., COM3, /dev/ttyACM0)
--baudrate BAUDRATE   # Serial baudrate (default: 115200)
```

**Example:**
```bash
python inference_capture.py --serial --port COM3 --baudrate 115200
```

## Dataset Collection  

The `capture_photos.py` script can help you collect data easily to add or make your own dataset. The script runs in Windows, macOS and Linux without problems.

### Using the Photo Capture Tool

```bash
# Run the capture script, category argument is positional
python capture_photos.py category

# Example
python capture_photos.py glass

# For more information
python capture_photos.py --help
```
### Controls
- `SPACE` - Take photo
- `Q` - Quit

> **Work in Progress:** The "others" category helps the model distinguish waste from non-waste items, reducing false positives. This will be removed after the final demo

### What to Include

Capture ~150 diverse photos of:
- Random household items (books, phones, remote controls, cups)
- Empty backgrounds (tables, walls, floors)
- Hands and arms (they appear during real usage)
- Food items (fruits, vegetables, packaged food)
- Furniture and electronics

---

## Extra Utilities

- `converter.py` to ensure all the photos are in the a compatible format (JPG,JPEG,jpg,jpeg).
- `count.sh` to easily count per category and the total number of photos.
- `ziping.sh` to easily zip all the categories.

## Acknowledgments

Developed as part of Embedded Systems Design course at National Taiwan University of Science and Technology (NTUST).
Special thanks to David for being an awesome and funny professor :)
