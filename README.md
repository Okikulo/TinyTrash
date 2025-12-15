# TinyTrash

Waste classification system using computer vision, PyTorch, and embedded systems. 
Classifies trash into 5 categories (glass, metal, paper, plastic, others) and 
automatically opens the corresponding bin lid using servo-controlled mechanisms.

**Complete system includes:**
- ML inference on laptop (MobileNetV2)
- Wio Terminal display and control hub
- PCA9685 PWM servo driver
- 4 automated trash bins with servo-actuated lids

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
- PlatformIO (for Wio Terminal firmware)
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
## Hardware Setup

### Wio Terminal Firmware

1. **Install PlatformIO:**
```bash
   pip install platformio
```

2. **Upload firmware:**
```bash
   cd arduino
   pio run --target upload
```

3. **Verify upload:**
   - LCD should show "TinyTrash Ready"
   - Serial monitor: `pio device monitor`

### Servo Wiring

**PCA9685 to Wio Terminal:**
- SDA → Wio Terminal SDA pin
- SCL → Wio Terminal SCL pin
- VCC → 5V
- GND → GND

**Servos to PCA9685:**
- Channel 0: Paper bin servo
- Channel 1: Metal bin servo
- Channel 2: Glass bin servo
- Channel 3: Plastic bin servo

**Power:**
- Servos require external 5V power supply (2A+)
- Connect to PCA9685 V+ and GND terminals
- **Do not power servos from Wio Terminal!**

### Mechanical Setup

Each trash bin requires:
- Hinged lid mechanism
- Servo horn attached to lid linkage
- 0° = closed, 90° = open
- Secure servo mounting to prevent vibration

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
- `f` - Toggle fullscreen (perfect for demos!)

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

## Performance Metrics

**Model Accuracy:**
- 4-category model: ~92% validation accuracy
- 5-category model: ~91% validation accuracy
- Inference time: ~50-100ms 

**System Response:**
- Classification: < 100ms
- Serial transmission: < 10ms
- Servo actuation: ~1 second (0° → 90°)
- Total time: ~1.5 seconds (capture to bin open)

**Tested on:**
- Laptop: NVIDIA 4060 laptop GPU / AMD Ryzen 9 CPU inference
- Wio Terminal: SAMD51 @ 120MHz
- Dataset: 1000 images across 5 categories

## Extra Utilities

- `converter.py` to ensure all the photos are in the a compatible format (JPG,JPEG,jpg,jpeg).
- `count.sh` to easily count per category and the total number of photos.
- `ziping.sh` to easily zip all the categories.

## Acknowledgments

Developed as part of Embedded Systems Design course at National Taiwan University of Science and Technology (NTUST).
Special thanks to David for being an awesome and funny professor :)
