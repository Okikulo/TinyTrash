import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import transforms
import cv2
import numpy as np
from PIL import Image
import time
import os
import argparse

# Try to import pyserial (optional for Wio Terminal communication)
try:
    import serial
    SERIAL_AVAILABLE = True
except ImportError:
    SERIAL_AVAILABLE = False

# ============================================================================
# COMMAND LINE ARGUMENTS
# ============================================================================

parser = argparse.ArgumentParser(description='TinyTrash Inference - Live Mode')
parser.add_argument('--serial', action='store_true', 
                    help='Enable serial communication with Wio Terminal')
parser.add_argument('--port', type=str, default=None,
                    help='Serial port (e.g., COM3 on Windows, /dev/ttyACM0 on Linux)')
parser.add_argument('--baudrate', type=int, default=115200,
                    help='Serial baudrate (default: 115200)')
args = parser.parse_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your trained model (update this!)
MODEL_PATH = 'models/tinytrash_model_5.pth'

if MODEL_PATH == 'models/tinytrash_model_4.pth':    

    # Class names (in alphabetical order - same as ImageFolder)
    CATEGORIES = ['glass', 'metal', 'paper', 'plastic']
    
    # Colors for each category (BGR format for OpenCV)
    COLORS = {
        'glass': (255, 200, 0),      # Cyan
        'metal': (200, 200, 200),    # Silver/Gray
        'paper': (0, 255, 255),      # Yellow
        'plastic': (255, 0, 0)       # Blue
    }
else:
    # Class names (in alphabetical order - same as ImageFolder)
    CATEGORIES = ['glass', 'metal', 'others', 'paper', 'plastic']
    
    # Colors for each category (BGR format for OpenCV)
    COLORS = {
        'glass': (255, 200, 0),      # Cyan
        'metal': (200, 200, 200),    # Silver/Gray
        'others': (128, 128, 128),   # Dark Gray
        'paper': (0, 255, 255),      # Yellow
        'plastic': (255, 0, 0)       # Blue
    }

# Confidence threshold
CONFIDENCE_THRESHOLD = 0.7  # Only show predictions above this confidence

# Image size (must match training)
IMG_SIZE = 224

# Create output directory for screenshots
os.makedirs('screenshots', exist_ok=True)

# ============================================================================
# LOAD MODEL
# ============================================================================

print("Loading model...")

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# Print hardware info
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
else:
    print("GPU: Not available - using CPU only")
    print("Note: Inference will be slower on CPU")

# Create model architecture (same as training)
model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(CATEGORIES))

# Load trained weights
try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"✓ Model loaded from: {MODEL_PATH}")
except FileNotFoundError:
    print(f"✗ Error: Model file not found at {MODEL_PATH}")
    print("Please update MODEL_PATH to point to your trained model.")
    exit(1)

# Set to evaluation mode
model.to(device)
model.eval()

print(f"✓ Model ready! Classes: {CATEGORIES}")

# ============================================================================
# PREPROCESSING
# ============================================================================

# Same preprocessing as training (without augmentation)
preprocess = transforms.Compose([
    transforms.Resize((640, 480)),
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                        std=[0.229, 0.224, 0.225])
])

# ============================================================================
# INFERENCE FUNCTION
# ============================================================================

def predict_image(frame):
    """
    Run inference on a single frame
    
    Args:
        frame: OpenCV BGR image
        
    Returns:
        category_idx: Predicted class index
        category_name: Predicted class name
        confidence: Prediction confidence (0-1)
    """
    # Convert BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(rgb_frame)
    
    # Preprocess
    input_tensor = preprocess(pil_image)
    input_batch = input_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_batch)
        probabilities = torch.nn.functional.softmax(output, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
        category_idx = predicted.item()
        conf = confidence.item()
        category_name = CATEGORIES[category_idx]
    
    return category_idx, category_name, conf

# ============================================================================
# SERIAL COMMUNICATION
# ============================================================================

def init_serial(port, baudrate):
    """Initialize serial connection to Wio Terminal"""
    if not SERIAL_AVAILABLE:
        print("✗ pyserial not installed!")
        print("  Install with: pip install pyserial")
        print("  Or: python3 -m pip install pyserial")
        return None
    
    if port is None:
        print("✗ No serial port specified. Use --port argument.")
        return None
    
    try:
        ser = serial.Serial(port, baudrate, timeout=1)
        time.sleep(2)  # Wait for Arduino to reset
        print(f"✓ Serial connection established: {port} @ {baudrate} baud")
        return ser
    except serial.SerialException as e:
        print(f"✗ Failed to open serial port {port}: {e}")
        print("  Continuing without serial communication...")
        return None
    except Exception as e:
        print(f"✗ Serial error: {e}")
        return None


def send_to_wio(ser, category, confidence):
    """
    Send classification result to Wio Terminal
    Protocol: "CATEGORY:CONFIDENCE\n"
    """
    if ser is None or not ser.is_open:
        return
    
    try:
        message = f"{category.upper()}:{confidence*100:.1f}\n"
        ser.write(message.encode('utf-8'))
        ser.flush()
    except Exception as e:
        pass  # Silent fail in live mode to avoid spam

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("\n" + "="*60)
    print("TinyTrash Inference - Live Mode")
    print("="*60)
    print("Controls:")
    print("  'q' - Quit")
    print("  'p' - Pause/unpause")
    print("  's' - Save screenshot")
    
    # Show serial status
    if args.serial:
        print(f"\nSerial: ENABLED")
        print(f"  Port: {args.port if args.port else 'Not specified'}")
        print(f"  Baudrate: {args.baudrate}")
    else:
        print(f"\nSerial: DISABLED (use --serial to enable)")
    
    print("="*60 + "\n")
    
    # Initialize serial connection if requested
    ser = None
    if args.serial:
        ser = init_serial(args.port, args.baudrate)
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("✗ Error: Cannot open webcam")
        return
    
    print("✓ Webcam opened successfully")
    print("Starting inference...\n")
    
    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Pause flag
    paused = False
    
    # Last prediction (for smooth display)
    last_prediction = None
    last_confidence = 0.0
    last_probabilities = torch.zeros(len(CATEGORIES))  # Initialize probabilities
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Error: Cannot read frame")
                break
            
            # Run inference ONCE and get all probabilities
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            input_tensor = preprocess(pil_image)
            input_batch = input_tensor.unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(input_batch)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                confidence, predicted = torch.max(probabilities, 0)
                
                category_idx = predicted.item()
                last_confidence = confidence.item()
                last_prediction = CATEGORIES[category_idx]
                last_probabilities = probabilities  # Store for later use
                
                # Send to Wio Terminal if enabled
                if ser:
                    send_to_wio(ser, last_prediction, last_confidence)
            
            # FPS calculation
            fps_counter += 1
            if fps_counter % 10 == 0:
                fps_end_time = time.time()
                fps = 10 / (fps_end_time - fps_start_time)
                fps_start_time = time.time()
        else:
            # If paused, just read frame without inference
            ret, frame = cap.read()
            if not ret:
                break
        
        # ====================================================================
        # DRAW ON FRAME
        # ====================================================================
        
        # Create overlay for better readability
        overlay = frame.copy()
        
        # Get color for current category
        color = COLORS.get(last_prediction, (255, 255, 255))
        
        # Draw prediction box (top section)
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Draw prediction text
        if last_confidence > CONFIDENCE_THRESHOLD:
            # High confidence - show prediction
            label = f"{last_prediction.upper()}"
            conf_text = f"{last_confidence*100:.1f}%"
            
            cv2.putText(frame, label, (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
            cv2.putText(frame, conf_text, (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        else:
            # Low confidence - show "uncertain"
            cv2.putText(frame, "UNCERTAIN", (20, 50), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
            cv2.putText(frame, f"{last_confidence*100:.1f}%", (20, 90), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps:.1f}", (frame.shape[1]-150, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Draw status
        if paused:
            cv2.putText(frame, "PAUSED", (frame.shape[1]-150, 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Draw confidence bar
        bar_width = int((frame.shape[1] - 40) * last_confidence)
        bar_color = color if last_confidence > CONFIDENCE_THRESHOLD else (100, 100, 100)
        cv2.rectangle(frame, (20, 105), (20 + bar_width, 115), bar_color, -1)
        cv2.rectangle(frame, (20, 105), (frame.shape[1] - 20, 115), (255, 255, 255), 2)
        
        # Draw all class probabilities (bottom section)
        y_offset = frame.shape[0] - 120
        cv2.rectangle(frame, (0, y_offset), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
        
        # Use the stored probabilities from the single inference run
        for idx, category in enumerate(CATEGORIES):
            prob = last_probabilities[idx].item()
            y_pos = y_offset + 20 + (idx * 25)
            
            # Draw category name
            cv2.putText(frame, f"{category}:", (20, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Draw probability bar
            bar_w = int(200 * prob)
            cv2.rectangle(frame, (120, y_pos - 12), (120 + bar_w, y_pos), 
                         COLORS[category], -1)
            
            # Draw percentage
            cv2.putText(frame, f"{prob*100:.1f}%", (330, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display frame
        cv2.imshow('TinyTrash Inference', frame)
        
        # ====================================================================
        # KEYBOARD CONTROLS
        # ====================================================================
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
        elif key == ord('p'):
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord('s'):
            filename = f"./screenshots/screenshot_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"✓ Screenshot saved: {filename}")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if ser:
        ser.close()
        print("✓ Serial connection closed")
    
    print("\n✓ Inference stopped")

# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
