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
parser.add_argument('--interval', type=int, default=5,
                    help='Serial update interval in seconds (default: 5)')
args = parser.parse_args()

# ============================================================================
# CONFIGURATION
# ============================================================================

# Path to your trained model (update this!)
MODEL_PATH = 'models/tinytrash_model_5_categories.pth'

if MODEL_PATH == 'models/tinytrash_model_4_categories.pth':    

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
CONFIDENCE_THRESHOLD = 0.0  # Only show predictions above this confidence

# Image size (must match training)
IMG_SIZE = 224

# Window sizes
CAMERA_WINDOW_WIDTH = 1280
CAMERA_WINDOW_HEIGHT = 720
RESULT_WINDOW_WIDTH = 800
RESULT_WINDOW_HEIGHT = 600

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
        return False
    
    try:
        message = f"{category.upper()}:{confidence*100:.1f}\n"
        ser.write(message.encode('utf-8'))
        ser.flush()
        print(f"  → Sent to Wio: {message.strip()}")
        return True
    except Exception as e:
        print(f"  ✗ Serial send error: {e}")
        return False

# ============================================================================
# DISPLAY FUNCTIONS
# ============================================================================

def create_results_display(prediction, confidence, probabilities, frame_shape=(480, 640, 3)):
    """Create results display image matching camera frame size"""
    # Create black canvas same size as camera frame
    results_img = np.zeros(frame_shape, dtype=np.uint8)
    
    h, w = results_img.shape[:2]
    
    # Get color for current category
    color = COLORS.get(prediction, (255, 255, 255))
    
    # Title
    cv2.putText(results_img, "RESULTS", (20, 40), 
               cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    
    # Main prediction
    if confidence > CONFIDENCE_THRESHOLD:
        label = f"{prediction.upper()}"
        conf_text = f"{confidence*100:.1f}%"
        
        cv2.putText(results_img, label, (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.8, color, 3)
        cv2.putText(results_img, conf_text, (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
    else:
        cv2.putText(results_img, "UNCERTAIN", (20, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 165, 255), 3)
        cv2.putText(results_img, f"{confidence*100:.1f}%", (20, 160), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 165, 255), 2)
    
    # Confidence bar
    bar_width = int((w - 40) * confidence)
    bar_color = color if confidence > CONFIDENCE_THRESHOLD else (100, 100, 100)
    cv2.rectangle(results_img, (20, 180), (20 + bar_width, 200), bar_color, -1)
    cv2.rectangle(results_img, (20, 180), (w - 20, 200), (255, 255, 255), 2)
    
    # All class probabilities
    cv2.putText(results_img, "All Classes:", (20, 240), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    for idx, category in enumerate(CATEGORIES):
        prob = probabilities[idx].item()
        y_pos = 270 + (idx * 35)
        
        # Category name
        cv2.putText(results_img, f"{category.capitalize()}:", (30, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Probability bar
        bar_w = int(300 * prob)
        cv2.rectangle(results_img, (170, y_pos - 15), (170 + bar_w, y_pos), 
                     COLORS[category], -1)
        
        # Percentage
        cv2.putText(results_img, f"{prob*100:.1f}%", (490, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return results_img

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("\n" + "="*60)
    print("TinyTrash Inference - Live Mode (Combined View)")
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
        print(f"  Update interval: {args.interval} seconds")
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
    
    # Set resolution to 640x480 to match training
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Create single combined window
    cv2.namedWindow('TinyTrash Live')
    cv2.resizeWindow('TinyTrash Live', 1600, 600)
    
    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps = 0
    
    # Serial timing
    last_serial_time = time.time()
    serial_interval = args.interval  # seconds
    next_serial_update = last_serial_time + serial_interval
    
    # Pause flag
    paused = False
    
    # Last prediction (for smooth display)
    last_prediction = None
    last_confidence = 0.0
    last_probabilities = torch.zeros(len(CATEGORIES))
    
    while True:
        if not paused:
            ret, frame = cap.read()
            
            if not ret:
                print("✗ Error: Cannot read frame")
                break
            
            # Run inference
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
                last_probabilities = probabilities
            
            # Check if it's time to send serial update
            current_time = time.time()
            if ser and current_time >= next_serial_update:
                print(f"\n[{time.strftime('%H:%M:%S')}] Serial Update:")
                if send_to_wio(ser, last_prediction, last_confidence):
                    next_serial_update = current_time + serial_interval
                    last_serial_time = current_time
            
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
        # CREATE CAMERA DISPLAY
        # ====================================================================
        
        display_frame = frame.copy()
        
        # Add clean overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, display_frame.shape[0] - 60), 
                     (display_frame.shape[1], display_frame.shape[0]), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        # FPS and status
        status_text = "PAUSED" if paused else "LIVE"
        status_color = (0, 0, 255) if paused else (0, 255, 0)
        cv2.putText(display_frame, f"FPS: {fps:.1f} | {status_text}", 
                   (10, display_frame.shape[0] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 2)
        
        # Serial countdown
        if ser:
            time_to_next = max(0, next_serial_update - time.time())
            cv2.putText(display_frame, f"Next: {time_to_next:.1f}s", 
                       (10, display_frame.shape[0] - 10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Controls
        cv2.putText(display_frame, "P: Pause | S: Save | Q: Quit", 
                   (display_frame.shape[1] - 260, display_frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # ====================================================================
        # CREATE RESULTS DISPLAY
        # ====================================================================
        
        if last_prediction is not None:
            results_display = create_results_display(last_prediction, last_confidence, 
                                                     last_probabilities, frame.shape)
        else:
            # Placeholder
            results_display = np.zeros_like(frame)
            cv2.putText(results_display, "Processing...", 
                       (results_display.shape[1]//2 - 100, results_display.shape[0]//2),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (128, 128, 128), 2)
        
        # ====================================================================
        # COMBINE AND DISPLAY
        # ====================================================================
        
        # Both frames are same size (640x480), combine directly
        combined = np.hstack([display_frame, results_display])
        
        # Show combined window
        cv2.imshow('TinyTrash Live', combined)
        
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
            timestamp = int(time.time())
            cv2.imwrite(f"./screenshots/screenshot_{timestamp}.jpg", combined)
            print(f"✓ Screenshot saved: screenshot_{timestamp}.jpg")
    
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
