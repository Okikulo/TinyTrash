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

parser = argparse.ArgumentParser(description='TinyTrash Inference - Photo Capture Mode')
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
MODEL_PATH = 'tinytrash_model.pth'

# Class names (in alphabetical order - same as ImageFolder)
CATEGORIES = ['glass', 'metal', 'paper', 'plastic']

# Colors for each category (BGR format for OpenCV)
COLORS = {
    'glass': (255, 200, 0),      # Cyan
    'metal': (200, 200, 200),    # Silver/Gray
    'paper': (0, 255, 255),      # Yellow
    'plastic': (255, 0, 0)       # Blue
}

# Image size (must match training)
IMG_SIZE = 224

# Create output directory for captured images
os.makedirs('captures', exist_ok=True)

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
else:
    print("GPU: Not available - using CPU")

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

print(f"✓ Model ready! Classes: {CATEGORIES}\n")

# ============================================================================
# PREPROCESSING
# ============================================================================

# Match your training pipeline
preprocess = transforms.Compose([
    transforms.Resize((640, 480)),  # Match laptop camera resolution first
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
        category_name: Predicted class name
        confidence: Prediction confidence (0-1)
        all_probabilities: Dictionary with all class probabilities
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
        probabilities = torch.nn.functional.softmax(output, dim=1)[0]
        confidence, predicted = torch.max(probabilities, 0)
        
        category_idx = predicted.item()
        conf = confidence.item()
        category_name = CATEGORIES[category_idx]
        
        # Get all probabilities
        all_probs = {CATEGORIES[i]: probabilities[i].item() 
                     for i in range(len(CATEGORIES))}
    
    return category_name, conf, all_probs

# ============================================================================
# DISPLAY RESULTS
# ============================================================================

def display_results(frame, category, confidence, all_probs):
    """
    Create a results window with prediction details
    """
    # Create results display (480x480 to match camera preview height)
    results_img = np.zeros((480, 500, 3), dtype=np.uint8)
    
    # Header
    cv2.putText(results_img, "CLASSIFICATION RESULTS", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.line(results_img, (20, 50), (480, 50), (255, 255, 255), 2)
    
    # Main prediction
    color = COLORS[category]
    cv2.putText(results_img, "Predicted:", (20, 90), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    cv2.putText(results_img, category.upper(), (20, 125), 
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, color, 3)
    
    # Confidence
    cv2.putText(results_img, f"Confidence: {confidence*100:.1f}%", (20, 165), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    # Confidence bar
    bar_width = int(460 * confidence)
    cv2.rectangle(results_img, (20, 180), (20 + bar_width, 200), color, -1)
    cv2.rectangle(results_img, (20, 180), (480, 200), (255, 255, 255), 2)
    
    # All probabilities
    cv2.line(results_img, (20, 220), (480, 220), (255, 255, 255), 1)
    cv2.putText(results_img, "All Classes:", (20, 250), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
    
    # Sort by probability
    sorted_probs = sorted(all_probs.items(), key=lambda x: x[1], reverse=True)
    
    y_pos = 280
    for cat, prob in sorted_probs:
        # Category name
        cv2.putText(results_img, f"{cat.capitalize()}:", (30, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Probability bar
        bar_w = int(250 * prob)
        cv2.rectangle(results_img, (150, y_pos - 12), (150 + bar_w, y_pos + 2), 
                     COLORS[cat], -1)
        
        # Percentage
        cv2.putText(results_img, f"{prob*100:.1f}%", (410, y_pos), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        y_pos += 25
    
    # Instructions at bottom
    cv2.line(results_img, (20, 400), (480, 400), (100, 100, 100), 1)
    cv2.putText(results_img, "Press 'S' to save this result", (20, 430), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    cv2.putText(results_img, "Press 'C' to capture again", (20, 460), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    return results_img

# ============================================================================
# SERIAL COMMUNICATION
# ============================================================================

def init_serial(port, baudrate):
    """
    Initialize serial connection to Wio Terminal
    
    Args:
        port: Serial port name
        baudrate: Baud rate
        
    Returns:
        serial object or None if failed
    """
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
    Example: "GLASS:85.7\n"
    
    Args:
        ser: Serial object
        category: Predicted category name
        confidence: Confidence value (0-1)
    """
    if ser is None or not ser.is_open:
        return
    
    try:
        # Format message: "CATEGORY:CONFIDENCE\n"
        message = f"{category.upper()}:{confidence*100:.1f}\n"
        ser.write(message.encode('utf-8'))
        ser.flush()
        print(f"  → Sent to Wio: {message.strip()}")
    except Exception as e:
        print(f"  ✗ Serial send error: {e}")

# ============================================================================
# MAIN LOOP
# ============================================================================

def main():
    print("="*60)
    print("TinyTrash - Photo Capture Mode")
    print("="*60)
    print("Controls:")
    print("  'c' - Capture photo and classify")
    print("  's' - Save current result")
    print("  'q' - Quit")
    
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
        if ser:
            ser.close()
        return
    
    print("✓ Webcam opened successfully")
    print("Waiting for capture command...\n")
    
    # Set resolution to 640x480 to match training
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Position windows side by side
    cv2.namedWindow('Camera Preview')
    cv2.moveWindow('Camera Preview', 50, 50)
    
    # Last results (for display)
    last_results = None
    last_captured_frame = None
    last_category = None
    last_confidence = None
    last_timestamp = None
    capture_count = 0
    saved_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("✗ Error: Cannot read frame")
            break
        
        # Create a clean preview frame
        display_frame = frame.copy()
        
        # Add minimal instructions overlay
        overlay = display_frame.copy()
        cv2.rectangle(overlay, (0, display_frame.shape[0] - 40), 
                     (display_frame.shape[1], display_frame.shape[0]), 
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.7, display_frame, 0.3, 0, display_frame)
        
        cv2.putText(display_frame, "Press 'C' to capture | 'S' to save | 'Q' to quit", 
                   (10, display_frame.shape[0] - 15), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show camera preview
        cv2.imshow('Camera Preview', display_frame)
        
        # Show last results if available
        if last_results is not None:
            cv2.namedWindow('Results')
            cv2.moveWindow('Results', 700, 50)  # Position to the right
            cv2.imshow('Results', last_results)
        
        # Keyboard controls
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("\nQuitting...")
            break
            
        elif key == ord('c'):
            capture_count += 1
            print(f"\n{'='*60}")
            print(f"Capture #{capture_count}")
            print(f"{'='*60}")
            
            # Store the frame and timestamp for potential saving
            last_captured_frame = frame.copy()
            last_timestamp = int(time.time())
            
            # Run classification
            print("Running classification...")
            start_time = time.time()
            category, confidence, all_probs = predict_image(frame)
            inference_time = (time.time() - start_time) * 1000
            
            # Store category for saving
            last_category = category
            last_confidence = confidence
            
            # Send to Wio Terminal if enabled
            if ser:
                send_to_wio(ser, category, confidence)
            
            # Print results to console
            print(f"\n--- RESULTS ---")
            print(f"Predicted: {category.upper()}")
            print(f"Confidence: {confidence*100:.1f}%")
            print(f"Inference time: {inference_time:.1f}ms")
            print(f"\nAll probabilities:")
            for cat in sorted(all_probs.keys()):
                print(f"  {cat.capitalize()}: {all_probs[cat]*100:.1f}%")
            print(f"\nPress 'S' to save this result")
            print(f"{'='*60}\n")
            
            # Create results display
            last_results = display_results(frame, category, confidence, all_probs)
        
        elif key == ord('s'):
            if last_captured_frame is not None:
                saved_count += 1
                # Create organized folder structure
                category_folder = f"captures/{last_category}"
                os.makedirs(category_folder, exist_ok=True)
                
                filename = f"{category_folder}/capture_{last_timestamp}.jpg"
                cv2.imwrite(filename, last_captured_frame)
                print(f"✓ Image saved: {filename}")
                print(f"  Category: {last_category.upper()}")
                print(f"  Total saved: {saved_count}\n")
            else:
                print("! No capture to save. Press 'C' first to capture an image.\n")
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    if ser:
        ser.close()
        print("✓ Serial connection closed")
    
    print(f"\n✓ Session complete!")
    print(f"  Captures: {capture_count}")
    print(f"  Saved: {saved_count}")
    if saved_count > 0:
        print(f"  Saved images in: ./captures/")
        print(f"  Organized by category folders")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    main()
