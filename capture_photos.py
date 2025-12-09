"""
TinyTrash Dataset Photo Capture Tool
=====================================
Easy photo capture for creating "others" category dataset.
Works on Windows, Linux, and macOS.

Requirements:
    pip install opencv-python

Usage:
    python capture_photos.py
    
Controls:
    SPACE - Take photo
    Q     - Quit
"""

import cv2
import os
from datetime import datetime

# ============================================================================
# CONFIGURATION
# ============================================================================

# Output folder (will be created if doesn't exist)
OUTPUT_FOLDER = "dataset/metal"

# Image resolution
CAMERA_WIDTH = 1280
CAMERA_HEIGHT = 720

# Save image resolution (can be different from camera)
SAVE_WIDTH = 640
SAVE_HEIGHT = 480

# Photo counter
photo_count = 0

# ============================================================================
# SETUP
# ============================================================================

print("="*60)
print("TinyTrash Dataset Photo Capture Tool")
print("="*60)
print(f"\nPhotos will be saved to: ./{OUTPUT_FOLDER}/")
print("\nControls:")
print("  SPACE - Take photo")
print("  Q     - Quit")
print("\nTips for 'others' category:")
print("  - Take photos of random household items")
print("  - Include books, phones, clothes, furniture")
print("  - Take photos of empty backgrounds (tables, walls)")
print("  - Include your hands/arms (they appear in real usage)")
print("  - Capture food items and packaging")
print("  - Aim for 100-200 diverse photos")
print("="*60 + "\n")

# Create output folder if it doesn't exist
if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)
    print(f"✓ Created folder: {OUTPUT_FOLDER}/")
else:
    # Count existing photos
    existing_photos = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith('.jpg')]
    photo_count = len(existing_photos)
    print(f"✓ Found {photo_count} existing photos in {OUTPUT_FOLDER}/")

print("\nStarting webcam...\n")

# ============================================================================
# OPEN WEBCAM
# ============================================================================

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("✗ Error: Cannot open webcam")
    print("\nTroubleshooting:")
    print("  1. Make sure no other app is using the webcam")
    print("  2. Check camera permissions in Windows Settings")
    print("  3. Try unplugging and replugging your webcam")
    exit(1)

print("✓ Webcam opened successfully")

# Set camera resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_WIDTH)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_HEIGHT)

# Get actual resolution (camera might not support requested resolution)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"Camera resolution: {actual_width}x{actual_height}")

# Create window
cv2.namedWindow('Photo Capture - Press SPACE to capture, Q to quit', cv2.WINDOW_NORMAL)

print("\nReady to capture! Press SPACE to take photos.\n")

# ============================================================================
# MAIN LOOP
# ============================================================================

last_capture_time = 0

while True:
    ret, frame = cap.read()
    
    if not ret:
        print("✗ Error: Cannot read frame")
        break
    
    # Create display frame with instructions
    display_frame = frame.copy()
    
    # Draw semi-transparent black bar at top
    overlay = display_frame.copy()
    cv2.rectangle(overlay, (0, 0), (display_frame.shape[1], 80), (0, 0, 0), -1)
    display_frame = cv2.addWeighted(overlay, 0.6, display_frame, 0.4, 0)
    
    # Draw instructions
    cv2.putText(display_frame, f"Photos captured: {photo_count}", 
                (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display_frame, "Press SPACE to capture | Q to quit", 
                (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    
    # Draw target box (helps with framing)
    center_x = display_frame.shape[1] // 2
    center_y = display_frame.shape[0] // 2
    box_size = 300
    cv2.rectangle(display_frame, 
                  (center_x - box_size, center_y - box_size),
                  (center_x + box_size, center_y + box_size),
                  (0, 255, 0), 2)
    
    # Show frame
    cv2.imshow('Photo Capture - Press SPACE to capture, Q to quit', display_frame)
    
    # Handle keyboard input
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('q') or key == ord('Q'):
        print("\nQuitting...")
        break
    
    elif key == ord(' '):  # SPACE key
        # Prevent accidental rapid captures
        current_time = datetime.now().timestamp()
        if current_time - last_capture_time < 0.5:  # 0.5 second cooldown
            continue
        last_capture_time = current_time
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{OUTPUT_FOLDER}/others_{timestamp}_{photo_count:04d}.jpg"
        
        # Resize frame if needed
        if SAVE_WIDTH != actual_width or SAVE_HEIGHT != actual_height:
            save_frame = cv2.resize(frame, (SAVE_WIDTH, SAVE_HEIGHT))
        else:
            save_frame = frame
        
        # Save photo
        cv2.imwrite(filename, save_frame)
        photo_count += 1
        
        print(f"✓ Photo #{photo_count} saved: {filename}")
        
        # Visual feedback - flash the screen
        white_frame = display_frame.copy()
        white_frame[:] = (255, 255, 255)
        cv2.imshow('Photo Capture - Press SPACE to capture, Q to quit', white_frame)
        cv2.waitKey(100)  # Show white flash for 100ms

# ============================================================================
# CLEANUP
# ============================================================================

cap.release()
cv2.destroyAllWindows()

print("\n" + "="*60)
print("Photo Capture Complete!")
print("="*60)
print(f"Total photos captured: {photo_count}")
print(f"Photos saved to: ./{OUTPUT_FOLDER}/")
print("\nNext steps:")
print(f"  1. Review photos in {OUTPUT_FOLDER}/ folder")
print("  2. Delete any blurry or bad photos")
print("  3. Zip the folder: {OUTPUT_FOLDER}.zip")
print("  4. Upload to Google Drive for training")
print("="*60)
