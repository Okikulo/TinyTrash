from PIL import Image
import os
from pathlib import Path

# Register HEIF opener (for HEIC/HEIF support)
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # HEIF support optional

# Define your category folders
CATEGORIES = ['glass', 'paper', 'metal', 'plastic', 'others']
BASE_PATH = '.'  # Current directory

# Image formats to convert
IMAGE_FORMATS = [
    '.png', '.PNG',
    '.bmp', '.BMP', 
    '.gif', '.GIF',
    '.tiff', '.TIFF', '.tif', '.TIF',
    '.webp', '.WEBP',
    '.heic', '.HEIC',
    '.heif', '.HEIF'
]

# JPEG quality
JPEG_QUALITY = 95

def is_jpeg(filename):
    """Check if file is already JPEG"""
    ext = filename.rsplit('.', 1)[-1].lower() if '.' in filename else ''
    return ext in ['jpg', 'jpeg']

def convert_to_jpeg(folder_path):
    """Convert all non-JPEG images in folder to JPEG"""
    
    converted_count = 0
    failed_count = 0
    skipped_count = 0
    
    if not os.path.exists(folder_path):
        print(f"✗ Folder not found: {folder_path}")
        return converted_count, failed_count, skipped_count
    
    print(f"\n{'='*60}")
    print(f"Processing: {folder_path}")
    print(f"{'='*60}")
    
    # Get all image files (non-JPEG)
    files = os.listdir(folder_path)
    image_files = [f for f in files if any(f.endswith(ext) for ext in IMAGE_FORMATS)]
    
    if not image_files:
        print(f"ℹ️  No non-JPEG images found")
        return converted_count, failed_count, skipped_count
    
    print(f"Found {len(image_files)} images to convert\n")
    
    for i, filename in enumerate(image_files, 1):
        filepath = os.path.join(folder_path, filename)
        
        try:
            # Generate new JPEG filename
            new_filename = filename.rsplit('.', 1)[0] + '.jpg'
            new_filepath = os.path.join(folder_path, new_filename)
            
            if os.path.exists(new_filepath):
                print(f"[{i}/{len(image_files)}] ⏭️  SKIP: {filename} (already exists)")
                skipped_count += 1
                continue
            
            # Open and convert image
            img = Image.open(filepath)
            
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                rgb_img = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'RGBA':
                    rgb_img.paste(img, mask=img.split()[-1])
                else:
                    rgb_img.paste(img)
                img = rgb_img
            elif img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Save as JPEG
            img.save(new_filepath, 'JPEG', quality=JPEG_QUALITY)
            
            print(f"[{i}/{len(image_files)}] ✓ {filename} → {new_filename}")
            converted_count += 1
            
        except Exception as e:
            print(f"[{i}/{len(image_files)}] ✗ FAILED: {filename} - {str(e)}")
            failed_count += 1
    
    return converted_count, failed_count, skipped_count


def main():
    print("\n" + "="*60)
    print("TinyTrash Image → JPEG Batch Converter")
    print("="*60)
    
    total_converted = 0
    total_failed = 0
    total_skipped = 0
    
    for category in CATEGORIES:
        folder_path = os.path.join(BASE_PATH, category)
        converted, failed, skipped = convert_to_jpeg(folder_path)
        
        total_converted += converted
        total_failed += failed
        total_skipped += skipped
    
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"✓ Converted: {total_converted}")
    print(f"✗ Failed: {total_failed}")
    print(f"⏭️  Skipped: {total_skipped}")
    print(f"{'='*60}\n")
    
    if total_failed == 0 and total_converted > 0:
        print("✓ Conversion complete!\n")
    elif total_converted == 0:
        print("ℹ️  No files to convert.\n")


if __name__ == "__main__":
    main()
