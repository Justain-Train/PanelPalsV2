#!/usr/bin/env python3
"""
Test the complete pipeline with multiple panels
"""
import requests
import sys

# Configuration
API_URL = "http://localhost:8000/process/chapter"
CHAPTER_ID = "webtoon_full_chapter"

# Image files
image_files = [f"screenshots/chapter_01/panel_{i}.png" for i in range(1,150)]
#image_files = [f"screenshots/chapter_01/panel_103.png"]

print(f"📸 Testing pipeline with {len(image_files)} panels")
print(f"📦 Chapter ID: {CHAPTER_ID}")
print(f"🔗 Endpoint: {API_URL}")
print("-" * 60)

# Prepare multipart form data
files = []
for img_path in image_files:
    try:
        with open(img_path, 'rb') as f:
            img_bytes = f.read()
            files.append(('images', (img_path, img_bytes, 'image/png')))
        print(f"✅ Loaded: {img_path} ({len(img_bytes):,} bytes)")
    except FileNotFoundError:
        print(f"❌ ERROR: File not found: {img_path}")
        sys.exit(1)

print("-" * 60)
print("⏳ Sending request to pipeline...")

# Prepare form data
data = {
    "chapter_id": CHAPTER_ID
}

# Send request
try:
    response = requests.post(
        API_URL,
        data=data,
        files=files,
        timeout=300  # 5 minute timeout for multiple panels
    )
    
    print(f"📡 Response Status: {response.status_code}")
    print("-" * 60)
    
    if response.status_code == 200:
        # Save the MP3 file
        output_path = f"{CHAPTER_ID}.mp3"
        with open(output_path, 'wb') as f:
            f.write(response.content)
        
        # Get metadata from headers
        bubble_count = response.headers.get('X-Bubble-Count', 'unknown')
        duration_ms = response.headers.get('X-Duration-Ms', 'unknown')
        audio_format = response.headers.get('X-Audio-Format', 'mp3')
        
        print("✅ SUCCESS! Audio file generated!")
        print(f"💾 Saved to: {output_path}")
        print(f"📊 File size: {len(response.content):,} bytes")
        print()
        print("📋 Metadata:")
        print(f"  - Chapter ID: {CHAPTER_ID}")
        print(f"  - Panels: {len(image_files)}")
        print(f"  - Bubble Count: {bubble_count}")
        print(f"  - Duration: {duration_ms} ms")
        print(f"  - Format: {audio_format}")
        print()
        print(f"🎧 Play the audio with: afplay {output_path}")
    else:
        print(f"❌ ERROR: {response.status_code}")
        print(f"Response: {response.text}")
        sys.exit(1)
        
except requests.exceptions.Timeout:
    print("❌ ERROR: Request timed out (>300s)")
    print("This might indicate an issue with the OCR or TTS API")
    sys.exit(1)
except Exception as e:
    print(f"❌ ERROR: {e}")
    sys.exit(1)
