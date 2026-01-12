#!/usr/bin/env python3
"""
Test script to test the full pipeline with webtoonScreenshot4.png
"""
import requests
import sys
from pathlib import Path

def test_pipeline(image_path: str, chapter_id: str = "test_chapter_001"):
    """Test the /process/chapter endpoint with a real image"""
    
    # Check if image exists
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return False
    
    print(f"📸 Testing pipeline with: {img_file.name}")
    print(f"📦 Chapter ID: {chapter_id}")
    print(f"🔗 Endpoint: http://localhost:8000/process/chapter")
    print("-" * 60)
    
    # Prepare the request
    url = "http://localhost:8000/process/chapter"
    
    files = [
        ('images', (img_file.name, open(img_file, 'rb'), 'image/png'))
    ]
    
    data = {
        'chapter_id': chapter_id
    }
    
    try:
        print("⏳ Sending request to pipeline...")
        response = requests.post(url, files=files, data=data, timeout=120)
        
        print(f"📡 Response Status: {response.status_code}")
        print("-" * 60)
        
        if response.status_code == 200:
            # Success - save the MP3 file
            output_file = f"{chapter_id}.mp3"
            with open(output_file, 'wb') as f:
                f.write(response.content)
            
            # Print metadata from headers
            print("✅ SUCCESS! Audio file generated!")
            print(f"💾 Saved to: {output_file}")
            print(f"📊 File size: {len(response.content):,} bytes")
            print()
            print("📋 Metadata:")
            print(f"  - Chapter ID: {response.headers.get('X-Chapter-ID', 'N/A')}")
            print(f"  - Bubble Count: {response.headers.get('X-Bubble-Count', 'N/A')}")
            print(f"  - Duration: {response.headers.get('X-Duration-MS', 'N/A')} ms")
            print(f"  - Format: {response.headers.get('X-Format', 'N/A')}")
            print()
            print(f"🎧 Play the audio with: afplay {output_file}")
            return True
            
        else:
            # Error response
            print(f"❌ ERROR: Request failed with status {response.status_code}")
            print()
            try:
                error_data = response.json()
                print("Error details:")
                for key, value in error_data.items():
                    print(f"  - {key}: {value}")
            except:
                print("Response body:")
                print(response.text[:500])
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ ERROR: Could not connect to server")
        print("Make sure the FastAPI server is running:")
        print("  python -m uvicorn backend.main:app --reload --port 8000")
        return False
        
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out (>120s)")
        print("This might indicate an issue with the OCR or TTS API")
        return False
        
    except Exception as e:
        print(f"❌ ERROR: Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    # Default to webtoonScreenshot4.png
    image_path = sys.argv[1] if len(sys.argv) > 1 else "webtoonScreenshot4.png"
    chapter_id = sys.argv[2] if len(sys.argv) > 2 else "webtoon_test_001"
    
    success = test_pipeline(image_path, chapter_id)
    sys.exit(0 if success else 1)
