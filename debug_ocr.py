#!/usr/bin/env python3
"""
Debug OCR results to see what Google Vision is detecting
"""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.services.vision import GoogleVisionOCRService
from backend.services.text_grouping import TextBubbleGrouper

def debug_ocr(image_path: str):
    """Debug OCR detection and text grouping"""
    
    img_file = Path(image_path)
    if not img_file.exists():
        print(f"❌ Error: Image file not found: {image_path}")
        return
    
    print(f"🔍 Debugging OCR for: {img_file.name}")
    print("=" * 80)
    
    # Read image
    with open(img_file, 'rb') as f:
        image_bytes = f.read()
    
    print(f"📸 Image size: {len(image_bytes):,} bytes")
    print()
    
    # Perform OCR
    print("🔍 Running Google Vision OCR...")
    ocr_service = GoogleVisionOCRService()
    ocr_results = ocr_service.detect_text_batch([image_bytes])
    
    if not ocr_results or not ocr_results[0]:
        print("❌ No text detected")
        return
    
    # Flatten results
    all_results = ocr_results[0]
    
    print(f"✅ Detected {len(all_results)} OCR results")
    print()
    print("=" * 80)
    print("📝 RAW OCR RESULTS:")
    print("=" * 80)
    
    for idx, result in enumerate(all_results, 1):
        bbox = result.bounding_box
        vertices = bbox.vertices if hasattr(bbox, 'vertices') else []
        
        # Calculate approximate position
        if vertices:
            avg_x = sum(v.get('x', 0) for v in vertices) / len(vertices)
            avg_y = sum(v.get('y', 0) for v in vertices) / len(vertices)
        else:
            avg_x = avg_y = 0
        
        print(f"\n{idx:3d}. Text: '{result.text}'")
        print(f"      Position: (x={avg_x:.0f}, y={avg_y:.0f})")
        print(f"      Vertices: {vertices}")
    
    print()
    print("=" * 80)
    print("🔗 TEXT GROUPING:")
    print("=" * 80)
    
    # Group into bubbles
    text_grouper = TextBubbleGrouper()
    bubbles = text_grouper.group_into_bubbles(all_results)
    
    print(f"✅ Formed {len(bubbles)} text bubbles")
    print()
    
    for idx, bubble in enumerate(bubbles, 1):
        bbox = bubble.bounding_box
        vertices = bbox.vertices if hasattr(bbox, 'vertices') else []
        
        if vertices:
            avg_x = sum(v.get('x', 0) for v in vertices) / len(vertices)
            avg_y = sum(v.get('y', 0) for v in vertices) / len(vertices)
        else:
            avg_x = avg_y = 0
        
        print(f"\n📦 Bubble {idx}:")
        print(f"   Text: '{bubble.text}'")
        print(f"   Position: (x={avg_x:.0f}, y={avg_y:.0f})")
        print(f"   Reading Order: {bubble.reading_order}")
    
    print()
    print("=" * 80)
    print("💡 ANALYSIS:")
    print("=" * 80)
    print(f"OCR Results: {len(all_results)}")
    print(f"Text Bubbles: {len(bubbles)}")
    print(f"Ratio: {len(all_results) / len(bubbles):.1f}x (avg words per bubble)")
    print()
    
    if len(bubbles) > 5:
        print("⚠️  WARNING: More than 5 bubbles detected for a single panel!")
        print("This might indicate:")
        print("  1. OCR is detecting individual words/characters instead of phrases")
        print("  2. Text grouping parameters need adjustment")
        print("  3. Image contains more text than expected")

if __name__ == "__main__":
    image_path = sys.argv[1] if len(sys.argv) > 1 else "webtoonScreenshot4.png"
    debug_ocr(image_path)
