#!/usr/bin/env python3
"""
Debug script to show how text is being ordered within bubbles
"""
import sys
from backend.services.vision import GoogleVisionOCRService
from backend.services.text_grouping import TextBubbleGrouper

if len(sys.argv) < 2:
    print("Usage: python debug_word_ordering.py <image_file>")
    sys.exit(1)

image_path = sys.argv[1]

print(f"📸 Analyzing word ordering in: {image_path}")
print("=" * 80)

# Read image
with open(image_path, 'rb') as f:
    image_bytes = f.read()

# Perform OCR
print("\n🔍 Step 1: Running OCR...")
ocr_service = GoogleVisionOCRService()
ocr_results = ocr_service.detect_text_batch([image_bytes])[0]
print(f"   Detected {len(ocr_results)} words")

# Group into bubbles
print("\n📦 Step 2: Grouping into text bubbles...")
grouper = TextBubbleGrouper()
bubbles = grouper.group_into_bubbles(ocr_results)
print(f"   Created {len(bubbles)} bubbles")

# Show detailed bubble analysis
print("\n" + "=" * 80)
print("📋 BUBBLE ANALYSIS")
print("=" * 80)

for idx, bubble in enumerate(bubbles, 1):
    print(f"\n🗨️  Bubble {idx} (Reading Order: {bubble.reading_order})")
    print(f"   📝 Final Text: \"{bubble.text}\"")
    print(f"   📍 Bounding Box: top={bubble.bounding_box.top:.0f}, left={bubble.bounding_box.left:.0f}")
    print(f"   📊 Word Count: {len(bubble.ocr_results)}")
    
    # Show word-by-word breakdown
    print(f"\n   📖 Word-by-word breakdown (in reading order):")
    for word_idx, ocr in enumerate(bubble.ocr_results, 1):
        bbox = ocr.bounding_box
        print(f"      {word_idx:2d}. \"{ocr.text:20s}\" @ (top={bbox.top:5.0f}, left={bbox.left:5.0f}, "
              f"bottom={bbox.bottom:5.0f}, right={bbox.right:5.0f})")
    
    # Show line grouping
    print(f"\n   📏 Line grouping:")
    current_line = []
    current_line_top = bubble.ocr_results[0].bounding_box.top if bubble.ocr_results else 0
    line_num = 1
    
    for ocr in bubble.ocr_results:
        # Check if this word is on a new line (>50% different top position)
        top_diff = abs(ocr.bounding_box.top - current_line_top)
        avg_height = (ocr.bounding_box.height + 
                     (bubble.ocr_results[0].bounding_box.height if bubble.ocr_results else ocr.bounding_box.height)) / 2
        
        if top_diff > avg_height * 0.5 and current_line:
            # Print current line
            line_text = " ".join(w.text for w in current_line)
            print(f"      Line {line_num}: \"{line_text}\"")
            line_num += 1
            current_line = [ocr]
            current_line_top = ocr.bounding_box.top
        else:
            current_line.append(ocr)
    
    # Print last line
    if current_line:
        line_text = " ".join(w.text for w in current_line)
        print(f"      Line {line_num}: \"{line_text}\"")
    
    print()

print("=" * 80)
print("✅ Analysis complete!")
