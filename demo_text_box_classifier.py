#!/usr/bin/env python3
"""
Demo script to test text box classifier with real Webtoon screenshots
Shows which text regions are classified as TEXT BOX vs BACKGROUND
"""

import sys
sys.path.insert(0, '/Users/test/PanelPalsV2')

from backend.services.text_box_classifier import TextBoxClassifier
from backend.services.text_grouping import TextBubbleGrouper
from backend.services.vision import GoogleVisionOCRService
from PIL import Image
import os

# Image to test
IMAGE_PATH = "18.png"

print("=" * 70)
print("TEXT BOX CLASSIFIER DEMO")
print("=" * 70)
print(f"Image: {IMAGE_PATH}")
print()

# Load image to get dimensions
try:
    img = Image.open(IMAGE_PATH)
    width, height = img.size
    print(f"Image dimensions: {width}x{height}")
except Exception as e:
    print(f"❌ Error loading image: {e}")
    sys.exit(1)

# Initialize services
print("\n🔧 Initializing services...")
vision_service = GoogleVisionOCRService()
classifier = TextBoxClassifier()

print(f"  - Classification threshold: {classifier.threshold}")
print(f"  - Feature weights: {classifier.weights}")

# Perform OCR
print(f"\n🔍 Running Google Vision OCR on {IMAGE_PATH}...")
with open(IMAGE_PATH, 'rb') as f:
    image_bytes = f.read()

try:
    ocr_results = vision_service.detect_text(image_bytes)
    print(f"  - Detected {len(ocr_results)} text regions")
except Exception as e:
    print(f"❌ OCR failed: {e}")
    sys.exit(1)

if not ocr_results:
    print("⚠️  No text detected!")
    sys.exit(0)


#Group into text bubbles 

print(f"\n🔧 Grouping OCR results into text bubbles...")
grouper = TextBubbleGrouper()
image_grouped = grouper.group_into_bubbles(ocr_results)


# Classify regions
print(f"\n🤖 Classifying {len(image_grouped)} regions...")
classification_results = classifier.classify_regions(image_grouped, width, height)

# Separate into TEXT BOX and BACKGROUND
text_boxes = [r for r in classification_results if r.is_text_box]
background = [r for r in classification_results if not r.is_text_box]

print(f"\n📊 RESULTS:")
print(f"  - TEXT BOX (dialogue for TTS): {len(text_boxes)}")
print(f"  - BACKGROUND (filtered out): {len(background)}")

# Show TEXT BOX regions
if text_boxes:
    print(f"\n✅ TEXT BOX REGIONS ({len(text_boxes)}):")
    print("-" * 70)
    for i, result in enumerate(text_boxes, 1):
        text = result.ocr_result.text
        score = result.features
        print(f"{i}. \"{text}\" (score={result.score:.3f})")
        print(f"   Features: area={score['bbox_area']:.2f}, words={score['word_count']:.2f}, "
              f"density={score['text_density']:.2f}, aspect={score['aspect_ratio']:.2f}, "
              f"punct={score['punctuation']:.2f}")
        print(f"   Raw: {score['raw_word_count']} words, density={score['raw_density']:.6f}, "
              f"bbox={score['raw_bbox_area']}px²")
        print()

# Show BACKGROUND regions
if background:
    print(f"❌ BACKGROUND REGIONS ({len(background)}) - FILTERED OUT:")
    print("-" * 70)
    for i, result in enumerate(background, 1):
        text = result.ocr_result.text
        score = result.features
        print(f"{i}. \"{text}\" (score={result.score:.3f})")
        print(f"   Features: area={score['bbox_area']:.2f}, words={score['word_count']:.2f}, "
              f"density={score['text_density']:.2f}, aspect={score['aspect_ratio']:.2f}, "
              f"punct={score['punctuation']:.2f}")
        print(f"   Raw: {score['raw_word_count']} words, density={score['raw_density']:.6f}, "
              f"bbox={score['raw_bbox_area']}px²")
        print()

# Summary
print("=" * 70)
print(f"SUMMARY: {len(text_boxes)}/{len(ocr_results)} regions will proceed to TTS")
print(f"         {len(background)}/{len(ocr_results)} regions filtered out as background")
print("=" * 70)
