#!/usr/bin/env python3
"""
Debug script to analyze why 1.png warning text is filtered but 3.png timestamp is accepted.
"""
import sys
import os

# Add backend to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))

from google.cloud import vision
from backend.services.text_grouping import TextBubbleGrouper
from backend.services.text_box_classifier import TextBoxClassifier
from backend.services.vision import OCRResult, BoundingBox
from PIL import Image

def analyze_image(image_path: str, image_name: str):
    """Analyze a single image for classification behavior."""
    
    # Load image dimensions
    with Image.open(image_path) as img:
        image_width, image_height = img.size
        total_pixels = image_width * image_height
    
    print(f"\n{'='*80}")
    print(f"ANALYZING: {image_name}")
    print(f"Image dimensions: {image_width}x{image_height} ({total_pixels:,} pixels)")
    print(f"{'='*80}\n")
    
    # Run OCR
    client = vision.ImageAnnotatorClient()
    with open(image_path, 'rb') as image_file:
        content = image_file.read()
    
    image = vision.Image(content=content)
    response = client.text_detection(image=image)
    annotations = response.text_annotations[1:]  # Skip first (full text)
    
    print(f"OCR detected {len(annotations)} words\n")
    
    # Convert to OCRResult objects
    ocr_results = []
    for annotation in annotations:
        vertices = annotation.bounding_poly.vertices
        vertices_dict = [{"x": v.x, "y": v.y} for v in vertices]
        bbox = BoundingBox(vertices_dict)
        ocr_results.append(OCRResult(text=annotation.description, bounding_box=bbox))
    
    # Group into bubbles
    grouper = TextBubbleGrouper()
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    print(f"Grouped into {len(bubbles)} text bubbles\n")
    
    # Classify bubbles
    classifier = TextBoxClassifier()
    image_area = image_width * image_height
    
    print(f"CLASSIFICATION RESULTS:")
    print(f"-" * 80)
    
    # Create pseudo OCR results from bubbles for feature computation
    pseudo_ocr_results = []
    for bubble in bubbles:
        pseudo_ocr = type('obj', (object,), {
            'text': bubble.text,
            'bounding_box': bubble.bounding_box
        })()
        pseudo_ocr_results.append(pseudo_ocr)
    
    for i, (bubble, pseudo_ocr) in enumerate(zip(bubbles, pseudo_ocr_results), 1):
        # Calculate bbox area
        bbox = bubble.bounding_box
        bbox_area = bbox.width * bbox.height
        bbox_percent = (bbox_area / total_pixels) * 100
        
        # Get features and score
        features = classifier._compute_features(pseudo_ocr, image_area, pseudo_ocr_results)
        final_score = classifier._compute_score(features)
        is_dialogue = final_score >= classifier.threshold
        
        print(f"\nBubble #{i}: {'✓ DIALOGUE' if is_dialogue else '✗ FILTERED'}")
        print(f"  Text: {bubble.text[:80]}{'...' if len(bubble.text) > 80 else ''}")
        print(f"  Bbox: {bbox.width}x{bbox.height} = {bbox_area:,} pixels ({bbox_percent:.2f}%)")
        print(f"  Word count: {len(bubble.text.split())}")
        print(f"  Final Score: {final_score:.3f} (threshold: {classifier.threshold})")
        print(f"  Feature breakdown:")
        print(f"    - bbox_area: {features['bbox_area']:.3f}")
        print(f"    - word_count: {features['word_count']:.3f}")
        print(f"    - text_density: {features['text_density']:.3f}")
        print(f"    - aspect_ratio: {features['aspect_ratio']:.3f}")
        print(f"    - punctuation: {features['punctuation']:.3f}")

def main():
    # Analyze both images
    analyze_image('1.png', '1.png (Warning Text)')
    analyze_image('3.png', '3.png (Timestamp)')
    
    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}\n")

if __name__ == '__main__':
    main()
