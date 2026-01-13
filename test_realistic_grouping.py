#!/usr/bin/env python3
"""
Test realistic webtoon bubble scenarios where bubbles merge incorrectly.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from backend.services.text_grouping import TextBubbleGrouper
from backend.services.vision import OCRResult, BoundingBox


def create_word(text: str, left: int, top: int, width: int, height: int) -> OCRResult:
    """Helper to create a mock OCR result."""
    vertices = [
        {"x": left, "y": top},
        {"x": left + width, "y": top},
        {"x": left + width, "y": top + height},
        {"x": left, "y": top + height}
    ]
    bbox = BoundingBox(vertices)
    return OCRResult(text=text, bounding_box=bbox, confidence=0.99)


def test_slightly_offset_bubbles():
    """
    Test case: Two bubbles side-by-side with SLIGHT vertical offset.
    
    This is the REAL PROBLEM in webtoons - bubbles that are:
    - Horizontally separated (different speakers)
    - Slightly vertically offset (not perfectly aligned)
    - Should NOT merge
    
    Layout:
    
    ┌─────────────┐       
    │ Left bubble │       ┌─────────────┐
    │ goes here   │       │Right bubble │
    └─────────────┘       │ goes here   │
                          └─────────────┘
    
    Left bubble: x=50-200, y=100-160
    Right bubble: x=350-500, y=120-180  (20px lower)
    
    Horizontal gap: 150px
    Vertical offset: 20px (some overlap)
    
    Problem: vertical_overlap will be > 0, so they might merge!
    """
    
    grouper = TextBubbleGrouper()
    
    # Left bubble (3 words)
    word1 = create_word("Left", left=50, top=100, width=60, height=30)
    word2 = create_word("bubble", left=120, top=100, width=80, height=30)
    word3 = create_word("here", left=50, top=135, width=60, height=25)
    
    # Right bubble (3 words) - 20px lower, 150px to the right
    word4 = create_word("Right", left=350, top=120, width=70, height=30)
    word5 = create_word("bubble", left=430, top=120, width=70, height=30)
    word6 = create_word("here", left=350, top=155, width=60, height=25)
    
    words = [word1, word2, word3, word4, word5, word6]
    
    print("\n" + "="*80)
    print("TEST: Side-by-Side Bubbles with Vertical Offset (REALISTIC)")
    print("="*80)
    print("\nWord positions:")
    for i, word in enumerate(words, 1):
        bbox = word.bounding_box
        print(f"  {i}. '{word.text:10s}' @ (left={bbox.left:3d}, top={bbox.top:3d}, "
              f"right={bbox.right:3d}, bottom={bbox.bottom:3d})")
    
    # Calculate overlaps/gaps
    print("\nSpatial relationships:")
    print(f"  Horizontal gap (left→right): {350 - 200}px")
    print(f"  Vertical offset: 20px")
    print(f"  Left bubble height: {160 - 100}px")
    print(f"  Right bubble height: {180 - 120}px")
    
    # Check if words connect
    print("\nChecking word connectivity:")
    word_right_edge = word2  # "bubble" from left
    word_left_edge = word4   # "Right" from right
    
    # These are on DIFFERENT lines (vertical overlap < 50%)
    v_overlap = min(word_right_edge.bounding_box.bottom, word_left_edge.bounding_box.bottom) - \
                max(word_right_edge.bounding_box.top, word_left_edge.bounding_box.top)
    max_h = max(word_right_edge.bounding_box.height, word_left_edge.bounding_box.height)
    v_overlap_ratio = v_overlap / max_h
    
    print(f"  Vertical overlap between 'bubble'(L) and 'Right'(R): {v_overlap}px ({v_overlap_ratio:.1%})")
    
    # Test clustering
    bubbles = grouper.group_into_bubbles(words)
    
    print(f"\n{'='*80}")
    print(f"RESULT: Created {len(bubbles)} bubbles")
    print(f"{'='*80}")
    
    for i, bubble in enumerate(bubbles, 1):
        print(f"\nBubble {i}:")
        print(f"  Text: \"{bubble.text}\"")
        print(f"  Words: {[w.text for w in bubble.ocr_results]}")
        print(f"  Bbox: (left={bubble.bounding_box.left}, top={bubble.bounding_box.top}, "
              f"right={bubble.bounding_box.right}, bottom={bubble.bounding_box.bottom})")
    
    # Verify
    print(f"\n{'='*80}")
    if len(bubbles) == 2:
        print("✅ PASS: Correctly created 2 separate bubbles")
        return True
    else:
        print(f"❌ FAIL: Expected 2 bubbles, got {len(bubbles)}")
        if len(bubbles) == 1:
            print(f"   MERGED: \"{bubbles[0].text}\"")
            print("   → This is the PROBLEM - side-by-side bubbles merged!")
        return False


def test_close_bubbles_different_heights():
    """
    Another realistic case: bubbles at very different heights but close horizontally.
    
    Layout:
    
    ┌─────────────┐
    │ Top bubble  │
    └─────────────┘
    
    
    
        ┌─────────────┐
        │Bottom bubble│
        └─────────────┘
    
    Should NOT merge despite being close horizontally.
    """
    
    grouper = TextBubbleGrouper()
    
    # Top bubble
    word1 = create_word("Top", left=50, top=100, width=60, height=30)
    word2 = create_word("bubble", left=120, top=100, width=70, height=30)
    
    # Bottom bubble - 120px below (beyond max_vertical_gap of 100)
    word3 = create_word("Bottom", left=80, top=250, width=80, height=30)
    word4 = create_word("bubble", left=170, top=250, width=70, height=30)
    
    words = [word1, word2, word3, word4]
    
    print("\n" + "="*80)
    print("TEST: Vertically Distant Bubbles")
    print("="*80)
    
    bubbles = grouper.group_into_bubbles(words)
    
    print(f"Vertical gap: {250 - 130}px (beyond max_vertical_gap)")
    print(f"RESULT: Created {len(bubbles)} bubbles")
    
    for i, bubble in enumerate(bubbles, 1):
        print(f"  Bubble {i}: \"{bubble.text}\"")
    
    if len(bubbles) == 2:
        print("✅ PASS: Correctly separated distant bubbles")
        return True
    else:
        print("❌ FAIL: Distant bubbles merged incorrectly")
        return False


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# Realistic Webtoon Bubble Grouping Tests")
    print("#"*80)
    
    result1 = test_slightly_offset_bubbles()
    result2 = test_close_bubbles_different_heights()
    
    print("\n" + "#"*80)
    print("# Summary")
    print("#"*80)
    if result1 and result2:
        print("✅ All tests passed - grouping heuristic works correctly")
    else:
        print("❌ Tests failed - grouping heuristic needs fixing")
        print("\nPROBLEM: The max_horizontal_gap_multiline (150px) is too large")
        print("SOLUTION: Reduce it or add stricter horizontal alignment checks")
    print()
