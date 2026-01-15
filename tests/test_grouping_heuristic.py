#!/usr/bin/env python3
"""
Test to demonstrate side-by-side bubble merging problem.

This creates a scenario where two bubbles are:
- Vertically close (within max_vertical_gap)
- Horizontally separate (side-by-side)
- Should NOT merge
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


def test_side_by_side_bubbles():
    """
    Test case: Two bubbles side-by-side should NOT merge.
    
    Layout:
    
    ┌─────────────┐       ┌─────────────┐
    │ Left bubble │       │Right bubble │
    │ text here   │       │ text here   │
    └─────────────┘       └─────────────┘
    
    Left bubble: x=50-250, y=100-150
    Right bubble: x=400-600, y=100-150
    
    Horizontal gap: 150px
    Vertical gap: 0px (same line vertically)
    
    Expected: 2 separate bubbles
    """
    
    grouper = TextBubbleGrouper()
    
    # Left bubble (2 words)
    word1 = create_word("Left", left=50, top=100, width=80, height=40)
    word2 = create_word("bubble", left=140, top=100, width=90, height=40)
    
    # Right bubble (2 words) - 150px gap from left bubble
    word3 = create_word("Right", left=400, top=100, width=80, height=40)
    word4 = create_word("bubble", left=490, top=100, width=90, height=40)
    
    words = [word1, word2, word3, word4]
    
    print("\n" + "="*80)
    print("TEST: Side-by-Side Bubbles Should NOT Merge")
    print("="*80)
    print("\nWord positions:")
    for i, word in enumerate(words, 1):
        bbox = word.bounding_box
        print(f"  {i}. '{word.text:10s}' @ (left={bbox.left:3d}, top={bbox.top:3d}, "
              f"right={bbox.right:3d}, bottom={bbox.bottom:3d})")
    
    # Calculate gaps
    gap_left_to_right = 400 - 250  # 150px
    print(f"\nHorizontal gap between bubbles: {gap_left_to_right}px")
    
    # Test clustering
    bubbles = grouper.group_into_bubbles(words)
    
    print(f"\n{'='*80}")
    print(f"RESULT: Created {len(bubbles)} bubbles")
    print(f"{'='*80}")
    
    for i, bubble in enumerate(bubbles, 1):
        print(f"\nBubble {i}:")
        print(f"  Text: \"{bubble.text}\"")
        print(f"  Words: {[w.text for w in bubble.ocr_results]}")
        print(f"  Position: (left={bubble.bounding_box.left}, top={bubble.bounding_box.top})")
    
    # Verify
    print(f"\n{'='*80}")
    if len(bubbles) == 2:
        print("✅ PASS: Correctly created 2 separate bubbles")
        print("   Side-by-side bubbles were NOT merged")
        return True
    else:
        print(f"❌ FAIL: Expected 2 bubbles, got {len(bubbles)}")
        print("   Side-by-side bubbles were INCORRECTLY merged")
        if len(bubbles) == 1:
            print(f"   Combined text: \"{bubbles[0].text}\"")
        return False


def test_stacked_bubbles():
    """
    Test case: Two bubbles vertically stacked should NOT merge if horizontally offset.
    
    Layout:
    
    ┌─────────────┐
    │ Top bubble  │
    └─────────────┘
              ┌─────────────┐
              │Bottom bubble│
              └─────────────┘
    
    Top bubble: x=50-200, y=100-140
    Bottom bubble: x=150-300, y=160-200
    
    Vertical gap: 20px
    Horizontal offset: 100px (partial overlap)
    
    Expected: Should these merge? Depends on heuristic.
    """
    
    grouper = TextBubbleGrouper()
    
    # Top bubble
    word1 = create_word("Top", left=50, top=100, width=70, height=40)
    word2 = create_word("bubble", left=130, top=100, width=70, height=40)
    
    # Bottom bubble - 20px below, offset 100px to the right
    word3 = create_word("Bottom", left=150, top=160, width=80, height=40)
    word4 = create_word("bubble", left=240, top=160, width=60, height=40)
    
    words = [word1, word2, word3, word4]
    
    print("\n" + "="*80)
    print("TEST: Vertically Stacked with Offset")
    print("="*80)
    print("\nWord positions:")
    for i, word in enumerate(words, 1):
        bbox = word.bounding_box
        print(f"  {i}. '{word.text:10s}' @ (left={bbox.left:3d}, top={bbox.top:3d}, "
              f"right={bbox.right:3d}, bottom={bbox.bottom:3d})")
    
    bubbles = grouper.group_into_bubbles(words)
    
    print(f"\n{'='*80}")
    print(f"RESULT: Created {len(bubbles)} bubbles")
    print(f"{'='*80}")
    
    for i, bubble in enumerate(bubbles, 1):
        print(f"\nBubble {i}:")
        print(f"  Text: \"{bubble.text}\"")
        print(f"  Words: {[w.text for w in bubble.ocr_results]}")


if __name__ == "__main__":
    print("\n" + "#"*80)
    print("# Text Grouping Heuristic Tests")
    print("#"*80)
    
    result1 = test_side_by_side_bubbles()
    test_stacked_bubbles()
    
    print("\n" + "#"*80)
    print("# Summary")
    print("#"*80)
    if result1:
        print("✅ All critical tests passed")
    else:
        print("❌ Some tests failed - horizontal overlap heuristic needs fixing")
    print()
