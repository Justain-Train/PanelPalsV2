#!/usr/bin/env python3
"""
Test to verify that side-by-side speech bubbles are properly separated
"""
from backend.services.vision import GoogleVisionOCRService, OCRResult, BoundingBox
from backend.services.text_grouping import TextBubbleGrouper

# Create mock OCR results for a test case with two side-by-side bubbles
# Left bubble: "HELLO WORLD" at x=100-300
# Right bubble: "GOODBYE FRIEND" at x=500-700 (200px gap)

def create_mock_word(text: str, top: int, left: int, width: int, height: int) -> OCRResult:
    """Create a mock OCR result for testing."""
    vertices = [
        {"x": left, "y": top},
        {"x": left + width, "y": top},
        {"x": left + width, "y": top + height},
        {"x": left, "y": top + height}
    ]
    bbox = BoundingBox(vertices)
    return OCRResult(text=text, bounding_box=bbox, confidence=0.99)

print("🧪 Testing Side-by-Side Bubble Separation")
print("=" * 80)

# Test Case 1: Two bubbles on the same horizontal line, 200px apart
print("\n📋 Test Case 1: Horizontally adjacent bubbles (200px gap)")
print("-" * 80)

test_words = [
    # Left bubble: "HELLO WORLD"
    create_mock_word("HELLO", top=100, left=100, width=80, height=30),
    create_mock_word("WORLD", top=100, left=190, width=80, height=30),
    
    # Right bubble: "GOODBYE FRIEND" (200px gap from left bubble)
    create_mock_word("GOODBYE", top=100, left=500, width=100, height=30),
    create_mock_word("FRIEND", top=100, left=610, width=80, height=30),
]

grouper = TextBubbleGrouper()
bubbles = grouper.group_into_bubbles(test_words)

print(f"   Words: {len(test_words)}")
print(f"   Expected bubbles: 2")
print(f"   Actual bubbles: {len(bubbles)}")

if len(bubbles) == 2:
    print(f"   ✅ PASS: Correctly separated into 2 bubbles")
    for idx, bubble in enumerate(bubbles, 1):
        print(f"      Bubble {idx}: \"{bubble.text}\"")
else:
    print(f"   ❌ FAIL: Expected 2 bubbles, got {len(bubbles)}")
    for idx, bubble in enumerate(bubbles, 1):
        print(f"      Bubble {idx}: \"{bubble.text}\"")

# Test Case 2: Two bubbles with small gap (should NOT merge)
print("\n📋 Test Case 2: Bubbles with 100px gap (should NOT merge)")
print("-" * 80)

test_words_2 = [
    create_mock_word("LEFT", top=100, left=100, width=60, height=30),
    create_mock_word("TEXT", top=100, left=170, width=60, height=30),
    
    # 100px gap
    create_mock_word("RIGHT", top=100, left=330, width=70, height=30),
    create_mock_word("TEXT", top=100, left=410, width=60, height=30),
]

bubbles_2 = grouper.group_into_bubbles(test_words_2)

print(f"   Words: {len(test_words_2)}")
print(f"   Expected bubbles: 2")
print(f"   Actual bubbles: {len(bubbles_2)}")

if len(bubbles_2) == 2:
    print(f"   ✅ PASS: Correctly separated into 2 bubbles")
    for idx, bubble in enumerate(bubbles_2, 1):
        print(f"      Bubble {idx}: \"{bubble.text}\"")
else:
    print(f"   ❌ FAIL: Expected 2 bubbles, got {len(bubbles_2)}")
    for idx, bubble in enumerate(bubbles_2, 1):
        print(f"      Bubble {idx}: \"{bubble.text}\"")

# Test Case 3: Multi-line bubble (should merge)
print("\n📋 Test Case 3: Multi-line bubble (should merge into 1)")
print("-" * 80)

test_words_3 = [
    # Line 1
    create_mock_word("THIS", top=100, left=100, width=60, height=30),
    create_mock_word("IS", top=100, left=170, width=30, height=30),
    # Line 2 (50px below, aligned)
    create_mock_word("MULTIPLE", top=150, left=100, width=100, height=30),
    create_mock_word("LINES", top=150, left=210, width=70, height=30),
]

bubbles_3 = grouper.group_into_bubbles(test_words_3)

print(f"   Words: {len(test_words_3)}")
print(f"   Expected bubbles: 1")
print(f"   Actual bubbles: {len(bubbles_3)}")

if len(bubbles_3) == 1:
    print(f"   ✅ PASS: Correctly merged into 1 bubble")
    print(f"      Bubble 1: \"{bubbles_3[0].text}\"")
else:
    print(f"   ❌ FAIL: Expected 1 bubble, got {len(bubbles_3)}")
    for idx, bubble in enumerate(bubbles_3, 1):
        print(f"      Bubble {idx}: \"{bubble.text}\"")

# Test Case 4: Words with tight spacing (normal bubble, should merge)
print("\n📋 Test Case 4: Normal word spacing (should merge into 1)")
print("-" * 80)

test_words_4 = [
    create_mock_word("I", top=100, left=100, width=20, height=30),
    create_mock_word("LOVE", top=100, left=130, width=60, height=30),
    create_mock_word("YOU", top=100, left=200, width=50, height=30),
]

bubbles_4 = grouper.group_into_bubbles(test_words_4)

print(f"   Words: {len(test_words_4)}")
print(f"   Expected bubbles: 1")
print(f"   Actual bubbles: {len(bubbles_4)}")

if len(bubbles_4) == 1:
    print(f"   ✅ PASS: Correctly merged into 1 bubble")
    print(f"      Bubble 1: \"{bubbles_4[0].text}\"")
else:
    print(f"   ❌ FAIL: Expected 1 bubble, got {len(bubbles_4)}")
    for idx, bubble in enumerate(bubbles_4, 1):
        print(f"      Bubble {idx}: \"{bubble.text}\"")

print("\n" + "=" * 80)
print("✅ Test suite complete!")
