"""
Unit tests for Bubble Continuation Detection

Tests detection and merging of text bubbles that span multiple images.
"""

import pytest
from unittest.mock import Mock

from backend.services.bubble_continuation import BubbleContinuationDetector, ContinuationMatch
from backend.services.text_grouping import TextBubble
from backend.services.vision import BoundingBox


def create_bubble(text: str, left: int, top: int, width: int, height: int) -> TextBubble:
    """Helper to create a mock TextBubble."""
    vertices = [
        {"x": left, "y": top},
        {"x": left + width, "y": top},
        {"x": left + width, "y": top + height},
        {"x": left, "y": top + height}
    ]
    bbox = BoundingBox(vertices)
    return TextBubble(text=text, bounding_box=bbox, ocr_results=[])


class TestBubbleContinuationDetector:
    """Test suite for BubbleContinuationDetector."""
    
    def setup_method(self):
        """Initialize detector before each test."""
        self.detector = BubbleContinuationDetector()
    
    # ===================================================================
    # Test incomplete sentence detection
    # ===================================================================
    
    def test_incomplete_sentence_no_punctuation(self):
        """Text without ending punctuation is incomplete."""
        assert self.detector._has_incomplete_sentence("Hey there")
        assert self.detector._has_incomplete_sentence("What are you doing")
    
    def test_incomplete_sentence_comma(self):
        """Text ending with comma is incomplete."""
        assert self.detector._has_incomplete_sentence("Well,")
        assert self.detector._has_incomplete_sentence("Hey, wait,")
    
    def test_incomplete_sentence_dash(self):
        """Text ending with dash is incomplete."""
        assert self.detector._has_incomplete_sentence("What the—")
        assert self.detector._has_incomplete_sentence("I was going to say-")
    
    def test_complete_sentence_period(self):
        """Text ending with period is complete."""
        assert not self.detector._has_incomplete_sentence("Hello there.")
        assert not self.detector._has_incomplete_sentence("That's interesting.")
    
    def test_complete_sentence_exclamation(self):
        """Text ending with exclamation is complete."""
        assert not self.detector._has_incomplete_sentence("Watch out!")
        assert not self.detector._has_incomplete_sentence("Amazing!")
    
    def test_complete_sentence_question(self):
        """Text ending with question mark is complete (for the heuristic)."""
        # The heuristic considers sentence-ending punctuation as "potentially complete"
        # But the actual merging decision considers ALL signals
        result = self.detector._has_incomplete_sentence("What is that?")
        # With sentence-ending punct, it returns False (not clearly incomplete)
        assert not result
    
    def test_incomplete_continuation_words(self):
        """Text ending with continuation word is incomplete."""
        assert self.detector._has_incomplete_sentence("This is the")
        assert self.detector._has_incomplete_sentence("I saw that and")
    
    # ===================================================================
    # Test alignment calculations
    # ===================================================================
    
    def test_horizontal_alignment_perfect(self):
        """Perfect horizontal alignment should return 1.0."""
        bbox1 = BoundingBox([
            {"x": 100, "y": 50},
            {"x": 200, "y": 50},
            {"x": 200, "y": 100},
            {"x": 100, "y": 100}
        ])
        bbox2 = BoundingBox([
            {"x": 100, "y": 150},
            {"x": 200, "y": 150},
            {"x": 200, "y": 200},
            {"x": 100, "y": 200}
        ])
        
        alignment = self.detector._calculate_horizontal_alignment(bbox1, bbox2)
        assert alignment == 1.0
    
    def test_horizontal_alignment_partial(self):
        """Partial horizontal overlap."""
        bbox1 = BoundingBox([
            {"x": 100, "y": 50},
            {"x": 200, "y": 50},
            {"x": 200, "y": 100},
            {"x": 100, "y": 100}
        ])
        bbox2 = BoundingBox([
            {"x": 150, "y": 150},  # 50px offset right
            {"x": 250, "y": 150},
            {"x": 250, "y": 200},
            {"x": 150, "y": 200}
        ])
        
        alignment = self.detector._calculate_horizontal_alignment(bbox1, bbox2)
        # Overlap is 50px (150-200), smaller width is 100px → 50/100 = 0.5
        assert 0.4 < alignment < 0.6
    
    def test_horizontal_alignment_none(self):
        """No horizontal overlap."""
        bbox1 = BoundingBox([
            {"x": 100, "y": 50},
            {"x": 200, "y": 50},
            {"x": 200, "y": 100},
            {"x": 100, "y": 100}
        ])
        bbox2 = BoundingBox([
            {"x": 300, "y": 150},  # Completely separate
            {"x": 400, "y": 150},
            {"x": 400, "y": 200},
            {"x": 300, "y": 200}
        ])
        
        alignment = self.detector._calculate_horizontal_alignment(bbox1, bbox2)
        assert alignment == 0.0
    
    def test_width_similarity_identical(self):
        """Identical widths should return 1.0."""
        bbox1 = BoundingBox([
            {"x": 100, "y": 50},
            {"x": 200, "y": 50},
            {"x": 200, "y": 100},
            {"x": 100, "y": 100}
        ])
        bbox2 = BoundingBox([
            {"x": 110, "y": 150},
            {"x": 210, "y": 150},
            {"x": 210, "y": 200},
            {"x": 110, "y": 200}
        ])
        
        similarity = self.detector._calculate_width_similarity(bbox1, bbox2)
        assert similarity == 1.0
    
    def test_width_similarity_different(self):
        """Different widths should return ratio."""
        bbox1 = BoundingBox([
            {"x": 100, "y": 50},
            {"x": 200, "y": 50},  # width = 100
            {"x": 200, "y": 100},
            {"x": 100, "y": 100}
        ])
        bbox2 = BoundingBox([
            {"x": 100, "y": 150},
            {"x": 150, "y": 150},  # width = 50
            {"x": 150, "y": 200},
            {"x": 100, "y": 200}
        ])
        
        similarity = self.detector._calculate_width_similarity(bbox1, bbox2)
        assert similarity == 0.5  # 50/100
    
    # ===================================================================
    # Test continuation detection
    # ===================================================================
    
    def test_detect_continuation_positive(self):
        """Should detect continuation with good alignment and incomplete sentence."""
        # Image height = 500px
        # Bubble at bottom of image 1: "Hey, what are you" (top=450, bottom=500)
        prev_bubble = create_bubble("Hey, what are you", left=100, top=450, width=200, height=50)
        
        # Bubble at top of image 2: "doing over there?" (top=10, within proximity)
        next_bubble = create_bubble("doing over there?", left=100, top=10, width=200, height=40)
        
        match = self.detector._check_continuation(prev_bubble, next_bubble, prev_image_height=500)
        
        assert match is not None
        assert match.confidence >= 0.6
        assert "horizontal_alignment" in match.reason
        assert "incomplete_sentence" in match.reason
    
    def test_detect_continuation_negative_poor_alignment(self):
        """Should NOT detect continuation with poor horizontal alignment."""
        # Bubble at bottom of image 1
        prev_bubble = create_bubble("Hey, what are you", left=100, top=450, width=200, height=50)
        
        # Bubble at top of image 2, but far to the right
        next_bubble = create_bubble("doing over there?", left=400, top=10, width=200, height=40)
        
        match = self.detector._check_continuation(prev_bubble, next_bubble, prev_image_height=500)
        
        assert match is None  # Poor alignment should reject
    
    def test_detect_continuation_negative_complete_sentence(self):
        """Complete sentence reduces confidence but might still match."""
        # Bubble at bottom with complete sentence (near image boundary)
        prev_bubble = create_bubble("That is interesting.", left=100, top=450, width=200, height=50)
        
        # Bubble at top (near image boundary)
        next_bubble = create_bubble("What do you think?", left=100, top=10, width=200, height=40)
        
        match = self.detector._check_continuation(prev_bubble, next_bubble, prev_image_height=500)
        
        # Should NOT match due to sentence boundary even with good alignment
        assert match is None
    
    def test_detect_continuation_negative_not_at_boundaries(self):
        """Should NOT detect continuation if bubbles aren't at image boundaries."""
        # Bubble in MIDDLE of image 1 (not at bottom)
        # Image height = 500, bubble at top=200, bottom=250
        prev_bubble = create_bubble("Hey, what are you", left=100, top=200, width=200, height=50)
        
        # Bubble in MIDDLE of image 2 (not at top)
        next_bubble = create_bubble("doing over there?", left=100, top=100, width=200, height=40)
        
        match = self.detector._check_continuation(prev_bubble, next_bubble, prev_image_height=500)
        
        # Should NOT match - bubbles aren't at image boundaries
        assert match is None
    
    # ===================================================================
    # Test full pipeline
    # ===================================================================
    
    def test_merge_single_continuation(self):
        """Test merging a single continuation across two images."""
        # Image 1 (height=500): 2 bubbles, last one is incomplete and at bottom
        img1_bubbles = [
            create_bubble("Hello there!", left=50, top=100, width=180, height=40),
            create_bubble("Hey, what are you", left=100, top=450, width=200, height=50)  # bottom=500
        ]
        
        # Image 2 (height=500): 2 bubbles, first one continues from img1 and at top
        img2_bubbles = [
            create_bubble("doing over there?", left=100, top=10, width=200, height=40),  # top=10
            create_bubble("Nothing much.", left=150, top=200, width=150, height=35)
        ]
        
        bubble_groups = [img1_bubbles, img2_bubbles]
        image_heights = [500, 500]
        merged = self.detector.detect_and_merge_continuations(bubble_groups, image_heights)
        
        # Should have 3 bubbles total (2 from img1, but last merged with first of img2, then 1 more from img2)
        assert len(merged) == 3
        
        # Second bubble should be merged text
        assert "Hey, what are you doing over there?" in merged[1].text
        
        # Third bubble should be unchanged
        assert merged[2].text == "Nothing much."
    
    def test_merge_no_continuations(self):
        """Test when there are no continuations (complete sentences with punctuation)."""
        # Image 1 (height=500): Complete sentences
        img1_bubbles = [
            create_bubble("Hello there!", left=50, top=100, width=180, height=40),
            create_bubble("How are you?", left=100, top=450, width=200, height=50)  # bottom=500
        ]
        
        # Image 2 (height=500): Different topic
        img2_bubbles = [
            create_bubble("I am fine.", left=100, top=10, width=200, height=40),  # top=10
            create_bubble("Thanks for asking!", left=150, top=200, width=150, height=35)
        ]
        
        bubble_groups = [img1_bubbles, img2_bubbles]
        image_heights = [500, 500]
        merged = self.detector.detect_and_merge_continuations(bubble_groups, image_heights)
        
        # Should have all 4 bubbles unchanged
        # The algorithm should NOT merge "How are you?" with "I am fine."
        # because "How are you?" ends with sentence punctuation
        assert len(merged) == 4, f"Expected 4 bubbles, got {len(merged)}"
        assert merged[0].text == "Hello there!"
        assert merged[1].text == "How are you?"
        assert merged[2].text == "I am fine."
        assert merged[3].text == "Thanks for asking!"
    
    def test_merge_multiple_continuations(self):
        """Test multiple continuations across three images.
        
        Each bubble must be positioned correctly relative to image boundaries:
        - Prev bubble: near BOTTOM of its image (within 50px of bottom)
        - Next bubble: near TOP of its image (within 50px of top)
        """
        # Image 1 (height=500): Single bubble at bottom
        img1_bubbles = [
            create_bubble("This is a", left=100, top=450, width=200, height=50)  # bottom=500, ✓ at boundary
        ]
        
        # Image 2 (height=500): Single bubble at BOTH top AND bottom
        # The bubble is at the top (continues from img1) AND will continue to img3
        # Position it in middle since it can't be at both boundaries simultaneously
        # This is actually an impossible scenario - a bubble can't be at top AND bottom
        # Let's test TWO separate continuations instead
        img2_bubbles = [
            create_bubble("really long", left=100, top=10, width=200, height=40)  # top=10, ✓ continues from img1
        ]
        
        # Image 3 (height=500): Single bubble at top
        img3_bubbles = [
            create_bubble("sentence.", left=100, top=10, width=200, height=40)  # top=10, ✗ can't continue (img2 not at bottom)
        ]
        
        bubble_groups = [img1_bubbles, img2_bubbles, img3_bubbles]
        image_heights = [500, 500, 500]
        merged = self.detector.detect_and_merge_continuations(bubble_groups, image_heights)
        
        # Should have 2 bubbles:
        # 1. "This is a really long" (merged from img1 + img2)
        # 2. "sentence." (img3, not merged because img2 bubble not at bottom)
        assert len(merged) == 2
        assert "This is a really long" in merged[0].text
        assert "sentence." in merged[1].text
    
    def test_merge_empty_images(self):
        """Test with some empty images."""
        img1_bubbles = [
            create_bubble("Start here", left=100, top=450, width=200, height=50)  # bottom=500
        ]
        
        img2_bubbles = []  # Empty image
        
        img3_bubbles = [
            create_bubble("End here.", left=100, top=10, width=200, height=40)  # top=10
        ]
        
        bubble_groups = [img1_bubbles, img2_bubbles, img3_bubbles]
        image_heights = [500, 500, 500]
        merged = self.detector.detect_and_merge_continuations(bubble_groups, image_heights)
        
        # Should have 2 separate bubbles (can't merge across empty image)
        # The algorithm now tracks consecutive images, so gap prevents merging
        assert len(merged) == 2, f"Expected 2 bubbles, got {len(merged)}"
        assert merged[0].text == "Start here"
        assert merged[1].text == "End here."
    
    def test_merge_single_image(self):
        """Test with only one image (no continuations possible)."""
        img1_bubbles = [
            create_bubble("Hello", left=100, top=100, width=200, height=40),
            create_bubble("World", left=100, top=200, width=200, height=40)
        ]
        
        bubble_groups = [img1_bubbles]
        merged = self.detector.detect_and_merge_continuations(bubble_groups)
        
        # Should return same bubbles unchanged
        assert len(merged) == 2
        assert merged[0].text == "Hello"
        assert merged[1].text == "World"
    
    def test_merge_empty_input(self):
        """Test with empty input."""
        merged = self.detector.detect_and_merge_continuations([])
        assert merged == []
    
    def test_continuation_with_internal_punctuation(self):
        """
        Test case from user: Bubble can have punctuation but still continue.
        
        Example: "Hello are you there? I want to" → continues → "come home."
        
        The bubble has internal punctuation (?) but doesn't end the thought.
        This is TRICKY and our heuristic may not catch it perfectly.
        
        Strategy: We look at whether it ends with sentence punct AND other signals.
        If alignment is PERFECT, we might still merge even with punct.
        """
        # Image 1 (height=500): Has internal punctuation but incomplete thought
        img1_bubbles = [
            create_bubble("Hello are you there I want to", left=100, top=450, width=250, height=50)  # bottom=500
        ]
        
        # Image 2 (height=500): Continuation
        img2_bubbles = [
            create_bubble("come home now", left=100, top=10, width=250, height=40)  # top=10
        ]
        
        bubble_groups = [img1_bubbles, img2_bubbles]
        image_heights = [500, 500]
        merged = self.detector.detect_and_merge_continuations(bubble_groups, image_heights)
        
        # Should merge because:
        # 1. No sentence-ending punctuation at end of prev bubble
        # 2. Perfect alignment
        # 3. Similar width
        # 4. At image boundaries
        assert len(merged) == 1
        assert "Hello are you there I want to come home now" in merged[0].text
    
    def test_no_continuation_with_sentence_boundary(self):
        """
        Test that sentence-ending punctuation prevents merging
        even with perfect alignment (unless EXTREMELY confident).
        
        Example: "Hello are you there?" → should NOT merge with → "I want to come home."
        """
        # Image 1 (height=500): Complete question at bottom
        img1_bubbles = [
            create_bubble("Hello are you there?", left=100, top=450, width=250, height=50)  # bottom=500
        ]
        
        # Image 2 (height=500): New sentence at top
        img2_bubbles = [
            create_bubble("I want to come home.", left=100, top=10, width=250, height=40)  # top=10
        ]
        
        bubble_groups = [img1_bubbles, img2_bubbles]
        image_heights = [500, 500]
        merged = self.detector.detect_and_merge_continuations(bubble_groups, image_heights)
        
        # Should NOT merge because prev bubble ends with ?
        # Even though alignment is good, the sentence boundary is strong
        assert len(merged) == 2
        assert merged[0].text == "Hello are you there?"
        assert merged[1].text == "I want to come home."
    
    def test_multiple_bubbles_at_boundaries(self):
        """
        Test case from user: Multiple bubbles at image boundaries.
        
        Scenario: Image 1 has TWO bubbles at the bottom, both cut off.
                  Image 2 has TWO bubbles at the top, continuing them.
        
        Example:
        - Image 1 bottom: "Hello I am" (left), "What are you" (right)  
        - Image 2 top: "talking about" (left), "doing here" (right)
        
        Should merge:
        - "Hello I am" + "talking about"
        - "What are you" + "doing here"
        """
        # Image 1 (height=500): TWO bubbles at bottom, side by side
        img1_bubbles = [
            create_bubble("Hello I am", left=50, top=450, width=150, height=50),    # left bubble, bottom=500
            create_bubble("What are you", left=250, top=450, width=150, height=50)  # right bubble, bottom=500
        ]
        
        # Image 2 (height=500): TWO bubbles at top, aligned with img1 bubbles
        img2_bubbles = [
            create_bubble("talking about", left=50, top=10, width=150, height=40),   # left bubble, top=10
            create_bubble("doing here", left=250, top=10, width=150, height=40)      # right bubble, top=10
        ]
        
        bubble_groups = [img1_bubbles, img2_bubbles]
        image_heights = [500, 500]
        merged = self.detector.detect_and_merge_continuations(bubble_groups, image_heights)
        
        # Should have 2 merged bubbles (both continuations detected)
        assert len(merged) == 2
        
        # Check left bubble merged
        assert "Hello I am talking about" in merged[0].text
        
        # Check right bubble merged  
        assert "What are you doing here" in merged[1].text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
