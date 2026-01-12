"""
Unit tests for Text Bubble Grouping Service

Section 14.3: Text Bubble Grouping Test Cases
Tests spatial proximity grouping algorithm with various edge cases.
"""

import pytest
from backend.services.text_grouping import TextBubbleGrouper, TextBubble
from backend.services.vision import OCRResult, BoundingBox


@pytest.fixture
def grouper():
    """Create a TextBubbleGrouper with default settings."""
    return TextBubbleGrouper(max_vertical_gap=35, max_center_shift=40)


def create_ocr_result(text: str, left: int, top: int, right: int, bottom: int) -> OCRResult:
    """Helper to create OCRResult for testing."""
    vertices = [
        {"x": left, "y": top},
        {"x": right, "y": top},
        {"x": right, "y": bottom},
        {"x": left, "y": bottom}
    ]
    bbox = BoundingBox(vertices)
    return OCRResult(text=text, bounding_box=bbox)


# Basic Grouping Tests

@pytest.mark.unit
def test_single_word_bubble(grouper):
    """
    Test single-word bubble.
    Section 14.3: Single-word bubbles
    """
    ocr_results = [
        create_ocr_result("Hello", 10, 20, 50, 40)
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    assert len(bubbles) == 1
    assert bubbles[0].text == "Hello"
    assert bubbles[0].reading_order == 1


@pytest.mark.unit
def test_multi_line_dialogue(grouper):
    """
    Test multi-line dialogue in same bubble.
    Section 14.3: Multi-line dialogue
    """
    ocr_results = [
        create_ocr_result("Hello", 10, 20, 50, 40),
        create_ocr_result("world", 10, 45, 50, 65),  # Close vertically, aligned horizontally
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    assert len(bubbles) == 1
    assert bubbles[0].text == "Hello world"
    assert bubbles[0].reading_order == 1
    assert len(bubbles[0].ocr_results) == 2


@pytest.mark.unit
def test_separate_bubbles(grouper):
    """
    Test two separate bubbles that should not merge.
    Section 14.3: Closely spaced bubbles
    """
    ocr_results = [
        create_ocr_result("First", 10, 20, 50, 40),
        create_ocr_result("bubble", 10, 45, 50, 65),
        # Large vertical gap - should be separate bubble
        create_ocr_result("Second", 10, 120, 60, 140),
        create_ocr_result("bubble", 10, 145, 60, 165),
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    assert len(bubbles) == 2
    assert bubbles[0].text == "First bubble"
    assert bubbles[1].text == "Second bubble"
    assert bubbles[0].reading_order == 1
    assert bubbles[1].reading_order == 2


@pytest.mark.unit
def test_horizontal_shift_separates_bubbles(grouper):
    """
    Test that large horizontal shift creates separate bubbles.
    """
    ocr_results = [
        create_ocr_result("Left", 10, 20, 50, 40),
        # Large horizontal shift but close vertically
        create_ocr_result("Right", 150, 25, 200, 45),
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    assert len(bubbles) == 2
    assert bubbles[0].text == "Left"
    assert bubbles[1].text == "Right"


# Reading Order Tests

@pytest.mark.unit
def test_reading_order_top_to_bottom(grouper):
    """
    Test reading order is top-to-bottom.
    Section 14.3: Correct reading order
    """
    ocr_results = [
        create_ocr_result("Bottom", 10, 150, 60, 170),  # Larger gaps to ensure separation
        create_ocr_result("Top", 10, 20, 40, 40),
        create_ocr_result("Middle", 10, 80, 60, 100),
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    assert len(bubbles) == 3
    assert bubbles[0].text == "Top"
    assert bubbles[1].text == "Middle"
    assert bubbles[2].text == "Bottom"
    assert bubbles[0].reading_order == 1
    assert bubbles[1].reading_order == 2
    assert bubbles[2].reading_order == 3


@pytest.mark.unit
def test_reading_order_left_to_right_same_height(grouper):
    """
    Test reading order is left-to-right when at same height.
    Words close together on the same line should merge into one bubble.
    """
    ocr_results = [
        create_ocr_result("Right", 100, 20, 150, 40),
        create_ocr_result("Left", 10, 20, 50, 40),
        create_ocr_result("Center", 60, 20, 95, 40),
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    # Words on same line with small gaps should merge
    assert len(bubbles) == 1
    # Reading order should be left-to-right
    assert bubbles[0].text == "Left Center Right"


# Edge Cases

@pytest.mark.unit
def test_empty_input(grouper):
    """Test handling of empty input."""
    bubbles = grouper.group_into_bubbles([])
    
    assert bubbles == []


@pytest.mark.unit
def test_overlapping_bounding_boxes(grouper):
    """
    Test handling of overlapping bounding boxes.
    Section 14.2: Overlapping speech bubbles
    """
    ocr_results = [
        create_ocr_result("Overlap", 10, 20, 60, 40),
        create_ocr_result("ping", 50, 30, 90, 50),  # Overlaps vertically and horizontally
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    # Should merge due to close proximity
    assert len(bubbles) == 1
    assert bubbles[0].text == "Overlap ping"


@pytest.mark.unit
def test_bubbles_near_panel_edges(grouper):
    """
    Test bubbles near panel edges (coordinate 0).
    Section 14.3: Bubbles near panel edges
    """
    ocr_results = [
        create_ocr_result("Edge", 0, 0, 40, 20),
        create_ocr_result("text", 0, 25, 40, 45),
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    assert len(bubbles) == 1
    assert bubbles[0].text == "Edge text"
    assert bubbles[0].bounding_box.left == 0
    assert bubbles[0].bounding_box.top == 0


@pytest.mark.unit
def test_long_multi_line_bubble(grouper):
    """Test bubble with many lines."""
    ocr_results = [
        create_ocr_result("Line", 10, 20, 50, 40),
        create_ocr_result("one", 10, 45, 50, 65),
        create_ocr_result("Line", 10, 70, 50, 90),
        create_ocr_result("two", 10, 95, 50, 115),
        create_ocr_result("Line", 10, 120, 50, 140),
        create_ocr_result("three", 10, 145, 50, 165),
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    assert len(bubbles) == 1
    assert bubbles[0].text == "Line one Line two Line three"
    assert len(bubbles[0].ocr_results) == 6


# Bounding Box Merging Tests

@pytest.mark.unit
def test_merged_bounding_box_encompasses_all(grouper):
    """Test that merged bounding box encompasses all constituent boxes."""
    ocr_results = [
        create_ocr_result("Top", 20, 10, 60, 30),
        create_ocr_result("Bottom", 10, 35, 70, 55),  # Wider and lower
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    assert len(bubbles) == 1
    merged_bbox = bubbles[0].bounding_box
    
    # Should encompass both boxes
    assert merged_bbox.left == 10
    assert merged_bbox.right == 70
    assert merged_bbox.top == 10
    assert merged_bbox.bottom == 55


@pytest.mark.unit
def test_single_box_merge_returns_same_box(grouper):
    """Test that merging a single box returns the same box."""
    ocr_results = [
        create_ocr_result("Solo", 10, 20, 50, 40)
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    
    original_bbox = ocr_results[0].bounding_box
    merged_bbox = bubbles[0].bounding_box
    
    assert merged_bbox.left == original_bbox.left
    assert merged_bbox.right == original_bbox.right
    assert merged_bbox.top == original_bbox.top
    assert merged_bbox.bottom == original_bbox.bottom


# Metadata Tests

@pytest.mark.unit
def test_bubble_to_dict(grouper):
    """Test TextBubble serialization."""
    ocr_results = [
        create_ocr_result("Hello", 10, 20, 50, 40),
        create_ocr_result("world", 10, 45, 50, 65),
    ]
    
    bubbles = grouper.group_into_bubbles(ocr_results)
    bubble_dict = bubbles[0].to_dict()
    
    assert bubble_dict["text"] == "Hello world"
    assert bubble_dict["reading_order"] == 1
    assert bubble_dict["word_count"] == 2
    assert "bounding_box" in bubble_dict


@pytest.mark.unit
def test_group_with_metadata(grouper):
    """Test grouping with metadata output."""
    ocr_results = [
        create_ocr_result("First", 10, 20, 50, 40),
        create_ocr_result("bubble", 10, 45, 50, 65),
        create_ocr_result("Second", 10, 120, 60, 140),
    ]
    
    result = grouper.group_into_bubbles_with_metadata(ocr_results)
    
    assert result["total_bubbles"] == 2
    assert result["total_words"] == 3
    assert result["config"]["max_vertical_gap"] == 35
    assert result["config"]["max_center_shift"] == 40
    assert len(result["bubbles"]) == 2


# Custom Configuration Tests

@pytest.mark.unit
def test_custom_vertical_gap():
    """Test custom vertical gap configuration."""
    # Very strict vertical gap
    strict_grouper = TextBubbleGrouper(max_vertical_gap=5, max_center_shift=40)
    
    ocr_results = [
        create_ocr_result("Line1", 10, 20, 50, 40),
        create_ocr_result("Line2", 10, 50, 50, 70),  # Gap of 10 pixels
    ]
    
    bubbles = strict_grouper.group_into_bubbles(ocr_results)
    
    # Should NOT merge with strict gap
    assert len(bubbles) == 2


@pytest.mark.unit
def test_custom_center_shift():
    """
    Test custom center shift configuration.
    Note: center_shift is no longer used for multi-line bubbles in the updated algorithm.
    Lines are merged based on vertical gap only, as speech bubbles have varying line lengths.
    """
    # Very strict center shift (now only affects same-line merging via horizontal gap)
    strict_grouper = TextBubbleGrouper(max_vertical_gap=35, max_center_shift=5)
    
    ocr_results = [
        create_ocr_result("Left", 10, 20, 40, 40),
        create_ocr_result("Shift", 25, 45, 65, 65),  # Small vertical gap
    ]
    
    bubbles = strict_grouper.group_into_bubbles(ocr_results)
    
    # Should merge based on small vertical gap (5 pixels)
    # center_shift no longer prevents multi-line merging
    assert len(bubbles) == 1
    assert bubbles[0].text == "Left Shift"
