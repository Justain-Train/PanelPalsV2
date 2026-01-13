"""
Unit tests for Text Box Classification Service

Tests the classifier's ability to distinguish between:
- Dialogue/narration text boxes (should pass through)
- Background text like sound effects, signs (should be filtered)
"""

import pytest
from backend.services.text_box_classifier import TextBoxClassifier, ClassificationResult
from backend.services.vision import OCRResult, BoundingBox


def create_ocr_result(text: str, left: int, top: int, width: int, height: int) -> OCRResult:
    """Helper to create OCR result with bounding box."""
    vertices = [
        {"x": left, "y": top},
        {"x": left + width, "y": top},
        {"x": left + width, "y": top + height},
        {"x": left, "y": top + height}
    ]
    bbox = BoundingBox(vertices)
    return OCRResult(text=text, bounding_box=bbox, confidence=0.95)


class TestTextBoxClassifier:
    """Test suite for TextBoxClassifier."""
    
    def test_classifier_initialization(self):
        """Test classifier initializes with correct defaults."""
        classifier = TextBoxClassifier()
        assert classifier.threshold == 0.65
        assert abs(sum(classifier.weights.values()) - 1.0) < 0.001
    
    def test_invalid_weights_raise_error(self):
        """Test that invalid weights (not summing to 1.0) raise error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            TextBoxClassifier(
                weight_bbox_area=0.5,
                weight_word_count=0.3,
                weight_text_density=0.1,
                weight_aspect_ratio=0.1,
                weight_punctuation=0.1  # Sums to 1.1
            )
    
    def test_large_dialogue_box_classified_as_text_box(self):
        """Large dialogue box with multiple words should be TEXT BOX."""
        classifier = TextBoxClassifier()
        
        # Typical dialogue: large box (400x120), multiple words, sentence
        dialogue = create_ocr_result(
            text="Hey, what are you doing over there?",
            left=100, top=500, width=400, height=120
        )
        
        results = classifier.classify_regions(
            [dialogue],
            image_width=1400,
            image_height=2000
        )
        
        assert len(results) == 1
        assert results[0].is_text_box is True
        assert results[0].score >= 0.55
        # Should score high on bbox_area, word_count, punctuation
        assert results[0].features['raw_word_count'] >= 5
    
    def test_small_background_text_classified_as_background(self):
        """Small background text should be BACKGROUND."""
        classifier = TextBoxClassifier()
        
        # Small sign or label: tiny box (80x25), few words
        background = create_ocr_result(
            text="EXIT",
            left=50, top=100, width=80, height=25
        )
        
        results = classifier.classify_regions(
            [background],
            image_width=1400,
            image_height=2000
        )
        
        assert len(results) == 1
        assert results[0].is_text_box is False
        assert results[0].score < 0.55
        # Should score low on bbox_area and word_count
        assert results[0].features['raw_word_count'] == 1
    
    def test_single_word_sound_effect_rejected(self):
        """Single-word sound effects should be BACKGROUND even if large font."""
        classifier = TextBoxClassifier()
        
        # Sound effect: medium box, single word, punctuation
        sound_effect = create_ocr_result(
            text="BAM!",
            left=200, top=300, width=120, height=80
        )
        
        results = classifier.classify_regions(
            [sound_effect],
            image_width=1400,
            image_height=2000
        )
        
        assert len(results) == 1
        assert results[0].is_text_box is False
        # Low word count should dominate even with punctuation
        assert results[0].features['raw_word_count'] == 1
    
    def test_dialogue_without_punctuation_accepted(self):
        """Dialogue without punctuation should still be TEXT BOX."""
        classifier = TextBoxClassifier()
        
        # Dialogue mid-sentence: large box, multiple words, NO punctuation
        dialogue = create_ocr_result(
            text="I think we should go back now",
            left=150, top=600, width=450, height=110
        )
        
        results = classifier.classify_regions(
            [dialogue],
            image_width=1400,
            image_height=2000
        )
        
        assert len(results) == 1
        assert results[0].is_text_box is True
        # Should pass despite no punctuation
        assert results[0].features['raw_has_punctuation'] is False
        assert results[0].score >= 0.55
    
    def test_background_text_with_punctuation_rejected(self):
        """Background text WITH punctuation should still be BACKGROUND."""
        classifier = TextBoxClassifier()
        
        # Small sign with punctuation: tiny box, few words
        sign = create_ocr_result(
            text="STOP!",
            left=80, top=150, width=90, height=30
        )
        
        results = classifier.classify_regions(
            [sign],
            image_width=1400,
            image_height=2000
        )
        
        assert len(results) == 1
        assert results[0].is_text_box is False
        # Punctuation present but other features too weak
        assert results[0].features['raw_has_punctuation'] is True
        assert results[0].score < 0.55
    
    def test_multiple_regions_classification(self):
        """Test classifying multiple mixed regions."""
        classifier = TextBoxClassifier()
        
        ocr_results = [
            # Dialogue box 1
            create_ocr_result(
                "This is a long dialogue with many words.",
                left=100, top=400, width=500, height=130
            ),
            # Sound effect
            create_ocr_result(
                "CRASH",
                left=300, top=200, width=100, height=60
            ),
            # Dialogue box 2
            create_ocr_result(
                "Are you sure about that",
                left=120, top=800, width=420, height=115
            ),
            # Small label
            create_ocr_result(
                "03",
                left=50, top=50, width=40, height=20
            ),
        ]
        
        results = classifier.classify_regions(
            ocr_results,
            image_width=1400,
            image_height=2000
        )
        
        assert len(results) == 4
        # First region: dialogue - TEXT BOX
        assert results[0].is_text_box is True
        # Second region: sound effect - BACKGROUND
        assert results[1].is_text_box is False
        # Third region: dialogue - TEXT BOX
        assert results[2].is_text_box is True
        # Fourth region: small label - BACKGROUND
        assert results[3].is_text_box is False
    
    def test_filter_text_boxes_method(self):
        """Test convenience method that filters to only text boxes."""
        classifier = TextBoxClassifier()
        
        ocr_results = [
            create_ocr_result(
                "This is dialogue with several words here.",
                left=100, top=400, width=480, height=125
            ),
            create_ocr_result(
                "BANG",
                left=200, top=150, width=90, height=55
            ),
            create_ocr_result(
                "Another dialogue bubble appears below.",
                left=110, top=700, width=460, height=120
            ),
        ]
        
        filtered = classifier.filter_text_boxes(
            ocr_results,
            image_width=1400,
            image_height=2000
        )
        
        # Should only return the 2 dialogue boxes, not the sound effect
        assert len(filtered) == 2
        assert filtered[0].text == "This is dialogue with several words here."
        assert filtered[1].text == "Another dialogue bubble appears below."
    
    def test_empty_input(self):
        """Test classifier handles empty input gracefully."""
        classifier = TextBoxClassifier()
        
        results = classifier.classify_regions([], 1400, 2000)
        assert results == []
        
        filtered = classifier.filter_text_boxes([], 1400, 2000)
        assert filtered == []
    
    def test_aspect_ratio_feature(self):
        """Test aspect ratio scoring favors wide dialogue boxes."""
        classifier = TextBoxClassifier()
        
        # Wide dialogue box (good aspect ratio ~3:1)
        wide_dialogue = create_ocr_result(
            "This is a normal dialogue bubble shape.",
            left=100, top=500, width=450, height=150
        )
        
        # Vertical text (poor aspect ratio)
        vertical_text = create_ocr_result(
            "UP",
            left=50, top=100, width=30, height=90
        )
        
        results = classifier.classify_regions(
            [wide_dialogue, vertical_text],
            image_width=1400,
            image_height=2000
        )
        
        # Wide dialogue should score higher on aspect ratio
        wide_aspect_score = results[0].features['aspect_ratio']
        vertical_aspect_score = results[1].features['aspect_ratio']
        assert wide_aspect_score > vertical_aspect_score
    
    def test_text_density_feature(self):
        """Test text density penalizes extremely dense tiny text."""
        classifier = TextBoxClassifier()
        
        # Normal dialogue: moderate density
        normal = create_ocr_result(
            "This has good spacing for reading.",
            left=100, top=500, width=400, height=120
        )
        
        # Tiny dense text (like a small watermark or label)
        dense = create_ocr_result(
            "ABCDEFGHIJKLMNOP",
            left=50, top=50, width=60, height=15
        )
        
        results = classifier.classify_regions(
            [normal, dense],
            image_width=1400,
            image_height=2000
        )
        
        # Normal should have better density score
        normal_density_score = results[0].features['text_density']
        dense_density_score = results[1].features['text_density']
        assert normal_density_score > dense_density_score
    
    def test_custom_threshold(self):
        """Test classifier with custom threshold."""
        # Lower threshold = more permissive
        permissive_classifier = TextBoxClassifier(classification_threshold=0.40)
        
        # Borderline case: medium box, 2 words
        borderline = create_ocr_result(
            "Maybe not",
            left=150, top=400, width=200, height=80
        )
        
        results = permissive_classifier.classify_regions(
            [borderline],
            image_width=1400,
            image_height=2000
        )
        
        # Should pass with lower threshold
        assert results[0].is_text_box is True or results[0].score >= 0.40
    
    def test_timestamp_pattern_filtered(self):
        """Test that timestamp patterns are immediately filtered."""
        classifier = TextBoxClassifier()
        
        # Timestamp patterns should be rejected
        timestamp_texts = [
            "26:20/60:00",
            "00:00",
            "12:34",
            "1:23/45:67",
        ]
        
        for text in timestamp_texts:
            timestamp_ocr = create_ocr_result(
                text,
                left=100,
                top=100,
                width=300,
                height=40
            )
            
            results = classifier.classify_regions(
                [timestamp_ocr],
                image_width=1600,
                image_height=2000
            )
            
            # Should be classified as BACKGROUND (False)
            assert results[0].is_text_box is False, f"Timestamp '{text}' should be filtered"
            assert results[0].score == 0.0, f"Timestamp score should be 0.0"
    
    def test_warning_text_accepted(self):
        """Test that warning/disclaimer text is accepted even if large."""
        classifier = TextBoxClassifier()
        
        warning_texts = [
            "WARNING THIS EPISODE CONTAINS STRONG LANGUAGE",
            "Disclaimer: viewer discretion advised",
            "This episode contains violence that may be unsuitable",
            "TRIGGER WARNING: violence and gore",
        ]
        
        for text in warning_texts:
            # Large bbox (14%+ of image) would normally be rejected
            # but warning text should be accepted
            warning_ocr = create_ocr_result(
                text,
                left=100,
                top=500,
                width=1400,
                height=300  # Very large
            )
            
            results = classifier.classify_regions(
                [warning_ocr],
                image_width=1600,
                image_height=2000
            )
            
            # Should be classified as TEXT BOX (True)
            assert results[0].is_text_box is True, f"Warning text '{text[:30]}...' should be accepted"
            assert results[0].score >= classifier.threshold, f"Warning score should be >= {classifier.threshold}"
