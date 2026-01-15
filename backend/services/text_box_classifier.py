"""
Text Box Classification Service

Classifies OCR-detected text regions as either:
- TEXT BOX: Dialogue or narration (proceed to TTS)
- BACKGROUND TEXT: Sound effects, signs, decorative text (filter out)

Uses multiple heuristic features with weighted scoring.
"""

import logging
from typing import List, Tuple, TYPE_CHECKING
from dataclasses import dataclass

from backend.services.vision import OCRResult

if TYPE_CHECKING:
    from backend.services.text_grouping import TextBubble

logger = logging.getLogger(__name__)


@dataclass
class ClassificationResult:
    """Result of text box classification."""
    ocr_result: OCRResult
    is_text_box: bool
    score: float
    features: dict  # Feature values for debugging/logging


class TextBoxClassifier:
    """
    Classifies OCR text regions as dialogue/narration vs background text.
    
    Uses a weighted heuristic scoring system based on:
    - Bounding box size relative to image
    - Word count
    - Text density (chars per pixel)
    - Aspect ratio
    - Punctuation presence (weak signal)
    """
    
    def __init__(
        self,
        classification_threshold: float = 0.60,  # Tuned for bubble-level classification
        # Feature weights (must sum to 1.0)
        weight_bbox_area: float = 0.30,     
        weight_word_count: float = 0.20,   
        weight_text_density: float = 0.20,
        weight_aspect_ratio: float = 0.10, 
        weight_punctuation: float = 0.20
    ):
        """
        Initialize text box classifier.
        
        Args:
            classification_threshold: Minimum score to classify as TEXT BOX
            weight_*: Feature weights (must sum to 1.0)
        """
        self.threshold = classification_threshold
        self.weights = {
            'bbox_area': weight_bbox_area,
            'word_count': weight_word_count,
            'text_density': weight_text_density,
            'aspect_ratio': weight_aspect_ratio,
            'punctuation': weight_punctuation
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Feature weights must sum to 1.0, got {total_weight}")
        
        logger.info(
            f"TextBoxClassifier initialized: threshold={classification_threshold}, "
            f"weights={self.weights}"
        )
    
    def classify_regions(
        self,
        ocr_results: List[OCRResult],
        image_width: int,
        image_height: int
    ) -> List[ClassificationResult]:
        """
        Classify all OCR regions in an image.
        
        Args:
            ocr_results: List of OCR results to classify
            image_width: Width of source image (pixels)
            image_height: Height of source image (pixels)
            
        Returns:
            List of classification results
        """
        if not ocr_results:
            logger.info("No OCR results to classify")
            return []
        
        image_area = image_width * image_height
        
        # Compute features for all regions
        all_features = []
        for ocr in ocr_results:
            features = self._compute_features(ocr, image_area, ocr_results)
            all_features.append(features)
        
        # Classify each region
        results = []
        for ocr, features in zip(ocr_results, all_features):
            score = self._compute_score(features)
            is_text_box = score >= self.threshold
            
            result = ClassificationResult(
                ocr_result=ocr,
                is_text_box=is_text_box,
                score=score,
                features=features
            )
            results.append(result)
            
            logger.debug(
                f"Classified '{ocr.text[:30]}': "
                f"{'TEXT_BOX' if is_text_box else 'BACKGROUND'} "
                f"(score={score:.2f})"
            )
        
        # Summary
        text_box_count = sum(1 for r in results if r.is_text_box)
        background_count = len(results) - text_box_count
        logger.info(
            f"Classified {len(results)} regions: "
            f"{text_box_count} TEXT_BOX, {background_count} BACKGROUND"
        )
        
        return results
    
    def _compute_features(
        self,
        ocr: OCRResult,
        image_area: int,
        all_ocr_results: List[OCRResult]
    ) -> dict:
        """
        Compute heuristic features for a single OCR region.
        
        Returns:
            Dict of feature names to normalized scores [0, 1]
        """
        bbox = ocr.bounding_box
        text = ocr.text.strip()
        
        # Pattern detection for special cases
        import re
        
        # Timestamp pattern (XX:XX or XX:XX/XX:XX etc.)
        timestamp_pattern = r'^\d{1,2}\s*:\s*\d{2}(/\d{1,2}\s*:\s*\d{2})?$'
        is_timestamp = bool(re.match(timestamp_pattern, text))
        
        # Warning/disclaimer pattern
        warning_keywords = ['warning', 'disclaimer', 'episode contains', 'viewer discretion',
                           'may be unsuitable', 'trigger warning', 'content warning']
        text_lower = text.lower()
        is_warning_text = any(keyword in text_lower for keyword in warning_keywords)
        
        # If timestamp, immediately classify as background
        if is_timestamp:
            return {
                'bbox_area': 0.0,
                'word_count': 0.0,
                'text_density': 0.0,
                'aspect_ratio': 0.0,
                'punctuation': 0.0,
                'raw_bbox_area': 0,
                'raw_word_count': 0,
                'raw_density': 0,
                'raw_aspect_ratio': 0,
                'raw_has_punctuation': False,
                'raw_has_any_punct': False,
                'is_timestamp': True
            }
        
        # 1. Bounding box area (relative to image)
        bbox_area = bbox.width * bbox.height
        bbox_area_ratio = bbox_area / image_area if image_area > 0 else 0
        
        # Dialogue boxes typically 1-5% of image area
        # Background text is usually either:
        # - Very small (<0.5% of image) - small signs, labels
        # - Very large (>10% of image) - huge decorative text
        # BUT: Warning/disclaimer text can be large (10-20%) and should still be accepted
        
        # Special handling for warning/disclaimer text
        if is_warning_text and bbox_area_ratio > 0.06:
            # Warning text gets good score even if large
            bbox_area_score = 0.9
        elif bbox_area_ratio < 0.003:
            # Very small boxes (<0.3% of image) - likely small labels/signs
            bbox_area_score = 0.2
        elif 0.003 <= bbox_area_ratio < 0.008:
            # Small but reasonable (0.3-0.8%) - could be dialogue
            bbox_area_score = 0.6
        elif 0.008 <= bbox_area_ratio <= 0.06:
            # Optimal range (0.8-6%) - typical dialogue boxes
            bbox_area_score = 1.0
        elif 0.06 < bbox_area_ratio <= 0.13:
            # Large (6-12%) - could be dialogue but suspicious
            bbox_area_score = 0.5
        elif bbox_area_ratio > 0.13:
            # Very large (>12%) - likely decorative/background text
            bbox_area_score = 0.2
        else:
            bbox_area_score = 0.4
        
        # 2. Word count (discriminator - but not too harsh)
        words = text.split()
        word_count = len(words)
        
        # Check for UI-specific patterns (usernames, buttons, labels)
        ui_keywords = ['follow', 'search', 'user', 'followers', 'settings', 'profile',
                       'back', 'next', 'cancel', 'submit', 'login', 'logout']
        text_lower = text.lower()
        has_ui_keyword = any(keyword in text_lower for keyword in ui_keywords)
        
        # Balance: Short dialogue exists ("Oh!", "Wait!") vs background ("Search", "Follow")
        # Rely more on OTHER features (punctuation, size, density) to distinguish
        if word_count == 1:
            if has_ui_keyword:
                word_count_score = 0.0  # Strong penalty for UI keywords
            else:
                word_count_score = 0.3  # Reduced penalty - let other features decide
        elif word_count == 2:
            if has_ui_keyword:
                word_count_score = 0.1  # Penalize UI patterns
            else:
                word_count_score = 0.5
        elif word_count == 3:
            word_count_score = 0.6
        elif word_count == 4:
            word_count_score = 0.65
        elif word_count == 5:
            word_count_score = 0.7
        elif word_count >= 9:
            if has_ui_keyword:
                word_count_score = 0.4  # Even long text with UI keywords suspicious
            else:
                word_count_score = min(1.0, 0.7 + (word_count - 3) * 0.15)
        else:
            word_count_score = 0.0
        
        # 3. Text density (chars per pixel)
        char_count = len(text)
        density = char_count / bbox_area if bbox_area > 0 else 0
        
        # Dialogue typically has LOWER density (large boxes with spacing)
        # Background text is often DENSE (small boxes with compact text)
        if density < 0.0005:
            density_score = 0.6  # Very sparse
        elif 0.0005 <= density <= 0.0015:
            density_score = 1.0  # Optimal density for dialogue (lower is better)
        elif 0.0015 < density <= 0.005:
            density_score = 0.5  # Higher density - could be background
        elif 0.005 < density <= 0.02:
            density_score = 0.3  # High density - likely background
        elif density > 0.02:
            density_score = 0.1  # Very dense - definitely background
        else:
            density_score = 0.5
        
        # 4. Aspect ratio
        aspect_ratio = bbox.width / bbox.height if bbox.height > 0 else 1.0
        
        # Dialogue boxes typically 2:1 to 4:1 (wider than tall)
        if 2.0 <= aspect_ratio <= 5.0:
            aspect_ratio_score = 1.0
        elif 1.0 <= aspect_ratio < 6.0:
            aspect_ratio_score = 0.7  # Slightly square is ok
        elif 6.0 < aspect_ratio <= 8.0:
            aspect_ratio_score = 0.5  # Very wide
        else:
            aspect_ratio_score = 0.3  # Extreme aspect ratio
        
        # 5. Punctuation presence (helps distinguish short dialogue from UI text)
        sentence_endings = ['.', '!', '?', '...', '…']
        has_ending_punct = any(text.endswith(p) for p in sentence_endings)
        
        # Also check for ANY punctuation (helps catch "SURE ! ~" style dialogue)
        has_any_punct = any(p in text for p in '.!?,;:—-~')
        
        if has_ending_punct:
            punctuation_score = 1.0  # Clear sentence ending
        elif has_any_punct:
            punctuation_score = 0.6  # Has punctuation but not at end
        else:
            punctuation_score = 0.2  # No punctuation - likely UI text or incomplete
        
        return {
            'bbox_area': bbox_area_score,
            'word_count': word_count_score,
            'text_density': density_score,
            'aspect_ratio': aspect_ratio_score,
            'punctuation': punctuation_score,
            # Raw values for debugging
            'raw_bbox_area': bbox_area,
            'raw_word_count': word_count,
            'raw_density': density,
            'raw_aspect_ratio': aspect_ratio,
            'raw_has_punctuation': has_ending_punct,
            'raw_has_any_punct': has_any_punct
        }
    
    def _compute_score(self, features: dict) -> float:
        """
        Compute weighted final score from features.
        
        Args:
            features: Dict of feature scores
            
        Returns:
            Final score [0, 1]
        """
        score = 0.0
        for feature_name, weight in self.weights.items():
            feature_value = features.get(feature_name, 0.0)
            score += feature_value * weight
        
        return score
    
    def filter_text_boxes(
        self,
        ocr_results: List[OCRResult],
        image_width: int,
        image_height: int
    ) -> List[OCRResult]:
        """
        Filter OCR results to only include TEXT BOX regions.
        
        Convenience method that classifies and returns only text boxes.
        
        Args:
            ocr_results: List of OCR results
            image_width: Width of source image
            image_height: Height of source image
            
        Returns:
            Filtered list containing only TEXT BOX regions
        """
        classifications = self.classify_regions(ocr_results, image_width, image_height)
        text_boxes = [c.ocr_result for c in classifications if c.is_text_box]
        
        logger.info(
            f"Filtered {len(ocr_results)} regions → {len(text_boxes)} text boxes "
            f"({len(ocr_results) - len(text_boxes)} background text filtered)"
        )
        
        return text_boxes
    
    def filter_text_bubbles(
        self,
        bubbles: List["TextBubble"],
        image_width: int,
        image_height: int
    ) -> List["TextBubble"]:
        """
        Filter text bubbles to only include dialogue/narration.
        
        Removes background text like sound effects and signs.
        Uses the grouped bubble's combined text and bounding box for classification.
        
        Args:
            bubbles: List of TextBubble objects
            image_width: Width of source image
            image_height: Height of source image
            
        Returns:
            Filtered list containing only dialogue/narration bubbles
        """
        if not bubbles:
            return []
        
        image_area = image_width * image_height
        
        # Create pseudo OCR results from bubbles for feature computation
        pseudo_ocr_results = []
        for bubble in bubbles:
            # Create a temporary OCRResult-like object with bubble's text and bbox
            pseudo_ocr = type('obj', (object,), {
                'text': bubble.text,
                'bounding_box': bubble.bounding_box
            })()
            pseudo_ocr_results.append(pseudo_ocr)
        
        # Compute features and scores for each bubble
        filtered_bubbles = []
        for bubble, pseudo_ocr in zip(bubbles, pseudo_ocr_results):
            features = self._compute_features(pseudo_ocr, image_area, pseudo_ocr_results)
            score = self._compute_score(features)
            is_text_box = score >= self.threshold
            
            if is_text_box:
                filtered_bubbles.append(bubble)
            else:
                logger.debug(
                    f"Filtered background bubble: '{bubble.text[:30]}...' (score={score:.2f})"
                )
        
        logger.info(
            f"Filtered {len(bubbles)} bubbles → {len(filtered_bubbles)} dialogue bubbles "
            f"({len(bubbles) - len(filtered_bubbles)} background bubbles filtered)"
        )
        
        return filtered_bubbles
    
