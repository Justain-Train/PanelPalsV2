"""
Text Box Classification Service

Classifies OCR-detected text regions as either:
- TEXT BOX: Dialogue or narration (proceed to TTS)
- BACKGROUND TEXT: Sound effects, signs, decorative text (filter out)

Uses multiple heuristic features with weighted scoring.
"""

import logging
from typing import List, Tuple, Optional
from dataclasses import dataclass

from backend.services.vision import OCRResult
from backend.services.text_preprocessing import TextPreprocessor
from backend.services.language_features import (
    compute_dictionary_ratio,
    compute_alphabet_ratio,
    compute_word_frequency_score,
    compute_trigram_language_score,
    compute_ocr_noise_score,
    handle_short_dialogue
)


from backend.services.text_grouping import TextBubble
from backend.ml.data_collector import MLDataCollector


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
    
    Spatial Features:
    - Bounding box size relative to image
    - Word count
    - Text density (chars per pixel)
    - Aspect ratio
    - Punctuation presence
    
    Language Features:
    - Dictionary word ratio (valid English words)
    - Alphabetic character ratio (letters vs symbols)
    - Word frequency score (common vs rare words)
    - Character trigram language score (English-like patterns)
    - OCR noise detection (repeated chars, symbols)
    """
    
    def __init__(
        self,
        classification_threshold: float = 0.60,  # Tuned for bubble-level classification
        # Spatial feature weights
        weight_bbox_area: float = 0.03,     
        weight_word_count: float = 0.07,    
        weight_text_density: float = 0.10, 
        weight_aspect_ratio: float = 0.05,
        weight_punctuation: float = 0.15,
        # Language feature weights
        weight_dictionary_ratio: float = 0.15,
        weight_alphabet_ratio: float = 0.20,
        weight_word_frequency: float = 0.15,
        weight_trigram_score: float = 0.07,
        weight_ocr_noise: float = 0.03
    ):
        """
        Initialize text box classifier with spatial and language features.
        
        Args:
            classification_threshold: Minimum score to classify as TEXT BOX
            weight_*: Feature weights (must sum to 1.0)
                Spatial features: bbox_area, word_count, text_density, aspect_ratio, punctuation
                Language features: dictionary_ratio, alphabet_ratio, word_frequency, 
                                   trigram_score, ocr_noise
        """
        self.threshold = classification_threshold
        self.weights = {
            # Spatial features
            'bbox_area': weight_bbox_area,
            'word_count': weight_word_count,
            'text_density': weight_text_density,
            'aspect_ratio': weight_aspect_ratio,
            'punctuation': weight_punctuation,
            # Language features
            'dictionary_ratio': weight_dictionary_ratio,
            'alphabet_ratio': weight_alphabet_ratio,
            'word_frequency': weight_word_frequency,
            'trigram_score': weight_trigram_score,
            'ocr_noise': weight_ocr_noise
        }
        
        # Validate weights sum to 1.0
        total_weight = sum(self.weights.values())
        if abs(total_weight - 1.0) > 0.001:
            raise ValueError(f"Feature weights must sum to 1.0, got {total_weight}")
        
        # Initialize text preprocessor
        self.preprocessor = TextPreprocessor()
        
        # ML data collection (always enabled for training)
        self.ml_data_collector = MLDataCollector()
   
        logger.info(
            f"TextBoxClassifier initialized with language features: "
            f"threshold={classification_threshold}, weights={self.weights}"
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
            
            # Apply edge case handling for short dialogue
            # This boosts scores for valid short exclamations like "NO", "WAIT!", etc.
            if len(ocr.text.split()) <= 2:
                features = handle_short_dialogue(ocr.text, features)
            
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
            
            # Enhanced logging with feature breakdown
            if is_text_box:
                # Log detailed features for accepted text (especially non-English)
                logger.info(
                    f"✓ ACCEPTED as dialogue: '{ocr.text}' (score={score:.3f})\n"
                    f"  Spatial: bbox_area={features.get('bbox_area', 0):.2f}, "
                    f"word_count={features.get('word_count', 0):.2f}, "
                    f"text_density={features.get('text_density', 0):.2f}, "
                    f"aspect_ratio={features.get('aspect_ratio', 0):.2f}, "
                    f"punctuation={features.get('punctuation', 0):.2f}\n"
                    f"  Language: dict_ratio={features.get('dictionary_ratio', 0):.2f} (raw={features.get('raw_dict_ratio', 0):.2f}), "
                    f"alpha_ratio={features.get('alphabet_ratio', 0):.2f} (raw={features.get('raw_alpha_ratio', 0):.2f}), "
                    f"word_freq={features.get('word_frequency', 0):.2f}, "
                    f"trigram={features.get('trigram_score', 0):.2f}, "
                    f"ocr_noise={features.get('ocr_noise', 0):.2f}"
                )
            else:
                logger.info(
                    f"✗ Filtered background: '{ocr.text[:30]}' (score={score:.2f})"
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
            bbox_area_score = 0.6
        else:
            bbox_area_score = 0.4
        
        # 2. Word count (weak discriminator - dialogue can be any length)
        words = text.split()
        word_count = len(words)

        # Check for UI-specific patterns (usernames, buttons, labels)
        ui_keywords = ['follow', 'search', 'user', 'followers', 'settings', 'profile',
                       'back', 'next', 'cancel', 'submit', 'login', 'logout']
        text_lower = text.lower()
        has_ui_keyword = any(keyword in text_lower for keyword in ui_keywords)

        # Word count is a WEAK signal - dialogue can be 1 word ("Wait!") or 50+ words
        # Only penalize obvious UI patterns or extremely short text without context
        if has_ui_keyword:
            # UI keywords get penalized regardless of length
            if word_count == 1:
                word_count_score = 0.0  # Single word UI label ("Follow", "Search")
            elif word_count == 2:
                word_count_score = 0.1  # Two-word UI ("Log In", "Sign Up")
            else:
                word_count_score = 0.3  # Longer UI text still suspicious
        elif word_count == 1:
            # Single word without UI keyword - could be dialogue ("Wait!", "No!")
            # Let other features (punctuation, size, position) decide
            word_count_score = 0.5
        elif word_count == 2:
            # Two words - common for short dialogue ("Oh no!", "Wait up!")
            word_count_score = 0.6
        elif word_count <= 5:
            # Short dialogue (3-5 words) - very common
            word_count_score = 0.75
        elif word_count <= 15:
            # Normal dialogue length (6-15 words) - optimal
            word_count_score = 1.0
        elif word_count <= 30:
            # Long dialogue (16-30 words) - still valid
            word_count_score = 0.9
        else:
            # Very long text (30+ words) - could be narration or dialogue
            # Slight penalty but don't reject
            word_count_score = 0.8
        
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
            density_score = 0.0  # Very dense - definitely background
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
        
        # ========================================================================
        # TEXT PREPROCESSING FOR LANGUAGE FEATURES
        # ========================================================================
        # Clean OCR artifacts and normalize text before computing language features
        # This improves accuracy by removing noise, fixing common OCR errors
        preprocessed_text = self.preprocessor.preprocess_for_classification(text)
        
        # Debug: Log preprocessing changes
        if preprocessed_text != text:
            logger.debug(f"Preprocessed: '{text}' → '{preprocessed_text}'")
        
        # Check if text is non-English (Korean, Japanese, Chinese, etc.)
        # Count non-ASCII characters
        non_ascii_count = sum(1 for c in preprocessed_text if ord(c) > 127)
        total_chars = len(preprocessed_text.replace(' ', ''))
        non_ascii_ratio = non_ascii_count / total_chars if total_chars > 0 else 0
        
        if non_ascii_ratio > 0.5:
            logger.warning(
                f"Non-English text detected: '{text}' "
                f"({non_ascii_ratio*100:.0f}% non-ASCII chars) - "
                f"Language features will score LOW"
            )
        
        # ========================================================================
        # LANGUAGE-BASED FEATURES (NEW)
        # ========================================================================
        
        # 6. Dictionary word ratio - filters gibberish and OCR errors
        dict_ratio = compute_dictionary_ratio(preprocessed_text)
        
        # Normalize to [0, 1] with boosting for high values
        if dict_ratio >= 0.8:
            dictionary_ratio_score = 1.0
        elif dict_ratio >= 0.5:
            dictionary_ratio_score = 0.7
        elif dict_ratio >= 0.3: 
            dictionary_ratio_score = 0.4
        else:
            dictionary_ratio_score = dict_ratio * 0.3
        
        # 7. Alphabetic character ratio - filters symbols and numbers
        alpha_ratio = compute_alphabet_ratio(preprocessed_text)
        
        # Normalize - prefer high alphabet content
        if alpha_ratio >= 0.7:
            alphabet_ratio_score = 1.0
        elif alpha_ratio >= 0.5:
            alphabet_ratio_score = 0.8
        elif alpha_ratio >= 0.3:
            alphabet_ratio_score = 0.5
        else:
            alphabet_ratio_score = alpha_ratio
        
        # 8. Word frequency score - dialogue uses common words
        freq_score = compute_word_frequency_score(preprocessed_text)
        
        # Normalize from [0, 5] to [0, 1]
        freq_normalized = freq_score / 5.0
        if freq_normalized >= 0.6:
            word_frequency_score = 1.0
        elif freq_normalized >= 0.4:
            word_frequency_score = 0.8
        else:
            word_frequency_score = freq_normalized
        
        # 9. Character trigram language score - English-like patterns
        trigram = compute_trigram_language_score(preprocessed_text)
        
        # Normalize from [0, 5] to [0, 1]
        trigram_score = trigram / 5.0
        
        # 10. OCR noise detection - inverted (low noise = good)
        noise = compute_ocr_noise_score(preprocessed_text)
        ocr_noise_score = 1.0 - noise  # Invert: high score = clean text
        
        # ========================================================================
        # NON-ENGLISH DETECTION AND PENALTY
        # ========================================================================
        # If text is primarily non-English (Korean, Japanese, Chinese, etc.),
        # apply a penalty since our language features are English-only
        
        # Count non-ASCII characters (Korean, Japanese, Chinese, etc.)
        non_ascii_count = sum(1 for c in text if ord(c) > 127)
        total_chars = len(text.replace(' ', ''))
        non_ascii_ratio = non_ascii_count / total_chars if total_chars > 0 else 0
        
        # If >50% non-English characters, this is likely non-English text
        # Apply heavy penalty to dictionary_ratio and alphabet_ratio scores
        if non_ascii_ratio > 0.5:
            logger.warning(
                f"Non-English text detected (should be filtered): '{text}' "
                f"({non_ascii_ratio*100:.0f}% non-ASCII) - applying penalty"
            )
            # Force language feature scores to 0 for non-English text
            dictionary_ratio_score = 0.0
            alphabet_ratio_score = 0.0
            word_frequency_score = 0.0
            trigram_score = 0.0
            # Also penalize punctuation (Korean sound effects rarely have English punctuation)
            if not has_ending_punct:
                punctuation_score = 0.0
        
        # ========================================================================
        
        return {
            # Spatial features
            'bbox_area': bbox_area_score,
            'word_count': word_count_score,
            'text_density': density_score,
            'aspect_ratio': aspect_ratio_score,
            'punctuation': punctuation_score,
            # Language features
            'dictionary_ratio': dictionary_ratio_score,
            'alphabet_ratio': alphabet_ratio_score,
            'word_frequency': word_frequency_score,
            'trigram_score': trigram_score,
            'ocr_noise': ocr_noise_score,
            # Raw values for debugging
            'raw_bbox_area': bbox_area,
            'raw_word_count': word_count,
            'raw_density': density,
            'raw_aspect_ratio': aspect_ratio,
            'raw_has_punctuation': has_ending_punct,
            'raw_has_any_punct': has_any_punct,
            'raw_dict_ratio': dict_ratio,
            'raw_alpha_ratio': alpha_ratio,
            'raw_freq_score': freq_score,
            'raw_trigram': trigram,
            'raw_noise': noise
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
        contributions = []
        
        for feature_name, weight in self.weights.items():
            feature_value = features.get(feature_name, 0.0)
            contribution = feature_value * weight
            score += contribution
            contributions.append(f"{feature_name}={feature_value:.2f}×{weight:.2f}={contribution:.3f}")
        
        # Log detailed breakdown for debugging
       # if logger.isEnabledFor(logging.DEBUG):
        logger.info(f"Score={score:.3f}: " + ", ".join(contributions))
        
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
            
            # Apply edge case handling for short dialogue
            if len(bubble.text.split()) <= 2:
                features = handle_short_dialogue(bubble.text, features)
            
            score = self._compute_score(features)
            is_text_box = score >= self.threshold
            
            # Collect ML training data (always enabled)
            if self.collect_ml_data and self.ml_data_collector:
                self.ml_data_collector.collect_sample(
                    text=bubble.text,
                    features=features,
                    score=score,
                    bbox=bubble.bounding_box,
                    panel_id=getattr(bubble, 'panel_id', None),
                    metadata={
                        'image_width': image_width,
                        'image_height': image_height,
                        'is_text_box': is_text_box,
                        'threshold': self.threshold
                    })
                logger.debug(f"📊 Collected ML sample: '{bubble.text[:30]}' (score={score:.2f})")
            
            if is_text_box:
                filtered_bubbles.append(bubble)
            else:
                logger.info(
                    f"Filtered background bubble: '{bubble.text[:30]}...' (score={score:.2f}) {features}"
                )
        
        logger.info(
            f"Filtered {len(bubbles)} bubbles → {len(filtered_bubbles)} dialogue bubbles "
            f"({len(bubbles) - len(filtered_bubbles)} background bubbles filtered)"
        
        )
        
        return filtered_bubbles

