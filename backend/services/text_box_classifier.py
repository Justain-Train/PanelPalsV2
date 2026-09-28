"""
Text Box Classification Service

Classifies OCR-detected text regions as either:
- TEXT BOX: Dialogue or narration (proceed to TTS)
- BACKGROUND TEXT: Sound effects, signs, decorative text (filter out)

Uses multiple heuristic features with weighted scoring.
"""

import json
import logging
from pathlib import Path
from typing import List, Tuple, Optional
from dataclasses import dataclass

from backend.config import settings
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
    score: float  # Heuristic formula score
    features: dict  # Feature values for debugging/logging
    model_prob: Optional[float] = None  # ML P(dialogue), when a model is loaded


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
        weight_ocr_noise: float = 0.03,
        # ML classifier (defaults come from settings)
        mode: Optional[str] = None,
        model_path: Optional[str] = None,
        model_threshold: Optional[float] = None
    ):
        """
        Initialize text box classifier with spatial and language features.

        Args:
            classification_threshold: Minimum score to classify as TEXT BOX
            weight_*: Feature weights (must sum to 1.0)
                Spatial features: bbox_area, word_count, text_density, aspect_ratio, punctuation
                Language features: dictionary_ratio, alphabet_ratio, word_frequency,
                                   trigram_score, ocr_noise
            mode: 'model' or 'heuristic' (default: settings.CLASSIFIER_MODE)
            model_path: Trained model file (default: settings.ML_MODEL_PATH)
            model_threshold: Min P(dialogue) in model mode (default: settings.ML_THRESHOLD)
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
        self.collect_ml_data = True
        self.ml_data_collector = MLDataCollector()

        # ML classifier. In heuristic mode the model (if present) still runs so
        # its probability is logged and collected for comparison.
        self.mode = (mode or settings.CLASSIFIER_MODE).lower()
        if self.mode not in ("model", "heuristic"):
            logger.warning(f"Unknown CLASSIFIER_MODE '{self.mode}', using heuristic")
            self.mode = "heuristic"
        self.model_threshold = model_threshold if model_threshold is not None else settings.ML_THRESHOLD
        self.model, self.model_feature_names = self._load_model(model_path or settings.ML_MODEL_PATH)
        if self.mode == "model" and self.model is None:
            logger.warning("CLASSIFIER_MODE=model but no usable model loaded - falling back to heuristic")

        logger.info(
            f"TextBoxClassifier initialized with language features: "
            f"threshold={classification_threshold}, weights={self.weights}, "
            f"mode={self.mode}, model={'loaded' if self.model is not None else 'none'}, "
            f"model_threshold={self.model_threshold}"
        )

    @staticmethod
    def _load_model(model_path: str):
        """
        Load the trained classifier and its feature order.

        Returns (model, feature_names), or (None, None) if the model is missing
        or unusable - the caller then falls back to the heuristic.
        """
        path = Path(model_path)
        if not path.exists():
            logger.info(f"No ML model at {path}")
            return None, None

        try:
            import joblib
            model = joblib.load(path)
            metadata = json.loads((path.parent / "model_metadata.json").read_text())
            feature_names = metadata["feature_names"]

            if not all(name.startswith("feature_") for name in feature_names):
                raise ValueError("feature names must start with 'feature_'")
            if getattr(model, "n_features_in_", len(feature_names)) != len(feature_names):
                raise ValueError(
                    f"model expects {model.n_features_in_} features, metadata lists {len(feature_names)}"
                )
            model_names = getattr(model, "feature_names_in_", None)
            if model_names is not None and list(model_names) != feature_names:
                raise ValueError("feature order in metadata doesn't match the model")

            # Training used n_jobs=-1; per-panel batches are tiny, so threads only add overhead
            if hasattr(model, "n_jobs"):
                model.n_jobs = 1

            logger.info(f"Loaded ML classifier {metadata.get('best_model', type(model).__name__)} from {path}")
            return model, feature_names
        except Exception as e:
            logger.warning(f"Could not load ML model from {path}: {e}")
            return None, None

    def _model_probabilities(self, features_list: List[dict]) -> Optional[List[float]]:
        """
        P(dialogue) for each feature dict, or None if no model / prediction fails.

        Converts features the same way the training pipeline did: booleans → 1/0,
        missing values → 0.0.
        """
        if self.model is None or not features_list:
            return None

        import pandas as pd

        def to_float(value) -> float:
            if value is None:
                return 0.0
            return float(value)  # bool → 1.0/0.0

        rows = [
            [to_float(features.get(name[len("feature_"):])) for name in self.model_feature_names]
            for features in features_list
        ]
        try:
            X = pd.DataFrame(rows, columns=self.model_feature_names)
            return [float(p) for p in self.model.predict_proba(X)[:, 1]]
        except Exception as e:
            logger.warning(f"ML prediction failed, using heuristic: {e}")
            return None

    def _decide(self, features_list: List[dict]) -> List[Tuple[float, Optional[float], bool]]:
        """
        Classify a batch of feature dicts.

        Returns (heuristic_score, model_prob, is_text_box) per item. The model
        decides when mode is 'model' and a probability is available; otherwise
        the heuristic score is compared against the formula threshold.
        """
        probs = self._model_probabilities(features_list)
        decisions = []
        for i, features in enumerate(features_list):
            score = self._compute_score(features)
            prob = probs[i] if probs is not None else None
            if self.mode == "model" and prob is not None:
                is_text_box = prob >= self.model_threshold
            else:
                is_text_box = score >= self.threshold
            decisions.append((score, prob, is_text_box))
        return decisions
    
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
            if len(ocr.text.split()) <= 2:
                features = handle_short_dialogue(ocr.text, features)

            all_features.append(features)

        results = []
        for ocr, features, (score, prob, is_text_box) in zip(
            ocr_results, all_features, self._decide(all_features)
        ):
            result = ClassificationResult(
                ocr_result=ocr,
                is_text_box=is_text_box,
                score=score,
                features=features,
                model_prob=prob
            )
            results.append(result)

            prob_str = f", model={prob:.2f}" if prob is not None else ""
            if is_text_box:
                logger.info(
                    f"✓ ACCEPTED as dialogue: '{ocr.text}' (score={score:.3f}{prob_str})\n"
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
                    f"✗ Filtered background: '{ocr.text[:30]}' (score={score:.2f}{prob_str})"
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
            punctuation_score = 1.0
        elif has_any_punct:
            punctuation_score = 0.6
        else:
            punctuation_score = 0.2

        # Preprocess text before computing language features
        preprocessed_text = self.preprocessor.preprocess_for_classification(text)

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

        # Language features
        # 6. Dictionary word ratio
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
        
        # Non-English penalty: force language scores to 0 if >50% non-ASCII
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
            dictionary_ratio_score = 0.0
            alphabet_ratio_score = 0.0
            word_frequency_score = 0.0
            trigram_score = 0.0
            if not has_ending_punct:
                punctuation_score = 0.0

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
            # Raw values
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
        
        # Compute features for every bubble, then classify the panel in one batch
        all_features = []
        for bubble, pseudo_ocr in zip(bubbles, pseudo_ocr_results):
            features = self._compute_features(pseudo_ocr, image_area, pseudo_ocr_results)

            # Apply edge case handling for short dialogue
            if len(bubble.text.split()) <= 2:
                features = handle_short_dialogue(bubble.text, features)

            all_features.append(features)

        filtered_bubbles = []
        for bubble, features, (score, prob, is_text_box) in zip(
            bubbles, all_features, self._decide(all_features)
        ):
            # Collect ML training data (always enabled). `score` stays the
            # heuristic score so collected data keeps a formula baseline.
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
                        'threshold': self.threshold,
                        'model_prob': prob
                    })
                logger.debug(f"📊 Collected ML sample: '{bubble.text[:30]}' (score={score:.2f})")

            prob_str = f", model={prob:.2f}" if prob is not None else ""
            if is_text_box:
                filtered_bubbles.append(bubble)
            else:
                logger.info(
                    f"Filtered background bubble: '{bubble.text[:30]}...' (score={score:.2f}{prob_str}) {features}"
                )
        
        logger.info(
            f"Filtered {len(bubbles)} bubbles → {len(filtered_bubbles)} dialogue bubbles "
            f"({len(bubbles) - len(filtered_bubbles)} background bubbles filtered)"
        
        )
        
        return filtered_bubbles

