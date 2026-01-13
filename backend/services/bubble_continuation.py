"""
Bubble Continuation Detection Service

Handles text bubbles that span multiple images/panels in webtoon chapters.

PROBLEM:
When a single speech bubble is cut across two images:
- Image N has the top part: "Hey, what are you"
- Image N+1 has the bottom part: "doing over there?"

They are currently treated as 2 separate bubbles, causing:
1. Incorrect TTS output (two separate audio clips)
2. Awkward pauses where there shouldn't be any

SOLUTION:
Detect when the last bubble of image N and first bubble of image N+1
are likely continuations of the same bubble, then merge them.

HEURISTICS:
1. Vertical proximity: Bottom of bubble N close to top of image N+1
2. Horizontal alignment: Similar horizontal position/width
3. Text characteristics: Incomplete sentence in bubble N
4. Bounding box similarity: Similar width suggests same bubble
"""

import logging
from typing import List, Optional, Tuple
from dataclasses import dataclass

from backend.services.text_grouping import TextBubble
from backend.services.vision import BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class ContinuationMatch:
    """Details about a detected bubble continuation."""
    prev_bubble_idx: int
    next_bubble_idx: int
    confidence: float
    reason: str


class BubbleContinuationDetector:
    """
    Detects and merges text bubbles that continue across multiple images.
    """
    
    def __init__(
        self,
        max_vertical_gap: int = 50,
        min_horizontal_alignment: float = 0.6,
        min_width_similarity: float = 0.7
    ):
        """
        Initialize continuation detector.
        
        Args:
            max_vertical_gap: Max vertical distance between bottom of prev bubble
                             and top of next image (pixels)
            min_horizontal_alignment: Minimum horizontal overlap ratio (0-1)
            min_width_similarity: Minimum width similarity ratio (0-1)
        """
        self.max_vertical_gap = max_vertical_gap
        self.min_horizontal_alignment = min_horizontal_alignment
        self.min_width_similarity = min_width_similarity
        
        logger.info(
            f"BubbleContinuationDetector initialized: "
            f"max_vertical_gap={max_vertical_gap}px, "
            f"min_horizontal_alignment={min_horizontal_alignment}, "
            f"min_width_similarity={min_width_similarity}"
        )
    
    def _has_incomplete_sentence(self, text: str) -> bool:
        """
        Check if text appears to be an incomplete sentence.
        
        Indicators of INCOMPLETE:
        - Doesn't end with sentence-ending punctuation (. ! ?)
        - Ends with comma or dash (strong continuation signal)
        - Very short text (< 10 chars) even with punctuation
        
        Indicators of COMPLETE:
        - Ends with sentence-ending punctuation AND
        - Not too short AND
        - Doesn't end with continuation words
        
        Note: This is a heuristic, not perfect. We rely on other signals
        (horizontal alignment, width similarity) to make the final decision.
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears incomplete
        """
        text = text.strip()
        if not text:
            return False
        
        # Check ending punctuation first (before length checks)
        last_char = text[-1]
        
        # Ends with comma or dash - strong continuation signal
        if last_char in ',-—':
            return True
        
        # Has sentence-ending punctuation - complete sentence
        if last_char in '.!?':
            return False
        
        # No sentence-ending punctuation - incomplete
        return True
    
    def _calculate_horizontal_alignment(
        self,
        bbox1: BoundingBox,
        bbox2: BoundingBox
    ) -> float:
        """
        Calculate horizontal alignment between two bounding boxes.
        
        This checks BOTH:
        1. How much the boxes overlap horizontally (overlap ratio)
        2. How far apart their centers are
        
        Returns:
            Overlap ratio (0-1), where 1 = perfect alignment
        """
        # Calculate horizontal overlap
        overlap_left = max(bbox1.left, bbox2.left)
        overlap_right = min(bbox1.right, bbox2.right)
        overlap_width = max(0, overlap_right - overlap_left)
        
        # If there's NO overlap at all, return 0 immediately
        # This prevents merging bubbles on opposite sides of the panel
        if overlap_width == 0:
            return 0.0
        
        # Calculate alignment ratio based on overlap
        min_width = min(bbox1.width, bbox2.width)
        if min_width == 0:
            return 0.0
        
        alignment_ratio = overlap_width / min_width
        
        # Additional check: penalize if centers are far apart
        # This helps distinguish between actual continuations vs side-by-side bubbles
        center1 = (bbox1.left + bbox1.right) / 2
        center2 = (bbox2.left + bbox2.right) / 2
        center_distance = abs(center1 - center2)
        avg_width = (bbox1.width + bbox2.width) / 2
        
        # If centers are more than 50% of average width apart, penalize
        if avg_width > 0 and center_distance > (avg_width * 0.5):
            # Reduce alignment score based on how far apart centers are
            penalty = min(1.0, center_distance / avg_width)
            alignment_ratio *= (1 - penalty * 0.5)  # Up to 50% penalty
        
        return min(1.0, alignment_ratio)
    
    def _calculate_width_similarity(
        self,
        bbox1: BoundingBox,
        bbox2: BoundingBox
    ) -> float:
        """
        Calculate width similarity between two bounding boxes.
        
        Returns:
            Similarity ratio (0-1), where 1 = identical width
        """
        if bbox1.width == 0 or bbox2.width == 0:
            return 0.0
        
        width_ratio = min(bbox1.width, bbox2.width) / max(bbox1.width, bbox2.width)
        return width_ratio
    
    def _check_continuation(
        self,
        prev_bubble: TextBubble,
        next_bubble: TextBubble,
        prev_image_height: Optional[int] = None
    ) -> Optional[ContinuationMatch]:
        """
        Check if next_bubble is a continuation of prev_bubble.
        
        Args:
            prev_bubble: Last bubble from previous image
            next_bubble: First bubble from next image
            prev_image_height: Height of previous image (for proximity check)
            
        Returns:
            ContinuationMatch if continuation detected, None otherwise
        """
        prev_bbox = prev_bubble.bounding_box
        next_bbox = next_bubble.bounding_box
        
        # CRITICAL: Check vertical proximity to image boundaries
        # prev_bubble should be near BOTTOM of prev image
        # next_bubble should be near TOP of next image (y=0)
        if prev_image_height is not None:
            # Check if prev bubble is near bottom of its image
            # (within max_vertical_gap pixels from bottom)
            prev_bottom = prev_bbox.bottom
            distance_from_bottom = prev_image_height - prev_bottom
            
            # Check if next bubble is near top of its image
            next_top = next_bbox.top
            
            logger.debug(
                f"        Vertical proximity check: prev_bottom={prev_bottom}, "
                f"img_height={prev_image_height}, distance_from_bottom={distance_from_bottom} "
                f"(max={self.max_vertical_gap}), next_top={next_top} (max={self.max_vertical_gap})"
            )
            
            # Both should be within max_vertical_gap of their respective edges
            if distance_from_bottom > self.max_vertical_gap or next_top > self.max_vertical_gap:
                # Not at image boundaries - unlikely to be a split bubble
                logger.debug(
                    f"        ❌ REJECTED: Not at boundaries "
                    f"(distance_from_bottom={distance_from_bottom}, next_top={next_top})"
                )
                return None
        
        # Calculate metrics
        horizontal_alignment = self._calculate_horizontal_alignment(prev_bbox, next_bbox)
        width_similarity = self._calculate_width_similarity(prev_bbox, next_bbox)
        has_incomplete = self._has_incomplete_sentence(prev_bubble.text)
        
        logger.debug(
            f"        Metrics: h_align={horizontal_alignment:.2f} (min={self.min_horizontal_alignment}), "
            f"w_sim={width_similarity:.2f} (min={self.min_width_similarity}), "
            f"incomplete={has_incomplete}"
        )
        
        # Check if previous bubble ends with sentence-ending punctuation
        prev_text = prev_bubble.text.strip()
        ends_with_sentence_punct = prev_text and prev_text[-1] in '.!?'
        
        # Score the match
        confidence = 0.0
        reasons = []
        
        # Check horizontal alignment (REQUIRED)
        if horizontal_alignment >= self.min_horizontal_alignment:
            confidence += 0.4
            reasons.append(f"horizontal_alignment={horizontal_alignment:.2f}")
        else:
            # Poor alignment - unlikely to be continuation
            logger.debug(
                f"        ❌ REJECTED: Poor horizontal alignment "
                f"({horizontal_alignment:.2f} < {self.min_horizontal_alignment})"
            )
            return None
        
        # Check width similarity
        if width_similarity >= self.min_width_similarity:
            confidence += 0.3
            reasons.append(f"width_similarity={width_similarity:.2f}")
        
        # Check for incomplete sentence
        if has_incomplete:
            confidence += 0.3
            reasons.append("incomplete_sentence")
        
        # CRITICAL: If prev bubble ends with sentence punctuation,
        # require PERFECT alignment and width match to merge
        if ends_with_sentence_punct:
            logger.debug(f"        Sentence boundary detected (ends with: '{prev_text[-1]}')")
            # Require near-perfect metrics to override sentence boundary
            if horizontal_alignment < 0.95 or width_similarity < 0.9:
                # Not confident enough to merge across sentence boundary
                logger.debug(
                    f"        ❌ REJECTED: Sentence boundary with insufficient metrics "
                    f"(needs h_align≥0.95 AND w_sim≥0.9)"
                )
                return None
            # Even with perfect alignment, reduce confidence
            confidence *= 0.7
            reasons.append("has_sentence_punct(reduced_confidence)")
        
        # Minimum confidence threshold
        if confidence >= 0.6:
            logger.debug(f"        ✅ MATCH: confidence={confidence:.2f}, reasons=[{', '.join(reasons)}]")
            return ContinuationMatch(
                prev_bubble_idx=-1,  # Will be set by caller
                next_bubble_idx=-1,   # Will be set by caller
                confidence=confidence,
                reason=", ".join(reasons)
            )
        else:
            logger.debug(f"        ❌ REJECTED: Insufficient confidence ({confidence:.2f} < 0.6)")
            return None
    
    def detect_and_merge_continuations(
        self,
        bubble_groups: List[List[TextBubble]],
        image_heights: Optional[List[int]] = None
    ) -> List[TextBubble]:
        """
        Detect and merge bubble continuations across multiple images.
        
        Args:
            bubble_groups: List of bubble lists, one per image
            image_heights: Heights of each image (optional, for better detection)
            
        Returns:
            Merged list of bubbles with continuations combined
        """
        if not bubble_groups:
            return []
        
        if len(bubble_groups) == 1:
            # Single image - no continuations possible
            return bubble_groups[0]
        
        logger.info(f"Checking for bubble continuations across {len(bubble_groups)} images")
        
        # Track merges
        merges_detected = 0
        
        # Build merged result
        merged_bubbles = []
        last_image_idx = -1  # Track which image the last bubble came from
        
        for img_idx in range(len(bubble_groups)):
            current_bubbles = bubble_groups[img_idx]
            
            logger.info(f"Processing image {img_idx}: {len(current_bubbles)} bubbles")
            for idx, bubble in enumerate(current_bubbles):
                logger.info(f"  Bubble[{idx}]: \"{bubble.text}\" bbox=(top:{bubble.bounding_box.top}, bottom:{bubble.bounding_box.bottom})")
            
            if not current_bubbles:
                logger.info(f"Image {img_idx} has no bubbles, skipping")
                continue
            
            # Check for continuations from previous image
            # ONLY if they're from consecutive images (no empty images between)
            if img_idx > 0 and merged_bubbles and last_image_idx == img_idx - 1:
                prev_img_height = image_heights[img_idx - 1] if image_heights else None
                logger.debug(f"Checking continuations from img {last_image_idx}→{img_idx}, prev_height={prev_img_height}")
                
                # Track which bubbles from current image have been merged
                merged_current_indices = set()
                
                # Find all bubbles at the BOTTOM of previous image
                # We need to check BACKWARDS through merged_bubbles to find all from prev image
                prev_image_bubbles = []
                for i in range(len(merged_bubbles) - 1, -1, -1):
                    bubble = merged_bubbles[i]
                    # Check if this bubble is actually near the BOTTOM of prev image
                    if prev_img_height is not None:
                        distance_from_bottom = prev_img_height - bubble.bounding_box.bottom
                        if distance_from_bottom > self.max_vertical_gap:
                            # This bubble is not at the bottom, skip it
                            logger.debug(f"  Skipping prev bubble[{i}]: not at bottom (distance={distance_from_bottom})")
                            continue
                    
                    prev_image_bubbles.insert(0, (i, bubble))
                    # Stop after checking a reasonable number
                    if len(prev_image_bubbles) >= 10:  # Max bubbles to check
                        break
                
                # Find all bubbles at the TOP of current image
                current_top_bubbles = []
                for curr_idx, curr_bubble in enumerate(current_bubbles):
                    curr_top = curr_bubble.bounding_box.top
                    if curr_top <= self.max_vertical_gap:
                        # This bubble is at the top
                        current_top_bubbles.append((curr_idx, curr_bubble))
                    else:
                        logger.debug(
                            f"  Skipping curr bubble[{curr_idx}]: not at top "
                            f"(top={curr_top} > {self.max_vertical_gap})"
                        )
                
                logger.info(
                    f"Found {len(prev_image_bubbles)} bubbles at bottom of prev image, "
                    f"{len(current_top_bubbles)} bubbles at top of current image"
                )
                
                # For each bubble at bottom of prev image, try to find continuation at top of current
                for prev_idx, prev_bubble in prev_image_bubbles:
                    prev_bbox = prev_bubble.bounding_box
                    logger.info(
                        f"  Prev bubble[{prev_idx}]: \"{prev_bubble.text}\" "
                        f"bbox=(top:{prev_bbox.top}, bottom:{prev_bbox.bottom}, left:{prev_bbox.left}, right:{prev_bbox.right})"
                    )
                    
                    for curr_idx, curr_bubble in current_top_bubbles:
                        # Skip if already merged
                        if curr_idx in merged_current_indices:
                            logger.info(f"    Curr bubble[{curr_idx}] already merged, skipping")
                            continue
                        
                        curr_bbox = curr_bubble.bounding_box
                        logger.info(
                            f"    Checking curr bubble[{curr_idx}]: \"{curr_bubble.text}\" "
                            f"bbox=(top:{curr_bbox.top}, bottom:{curr_bbox.bottom}, left:{curr_bbox.left}, right:{curr_bbox.right})"
                        )
                        
                        match = self._check_continuation(
                            prev_bubble,
                            curr_bubble,
                            prev_img_height
                        )
                        
                        if match:
                            # Merge the bubbles
                            merged_text = prev_bubble.text + " " + curr_bubble.text
                            merged_bbox = curr_bubble.bounding_box
                            merged_ocr = prev_bubble.ocr_results + curr_bubble.ocr_results
                            
                            # Update the bubble in merged_bubbles
                            merged_bubbles[prev_idx] = TextBubble(
                                text=merged_text,
                                bounding_box=merged_bbox,
                                ocr_results=merged_ocr,
                                reading_order=prev_bubble.reading_order
                            )
                            
                            # Mark this current bubble as merged
                            merged_current_indices.add(curr_idx)
                            merges_detected += 1
                            
                            logger.info(
                                f"✓ Merged continuation (img {last_image_idx}→{img_idx}): "
                                f"\"{prev_bubble.text[:30]}...\" + \"{curr_bubble.text[:30]}...\" "
                                f"[confidence={match.confidence:.2f}, {match.reason}]"
                            )
                            
                            # Only merge once per prev bubble
                            break
                        else:
                            logger.debug(f"      No match (position or alignment check failed)")
                
                # Add unmerged bubbles from current image
                unmerged_count = 0
                for curr_idx, curr_bubble in enumerate(current_bubbles):
                    if curr_idx not in merged_current_indices:
                        merged_bubbles.append(curr_bubble)
                        unmerged_count += 1
                
                logger.debug(f"Added {unmerged_count} unmerged bubbles from image {img_idx}")
                last_image_idx = img_idx
            else:
                # First image or gap after empty images - add all
                logger.debug(f"First image or gap detected, adding all {len(current_bubbles)} bubbles")
                merged_bubbles.extend(current_bubbles)
                last_image_idx = img_idx
        
        logger.info(
            f"Bubble continuation detection complete: "
            f"{merges_detected} merges detected, "
            f"{sum(len(g) for g in bubble_groups)} → {len(merged_bubbles)} bubbles"
        )
        
        return merged_bubbles
