"""
Text Bubble Grouping Service

Section 6.1: Bubble-Level Text Grouping
Groups OCR word-level detections into text bubbles using spatial proximity heuristics.
"""

import logging
from typing import List, Dict, Any, Set
from dataclasses import dataclass, field

from backend.config import settings
from backend.services.vision import OCRResult, BoundingBox

logger = logging.getLogger(__name__)


@dataclass
class TextBubble:
    """
    Represents a grouped text bubble with reading order.
    
    Section 6.1: Each text bubble should be treated as a single dialogue unit.
    """
    text: str
    bounding_box: BoundingBox
    ocr_results: List[OCRResult] = field(default_factory=list)
    reading_order: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "text": self.text,
            "bounding_box": self.bounding_box.to_dict(),
            "reading_order": self.reading_order,
            "word_count": len(self.ocr_results)
        }


class TextBubbleGrouper:
    """
    Groups OCR results into text bubbles using spatial proximity.
    
    Section 6.1: Bubble-Level Text Grouping
    - Bounding box proximity
    - Line alignment
    - Overlap heuristics
    """
    
    def __init__(
        self,
        max_vertical_gap: int = None,
        max_center_shift: int = None
    ):
        """
        Initialize text bubble grouper.
        
        Args:
            max_vertical_gap: Maximum vertical gap between lines in same bubble (pixels)
            max_center_shift: Maximum horizontal center shift for same bubble (pixels)
        """
        self.max_vertical_gap = max_vertical_gap or settings.BUBBLE_MAX_VERTICAL_GAP
        self.max_center_shift = max_center_shift or settings.BUBBLE_MAX_CENTER_SHIFT
        
        logger.info(
            f"TextBubbleGrouper initialized: "
            f"max_vertical_gap={self.max_vertical_gap}, "
            f"max_center_shift={self.max_center_shift}"
        )
    
    def _words_are_close(self, word1: OCRResult, word2: OCRResult) -> bool:
        """
        Determine if two words are spatially close enough to be in the same bubble.
        
        Uses a combination of:
        1. Vertical proximity (for multi-line bubbles)
        2. Horizontal proximity (for same-line words)
        3. Bounding box overlap/proximity
        
        Args:
            word1: First OCR result
            word2: Second OCR result
            
        Returns:
            True if words should be in the same bubble
        """
        bbox1 = word1.bounding_box
        bbox2 = word2.bounding_box
        
        # Calculate vertical overlap (same line detection)
        vertical_overlap = min(bbox1.bottom, bbox2.bottom) - max(bbox1.top, bbox2.top)
        max_height = max(bbox1.height, bbox2.height)
        vertical_overlap_ratio = vertical_overlap / max_height if max_height > 0 else 0
        
        # Calculate gaps
        horizontal_gap = abs(bbox1.left - bbox2.right) if bbox1.left > bbox2.right else abs(bbox2.left - bbox1.right)
        vertical_gap = abs(bbox1.top - bbox2.bottom) if bbox1.top > bbox2.bottom else abs(bbox2.top - bbox1.bottom)
        
        # Same line words (>50% vertical overlap)
        if vertical_overlap_ratio > 0.5:
            # For same-line words, use STRICT horizontal proximity
            # This prevents merging side-by-side bubbles
            max_horizontal_gap = 60  # Reduced from 80 to be more conservative
            return horizontal_gap < max_horizontal_gap
        
        # Different lines - check if vertically adjacent
        # Use more lenient vertical gap for multi-line bubbles
        # But also check horizontal alignment to avoid linking distant bubbles
        else:
            # Words must be vertically close
            if vertical_gap >= self.max_vertical_gap:
                return False
            
            # AND they should have some horizontal overlap or be reasonably aligned
            # Calculate horizontal overlap/proximity
            horizontal_overlap = min(bbox1.right, bbox2.right) - max(bbox1.left, bbox2.left)
            
            # If there's horizontal overlap, they're likely in the same bubble
            if horizontal_overlap > 0:
                return True
            
            # If no overlap, check if horizontally close enough
            # Allow larger gaps for vertically adjacent lines
            max_horizontal_gap_multiline = 150
            return horizontal_gap < max_horizontal_gap_multiline
    
    def _cluster_words_into_bubbles(self, ocr_results: List[OCRResult]) -> List[List[OCRResult]]:
        """
        Cluster OCR words into speech bubbles using connected components.
        
        This is a graph-based clustering approach where:
        - Each word is a node
        - Edges connect spatially close words
        - Connected components form bubbles
        
        Args:
            ocr_results: List of OCR results
            
        Returns:
            List of bubble clusters (each cluster is a list of OCR results)
        """
        if not ocr_results:
            return []
        
        n = len(ocr_results)
        
        # Build adjacency graph - words that are close to each other
        graph: Dict[int, Set[int]] = {i: set() for i in range(n)}
        
        for i in range(n):
            for j in range(i + 1, n):
                if self._words_are_close(ocr_results[i], ocr_results[j]):
                    graph[i].add(j)
                    graph[j].add(i)
        
        # Find connected components using DFS
        visited = set()
        clusters = []
        
        def dfs(node: int, cluster: List[int]):
            """Depth-first search to find all connected words."""
            visited.add(node)
            cluster.append(node)
            for neighbor in graph[node]:
                if neighbor not in visited:
                    dfs(neighbor, cluster)
        
        for i in range(n):
            if i not in visited:
                cluster_indices = []
                dfs(i, cluster_indices)
                cluster_words = [ocr_results[idx] for idx in cluster_indices]
                clusters.append(cluster_words)
        
        logger.info(f"Clustered {n} words into {len(clusters)} bubbles using spatial proximity")
        return clusters
    
    def _merge_bounding_boxes(self, boxes: List[BoundingBox]) -> BoundingBox:
        """
        Merge multiple bounding boxes into one encompassing box.
        
        Args:
            boxes: List of bounding boxes to merge
            
        Returns:
            Merged bounding box
        """
        if not boxes:
            raise ValueError("Cannot merge empty list of bounding boxes")
        
        if len(boxes) == 1:
            return boxes[0]
        
        # Calculate encompassing bounds
        left = min(box.left for box in boxes)
        right = max(box.right for box in boxes)
        top = min(box.top for box in boxes)
        bottom = max(box.bottom for box in boxes)
        
        # Create new vertices for merged box
        vertices = [
            {"x": int(left), "y": int(top)},
            {"x": int(right), "y": int(top)},
            {"x": int(right), "y": int(bottom)},
            {"x": int(left), "y": int(bottom)}
        ]
        
        return BoundingBox(vertices)
    
    def _sort_by_reading_order(self, ocr_results: List[OCRResult]) -> List[OCRResult]:
        """
        Sort OCR results by reading order (top-to-bottom, left-to-right).
        
        Section 7: Preserve reading order based on vertical position and panel order
        
        Strategy:
        1. Group words into lines based on vertical overlap
        2. Sort lines by vertical position (top)
        3. Within each line, sort by horizontal position (left)
        
        Args:
            ocr_results: List of OCR results to sort
            
        Returns:
            Sorted list of OCR results
        """
        if not ocr_results:
            return []
        
        if len(ocr_results) == 1:
            return ocr_results
        
        # Group words into lines based on vertical overlap
        lines = []
        
        for result in ocr_results:
            # Find if this word belongs to an existing line
            placed = False
            for line in lines:
                # Check if this word overlaps vertically with any word in the line
                for line_word in line:
                    vertical_overlap = min(result.bounding_box.bottom, line_word.bounding_box.bottom) - \
                                     max(result.bounding_box.top, line_word.bounding_box.top)
                    max_height = max(result.bounding_box.height, line_word.bounding_box.height)
                    overlap_ratio = vertical_overlap / max_height if max_height > 0 else 0
                    
                    # If more than 50% vertical overlap, they're on the same line
                    if overlap_ratio > 0.5:
                        line.append(result)
                        placed = True
                        break
                if placed:
                    break
            
            # If not placed in any existing line, create new line
            if not placed:
                lines.append([result])
        
        # Sort lines by vertical position (average top of words in line)
        lines.sort(key=lambda line: sum(w.bounding_box.top for w in line) / len(line))
        
        # Within each line, sort words by horizontal position (left)
        for line in lines:
            line.sort(key=lambda w: w.bounding_box.left)
        
        # Flatten lines back into single list
        sorted_results = []
        for line in lines:
            sorted_results.extend(line)
        
        return sorted_results
    
    def group_into_bubbles(self, ocr_results: List[OCRResult]) -> List[TextBubble]:
        """
        Group OCR results into text bubbles using spatial clustering.
        
        Section 6.1: Group OCR words into text bubbles using proximity heuristics
        
        Algorithm:
        1. Cluster words into bubbles based on spatial proximity (graph-based)
        2. Sort bubbles by reading order (top-to-bottom, left-to-right)
        3. Within each bubble, sort words by reading order
        
        Args:
            ocr_results: List of OCR results from Vision API
            
        Returns:
            List of TextBubble objects in reading order
        """
        if not ocr_results:
            logger.info("No OCR results to group")
            return []
        
        logger.info(f"Grouping {len(ocr_results)} OCR results into text bubbles")
        
        # Step 1: Cluster words into bubbles using spatial proximity
        bubble_clusters = self._cluster_words_into_bubbles(ocr_results)
        
        # Step 2: Create TextBubble objects from clusters
        bubbles = []
        for cluster in bubble_clusters:
            bubble = self._create_bubble(cluster)
            bubbles.append(bubble)
        
        # Step 3: Sort bubbles by reading order (top-to-bottom, left-to-right)
        # Use the top-left corner of each bubble's bounding box
        bubbles.sort(key=lambda b: (b.bounding_box.top, b.bounding_box.left))
        
        # Step 4: Assign reading order
        for idx, bubble in enumerate(bubbles):
            bubble.reading_order = idx + 1
        
        logger.info(f"Created {len(bubbles)} text bubbles using spatial clustering")
        return bubbles
    
    def _create_bubble(self, ocr_results: List[OCRResult]) -> TextBubble:
        """
        Create a TextBubble from a list of OCR results.
        
        Args:
            ocr_results: List of OCR results to combine
            
        Returns:
            TextBubble object
        """
        # IMPORTANT: Re-sort OCR results within the bubble by reading order
        # This ensures multi-line text bubbles have correct word order
        # (top-to-bottom, then left-to-right within each line)
        sorted_results = self._sort_by_reading_order(ocr_results)
        
        # Combine text with spaces in correct reading order
        text = " ".join(result.text for result in sorted_results)
        
        # Merge bounding boxes
        bounding_boxes = [result.bounding_box for result in ocr_results]
        merged_bbox = self._merge_bounding_boxes(bounding_boxes)
        
        return TextBubble(
            text=text,
            bounding_box=merged_bbox,
            ocr_results=sorted_results  # Store sorted results
        )
    
    def group_into_bubbles_with_metadata(
        self,
        ocr_results: List[OCRResult]
    ) -> Dict[str, Any]:
        """
        Group OCR results and return with metadata.
        
        Args:
            ocr_results: List of OCR results from Vision API
            
        Returns:
            Dictionary with bubbles and processing metadata
        """
        bubbles = self.group_into_bubbles(ocr_results)
        
        return {
            "bubbles": [bubble.to_dict() for bubble in bubbles],
            "total_bubbles": len(bubbles),
            "total_words": len(ocr_results),
            "config": {
                "max_vertical_gap": self.max_vertical_gap,
                "max_center_shift": self.max_center_shift
            }
        }
