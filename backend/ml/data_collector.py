"""
ML Training Data Collector

Section 14.1: Unit Tests
Collects OCR region features and scores for machine learning model training.

Architecture:
- Collects samples during classification
- Auto-labels high-confidence predictions (>0.70 dialogue, <0.45 background)
- Flags uncertain cases (0.45-0.70) for manual review
- Exports to clean CSV format

Usage:
    collector = MLDataCollector(output_dir="ml_data/raw")
    collector.collect_sample(text, features, score, bbox, panel_id, metadata)
    collector.save()
"""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, List

import pandas as pd

logger = logging.getLogger(__name__)


class MLDataCollector:
    """
    Collects text box classification data for ML training.
    
    Auto-labels based on heuristic score:
    - score > 0.70: dialogue
    - score < 0.45: background
    - 0.45 <= score <= 0.70: needs_review
    """
    
    # Auto-labeling thresholds
    DIALOGUE_THRESHOLD = 0.70
    BACKGROUND_THRESHOLD = 0.45
    
    def __init__(self, output_dir: str = "ml_data/raw"):
        """
        Initialize ML data collector.
        
        Args:
            output_dir: Directory to save collected samples
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.samples: List[Dict[str, Any]] = []
        self.session_start = datetime.now()
        
        logger.info(f"🎓 MLDataCollector initialized: {self.output_dir}")
    
    def collect_sample(
        self,
        text: str,
        features: Dict[str, float],
        score: float,
        bbox: Any,  # BoundingBox
        panel_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Collect a single training sample.
        
        Args:
            text: OCR extracted text
            features: Classification features (10 values)
            score: Heuristic classification score (0.0-1.0)
            bbox: Bounding box of text region
            panel_id: Panel number in chapter
            metadata: Additional context (image size, etc)
        """
        # Auto-label based on heuristic score
        if score > self.DIALOGUE_THRESHOLD:
            auto_label = "dialogue"
            needs_review = False
        elif score < self.BACKGROUND_THRESHOLD:
            auto_label = "background"
            needs_review = False
        else:
            auto_label = None
            needs_review = True
        
        # Flatten features with 'feature_' prefix for CSV columns
        feature_dict = {f"feature_{k}": v for k, v in features.items()}
        
        # Extract bbox coordinates
        bbox_dict = {
            "bbox_x": bbox.x if hasattr(bbox, 'x') else 0,
            "bbox_y": bbox.y if hasattr(bbox, 'y') else 0,
            "bbox_width": bbox.width if hasattr(bbox, 'width') else 0,
            "bbox_height": bbox.height if hasattr(bbox, 'height') else 0,
        }
        
        sample = {
            # Identifiers
            "sample_id": len(self.samples),
            "timestamp": datetime.now().isoformat(),
            
            # Text data
            "text": text,
            "text_length": len(text),
            "word_count_raw": len(text.split()),
            
            # Classification
            "score": round(score, 4),
            "auto_label": auto_label,
            "needs_review": needs_review,
            "label": auto_label,  # Will be updated during manual review
            
            # Features (all 10 classification features)
            **feature_dict,
            
            # Spatial data
            **bbox_dict,
            "panel_id": panel_id,
            
            # Metadata
            "image_width": metadata.get("image_width") if metadata else None,
            "image_height": metadata.get("image_height") if metadata else None,
            "threshold_used": metadata.get("threshold") if metadata else None,
        }
        
        self.samples.append(sample)
        
        # Log collection (throttled)
        if len(self.samples) % 100 == 0:
            dialogue_count = sum(1 for s in self.samples if s["auto_label"] == "dialogue")
            background_count = sum(1 for s in self.samples if s["auto_label"] == "background")
            review_count = sum(1 for s in self.samples if s["needs_review"])
            
            logger.info(
                f"📊 Collected {len(self.samples)} samples: "
                f"{dialogue_count} dialogue, {background_count} background, "
                f"{review_count} needs_review"
            )
    
    def save(self, filename: Optional[str] = None) -> Path:
        """
        Save collected samples to CSV.
        
        Args:
            filename: Output filename (default: collected_TIMESTAMP.csv)
            
        Returns:
            Path to saved CSV file
        """
        if not self.samples:
            logger.warning("No samples collected, skipping save")
            return None
        
        # Generate filename
        if filename is None:
            timestamp = self.session_start.strftime("%Y%m%d_%H%M%S")
            filename = f"collected_{timestamp}.csv"
        
        output_path = self.output_dir / filename
        
        # Convert to DataFrame and save
        df = pd.DataFrame(self.samples)
        
        # Reorder columns for readability
        column_order = [
            "sample_id", "timestamp", "text", "score", "auto_label", "label", 
            "needs_review", "text_length", "word_count_raw", "panel_id",
            "bbox_x", "bbox_y", "bbox_width", "bbox_height",
        ]
        
        # Add feature columns
        feature_cols = sorted([c for c in df.columns if c.startswith("feature_")])
        column_order.extend(feature_cols)
        
        # Add remaining columns
        remaining = [c for c in df.columns if c not in column_order]
        column_order.extend(remaining)
        
        df = df[column_order]
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        # Generate summary
        dialogue_count = (df["auto_label"] == "dialogue").sum()
        background_count = (df["auto_label"] == "background").sum()
        review_count = df["needs_review"].sum()
        
        logger.info(
            f"💾 Saved {len(df)} samples to {output_path}\n"
            f"   Auto-labeled: {dialogue_count} dialogue, {background_count} background\n"
            f"   Needs review: {review_count} ({review_count/len(df)*100:.1f}%)"
        )
        
        return output_path
    
    def get_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        if not self.samples:
            return {"total": 0}
        
        df = pd.DataFrame(self.samples)
        
        return {
            "total": len(df),
            "dialogue": (df["auto_label"] == "dialogue").sum(),
            "background": (df["auto_label"] == "background").sum(),
            "needs_review": df["needs_review"].sum(),
            "avg_score": df["score"].mean(),
            "score_std": df["score"].std(),
            "duration_seconds": (datetime.now() - self.session_start).total_seconds(),
        }
