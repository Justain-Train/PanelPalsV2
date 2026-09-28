"""
ML Pipeline Unit Tests

Section 14.1: Unit Tests
Tests for ML training pipeline components.

Run with: pytest backend/ml/test_ml_pipeline.py -v
"""

import pytest
from pathlib import Path
from unittest.mock import Mock, MagicMock

from backend.ml.data_collector import MLDataCollector
from backend.services.text_box_classifier import TextBoxClassifier


class TestMLDataCollector:
    """Test MLDataCollector functionality."""
    
    def test_initialization(self, tmp_path):
        """Test collector initializes correctly."""
        collector = MLDataCollector(output_dir=str(tmp_path))
        
        assert collector.output_dir == tmp_path
        assert len(collector.samples) == 0
        assert collector.DIALOGUE_THRESHOLD == 0.70
        assert collector.BACKGROUND_THRESHOLD == 0.45
    
    def test_collect_sample_dialogue(self, tmp_path):
        """Test collecting high-confidence dialogue sample."""
        collector = MLDataCollector(output_dir=str(tmp_path))
        
        # Mock bounding box
        bbox = Mock()
        bbox.x = 100
        bbox.y = 200
        bbox.width = 300
        bbox.height = 50
        
        # Collect dialogue sample (score > 0.70)
        collector.collect_sample(
            text="Hello there!",
            features={
                "bbox_area": 0.05,
                "word_count": 0.75,
                "text_density": 0.8,
            },
            score=0.85,
            bbox=bbox,
            panel_id=5,
            metadata={"image_width": 800, "image_height": 1200}
        )
        
        assert len(collector.samples) == 1
        sample = collector.samples[0]
        
        assert sample["text"] == "Hello there!"
        assert sample["score"] == 0.85
        assert sample["auto_label"] == "dialogue"
        assert sample["label"] == "dialogue"
        assert sample["needs_review"] is False
        assert sample["panel_id"] == 5
        assert "feature_bbox_area" in sample
    
    def test_collect_sample_background(self, tmp_path):
        """Test collecting high-confidence background sample."""
        collector = MLDataCollector(output_dir=str(tmp_path))
        
        bbox = Mock()
        bbox.x = 50
        bbox.y = 100
        bbox.width = 150
        bbox.height = 30
        
        # Collect background sample (score < 0.45)
        collector.collect_sample(
            text="BOOM",
            features={"bbox_area": 0.02, "word_count": 0.0},
            score=0.30,
            bbox=bbox,
            panel_id=3
        )
        
        assert len(collector.samples) == 1
        sample = collector.samples[0]
        
        assert sample["auto_label"] == "background"
        assert sample["needs_review"] is False
    
    def test_collect_sample_uncertain(self, tmp_path):
        """Test collecting uncertain sample that needs review."""
        collector = MLDataCollector(output_dir=str(tmp_path))
        
        bbox = Mock()
        bbox.x = 0
        bbox.y = 0
        bbox.width = 200
        bbox.height = 40
        
        # Collect uncertain sample (0.45 <= score <= 0.70)
        collector.collect_sample(
            text="Wait...",
            features={"bbox_area": 0.03},
            score=0.58,
            bbox=bbox,
            panel_id=10
        )
        
        sample = collector.samples[0]
        
        assert sample["auto_label"] is None
        assert sample["needs_review"] is True
        assert sample["label"] is None  # Needs manual review
    
    def test_save_csv(self, tmp_path):
        """Test saving collected data to CSV."""
        collector = MLDataCollector(output_dir=str(tmp_path))
        
        # Collect multiple samples
        bbox = Mock()
        bbox.x = bbox.y = bbox.width = bbox.height = 100
        
        for i in range(5):
            collector.collect_sample(
                text=f"Sample {i}",
                features={"feature1": i * 0.1},
                score=0.5 + i * 0.1,
                bbox=bbox,
                panel_id=i
            )
        
        # Save
        csv_path = collector.save(filename="test_samples.csv")
        
        assert csv_path is not None
        assert csv_path.exists()
        assert csv_path.name == "test_samples.csv"
        
        # Verify CSV content
        import pandas as pd
        df = pd.read_csv(csv_path)
        
        assert len(df) == 5
        assert "text" in df.columns
        assert "score" in df.columns
        assert "auto_label" in df.columns
        assert "needs_review" in df.columns
    
    def test_get_stats(self, tmp_path):
        """Test statistics computation."""
        collector = MLDataCollector(output_dir=str(tmp_path))
        
        # Empty stats
        stats = collector.get_stats()
        assert stats["total"] == 0
        
        # Add samples
        bbox = Mock()
        bbox.x = bbox.y = bbox.width = bbox.height = 100
        
        # 2 dialogue, 1 background, 1 uncertain
        collector.collect_sample("Dialogue 1", {}, 0.85, bbox, 1)
        collector.collect_sample("Dialogue 2", {}, 0.75, bbox, 2)
        collector.collect_sample("Background", {}, 0.30, bbox, 3)
        collector.collect_sample("Uncertain", {}, 0.55, bbox, 4)
        
        stats = collector.get_stats()
        
        assert stats["total"] == 4
        assert stats["dialogue"] == 2
        assert stats["background"] == 1
        assert stats["needs_review"] == 1
        assert "avg_score" in stats
        assert "score_std" in stats


class TestTextBoxClassifierMLIntegration:
    """Test TextBoxClassifier ML data collection integration."""
    
    def test_enable_ml_collection(self, tmp_path):
        """Test enabling ML data collection."""
        classifier = TextBoxClassifier()
        
        assert classifier.collect_ml_data is False
        assert classifier.ml_data_collector is None
        
        classifier.enable_ml_collection(output_dir=str(tmp_path))
        
        assert classifier.collect_ml_data is True
        assert classifier.ml_data_collector is not None
        assert isinstance(classifier.ml_data_collector, MLDataCollector)
    
    def test_disable_ml_collection(self, tmp_path):
        """Test disabling ML data collection."""
        classifier = TextBoxClassifier()
        classifier.enable_ml_collection(output_dir=str(tmp_path))
        
        # Add mock sample
        bbox = Mock()
        bbox.x = bbox.y = bbox.width = bbox.height = 100
        classifier.ml_data_collector.collect_sample("Test", {}, 0.75, bbox, 1)
        
        # Disable (should save data)
        classifier.disable_ml_collection()
        
        assert classifier.collect_ml_data is False
        
        # Check that data was saved
        csv_files = list(tmp_path.glob("collected_*.csv"))
        assert len(csv_files) == 1
    
    def test_classification_with_ml_collection(self, tmp_path):
        """Test that classification collects ML data when enabled."""
        from backend.services.ocr_models import OCRResult, BoundingBox
        
        classifier = TextBoxClassifier()
        classifier.enable_ml_collection(output_dir=str(tmp_path))
        
        # Mock OCR results
        ocr_results = [
            OCRResult(
                text="Hello!",
                bounding_box=BoundingBox(x=100, y=200, width=300, height=50),
                confidence=0.95
            ),
            OCRResult(
                text="BOOM",
                bounding_box=BoundingBox(x=50, y=100, width=100, height=30),
                confidence=0.90
            ),
        ]
        
        # Classify
        results = classifier.classify_all(ocr_results, 800, 1200)
        
        # Check that samples were collected
        assert len(classifier.ml_data_collector.samples) == 2
        
        sample1 = classifier.ml_data_collector.samples[0]
        assert sample1["text"] == "Hello!"
        assert "score" in sample1
        assert "feature_bbox_area" in sample1
        
        sample2 = classifier.ml_data_collector.samples[1]
        assert sample2["text"] == "BOOM"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
