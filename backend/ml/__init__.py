"""
ML Training Pipeline

Section 14.1: Unit Tests
Machine learning components for text box classification.

Modules:
- data_collector: Collects training data during classification
- review_tool: Interactive manual review of uncertain samples
- prepare_dataset: Prepares labeled data for training
- train_model: Trains and evaluates ML models

Workflow:
1. Enable data collection: classifier.enable_ml_collection()
2. Run classification on chapters to collect samples
3. Review uncertain samples: python -m backend.ml.review_tool <csv>
4. Prepare dataset: python -m backend.ml.prepare_dataset <csv>
5. Train model: python -m backend.ml.train_model <data_dir>
6. Deploy model in production
"""

from .data_collector import MLDataCollector

__all__ = ["MLDataCollector"]
