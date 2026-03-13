#!/usr/bin/env python3
"""
ML Pipeline Workflow Example

Demonstrates the complete ML training pipeline from data collection to model deployment.

Usage:
    python backend/ml/example_workflow.py
"""

import logging
from pathlib import Path

from backend.services.text_box_classifier import TextBoxClassifier
from backend.ml.data_collector import MLDataCollector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_data_collection():
    """
    Example 1: Collect training data during classification.
    
    Run this while processing webtoon chapters to collect samples.
    """
    logger.info("="*80)
    logger.info("EXAMPLE 1: Data Collection")
    logger.info("="*80)
    
    # Initialize classifier
    classifier = TextBoxClassifier()
    
    # Enable ML data collection
    classifier.enable_ml_collection(output_dir="ml_data/raw")
    
    logger.info("""
    Now run your normal chapter processing:
    
    from backend.services.vision_service import VisionService
    
    # Process chapter images
    vision = VisionService()
    ocr_results = vision.extract_text_batch(images)
    
    # Classify with data collection enabled
    classification_results = classifier.classify_all(
        ocr_results, image_width, image_height
    )
    
    # Save collected data
    classifier.disable_ml_collection()  # Saves to ml_data/raw/collected_TIMESTAMP.csv
    """)
    
    # Example stats
    if classifier.ml_data_collector:
        stats = classifier.ml_data_collector.get_stats()
        logger.info(f"Collected samples: {stats}")


def example_review_workflow():
    """
    Example 2: Review uncertain samples.
    
    Run this after collecting data to manually review uncertain cases.
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 2: Manual Review")
    logger.info("="*80)
    
    logger.info("""
    Step 1: Find your collected CSV file:
        ls ml_data/raw/
    
    Step 2: Run the review tool:
        python -m backend.ml.review_tool ml_data/raw/collected_20240101_120000.csv
    
    Step 3: Review each uncertain sample:
        [1/25 done] Label? [d/b/s/q]: d  # Mark as dialogue
        [2/25 done] Label? [d/b/s/q]: b  # Mark as background
        [3/25 done] Label? [d/b/s/q]: s  # Skip this one
        [4/25 done] Label? [d/b/s/q]: q  # Quit and save
    
    Controls:
        d = dialogue
        b = background
        s = skip
        q = quit and save
    
    Progress is saved automatically, so you can quit and resume anytime.
    """)


def example_dataset_preparation():
    """
    Example 3: Prepare dataset for training.
    
    Run this after reviewing samples to create train/test splits.
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 3: Dataset Preparation")
    logger.info("="*80)
    
    logger.info("""
    Step 1: Prepare train/test datasets:
        python -m backend.ml.prepare_dataset \\
            ml_data/raw/collected_20240101_120000.csv \\
            --output-dir ml_data/prepared \\
            --test-size 0.2
    
    This will create:
        - ml_data/prepared/train.csv (80% of data)
        - ml_data/prepared/test.csv (20% of data)
    
    Both files contain:
        - 10 feature columns (feature_*)
        - Binary label (0=background, 1=dialogue)
        - Text content (for reference)
    """)


def example_model_training():
    """
    Example 4: Train ML models.
    
    Run this after preparing the dataset to train and evaluate models.
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 4: Model Training")
    logger.info("="*80)
    
    logger.info("""
    Step 1: Train models:
        python -m backend.ml.train_model ml_data/prepared \\
            --output-dir models
    
    This will:
        1. Train 3 models: Random Forest, Gradient Boosting, Logistic Regression
        2. Evaluate each on train/test sets
        3. Select the best model (by F1 score)
        4. Save to models/best_model.joblib
    
    Output files:
        - models/best_model.joblib          # Trained model (ready for deployment)
        - models/model_metadata.json        # Metrics and feature importance
        - models/classification_report.txt  # Detailed evaluation
    
    Example output:
        Training random_forest...
          random_forest → Train: 0.953, Test: 0.912, F1: 0.905
        Training gradient_boosting...
          gradient_boosting → Train: 0.967, Test: 0.923, F1: 0.918
        Training logistic_regression...
          logistic_regression → Train: 0.885, Test: 0.878, F1: 0.871
        🏆 Best model: gradient_boosting (F1=0.918)
        ✅ Model saved: models/best_model.joblib
    """)


def example_model_deployment():
    """
    Example 5: Deploy trained model.
    
    Integrate the ML model into production classification.
    """
    logger.info("\n" + "="*80)
    logger.info("EXAMPLE 5: Model Deployment")
    logger.info("="*80)
    
    logger.info("""
    Option 1: Load and use the model directly
    
    import joblib
    import pandas as pd
    
    # Load trained model
    model = joblib.load("models/best_model.joblib")
    
    # Compute features (same as heuristic classifier)
    features = classifier._compute_features(ocr_result, image_area, all_results)
    feature_vector = pd.DataFrame([features])
    
    # Predict
    is_dialogue = model.predict(feature_vector)[0]  # 1=dialogue, 0=background
    confidence = model.predict_proba(feature_vector)[0][1]  # Probability
    
    
    Option 2: Hybrid approach (use ML for uncertain cases)
    
    # Use heuristic for high-confidence cases
    heuristic_score = classifier._compute_score(features)
    
    if heuristic_score > 0.80 or heuristic_score < 0.40:
        # High confidence → use heuristic
        is_dialogue = heuristic_score > classifier.threshold
    else:
        # Uncertain → use ML model
        is_dialogue = model.predict(feature_vector)[0]
    
    
    Option 3: Replace heuristic entirely
    
    # Modify TextBoxClassifier to use ML model
    class MLTextBoxClassifier(TextBoxClassifier):
        def __init__(self, model_path, **kwargs):
            super().__init__(**kwargs)
            self.model = joblib.load(model_path)
        
        def classify_all(self, ocr_results, image_width, image_height):
            # Compute features
            features_list = []
            for ocr in ocr_results:
                features = self._compute_features(ocr, ...)
                features_list.append(features)
            
            # Predict with ML model
            X = pd.DataFrame(features_list)
            predictions = self.model.predict(X)
            
            # Return results
            results = []
            for ocr, is_dialogue, features in zip(ocr_results, predictions, features_list):
                results.append(ClassificationResult(
                    ocr_result=ocr,
                    is_text_box=bool(is_dialogue),
                    score=self.model.predict_proba([features])[0][1],
                    features=features
                ))
            
            return results
    """)


def main():
    """Run all workflow examples."""
    logger.info("\n" + "="*80)
    logger.info("ML TRAINING PIPELINE - COMPLETE WORKFLOW")
    logger.info("="*80)
    
    example_data_collection()
    example_review_workflow()
    example_dataset_preparation()
    example_model_training()
    example_model_deployment()
    
    logger.info("\n" + "="*80)
    logger.info("SUMMARY")
    logger.info("="*80)
    logger.info("""
    Complete workflow:
    
    1. Enable data collection → classifier.enable_ml_collection()
    2. Process chapters → collect samples automatically
    3. Review uncertain cases → python -m backend.ml.review_tool <csv>
    4. Prepare dataset → python -m backend.ml.prepare_dataset <csv>
    5. Train models → python -m backend.ml.train_model <data_dir>
    6. Deploy model → Load and use best_model.joblib
    
    For more details, see: backend/ml/README.md
    """)


if __name__ == "__main__":
    main()
