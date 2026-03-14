"""
ML Model Training

Trains machine learning models for text box classification.

Architecture:
- Loads prepared train/test datasets
- Trains multiple model types (Random Forest, Gradient Boosting, etc.)
- Evaluates model performance
- Saves best model for deployment
- Generates feature importance analysis

Usage:
    python -m backend.ml.train_model ml_data/prepared
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


class ModelTrainer:
    """
    Trains and evaluates classification models.
    
    Models trained:
    - Random Forest
    - Gradient Boosting
    - Logistic Regression
    
    Selects best model based on F1 score.
    """
    
    def __init__(self, data_dir: Path, output_dir: Path):
        """
        Initialize model trainer.
        
        Args:
            data_dir: Directory containing train.csv and test.csv
            output_dir: Directory to save trained models
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"🎓 ModelTrainer initialized: {data_dir} → {output_dir}")
    
    def train_all(self) -> Dict[str, Any]:
        """
        Train all model types and select best.
        
        Returns:
            Dictionary with training results and best model info
        """
        # Load data
        logger.info("Loading train/test data")
        train_df = pd.read_csv(self.data_dir / "train.csv")
        test_df = pd.read_csv(self.data_dir / "test.csv")
        
        # Extract features and labels
        feature_cols = [c for c in train_df.columns if c.startswith("feature_")]
        
        X_train = train_df[feature_cols]
        y_train = train_df["label"]
        
        X_test = test_df[feature_cols]
        y_test = test_df["label"]
        
        logger.info(
            f"Loaded: {len(train_df)} train, {len(test_df)} test, {len(feature_cols)} features"
        )
        
        # Define models to train
        models = {
            "random_forest": RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                random_state=42,
                n_jobs=-1
            ),
            "gradient_boosting": GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            "logistic_regression": LogisticRegression(
                max_iter=1000,
                random_state=42
            ),
        }
        
        # Train and evaluate each model
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training {name}...")
            
            # Train
            model.fit(X_train, y_train)
            
            # Predict
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Evaluate
            train_metrics = self._compute_metrics(y_train, y_pred_train)
            test_metrics = self._compute_metrics(y_test, y_pred_test)
            
            # Store results
            results[name] = {
                "model": model,
                "train_metrics": train_metrics,
                "test_metrics": test_metrics,
                "feature_importance": self._get_feature_importance(model, feature_cols),
            }
            
            logger.info(
                f"  {name} → Train: {train_metrics['accuracy']:.3f}, "
                f"Test: {test_metrics['accuracy']:.3f}, "
                f"F1: {test_metrics['f1']:.3f}"
            )
        
        # Select best model (by test F1 score)
        best_name = max(results.keys(), key=lambda k: results[k]["test_metrics"]["f1"])
        best_result = results[best_name]
        
        logger.info(f"🏆 Best model: {best_name} (F1={best_result['test_metrics']['f1']:.3f})")
        
        # Save best model
        model_path = self.output_dir / "best_model.joblib"
        joblib.dump(best_result["model"], model_path)
        
        # Save metadata
        metadata = {
            "best_model": best_name,
            "train_metrics": best_result["train_metrics"],
            "test_metrics": best_result["test_metrics"],
            "feature_names": feature_cols,
            "feature_importance": best_result["feature_importance"],
            "model_path": str(model_path),
        }
        
        metadata_path = self.output_dir / "model_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        # Generate classification report
        y_pred_best = best_result["model"].predict(X_test)
        report = classification_report(y_test, y_pred_best, target_names=["background", "dialogue"])
        
        report_path = self.output_dir / "classification_report.txt"
        with open(report_path, "w") as f:
            f.write(report)
        
        logger.info(f"\n{report}")
        logger.info(f"✅ Model saved: {model_path}")
        logger.info(f"   Metadata: {metadata_path}")
        logger.info(f"   Report: {report_path}")
        
        return metadata
    
    def _compute_metrics(self, y_true, y_pred) -> Dict[str, float]:
        """Compute classification metrics."""
        return {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
        }
    
    def _get_feature_importance(self, model, feature_names) -> Dict[str, float]:
        """Extract feature importance if available."""
        if hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
            return dict(zip(feature_names, importance.tolist()))
        elif hasattr(model, "coef_"):
            importance = abs(model.coef_[0])
            return dict(zip(feature_names, importance.tolist()))
        else:
            return {}


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Train ML models for text box classification"
    )
    parser.add_argument(
        "data_dir",
        type=Path,
        help="Directory containing train.csv and test.csv"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("models"),
        help="Output directory for trained models"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Validate input
    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    
    if not train_path.exists():
        print(f"❌ Error: Train file not found: {train_path}")
        return 1
    
    if not test_path.exists():
        print(f"❌ Error: Test file not found: {test_path}")
        return 1
    
    # Train models
    trainer = ModelTrainer(args.data_dir, args.output_dir)
    metadata = trainer.train_all()
    
    print(f"\n✅ Training complete!")
    print(f"   Best model: {metadata['best_model']}")
    print(f"   Test accuracy: {metadata['test_metrics']['accuracy']:.3f}")
    print(f"   Test F1: {metadata['test_metrics']['f1']:.3f}")
    print(f"   Saved to: {metadata['model_path']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
