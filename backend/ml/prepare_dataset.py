"""
ML Dataset Preparation

Section 14.1: Unit Tests
Prepares labeled data for machine learning model training.

Architecture:
- Loads reviewed/labeled samples from CSV
- Splits into train/test sets
- Extracts features and labels
- Validates data quality
- Exports ready-to-train datasets

Usage:
    python -m backend.ml.prepare_dataset ml_data/raw/collected_TIMESTAMP.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Tuple

import pandas as pd
from sklearn.model_selection import train_test_split

logger = logging.getLogger(__name__)


class DatasetPreparator:
    """
    Prepares labeled classification data for ML training.
    
    Steps:
    1. Load labeled CSV
    2. Filter out unreviewed samples
    3. Extract feature columns and labels
    4. Split train/test (80/20)
    5. Validate data quality
    6. Export to separate CSVs
    """
    
    def __init__(self, csv_path: Path, output_dir: Path):
        """
        Initialize dataset preparator.
        
        Args:
            csv_path: Path to labeled samples CSV
            output_dir: Directory to save train/test datasets
        """
        self.csv_path = csv_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"📊 DatasetPreparator initialized: {csv_path} → {output_dir}")
    
    def prepare(self, test_size: float = 0.2, random_state: int = 42) -> dict:
        """
        Prepare train/test datasets.
        
        Args:
            test_size: Proportion of data for testing (default: 0.2)
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary with dataset statistics
        """
        # Load data
        logger.info(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        # Filter labeled samples only
        labeled = df[df["label"].notna()].copy()
        
        if len(labeled) == 0:
            raise ValueError("No labeled samples found in CSV")
        
        logger.info(f"Loaded {len(labeled)} labeled samples (from {len(df)} total)")
        
        # Validate labels
        valid_labels = {"dialogue", "background"}
        invalid = labeled[~labeled["label"].isin(valid_labels)]
        if len(invalid) > 0:
            logger.warning(f"Found {len(invalid)} samples with invalid labels, removing")
            labeled = labeled[labeled["label"].isin(valid_labels)]
        
        # Extract features (all columns starting with 'feature_')
        feature_cols = sorted([c for c in labeled.columns if c.startswith("feature_")])
        
        if len(feature_cols) == 0:
            raise ValueError("No feature columns found (expected 'feature_*' columns)")
        
        X = labeled[feature_cols].copy()
        
        # Convert labels to binary (dialogue=1, background=0)
        y = (labeled["label"] == "dialogue").astype(int)
        
        # Check for missing values
        missing_features = X.isnull().sum().sum()
        missing_labels = y.isnull().sum()
        
        if missing_features > 0:
            logger.warning(f"Found {missing_features} missing feature values, filling with 0.0")
            X = X.fillna(0.0)
        
        if missing_labels > 0:
            raise ValueError(f"Found {missing_labels} missing labels")
        
        # Split train/test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        # Add labels back for export
        train_df = X_train.copy()
        train_df["label"] = y_train.values
        train_df["label_str"] = y_train.map({1: "dialogue", 0: "background"}).values
        
        test_df = X_test.copy()
        test_df["label"] = y_test.values
        test_df["label_str"] = y_test.map({1: "dialogue", 0: "background"}).values
        
        # Add text for reference (helpful for debugging)
        train_df["text"] = labeled.loc[X_train.index, "text"].values
        test_df["text"] = labeled.loc[X_test.index, "text"].values
        
        # Save datasets
        train_path = self.output_dir / "train.csv"
        test_path = self.output_dir / "test.csv"
        
        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)
        
        # Generate statistics
        stats = {
            "total_samples": len(labeled),
            "train_samples": len(train_df),
            "test_samples": len(test_df),
            "num_features": len(feature_cols),
            "feature_names": feature_cols,
            "train_dialogue_ratio": (y_train == 1).sum() / len(y_train),
            "test_dialogue_ratio": (y_test == 1).sum() / len(y_test),
            "train_path": str(train_path),
            "test_path": str(test_path),
        }
        
        # Log summary
        logger.info(
            f"✅ Dataset prepared:\n"
            f"   Train: {len(train_df)} samples ({stats['train_dialogue_ratio']:.1%} dialogue)\n"
            f"   Test:  {len(test_df)} samples ({stats['test_dialogue_ratio']:.1%} dialogue)\n"
            f"   Features: {len(feature_cols)}\n"
            f"   Saved to: {self.output_dir}"
        )
        
        return stats


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Prepare labeled data for ML model training"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to labeled samples CSV"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("ml_data/prepared"),
        help="Output directory for train/test datasets"
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Proportion of data for testing (default: 0.2)"
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed for reproducibility"
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
    if not args.csv_path.exists():
        print(f"❌ Error: CSV file not found: {args.csv_path}")
        return 1
    
    # Prepare dataset
    preparator = DatasetPreparator(args.csv_path, args.output_dir)
    stats = preparator.prepare(
        test_size=args.test_size,
        random_state=args.random_state
    )
    
    print(f"\n✅ Dataset preparation complete!")
    print(f"   Train: {stats['train_samples']} samples → {stats['train_path']}")
    print(f"   Test:  {stats['test_samples']} samples → {stats['test_path']}")
    
    return 0


if __name__ == "__main__":
    exit(main())
