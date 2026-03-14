"""
Interactive ML Training Data Review Tool

Fast keyboard-driven interface for manually reviewing uncertain classification samples.

Architecture:
- Loads samples with needs_review=True from CSV
- Shows text, score, and context for each sample
- Keyboard controls: d=dialogue, b=background, s=skip, q=quit
- Tracks progress (X/Y completed)
- Saves reviewed labels back to CSV

Usage:
    python -m backend.ml.review_tool ml_data/raw/collected_TIMESTAMP.csv
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ReviewTool:
    """
    Interactive tool for reviewing uncertain classification samples.
    
    Keyboard shortcuts:
    - d: Label as dialogue
    - b: Label as background
    - s: Skip (no label change)
    - q: Quit and save progress
    """
    
    def __init__(self, csv_path: Path):
        """
        Initialize review tool.
        
        Args:
            csv_path: Path to collected samples CSV
        """
        self.csv_path = csv_path
        self.df = pd.read_csv(csv_path)
        
        # Filter uncertain samples
        self.uncertain = self.df[self.df["needs_review"] == True].copy()
        
        self.reviewed_count = 0
        self.session_labels = {}  # sample_id -> label
        
        logger.info(f"📋 Loaded {len(self.df)} samples from {csv_path}")
        logger.info(f"   {len(self.uncertain)} need review ({len(self.uncertain)/len(self.df)*100:.1f}%)")
    
    def review_all(self) -> int:
        """
        Review all uncertain samples interactively.
        
        Returns:
            Number of samples reviewed
        """
        if len(self.uncertain) == 0:
            print("\n✅ No samples need review!")
            return 0
        
        print(f"\n{'='*80}")
        print(f"📝 REVIEW TOOL - {len(self.uncertain)} samples need your review")
        print(f"{'='*80}")
        print("\nControls:")
        print("  [d] → dialogue")
        print("  [b] → background")
        print("  [s] → skip")
        print("  [q] → quit and save")
        print(f"{'='*80}\n")
        
        try:
            for idx, (sample_idx, row) in enumerate(self.uncertain.iterrows()):
                self._review_sample(idx + 1, sample_idx, row)
        
        except KeyboardInterrupt:
            print("\n\n⚠️  Review interrupted by user")
        
        # Save progress
        self._save_progress()
        
        return self.reviewed_count
    
    def _review_sample(self, display_idx: int, sample_idx: int, row: pd.Series) -> None:
        """
        Review a single sample.
        
        Args:
            display_idx: Display number (1-indexed)
            sample_idx: DataFrame index
            row: Sample data
        """
        # Display sample info
        print(f"\n{'─'*80}")
        print(f"Sample {display_idx}/{len(self.uncertain)}")
        print(f"{'─'*80}")
        print(f"📝 Text: \"{row['text']}\"")
        print(f"📊 Score: {row['score']:.3f} (threshold uncertain: 0.45-0.70)")
        print(f"📍 Panel: {row.get('panel_id', 'N/A')}")
        print(f"📏 Bbox: ({row.get('bbox_x', 0):.0f}, {row.get('bbox_y', 0):.0f}) "
              f"{row.get('bbox_width', 0):.0f}×{row.get('bbox_height', 0):.0f}px")
        
        # Show key features
        print(f"\n🔍 Features:")
        print(f"   Word count: {row.get('feature_word_count', 0):.2f}")
        print(f"   Dictionary: {row.get('feature_dictionary_ratio', 0):.2f}")
        print(f"   Alphabet: {row.get('feature_alphabet_ratio', 0):.2f}")
        print(f"   Punctuation: {row.get('feature_punctuation', 0):.2f}")
        
        # Get user input
        while True:
            print(f"\n[{self.reviewed_count}/{len(self.uncertain)} done]", end=" ")
            choice = input("Label? [d/b/s/q]: ").strip().lower()
            
            if choice == 'd':
                self._label_sample(sample_idx, "dialogue")
                print("✅ Labeled as DIALOGUE")
                break
            
            elif choice == 'b':
                self._label_sample(sample_idx, "background")
                print("✅ Labeled as BACKGROUND")
                break
            
            elif choice == 's':
                print("⏭️  Skipped")
                break
            
            elif choice == 'q':
                print("\n💾 Quitting and saving progress...")
                raise KeyboardInterrupt
            
            else:
                print("❌ Invalid choice. Use d/b/s/q")
    
    def _label_sample(self, sample_idx: int, label: str) -> None:
        """
        Apply label to sample.
        
        Args:
            sample_idx: DataFrame index
            label: "dialogue" or "background"
        """
        self.session_labels[sample_idx] = label
        self.reviewed_count += 1
    
    def _save_progress(self) -> None:
        """Save reviewed labels back to CSV."""
        if not self.session_labels:
            print("\n⚠️  No labels to save")
            return
        
        # Update labels in DataFrame
        for sample_idx, label in self.session_labels.items():
            self.df.at[sample_idx, "label"] = label
            self.df.at[sample_idx, "needs_review"] = False
        
        # Save to CSV
        self.df.to_csv(self.csv_path, index=False)
        
        # Print summary
        dialogue_count = sum(1 for l in self.session_labels.values() if l == "dialogue")
        background_count = len(self.session_labels) - dialogue_count
        
        remaining = (self.df["needs_review"] == True).sum()
        
        print(f"\n{'='*80}")
        print(f"💾 SAVED {len(self.session_labels)} labels to {self.csv_path}")
        print(f"{'='*80}")
        print(f"   Dialogue: {dialogue_count}")
        print(f"   Background: {background_count}")
        print(f"   Remaining to review: {remaining}")
        print(f"{'='*80}\n")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Review uncertain classification samples for ML training"
    )
    parser.add_argument(
        "csv_path",
        type=Path,
        help="Path to collected samples CSV"
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
    
    # Validate CSV path
    if not args.csv_path.exists():
        print(f"❌ Error: CSV file not found: {args.csv_path}")
        sys.exit(1)
    
    # Run review
    tool = ReviewTool(args.csv_path)
    reviewed = tool.review_all()
    
    print(f"\n✅ Review session complete: {reviewed} samples reviewed")


if __name__ == "__main__":
    main()
