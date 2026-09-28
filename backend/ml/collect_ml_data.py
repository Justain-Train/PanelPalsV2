#!/usr/bin/env python3
"""
Collect ML training data without generating audio.

Runs the same OCR → bubble grouping → classification steps as
POST /process/chapter, then saves the classifier's ML samples. Stops before
TTS, so no ElevenLabs credits are used (Google Vision OCR is still called).
Doesn't need the server running.

Usage:
    python -m backend.ml.collect_ml_data screenshots/<episode_dir> [more dirs ...] [--out-dir ml_data/raw]

Each directory is processed as one chapter and saved to
<out-dir>/collected_<dir name>.csv
"""

import argparse
import io
import logging
import re
import sys
from pathlib import Path

from PIL import Image

from backend.services import GoogleVisionOCRService, TextBubbleGrouper, TextBoxClassifier

logger = logging.getLogger("collect_ml_data")

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".webp"}


def panel_number(path: Path) -> int:
    match = re.search(r"(\d+)", path.stem)
    return int(match.group(1)) if match else 0


def collect_episode(
    episode_dir: Path,
    ocr: GoogleVisionOCRService,
    grouper: TextBubbleGrouper,
    out_dir: Path,
) -> Path:
    panels = sorted(
        (p for p in episode_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS),
        key=panel_number,
    )
    if not panels:
        raise ValueError(f"No images found in {episode_dir}")

    logger.info(f"📂 {episode_dir.name}: {len(panels)} panels")
    image_bytes_list = [p.read_bytes() for p in panels]

    # Fresh classifier per episode so each gets its own sample list / CSV
    classifier = TextBoxClassifier()
    out_dir.mkdir(parents=True, exist_ok=True)
    classifier.ml_data_collector.output_dir = out_dir

    ocr_results = ocr.detect_text_batch(image_bytes_list)

    for panel_idx, panel_ocr in enumerate(ocr_results):
        if not panel_ocr:
            continue
        bubbles = grouper.group_into_bubbles(panel_ocr, panel_id=panel_idx)
        if not bubbles:
            continue
        width, height = Image.open(io.BytesIO(image_bytes_list[panel_idx])).size
        classifier.filter_text_bubbles(bubbles, width, height)

    csv_path = classifier.ml_data_collector.save(filename=f"collected_{episode_dir.name}.csv")
    stats = classifier.ml_data_collector.get_stats()
    logger.info(
        f"💾 {csv_path} — Total: {stats['total']}, Dialogue: {stats['dialogue']}, "
        f"Background: {stats['background']}, Needs review: {stats['needs_review']}"
    )
    return csv_path


def main():
    parser = argparse.ArgumentParser(description="Collect ML data without TTS")
    parser.add_argument("dirs", nargs="+", type=Path, help="Episode directories of panel images")
    parser.add_argument("--out-dir", type=Path, default=Path("ml_data/raw"), help="Where to save CSVs")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    # Per-bubble classifier logs are very noisy; keep this script's summary readable
    logging.getLogger("backend").setLevel(logging.WARNING)

    ocr = GoogleVisionOCRService()
    grouper = TextBubbleGrouper()

    failed = []
    for episode_dir in args.dirs:
        try:
            collect_episode(episode_dir, ocr, grouper, args.out_dir)
        except Exception as e:
            logger.error(f"❌ {episode_dir}: {e}")
            failed.append(episode_dir)

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
