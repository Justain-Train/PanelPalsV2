#!/usr/bin/env python3
"""
Start ML Data Collection Pipeline

This script processes a webtoon chapter to collect ML training samples.
ML data collection is always enabled in the server.

Usage:
    # Start the server first:
    uvicorn backend.main:app --reload
    
    # Then in another terminal, run:
    python start_ml_collection.py
"""

import logging
import sys
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def check_server_running():
    """Check if FastAPI server is running."""
    import requests
    try:
        response = requests.get("http://localhost:8000/health", timeout=2)
        return response.status_code == 200
    except:
        return False


def process_test_chapter():
    """Process test chapter with ML collection enabled."""
    import requests
    
    # Configuration
    API_URL = "http://localhost:8000/process/chapter"
    CHAPTER_ID = "ml_training_chapter_01"
    
    # Find available test images
    screenshots_dir = Path("screenshots/chapter_01")
    if not screenshots_dir.exists():
        # Try webtoon screenshots in root
        test_images = list(Path(".").glob("webtoonScreenshot*.png"))
        if not test_images:
            logger.error("❌ No test images found!")
            logger.info("   Expected: screenshots/chapter_01/panel_*.png")
            logger.info("   Or: webtoonScreenshot*.png in current directory")
            return False
        
        image_files = [str(img) for img in test_images]
        logger.info(f"📸 Using {len(image_files)} test images from current directory")
    else:
        image_files = [str(f) for f in sorted(screenshots_dir.glob("panel_*.png"))[:10]]  # First 10 panels
        logger.info(f"📸 Using first 10 panels from screenshots/chapter_01/")
    
    if not image_files:
        logger.error("❌ No images found to process")
        return False
    
    logger.info(f"📦 Chapter ID: {CHAPTER_ID}")
    logger.info(f"🔗 Endpoint: {API_URL}")
    logger.info(f"🎓 ML Collection: Always enabled in server")
    logger.info("-" * 60)
    
    # Prepare multipart form data
    files = []
    for img_path in image_files:
        try:
            with open(img_path, 'rb') as f:
                img_bytes = f.read()
                files.append(('images', (img_path, img_bytes, 'image/png')))
            logger.info(f"✅ Loaded: {img_path} ({len(img_bytes):,} bytes)")
        except FileNotFoundError:
            logger.error(f"❌ File not found: {img_path}")
            return False
    
    logger.info("-" * 60)
    logger.info("⏳ Sending request to pipeline (this will collect ML data)...")
    
    # Prepare form data
    data = {
        "chapter_id": CHAPTER_ID
    }
    
    # Send request
    try:
        response = requests.post(
            API_URL,
            data=data,
            files=files,
            timeout=300
        )
        
        logger.info(f"📡 Response Status: {response.status_code}")
        logger.info("-" * 60)
        
        if response.status_code == 200:
            # Save the MP3 file
            output_path = f"{CHAPTER_ID}.mp3"
            with open(output_path, 'wb') as f:
                f.write(response.content)
            
            logger.info("✅ SUCCESS! Audio file generated and ML data collected!")
            logger.info(f"💾 Audio saved to: {output_path}")
            logger.info(f"📊 File size: {len(response.content):,} bytes")
            
            # Check for collected data
            ml_data_dir = Path("ml_data/raw")
            if ml_data_dir.exists():
                csv_files = list(ml_data_dir.glob("collected_*.csv"))
                if csv_files:
                    latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
                    logger.info(f"📊 ML data saved to: {latest_csv}")
                    
                    # Show quick stats
                    import pandas as pd
                    df = pd.read_csv(latest_csv)
                    dialogue_count = (df["auto_label"] == "dialogue").sum()
                    background_count = (df["auto_label"] == "background").sum()
                    review_count = df["needs_review"].sum()
                    
                    logger.info(f"\n📈 Collection Statistics:")
                    logger.info(f"   Total samples: {len(df)}")
                    logger.info(f"   Auto-labeled dialogue: {dialogue_count}")
                    logger.info(f"   Auto-labeled background: {background_count}")
                    logger.info(f"   Needs review: {review_count} ({review_count/len(df)*100:.1f}%)")
                else:
                    logger.warning("⚠️  No CSV files found in ml_data/raw/")
            else:
                logger.warning("⚠️  ml_data/raw/ directory not created - check server logs")
            
            return True
        else:
            logger.error(f"❌ ERROR: {response.status_code}")
            logger.error(f"Response: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return False


def show_next_steps(csv_path: Path):
    """Show next steps after data collection."""
    print("\n" + "=" * 80)
    print("🎉 ML DATA COLLECTION COMPLETE!")
    print("=" * 80)
    print("\nNext steps:")
    print("\n1️⃣  Review uncertain samples (fast keyboard interface):")
    print(f"   python -m backend.ml.review_tool {csv_path}")
    print("\n2️⃣  Prepare dataset for training:")
    print(f"   python -m backend.ml.prepare_dataset {csv_path}")
    print("\n3️⃣  Train ML models:")
    print("   python -m backend.ml.train_model ml_data/prepared")
    print("\n📚 For more info, see: ML_QUICK_START.md")
    print("=" * 80)


def main():
    """Main entry point."""
    print("\n" + "=" * 80)
    print("🎓 ML DATA COLLECTION PIPELINE")
    print("=" * 80)
    print()
    
    # Check if server is running
    logger.info("🔍 Checking if FastAPI server is running...")
    if not check_server_running():
        logger.error("❌ FastAPI server is not running!")
        logger.info("\n📝 Start the server first:")
        logger.info("   uvicorn backend.main:app --reload")
        logger.info("\nOr in Docker:")
        logger.info("   docker-compose up")
        return 1
    
    logger.info("✅ Server is running")
    logger.info("")
    
    # Process test chapter
    logger.info("🔄 Processing test chapter with ML data collection...")
    success = process_test_chapter()
    
    if success:
        # Find latest CSV
        ml_data_dir = Path("ml_data/raw")
        if ml_data_dir.exists():
            csv_files = list(ml_data_dir.glob("collected_*.csv"))
            if csv_files:
                latest_csv = max(csv_files, key=lambda p: p.stat().st_mtime)
                show_next_steps(latest_csv)
            else:
                logger.warning("\n⚠️  No CSV file was created. Check server logs for errors.")
        
        return 0
    else:
        logger.error("\n❌ Failed to process chapter")
        return 1


if __name__ == "__main__":
    sys.exit(main())

