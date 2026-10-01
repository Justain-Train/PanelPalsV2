#!/usr/bin/env python3
"""
Download the panel images of a WEBTOON episode for local pipeline testing.

Panels live in <div id="_imageList"> as <img class="_images">. The `src`
attribute is a lazy-load placeholder; the real image URL is in `data-url`.
The image CDN rejects requests without a webtoons.com Referer.

Usage:
    python backend/utilities/download_episode.py "<episode viewer url>" [--out screenshots/<name>]

Images are for personal testing only - don't redistribute them.
"""

import argparse
import re
import sys
import time
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urlparse

import requests

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    ),
    "Referer": "https://www.webtoons.com/",
}


class ImageListParser(HTMLParser):
    """Collects data-url of every <img> inside #_imageList."""

    def __init__(self):
        super().__init__()
        self.depth = 0  # >0 while inside #_imageList
        self.urls = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if self.depth:
            if tag == "img":
                url = attrs.get("data-url") or attrs.get("src")
                if url and "bg_transparency" not in url:
                    self.urls.append(url)
            elif tag == "div":
                self.depth += 1
        elif tag == "div" and attrs.get("id") == "_imageList":
            self.depth = 1

    def handle_endtag(self, tag):
        if self.depth and tag == "div":
            self.depth -= 1


def default_out_dir(url: str) -> Path:
    # /en/fantasy/<series>/episode-1/viewer -> screenshots/<series>_ep1
    parts = urlparse(url).path.strip("/").split("/")
    series = parts[2] if len(parts) > 2 else "episode"
    ep = re.search(r"episode_no=(\d+)", url)
    return Path("screenshots") / f"{series}_ep{ep.group(1) if ep else ''}"


def main():
    parser = argparse.ArgumentParser(description="Download WEBTOON episode panels")
    parser.add_argument("url", help="Episode viewer URL")
    parser.add_argument("--out", type=Path, help="Output directory")
    args = parser.parse_args()

    out_dir = args.out or default_out_dir(args.url)
    out_dir.mkdir(parents=True, exist_ok=True)

    session = requests.Session()
    session.headers.update(HEADERS)

    page = session.get(args.url, timeout=30)
    page.raise_for_status()

    image_list = ImageListParser()
    image_list.feed(page.text)
    if not image_list.urls:
        print("❌ No images found in #_imageList")
        return 1

    print(f"📸 Found {len(image_list.urls)} panels → {out_dir}")

    for n, img_url in enumerate(image_list.urls, 1):
        ext = Path(urlparse(img_url).path).suffix or ".jpg"
        dest = out_dir / f"panel_{n}{ext}"
        if dest.exists():
            continue
        resp = session.get(img_url, timeout=30)
        resp.raise_for_status()
        dest.write_bytes(resp.content)
        print(f"  ✅ {dest.name} ({len(resp.content):,} bytes)")
        time.sleep(0.2)  # be polite to the CDN

    print("✅ Done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
