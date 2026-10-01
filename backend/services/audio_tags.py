"""
Audio Tag Service

Turns the sound words webtoons draw for emotions (SOB, GIGGLE, SIGH, GASP,
WHISPER...) into ElevenLabs v3/v4 audio tags like [crying] or [laughs], so the
narrator performs the emotion instead of reading the word aloud.

Cues come from two places:
- Inside a dialogue bubble, at its very start or end, where OCR merges a
  nearby sound effect into the speech ("WHISPER YOU'RE TOO LOUD").
  The same word in the middle of a sentence is left alone.
- A neighbouring bubble the classifier filtered out as background, attached to
  the nearest dialogue bubble in the same panel.

Tags are applied after TTS preprocessing, which would otherwise strip the brackets.
"""

import logging
import math
import re
from typing import List, Optional, Sequence, Tuple

logger = logging.getLogger(__name__)

# Sound word (matched against the whole word, letters only, uppercase) → audio tag
CUE_TAGS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"(?:BW)?A?(?:HA){2,}H?|LAUGHS?|LAUGHING"), "[laughs]"),
    (re.compile(r"HEH|HE(?:HE)+H?|GIGGLES?|CHUCKLES?|SNICKERS?"), "[chuckles]"),
    (re.compile(r"SOBS?|SNIFF(?:LE)?S?|WEEPS?|HIC"), "[crying]"),
    (re.compile(r"SIGHS?|HAA+"), "[sighs]"),
    (re.compile(r"GASPS?|STARTLED"), "[gasps]"),
    (re.compile(r"WHISPERS?|MURMURS?|MUTTERS?|MUMBLES?"), "[whispers]"),
    (re.compile(r"SHOUTS?|YELLS?"), "[shouting]"),
    (re.compile(r"GROANS?"), "[groans]"),
    (re.compile(r"SHIVERS?|TREMBLES?"), "[shivering]"),
    (re.compile(r"GULPS?"), "[gulps]"),
    (re.compile(r"SMIRKS?"), "[mischievously]"),
]

MAX_TAGS_PER_BUBBLE = 2

# Max distance (pixels, between box centres) for a standalone sound word to
# attach to a dialogue bubble in the same panel
MAX_NEIGHBOUR_DISTANCE = 400


def _letters(token: str) -> str:
    return re.sub(r"[^A-Za-z]", "", token).upper()


def cue_tag(token: str) -> Optional[str]:
    """Audio tag for a single sound word, or None. 'HAHAHA!!' → '[laughs]'."""
    word = _letters(token)
    if not word:
        return None
    for pattern, tag in CUE_TAGS:
        if pattern.fullmatch(word):
            return tag
    return None


def _is_punctuation(token: str) -> bool:
    return not _letters(token)


def merge_tags(existing: Sequence[str], new: Sequence[str]) -> List[str]:
    """Combine tag lists, keeping order, dropping repeats, capping the count."""
    merged: List[str] = []
    for tag in list(existing) + list(new):
        if tag not in merged:
            merged.append(tag)
    return merged[:MAX_TAGS_PER_BUBBLE]


def extract_inline_cues(text: str) -> Tuple[str, List[str]]:
    """
    Remove sound words from the start/end of a bubble and return their tags.

    "HAHAHA !! THAT'S HILARIOUS !!" → ("THAT'S HILARIOUS !!", ["[laughs]"])
    "DON'T SHOUT AT ME !"          → unchanged (cue word is mid-sentence)
    "SOB"                          → ("", ["[crying]"])
    """
    tokens = text.split()
    tags: List[str] = []

    # Leading cues, plus punctuation that belonged to them ("HAHA !!")
    start = 0
    while start < len(tokens):
        tag = cue_tag(tokens[start])
        if tag:
            tags.append(tag)
            start += 1
        elif tags and _is_punctuation(tokens[start]):
            start += 1
        else:
            break

    # Trailing cues
    end = len(tokens)
    trailing: List[str] = []
    while end > start:
        tag = cue_tag(tokens[end - 1])
        if not tag:
            break
        trailing.insert(0, tag)
        end -= 1

    clean = " ".join(tokens[start:end])
    return clean, merge_tags([], tags + trailing)


def bubble_cue_tags(text: str) -> List[str]:
    """
    Tags for a bubble made only of sound words (and punctuation), e.g. 'SOB',
    'MUTTER MUMBLE'. Returns [] if any real word is present.
    """
    tokens = [t for t in text.split() if not _is_punctuation(t)]
    if not tokens:
        return []
    tags = [cue_tag(t) for t in tokens]
    if not all(tags):
        return []
    return merge_tags([], tags)


def _distance(a, b) -> float:
    return math.hypot(a.center_x - b.center_x, a.center_y - b.center_y)


def tag_panel(kept: List, removed: List, max_distance: float = MAX_NEIGHBOUR_DISTANCE) -> int:
    """
    Add audio tags to a panel's dialogue bubbles (mutates them in place).

    Args:
        kept: Bubbles classified as dialogue (TextBubble-like: text,
            bounding_box, audio_tags)
        removed: Bubbles classified as background in the same panel
        max_distance: Max centre distance for a background sound word to
            attach to a dialogue bubble

    Returns:
        Number of tags added
    """
    added = 0

    for bubble in kept:
        clean, tags = extract_inline_cues(bubble.text)
        if tags:
            logger.info(f"🎭 Inline cue {tags} in '{bubble.text[:40]}' → '{clean[:40]}'")
            bubble.text = clean
            before = len(bubble.audio_tags)
            bubble.audio_tags = merge_tags(bubble.audio_tags, tags)
            added += len(bubble.audio_tags) - before

    for sfx in removed:
        tags = bubble_cue_tags(sfx.text)
        if not tags or not kept:
            continue
        nearest = min(kept, key=lambda b: _distance(b.bounding_box, sfx.bounding_box))
        distance = _distance(nearest.bounding_box, sfx.bounding_box)
        if distance > max_distance:
            logger.info(f"🎭 '{sfx.text}' {tags} too far from dialogue ({distance:.0f}px), skipped")
            continue
        before = len(nearest.audio_tags)
        nearest.audio_tags = merge_tags(nearest.audio_tags, tags)
        if len(nearest.audio_tags) > before:
            logger.info(f"🎭 '{sfx.text}' → {tags} on '{nearest.text[:40]}' ({distance:.0f}px)")
            added += len(nearest.audio_tags) - before

    return added


def apply_tags(text: str, tags: Sequence[str]) -> str:
    """Prefix preprocessed TTS text with its audio tags: '[laughs] That was great!'."""
    tags = list(tags)[:MAX_TAGS_PER_BUBBLE]
    if not tags:
        return text
    prefix = " ".join(tags)
    return f"{prefix} {text}" if text else prefix
