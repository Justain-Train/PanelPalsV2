"""
Unit tests for audio tags (sound words → ElevenLabs tags) and the expressive
TTS preprocessing they're paired with.
"""

import pytest

from backend.services.audio_tags import (
    apply_tags,
    bubble_cue_tags,
    cue_tag,
    extract_inline_cues,
    merge_tags,
    tag_panel,
)
from backend.services.bubble_continuation import BubbleContinuationDetector
from backend.services.text_grouping import TextBubble
from backend.services.text_preprocessing import TextPreprocessor
from backend.services.vision import BoundingBox


def bubble(text, left=100, top=100, width=200, height=60, panel_id=0, tags=None):
    bbox = BoundingBox([
        {"x": left, "y": top}, {"x": left + width, "y": top},
        {"x": left + width, "y": top + height}, {"x": left, "y": top + height},
    ])
    return TextBubble(text=text, bounding_box=bbox, panel_id=panel_id, audio_tags=list(tags or []))


# cue_tag

@pytest.mark.unit
@pytest.mark.parametrize("word, tag", [
    ("HAHAHA", "[laughs]"), ("AHAHA", "[laughs]"), ("BWAHAHA", "[laughs]"), ("HAHAHA!!", "[laughs]"),
    ("HEH", "[chuckles]"), ("HEHEHE", "[chuckles]"), ("GIGGLE", "[chuckles]"), ("snicker", "[chuckles]"),
    ("SOB", "[crying]"), ("SNIFFLE", "[crying]"), ("HIC", "[crying]"),
    ("SIGH", "[sighs]"), ("HAA", "[sighs]"), ("HAAA...", "[sighs]"),
    ("GASP", "[gasps]"), ("WHISPER", "[whispers]"), ("MUTTER", "[whispers]"),
    ("SHOUT", "[shouting]"), ("GROAN", "[groans]"), ("TREMBLE", "[shivering]"),
    ("GULP", "[gulps]"), ("SMIRK", "[mischievously]"),
])
def test_cue_tag_maps_sound_words(word, tag):
    assert cue_tag(word) == tag


@pytest.mark.unit
@pytest.mark.parametrize("word", ["HA", "HELLO", "HEAT", "STEP", "SLAM", "GLANCE", "UGH", "?!", ""])
def test_cue_tag_ignores_other_words(word):
    """A single HA stays speech; action words without an emotion get no tag."""
    assert cue_tag(word) is None


# extract_inline_cues

@pytest.mark.unit
def test_leading_laughter_becomes_tag():
    assert extract_inline_cues("HAHAHA AHAHA !! THAT WAS GREAT !!") == ("THAT WAS GREAT !!", ["[laughs]"])


@pytest.mark.unit
def test_leading_and_trailing_cues():
    assert extract_inline_cues("WHISPER KEEP YOUR VOICE DOWN . SOB") == (
        "KEEP YOUR VOICE DOWN .", ["[whispers]", "[crying]"]
    )


@pytest.mark.unit
def test_cue_word_mid_sentence_is_kept():
    text = "PLEASE DON'T SHOUT AT ME !"
    assert extract_inline_cues(text) == (text, [])


@pytest.mark.unit
def test_bubble_of_only_cues_becomes_empty():
    assert extract_inline_cues("HAHAHA !!") == ("", ["[laughs]"])


@pytest.mark.unit
def test_inline_tags_capped_and_deduped():
    _, tags = extract_inline_cues("SOB SNIFFLE GASP WHY ME SIGH")
    assert tags == ["[crying]", "[gasps]"]


# bubble_cue_tags

@pytest.mark.unit
def test_bubble_cue_tags_only_for_pure_sound_word_bubbles():
    assert bubble_cue_tags("MUTTER MUMBLE") == ["[whispers]"]
    assert bubble_cue_tags("SOB !") == ["[crying]"]
    assert bubble_cue_tags("CLAP CLAP") == []
    assert bubble_cue_tags("SOB WHY") == []


# tag_panel

@pytest.mark.unit
def test_nearby_background_cue_attaches_to_nearest_dialogue():
    near = bubble("I CAN'T BELIEVE IT ...", left=100, top=100)
    far = bubble("WHERE ARE YOU GOING ?", left=100, top=500)
    sob = bubble("SOB", left=320, top=110, width=60, height=40)

    added = tag_panel([near, far], [sob])

    assert added == 1
    assert near.audio_tags == ["[crying]"]
    assert far.audio_tags == []


@pytest.mark.unit
def test_background_cue_too_far_is_ignored():
    line = bubble("OKAY .", left=0, top=0)
    sigh = bubble("SIGH", left=600, top=900, width=60, height=40)
    assert tag_panel([line], [sigh], max_distance=400) == 0
    assert line.audio_tags == []


@pytest.mark.unit
def test_non_emotion_background_is_ignored():
    line = bubble("OVER HERE !")
    assert tag_panel([line], [bubble("SLAM", left=120, top=120)]) == 0


@pytest.mark.unit
def test_inline_cue_rewrites_bubble_text():
    line = bubble("GIGGLE YOU'RE SO FUNNY")
    tag_panel([line], [])
    assert line.text == "YOU'RE SO FUNNY"
    assert line.audio_tags == ["[chuckles]"]


@pytest.mark.unit
def test_no_dialogue_in_panel_is_safe():
    assert tag_panel([], [bubble("SOB")]) == 0


# apply_tags / merge_tags

@pytest.mark.unit
def test_apply_tags():
    assert apply_tags("That was great!", ["[laughs]"]) == "[laughs] That was great!"
    assert apply_tags("", ["[laughs]"]) == "[laughs]"
    assert apply_tags("Hi.", []) == "Hi."
    assert apply_tags("Hi.", ["[a]", "[b]", "[c]"]) == "[a] [b] Hi."


@pytest.mark.unit
def test_merge_tags():
    assert merge_tags(["[sighs]"], ["[sighs]", "[crying]", "[gasps]"]) == ["[sighs]", "[crying]"]


# Tags survive page-break merges

@pytest.mark.unit
def test_continuation_merge_keeps_both_bubbles_tags():
    """A bubble split across a page break keeps the tags from both halves."""
    detector = BubbleContinuationDetector()
    top_half = bubble("I REALLY THOUGHT THAT", left=300, top=1150, height=80, panel_id=0, tags=["[sighs]"])
    bottom_half = bubble("YOU WOULD COME BACK .", left=300, top=10, height=80, panel_id=1, tags=["[crying]"])

    merged = detector.detect_and_merge_continuations([[top_half], [bottom_half]], [1280, 1280])

    assert len(merged) == 1, "fixture should be detected as a continuation"
    assert merged[0].audio_tags == ["[sighs]", "[crying]"]


# Expressive TTS preprocessing

@pytest.mark.unit
@pytest.mark.parametrize("raw, expected", [
    ("UGH ... I'M SO TIRED ...", "Ugh... I'm so tired..."),
    ("... WHAT ?!", "...What?!"),
    ("OH , NO ... !! NOT AGAIN !!", "Oh, no...!! Not again!!"),
    ("W - WAIT ?!", "W-wait?!"),
    ("I - I'M SORRY", "I-I'm sorry"),
    ("Y - Y - YOU SCARED ME !!!", "Y-y-you scared me!"),
    ("SO THEN ... WHY AM I HERE ? I DUNNO .", "So then... why am I here? I dunno."),
    ("WAIT…THERE", "Wait... there"),
])
def test_expressive_tts_preprocessing(raw, expected):
    assert TextPreprocessor().preprocess_for_tts(raw, expressive=True) == expected


@pytest.mark.unit
def test_expressive_leaves_mixed_case_text_alone():
    assert TextPreprocessor().preprocess_for_tts("Hey , where are you ?", expressive=True) == "Hey, where are you?"


@pytest.mark.unit
def test_default_tts_preprocessing_unchanged():
    """Without expressive, ellipses are still turned into spaces as before."""
    assert TextPreprocessor().preprocess_for_tts("UGH ... I'M SO TIRED ...") == "UGH I'M SO TIRED"


@pytest.mark.unit
def test_classification_preprocessing_unchanged():
    """The classifier's features must not change, or the trained model would no longer match."""
    assert TextPreprocessor().preprocess_for_classification("UGH ... I'M SO TIRED ...") == "ugh i'm so tired"
