"""
Unit tests for the punctuation-independent classifier features
(text shape, layout, image background, neighbours) and their wiring.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.config import settings
from backend.ml.data_collector import MLDataCollector
from backend.services.language_features import (
    context_features,
    image_background_features,
    layout_features,
    text_shape_features,
    to_grayscale_array,
)
from backend.services.text_box_classifier import TextBoxClassifier
from backend.services.vision import BoundingBox


def box(left, top, width, height):
    return BoundingBox([
        {"x": left, "y": top}, {"x": left + width, "y": top},
        {"x": left + width, "y": top + height}, {"x": left, "y": top + height},
    ])


# Text shape

@pytest.mark.unit
def test_text_shape_interface_line():
    f = text_shape_features("RP NEEDED : 15")
    assert f["label_colon"] == 1.0
    assert f["digit_ratio"] > 0.1
    assert f["lower_ratio"] == 0.0


@pytest.mark.unit
def test_text_shape_repeated_sound_effect():
    f = text_shape_features("STEP STEP STEP")
    assert f["repeat_ratio"] == pytest.approx(2 / 3)
    assert f["sfx_ratio"] == 1.0


@pytest.mark.unit
def test_text_shape_mixed_case_credit_line():
    f = text_shape_features("Art by Someone")
    assert f["lower_ratio"] > 0.5
    assert f["sfx_ratio"] == 0.0


@pytest.mark.unit
def test_text_shape_ordinary_dialogue():
    f = text_shape_features("WHERE ARE YOU GOING ?")
    assert f == {"lower_ratio": 0.0, "digit_ratio": 0.0, "label_colon": 0.0,
                 "repeat_ratio": 0.0, "sfx_ratio": 0.0}


@pytest.mark.unit
def test_text_shape_empty():
    assert text_shape_features("") == {"lower_ratio": 0.0, "digit_ratio": 0.0, "label_colon": 0.0,
                                       "repeat_ratio": 0.0, "sfx_ratio": 0.0}


# Layout

@pytest.mark.unit
def test_layout_relative_position_and_letter_height():
    f = layout_features(box(200, 500, 400, 100), 800, 1000,
                        word_boxes=[box(0, 0, 10, 20), box(0, 0, 10, 30), box(0, 0, 10, 40)])
    assert f["rel_x"] == pytest.approx(0.5)
    assert f["rel_y"] == pytest.approx(0.55)
    assert f["rel_width"] == pytest.approx(0.5)
    assert f["letter_height"] == pytest.approx(30 / 800)   # median word height / panel width


@pytest.mark.unit
def test_layout_without_word_boxes_uses_box_height():
    assert layout_features(box(0, 0, 100, 80), 800, 1000)["letter_height"] == pytest.approx(80 / 800)


# Image background

def _noise(shape, seed=0):
    return np.random.default_rng(seed).random(shape).astype(np.float32)


@pytest.mark.unit
def test_image_background_bubble_vs_artwork():
    """Text in a white bubble has a light, flat ring; text on artwork doesn't."""
    gray = _noise((400, 400))
    gray[100:200, 100:300] = 1.0            # white speech bubble
    gray[140:160, 140:260] = 0.0            # dark text inside it
    bubble = image_background_features(gray, box(130, 135, 140, 30))
    artwork = image_background_features(_noise((400, 400), 1), box(130, 135, 140, 30))

    assert bubble["ring_mean"] > 0.9 and bubble["ring_std"] < 0.1
    assert artwork["ring_std"] > 0.2
    assert set(bubble) == {"inside_mean", "inside_std", "ring_mean", "ring_std"}


@pytest.mark.unit
def test_image_background_clips_to_image_and_handles_edges():
    gray = np.ones((100, 100), dtype=np.float32)
    f = image_background_features(gray, box(-10, -5, 50, 20))     # partly outside the panel
    assert f["inside_mean"] == 1.0


@pytest.mark.unit
def test_image_background_without_image_or_empty_box():
    assert image_background_features(None, box(0, 0, 10, 10)) == {}
    assert image_background_features(np.ones((50, 50)), box(60, 60, 10, 10)) == {}


@pytest.mark.unit
def test_to_grayscale_array():
    arr = to_grayscale_array(Image.new("RGB", (4, 3), "white"))
    assert arr.shape == (3, 4) and arr.max() == pytest.approx(1.0)
    assert to_grayscale_array(None) is None


# Neighbours

@pytest.mark.unit
def test_context_split_sentence():
    """Second half of a split line: previous bubble ends mid-sentence, close and aligned."""
    texts = ["I REALLY THOUGHT THAT", "YOU WOULD COME .", "SLAM"]
    boxes = [box(300, 100, 200, 40), box(300, 150, 200, 40), box(0, 800, 300, 200)]
    f = context_features(texts, boxes, [0.8, 0.7, 0.4], 800, 1000)

    assert f[1]["prev_open"] == 1.0
    assert f[1]["nb_gap"] == pytest.approx(10 / 1000)
    assert f[1]["nb_aligned"] == pytest.approx(1.0)
    assert f[1]["nb_max_score"] == pytest.approx(0.8)
    assert f[2]["prev_open"] == 0.0            # previous line ends with '.'
    assert all(x["panel_bubbles"] == 3.0 for x in f)


@pytest.mark.unit
def test_context_single_bubble_panel():
    f = context_features(["HELLO ."], [box(0, 0, 10, 10)], [0.9], 800, 1000)
    assert f == [{"nb_gap": 1.0, "nb_max_score": 0.0, "prev_open": 0.0,
                  "nb_aligned": 0.0, "panel_bubbles": 1.0}]


@pytest.mark.unit
def test_context_overlapping_boxes_have_zero_gap():
    f = context_features(["A", "B"], [box(0, 0, 100, 50), box(0, 30, 100, 50)], [0.5, 0.5], 800, 1000)
    assert f[0]["nb_gap"] == 0.0


# Wiring into the classifier and collector

EXTRA_KEYS = {"lower_ratio", "digit_ratio", "label_colon", "repeat_ratio", "sfx_ratio",
              "rel_x", "rel_y", "rel_width", "letter_height",
              "nb_gap", "nb_max_score", "prev_open", "nb_aligned", "panel_bubbles"}
IMAGE_KEYS = {"inside_mean", "inside_std", "ring_mean", "ring_std"}


@pytest.fixture
def heuristic_classifier(monkeypatch, tmp_path):
    monkeypatch.setattr(settings, "CLASSIFIER_MODE", "heuristic")
    monkeypatch.setattr(settings, "ML_MODEL_PATH", str(tmp_path / "none.joblib"))
    monkeypatch.setattr(settings, "ML_COLLECT_DATA", True)
    clf = TextBoxClassifier()
    clf.ml_data_collector.samples.clear()
    return clf


def _bubbles():
    return [SimpleNamespace(text="WHERE ARE YOU GOING ?", bounding_box=box(100, 100, 300, 60),
                            ocr_results=[], panel_id=0),
            SimpleNamespace(text="SLAM", bounding_box=box(400, 600, 300, 200),
                            ocr_results=[], panel_id=0)]


@pytest.mark.unit
def test_classifier_records_extra_features_with_image(heuristic_classifier):
    image = Image.new("RGB", (800, 1000), "white")
    heuristic_classifier.filter_text_bubbles(_bubbles(), 800, 1000, image=image)
    sample = heuristic_classifier.ml_data_collector.samples[0]
    assert {f"feature_{k}" for k in EXTRA_KEYS | IMAGE_KEYS} <= set(sample)


@pytest.mark.unit
def test_classifier_without_image_skips_only_image_features(heuristic_classifier):
    heuristic_classifier.filter_text_bubbles(_bubbles(), 800, 1000)
    sample = heuristic_classifier.ml_data_collector.samples[0]
    assert {f"feature_{k}" for k in EXTRA_KEYS} <= set(sample)
    assert not {f"feature_{k}" for k in IMAGE_KEYS} & set(sample)


@pytest.mark.unit
def test_extra_features_do_not_change_formula_score(heuristic_classifier):
    clf = heuristic_classifier
    bubble = _bubbles()[0]
    pseudo = SimpleNamespace(text=bubble.text, bounding_box=bubble.bounding_box)
    base = clf._compute_features(pseudo, 800 * 1000, [pseudo])
    before = clf._compute_score(dict(base))
    extended = dict(base)
    clf._add_extra_features([bubble.text], [bubble.bounding_box], [[]], [extended], 800, 1000)
    assert set(extended) > set(base)
    assert clf._compute_score(extended) == before


@pytest.mark.unit
def test_collector_records_real_box_position():
    collector = MLDataCollector(output_dir="/tmp/panelpals_test_collector")
    collector.collect_sample(text="HI .", features={}, score=0.9, bbox=box(120, 340, 50, 20))
    sample = collector.samples[-1]
    assert (sample["bbox_x"], sample["bbox_y"]) == (120, 340)
