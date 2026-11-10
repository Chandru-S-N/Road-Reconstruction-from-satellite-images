"""Metric definitions on small hand-made masks."""

import numpy as np
import pytest

from core.evaluation import count_components, gap_counts, gap_metrics_from_counts, metrics_from_counts, pixel_counts


def test_pixel_metrics():
    ref = np.zeros((10, 10), np.uint8); ref[2:8, 4:6] = 255
    pred = np.zeros_like(ref); pred[2:8, 5:7] = 255
    m = metrics_from_counts(pixel_counts(pred, ref))
    assert m["precision"] == pytest.approx(0.5) and m["iou"] == pytest.approx(6 / 18)


def test_gap_metrics_and_components():
    ref = np.zeros((10, 30), np.uint8); ref[4:6, :] = 255
    inp = ref.copy(); inp[:, 10:20] = 0
    rec = inp.copy(); rec[4:6, 10:15] = 255
    g = gap_metrics_from_counts(gap_counts(inp, rec, ref, 0))
    assert g["gap_precision"] == 1 and g["gap_recall"] == pytest.approx(0.5)
    assert count_components(inp) == 2 and count_components(ref) == 1
