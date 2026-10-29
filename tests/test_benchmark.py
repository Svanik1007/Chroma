import numpy as np
import pytest
from swcf.benchmark import read_ground_truth, reference_bpm, metrics


def test_ubfc_rows_and_timestamp_normalization(tmp_path):
    p = tmp_path / "ground_truth.txt"
    p.write_text("1 2 3 4\n72 72 74 74\n100 101 102 103\n")
    t, bpm, ppg = read_ground_truth(p)
    np.testing.assert_equal(t, [0,1,2,3])
    assert reference_bpm(t, bpm, 0, 3) == 73
    assert reference_bpm(t, bpm, 0, 4) is None
    p.write_text("1 2\n72 72\n")
    with pytest.raises(ValueError):
        read_ground_truth(p)


def test_metrics_report_coverage_and_do_not_fill_missing_predictions():
    rows = [{"prediction_bpm": 72, "reference_bpm": 70},
            {"prediction_bpm": 68, "reference_bpm": 70},
            {"prediction_bpm": None, "reference_bpm": 70}]
    m = metrics(rows)
    assert m["mae_bpm"] == 2 and m["rmse_bpm"] == 2
    assert m["coverage"] == 2/3
    assert metrics([])["mae_bpm"] is None
