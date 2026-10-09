"""Phase 4 Tests — Big Data Streaming Algorithms (DGIM and Flajolet-Martin)."""

import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


class TestDGIMAlgorithm:
    """Tests for bigdata/streaming_algorithms/dgim/dgim.py"""

    def test_dgim_basic_window(self):
        from bigdata.streaming_algorithms.dgim.dgim import DGIMWindow
        window = DGIMWindow(window_size=8)
        for _ in range(8):
            window.process_bit(1)
        est = window.estimate_count()
        assert 4 <= est <= 8, f"Expected estimate in [4, 8], got {est}"

    def test_dgim_all_zeros(self):
        from bigdata.streaming_algorithms.dgim.dgim import DGIMWindow
        window = DGIMWindow(window_size=10)
        for _ in range(10):
            window.process_bit(0)
        assert window.estimate_count() == 0

    def test_dgim_bucket_expiration(self):
        from bigdata.streaming_algorithms.dgim.dgim import DGIMWindow
        window = DGIMWindow(window_size=5)
        window.process_bit(1)
        for _ in range(5):
            window.process_bit(0)
        assert window.estimate_count() == 0


class TestFlajoletMartinAlgorithm:
    """Tests for bigdata/streaming_algorithms/flajolet_martin/flajolet_martin.py"""

    def test_flajolet_martin_distinct_count(self):
        from bigdata.streaming_algorithms.flajolet_martin.flajolet_martin import FlajoletMartinSketch
        sketch = FlajoletMartinSketch(num_trials=30)
        items = [f"district_{i}" for i in range(50)]
        for item in items:
            sketch.add(item)
            sketch.add(item)  # duplicate should not inflate count significantly

        est = sketch.estimate()
        # Should be order of magnitude reasonable for 50 distinct items
        assert est > 10, f"Estimate too low: {est}"
        assert est < 250, f"Estimate too high: {est}"

    def test_flajolet_martin_empty(self):
        from bigdata.streaming_algorithms.flajolet_martin.flajolet_martin import FlajoletMartinSketch
        sketch = FlajoletMartinSketch(num_trials=10)
        assert sketch.estimate() == 0
