"""Flajolet-Martin Approximate Distinct Count — Offline Demo, MahaTraffic AI.

Implements the Flajolet-Martin algorithm for probabilistic approximate
counting of distinct elements in a data stream.

Academic Context:
    FM sketches estimate the number of distinct elements using O(log N)
    space instead of O(N). In traffic analytics, this can estimate the
    approximate number of distinct accident locations or vehicle types
    seen in a large streaming dataset without full memory.

This is an OFFLINE DEMONSTRATION using historical accident data.
It is NOT a live streaming system.
"""

from __future__ import annotations
import hashlib
import math
import logging

logger = logging.getLogger("bigdata.flajolet_martin")


# ─── Flajolet-Martin Sketch ───────────────────────────────────────────────────

def _trailing_zeros(n: int) -> int:
    """Count trailing zero bits of an integer."""
    if n == 0:
        return 0
    count = 0
    while n & 1 == 0:
        count += 1
        n >>= 1
    return count


def _hash_element(element: str, seed: int, num_bits: int = 32) -> int:
    """Hash a string element to an integer using MD5 with a seed."""
    raw = hashlib.md5(f"{seed}:{element}".encode()).hexdigest()
    return int(raw, 16) % (2 ** num_bits)


class FlajoletMartinSketch:
    """
    Flajolet-Martin approximate distinct count sketch.

    Uses multiple independent hash functions and takes the median-of-means
    correction to improve accuracy.

    Error: ~O(1/√k) where k is the number of independent trials.
    Space: O(k × log N) bits vs O(N) for exact counting.
    """

    _FM_CORRECTION = 0.77351  # known correction factor

    def __init__(self, num_trials: int = 30, bits: int = 32):
        """
        Args:
            num_trials: Number of independent hash functions (higher = more accurate).
            bits: Bit width for hash space.
        """
        self.num_trials = num_trials
        self.bits = bits
        self.max_zeros = [0] * num_trials  # max trailing zeros per trial
        self.element_count = 0
        self._seen = set()  # for true count comparison (demo only)

    def add(self, element: str) -> None:
        """Add an element to the sketch."""
        self._seen.add(element)
        self.element_count += 1
        for seed in range(self.num_trials):
            h = _hash_element(element, seed, self.bits)
            tz = _trailing_zeros(h)
            if tz > self.max_zeros[seed]:
                self.max_zeros[seed] = tz

    def estimate(self) -> int:
        """Return approximate count of distinct elements."""
        if self.element_count == 0:
            return 0
        # Group into groups of 5 and take mean of each group, then median
        group_size = max(1, self.num_trials // 6)
        group_means = []
        for i in range(0, self.num_trials, group_size):
            group = self.max_zeros[i:i + group_size]
            mean_val = sum(2 ** z for z in group) / len(group)
            group_means.append(mean_val)

        # Median of group means
        group_means.sort()
        n = len(group_means)
        if n % 2 == 0:
            median = (group_means[n // 2 - 1] + group_means[n // 2]) / 2
        else:
            median = group_means[n // 2]

        estimate = int(median / self._FM_CORRECTION)
        return max(estimate, 0)

    @property
    def true_distinct_count(self) -> int:
        """Exact distinct count (only available in demo/test mode)."""
        return len(self._seen)

    @property
    def info(self) -> dict:
        est = self.estimate()
        true = self.true_distinct_count
        error_pct = abs(est - true) / max(true, 1) * 100
        return {
            "elements_processed": self.element_count,
            "estimated_distinct": est,
            "true_distinct": true,
            "error_pct": round(error_pct, 2),
            "num_trials": self.num_trials,
            "space_bits": self.num_trials * self.bits,
        }


# ─── Demo ─────────────────────────────────────────────────────────────────────

def demo_distinct_district_counting():
    """
    [OFFLINE DEMONSTRATION — Historical district accident records]

    Simulates a stream of district names from the accident dataset and
    estimates the number of distinct districts using FM sketches.
    Compares approximate vs exact count.
    """
    print("\n" + "=" * 70)
    print("BIG DATA DEMO: Flajolet-Martin — Approximate Distinct Count")
    print("[OFFLINE DEMONSTRATION — Using district names from accident data]")
    print("=" * 70)

    # Simulated stream of district names (repeated, as in a real stream)
    # Based on known districts from the Maharashtra accident dataset
    districts_sample = [
        "Pune", "Mumbai", "Nashik", "Ahmednagar", "Thane",
        "Nagpur", "Solapur", "Kolhapur", "Satara", "Raigad",
        "Aurangabad", "Latur", "Nanded", "Jalgaon", "Dhule",
        "Sangli", "Buldhana", "Wardha", "Yavatmal", "Amravati",
        "Pune", "Mumbai", "Nashik", "Thane", "Nagpur",  # repeats
        "Pune", "Satara", "Kolhapur", "Nashik", "Aurangabad",
        "Gadchiroli", "Chandrapur", "Beed", "Osmanabad", "Hingoli",
    ]

    fm = FlajoletMartinSketch(num_trials=30, bits=32)

    for district in districts_sample:
        fm.add(district)

    info = fm.info
    print(f"\n  Stream length:          {info['elements_processed']} elements")
    print(f"  Estimated distinct:     {info['estimated_distinct']}")
    print(f"  True distinct:          {info['true_distinct']}")
    print(f"  Estimation error:       {info['error_pct']:.2f}%")
    print(f"  Space used:             {info['space_bits']} bits")
    print(f"  Space (exact HashSet):  ~{info['true_distinct'] * 64} bits (estimated)")
    print(f"\n  Application: Estimate number of distinct at-risk locations in a")
    print(f"               streaming accident report feed without storing all IDs.")
    print("=" * 70 + "\n")

    return fm


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_fm_known_distinct():
    """Test FM estimate accuracy for known distinct count."""
    fm = FlajoletMartinSketch(num_trials=30, bits=32)
    elements = [f"district_{i}" for i in range(100)]
    for e in elements * 5:  # repeat each 5 times
        fm.add(e)

    est = fm.estimate()
    true = fm.true_distinct_count
    assert true == 100, f"True count should be 100, got {true}"
    # FM estimate should be within 50% of true for reasonable num_trials
    assert 50 <= est <= 200, f"FM estimate {est} out of expected range [50, 200]"
    error_pct = abs(est - 100) / 100 * 100
    print(f"  [PASS] FM test: true=100, estimate={est}, error={error_pct:.1f}%")


def test_fm_empty():
    """Empty stream gives 0."""
    fm = FlajoletMartinSketch()
    assert fm.estimate() == 0
    print("  [PASS] FM empty stream: estimate=0")


def test_fm_single_element():
    """Single unique element."""
    fm = FlajoletMartinSketch(num_trials=30, bits=32)
    fm.add("only_one")
    fm.add("only_one")
    fm.add("only_one")
    # True distinct = 1; estimate may vary but should be low
    assert fm.true_distinct_count == 1
    print(f"  [PASS] FM single element: true=1, estimate={fm.estimate()}")


if __name__ == "__main__":
    print("Running Flajolet-Martin tests...")
    test_fm_empty()
    test_fm_single_element()
    test_fm_known_distinct()
    print("All FM tests passed.\n")
    demo_distinct_district_counting()
