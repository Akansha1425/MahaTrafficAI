"""Flajolet-Martin Algorithm — Academic Big Data Streaming Demo.

Approximates cardinality (number of distinct elements) in a data stream
without storing all elements. Used in big data systems like HyperLogLog
in Redis, BigQuery, and Apache Spark.

Academic Context:
- Exact distinct counting requires O(n) memory.
- FM-Sketch provides approximate distinct count in O(log n) memory.
- In a traffic analytics pipeline, FM-sketch can estimate distinct
  districts or accident locations observed in a streaming batch.
"""

import hashlib
import math
import random


class FlajoletMartinSketch:
    """Approximate distinct element counting using Flajolet-Martin algorithm.

    Uses the trailing-zeros heuristic on hash values to estimate cardinality.
    Multiple hash functions (trials) averaged for better accuracy.
    """

    def __init__(self, num_trials: int = 10, num_groups: int = 4):
        """Initialize FM Sketch.

        Args:
            num_trials: Number of independent hash function trials.
            num_groups: Number of groups for median-of-means aggregation.
        """
        self.num_trials = num_trials
        self.num_groups = num_groups
        self.BMAX = 32  # bit precision
        self.bitmap = [[0] * self.BMAX for _ in range(num_trials)]
        self.count = 0

    def _hash(self, item: str, seed: int) -> int:
        """Hash item with seed to a 32-bit integer."""
        h = hashlib.sha256(f"{seed}:{item}".encode()).hexdigest()
        return int(h[:8], 16)

    def _trailing_zeros(self, n: int) -> int:
        """Count trailing zeros in binary representation."""
        if n == 0:
            return self.BMAX
        count = 0
        while n & 1 == 0:
            count += 1
            n >>= 1
        return count

    def add(self, item: str) -> None:
        """Process an element from the stream."""
        for i in range(self.num_trials):
            h = self._hash(item, seed=i)
            r = self._trailing_zeros(h)
            if r < self.BMAX:
                self.bitmap[i][r] = 1
        self.count += 1

    def estimate(self) -> int:
        """Estimate the number of distinct elements seen."""
        # For each trial, find position of leftmost 0
        estimates = []
        for trial in self.bitmap:
            r = 0
            while r < self.BMAX and trial[r] == 1:
                r += 1
            estimates.append(2 ** r)

        # Apply median-of-means: split into groups, take mean per group, then median
        group_size = max(1, self.num_trials // self.num_groups)
        group_means = []
        for g in range(self.num_groups):
            start = g * group_size
            end = min(start + group_size, self.num_trials)
            group = estimates[start:end]
            if group:
                group_means.append(sum(group) / len(group))

        group_means.sort()
        mid = len(group_means) // 2
        if len(group_means) % 2 == 0:
            estimate = (group_means[mid - 1] + group_means[mid]) / 2
        else:
            estimate = group_means[mid]

        return round(estimate)


# ─── Demo ─────────────────────────────────────────────────────────────────────

def demo_distinct_district_counter():
    """Demonstrate FM-Sketch for estimating distinct districts in an accident stream."""
    import pandas as pd
    from pathlib import Path

    print("\n" + "=" * 65)
    print("BIG DATA DEMO: Flajolet-Martin — Distinct District Counter")
    print("=" * 65)

    # Try to load actual data, otherwise simulate
    data_path = Path(__file__).resolve().parent.parent.parent.parent / \
                "data" / "processed" / "parquet" / "accidents" / "maharashtra_accidents_clean.parquet"

    if data_path.exists():
        df = pd.read_parquet(data_path, engine="pyarrow")
        stream = df["district"].tolist()
        true_distinct = df["district"].nunique()
        print(f"  Stream source: Real accident dataset ({len(stream)} records)")
    else:
        # Simulate: 34 distinct districts repeated across 2460 records
        districts = [f"District_{i}" for i in range(34)]
        stream = districts * 72 + districts[:12]
        true_distinct = 34
        print(f"  Stream source: Simulated stream ({len(stream)} records)")

    print(f"  True distinct count: {true_distinct}")
    print(f"  Stream length:       {len(stream)}")

    # FM Sketch estimation
    fm = FlajoletMartinSketch(num_trials=20, num_groups=4)
    for item in stream:
        fm.add(item)

    estimated = fm.estimate()
    error_pct = abs(estimated - true_distinct) / true_distinct * 100

    print(f"\n  FM-Sketch Results:")
    print(f"    Estimated distinct: {estimated}")
    print(f"    True distinct:      {true_distinct}")
    print(f"    Error:              {error_pct:.1f}%")
    print(f"    Memory used:        {fm.num_trials * fm.BMAX} bits (vs ~{true_distinct*50*8} bits exact)")
    print(f"    Compression ratio:  {(true_distinct*50*8) / (fm.num_trials * fm.BMAX):.1f}x")
    print("=" * 65 + "\n")

    return fm, estimated, true_distinct


if __name__ == "__main__":
    demo_distinct_district_counter()
