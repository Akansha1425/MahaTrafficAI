"""DGIM Algorithm — Offline Demonstration, MahaTraffic AI.

Implements the Datar-Gionis-Indyk-Motwani (DGIM) algorithm for
approximate counting of 1s in a sliding window over a bit stream.

Academic Context:
    DGIM is used in streaming data scenarios where:
      - The full stream cannot be stored in memory
      - Approximate counts over a recent window are sufficient
    In a traffic monitoring context, it can estimate the approximate
    number of accident-flagged time windows in the last N time slots
    without storing the full history.

This is an OFFLINE DEMONSTRATION using simulated bit stream data.
It is NOT connected to live traffic or real-time data.
"""

from __future__ import annotations
from collections import deque
from typing import Optional


class DGIMBucket:
    """Represents a single DGIM bucket."""
    def __init__(self, timestamp: int, size: int):
        self.timestamp = timestamp  # most recent 1 in this bucket
        self.size = size            # power of 2


class DGIMWindow:
    """
    DGIM sliding-window approximate counter.

    Maintains a set of buckets satisfying:
      - Each bucket size is a power of 2
      - At most 2 buckets of each size exist
      - Total buckets ≤ O(log N)

    Space: O(log²N) instead of O(N)
    Error bound: ≤ 50% of true count
    """

    def __init__(self, window_size: int):
        """
        Args:
            window_size: Number of most recent bits to consider.
        """
        self.N = window_size
        self.buckets: deque[DGIMBucket] = deque()  # ordered newest → oldest
        self.current_time = 0

    def process_bit(self, bit: int) -> None:
        """Process the next bit in the stream (0 or 1)."""
        self.current_time += 1

        # Remove buckets that have fallen outside the window
        self._expire_old_buckets()

        if bit == 1:
            # Create new bucket of size 1
            self.buckets.appendleft(DGIMBucket(self.current_time, 1))
            # Merge if needed
            self._merge()

    def _expire_old_buckets(self) -> None:
        """Remove buckets whose timestamps are outside the window."""
        while self.buckets and (self.current_time - self.buckets[-1].timestamp >= self.N):
            self.buckets.pop()

    def _merge(self) -> None:
        """Merge buckets of the same size to maintain DGIM invariant (≤2 per size)."""
        sizes: dict[int, list] = {}
        for b in self.buckets:
            sizes.setdefault(b.size, []).append(b)

        for size, group in sizes.items():
            if len(group) >= 3:
                # Merge oldest two of this size into one of size*2
                group.sort(key=lambda b: b.timestamp)
                old1, old2 = group[0], group[1]
                # Remove them from deque
                self.buckets = deque(b for b in self.buckets if b is not old1 and b is not old2)
                # Add merged bucket with oldest timestamp
                merged = DGIMBucket(max(old1.timestamp, old2.timestamp), size * 2)
                # Insert in correct position
                new_deque = deque()
                inserted = False
                for b in self.buckets:
                    if not inserted and b.timestamp < merged.timestamp:
                        new_deque.append(merged)
                        inserted = True
                    new_deque.append(b)
                if not inserted:
                    new_deque.append(merged)
                self.buckets = new_deque
                # Recurse to check new size
                self._merge()
                break

    def estimate_count(self) -> int:
        """Return approximate count of 1s in the current window."""
        if not self.buckets:
            return 0
        # Sum all buckets except the oldest, then add half of the oldest
        total = sum(b.size for b in list(self.buckets)[:-1])
        oldest_size = list(self.buckets)[-1].size
        return total + oldest_size // 2

    @property
    def bucket_summary(self) -> list[dict]:
        return [{"timestamp": b.timestamp, "size": b.size} for b in self.buckets]


# ─── Demo ─────────────────────────────────────────────────────────────────────

def demo_accident_window_counting():
    """
    [OFFLINE DEMONSTRATION — NOT LIVE DATA]

    Simulates a bit stream representing whether each hourly time slot had
    an accident report (1) or not (0) for a district.

    Demonstrates DGIM approximate counting over a sliding window of 20 slots.
    """
    print("\n" + "=" * 70)
    print("BIG DATA DEMO: DGIM Algorithm — Sliding Window 1-Counter")
    print("[OFFLINE DEMONSTRATION — Simulated accident-flag bit stream]")
    print("=" * 70)

    # Simulated accident flags: 1 = accident reported in this hour, 0 = none
    bit_stream = [
        1, 0, 0, 1, 1, 0, 1, 0, 0, 1,
        1, 1, 0, 0, 1, 0, 1, 1, 0, 0,
        1, 0, 1, 0, 0, 1, 1, 0, 1, 0,
    ]

    window_size = 20
    dgim = DGIMWindow(window_size=window_size)

    print(f"\n  Window size: {window_size} time slots")
    print(f"  Stream length: {len(bit_stream)} bits")
    print(f"\n  Processing stream...")

    checkpoints = [10, 20, 25, 30]
    for t, bit in enumerate(bit_stream, 1):
        dgim.process_bit(bit)
        if t in checkpoints:
            true_count = sum(bit_stream[max(0, t - window_size):t])
            approx_count = dgim.estimate_count()
            error = abs(true_count - approx_count)
            error_pct = (error / max(true_count, 1)) * 100
            print(f"\n  [t={t:>3}] Exact={true_count}  DGIM≈{approx_count}  "
                  f"Error={error} ({error_pct:.1f}%)")
            print(f"   Buckets: {dgim.bucket_summary}")

    print("\n  Algorithm Properties:")
    print(f"    Space: O(log² N) = O({len(dgim.buckets)}) buckets for N={window_size}")
    print(f"    Error guarantee: ≤ 50% of true count")
    print(f"    Use case: Approximate accident frequency over sliding time window")
    print("=" * 70 + "\n")

    return dgim


# ─── Tests ────────────────────────────────────────────────────────────────────

def test_dgim_basic():
    """Basic correctness test."""
    # Stream of all 1s in window of 8
    dgim = DGIMWindow(window_size=8)
    for _ in range(8):
        dgim.process_bit(1)
    est = dgim.estimate_count()
    # True count is 8; estimate should be 4–8 (within 50%)
    assert est >= 4, f"Estimate too low: {est}"
    assert est <= 8, f"Estimate too high: {est}"
    print(f"  [PASS] all-ones stream: true=8, estimate={est}")


def test_dgim_all_zeros():
    """Zero stream should give 0 count."""
    dgim = DGIMWindow(window_size=10)
    for _ in range(10):
        dgim.process_bit(0)
    assert dgim.estimate_count() == 0
    print("  [PASS] all-zeros stream: estimate=0")


def test_dgim_single_one():
    """Single 1 at start, then expires."""
    dgim = DGIMWindow(window_size=5)
    dgim.process_bit(1)
    for _ in range(5):
        dgim.process_bit(0)
    # The 1 should have expired
    assert dgim.estimate_count() == 0
    print("  [PASS] expired 1: estimate=0 after window")


if __name__ == "__main__":
    print("Running DGIM tests...")
    test_dgim_basic()
    test_dgim_all_zeros()
    test_dgim_single_one()
    print("All DGIM tests passed.\n")
    demo_accident_window_counting()
