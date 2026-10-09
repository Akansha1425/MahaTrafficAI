"""Bloom Filter — Academic Big Data Streaming Algorithm Demo.

Implements a Bloom Filter for approximate district name membership testing.
Used as an academic demonstration of probabilistic data structures for
efficient set membership in streaming/big data contexts.

Academic Context:
- Bloom Filters are used in Apache Cassandra, HBase, and BigQuery for
  efficient key lookup without full table scans.
- In a traffic system, they can be used to quickly check if a location
  has been flagged as a high-risk black spot without a full DB query.
"""

import hashlib
import math
from typing import List


class BloomFilter:
    """Probabilistic set-membership data structure.

    False positives are possible; false negatives are not.
    Space-efficient: uses bit array instead of storing elements.
    """

    def __init__(self, expected_elements: int, false_positive_rate: float = 0.01):
        """Initialize Bloom Filter.

        Args:
            expected_elements: Expected number of elements to insert.
            false_positive_rate: Acceptable false positive probability (0-1).
        """
        self.n = expected_elements
        self.p = false_positive_rate
        # Optimal bit array size: m = -n * ln(p) / (ln 2)^2
        self.m = math.ceil(-expected_elements * math.log(false_positive_rate) / (math.log(2) ** 2))
        # Optimal number of hash functions: k = (m/n) * ln 2
        self.k = max(1, round((self.m / expected_elements) * math.log(2)))
        self.bit_array = bytearray(math.ceil(self.m / 8))
        self.count = 0

    def _get_bit_positions(self, item: str) -> List[int]:
        """Generate k hash-based bit positions for an item."""
        positions = []
        for seed in range(self.k):
            h = hashlib.md5(f"{seed}:{item}".encode()).hexdigest()
            pos = int(h, 16) % self.m
            positions.append(pos)
        return positions

    def _set_bit(self, pos: int) -> None:
        byte_idx, bit_idx = divmod(pos, 8)
        self.bit_array[byte_idx] |= (1 << bit_idx)

    def _get_bit(self, pos: int) -> bool:
        byte_idx, bit_idx = divmod(pos, 8)
        return bool(self.bit_array[byte_idx] & (1 << bit_idx))

    def add(self, item: str) -> None:
        """Add an element to the Bloom Filter."""
        for pos in self._get_bit_positions(item):
            self._set_bit(pos)
        self.count += 1

    def __contains__(self, item: str) -> bool:
        """Check if element might be in the set (probabilistic)."""
        return all(self._get_bit(pos) for pos in self._get_bit_positions(item))

    def estimated_false_positive_rate(self) -> float:
        """Current estimated false positive rate given actual insertions."""
        return (1 - math.exp(-self.k * self.count / self.m)) ** self.k

    @property
    def info(self) -> dict:
        return {
            "expected_elements": self.n,
            "target_fp_rate": self.p,
            "bit_array_size_bits": self.m,
            "bit_array_size_kb": round(self.m / 8 / 1024, 4),
            "hash_functions": self.k,
            "elements_inserted": self.count,
            "estimated_fp_rate": round(self.estimated_false_positive_rate(), 6),
        }


# ─── Demo ─────────────────────────────────────────────────────────────────────

def demo_accident_blackspot_filter():
    """Demonstrate Bloom Filter for high-risk district lookup in a streaming context."""
    print("\n" + "=" * 65)
    print("BIG DATA DEMO: Bloom Filter — Black Spot District Lookup")
    print("=" * 65)

    # Known high-risk districts from our analytics
    HIGH_RISK_DISTRICTS = [
        "Pune", "Mumbai", "Nashik", "Ahmednagar", "Thane",
        "Nagpur", "Solapur", "Kolhapur", "Satara", "Raigad"
    ]

    # Initialize filter for 50 districts, 1% FP rate
    bf = BloomFilter(expected_elements=50, false_positive_rate=0.01)
    for district in HIGH_RISK_DISTRICTS:
        bf.add(district)

    print(f"\n  Filter Configuration:")
    for k, v in bf.info.items():
        print(f"    {k:<30} {v}")

    test_queries = [
        ("Pune", True),
        ("Mumbai", True),
        ("Nashik", True),
        ("Latur", False),
        ("Dhule", False),
        ("Kolhapur", True),
        ("Gadchiroli", False),
        ("Nagpur", True),
    ]

    print(f"\n  Membership Queries:")
    correct = 0
    for district, is_high_risk in test_queries:
        result = district in bf
        expected = is_high_risk
        mark = "OK" if result == expected else "FP"  # FP = false positive
        print(f"    [{mark}] '{district}': filter={result}, actual_high_risk={expected}")
        if result == expected:
            correct += 1

    accuracy = correct / len(test_queries) * 100
    print(f"\n  Accuracy: {correct}/{len(test_queries)} ({accuracy:.1f}%)")
    print(f"  Estimated FP rate: {bf.estimated_false_positive_rate():.6f}")
    print(f"\n  Space saved vs Python set: ~{len(HIGH_RISK_DISTRICTS) * 50}B -> {bf.m//8}B")
    print("=" * 65 + "\n")

    return bf


if __name__ == "__main__":
    demo_accident_blackspot_filter()
