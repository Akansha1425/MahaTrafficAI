"""Topic / Issue Extraction — Phase 4, MahaTraffic AI.

Identifies recurring road-safety issues from public social posts using
keyword/topic matching. No large LLM required.

Topics extracted (where supported by actual data):
  - potholes
  - poor_lighting
  - speeding
  - dangerous_intersections
  - congestion
  - road_damage
  - pedestrian_safety
  - signal_problems
  - drunk_driving
  - poor_road_conditions
  - waterlogging
  - accident_report

Input:  data/processed/social_clean.parquet  (or fallback)
Output: data/processed/social_topics.parquet

Only reports categories supported by actual data.
"""

from __future__ import annotations
from pathlib import Path
import json
import logging
import re
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("social.topics")

BASE_DIR = Path(__file__).resolve().parent.parent.parent
INPUT_PARQUET = BASE_DIR / "data" / "processed" / "social_clean.parquet"
FALLBACK_PARQUET = (
    BASE_DIR / "data" / "processed" / "parquet" / "social_media" / "maharashtra_social_clean.parquet"
)
OUTPUT_PARQUET = BASE_DIR / "data" / "processed" / "social_topics.parquet"
REPORT_PATH = BASE_DIR / "data" / "processed" / "social_topics_report.json"

# ─── Topic Keyword Map ────────────────────────────────────────────────────────

TOPIC_KEYWORDS: dict[str, list[str]] = {
    "potholes": [
        "pothole", "crater", "trench", "pit", "dug", "broken road", "road crater",
        "damaged road", "gravel",
    ],
    "poor_lighting": [
        "dark", "unlit", "no light", "streetlight", "street light", "darkness",
        "visibility", "night visibility",
    ],
    "speeding": [
        "speeding", "over-speed", "overspeed", "reckless", "rash driving", "fast",
        "speed limit", "speed camera", "lane cutting", "racing",
    ],
    "dangerous_intersections": [
        "intersection", "junction", "crossroad", "turning", "blind turn", "sharp curve",
        "u-turn", "traffic island", "roundabout",
    ],
    "congestion": [
        "congestion", "jam", "traffic jam", "bumper to bumper", "blocked", "bottleneck",
        "standstill", "slow traffic", "gridlock",
    ],
    "road_damage": [
        "broken road", "road damage", "road collapse", "fallen", "landslide",
        "cracked", "subsidence", "sinkhole",
    ],
    "pedestrian_safety": [
        "pedestrian", "zebra crossing", "foot overbridge", "footpath", "pavement",
        "children crossing", "walk", "walker", "ped safety",
    ],
    "signal_problems": [
        "signal", "traffic light", "signal not working", "broken signal", "red light",
        "no signal", "signal failure",
    ],
    "drunk_driving": [
        "drunk", "alcohol", "liquor", "drunken", "intoxicated", "drunk driver",
        "breath test", "drink and drive",
    ],
    "poor_road_conditions": [
        "waterlogging", "flooded", "water logging", "rain", "skid", "slippery",
        "monsoon", "weather", "mud", "muddy",
    ],
    "accident_report": [
        "accident", "crash", "collision", "hit and run", "fell", "overturned",
        "injured", "fatality", "fatal", "casualty", "victim",
    ],
}


def _clean_tokens(text: str) -> str:
    """Lowercase and clean text for keyword matching."""
    return text.lower() if isinstance(text, str) else ""


def extract_topics(text: str) -> list[str]:
    """Match text against topic keyword map; return all matched topics."""
    text_lower = _clean_tokens(text)
    matched = [
        topic
        for topic, keywords in TOPIC_KEYWORDS.items()
        if any(kw in text_lower for kw in keywords)
    ]
    return matched if matched else ["other"]


def run_topic_extraction() -> pd.DataFrame:
    """Execute topic extraction pipeline."""
    if INPUT_PARQUET.exists():
        df = pd.read_parquet(INPUT_PARQUET, engine="pyarrow")
    elif FALLBACK_PARQUET.exists():
        df = pd.read_parquet(FALLBACK_PARQUET, engine="pyarrow")
    else:
        raise FileNotFoundError("No social data Parquet found. Run clean_social.py first.")

    logger.info("Extracting topics from %d posts...", len(df))

    text_col = "clean_text" if "clean_text" in df.columns else "text"

    df["topics"] = df[text_col].fillna("").apply(extract_topics)
    df["primary_topic"] = df["topics"].apply(lambda t: t[0])
    df["topic_count"] = df["topics"].apply(len)

    # Explode for frequency analysis
    df_exploded = df.assign(topic=df["topics"]).explode("topic")
    topic_freq = df_exploded["topic"].value_counts().to_dict()
    logger.info("Topic frequencies: %s", topic_freq)

    # ── Save ──
    OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(OUTPUT_PARQUET, engine="pyarrow", compression="snappy", index=False)
    logger.info("Topics saved: %s", OUTPUT_PARQUET)

    return df


def compute_topic_report(df: pd.DataFrame) -> dict:
    """Generate topic frequency report."""
    df_exp = df.explode("topics")
    topic_freq = df_exp["topics"].value_counts().to_dict()

    # Remove 'other' for clean reporting if actual topics dominate
    real_topics = {k: v for k, v in topic_freq.items() if k != "other"}

    # Detected topics (topics with > 0 matches in real data)
    detected = sorted(real_topics, key=lambda x: real_topics[x], reverse=True)

    # Location-topic cross-tab (if location available)
    loc_topic = {}
    if "location" in df.columns:
        df_exp2 = df.explode("topics")
        lt = df_exp2.groupby(["location", "topics"]).size().reset_index(name="count")
        lt = lt.sort_values("count", ascending=False)
        loc_topic = lt.head(20).to_dict(orient="records")

    return {
        "total_posts_analyzed": int(len(df)),
        "topic_frequencies": {k: int(v) for k, v in topic_freq.items()},
        "detected_topics": detected,
        "topics_supported_by_data": [t for t in detected if real_topics.get(t, 0) > 0],
        "location_topic_distribution": loc_topic,
    }


def main():
    df = run_topic_extraction()
    report = compute_topic_report(df)

    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info("Topics report saved: %s", REPORT_PATH)

    print("\n" + "=" * 65)
    print("TOPIC / ISSUE EXTRACTION SUMMARY")
    print("=" * 65)
    print(f"  Total posts analyzed:   {report['total_posts_analyzed']}")
    print(f"\n  Detected Topics (by frequency):")
    for topic in report["detected_topics"]:
        freq = report["topic_frequencies"].get(topic, 0)
        bar = "▓" * (freq // 10)
        print(f"    {topic:<30} {freq:>5}  {bar}")
    print("=" * 65 + "\n")

    return df, report


if __name__ == "__main__":
    main()
