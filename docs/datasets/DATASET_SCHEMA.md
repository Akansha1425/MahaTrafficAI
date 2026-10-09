# MahaTraffic AI — Dataset Schema Documentation

**Generated:** 2026-10-09  
**Phase:** 4 — Intelligence Layer

---

## 1. maharashtra_district_accidents_2019_2023.csv

| Property | Value |
|----------|-------|
| **Path** | `data/raw/accidents/maharashtra_district_accidents_2019_2023.csv` |
| **Source** | Maharashtra Traffic Police / MoRTH district-level records (2019–2023) |
| **Rows** | 2,460 |
| **Columns** | 13 |

### Column Schema

| Column | Dtype | Missing | Description |
|--------|-------|---------|-------------|
| `year` | int64 | 0 | Year of record (2019–2023) |
| `month` | int64 | 0 | Month of record (1–12) |
| `state` | object | 0 | Always "Maharashtra" |
| `district` | object | 0 | District name (location field) |
| `police_unit` | object | 0 | Responsible police unit |
| `jurisdiction` | object | 0 | Rural / Urban |
| `road_type` | object | 0 | National Highway / State Highway / Expressway / Urban Road |
| `accident_count` | int64 | 0 | Total accidents in period |
| `fatal_accidents` | int64 | 0 | Accidents resulting in fatalities |
| `deaths` | int64 | 0 | Total deaths |
| `injuries` | int64 | 0 | Total injured |
| `primary_cause` | object | 0 | Primary reported cause (Over-speeding, Drunk Driving, etc.) |
| `time_period` | object | 0 | Day (06:00–18:00) / Night (18:00–06:00) |

**Derived columns (in `accidents_clean.csv`)**:
`fatality_ratio`, `injury_ratio`, `severity_index`, `season`

**Location field:** `district`  
**Date fields:** `year`, `month`  
**Severity fields:** `fatal_accidents`, `deaths`, `injuries`, `severity_index`  
**Road/Cause fields:** `road_type`, `primary_cause`, `jurisdiction`  
**Missing values:** None  

---

## 2. maharashtra_road_safety_public_posts.csv

| Property | Value |
|----------|-------|
| **Path** | `data/raw/social_media/maharashtra_road_safety_public_posts.csv` |
| **Source** | Twitter/X Public Archive — road safety posts from Maharashtra (curated dataset) |
| **Rows** | 1,200 |
| **Columns** | 10 |

### Column Schema

| Column | Dtype | Missing | Description |
|--------|-------|---------|-------------|
| `post_id` | object | 0 | Unique post identifier (MAHA_SOC_XXXXX) |
| `timestamp` | object | 0 | Post datetime (YYYY-MM-DD HH:MM:SS) |
| `location` | object | 0 | City/district name |
| `text` | object | 0 | Post text content |
| `source` | object | 0 | Platform (Twitter/X Public Archive) |
| `likes` | int64 | 0 | Number of likes |
| `shares` | int64 | 0 | Number of shares/retweets |
| `comments` | int64 | 0 | Number of comments |
| `hashtags` | object | 0 | Semicolon-separated hashtag list |
| `category_tag` | object | 0 | Pre-labelled category (pothole, speeding, etc.) |

**Social media fields:** `text`, `source`, `likes`, `shares`, `comments`, `hashtags`  
**Location field:** `location`  
**Date fields:** `timestamp`  
**Author/user:** Not present — no user-level network graph possible  
**Engagement:** `likes`, `shares`, `comments`  

> **NOTE:** No author/mention/reply fields exist. Graph-based community detection cannot be applied to real data. See `social_analytics/community_detection/social_graph.py` for the academic demonstration.

---

## 3. MoRTH Reference Datasets (2020–2024)

### 3a. morth_statewise_accidents_2020_2024.csv

| Property | Value |
|----------|-------|
| **Path** | `data/raw/accidents/morth_statewise_accidents_2020_2024.csv` |
| **Source** | Ministry of Road Transport & Highways (MoRTH) Annual Report |
| **Rows** | 39 | **Columns** | 14 |

Columns: `Sl No`, `State`, `2020–2024 Accidents`, `Change from 2023 to 2024`, `% change`, `2020–2024 Ranking`

### 3b. morth_statewise_fatalities_2020_2024.csv

- 39 rows × 13 columns  
- Columns: `Sl No`, `State`, `2020–2024 Killed`, `% change from 2023 to 2024`, `2020–2024 Ranking`

### 3c. morth_major_cities_accidents_2022.csv

- 51 rows × 13 columns  
- Columns: `Cities`, Accidents/Deaths/Injured for 2021 and 2022 with Rankings

### 3d. morth_major_cities_accidents_2024.csv

- 51 rows × 14 columns  
- Columns: `Sl No`, `City`, Accidents/Killed/Injured for 2023 and 2024 with Rankings

### 3e. morth_cities_fatalities_by_mode_2024.csv

- 51 rows × 11 columns  
- Columns: `City`, fatalities by transport mode (Pedestrians, Two-Wheelers, Cars, Trucks, etc.)

### 3f. morth_cities_traffic_violations_2024.csv

- 51 rows × 9 columns  
- Columns: `City`, `Over-speeding`, `Drunk Driving`, `Wrong Side`, `Jumping Red Light`, `Mobile Phone`, `Others`, `Total`

### 3g. morth_collision_types_2024.csv

- 19 rows × 10 columns  
- 2023 and 2024 collision type data (Accidents/Killed/Injured + % change)

---

## 4. Processed Datasets

| File | Path | Description |
|------|------|-------------|
| `accidents_clean.csv` | `data/processed/` | Cleaned accident data + derived features (2,460 rows × 17 cols) |
| `social_media_clean.csv` | `data/processed/` | Cleaned social posts (1,200 rows) |
| `maharashtra_accidents_clean.parquet` | `data/processed/parquet/accidents/` | Parquet version of cleaned accidents |
| `maharashtra_social_clean.parquet` | `data/processed/parquet/social_media/` | Parquet version of cleaned social posts |
| `social_features.parquet` | `data/processed/parquet/social_media/` | Social features with sentiment scores |
| `risk_features.parquet` | `data/processed/features/` | ML feature matrix (2,460 rows × 19 cols) |

---

## 5. RAG Document Corpus

| File | Path | Description |
|------|------|-------------|
| `morth_road_safety_guidelines_2022.md` | `rag/documents/` | MoRTH road safety guidelines summary |
| `maharashtra_motor_vehicles_rules_safety_summary.md` | `rag/documents/` | Maharashtra MV Rules safety provisions |
| `irc_sp_88_road_safety_audit_manual.md` | `rag/documents/` | IRC Road Safety Audit Manual excerpts |

---

## 6. Key Notes for Phase 4

- **Primary accident dataset:** `maharashtra_district_accidents_2019_2023.csv` — 2,460 records, fully populated, no missing values
- **Primary social dataset:** `maharashtra_road_safety_public_posts.csv` — 1,200 records, no author fields
- **Graph analysis:** Not applicable to real data (no user/mention/reply columns). Academic demo provided.
- **Temporal coverage:** Accident data spans 2019–2023 (5 years). Social data spans ~2021–2023.
- **Location granularity:** District level for accidents, city level for social posts.
