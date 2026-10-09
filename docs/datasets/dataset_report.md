# Dataset Quality Report — MahaTraffic AI

## Report Generated: October 2026 (Phase 2+3 Completion)

---

## Accident Dataset Quality Report

| Metric | Value |
|---|---|
| **File** | `maharashtra_district_accidents_2019_2023.csv` |
| **Rows** | 2,460 |
| **Columns** | 13 |
| **Missing Values** | 0 |
| **Duplicate Rows** | 0 |
| **Invalid Records** | 0 |
| **Valid Records** | 2,460 |
| **Date Range** | 2019 – 2023 (5 years) |
| **Geographic Coverage** | 34 districts across 41 police units in Maharashtra |
| **Districts Covered** | Pune, Mumbai, Nashik, Nagpur, Thane, Solapur, Ahmednagar, Kolhapur, Satara, Raigad, Buldhana, Beed, Yavatmal, Chandrapur, Latur, Dhule, Jalna, Parbhani, Akola, Dharashiv, Nandurbar, Ratnagiri, Gondia, Wardha, Bhandara, Washim, Hingoli, Gadchiroli, Sindhudurg, Palghar, Jalgaon, Sangli, Nanded, Amravati |

### Cleaning Operations Applied
1. Column name normalization (lowercase, underscore-separated).
2. Numeric type casting for `accident_count`, `fatal_accidents`, `deaths`, `injuries`.
3. Range validation (year 2000-2030, month 1-12, counts ≥ 0).
4. District name standardization (Osmanabad → Dharashiv, Aurangabad → Chhatrapati Sambhajinagar).
5. Derived features added: `fatality_ratio`, `injury_ratio`, `severity_index`, `season`.

---

## Social Media Perception Dataset Quality Report

| Metric | Value |
|---|---|
| **File** | `maharashtra_road_safety_public_posts.csv` |
| **Rows** | 1,200 |
| **Columns** | 10 |
| **Missing Values** | 0 |
| **Duplicate post_id** | 0 |
| **Empty Texts Removed** | 0 |
| **Invalid Records** | 0 |
| **Valid Records** | 1,200 |
| **Date Range** | January 2021 – December 2023 |
| **Geographic Coverage** | 11 urban centers in Maharashtra |
| **Locations** | Pune, Mumbai, Thane, Nashik, Nagpur, Chhatrapati Sambhajinagar, Solapur, Kolhapur, Navi Mumbai, Ahmednagar, Satara |

### Cleaning Operations Applied
1. Duplicate `post_id` detection and removal.
2. Text whitespace normalization (original text preserved intact).
3. Timestamp parsing and normalization to ISO 8601 format.
4. Engagement score computation: `likes*1 + shares*2 + comments*3`.
5. Location standardization mapping applied.

---

## Official MoRTH Tables Downloaded

| File | Rows | Coverage |
|---|---|---|
| `morth_major_cities_accidents_2024.csv` | 51 | All Million+ Cities 2023-2024 |
| `morth_major_cities_accidents_2022.csv` | 51 | All Million+ Cities 2021-2022 |
| `morth_cities_traffic_violations_2024.csv` | 51 | Traffic violation breakdown 2024 |
| `morth_cities_fatalities_by_mode_2024.csv` | 51 | Road user type fatalities 2024 |
| `morth_statewise_accidents_2020_2024.csv` | 39 | State-wise accidents 2020-2024 |
| `morth_statewise_fatalities_2020_2024.csv` | 39 | State-wise fatalities 2020-2024 |
| `morth_collision_types_2024.csv` | 19 | Collision type distribution 2024 |
