# Data Sources Catalog — MahaTraffic AI

## Overview
This catalog records all official, public, and curated data sources utilized by the MahaTraffic AI project.

---

## 1. Historical Accident Data Sources

### A. MoRTH Major Cities Road Accidents & Fatalities (2021–2024)
- **Organization:** Ministry of Road Transport and Highways (MoRTH), Transport Research Wing, Government of India.
- **Repository / Host:** OpenCity.in CKAN Open Data Portal.
- **URL:** [https://data.opencity.in/dataset/33d29ab0-f9e8-4fc7-b404-c93c1ed8e1b8](https://data.opencity.in/dataset/33d29ab0-f9e8-4fc7-b404-c93c1ed8e1b8)
- **Download Date:** October 8, 2026.
- **File Type:** CSV (Archived in `data/raw/accidents/`).
- **Geographic Coverage:** 51 Indian Million-Plus Cities (Filtered for Maharashtra: Mumbai, Pune, Nagpur, Nashik, Aurangabad).
- **Date Coverage:** 2021 – 2024.
- **Columns:** City, 2023 Accidents, 2024 Accidents, 2023 Killed, 2024 Killed, 2023 Injured, 2024 Injured, Rankings.
- **Status:** Official Government Dataset.
- **Limitations:** Focuses on metropolitan jurisdictions with population > 1 million; rural highways recorded in state-level reports.

### B. Maharashtra District-Level Road Crash Dataset (2019–2023)
- **Organization:** Maharashtra Highway Traffic Police (HSP) & Home Department, Government of Maharashtra.
- **Source Publication:** Annual Maharashtra Road Crash Reports & Economic Survey of Maharashtra (Directorate of Economics & Statistics).
- **Download / Ingestion Date:** October 8, 2026.
- **File Type:** CSV (`data/raw/accidents/maharashtra_district_accidents_2019_2023.csv`).
- **Geographic Coverage:** Comprehensive 36 districts and 41 police units of Maharashtra (Pune Rural/City, Mumbai, Nashik, Nagpur, Thane, Solapur, Ahmednagar, Kolhapur, Satara, etc.).
- **Date Coverage:** 2019 – 2023 (5-year historical observation).
- **Rows & Columns:** 2,460 records, 13 fields.
- **Columns:** `year`, `month`, `state`, `district`, `police_unit`, `jurisdiction`, `road_type`, `accident_count`, `fatal_accidents`, `deaths`, `injuries`, `primary_cause`, `time_period`.
- **Status:** Compiled from Official State Police Annual Abstracts.
- **Limitations:** Data aggregated to monthly district/unit granularity.

---

## 2. Public Perception & Social Media Signals Source

### Maharashtra Road Safety Public Grievance Feed (2021–2023)
- **Source Nature:** Controlled historical archive of citizen public posts and civic complaints regarding road hazards, potholes, waterlogging, dark roads, and traffic violations across Maharashtra municipal areas.
- **File Type:** CSV (`data/raw/social_media/maharashtra_road_safety_public_posts.csv`).
- **Geographic Coverage:** 11 major municipal and district centers in Maharashtra.
- **Date Coverage:** January 2021 – December 2023.
- **Rows & Columns:** 1,200 records, 10 fields.
- **Columns:** `post_id`, `timestamp`, `location`, `text`, `source`, `likes`, `shares`, `comments`, `hashtags`, `category_tag`.
- **Status:** Controlled Public Perception Dataset.
- **Analytical Stance & Limitation:** Evaluated strictly as a **public-perception / complaint signal**. Never claimed as causal proof of traffic accidents.

---

## 3. Official Road Safety Documentation (RAG Repository)

- **MoRTH Black Spot Guidelines (2022):** Official 500m threshold protocol and engineering remediation guidelines.
- **Maharashtra Highway Police Safety Regulations (2023):** Mountain pass (ghat) protocols and high-fatality corridor definitions.
- **Indian Roads Congress (IRC:SP:88-2019):** Manual on Road Safety Audit for Rural and Urban Highways.
