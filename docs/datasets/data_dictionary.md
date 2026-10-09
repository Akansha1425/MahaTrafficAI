# Data Dictionary — MahaTraffic AI

## 1. Processed Accident Records Schema
**Storage File:** `data/processed/parquet/accidents/maharashtra_accidents_clean.parquet`

| Field Name | Type | Description | Values / Range |
|---|---|---|---|
| `year` | Integer | Observation year | `2019 – 2023` |
| `month` | Integer | Month of occurrence | `1 – 12` |
| `state` | String | State name | `Maharashtra` |
| `district` | String | Standardized district name | `Pune`, `Mumbai`, `Nashik`, `Nagpur`, etc. |
| `police_unit` | String | Specific police administration unit | e.g., `Pune Rural`, `Pune City` |
| `jurisdiction` | String | Administrative boundary classification | `City`, `Rural` |
| `road_type` | String | Road network classification | `National Highway`, `State Highway`, `Urban / City Road`, `Urban / Expressway` |
| `accident_count` | Integer | Total recorded crash incidents in interval | `>= 0` |
| `fatal_accidents`| Integer | Incidents resulting in at least 1 death | `<= accident_count` |
| `deaths` | Integer | Total deceased casualties | `>= fatal_accidents` |
| `injuries` | Integer | Total non-fatal injured individuals | `>= 0` |
| `primary_cause` | String | Primary contributing factor | `Over-speeding`, `Dangerous Overtaking`, `Drunk Driving`, `Adverse Weather / Monsoon`, `Road Defect / Pothole` |
| `time_period` | String | Diurnal time window | `Day (06:00-18:00)`, `Night (18:00-06:00)` |
| `fatality_ratio`| Float | Computed ratio: `deaths / accident_count` | `0.0 – 1.0` |
| `injury_ratio` | Float | Computed ratio: `injuries / accident_count` | `0.0 – 2.0` |
| `severity_index`| Float | Weighted index: `(deaths*1.0 + injuries*0.3) / accidents` | Continuous |
| `season` | String | Seasonal meteorological category | `Monsoon`, `Post-Monsoon`, `Winter`, `Summer` |

---

## 2. Processed Social Media Perception Schema
**Storage File:** `data/processed/parquet/social_media/social_features.parquet`

| Field Name | Type | Description | Values / Range |
|---|---|---|---|
| `post_id` | String | Unique post identifier | `MAHA_SOC_00001` ... |
| `timestamp` | Datetime | Publication timestamp | ISO 8601 string |
| `location` | String | Identified urban center | `Pune`, `Mumbai`, `Nashik`, `Thane`, etc. |
| `original_text` | String | Complete unaltered post content | Text string |
| `clean_text` | String | Whitespace-normalized text | Text string |
| `source` | String | Civic platform / channel | `Twitter/X Public Archive`, `Citizen Grievance Portal`, etc. |
| `likes` | Integer | Interaction reaction count | `>= 0` |
| `shares` | Integer | Forward / amplification count | `>= 0` |
| `comments` | Integer | Civic feedback replies count | `>= 0` |
| `hashtags` | String | Semicolon-delimited hashtags | e.g. `#PotholeAlert;#RoadSafety` |
| `category_tag` | String | Initial civic problem category | `potholes`, `waterlogging`, `speeding`, etc. |
| `engagement_score` | Integer | Computed interaction weight: `likes + shares*2 + comments*3` | `>= 0` |
| `pothole_signal` | Integer | Binary keyword indicator | `0` or `1` |
| `monsoon_flood_signal` | Integer | Binary keyword indicator | `0` or `1` |
| `speeding_signal` | Integer | Binary keyword indicator | `0` or `1` |
| `infrastructure_signal`| Integer | Binary keyword indicator | `0` or `1` |
