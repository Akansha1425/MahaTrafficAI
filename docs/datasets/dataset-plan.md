# Dataset Acquisition and Schema Plan — MahaTraffic AI

## Purpose
This document outlines the planned discovery, validation, and storage strategy for datasets utilized by the MahaTraffic AI system.

> **Status:** Phase 1 Initialization (No datasets downloaded yet. Scheduled for Phase 2).

## Planned Datasets

### 1. Historical Road Accident Data (Maharashtra)
- **Planned Sources:** Official open government data portals (Open Government Data - data.gov.in, Maharashtra Police Crime & Accident reports, Ministry of Road Transport and Highways - MoRTH).
- **Provisional Schema Fields:**
  - `state` (str)
  - `city` / `district` (str)
  - `year` (int)
  - `month` (int)
  - `time_period` (str)
  - `accident_count` (int)
  - `fatal_accidents` (int)
  - `serious_accidents` (int)
  - `minor_accidents` (int)
  - `deaths` (int)
  - `injuries` (int)
  - `road_type` (str)
  - `road_condition` (str)
  - `weather` (str)
  - `cause` (str)

### 2. Public Perception & Social Media Signals
- **Planned Nature:** Controlled public civic complaint and road perception samples (e.g. municipal grievance feeds, civic posts).
- **Provisional Schema Fields:**
  - `post_id` (str)
  - `timestamp` (datetime)
  - `location` (str)
  - `text` (str)
  - `source` (str)
  - `likes` / `shares` / `comments` (int)
  - `hashtags` (list)
  - `sentiment` (str)
- **Analytical Stance:** Evaluated strictly as a **public-perception / complaint signal**, not as direct causal evidence for traffic accidents.

### 3. Road Safety Guidelines & MoRTH Manuals
- **Planned Nature:** Official road safety manuals, IRC (Indian Roads Congress) standards, and accident prevention policy documents for RAG indexing.

## Storage Hierarchy
- `data/raw/`: Original, unmodified files.
- `data/processed/parquet/`: Columnar, compressed format partitioned for Spark.
- `data/sample/`: Small representative snippets for fast unit tests.
