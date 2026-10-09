# Apache Hive — Academic Query Architecture

## Overview
This directory contains HiveQL (`.hql`) DDL scripts and analytical queries illustrating how the MahaTraffic AI Parquet data lake maps into an **Apache Hive** distributed data warehouse.

> ⚠️ **Academic Status:**  
> Labeled as **Academic Big Data Query Demonstrations**.  
> Hive is **NOT** a required runtime dependency. Hive execution is not fabricated; these queries illustrate enterprise data warehouse compatibility for academic examination and vivas.

## Script Catalog
1. [01_create_database.hql](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/bigdata/hive/01_create_database.hql): Initializes database namespace.
2. [02_create_accident_table.hql](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/bigdata/hive/02_create_accident_table.hql): Defines external Parquet schema partitioned by `year`.
3. [03_yearly_analysis.hql](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/bigdata/hive/03_yearly_analysis.hql): Aggregates accident counts, fatalities, and injury percentages.
4. [04_location_analysis.hql](file:///c:/Users/nikit/OneDrive/Pictures/Documents/MahaTrafficAI/bigdata/hive/04_location_analysis.hql): Computes severity scores and window ranking functions.
