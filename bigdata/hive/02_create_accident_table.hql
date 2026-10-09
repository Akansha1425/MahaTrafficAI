-- ==============================================================================
-- MahaTraffic AI: Academic Big Data Hive Script 02
-- Description: Create external Parquet-backed table partitioned by year
-- ==============================================================================

USE mahatraffic_warehouse;

CREATE EXTERNAL TABLE IF NOT EXISTS maharashtra_accidents (
    month INT,
    state STRING,
    district STRING,
    police_unit STRING,
    jurisdiction STRING,
    road_type STRING,
    accident_count INT,
    fatal_accidents INT,
    deaths INT,
    injuries INT,
    primary_cause STRING,
    time_period STRING,
    fatality_ratio DOUBLE,
    injury_ratio DOUBLE,
    severity_index DOUBLE,
    season STRING
)
PARTITIONED BY (year INT)
STORED AS PARQUET
LOCATION '/mahatraffic/processed/parquet/accidents/';

-- Synchronize partition metadata
MSCK REPAIR TABLE maharashtra_accidents;
