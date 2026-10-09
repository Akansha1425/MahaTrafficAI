-- ==============================================================================
-- MahaTraffic AI: Academic Big Data Hive Script 01
-- Description: Create dedicated database for MahaTraffic intelligence warehouse
-- Status: Academic query specification (No live Hive server required)
-- ==============================================================================

CREATE DATABASE IF NOT EXISTS mahatraffic_warehouse
COMMENT 'Historical road accident and risk intelligence warehouse for Maharashtra'
LOCATION '/user/hive/warehouse/mahatraffic.db';

USE mahatraffic_warehouse;
