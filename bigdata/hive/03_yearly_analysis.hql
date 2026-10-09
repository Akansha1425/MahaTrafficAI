-- ==============================================================================
-- MahaTraffic AI: Academic Big Data Hive Script 03
-- Description: Multi-year analytical aggregation query
-- ==============================================================================

USE mahatraffic_warehouse;

SELECT 
    year,
    SUM(accident_count) AS total_accidents,
    SUM(fatal_accidents) AS total_fatal_accidents,
    SUM(deaths) AS total_fatalities,
    SUM(injuries) AS total_injuries,
    ROUND((SUM(deaths) / SUM(accident_count)) * 100, 2) AS fatality_percentage
FROM maharashtra_accidents
GROUP BY year
ORDER BY year ASC;
