-- ==============================================================================
-- MahaTraffic AI: Academic Big Data Hive Script 04
-- Description: District-level crash risk and fatality distribution query
-- ==============================================================================

USE mahatraffic_warehouse;

SELECT 
    district,
    SUM(accident_count) AS total_accidents,
    SUM(deaths) AS total_deaths,
    SUM(injuries) AS total_injuries,
    ROUND(SUM(deaths) * 1.0 + SUM(injuries) * 0.3, 2) AS raw_severity_score,
    DENSE_RANK() OVER (ORDER BY (SUM(deaths) * 1.0 + SUM(injuries) * 0.3) DESC) AS analytical_rank
FROM maharashtra_accidents
GROUP BY district
ORDER BY analytical_rank ASC
LIMIT 10;
