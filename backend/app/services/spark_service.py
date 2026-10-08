"""PySpark analytics service interface."""

from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


class SparkService:
    """Service facade for PySpark computations, Parquet queries, and aggregations."""

    def __init__(self, master: str = "local[*]", app_name: str = "MahaTrafficAI-Spark"):
        self.master = master
        self.app_name = app_name
        self._spark_session = None

    def get_session(self):
        """Lazy-initialize and return active SparkSession.

        Will be implemented in Phase 5.
        """
        raise NotImplementedError("SparkSession initialization scheduled for Phase 5.")

    def query_yearly_accidents(self) -> List[Dict[str, Any]]:
        """Query aggregated yearly metrics from Parquet store."""
        raise NotImplementedError("Spark yearly query scheduled for Phase 5.")

    def query_monthly_accidents(self) -> List[Dict[str, Any]]:
        """Query monthly distribution metrics from Parquet store."""
        raise NotImplementedError("Spark monthly query scheduled for Phase 5.")

    def query_district_accidents(self, district: Optional[str] = None) -> List[Dict[str, Any]]:
        """Query district/city-level metrics from Parquet store."""
        raise NotImplementedError("Spark district query scheduled for Phase 5.")
