"""Test suite for application configuration."""

from backend.app.config import get_settings


def test_settings_defaults():
    """Verify default configuration parameters."""
    settings = get_settings()
    assert settings.APP_NAME == "MahaTraffic AI"
    assert settings.API_V1_PREFIX == "/api/v1"
    assert len(settings.allowed_tools_list) > 0
    assert "get_city_accident_statistics" in settings.allowed_tools_list
