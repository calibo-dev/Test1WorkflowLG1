from __future__ import annotations

import pytest

from latest_ai_development.config.settings import (
    get_graph_config,
    get_models_config,
    get_settings,
)


@pytest.fixture(autouse=True)
def clear_cached_settings():
    get_settings.cache_clear()
    get_models_config.cache_clear()
    get_graph_config.cache_clear()
    yield
    get_settings.cache_clear()
    get_models_config.cache_clear()
    get_graph_config.cache_clear()
