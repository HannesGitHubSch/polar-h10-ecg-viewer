"""
Pytest configuration and shared fixtures.
"""

import sys
from pathlib import Path

import pytest

# Ensure repository root is on sys.path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from polar_ecg.settings import Settings, load_settings


@pytest.fixture
def settings() -> Settings:
    """Fixture providing default settings instance."""
    return load_settings()
