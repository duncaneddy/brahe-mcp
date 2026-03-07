import brahe
import pytest
from pathlib import Path


# Initialize EOP and space weather data once for the entire test session.
# Production code does this in server.py; tests need it before importing
# any module that uses brahe frame transforms or propagation.
@pytest.fixture(autouse=True, scope="session")
def _init_brahe():
    brahe.initialize_eop()
    brahe.initialize_sw()


@pytest.fixture
def tmp_db_path(tmp_path: Path) -> Path:
    return tmp_path / "test.db"
