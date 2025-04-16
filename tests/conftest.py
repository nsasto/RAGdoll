import pytest
import logging

logging.getLogger("ragdoll.config.config_manager").setLevel(logging.ERROR)

@pytest.fixture
def sample_fixture():
    return "sample data"