import os
import pytest

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")


@pytest.fixture
def fixtures_dir():
    return FIXTURES_DIR


@pytest.fixture
def optimist_dir():
    return os.path.join(FIXTURES_DIR, "optimist")


@pytest.fixture
def pessimist_dir():
    return os.path.join(FIXTURES_DIR, "pessimist")
