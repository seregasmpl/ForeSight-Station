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
# tests/conftest_playwright.py
import time
import httpx
import pytest

BASE_URL = "http://localhost:8001"
ALL_CODES = ["ЛУНА-01", "ЛУНА-02", "ОРБИТА-01", "ОРБИТА-02",
             "ЗВЕЗДА-01", "ЗВЕЗДА-02", "ЗЕМЛЯ-01", "ЗЕМЛЯ-02"]


@pytest.fixture(scope="session")
def live_server():
    """Assumes server is already running on 8001. Just checks availability."""
    for _ in range(10):
        try:
            httpx.get(f"{BASE_URL}/", timeout=2)
            return BASE_URL
        except (httpx.TimeoutException, httpx.ConnectError, httpx.RequestError):
            time.sleep(1)
    raise RuntimeError("Server not available on port 8001")


@pytest.fixture()
def fresh_sessions(live_server):
    """Resets all sessions before each test."""
    try:
        httpx.post(f"{live_server}/api/sessions/reset-all", timeout=10)
    except Exception as e:
        pytest.fail(f"Failed to reset sessions: {e}")
    return live_server


@pytest.fixture()
def joined_session(fresh_sessions):
    """Creates and activates ЛУНА-01 session, returns (base_url, session_data)."""
    response = httpx.post(
        f"{fresh_sessions}/api/sessions/join",
        json={"code": "ЛУНА-01"},
        timeout=10,
    )
    response.raise_for_status()
    data = response.json()
    return fresh_sessions, data
