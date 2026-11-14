import pytest

from app import odormeta


@pytest.fixture(autouse=True)
def no_aliases_tab(monkeypatch):
    """The odor sheet's aliases tab is not asked of Google (tests stand in for its CSV alone,
    `fetch_sheet`): there is none, unless a test gives one."""
    monkeypatch.setattr(odormeta, "fetch_aliases_tab", lambda url, timeout=30: None)
