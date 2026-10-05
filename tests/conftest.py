from __future__ import annotations

import pytest

from xaglab.data.panel import load_default_panel, load_raw
from xaglab.data.store import Snapshot
from xaglab.features.build import build_features


@pytest.fixture(scope="session")
def prices() -> Snapshot:
    return Snapshot.open("dissertation_v1")


@pytest.fixture(scope="session")
def calendar() -> Snapshot:
    return Snapshot.open("calendar_v1")


@pytest.fixture(scope="session")
def raw(prices):
    return load_raw(prices)


@pytest.fixture(scope="session")
def panel():
    return load_default_panel()


@pytest.fixture(scope="session")
def features(panel):
    return build_features(panel)
