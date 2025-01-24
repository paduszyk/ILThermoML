from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING

import pytest
from semver import Version

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def package() -> ModuleType | None:
    try:
        return import_module("ilthermoml")
    except ModuleNotFoundError:
        pytest.fail("importing 'ilthermoml' package failed unexpectedly")

    return None


def test_is_importable(package: ModuleType) -> None:
    pass


def test_has_version(package: ModuleType) -> None:
    # Assert.
    assert hasattr(package, "__version__")


def test_version_follows_semver(package: ModuleType) -> None:
    # Arrange.
    version = package.__version__

    # Assert.
    assert Version.is_valid(version)
