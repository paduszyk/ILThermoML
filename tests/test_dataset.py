from __future__ import annotations

import pandas as pd
import pytest

from ilthermoml.dataset import Entry
from ilthermoml.exceptions import EntryError


def test_retrive_entry() -> None:
    # Arrange
    kepte = pd.DataFrame(
        {
            "Temperature, K": {0: 298.15},
            "Pressure, kPa": {0: 101.0},
            "Viscosity, Pa&#8226;s => Liquid": {0: 0.255},
            "Error of viscosity, Pa&#8226;s => Liquid": {0: 0.01},
        }
    )

    entry = Entry("kepte")

    # Assert
    assert entry.data.equals(kepte)


def test_failed_to_retrive_entry() -> None:
    # Assert
    with pytest.raises(EntryError):
        Entry("invalidID")
