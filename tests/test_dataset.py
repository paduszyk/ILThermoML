from __future__ import annotations

from dataclasses import dataclass
from unittest import mock

import pandas as pd
import pytest

from ilthermoml.dataset import Entry
from ilthermoml.exceptions import EntryError


def test_retrive_entry() -> None:
    # Arrange
    @dataclass()
    class MockIlthermopyEntry:
        header: dict[str, str]
        data: pd.DataFrame

    result = pd.DataFrame({"Replacement": [10]})

    with mock.patch(
        "ilthermopy.GetEntry",
        return_value=MockIlthermopyEntry(
            header={"To_be_replaced": "Replacement"},
            data=pd.DataFrame({"To_be_replaced": [10]}),
        ),
    ) as mock_get_entry:
        entry = Entry("id")

    # Assert
    mock_get_entry.assert_called_once_with("id")
    assert entry.data.equals(result)


def test_failed_to_retrive_entry() -> None:
    # Assert
    with (
        mock.patch("ilthermopy.GetEntry", mock.Mock(side_effect=Exception)),
        pytest.raises(EntryError),
    ):
        Entry("invalidID")
