from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import pandas as pd
import pytest

from ilthermoml.dataset import Entry
from ilthermoml.exceptions import EntryError

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


def test_retrive_entry(mocker: MockerFixture) -> None:
    # Arrange
    @dataclass()
    class MockIlthermopyEntry:
        header: dict[str, str]
        data: pd.DataFrame

    result = pd.DataFrame({"Replacement": [10]})
    mock_get_entry = mocker.patch(
        "ilthermopy.GetEntry",
        return_value=MockIlthermopyEntry(
            header={"To_be_replaced": "Replacement"},
            data=pd.DataFrame({"To_be_replaced": [10]}),
        ),
    )

    # Act
    entry = Entry("id")

    # Assert
    mock_get_entry.assert_called_once_with("id")
    assert entry.data.equals(result)


def test_failed_to_retrive_entry(mocker: MockerFixture) -> None:
    # Assert
    mocker.patch("ilthermopy.GetEntry", mocker.Mock(side_effect=Exception))
    with pytest.raises(EntryError):
        Entry("invalidID")
