from __future__ import annotations

from typing import TYPE_CHECKING

import pandas as pd
import pytest

from ilthermoml.dataset import Dataset, Entry
from ilthermoml.exceptions import EntryError

if TYPE_CHECKING:
    from collections.abc import Callable

    from pytest_mock import MockerFixture


class MockError(Exception):
    pass


@pytest.fixture
def create_mock_entry(mocker: MockerFixture) -> Callable[..., None]:
    def setup_mock(
        error: Exception | None = None,
        header: dict[str, str] | None = None,
        data: pd.DataFrame | None = None,
    ) -> None:
        header = header or {"V1": "mock_header"}
        data = data or pd.DataFrame({"V1": []})
        if error:
            mocker.patch("ilthermopy.GetEntry", side_effect=error)
        else:
            mock = mocker.Mock(header=header, data=data)
            mocker.patch("ilthermopy.GetEntry", return_value=mock)

    return setup_mock


def test_entry_attempts_to_retrieve_entry_from_ilthermo(
    mocker: MockerFixture,
) -> None:
    # Mock.
    mock_get_entry = mocker.patch("ilthermopy.GetEntry")

    # Act.
    Entry("mock_id")

    # Assert.
    mock_get_entry.assert_called_once_with("mock_id")


def test_entry_raises_entry_error_if_ilthermo_entry_cannot_be_retrieved(
    create_mock_entry: MockerFixture.Mock,
) -> None:
    # Mock.
    create_mock_entry(error=MockError)

    # Act & assert.
    with pytest.raises(EntryError):
        Entry("mock_id")


def test_entry_updates_ilthermo_entry_data_columns_with_header(
    create_mock_entry: MockerFixture.Mock,
) -> None:
    # Mock.
    create_mock_entry()

    # Act.
    entry = Entry("mock_id")

    # Assert.
    assert entry.data.columns == ["mock_header"]


def test_entry_attempts_to_prepare_entry(
    mocker: MockerFixture, create_mock_entry: MockerFixture.Mock
) -> None:
    # Mock.
    create_mock_entry()
    mock_dataset = mocker.Mock()

    # Act.
    Entry("mock_id", mock_dataset)

    # Assert.
    mock_dataset.prepare_entry.assert_called_once()


class DatasetTest(Dataset):
    @staticmethod
    def get_entry_ids() -> list[str]:
        return ["mock_id"]

    @staticmethod
    def prepare_entry(entry: Entry) -> None:
        _ = entry


@pytest.fixture
def dataset() -> DatasetTest:
    return DatasetTest()


def test_dataset_attempts_to_retrive_ids(
    dataset: DatasetTest, mocker: MockerFixture
) -> None:
    # Mock.
    mock_get_entry_ids = mocker.patch.object(dataset, "get_entry_ids", return_value=[])

    # Act.
    dataset.populate()

    # Assert.
    mock_get_entry_ids.assert_called_once()


def test_dataset_attempts_to_retrive_entries(
    dataset: DatasetTest, mocker: MockerFixture
) -> None:
    # Mock.
    mock_entry = mocker.patch("ilthermoml.dataset.Entry", return_value="Mock_entry")

    # Act.
    dataset.populate()

    # Assert.
    assert len(dataset.entries) == 1
    mock_entry.assert_called_once()


def test_dataset_skips_on_entryerror(
    dataset: DatasetTest,
    create_mock_entry: MockerFixture.Mock,
) -> None:
    # Mock.
    create_mock_entry(error=MockError)

    # Act.
    dataset.populate()

    # Assert.

    assert len(dataset.entries) == 0
