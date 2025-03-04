from __future__ import annotations

from typing import TYPE_CHECKING, Any

import pandas as pd
import pytest

from ilthermoml.dataset import Dataset, Entry
from ilthermoml.exceptions import EntryError

if TYPE_CHECKING:
    from pytest_mock import MockerFixture


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
    mocker: MockerFixture,
) -> None:
    # Mock.
    class MockError(Exception):
        pass

    def mock_get_entry(code: str) -> Any:  # noqa: ANN401
        if code == "mock_id":
            raise MockError

    mocker.patch("ilthermopy.GetEntry", side_effect=mock_get_entry)

    # Act & assert.
    with pytest.raises(EntryError):
        Entry("mock_id")


def test_entry_updates_ilthermo_entry_data_columns_with_header(
    mocker: MockerFixture,
) -> None:
    # Mock.
    mocker.patch(
        "ilthermopy.GetEntry",
        return_value=mocker.Mock(
            header={"V1": "mock_header"},
            data=pd.DataFrame({"V1": []}),
        ),
    )

    # Act.
    entry = Entry("mock_id")

    # Assert.
    assert entry.data.columns == ["mock_header"]


def test_entry_attempts_to_prepare_entry(mocker: MockerFixture) -> None:
    # Mock.
    mocker.patch("ilthermopy.GetEntry")
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
    dataset: DatasetTest, mocker: MockerFixture
) -> None:
    # Mock.
    class MockError(Exception):
        pass

    def mock_get_entry(code: str) -> Any:  # noqa: ANN401
        if code == "mock_id":
            raise MockError

    mocker.patch("ilthermopy.GetEntry", side_effect=mock_get_entry)

    # Act & Assert.
    dataset.populate()
    assert len(dataset.entries) == 0
