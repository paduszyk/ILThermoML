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
    mock_get_entry = mocker.patch("ilthermoml.dataset.GetEntry")

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

    mocker.patch("ilthermoml.dataset.GetEntry", side_effect=mock_get_entry)

    # Act & assert.
    with pytest.raises(EntryError):
        Entry("mock_id")


def test_entry_updates_ilthermo_entry_data_columns_with_header(
    mocker: MockerFixture,
) -> None:
    # Mock.
    mocker.patch(
        "ilthermoml.dataset.GetEntry",
        return_value=mocker.Mock(
            header={"V1": "mock_header"},
            data=pd.DataFrame({"V1": []}),
        ),
    )

    # Act.
    entry = Entry("mock_id")

    # Assert.
    assert entry.data.columns == ["mock_header"]


def test_entry_is_prepared_when_instantiated_with_dataset(
    mocker: MockerFixture,
) -> None:
    # Arrange.
    mock_dataset = mocker.Mock()

    # Mock.
    mocker.patch("ilthermoml.dataset.GetEntry")

    # Act.
    Entry("mock_id", mock_dataset)

    # Assert.
    mock_dataset.prepare_entry.assert_called_once()


def test_dataset_populate_attempts_to_retrieve_entry_ids(
    mocker: MockerFixture,
) -> None:
    # Arrange.
    class TestDataset(Dataset):
        @staticmethod
        def get_entry_ids() -> list[str]:
            return []

        @staticmethod
        def prepare_entry(entry: Entry) -> None:
            pass

    dataset = TestDataset()

    # Mock.
    mocker.patch("ilthermoml.dataset.GetEntry")

    # Spy.
    spy_get_entry_ids = mocker.spy(dataset, "get_entry_ids")

    # Act.
    dataset.populate()

    # Assert.
    spy_get_entry_ids.assert_called_once()


def test_dataset_populate_append_entries_with_ids_retrieved(
    mocker: MockerFixture,
) -> None:
    # Arrange.
    class TestDataset(Dataset):
        @staticmethod
        def get_entry_ids() -> list[str]:
            return ["id_a", "id_b"]

        @staticmethod
        def prepare_entry(entry: Entry) -> None:
            pass

    dataset = TestDataset()

    # Mock.
    mocker.patch("ilthermoml.dataset.GetEntry")

    # Act.
    dataset.populate()
    dataset_entry_ids = [entry.id for entry in dataset.entries]

    # Assert.
    assert dataset_entry_ids == ["id_a", "id_b"]


def test_dataset_populate_skips_entries_that_cannot_be_retrieved(
    mocker: MockerFixture,
) -> None:
    # Arrange.
    class MockDataset(Dataset):
        @staticmethod
        def get_entry_ids() -> list[str]:
            return ["id_a", "id_b", "id_c"]

        @staticmethod
        def prepare_entry(entry: Entry) -> None:
            pass

    dataset = MockDataset()

    # Mock.
    def mock_get_entry(code: str) -> Any:  # noqa: ANN401
        if code == "id_b":
            raise EntryError

        return mocker.Mock()

    mocker.patch("ilthermoml.dataset.GetEntry", side_effect=mock_get_entry)

    # Act.
    dataset.populate()
    dataset_entry_ids = [entry.id for entry in dataset.entries]

    # Assert.
    assert dataset_entry_ids == ["id_a", "id_c"]


def test_dataset_data_returns_joined_entries(
    mocker: MockerFixture,
) -> None:
    # Mock.
    def mock_get_entry(code: str) -> Any:  # noqa: ANN401 def data(self) -> pd.DataFrame: ...
        if code == "id_a":
            return mocker.Mock(
                header={"A": "A"},
                data=pd.DataFrame({"A": [1, 2, 3]}),
            )
        if code == "id_b":
            return mocker.Mock(
                header={"A": "A"},
                data=pd.DataFrame({"A": [4, 5, 6]}),
            )
        return mocker.Mock()

    mocker.patch("ilthermoml.dataset.GetEntry", side_effect=mock_get_entry)

    # Arrange.
    expected_data = pd.DataFrame(
        {"A": [1, 2, 3, 4, 5, 6], "entry_id": [0, 0, 0, 1, 1, 1]}
    )

    class MockDataset(Dataset):
        @staticmethod
        def get_entry_ids() -> list[str]:
            return ["id_a", "id_b"]

        @staticmethod
        def prepare_entry(entry: Entry) -> None:
            pass

    dataset = MockDataset()
    dataset.populate()

    # Act & Assert.
    pd.testing.assert_frame_equal(expected_data, dataset.data)
