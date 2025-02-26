from __future__ import annotations

__all__ = [
    "Dataset",
    "Entry",
]

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field
from typing import TYPE_CHECKING

import ilthermopy as ilt

from .exceptions import EntryError

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Entry:
    id: str
    data: pd.DataFrame = field(init=False, repr=False)

    dataset: InitVar[Dataset | None] = None

    def __post_init__(self, dataset: Dataset | None) -> None:
        try:
            ilt_entry = ilt.GetEntry(self.id)
        except Exception as e:
            msg = f"failed to retrieve ILThermo entry {self.id!r}"

            raise EntryError(msg) from e

        self.data = ilt_entry.data.copy().rename(columns=ilt_entry.header)

        if dataset:
            dataset.prepare_entry(self)


@dataclass
class Dataset(ABC):
    entries: list[Entry] = field(init=False, repr=False)

    @staticmethod
    @abstractmethod
    def get_entry_ids() -> list[str]:
        pass

    @staticmethod
    @abstractmethod
    def prepare_entry(entry: Entry) -> None:
        pass

    def populate(self) -> None:
        entry_ids = self.get_entry_ids()

        entries: list[Entry] = []
        for entry_id in entry_ids:
            try:
                entry = Entry(entry_id, dataset=self)
            except EntryError:
                continue

            entries.append(entry)

        self.entries = entries
