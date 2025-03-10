from __future__ import annotations

__all__ = [
    "Dataset",
    "Entry",
]

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field

import ilthermopy as ilt
import pandas as pd
from tqdm import tqdm

from .exceptions import DatasetError, EntryError
from .memory import ilt_memory

GetEntry = ilt_memory.cache(ilt.GetEntry)


@dataclass
class Entry:
    id: str
    data: pd.DataFrame = field(init=False, repr=False)

    dataset: InitVar[Dataset | None] = None

    def __post_init__(self, dataset: Dataset | None) -> None:
        try:
            ilt_entry = GetEntry(self.id)
        except Exception as e:
            msg = f"failed to retrieve ILThermo entry {self.id!r}"

            raise EntryError(msg) from e

        self.data = ilt_entry.data.copy().rename(columns=ilt_entry.header)

        if dataset:
            dataset.prepare_entry(self)


@dataclass
class Dataset(ABC):
    entries: list[Entry] = field(default_factory=list, init=False, repr=False)

    @property
    def data(self) -> pd.DataFrame:
        if entries := self.entries:
            return pd.concat(
                {entry.id: entry.data for entry in entries},
                names=["entry_id", "data_point_id"],
            )

        msg = "dataset is empty"

        raise DatasetError(msg)

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

        for entry_id in tqdm(entry_ids, desc="Populating dataset"):
            try:
                entry = Entry(entry_id, dataset=self)
            except EntryError:
                continue

            self.entries.append(entry)
