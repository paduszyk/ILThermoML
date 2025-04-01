from __future__ import annotations

from typing import TYPE_CHECKING, cast

from ilthermoml.chemistry import Anion, Cation, IonicLiquid

if TYPE_CHECKING:
    from ilthermoml.chemistry import Ion

__all__ = [
    "Dataset",
    "Entry",
]

from abc import ABC, abstractmethod
from dataclasses import InitVar, dataclass, field

import ilthermopy as ilt
import pandas as pd
from tqdm import tqdm

from .exceptions import ChemistryError, DatasetError, EntryError
from .memory import ilt_memory

GetEntry = ilt_memory.cache(ilt.GetEntry)


@dataclass
class Entry:
    """Represents a single entry in the dataset."""

    id: str
    """The identifier of the entry."""

    ionic_liquid: IonicLiquid = field(init=False, repr=False)
    """The ionic liquid associated with the entry."""

    data: pd.DataFrame = field(init=False, repr=False)
    """The data associated with the entry."""

    dataset: InitVar[Dataset | None] = None
    """The dataset to which this entry belongs."""

    def __post_init__(self, dataset: Dataset | None) -> None:
        """Initialize the entry by retrieving data from ILThermo.

        Args:
            dataset: The dataset to which this entry belongs.

        Raises:
            EntryError: If the entry cannot be retrieved from ILThermo.
        """
        try:
            ilt_entry = GetEntry(self.id)
        except Exception as e:
            msg = f"failed to retrieve ILThermo entry {self.id!r}"

            raise EntryError(msg) from e

        if len(ilt_entry.components) > 1:
            msg = "entries with multiple components are not supported"

            raise EntryError(msg)

        self.data = ilt_entry.data.copy().rename(columns=ilt_entry.header)
        self.ionic_liquid_id = ilt_entry.components[0].id

        if smiles_error := ilt_entry.components[0].smiles_error:
            msg = f"entry {self.id!r} has no smiles: {smiles_error}"

            raise EntryError(msg)

        try:
            self.ionic_liquid = IonicLiquid(ilt_entry.components[0].smiles)
        except ChemistryError as e:
            msg = f"invalid smiles {ilt_entry.components[0].smiles}"

            raise EntryError(msg) from e

        if dataset:
            dataset.prepare_entry(self)


@dataclass
class Dataset(ABC):
    """Abstract base class for datasets."""

    entries: list[Entry] = field(default_factory=list, init=False, repr=False)
    """The list of entries in the dataset."""

    ionic_liquids: list[IonicLiquid] = field(
        default_factory=list, init=False, repr=False
    )

    ions: list[Ion | Cation | Anion] = field(
        default_factory=list, init=False, repr=False
    )

    @property
    def data(self) -> pd.DataFrame:
        """Concatenate and return the data from all entries in the dataset.

        Returns:
            The concatenated data from all entries.

        Raises:
            DatasetError: If the dataset is empty.
        """
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
        """Get the list of entry IDs.

        Returns:
            The list of entry IDs.
        """

    @staticmethod
    @abstractmethod
    def prepare_entry(entry: Entry) -> None:
        """Prepare an entry.

        Args:
            entry: The entry to prepare.
        """

    def populate(self) -> None:
        """Populate the dataset with entries."""
        entry_ids = self.get_entry_ids()

        for entry_id in tqdm(entry_ids, desc="Populating dataset"):
            try:
                entry = Entry(entry_id, dataset=self)
            except EntryError:
                continue

            if entry.ionic_liquid not in self.ionic_liquids:
                if entry.ionic_liquid.cation not in self.ions:
                    self.ions.append(entry.ionic_liquid.cation)
                else:
                    entry.ionic_liquid.cation = cast(
                        Cation, self.ions[self.ions.index(entry.ionic_liquid.cation)]
                    )

                if entry.ionic_liquid.anion not in self.ions:
                    self.ions.append(entry.ionic_liquid.anion)
                else:
                    entry.ionic_liquid.anion = cast(
                        Anion, self.ions[self.ions.index(entry.ionic_liquid.anion)]
                    )

                self.ionic_liquids.append(entry.ionic_liquid)
            else:
                entry.ionic_liquid = self.ionic_liquids[
                    self.ionic_liquids.index(entry.ionic_liquid)
                ]

            self.entries.append(entry)
