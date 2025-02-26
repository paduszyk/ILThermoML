from __future__ import annotations

__all__ = [
    "Entry",
]

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import ilthermopy as ilt

from .exceptions import EntryError

if TYPE_CHECKING:
    import pandas as pd


@dataclass
class Entry:
    id: str
    data: pd.DataFrame = field(init=False, repr=False)

    def __post_init__(self) -> None:
        try:
            ilt_entry = ilt.GetEntry(self.id)
        except Exception as e:
            msg = f"failed to retrieve ILThermo entry {self.id!r}"

            raise EntryError(msg) from e

        self.data = ilt_entry.data.copy().rename(columns=ilt_entry.header)
