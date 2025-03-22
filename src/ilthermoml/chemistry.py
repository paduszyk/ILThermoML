from __future__ import annotations

__all__ = [
    "Molecule",
]

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import NamedTuple, override

from rdkit import Chem

from .exceptions import (
    InvalidChargeError,
    IonicLiquidCationError,
    UnsupportedSaltTypeError,
)


@dataclass
class Molecule(ABC):
    """Abstract base class for molecules.

    This class basically wraps an RDKit molecule object to provide some additional
    functionality and checks.
    """

    smiles: str
    """The SMILES representation of the molecule."""

    _rdkit_mol: Chem.Mol = field(init=False, repr=False, compare=False)
    """The wrapped RDKit molecule object."""

    @abstractmethod
    def post_init_check(self) -> None:
        """A hook to perform additional checks after initialization."""

    def __post_init__(self) -> None:
        """Initialize the RDKit molecule and perform post-initialization checks."""
        self._rdkit_mol = Chem.MolFromSmiles(self.smiles)

        # SMILES is reassigned to ensure that it is canonical. Stereochemistry
        # and E/Z isomerism are also discarded.
        self.smiles = Chem.MolToSmiles(self._rdkit_mol, isomericSmiles=False)

        # Perform additional checks.
        self.post_init_check()

    # NOTE: Other relevant properties from the RDKIT `Mol` object can be wrapped here.

    def is_organic(self) -> bool:
        """Return `True` if the molecule is organic, `False` otherwise."""
        return any(
            atom.GetSymbol() == "C"
            for atom in self._rdkit_mol.GetAtoms()  # type: ignore[no-untyped-call]
            if atom.GetAtomicNum()
        )

    @property
    def charge_number(self) -> int:
        """Return the formal charge of the molecule."""
        return Chem.GetFormalCharge(self._rdkit_mol)


@dataclass
class Ion(Molecule):
    """Represent an ion, i.e. a charged molecule."""

    @override
    def post_init_check(self) -> None:
        if self.charge_number == 0:
            msg = "ions must have a non-zero charge"

            raise InvalidChargeError(msg)


@dataclass
class Cation(Ion):
    """Represents a cation, i.e. a positively charged ion."""

    @override
    def post_init_check(self) -> None:
        super().post_init_check()

        if not self.charge_number > 0:
            msg = "cations must have a positive charge"

            raise InvalidChargeError(msg)


@dataclass
class Anion(Ion):
    """Represents an anion, i.e. a negatively charged ion."""

    @override
    def post_init_check(self) -> None:
        super().post_init_check()

        if not self.charge_number < 0:
            msg = "anions must have a negative charge"

            raise InvalidChargeError(msg)


class Stoichiometry(NamedTuple):
    """Represents the stoichiometry of a salt."""

    cation: int
    """The number of cations in the salt."""

    anion: int
    """The number of anions in the salt."""


@dataclass
class Salt:
    """Represents a salt composed of a cation and an anion."""

    smiles: str
    """The SMILES representation of the salt."""

    cation: Cation = field(init=False, repr=False)
    """The cation of the salt."""

    anion: Anion = field(init=False, repr=False)
    """The anion of the salt."""

    def __post_init__(self) -> None:
        """Initialize the cation and anion from the SMILES string."""
        try:
            cation_smiles, anion_smiles = set(smiles_codes := self.smiles.split("."))
        except ValueError as e:
            msg = (
                f"salts must contain exactly one type of both cation and anion; "
                f"found {len(smiles_codes)} type(s)"
            )

            raise UnsupportedSaltTypeError(msg) from e

        if any(
            [
                (cation := Ion(cation_smiles)).charge_number < 0,
                (anion := Ion(anion_smiles)).charge_number > 0,
            ]
        ):
            cation, anion = anion, cation

        self.cation = Cation(cation.smiles)
        self.anion = Anion(anion.smiles)

    @property
    def stoichiometry(self) -> Stoichiometry:
        """Return the stoichiometry of the salt."""
        lcm = math.lcm(
            z_cation := abs(self.cation.charge_number),
            z_anion := abs(self.anion.charge_number),
        )

        return Stoichiometry(cation=lcm // z_cation, anion=lcm // z_anion)


@dataclass
class IonicLiquid(Salt):
    """Represents an ionic liquid."""

    def __post_init__(self) -> None:
        super().__post_init__()

        if not self.cation.is_organic():
            msg = "cations in ionic liquids must be organic"

            raise IonicLiquidCationError(msg)
