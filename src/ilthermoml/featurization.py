from abc import ABC, abstractmethod
from typing import Any

import padelpy  # type: ignore [import-untyped]
from rdkit.Chem.Descriptors import CalcMolDescriptors

from .chemistry import Molecule
from .exceptions import FeaturizerError
from .memory import ilt_memory

padel_calc_descriptors = ilt_memory.cache(padelpy.from_smiles)


class MoleculeFeaturizer(ABC):
    """Abstract class describing molecule featurizers."""

    def __call__(self, molecule: Molecule) -> dict[str, Any]:
        descriptors = self._featurize(molecule)

        if any(
            [
                not descriptors,
                all(not descriptor for descriptor in descriptors.values()),
            ]
        ):
            msg = f"Unable to calculate descriptors for {molecule!r}"

            raise FeaturizerError(msg)

        return descriptors

    @abstractmethod
    def _featurize(self, molecule: Molecule) -> dict[str, Any]:
        pass


class RDKitMoleculeFeaturizer(MoleculeFeaturizer):
    """Molecule featurizer class for calculating descriptors using RDKit."""

    def _featurize(self, molecule: Molecule) -> dict[str, Any]:
        return CalcMolDescriptors(molecule.rdkit_mol)  # type: ignore [no-untyped-call, no-any-return]


class PadelMoleculeFeaturizer(MoleculeFeaturizer):
    """Molecule featurizer class for calculating descriptors using padel."""

    def _featurize(self, molecule: Molecule) -> dict[str, Any]:
        return padel_calc_descriptors(molecule.smiles)  # type: ignore [no-any-return]
