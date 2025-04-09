from abc import ABC, abstractmethod
from typing import Any

from rdkit.Chem.Descriptors import CalcMolDescriptors

from .chemistry import Molecule
from .exceptions import FeaturizerError


class MoleculeFeaturizer(ABC):
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
    def _featurize(self, molecule: Molecule) -> dict[str, Any]:
        return CalcMolDescriptors(molecule.rdkit_mol)  # type: ignore [no-untyped-call, no-any-return]
