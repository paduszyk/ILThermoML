from abc import ABC, abstractmethod
from collections.abc import Callable
from typing import Any

import padelpy  # type: ignore [import-untyped]
from rdkit.Chem.Descriptors import CalcMolDescriptors

from .chemistry import Molecule, Salt
from .exceptions import FeaturizerError
from .memory import ilt_memory

padel_calc_descriptors = ilt_memory.cache(padelpy.from_smiles)


class MoleculeFeaturizer(ABC):
    """Abstract class describing molecule featurizers."""

    def __call__(self, molecule: Molecule) -> dict[str, Any]:
        descriptors = self._featurize(molecule)

        if all(not descriptor for descriptor in descriptors.values()):
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
        descriptors = padel_calc_descriptors(molecule.smiles)
        for key, value in descriptors.items():
            try:
                descriptors[key] = float(value)
            except ValueError:
                continue

        return descriptors  # type: ignore [no-any-return]


class CachingMoleculeFeaturizer(MoleculeFeaturizer):
    """Wrapper molecule featurizer class that caches calculated descriptors."""

    def __init__(self, featurizer: MoleculeFeaturizer) -> None:
        self._inner_featurize = featurizer
        self._cache: dict[str, dict[str, Any]] = {}

    def _featurize(self, molecule: Molecule) -> dict[str, Any]:
        if molecule.smiles not in self._cache:
            self._cache[molecule.smiles] = self._inner_featurize(molecule)
        return self._cache[molecule.smiles]


class SaltFeaturizer:
    """Class for calculating salt descriptors."""

    def __init__(
        self,
        combination_rule: Callable[[Any, Any], Any],
        featurize: MoleculeFeaturizer,
    ) -> None:
        self.combination_rule = combination_rule
        self.featurize = CachingMoleculeFeaturizer(featurize)

    def __call__(self, salt: Salt) -> dict[str, Any]:
        cation_features = self.featurize(salt.cation)
        anion_features = self.featurize(salt.anion)

        output_keys = set(list(cation_features) + list(anion_features))
        output_values: list[Any] = []

        for key in output_keys:
            if not cation_features[key] or not anion_features[key]:
                output_values.append(None)
                continue

            output_values.append(
                self.combination_rule(cation_features[key], anion_features[key])
            )

        return dict(zip(output_keys, output_values, strict=True))
