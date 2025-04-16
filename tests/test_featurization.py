from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from ilthermoml.chemistry import Ion
from ilthermoml.exceptions import FeaturizerError
from ilthermoml.featurization import (
    CachingMoleculeFeaturizer,
    MoleculeFeaturizer,
    PadelMoleculeFeaturizer,
    RDKitMoleculeFeaturizer,
)

if TYPE_CHECKING:
    from pytest_mock import MockerFixture

    from ilthermoml.chemistry import Molecule


def test_molecule_featurizer_raises_error_when_all_descriptors_none() -> None:
    # Arrange.
    class MockFeaturizer(MoleculeFeaturizer):
        def _featurize(self, _molecule: Molecule) -> dict[str, None]:
            return {
                "Test1": None,
                "Test2": None,
                "Test3": None,
                "Test4": None,
            }

    featurize = MockFeaturizer()
    molecule = Ion("[Na+]")

    # Act & assert.
    with pytest.raises(FeaturizerError):
        featurize(molecule)


def test_rdkit_molecule_featurizer_attempts_calculating_descriptors(
    mocker: MockerFixture,
) -> None:
    # Mock.
    mock_calc_descriptors = mocker.patch(
        "ilthermoml.featurization.CalcMolDescriptors",
        return_value={"Test": "Test"},
    )

    # Arrange.
    featurize = RDKitMoleculeFeaturizer()
    molecule = Ion("[Na+]")

    # Act.
    featurize(molecule)

    # Assert
    mock_calc_descriptors.assert_called_once_with(molecule.rdkit_mol)


def test_padel_molecule_featurizer_attempts_calculating_descriptors(
    mocker: MockerFixture,
) -> None:
    # Mock.
    mock_calc_descriptors = mocker.patch(
        "ilthermoml.featurization.padel_calc_descriptors",
        return_value={"Test": "Test"},
    )

    # Arrange.
    featurize = PadelMoleculeFeaturizer()
    molecule = Ion("[Na+]")

    # Act.
    featurize(molecule)

    # Assert
    mock_calc_descriptors.assert_called_once_with(molecule.smiles)


def test_padel_molecule_featurizer_converts_output_to_floats(
    mocker: MockerFixture,
) -> None:
    # Mock.
    mocker.patch(
        "ilthermoml.featurization.padel_calc_descriptors",
        return_value={
            "TestConvertable": "0.15",
            "TestUnconvertable": "Unconvertable",
        },
    )

    # Arrange.
    featurize = PadelMoleculeFeaturizer()
    molecule = Ion("[Na+]")

    # Act.
    desctriptors = featurize(molecule)

    # Assert
    assert type(desctriptors["TestConvertable"]) is float
    assert type(desctriptors["TestUnconvertable"]) is str


def test_cahing_molecule_featurizer_calls_inner_featurizer_only_once(
    mocker: MockerFixture,
) -> None:
    # Spy.
    spy_featurizer = mocker.spy(RDKitMoleculeFeaturizer(), "__call__")

    # Arrange.
    featurize = CachingMoleculeFeaturizer(spy_featurizer)
    molecule = Ion("[Na+]")

    # Act.
    featurize(molecule)
    featurize(molecule)

    # Assert
    spy_featurizer.assert_called_once_with(molecule)
