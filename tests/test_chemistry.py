from __future__ import annotations

import pytest

from ilthermoml.chemistry import Anion, Cation, Ion, IonicLiquid, Salt, Stoichiometry
from ilthermoml.exceptions import (
    InvalidChargeError,
    IonicLiquidCationError,
    UnsupportedSaltTypeError,
)


def test_ion_raises_invalid_charge_error_if_charge_zero() -> None:
    # Arrange.
    smiles = "O"

    # Act & assert.
    with pytest.raises(InvalidChargeError):
        _ = Ion(smiles)


def test_cation_raises_invalid_charge_error_if_charge_negative() -> None:
    # Arrange.
    smiles = "[Cl-]"

    # Act & assert.
    with pytest.raises(InvalidChargeError):
        _ = Cation(smiles)


def test_anion_raises_invalid_charge_error_if_charge_positive() -> None:
    # Arrange.
    smiles = "[Na+]"

    # Act & assert.
    with pytest.raises(InvalidChargeError):
        _ = Anion(smiles)


def test_salt_init_assigns_ions_based_on_smiles() -> None:
    # Arrange.
    smiles = "[Na+].[Cl-]"

    # Act.
    salt = Salt(smiles)

    # Assert.
    assert salt.cation.smiles == "[Na+]"
    assert salt.anion.smiles == "[Cl-]"


def test_salt_init_reverses_ions_if_anion_goes_first() -> None:
    # Arrange.
    smiles = "[Cl-].[Na+]"

    # Act.
    salt = Salt(smiles)

    # Assert.
    assert salt.cation.smiles == "[Na+]"
    assert salt.anion.smiles == "[Cl-]"


@pytest.mark.parametrize(
    ("smiles", "expected_stoichiometry"),
    [
        ("[Na+].[Cl-]", (1, 1)),
        ("[Mg+2].[S-2]", (1, 1)),
        ("[Mg+2].[Cl-].[Cl-]", (1, 2)),
        ("[Na+].[Na+].[S-2]", (2, 1)),
    ],
    ids=[
        "A1B1 (monovalent)",
        "A1B1 (multivalent)",
        "A2B1 ",
        "A1B2",
    ],
)
def test_salt_stoichiometry(
    smiles: str,
    expected_stoichiometry: tuple[int, int],
) -> None:
    # Act.
    salt = Salt(smiles)

    # Assert.
    assert salt.stoichiometry == Stoichiometry(*expected_stoichiometry)


def test_salt_raises_unsupported_salt_type_error_if_salt_is_mixed() -> None:
    # Arrange.
    smiles = "[Na+].[K+].C(=O)([O-])[O-]"

    # Act & assert.
    with pytest.raises(UnsupportedSaltTypeError):
        _ = IonicLiquid(smiles)


def test_ionic_liquid_raises_ionic_liquid_cation_error_if_cation_inorganic() -> None:
    # Arrange.
    smiles = "[Na+].CC(=O)[O-]"

    # Act & assert.
    with pytest.raises(IonicLiquidCationError):
        _ = IonicLiquid(smiles)
