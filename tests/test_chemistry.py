from __future__ import annotations

import pytest

from ilthermoml.chemistry import Anion, Cation, Ion, IonicLiquid, Stoichiometry
from ilthermoml.exceptions import (
    InvalidChargeError,
    IonicLiquidCationError,
    UnsupportedSaltTypeError,
)


def test_ionic_liquid_correct_ions() -> None:
    # Arrange.
    # ILThermo compound ID: AAiEIE.
    smiles = "CC[n+]1ccn(C)c1.N#C[N-]C#N"
    smiles_reverse = "N#C[N-]C#N.CC[n+]1ccn(C)c1"

    # Act.
    il = IonicLiquid(smiles)
    il_reverse = IonicLiquid(smiles_reverse)

    # Assert.
    assert il.cation.smiles == "CC[n+]1ccn(C)c1"
    assert il.anion.smiles == "N#C[N-]C#N"

    assert il_reverse.cation.smiles == "CC[n+]1ccn(C)c1"
    assert il_reverse.anion.smiles == "N#C[N-]C#N"


def test_ionic_liquid_correct_stoichiometry() -> None:
    # Arrange.
    # ILThermo compound ID: ACGpYk.
    smiles = "CCCCn1cc[n+](C)c1.CCCCn1cc[n+](C)c1.N#CS[Zn-2](SC#N)(SC#N)SC#N"

    expected_stoichiometry = Stoichiometry(2, 1)

    # Act.
    il = IonicLiquid(smiles)

    # Assert.
    assert il.stoichiometry == expected_stoichiometry


def test_ionic_liquid_value_error_if_more_than_one_anion() -> None:
    # Arrange.
    # ILThermo compound ID: AAiEIE (modified).
    smiles = "CC[n+]1c(Cl)cn(C)c1.CC[n+]1ccn(C)c1.N#C[N-]C#N"

    # Act & assert.
    with pytest.raises(UnsupportedSaltTypeError):
        _ = IonicLiquid(smiles)


def test_ion_charge_error_if_charge_zero() -> None:
    # Arrange.
    smiles = "O=C(O)CN"

    # Act & assert.
    with pytest.raises(InvalidChargeError):
        _ = Ion(smiles)


def test_cation_charge_error_if_charge_negative() -> None:
    # Arrange.
    smiles = "O=C([O-])CN"

    # Act & assert.
    with pytest.raises(InvalidChargeError):
        _ = Cation(smiles)


def test_anion_charge_error_if_charge_positive() -> None:
    # Arrange.
    smiles = "O=C(O)C[NH3+]"

    # Act & assert.
    with pytest.raises(InvalidChargeError):
        _ = Anion(smiles)


def test_ionic_liquid_cation_error_if_cation_inorganic() -> None:
    # Arrange.
    smiles = "[Na+].N#C[N-]C#N"

    # Act & assert.
    with pytest.raises(IonicLiquidCationError):
        _ = IonicLiquid(smiles)
