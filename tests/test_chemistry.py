from __future__ import annotations

import re

import pytest

from ilthermoml.chemistry import Anion, Cation, Ion, IonicLiquid, Stoichiometry
from ilthermoml.exceptions import InvalidChargeError


def test_ionic_liquid_correct_ions() -> None:
    # Arrange.
    smiles = "CC[n+]1ccn(C)c1.N#C[N-]C#N"  # cmpund id: AAiEIE
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
    # compund id: ACGpYk
    smiles = "CCCCn1cc[n+](C)c1.CCCCn1cc[n+](C)c1.N#CS[Zn-2](SC#N)(SC#N)SC#N"

    expected_stoichiometry = Stoichiometry(2, 1)

    # Act.
    il = IonicLiquid(smiles)

    # Assert.
    assert il.stoichiometry == expected_stoichiometry


def test_ionic_liquid_value_error_if_more_than_one_anion() -> None:
    # Arrange.
    smiles = (
        "CC[n+]1c(Cl)cn(C)c1.CC[n+]1ccn(C)c1.N#C[N-]C#N"  # modified cmpund id: AAiEIE
    )

    expected_msg = re.compile(
        r"salts must contain exactly one type of both cation and anion; "
        r"found \d+ type\(s\)"
    )

    # Act & Assert.
    with pytest.raises(ValueError, match=expected_msg):
        _ = IonicLiquid(smiles)


def test_ion_charge_error_if_charge_zero() -> None:
    # Arrange.
    smiles = "O=C(O)CN"

    # Act & Assert
    with pytest.raises(InvalidChargeError):
        _ = Ion(smiles)


def test_cation_charge_error_if_charge_negative() -> None:
    # Arrange.
    smiles = "O=C([O-])CN"

    # Act & Assert
    with pytest.raises(InvalidChargeError):
        _ = Cation(smiles)


def test_anion_charge_error_if_charge_positive() -> None:
    # Arrange.
    smiles = "O=C(O)C[NH3+]"

    # Act & Assert
    with pytest.raises(InvalidChargeError):
        _ = Anion(smiles)
