__all__ = [
    "ChemistryError",
    "DatasetError",
    "EntryError",
    "ILThermoMLException",
    "InvalidChargeError",
    "IonicLiquidCationError",
    "UnsupportedSaltTypeError",
]


class ILThermoMLException(Exception):  # noqa: N818
    """Base exception for ILThermoML errors."""


class EntryError(ILThermoMLException):
    """Exception raised for errors in the entry retrieval."""


class DatasetError(ILThermoMLException):
    """Exception raised for errors in the dataset operations."""


class ChemistryError(ILThermoMLException):
    """Exception raised for errors in the chemistry operations."""


class UnsupportedSaltTypeError(ChemistryError):
    """Exception raised for unsupported types of salts."""


class IonicLiquidCationError(UnsupportedSaltTypeError):
    """Exception raised for unsupported types of cations in ionic liquids."""


class InvalidChargeError(ChemistryError):
    """Exception raised for invalid charge values."""
