__all__ = [
    "DatasetError",
    "EntryError",
    "ILThermoMLException",
]


class ILThermoMLException(Exception):  # noqa: N818
    """Base exception for ILThermoML errors."""


class EntryError(ILThermoMLException):
    """Exception raised for errors in the entry retrieval."""


class DatasetError(ILThermoMLException):
    """Exception raised for errors in the dataset operations."""
