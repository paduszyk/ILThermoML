__all__ = [
    "DatasetError",
    "EntryError",
    "ILThermoMLException",
]


class ILThermoMLException(Exception):  # noqa: N818
    pass


class EntryError(ILThermoMLException):
    pass


class DatasetError(ILThermoMLException):
    pass
