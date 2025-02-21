__all__ = [
    "EntryError",
    "ILThermoMLException",
]


class ILThermoMLException(Exception):  # noqa: N818
    pass


class EntryError(ILThermoMLException):
    pass
