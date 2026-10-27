"""The driver library shared by every tab: one per application, saved on every change."""
from PySide6 import QtCore
from ..drivers import DriverLibrary

_STORE = None


class DriverStore(QtCore.QObject):
    changed = QtCore.Signal()

    def __init__(self, lib):
        super().__init__()
        self.lib = lib

    def names(self):
        return self.lib.names()

    def get(self, name):
        return self.lib.drivers.get(name)

    def touch(self):
        """Call after changing a driver: saves the library and tells the other tabs."""
        try:
            self.lib.save()
        except OSError:
            pass
        self.changed.emit()


def get_store():
    global _STORE
    if _STORE is None:
        _STORE = DriverStore(DriverLibrary.with_defaults())
    return _STORE
