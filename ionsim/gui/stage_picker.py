"""The 'Ions from' rows a stage puts in one of its setting groups: which model feeds it."""
from PySide6 import QtCore, QtWidgets


class SourcePicker(QtCore.QObject):
    """A combo of the models this stage can take from, an optional DC step, and a note.

    accepts: the controller classes that can feed this stage.
    attr: what the feeding controller hands on (exit_ions or exit_gas).
    changed fires when another source is picked, the step changes, or the source reruns."""

    changed = QtCore.Signal()

    def __init__(self, form, accepts, attr="exit_ions", label="Ions from", manual="Set below", step=True,
                 tip="", step_tip="", parent=None):
        super().__init__(parent)
        self.accepts, self.attr, self.manual = accepts, attr, manual
        self.sources = {}                       # model name -> controller
        self.combo = QtWidgets.QComboBox(); self.combo.addItem(manual)
        self.combo.setToolTip(tip)
        self.combo.setProperty("ionsim_nosave", True)       # the workspace saves links by model name
        self.step = QtWidgets.QDoubleSpinBox(); self.step.setRange(-1e4, 1e4); self.step.setDecimals(2)
        self.step.setValue(10.0); self.step.setSuffix(" V"); self.step.setToolTip(step_tip)
        self.note = QtWidgets.QLabel(); self.note.setWordWrap(True); self.note.hide()
        form.insertRow(0, label, self.combo)
        if step:
            form.insertRow(1, "  energy added on the way in", self.step)
        self.has_step = step
        self.form = form
        form.addRow(self.note)
        self.combo.currentIndexChanged.connect(self._picked)
        self.step.valueChanged.connect(self.changed)
        self._picked()

    def set_sources(self, sources):
        """sources: {model name: controller} that can feed this stage."""
        for tab in self.sources.values():
            try:
                tab.exit_changed.disconnect(self._updated)
            except (RuntimeError, TypeError):
                pass
        self.sources = {k: v for k, v in sources.items() if isinstance(v, self.accepts)}
        for tab in self.sources.values():
            tab.exit_changed.connect(self._updated)
        cur = self.combo.currentText()
        self.combo.blockSignals(True)
        self.combo.clear(); self.combo.addItem(self.manual)
        self.combo.addItems([f"{name} exit" for name in self.sources])
        self.combo.setCurrentIndex(max(self.combo.findText(cur), 0))
        self.combo.blockSignals(False)
        self._picked()

    def pick(self, name):
        """Take from the model called name; None sets it by hand."""
        self.combo.setCurrentIndex(max(self.combo.findText(f"{name} exit"), 0) if name else 0)

    def source(self):
        k = self.combo.currentIndex()
        return list(self.sources.values())[k - 1] if k > 0 else None

    def source_name(self):
        k = self.combo.currentIndex()
        return list(self.sources)[k - 1] if k > 0 else None

    def data(self):
        """What the source handed on (None when set by hand or not run yet)."""
        tab = self.source()
        return getattr(tab, self.attr, None) if tab is not None else None

    def linked(self):
        return self.source() is not None

    def set_note(self, text):
        self.note.setText(text)

    def _picked(self, *_):
        on = self.linked()
        if self.has_step:
            self.step.setVisible(on)
            lab = self.form.labelForField(self.step)
            if lab is not None:
                lab.setVisible(on)
        self.note.setVisible(on)
        self.changed.emit()

    def _updated(self):
        if self.sender() is self.source():
            self.changed.emit()
