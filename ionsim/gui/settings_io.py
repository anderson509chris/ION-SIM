"""Every input in a settings column, as plain values: saved into setup files and brought back.

Each input is named by the group it sits in and its label ("Collision gas / Pressure (0 = vacuum)"),
so a setup keeps working when inputs are added or moved. Inputs a later version no longer has are
skipped, and new ones keep their defaults.
"""
from PySide6 import QtWidgets

from .ion_table import IonTable

NOSAVE = "ionsim_nosave"          # widget property: leave this input out (e.g. the stage links, saved apart)


def _group(w, root):
    p = w.parentWidget()
    while p is not None and p is not root:
        if isinstance(p, QtWidgets.QGroupBox):
            return p
        p = p.parentWidget()
    return None


def _label(w):
    if isinstance(w, QtWidgets.QCheckBox):
        return w.text()
    par = w.parentWidget()
    lay = par.layout() if par is not None else None
    if isinstance(lay, QtWidgets.QFormLayout):
        lab = lay.labelForField(w)
        if isinstance(lab, QtWidgets.QLabel) and lab.text().strip():
            return lab.text().strip()
    return type(w).__name__


def _skip(w):
    p = w
    while p is not None:
        if p.property(NOSAVE):
            return True
        p = p.parentWidget()
    par = w.parentWidget()
    # the line edit inside a spin box or combo, and editors inside tables, are part of their owner
    return isinstance(w, QtWidgets.QLineEdit) and isinstance(par, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox)) \
        or isinstance(par, QtWidgets.QAbstractItemView) or (par is not None and isinstance(par.parentWidget(), QtWidgets.QAbstractItemView))


def inputs(root):
    """{name: widget} for every input under root, in layout order."""
    out, seen = {}, {}
    for w in root.findChildren(QtWidgets.QWidget):
        if isinstance(w, IonTable):
            key = f"{w.title()} / ions"
        elif isinstance(w, (QtWidgets.QAbstractSpinBox, QtWidgets.QComboBox, QtWidgets.QCheckBox,
                            QtWidgets.QTableWidget, QtWidgets.QLineEdit)):
            if _skip(w) or _ion_table_part(w):
                continue
            g = _group(w, root)
            key = f"{g.title() if g else ''} / {_label(w)}"
        else:
            continue
        n = seen.get(key, 0)
        seen[key] = n + 1
        out[key if n == 0 else f"{key} #{n + 1}"] = w
    return out


def _ion_table_part(w):
    p = w.parentWidget()
    while p is not None:
        if isinstance(p, IonTable):
            return not isinstance(w, QtWidgets.QComboBox)       # its own table and buttons; keep extras like polarity
        p = p.parentWidget()
    return False


def value(w):
    if isinstance(w, IonTable):
        try:
            return {"species": w.to_list()}
        except ValueError:
            return None
    if isinstance(w, QtWidgets.QAbstractSpinBox):
        return w.value()
    if isinstance(w, QtWidgets.QComboBox):
        return w.currentText()
    if isinstance(w, QtWidgets.QCheckBox):
        return w.isChecked()
    if isinstance(w, QtWidgets.QTableWidget):
        return [[(w.item(r, c).text() if w.item(r, c) else "") for c in range(w.columnCount())] for r in range(w.rowCount())]
    if isinstance(w, QtWidgets.QLineEdit):
        return w.text()
    return None


def set_value(w, v):
    """Returns False when the value no longer fits (an option that is gone)."""
    if v is None:
        return True
    if isinstance(w, IonTable):
        w.from_list(v.get("species", []))
    elif isinstance(w, QtWidgets.QAbstractSpinBox):
        w.setValue(type(w.value())(v))
    elif isinstance(w, QtWidgets.QComboBox):
        if v == "":                       # an empty choice (nothing to pick then) fits whatever is there now
            return True
        i = w.findText(v)
        if i < 0:
            return False
        w.setCurrentIndex(i)
    elif isinstance(w, QtWidgets.QCheckBox):
        w.setChecked(bool(v))
    elif isinstance(w, QtWidgets.QTableWidget):
        w.setRowCount(len(v))
        for r, row in enumerate(v):
            for c, text in enumerate(row[:w.columnCount()]):
                w.setItem(r, c, QtWidgets.QTableWidgetItem(text))
    elif isinstance(w, QtWidgets.QLineEdit):
        w.setText(v)
    return True


def save(root):
    return {k: value(w) for k, w in inputs(root).items()}


def restore(root, state, passes=3):
    """Set every saved input. Choices first and then everything else, a few times over, because a
    choice can show, hide or refill other inputs. Returns the names whose saved value no longer fits."""
    bad = set()
    for _ in range(passes):
        bad = set()
        now = inputs(root)
        order = sorted(now.items(), key=lambda kw: not isinstance(kw[1], QtWidgets.QComboBox))
        for k, w in order:
            if k in state and not set_value(w, state[k]):
                bad.add(k)
    return sorted(bad)
