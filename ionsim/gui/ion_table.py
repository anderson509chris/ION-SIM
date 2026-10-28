"""Ion mixture table shared by the tabs that fly ions: one row per species."""
import matplotlib
import numpy as np
from PySide6 import QtCore, QtWidgets

from ..ions import IonSpecies

COLUMNS = {"name": "Name", "mz": "m/z", "z": "z", "K0": "K0 (cm²/Vs)", "n": "Ions"}


def species_color(k):
    """Colour of species k in every plot."""
    return matplotlib.colormaps["tab10"](k % 10)


class IonTable(QtWidgets.QGroupBox):
    """Editable list of IonSpecies. columns: which fields to show (name, mz, z, K0, n);
    hidden fields keep their defaults. species() raises ValueError on bad input."""

    changed = QtCore.Signal()

    def __init__(self, title="Ions", species=None, columns=("name", "mz", "z", "K0", "n"), parent=None):
        super().__init__(title, parent)
        self.columns = list(columns)
        v = QtWidgets.QVBoxLayout(self)
        self.table = QtWidgets.QTableWidget(0, len(self.columns))
        self.table.setHorizontalHeaderLabels([COLUMNS[c] for c in self.columns])
        self.table.horizontalHeader().setSectionResizeMode(QtWidgets.QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setMinimumHeight(110)
        self.table.itemChanged.connect(self._edited)
        v.addWidget(self.table)
        row = QtWidgets.QHBoxLayout()
        add = QtWidgets.QPushButton("Add"); rem = QtWidgets.QPushButton("Remove")
        add.setToolTip("Add a species (copies the selected row)")
        add.clicked.connect(self._add)
        rem.clicked.connect(self._remove)
        row.addWidget(add); row.addWidget(rem)
        v.addLayout(row)
        self.set_species(species or [IonSpecies()])

    # ---- editing ----
    def _row_values(self, s):
        return {"name": s.name, "mz": f"{s.mz:g}", "z": f"{int(s.z)}", "K0": f"{s.K0:g}", "n": f"{int(s.n)}"}

    def _append(self, s):
        self.table.blockSignals(True)
        r = self.table.rowCount()
        self.table.insertRow(r)
        vals = self._row_values(s)
        for c, key in enumerate(self.columns):
            it = QtWidgets.QTableWidgetItem(vals[key])
            if c == 0:
                it.setData(QtCore.Qt.DecorationRole, _swatch(r))
            self.table.setItem(r, c, it)
        self.table.blockSignals(False)

    def _add(self):
        try:
            sp = self.species()
        except ValueError:
            sp = [IonSpecies()]
        r = self.table.currentRow()
        base = sp[r] if 0 <= r < len(sp) else sp[-1]
        new = IonSpecies(f"ion {self.table.rowCount() + 1}", base.mz, base.z, base.K0, base.n)
        self._append(new)
        self.changed.emit()

    def _remove(self):
        if self.table.rowCount() > 1:
            self.table.removeRow(max(self.table.currentRow(), 0))
            self._recolor()
            self.changed.emit()

    def _recolor(self):
        self.table.blockSignals(True)
        for r in range(self.table.rowCount()):
            it = self.table.item(r, 0)
            if it is not None:
                it.setData(QtCore.Qt.DecorationRole, _swatch(r))
        self.table.blockSignals(False)

    def _edited(self, *_):
        self.changed.emit()

    # ---- values ----
    def species(self):
        out = []
        for r in range(self.table.rowCount()):
            cell = {key: (self.table.item(r, c).text().strip() if self.table.item(r, c) else "")
                    for c, key in enumerate(self.columns)}
            d = IonSpecies(name=cell.get("name") or f"ion {r + 1}")
            try:
                if "mz" in cell:
                    d.mz = float(cell["mz"])
                if "z" in cell:
                    d.z = int(float(cell["z"] or 1))
                if "K0" in cell:
                    d.K0 = float(cell["K0"])
                if "n" in cell:
                    d.n = int(float(cell["n"] or 0))
            except ValueError:
                raise ValueError(f"Ion table row {r + 1} ({d.name}): numbers only in m/z, z, K0 and Ions")
            if d.mz <= 0 or d.z < 1 or d.K0 <= 0 or d.n < 0:
                raise ValueError(f"Ion table row {r + 1} ({d.name}): m/z and K0 must be positive, z at least 1")
            out.append(d)
        if not out or not any(s.n > 0 for s in out):
            raise ValueError("Add at least one ion species with ions to fly")
        return out

    def first(self):
        """The first species (the reference ion for operating points and scans)."""
        return self.species()[0]

    def set_species(self, species):
        self.table.setRowCount(0)
        for s in species:
            self._append(s)
        self.changed.emit()

    def to_list(self):
        return [s.to_dict() for s in self.species()]

    def from_list(self, rows):
        self.set_species([IonSpecies(**{k: v for k, v in r.items() if k in COLUMNS}) for r in rows] or [IonSpecies()])


def _swatch(k):
    from PySide6 import QtGui
    pm = QtGui.QPixmap(10, 10)
    pm.fill(QtGui.QColor.fromRgbF(*species_color(k)))
    return QtGui.QIcon(pm)


# ---- plotting and reporting helpers for mixtures ----
FATE_COLORS = {1: "tab:blue", 2: "tab:red"}


def path_style(status, species_idx, n_species):
    """Line colour and style for one recorded ion. One species: colour by fate (blue through,
    red lost, orange still flying). Several: colour by species, solid through, dotted lost,
    dashed still flying."""
    if n_species <= 1:
        return FATE_COLORS.get(int(status), "tab:orange"), "-"
    return species_color(int(species_idx)), {1: "-", 2: ":"}.get(int(status), "--")


def fate_caption(n_species):
    if n_species <= 1:
        return "blue: transmitted, red: lost, orange: still flying"
    return "colour: ion species; solid: transmitted, dotted: lost, dashed: still flying"


def species_legend(ax, species, **kw):
    from matplotlib.lines import Line2D
    if len(species) > 1:
        ax.legend([Line2D([], [], color=species_color(k)) for k in range(len(species))],
                  [s.name for s in species], fontsize=7, **kw)


def flight_metrics(res, species, z_len=None):
    """Headline numbers of a gas or vacuum flight, per species: {label: value}."""
    out = {}
    h = getattr(res, "heat", None)
    for d in species_stats(res, species, z_len):
        n = d["name"]
        out[f"{n} transmitted (%)"] = 100 * d["through"]
        out[f"{n} transit (ms)"] = d["transit"] * 1e3
        if h is not None:
            m = d["mask"] & (res.status >= 0)
            if m.any():
                out[f"{n} internal T peak (K)"] = float(np.median(h.tint_max[m]))
                out[f"{n} T_eff peak (K)"] = float(np.median(h.teff_max[m]))
                if h.model.ea_ev > 0:
                    out[f"{n} intact (%)"] = 100 * float(np.mean(h.survival[m]))
    return out


def species_stats(res, species, z_len=None):
    """Per species: dict(name, n, through, lost, flying, transit, speed). transit in s."""
    sp = res.species if res.species is not None else np.zeros(len(res.status), int)
    out = []
    for k, s in enumerate(species):
        m = sp == k
        st = res.status[m]
        tr = st == 1
        transit = float(np.nanmean(res.t_end[m][tr])) if tr.any() else np.nan
        out.append(dict(name=s.name, mz=s.mz, z=s.z, K0=s.K0, n=int(m.sum()), mask=m,
                        through=float(np.mean(tr)) if m.any() else np.nan,
                        lost=float(np.mean(st == 2)) if m.any() else np.nan,
                        flying=float(np.mean(st <= 0)) if m.any() else np.nan, transit=transit,
                        speed=(z_len / transit if z_len and np.isfinite(transit) else np.nan)))
    return out
