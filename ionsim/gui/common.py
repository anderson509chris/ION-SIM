"""Shared GUI pieces: parameter forms, plot panels, background worker."""
import traceback
from PySide6 import QtCore, QtWidgets
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT


class ParamForm(QtWidgets.QGroupBox):
    """A titled group of labelled numeric inputs built from a spec list:
    (key, label, default, min, max, decimals, step, suffix). decimals=0 gives an int box."""

    changed = QtCore.Signal()

    def __init__(self, title, spec, parent=None):
        super().__init__(title, parent)
        lay = QtWidgets.QFormLayout(self)
        lay.setFieldGrowthPolicy(QtWidgets.QFormLayout.AllNonFixedFieldsGrow)
        self.boxes = {}
        for key, label, default, lo, hi, dec, step, suffix in spec:
            if dec == 0:
                box = QtWidgets.QSpinBox()
                box.setRange(int(lo), int(hi)); box.setSingleStep(int(step)); box.setValue(int(default))
            else:
                box = QtWidgets.QDoubleSpinBox()
                box.setDecimals(dec); box.setRange(lo, hi); box.setSingleStep(step); box.setValue(default)
            if suffix:
                box.setSuffix(" " + suffix)
            box.valueChanged.connect(self.changed)
            lay.addRow(label, box)
            self.boxes[key] = box

    def values(self):
        return {k: b.value() for k, b in self.boxes.items()}

    def set(self, key, value):
        self.boxes[key].setValue(value)


class PlotPanel(QtWidgets.QTabWidget):
    """Tabs of matplotlib figures, each with the standard zoom/pan/save toolbar."""

    def __init__(self, names, parent=None):
        super().__init__(parent)
        self.figs = {}
        for name in names:
            w = QtWidgets.QWidget()
            v = QtWidgets.QVBoxLayout(w)
            fig = Figure(figsize=(7, 5), layout="tight")
            canvas = FigureCanvasQTAgg(fig)
            v.addWidget(NavigationToolbar2QT(canvas, w))
            v.addWidget(canvas)
            self.addTab(w, name)
            self.figs[name] = fig
            self.hint(name, "Press Run to fill this plot.")

    def hint(self, name, text):
        """Show what fills a plot until a run draws it (a plot is only drawn by the run modes that make it)."""
        f = self.figs[name]
        f.clear()
        ax = f.add_subplot(111); ax.axis("off")
        ax.text(0.5, 0.5, text, ha="center", va="center", fontsize=11, color="0.35", wrap=True,
                transform=ax.transAxes)
        f.canvas.draw_idle()

    def fig(self, name, show=True):
        f = self.figs[name]
        f.clear()
        if show:
            self.setCurrentIndex(list(self.figs).index(name))
        return f

    def draw(self, name):
        self.figs[name].canvas.draw_idle()

    def current_figure(self):
        return list(self.figs.values())[self.currentIndex()]


class Worker(QtCore.QObject):
    """Runs fn(progress) in a QThread. progress(fraction) returns True when Stop was pressed."""
    progressed = QtCore.Signal(int)
    finished = QtCore.Signal(object)
    failed = QtCore.Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        self.cancelled = False

    def progress(self, frac):
        self.progressed.emit(int(100 * min(max(frac, 0), 1)))
        return self.cancelled

    @QtCore.Slot()
    def run(self):
        try:
            self.finished.emit(self.fn(self.progress))
        except Exception:
            self.failed.emit(traceback.format_exc())


class SimTab(QtWidgets.QWidget):
    """Layout shared by every simulation tab: inputs on the left, plots on the right,
    Run / Stop / progress at the bottom of the inputs, a results box under the plots."""

    def __init__(self, plot_names, parent=None):
        super().__init__(parent)
        split = QtWidgets.QSplitter(QtCore.Qt.Horizontal)
        QtWidgets.QHBoxLayout(self).addWidget(split)

        left = QtWidgets.QWidget()
        self.form_layout = QtWidgets.QVBoxLayout(left)
        scroll = QtWidgets.QScrollArea(); scroll.setWidgetResizable(True); scroll.setWidget(left)
        scroll.setMinimumWidth(330)
        self.scroll = scroll

        self.run_btn = QtWidgets.QPushButton("Run")
        self.stop_btn = QtWidgets.QPushButton("Stop"); self.stop_btn.setEnabled(False)
        self.bar = QtWidgets.QProgressBar()
        self.status = QtWidgets.QLabel("Ready"); self.status.setWordWrap(True)
        controls = QtWidgets.QWidget()
        c = QtWidgets.QGridLayout(controls)
        c.addWidget(self.run_btn, 0, 0); c.addWidget(self.stop_btn, 0, 1)
        c.addWidget(self.bar, 1, 0, 1, 2); c.addWidget(self.status, 2, 0, 1, 2)

        leftcol = QtWidgets.QWidget()
        lc = QtWidgets.QVBoxLayout(leftcol); lc.setContentsMargins(0, 0, 0, 0)
        lc.addWidget(scroll, 1); lc.addWidget(controls)
        split.addWidget(leftcol)

        right = QtWidgets.QSplitter(QtCore.Qt.Vertical)
        self.plots = PlotPanel(plot_names)
        self.results = QtWidgets.QPlainTextEdit(); self.results.setReadOnly(True)
        self.results.setMaximumBlockCount(2000)
        right.addWidget(self.plots); right.addWidget(self.results)
        right.setSizes([600, 140])
        split.addWidget(right)
        split.setSizes([360, 900])

        self.stop_btn.clicked.connect(self.stop)
        self._thread = None
        self.last_table = None     # (header, rows) for CSV export

    def log(self, text):
        self.results.appendPlainText(text)

    def start(self, fn, on_done, label="Running..."):
        if self._thread is not None:
            return
        self.run_btn.setEnabled(False); self.stop_btn.setEnabled(True)
        self.bar.setValue(0); self.status.setText(label)
        self._worker = Worker(fn)
        self._thread = QtCore.QThread()
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.progressed.connect(self.bar.setValue)
        self._on_done = on_done
        self._worker.finished.connect(self._done)      # bound slot -> queued into the GUI thread
        self._worker.failed.connect(self._failed)
        self._thread.start()

    def stop(self):
        if self._thread is not None:
            self._worker.cancelled = True
            self.status.setText("Stopping...")

    def _cleanup(self):
        self._thread.quit(); self._thread.wait()
        self._thread = None
        self.run_btn.setEnabled(True); self.stop_btn.setEnabled(False)

    @QtCore.Slot(object)
    def _done(self, result):
        on_done = self._on_done
        cancelled = self._worker.cancelled
        self._cleanup()
        self.bar.setValue(100)
        self.status.setText("Stopped (partial results shown)" if cancelled else "Done")
        on_done(result)

    @QtCore.Slot(str)
    def _failed(self, tb):
        self._cleanup()
        self.status.setText("Error - see results box")
        self.log(tb)
