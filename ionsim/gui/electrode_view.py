"""Clickable electrodes on a matplotlib canvas, coloured by their DC voltage."""
import matplotlib
from matplotlib.colors import TwoSlopeNorm, Normalize
from matplotlib.collections import LineCollection


class ElectrodeView:
    """Keeps name -> artists for one figure. Click an electrode to select it (Ctrl / Shift
    click adds to the selection); hovering reports the electrode under the mouse."""

    def __init__(self, canvas, on_pick, on_hover=None):
        self.canvas, self.on_pick, self.on_hover = canvas, on_pick, on_hover
        self.artists = {}
        self._last_click = None
        self._hover_name = None
        canvas.mpl_connect("pick_event", self._pick)
        canvas.mpl_connect("motion_notify_event", self._motion)

    def reset(self):
        self.artists = {}

    def add(self, name, artist):
        artist.set_picker(4 if isinstance(artist, LineCollection) else True)
        artist.set_gid(name)
        self.artists.setdefault(name, []).append(artist)

    def norm(self, values):
        v = list(values.values()) or [0.0]
        lo, hi = min(v), max(v)
        if lo < 0 < hi:
            return TwoSlopeNorm(0.0, lo, hi)
        if lo == hi:
            return Normalize(lo - 1, hi + 1)
        return Normalize(lo, hi)

    def paint(self, values, selected=(), cmap="coolwarm", rf=()):
        """values: name -> DC volts; rf: names carrying RF (drawn hatched).
        Returns a ScalarMappable for a colourbar."""
        cm = matplotlib.colormaps[cmap]
        norm = self.norm(values)
        for name, arts in self.artists.items():
            c = cm(norm(values.get(name, 0.0)))
            sel = name in selected
            for a in arts:
                if isinstance(a, LineCollection):
                    a.set_color(c); a.set_linewidth(3.0 if sel else 1.3)
                else:
                    a.set_facecolor(c)
                    a.set_hatch("////" if name in rf else None)
                    a.set_edgecolor("black" if sel else (0.2, 0.2, 0.2, 0.6))
                    a.set_linewidth(2.4 if sel else 0.5)
                    a.set_zorder(3 if sel else 2)
        sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cm)
        sm.set_array([])
        return sm

    def _pick(self, ev):
        me = ev.mouseevent
        key = (me.x, me.y, me.button)
        if key == self._last_click:          # one click can pick overlapping artists
            return
        self._last_click = key
        name = ev.artist.get_gid()
        additive = bool(me.key) and any(k in me.key for k in ("control", "ctrl", "shift", "cmd"))
        if name:
            self.on_pick(name, additive)

    def _motion(self, ev):
        if self.on_hover is None or ev.inaxes is None:
            return
        found = None
        for name, arts in self.artists.items():
            if any(a.axes is ev.inaxes and a.contains(ev)[0] for a in arts):
                found = name
                break
        if found != self._hover_name:
            self._hover_name = found
            self.on_hover(found)
