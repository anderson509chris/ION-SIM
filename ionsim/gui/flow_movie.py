"""The inlet as a moving picture: gas and ions from the spray, through the capillary, into the
first vacuum chamber.

Three panels, each at its own scale, because the inlet spans a 5 mm spray region, a 0.5 mm wide
capillary tens of mm long and a few-mm free jet: the spray region, the capillary and the jet.
The background is the gas speed (log colour scale, walls grey). White dots are gas parcels carried
by the flow; the coloured dots are the ions, one lane per species, placed by their computed flight
along the axis. Gas speeds up about ten-thousand-fold from the spray to the jet, so each panel
plays at its own speed, given in its title, with the slowest ion taking a few seconds to cross.
"""
import numpy as np
from matplotlib.colors import LogNorm
from matplotlib import colormaps

from .ion_table import species_color

MM = 1e-3
N_GAS = 160
CROSS_S = 6.0          # seconds on screen for the slowest ion to cross a panel


def _fmt_t(t):
    for unit, k in (("s", 1), ("ms", 1e-3), ("µs", 1e-6), ("ns", 1e-9)):
        if t >= k:
            return f"{t / k:.3g} {unit}"
    return f"{t:.3g} s"


class _Panel:
    def __init__(self, name, x0, x1, half, wall, spawn, lane):
        self.name, self.x0, self.x1, self.half = name, x0, x1, half
        self.wall, self.spawn, self.lane = wall, spawn, lane     # wall(x, r) -> bool; spawn(n, rng) -> (x, r)


class FlowMovie:
    def __init__(self, fig):
        self.fig = fig
        self.inlet = None
        self.rng = np.random.default_rng(1)
        self.gas_dots, self.ion_dots = [], []
        self._bg = None          # the still picture, kept so each frame redraws only the dots
        fig.canvas.mpl_connect("draw_event", self._drawn)

    def _live(self):
        return self.inlet is not None and bool(self.gas_dots) and self.gas_dots[0].axes in self.fig.axes

    def _drawn(self, _event):
        if not self._live():
            return
        self._bg = self.fig.canvas.copy_from_bbox(self.fig.bbox)
        self._draw_dots()

    def _draw_dots(self):
        for art in self.gas_dots + self.ion_dots:
            art.axes.draw_artist(art)

    # ---- layout ----
    def show(self, inl, res):
        self.inlet, self.res = inl, res
        a, L, gap = inl.diameter / 2, inl.length, inl.spray_gap
        xm = inl.x_mach_disk
        X = max(1.25 * inl.skim_dist, 1.25 * xm)
        W = max(1.25 * inl.jet_radius(xm), 0.3 * X, 3 * a)
        rng = self.rng

        def spray_spawn(n, first=False):
            if first:
                return rng.uniform(-gap, 0, n), rng.uniform(-0.6 * gap, 0.6 * gap, n)
            th = rng.uniform(0.5 * np.pi, 1.5 * np.pi, n)            # on the far arc, then drawn in
            rho = rng.uniform(0.55, 0.65, n) * gap
            return rho * np.cos(th), rho * np.sin(th)

        def cap_spawn(n, first=False):
            s = a * np.sqrt(rng.random(n)) * rng.choice([-1, 1], n)
            return (rng.uniform(0, L, n) if first else np.zeros(n)), s

        def jet_spawn(n, first=False):
            xi = rng.uniform(0, X, n) if first else np.zeros(n)
            return L + xi, rng.uniform(-1, 1, n) * inl.jet_radius(xi)

        self.panels = [
            _Panel("Spray region", -gap, 0.08 * gap, 0.6 * gap, lambda x, r: (x >= 0) & (np.abs(r) > a), spray_spawn,
                   0.07 * gap),
            _Panel("Capillary", 0.0, L, 1.8 * a, lambda x, r: np.abs(r) > a, cap_spawn, 0.3 * a),
            _Panel("Free jet into the vacuum chamber", L - 0.05 * X, L + X, W,
                   lambda x, r: (x <= L) & (np.abs(r) > a), jet_spawn, 0.1 * W),
        ]
        u_cap = inl.cap["u"]
        u_jet = inl.jet_centre(np.linspace(0, min(X, xm), 400))[2]
        self.panels[0].note = f"gas drawn in at up to {u_cap[0]:.3g} m/s"
        self.panels[1].note = f"mean gas speed {u_cap[0]:.3g} → {u_cap[-1]:.3g} m/s"
        self.panels[2].note = f"gas on axis up to {u_jet.max():.3g} m/s"
        self.fig.clear()
        self._bg = None
        *axes, cax = self.fig.subplots(1, 4, gridspec_kw={"width_ratios": [1, 1.6, 1.3, 0.06]})
        maps = []
        for p in self.panels:
            xs = np.linspace(p.x0, p.x1, 240)
            rs = np.linspace(-p.half, p.half, 120)
            Xg, Rg = np.meshgrid(xs, rs)
            ux, ur = inl.velocity_2d(Xg, Rg)
            sp = np.hypot(ux, ur)
            maps.append(np.ma.masked_where(p.wall(Xg, Rg), sp))
        vmax = max(float(m.max()) for m in maps)
        norm = LogNorm(max(vmax * 1e-4, 1e-3), vmax)
        cmap = colormaps["viridis"].with_extremes(bad="0.55", under=colormaps["viridis"](0.0))
        self.gas_dots, self.ion_dots, self.gas, self.ions = [], [], [], []
        species = [d["ion"] for d in res]
        for ax, p, m in zip(axes, self.panels, maps):
            im = ax.imshow(np.ma.maximum(m, norm.vmin), origin="lower", aspect="auto", cmap=cmap, norm=norm,
                           extent=[p.x0 / MM, p.x1 / MM, -p.half / MM, p.half / MM])
            ax.set_xlabel("x (mm)")
            p.gx, p.gr = p.spawn(N_GAS, first=True)
            self.gas_dots.append(ax.scatter(p.gx / MM, p.gr / MM, s=3, c="w", alpha=0.8, linewidths=0, animated=True))
            p.tracks = []
            for k, d in enumerate(res):
                x, t = d["x"], d["t"]
                inside = (x >= p.x0) & (x <= p.x1)
                if not inside.any():
                    p.tracks.append(None)
                    continue
                t0 = np.interp(max(p.x0, x[0]), x, t)
                t1 = np.interp(min(p.x1, x[-1]), x, t)
                p.tracks.append((t0, t1, x, t, (k - (len(res) - 1) / 2) * p.lane))
            spans = [tr[1] - tr[0] for tr in p.tracks if tr is not None]
            p.period = (max(spans) if spans else 1e-6) * 1.15
            p.scale = (max(spans) if spans else 1e-6) / CROSS_S            # real seconds per screen second
            p.clock = 0.0
            self.ion_dots.append(ax.scatter(np.full(len(res), np.nan), np.zeros(len(res)), s=46, zorder=5,
                                            c=[species_color(k) for k in range(len(res))], edgecolors="k",
                                            linewidths=0.6, animated=True))
            ax.set_title(f"{p.name}\n{p.note}; 1 s here = {_fmt_t(p.scale)}", fontsize=8)
        a0, a1, a2 = axes
        a0.set_ylabel("r (mm)")
        tip_r = max(inl.tip_radius, 0.02 * gap)
        a0.fill([(-gap - 0.3 * gap) / MM, -gap / MM, (-gap - 0.3 * gap) / MM],
                [6 * tip_r / MM, 0, -6 * tip_r / MM], color="0.3")
        a0.annotate("emitter", (-gap / MM, 0), xytext=(4, 8), textcoords="offset points", fontsize=7, color="w")
        a0.set_xlim(-1.02 * gap / MM, 0.08 * gap / MM)
        xsk = L + inl.skim_dist
        a2.axvline(xsk / MM, color="w", ls="--", lw=1)
        a2.annotate("skimmer / lens", (xsk / MM, -0.9 * self.panels[2].half / MM), xytext=(3, 0),
                    textcoords="offset points", fontsize=7, color="w")
        if xm < X:
            rm = inl.jet_radius(xm)
            a2.plot([(L + xm) / MM] * 2, [-rm / MM, rm / MM], color="w", lw=1, ls=":")
            a2.annotate("Mach disk", ((L + xm) / MM, rm / MM), xytext=(3, 2), textcoords="offset points",
                        fontsize=7, color="w")
        xi = np.linspace(0, min(xm, X), 100)
        for sgn in (1, -1):
            a2.plot((L + xi) / MM, sgn * inl.jet_radius(xi) / MM, color="w", lw=0.6, alpha=0.6)
        self.fig.colorbar(im, cax=cax, label="gas speed (m/s)", extend="min")
        handles = [a2.scatter([], [], s=40, color=species_color(k), edgecolors="k", label=s.name)
                   for k, s in enumerate(species)]
        a1.legend(handles=handles, loc="upper right", fontsize=7, framealpha=0.7, title="ions", title_fontsize=7)
        self.fig.suptitle(f"{inl.flow_slm:.2f} L/min into the inlet; white dots: gas, coloured: ions "
                          "(drawn in lanes near the axis; each panel plays at its own speed)", fontsize=9)
        self.step(0.0)

    # ---- motion ----
    def step(self, dt_screen):
        """Advance every panel by dt_screen seconds of screen time."""
        if self.inlet is None:
            return
        inl = self.inlet
        a = inl.diameter / 2
        for p, gd, idots in zip(self.panels, self.gas_dots, self.ion_dots):
            dt = dt_screen * p.scale
            if dt > 0:
                for _ in range(8):
                    ux, ur = inl.velocity_2d(p.gx, p.gr)
                    p.gx = p.gx + ux * dt / 8
                    p.gr = p.gr + ur * dt / 8
                gone = (p.gx < p.x0) | (p.gx > p.x1) | (np.abs(p.gr) > p.half)
                if p.name == "Spray region":
                    gone |= (p.gx > -0.02 * a) & (np.abs(p.gr) < a)
                else:                                    # parcels that left the jet stop: start new ones
                    gone |= (ux == 0) & (ur == 0) & (p.gx > inl.length)
                if gone.any():
                    p.gx[gone], p.gr[gone] = p.spawn(int(gone.sum()))
            gd.set_offsets(np.column_stack([p.gx, p.gr]) / MM)
            p.clock = (p.clock + dt) % p.period
            pos = np.full((len(p.tracks), 2), np.nan)
            for k, tr in enumerate(p.tracks):
                if tr is None:
                    continue
                t0, t1, x, t, lane = tr
                if p.clock <= t1 - t0:
                    pos[k] = (np.interp(t0 + p.clock, t, x) / MM, lane / MM)
            idots.set_offsets(pos)
        c = self.fig.canvas
        if self._bg is None:
            c.draw_idle()
        else:
            c.restore_region(self._bg)
            self._draw_dots()
            c.blit(self.fig.bbox)
