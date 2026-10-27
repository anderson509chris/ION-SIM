"""TWAVE traveling-wave separation tab."""
import numpy as np
from PySide6 import QtWidgets

from ..twave import TWBoards, mobility, fly_mobility, mean_velocity_sine, fwhm
from .common import ParamForm, SimTab
from .heating_view import HeatingBox, draw_heating
from .ion_table import IonTable, species_color
from ..ions import IonSpecies

MODES = ["Separation", "Wave-speed sweep", "Velocity vs theory"]


class TwaveTab(SimTab):
    def __init__(self, parent=None):
        super().__init__(["TW field", "Arrival times", "Ion velocity", "Speed sweep", "Heating"], parent)
        self.plots.hint("Arrival times", "Set Run mode to “Separation” and press Run.")
        self.plots.hint("Speed sweep", "Set Run mode to “Wave-speed sweep” and press Run.")
        self.plots.hint("Heating", "Set Run mode to “Separation”, tick Ion heating and press Run.")
        self.boards = ParamForm("Boards", [
            ("pitch", "Electrode pitch", 0.508, 0.01, 10, 3, 0.01, "mm"),
            ("width", "Electrode width", 0.432, 0.01, 10, 3, 0.01, "mm"),
            ("gap", "Board gap", 2.75, 0.1, 50, 3, 0.05, "mm"),
            ("nwave", "Electrodes per wave", 8, 2, 64, 0, 1, ""),
            ("non", "Electrodes on", 4, 1, 63, 0, 1, ""),
        ])
        self.wave = ParamForm("Traveling wave", [
            ("amp", "Amplitude", 30.0, 0.1, 500, 2, 1, "V"),
            ("speed", "Speed", 200.0, 1, 5000, 1, 10, "m/s"),
        ])
        self.waveform = QtWidgets.QComboBox(); self.waveform.addItems(["electrodes", "sine"])
        self.wave.layout().addRow("Waveform", self.waveform)
        self.gas = ParamForm("Gas", [
            ("p", "Pressure", 4.0, 0.01, 760, 3, 0.1, "Torr"),
            ("T", "Temperature", 300.0, 1, 1000, 1, 5, "K"),
        ])
        self.ion_table = IonTable("Ions (one row per species, separated together)",
                                  [IonSpecies("K0 1.00", 500, 1, 1.00, 500), IonSpecies("K0 1.03", 500, 1, 1.03, 500)])
        self.ion_table.setToolTip("K0 sets the drift; m/z and z set diffusion and ion heating. "
                                  "The velocity check uses the first row's count and charge.")
        self.ions = ParamForm("Path and time step", [
            ("path", "Path length", 1.0, 0.01, 100, 3, 0.1, "m"),
            ("dt", "Time step", 0.1, 0.001, 10, 3, 0.01, "us"),
        ])
        self.mode = QtWidgets.QComboBox(); self.mode.addItems(MODES)
        mode_box = QtWidgets.QGroupBox("Run mode")
        QtWidgets.QVBoxLayout(mode_box).addWidget(self.mode)
        self.sweep = ParamForm("Speed sweep", [
            ("s0", "From", 130, 1, 5000, 0, 10, "m/s"),
            ("s1", "To", 300, 1, 5000, 0, 10, "m/s"),
            ("ns", "Points", 8, 2, 100, 0, 1, ""),
        ])
        self.heat = HeatingBox()
        self.heat.setToolTip("Tracks each ion's effective temperature from its drift K·E through the gas, "
                             "and its internal temperature (Separation mode)")
        for w in (self.boards, self.wave, self.gas, self.ion_table, self.ions, self.heat, mode_box, self.sweep):
            self.form_layout.addWidget(w)
        self.form_layout.addStretch(1)
        self.mode.currentIndexChanged.connect(lambda i: self.sweep.setVisible(i == 1))
        self.sweep.setVisible(False)
        self.run_btn.clicked.connect(self.run)
        self.boards.changed.connect(self._draw_field)
        self.wave.changed.connect(self._draw_field)
        self.waveform.currentIndexChanged.connect(self._draw_field)
        self._draw_field()

    def _tw(self, speed=None):
        b, w = self.boards.values(), self.wave.values()
        return TWBoards(pitch=b["pitch"] * 1e-3, width=min(b["width"], b["pitch"] * 0.999) * 1e-3,
                        gap=b["gap"] * 1e-3, n_per_wave=b["nwave"], n_on=min(b["non"], b["nwave"] - 1),
                        amplitude=w["amp"], speed=speed or w["speed"], waveform=self.waveform.currentText())


    def _draw_field(self):
        try:
            tw = self._tw()
        except Exception:
            return
        fig = self.plots.fig("TW field", show=False)
        a1, a2 = fig.subplots(2, 1, sharex=True)
        z = np.linspace(0, 2 * tw.wavelength, 1500)
        a1.plot(z * 1e3, tw.board_potential(z) * tw.amplitude)
        a1.set_ylabel("board potential (V)")
        a1.set_title(f"wavelength {tw.wavelength * 1e3:.3f} mm, peak Ez {tw.e_max:.0f} V/m, "
                     f"step every {tw.step_time * 1e6:.2f} µs", fontsize=9)
        for k in range(3):
            a2.plot(z * 1e3, tw.ez(k * tw.step_time * 1.0001, z), label=f"step {k}")
        a2.set_ylabel("midplane Ez (V/m)"); a2.set_xlabel("z (mm)"); a2.legend(fontsize=8)
        self.plots.draw("TW field")

    def run(self):
        try:
            species = [sp for sp in self.ion_table.species() if sp.n > 0]
            tw = self._tw()
        except Exception as e:
            self.log(f"Input error: {e}")
            return
        g, io, sw, mode = self.gas.values(), self.ions.values(), self.sweep.values(), self.mode.currentIndex()
        k0s, names = [sp.K0 for sp in species], [sp.name for sp in species]
        counts = [sp.n for sp in species]
        n, path, dt, zc = species[0].n, io["path"], io["dt"] * 1e-6, species[0].z
        Ks = [mobility(k, g["p"], g["T"]) for k in k0s]
        heat_model = self.heat.model()

        def separate(tw, path, counts, progress, heating=None):
            sp = np.repeat(np.arange(len(Ks)), counts)
            K = np.asarray(Ks)[sp]
            z0 = np.random.default_rng().random(len(K)) * tw.wavelength
            v_slow = max(float(mean_velocity_sine(min(Ks) * tw.e_max, tw.speed)), 0.02 * tw.speed)
            r = fly_mobility(tw, K, z0, path, dt, 3 * path / v_slow, g["T"], np.array([s.z for s in species])[sp], sp,
                             progress=progress, heating=heating, mz=np.array([s.mz for s in species])[sp])
            species_i = sp
            arr = [r.arrival[species_i == s] for s in range(len(Ks))]
            return (arr, r.heat, species_i) if heating is not None else arr

        def job(progress):
            out = {"mode": mode, "k0s": k0s, "names": names, "Ks": Ks, "tw": tw, "path": path}
            if mode == 0:
                if heat_model is not None:
                    out["t"], out["heat"], out["species"] = separate(tw, path, counts, progress, heat_model)
                else:
                    out["t"] = separate(tw, path, counts, progress)
            elif mode == 1:
                speeds = np.linspace(sw["s0"], sw["s1"], int(sw["ns"]))
                out["speeds"], out["sweep"] = speeds, []
                for j, s in enumerate(speeds):
                    sub = lambda f, j=j: progress((j + f) / len(speeds))
                    out["sweep"].append(separate(self._tw(s), path, counts, sub))
                    if progress((j + 1) / len(speeds)):
                        out["speeds"] = speeds[: j + 1]
                        break
            else:
                L = min(path, 0.1)
                K_grid = np.linspace(0.3, 1.8, 10) * tw.speed / tw.e_max
                vs = []
                rng = np.random.default_rng()
                for j, K in enumerate(K_grid):
                    z0 = rng.random(n) * tw.wavelength
                    v_th = max(float(mean_velocity_sine(K * tw.e_max, tw.speed)), 0.02 * tw.speed)
                    r = fly_mobility(tw, K, z0, L, dt, 3 * L / v_th, g["T"], zc,
                                     progress=lambda f, j=j: progress((j + f) / len(K_grid)))
                    vs.append(np.nanmean((L - z0) / r.arrival))
                out["cs"] = K_grid * tw.e_max / tw.speed
                out["v"] = np.array(vs)
            return out

        self.start(job, self._show)

    @staticmethod
    def _stats(t, k0s):
        means = [np.nanmedian(x) for x in t]
        widths = [fwhm(x) if np.isfinite(x).sum() > 3 else np.nan for x in t]
        rpp = [1.18 * abs(means[j + 1] - means[j]) / (widths[j] + widths[j + 1]) for j in range(len(t) - 1)]
        return means, widths, rpp

    def _show(self, out):
        k0s, names, mode = out["k0s"], out["names"], out["mode"]
        if mode == 0:
            t = out["t"]
            means, widths, rpp = self._stats(t, k0s)
            for name, k0, x, m, w in zip(names, k0s, t, means, widths):
                self.log(f"{name} (K0 {k0:.4f}): arrival {m * 1e3:.3f} ms, FWHM {w * 1e3:.3f} ms, "
                         f"resolving power {m / w:.0f}, arrived {np.isfinite(x).mean():.0%}")
            for j, r in enumerate(rpp):
                self.log(f"Rpp({names[j]} vs {names[j + 1]}) = {r:.2f}")
            self.last_table = (["species", "K0", "arrival_s"],
                               [[names[s], k0s[s], v] for s in range(len(t)) for v in t[s]])
            fig = self.plots.fig("Arrival times")
            ax = fig.add_subplot(111)
            allt = np.concatenate([x[np.isfinite(x)] for x in t])
            if len(allt):
                # bin width = whole wave steps, otherwise the stepped wave shows up as a comb
                step = out["tw"].step_time
                bw = step * max(1, int(np.ceil((allt.max() - allt.min()) / 80 / step)))
                bins = np.arange(allt.min(), allt.max() + bw, bw) * 1e3
                for k, (x, name) in enumerate(zip(t, names)):
                    ax.hist(x[np.isfinite(x)] * 1e3, bins=bins, alpha=0.6, color=species_color(k), label=name)
            ax.set_xlabel("arrival time (ms)"); ax.set_ylabel("ions"); ax.legend()
            ax.set_title(f"{out['path']} m, {out['tw'].amplitude:.0f} V, {out['tw'].speed:.0f} m/s" +
                         (f", Rpp = {', '.join(f'{r:.2f}' for r in rpp)}" if rpp else ""))
            self.plots.draw("Arrival times")
            self._draw_velocity(out, None)
            h = out.get("heat")
            if h is not None:
                for s, name in enumerate(names):
                    self.log(f"  Heating {name}: " + h.summary(out["species"] == s))
            draw_heating(self.plots.figs["Heating"], h,
                         title=f"{', '.join(names)}, {out['tw'].amplitude:.0f} V at {out['tw'].speed:.0f} m/s: "
                               "ion heating along the path")
            self.plots.draw("Heating")
        elif mode == 1:
            rows, curves = [], []
            for s, t in zip(out["speeds"], out["sweep"]):
                means, widths, rpp = self._stats(t, k0s)
                curves.append((means, widths, rpp))
                rows.append([s] + means + rpp)
                self.log(f"speed {s:.0f} m/s: arrivals " + ", ".join(f"{m * 1e3:.2f}" for m in means) + " ms"
                         + (f"; Rpp " + ", ".join(f"{r:.2f}" for r in rpp) if rpp else ""))
            self.last_table = (["speed_m_s"] + [f"t_{nm}" for nm in names] +
                               [f"Rpp_{names[j]}_{names[j + 1]}" for j in range(len(names) - 1)], rows)
            fig = self.plots.fig("Speed sweep")
            ax1, ax2 = fig.subplots(1, 2)
            sp = out["speeds"][: len(curves)]
            for j, name in enumerate(names):
                ax1.plot(sp, [c[0][j] * 1e3 for c in curves], "o-", color=species_color(j), label=name)
                ax1.set_xlabel("wave speed (m/s)"); ax1.set_ylabel("arrival time (ms)"); ax1.legend(fontsize=8)
            for j in range(len(k0s) - 1):
                ax2.plot(sp, [c[2][j] for c in curves], "o-", label=f"{names[j]} vs {names[j + 1]}")
            ax2.set_xlabel("wave speed (m/s)"); ax2.set_ylabel("Rpp"); ax2.set_title("two-peak resolution")
            if len(k0s) > 1:
                ax2.legend(fontsize=8)
            self.plots.draw("Speed sweep")
        else:
            self._draw_velocity(out, (out["cs"], out["v"]))
            self.last_table = (["c_over_s", "v_over_s"], [[c, v / out["tw"].speed] for c, v in zip(out["cs"], out["v"])])
            self.log("Velocity check done: points vs the analytic sine-wave curve on the Ion velocity tab")

    def _draw_velocity(self, out, sim):
        tw = out["tw"]
        fig = self.plots.fig("Ion velocity", show=sim is not None)
        ax = fig.add_subplot(111)
        cs = np.linspace(0, 2, 300)
        ax.plot(cs, mean_velocity_sine(cs * tw.speed, tw.speed) / tw.speed, "k-", label="theory (sine wave)")
        if sim is not None:
            ax.plot(sim[0], sim[1] / tw.speed, "s", label=f"simulated ({tw.waveform})")
        for k, (name, K) in enumerate(zip(out["names"], out["Ks"])):
            c = K * tw.e_max / tw.speed
            ax.axvline(c, ls=":", lw=1, color=species_color(k))
            ax.text(c, 0.03, f" {name}", rotation=90, fontsize=8, va="bottom")
        ax.set_xlabel("c / s  (K·Emax / wave speed)"); ax.set_ylabel("mean velocity / wave speed")
        ax.set_title("Surfing (c ≥ s) vs rolling over (c < s)"); ax.legend(fontsize=8)
        self.plots.draw("Ion velocity")
