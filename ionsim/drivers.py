"""Electronics drivers: the boxes built in the shop, defined once and wired to electrodes.

A Driver has one RF frequency, a set of excitation tones, a traveling-wave (TWAVE)
sequencer and any number of outputs. Each output's voltage is

    V(t) = dc + rf cos(2 pi f_rf t + phase)
              + tone_sign * sum_k A_k cos(2 pi f_k t + phase_k)
              + tw_amp * level_p(t)            (only if the output carries TWAVE phase p)

level_p(t) is the TWAVE gate for phase p: n_on of the n_phases outputs are high at a
time, the pattern steps one phase every step_time = pitch / speed, and each edge ramps
over rise_time. tone_sign is +1 / -1 / 0 per output, so the same tones can be added
in phase with the RF (quadrupolar excitation: + on one rod pair, - on the other), across
one rod pair (dipolar: + on x1, - on x2) or left off.

Because fields add linearly, a device is solved once per distinct component (DC, cos and
sin of each frequency, each TWAVE phase), not once per output, so 8 tones on a quad or
16 outputs on a TWAVE/RF combiner stay cheap (see components()).
"""
import json
import os
from dataclasses import dataclass, field, asdict
import numpy as np

LIBRARY_PATH = os.environ.get("IONSIM_DRIVERS", os.path.join(os.path.expanduser("~"), ".ionsim", "drivers.json"))


@dataclass
class Tone:
    freq: float = 100e3        # Hz
    amp: float = 1.0           # V zero-to-peak
    phase: float = 0.0         # degrees
    on: bool = True


@dataclass
class Output:
    name: str
    dc: float = 0.0            # V
    rf: float = 0.0            # V zero-to-peak at the driver's RF frequency
    rf_phase: float = 0.0      # degrees (0 = RF+, 180 = RF-)
    tw: int = -1               # TWAVE phase carried (0-based), -1 = none
    tone_sign: float = 0.0     # multiplier on the driver's tones: +1, -1 or 0


@dataclass
class Driver:
    name: str
    rf_freq: float = 1.0e6
    outputs: list = field(default_factory=list)
    tones: list = field(default_factory=list)
    tw_phases: int = 8
    tw_on: int = 4
    tw_amp: float = 20.0       # V added to an output while its phase is high
    tw_speed: float = 200.0    # m/s, at tw_pitch
    tw_pitch: float = 0.508e-3 # electrode pitch the speed refers to (m)
    tw_rise: float = 0.0       # edge ramp time (s)
    notes: str = ""

    # ---- TWAVE ----
    def step_time(self):
        return self.tw_pitch / self.tw_speed if self.tw_speed > 0 else np.inf

    def tw_frequency(self):
        """Full wave cycles per second (n_phases steps)."""
        st = self.step_time()
        return 1.0 / (self.tw_phases * st) if np.isfinite(st) else 0.0

    def has_twave(self):
        return any(0 <= o.tw < self.tw_phases for o in self.outputs)

    def tw_spec(self):
        return TwSpec(self.name, self.tw_phases, self.tw_on, self.tw_amp, self.step_time(), self.tw_rise)

    # ---- outputs ----
    def output(self, name):
        for o in self.outputs:
            if o.name == name:
                return o
        raise KeyError(f"{self.name} has no output '{name}'")

    def active_tones(self):
        return [t for t in self.tones if t.on and t.amp != 0]

    def components(self, out):
        """(dc, [(freq, amp, phase_deg)], tw) for one output: what the field solver needs.
        tw is (TwSpec, phase) or None."""
        o = self.output(out) if isinstance(out, str) else out
        sines = []
        if o.rf:
            sines.append((self.rf_freq, o.rf, o.rf_phase))
        if o.tone_sign:
            sines += [(t.freq, o.tone_sign * t.amp, t.phase) for t in self.active_tones()]
        tw = (self.tw_spec(), o.tw) if 0 <= o.tw < self.tw_phases and self.tw_amp else None
        return o.dc, sines, tw

    def waveform(self, out, t):
        """Voltage of an output at times t (array)."""
        t = np.asarray(t, float)
        dc, sines, tw = self.components(out)
        v = np.full(t.shape, float(dc))
        for f, a, ph in sines:
            v += a * np.cos(2 * np.pi * f * t + np.radians(ph))
        if tw:
            v += tw[0].amp * tw[0].level(tw[1], t)
        return v

    def max_freq(self):
        fs = [self.rf_freq] if any(o.rf for o in self.outputs) else []
        if any(o.tone_sign for o in self.outputs):
            fs += [t.freq for t in self.active_tones()]
        return max(fs) if fs else 0.0

    # ---- storage ----
    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        d["outputs"] = [Output(**o) for o in d.get("outputs", [])]
        d["tones"] = [Tone(**t) for t in d.get("tones", [])]
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})

    def copy(self, name=None):
        d = self.from_dict(self.to_dict())
        if name:
            d.name = name
        return d


@dataclass(frozen=True)
class TwSpec:
    """A TWAVE sequencer, hashable so the field solver can share parts between outputs."""
    key: str
    n_phases: int
    n_on: int
    amp: float
    step: float
    rise: float = 0.0

    def level(self, phase, t):
        """Gate of TWAVE phase p (0..1) at times t. Phase p is high while
        ((p - step) mod n) < n_on, step = floor(t / step_time); edges ramp over rise."""
        t = np.asarray(t, float)
        if not np.isfinite(self.step) or self.step <= 0:
            return np.zeros(t.shape)
        n, n_on = self.n_phases, self.n_on
        if n_on >= n:
            return np.ones(t.shape)
        s_on = phase - n_on + 1                            # step at which this phase switches on
        tau = np.mod(t / self.step - s_on, n)              # steps since switch-on
        r = min(self.rise / self.step, n_on, n - n_on)     # an edge cannot outlast its level
        if r <= 0:
            return (tau < n_on).astype(float)
        up = np.clip(tau / r, 0.0, 1.0)
        down = np.clip(1.0 - (tau - n_on) / r, 0.0, 1.0)
        return np.where(tau < n_on, up, down)


# ---------- templates ----------

def twave_rf_combiner(name="TWAVE/RF combiner"):
    """16 outputs: TWAVE phases 1-8, each with RF+ and RF-."""
    outs = []
    for k in range(8):
        outs.append(Output(f"TW{k + 1} RF+", rf=100.0, rf_phase=0.0, tw=k))
        outs.append(Output(f"TW{k + 1} RF-", rf=100.0, rf_phase=180.0, tw=k))
    return Driver(name, rf_freq=1.0e6, outputs=outs, tw_amp=20.0, tw_speed=200.0,
                  notes="TWAVE phases 1-8 superimposed on RF+ and RF- (16 outputs).")


def quad_excitation(name="Quad RF + 8 tones", dipolar=False):
    """Quadrupole RF/DC with 8 excitation tones superimposed. Quadrupolar: tones follow the
    RF (+ on rods A, - on rods B). Dipolar: tones across the x rod pair only."""
    amp = 1.0 if dipolar else 5.0       # quadrupolar (parametric) excitation needs ~1-2% of the RF
    tones = [Tone(freq=f, amp=amp) for f in (60e3, 80e3, 100e3, 120e3, 140e3, 160e3, 180e3, 200e3)]
    if dipolar:
        outs = [Output("Rod x+", rf=300.0, tone_sign=1.0), Output("Rod x-", rf=300.0, tone_sign=-1.0),
                Output("Rods y", rf=300.0, rf_phase=180.0)]
        note = "Dipolar excitation: tones on the x rods in antiphase, RF on all rods."
    else:
        outs = [Output("Rods A (x)", rf=300.0, tone_sign=1.0),
                Output("Rods B (y)", rf=300.0, rf_phase=180.0, tone_sign=-1.0)]
        note = "Tones superimposed on the RF: + with RF+ on rods A, - with RF- on rods B."
    return Driver(name, rf_freq=1.0e6, outputs=outs, tones=tones, notes=note)


def rf_pair(name="RF pair"):
    return Driver(name, rf_freq=1.0e6, outputs=[Output("RF+", rf=50.0), Output("RF-", rf=50.0, rf_phase=180.0)],
                  notes="Two-phase RF for funnels, ion guides and TWAVE RF guard strips.")


def dc_supply(name="DC supply", n=8):
    return Driver(name, outputs=[Output(f"DC{k + 1}") for k in range(n)], notes=f"{n} DC channels.")


TEMPLATES = {
    "TWAVE/RF combiner (16 outputs)": twave_rf_combiner,
    "Quad RF + 8 tones (on the RF)": quad_excitation,
    "Quad RF + 8 tones (dipolar, x rods)": lambda name="Quad dipolar": quad_excitation(name, dipolar=True),
    "RF pair (RF+ / RF-)": rf_pair,
    "DC supply (8 channels)": dc_supply,
    "Empty": lambda name="New driver": Driver(name, outputs=[Output("Out 1")]),
}


# ---------- library ----------

class DriverLibrary:
    """Named drivers, saved as one JSON file."""

    def __init__(self, path=LIBRARY_PATH):
        self.path = path
        self.drivers = {}

    def names(self):
        return list(self.drivers)

    def get(self, name):
        return self.drivers[name]

    def add(self, drv, replace=True):
        if not replace:
            base, k = drv.name, 2
            while drv.name in self.drivers:
                drv.name = f"{base} ({k})"; k += 1
        self.drivers[drv.name] = drv
        return drv.name

    def remove(self, name):
        self.drivers.pop(name, None)

    def rename(self, old, new):
        d = self.drivers.pop(old)
        d.name = new
        self.drivers[new] = d

    def resolve(self, ref):
        """'Driver / Output' -> (Driver, Output) or None."""
        if not ref or " / " not in ref:
            return None
        dn, on = ref.split(" / ", 1)
        d = self.drivers.get(dn)
        if d is None:
            return None
        try:
            return d, d.output(on)
        except KeyError:
            return None

    def to_list(self):
        return [d.to_dict() for d in self.drivers.values()]

    def load(self, path=None):
        path = path or self.path
        if os.path.exists(path):
            with open(path) as f:
                data = json.load(f)
            for d in data.get("drivers", data if isinstance(data, list) else []):
                drv = Driver.from_dict(d)
                self.drivers[drv.name] = drv
        return self

    def save(self, path=None, names=None):
        path = path or self.path
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        ds = [self.drivers[n].to_dict() for n in (names or self.drivers)]
        with open(path, "w") as f:
            json.dump({"drivers": ds}, f, indent=1)

    @classmethod
    def with_defaults(cls, path=LIBRARY_PATH):
        lib = cls(path).load()
        if not lib.drivers:
            for make in (twave_rf_combiner, quad_excitation, rf_pair):
                lib.add(make())
        return lib
