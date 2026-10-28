"""Orthogonal-acceleration time-of-flight (OA-TOF): pusher, acceleration, drift,
reflectron and detector.

The incoming beam travels along z through the pusher gap. A pulse on the pusher plate
(and optionally the puller grid) pushes a slice of it along x, through the acceleration
gap into the field-free drift tube, into a single- or two-stage reflectron and back to
the detector. With no reflectron the detector sits at the end of the drift tube.

Every gap is bounded by ideal grids, so the field in each gap is uniform along x and
depends only on time (the pusher pulse). The flight is solved exactly gap by gap:
inside a gap the motion is a parabola, so the time to reach the next grid is the root
of a quadratic. While the pulse is rising or falling the gap fields are held for short
steps (midpoint values), which is second-order accurate in the step; everywhere else the
flight is analytic to round-off. A run of 10 000 ions takes well under a second.

What it covers: spatial and turnaround (velocity) spread in the pusher, pulse rise and
shape, grid transmission, single- and two-stage reflectrons, the drift angle from the
beam energy (where ions land on the detector), detector size, timing jitter and
digitizer bins, duty cycle. What it does not: field penetration through grid meshes,
fringe fields at the edges of the plates, space charge, gridless reflectrons (use the
Device builder for those), metastable decay.

Coordinates: x is the flight axis (0 = pusher plate), z the incoming beam axis, y the
third direction. Ions keep their z and y velocity through the whole flight.
"""
import numpy as np
from dataclasses import dataclass, field, asdict, replace
from .constants import AMU, E_CHARGE
from .drivers import pulse_level

REFLECTRONS = ("two-stage", "single-stage", "none")
SHAPES = ("linear", "rc")

ST_FLYING, ST_DETECTED, ST_HIT, ST_MISSED, ST_STUCK = 0, 1, 2, 3, 4
STATUS_NAMES = {ST_DETECTED: "detected", ST_HIT: "hit a plate", ST_MISSED: "missed the detector",
                ST_STUCK: "never arrived"}


@dataclass
class Pulse:
    """Pusher timing. g(t) runs 0 -> 1 -> 0. rise and fall are 10-90 % times."""
    delay: float = 0.0          # s, from t = 0 (when the beam slice is frozen) to the start of the rise
    rise: float = 10e-9
    width: float = 5e-6         # s, start of rise to start of fall
    fall: float = 10e-9
    shape: str = "linear"       # "linear" ramp or "rc" (exponential, 1 - exp(-t / tau))
    rate: float = 10e3          # Hz, pulses per second (duty cycle and the m/z range without overlap)

    def _ramp_len(self, t10_90):
        if t10_90 <= 0:
            return 0.0
        return t10_90 / 0.8 if self.shape == "linear" else 10 * t10_90 / np.log(9)   # rc: settle to e^-10

    def level(self, t):
        return pulse_level(t, self.delay, self.rise, self.width, self.fall, self.shape)

    @classmethod
    def from_driver(cls, drv):
        return cls(delay=drv.pulse_delay, rise=drv.pulse_rise, width=drv.pulse_width, fall=drv.pulse_fall,
                   shape=drv.pulse_shape, rate=drv.pulse_rate)

    def schedule(self):
        """[(t0, t1, changing)] covering [0, inf): changing intervals are stepped."""
        r, f = self._ramp_len(self.rise), self._ramp_len(self.fall)
        pts = [(0.0, self.delay, False), (self.delay, self.delay + r, True),
               (self.delay + r, self.delay + self.width, False),
               (self.delay + self.width, self.delay + self.width + f, True),
               (self.delay + self.width + f, np.inf, False)]
        out = []
        for a, b, ch in pts:
            b = max(a, b)
            if b > a:
                out.append((a, b, ch))
            if b == np.inf:
                break
        return out


@dataclass
class OATof:
    # pusher and acceleration (potentials in V)
    push_gap: float = 5e-3       # pusher plate to puller grid (m)
    accel_gap: float = 15e-3     # puller grid to drift grid (m)
    v_rest: float = 0.0          # pusher plate between pulses (the beam's potential)
    v_rest_pull: float = None    # puller grid between pulses; None = same as the pusher plate
    v_push: float = 600.0        # added to the pusher plate during the pulse
    v_pull: float = -600.0       # added to the puller grid during the pulse
    v_drift: float = -6000.0     # drift tube and reflectron entrance grid
    pulse: Pulse = field(default_factory=Pulse)
    # flight tube
    drift_length: float = 0.60   # drift grid to reflectron entrance (or to the detector, linear)
    reflectron: str = "two-stage"
    r1_depth: float = 0.02       # first (decelerating) stage, m
    v_r1: float = -2000.0        # potential at the end of stage 1
    r2_depth: float = 0.20       # second stage (single-stage: the only stage), m
    v_r2: float = 4975.0         # reflectron back plate
    det_distance: float = 0.59   # reflectron entrance back to the detector (m); None = 98 % of drift_length
    det_z: float = None          # detector centre along z (m); None = centred on the first species
    det_width_z: float = 40e-3
    det_width_y: float = 40e-3
    grid_transmission: float = 0.90   # per grid crossing
    det_jitter: float = 0.2e-9        # s, 1 sigma (detector + timing electronics)
    det_bin: float = 0.25e-9          # s, digitizer bin

    # ---- planes along x ----
    def planes(self):
        """(x, kind, name) of every plane in order. kind: wall, grid or det."""
        x1 = self.push_gap
        x2 = x1 + self.accel_gap
        x3 = x2 + self.drift_length
        if self.reflectron == "none":
            return [(0.0, "wall", "pusher"), (x1, "grid", "puller"), (x2, "grid", "drift grid"), (x3, "det", "detector")]
        dd = self.det_dist()
        xd = x3 - dd
        if not x2 < xd < x3:
            raise ValueError(f"detector distance {dd * 1e3:.1f} mm puts it outside the drift tube "
                             f"(0 to {self.drift_length * 1e3:.1f} mm from the reflectron)")
        out = [(0.0, "wall", "pusher"), (x1, "grid", "puller"), (x2, "grid", "drift grid"), (xd, "det", "detector"),
               (x3, "grid", "reflectron entrance")]
        if self.reflectron == "two-stage":
            out.append((x3 + self.r1_depth, "grid", "reflectron stage 1"))
        out.append((out[-1][0] + self.r2_depth, "wall", "reflectron back"))
        return out

    def det_dist(self):
        return 0.98 * self.drift_length if self.det_distance is None else self.det_distance

    def potentials(self, t):
        """Potential of every plane at time t (scalar t)."""
        g = float(self.pulse.level(t))
        pull = self.v_rest if self.v_rest_pull is None else self.v_rest_pull
        v = [self.v_rest + self.v_push * g, pull + self.v_pull * g, self.v_drift, self.v_drift]
        if self.reflectron != "none":
            v.append(self.v_drift)
            if self.reflectron == "two-stage":
                v.append(self.v_r1)
            v.append(self.v_r2)
        return np.array(v)

    def fields(self, t):
        """E_x in each gap (V/m) at time t."""
        x = np.array([p[0] for p in self.planes()])
        return -np.diff(self.potentials(t)) / np.diff(x)

    def potential_profile(self, t, n=2000):
        """(x, V) along the flight axis."""
        x = np.array([p[0] for p in self.planes()])
        xs = np.linspace(x[0], x[-1], n)
        return xs, np.interp(xs, x, self.potentials(t))

    def copy(self, **kw):
        return replace(self, pulse=replace(self.pulse), **kw)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        d = dict(d)
        p = Pulse(**d.pop("pulse", {}))
        known = set(cls.__dataclass_fields__)
        return cls(pulse=p, **{k: v for k, v in d.items() if k in known})


@dataclass
class Beam:
    """The continuous beam in the pusher at the moment of the pulse."""
    energy_ev: float = 10.0       # kinetic energy along z (per charge: eV for z = 1)
    energy_spread_ev: float = 0.5 # 1 sigma
    x0: float = None              # beam centre, from the pusher plate (m); None = middle of the gap
    sigma_x: float = 0.5e-3       # beam size along the push direction, 1 sigma
    sigma_y: float = 0.5e-3
    divergence_deg: float = 0.5   # 1 sigma angle to the z axis, in x and in y (sets the turnaround time)
    window: float = 20e-3         # length of the pusher window along z (the slice that is pushed)
    temperature_k: float = 0.0    # extra thermal velocity spread (adds to the divergence), e.g. a cooled beam


@dataclass
class TofResult:
    tof: OATof
    species: np.ndarray         # per ion, index into names
    names: list
    mz: np.ndarray              # per species
    status: np.ndarray          # per ion
    t: np.ndarray               # arrival time (s), NaN if not detected
    weight: np.ndarray          # grid transmission along the path
    x0: np.ndarray
    vx0: np.ndarray
    z_hit: np.ndarray           # where each ion crossed the detector plane (z, y)
    y_hit: np.ndarray
    vz: np.ndarray
    traj: list                  # recorded ions: list of (t, x, v, a) segment arrays
    traj_ids: np.ndarray
    z0: np.ndarray
    y0: np.ndarray
    energy: np.ndarray          # kinetic energy along x in the drift tube (eV)
    beam: Beam = None
    det_z: float = 0.0
    t_end: np.ndarray = None    # per ion, when it stopped (detected, hit, missed)

    def arrival(self, k):
        m = (self.species == k) & (self.status == ST_DETECTED)
        return self.t[m], self.weight[m]

    def stats(self, rng=None):
        """Per species: dict(name, mz, n, detected, transmission, t_mean, fwhm, resolution, duty)."""
        rng = rng or np.random.default_rng(1)
        out = []
        for k, name in enumerate(self.names):
            sel = self.species == k
            t, w = self.arrival(k)
            tj = t + self.tof.det_jitter * rng.standard_normal(len(t)) if len(t) else t
            fw = fwhm(tj, w)
            tm = float(np.average(t, weights=w)) if len(t) and w.sum() > 0 else np.nan
            vz = float(np.median(self.vz[sel])) if sel.any() else np.nan
            det = (self.status[sel] == ST_DETECTED)
            frac = float(np.sum(self.weight[sel][det]) / max(sel.sum(), 1))
            duty = min(1.0, self.beam.window / (vz * (1 / self.tof.pulse.rate))) if self.beam and vz > 0 else np.nan
            out.append(dict(name=name, mz=float(self.mz[k]), n=int(sel.sum()), detected=int(det.sum()),
                            transmission=frac, t_mean=tm, fwhm=fw,
                            resolution=tm / (2 * fw) if fw > 0 else np.nan, duty=duty * frac if duty == duty else np.nan,
                            fates={STATUS_NAMES[s]: int(np.sum(self.status[sel] == s)) for s in STATUS_NAMES}))
        return out

    def calibration(self):
        """(t0, k) with t = t0 + k sqrt(m/z), fitted to the species' mean arrival times."""
        st = [s for s in self.stats() if np.isfinite(s["t_mean"])]
        if not st:
            return 0.0, np.nan
        sq = np.sqrt([s["mz"] for s in st]); tm = np.array([s["t_mean"] for s in st])
        if len(st) == 1 or np.ptp(sq) == 0:
            return 0.0, float(tm[0] / sq[0])
        k, t0 = np.polyfit(sq, tm, 1)
        return float(t0), float(k)

    def spectrum(self, rng=None):
        """Detected signal: (bin centres in s, counts per bin, m/z of each bin centre)."""
        rng = rng or np.random.default_rng(2)
        m = self.status == ST_DETECTED
        if not m.any():
            return np.zeros(0), np.zeros(0), np.zeros(0)
        t = self.t[m] + self.tof.det_jitter * rng.standard_normal(m.sum())
        b = max(self.tof.det_bin, 1e-12)
        lo, hi = np.floor(t.min() / b) - 20, np.ceil(t.max() / b) + 20
        if hi - lo > 2_000_000:
            b = (hi - lo) * b / 2_000_000
            lo, hi = np.floor(t.min() / b) - 20, np.ceil(t.max() / b) + 20
        edges = np.arange(lo, hi + 1) * b
        c, _ = np.histogram(t, edges, weights=self.weight[m])
        tc = 0.5 * (edges[1:] + edges[:-1])
        t0, k = self.calibration()
        mz = np.where(tc > t0, ((tc - t0) / k) ** 2, np.nan) if np.isfinite(k) else np.full(len(tc), np.nan)
        return tc, c, mz

    def paths(self, n_pts=80):
        """Sampled (t, x, z) of each recorded ion."""
        out = []
        for i, seg in zip(self.traj_ids, self.traj):
            ts, xs = [], []
            for j in range(len(seg)):
                t0, x0, v0, a = seg[j]
                t1 = seg[j + 1][0] if j + 1 < len(seg) else (self.t_end[i] if np.isfinite(self.t_end[i]) else t0)
                tt = np.linspace(t0, t1, max(2, n_pts // max(len(seg), 1))) if t1 > t0 else np.array([t0])
                s = tt - t0
                ts.append(tt); xs.append(x0 + v0 * s + 0.5 * a * s * s)
            t = np.concatenate(ts); x = np.concatenate(xs)
            out.append((t, x, self.z0[i] + self.vz[i] * t))
        return out


def fwhm(t, w=None):
    """Full width at half maximum of a peak of arrival times (histogram, interpolated edges)."""
    t = np.asarray(t, float)
    if len(t) < 5 or np.ptp(t) == 0:
        return 0.0 if len(t) else np.nan
    w = np.ones(len(t)) if w is None else np.asarray(w, float)
    nb = int(np.clip(np.sqrt(len(t)) * 1.5, 15, 200))
    sd = np.sqrt(np.cov(t, aweights=w)) if len(t) > 1 else 0
    lo, hi = np.percentile(t, [0.5, 99.5])
    span = max(hi - lo, sd * 1e-3, 1e-15)
    edges = np.linspace(lo - 0.1 * span, hi + 0.1 * span, nb + 1)
    c, _ = np.histogram(t, edges, weights=w)
    c = np.convolve(c, [0.25, 0.5, 0.25], mode="same")
    if c.max() <= 0:
        return np.nan
    half = c.max() / 2
    above = np.nonzero(c >= half)[0]
    i0, i1 = above[0], above[-1]
    x = 0.5 * (edges[1:] + edges[:-1])

    def cross(a, b):
        if c[b] == c[a]:
            return x[a]
        return x[a] + (half - c[a]) * (x[b] - x[a]) / (c[b] - c[a])
    left = cross(i0 - 1, i0) if i0 > 0 else x[i0]
    right = cross(i1, i1 + 1) if i1 + 1 < len(c) else x[i1]
    return float(max(right - left, edges[1] - edges[0]) if right > left else edges[1] - edges[0])


# ---------------------------------------------------------------------------------
def _first_hit(x, v, a, b):
    """Smallest time > 0 at which x + v t + a t^2 / 2 = b (inf if never)."""
    c = x - b
    A = 0.5 * a
    out = np.full(len(x), np.inf)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        lin = A == 0
        tl = np.where(lin & (v != 0), -c / np.where(v != 0, v, 1), np.inf)
        disc = v * v - 4 * A * c
        sq = np.sqrt(np.maximum(disc, 0))
        q = -0.5 * (v + np.where(v >= 0, 1.0, -1.0) * sq)
        r1 = np.where(A != 0, q / np.where(A != 0, A, 1), np.inf)
        r2 = np.where(q != 0, c / np.where(q != 0, q, 1), np.inf)
        r1 = np.where(r1 > 0, r1, np.inf); r2 = np.where(r2 > 0, r2, np.inf)
        tq = np.where(disc >= 0, np.minimum(r1, r2), np.inf)
        tl = np.where(tl > 0, tl, np.inf)
    out = np.where(lin, tl, tq)
    return out


class _State:
    def __init__(self, x, v, qm, n_rec):
        n = len(x)
        self.x, self.v, self.qm = x.astype(float), v.astype(float), qm
        self.t = np.zeros(n)
        self.reg = np.zeros(n, int)
        self.status = np.zeros(n, np.int8)
        self.weight = np.ones(n)
        self.t_det = np.full(n, np.nan)
        self.v_drift = np.full(n, np.nan)
        self.rec = list(range(min(n_rec, n)))
        self.segs = [[] for _ in self.rec]

    def record(self, idx, a):
        """Start a new segment for recorded ions in idx (with acceleration a)."""
        for j in np.nonzero(idx < len(self.segs))[0]:
            i = idx[j]
            self.segs[i].append((self.t[i], self.x[i], self.v[i], a[j]))


def _advance(st, t_cap, E, X, kinds, grid_T, max_hops=400):
    """Move every flying ion until t_cap with fixed gap fields E, hopping between gaps."""
    last = len(X) - 1
    for _ in range(max_hops):
        i = np.nonzero((st.status == ST_FLYING) & (st.t < t_cap))[0]
        if len(i) == 0:
            return
        r = st.reg[i]
        x, v = st.x[i], st.v[i]
        a = st.qm[i] * E[r]
        st.record(i, a)
        lo, hi = X[r], X[r + 1]
        tl, th = _first_hit(x, v, a, lo), _first_hit(x, v, a, hi)
        tau = np.minimum(tl, th)
        rem = t_cap - st.t[i]
        stays = tau >= rem
        if np.isinf(t_cap):
            stuck = np.isinf(tau)
            st.status[i[stuck]] = ST_STUCK
            stays = stuck
        dt = np.where(stays, np.where(np.isfinite(rem), rem, 0.0), tau)
        st.x[i] = np.where(stays, x + v * dt + 0.5 * a * dt * dt, 0.0)
        st.v[i] = v + a * dt
        st.t[i] = st.t[i] + dt
        cr = ~stays
        up = th < tl
        ic, upc = i[cr], up[cr]
        k = np.where(upc, st.reg[ic] + 1, st.reg[ic])          # plane crossed
        st.x[ic] = X[k]
        for kk in np.unique(k):
            m = k == kk
            ions, goes_up = ic[m], upc[m]
            kind = kinds[kk]
            if kind == "wall":
                st.status[ions] = ST_HIT
            elif kind == "det":
                # linear: detector at the end (moving +x); reflectron: on the way back (moving -x)
                hit = goes_up if kk == last else ~goes_up
                st.status[ions[hit]] = ST_DETECTED
                st.t_det[ions[hit]] = st.t[ions[hit]]
                if kk != last:
                    st.v_drift[ions[goes_up]] = st.v[ions[goes_up]]
                else:
                    st.v_drift[ions[hit]] = st.v[ions[hit]]
            else:
                st.weight[ions] *= grid_T
        move = st.status[ic] == ST_FLYING
        st.reg[ic[move]] = np.where(upc[move], st.reg[ic[move]] + 1, st.reg[ic[move]] - 1)
    st.status[(st.status == ST_FLYING) & (st.t < t_cap)] = ST_STUCK


def make_ions(tof, beam, species, rng):
    """Initial state in the pusher. species: list of IonSpecies (mz, z, n)."""
    from .ions import interleave
    from .constants import K_B
    idx = interleave([s.n for s in species])
    n = len(idx)
    mz = np.array([s.mz for s in species], float)[idx]
    zc = np.array([s.z for s in species], float)[idx]
    mass = mz * np.abs(zc) * AMU
    q = zc * E_CHARGE
    gp = tof.push_gap
    xc = gp / 2 if beam.x0 is None else beam.x0
    x0 = np.clip(xc + beam.sigma_x * rng.standard_normal(n), 0.02 * gp, 0.98 * gp)
    y0 = beam.sigma_y * rng.standard_normal(n)
    z0 = beam.window * (rng.random(n) - 0.5)
    ez = np.maximum(beam.energy_ev + beam.energy_spread_ev * rng.standard_normal(n), 1e-3) * np.abs(zc) * E_CHARGE
    vz = np.sqrt(2 * ez / mass)
    th = np.radians(beam.divergence_deg)
    sth = np.sqrt(K_B * beam.temperature_k / mass) if beam.temperature_k > 0 else 0.0
    vx = vz * np.tan(th * rng.standard_normal(n)) + sth * rng.standard_normal(n)
    vy = vz * np.tan(th * rng.standard_normal(n)) + sth * rng.standard_normal(n)
    return idx, mz, q, mass, x0, vx, y0, vy, z0, vz


def fly(tof, beam, species, rng=None, n_record=40, steps_per_edge=60):
    """Fly a pushed beam slice through the OA-TOF. species: list of IonSpecies."""
    rng = rng or np.random.default_rng()
    pl = tof.planes()
    X = np.array([p[0] for p in pl]); kinds = [p[1] for p in pl]
    idx, mz, q, mass, x0, vx, y0, vy, z0, vz = make_ions(tof, beam, species, rng)
    st = _State(x0, vx, q / mass, n_record)
    for a, b, changing in tof.pulse.schedule():
        if not changing:
            _advance(st, b, tof.fields(a if np.isfinite(b) else a + 1e-12), X, kinds, tof.grid_transmission)
            continue
        nst = max(steps_per_edge, 4)
        ts = np.linspace(a, b, nst + 1)
        for t0, t1 in zip(ts[:-1], ts[1:]):
            _advance(st, t1, tof.fields(0.5 * (t0 + t1)), X, kinds, tof.grid_transmission)
    # landing point on the detector plane
    t = st.t_det
    z_hit = z0 + vz * np.where(np.isfinite(t), t, st.t)
    y_hit = y0 + vy * np.where(np.isfinite(t), t, st.t)
    zc0 = float(np.median(z_hit[(idx == 0) & np.isfinite(t)])) if np.any((idx == 0) & np.isfinite(t)) else 0.0
    det_z = zc0 if tof.det_z is None else tof.det_z
    det = st.status == ST_DETECTED
    out = det & ((np.abs(z_hit - det_z) > tof.det_width_z / 2) | (np.abs(y_hit) > tof.det_width_y / 2))
    st.status[out] = ST_MISSED
    energy = 0.5 * mass * st.v_drift ** 2 / E_CHARGE
    names = [s.name for s in species]
    ids = np.array(st.rec, int)
    return TofResult(tof=tof, species=idx, names=names, mz=np.array([s.mz for s in species], float),
                     status=st.status, t=np.where(st.status == ST_DETECTED, st.t_det, np.nan), weight=st.weight,
                     x0=x0, vx0=vx, z_hit=z_hit, y_hit=y_hit, vz=vz, traj=[np.array(s) for s in st.segs],
                     traj_ids=ids, z0=z0, y0=y0, energy=energy, beam=beam, det_z=det_z, t_end=st.t)


# ---------------------------------------------------------------------------------
# what limits the resolution
def broadening(tof, beam, species, n=3000, rng=None):
    """FWHM of the first species' peak with one spread at a time: [(label, fwhm s)]."""
    sp = [replace(species[0], n=n)]
    base = dict(sigma_x=0.0, divergence_deg=0.0, temperature_k=0.0, energy_spread_ev=0.0)
    cases = [("Beam width (spatial spread)", dict(sigma_x=beam.sigma_x)),
             ("Turnaround (velocity spread)", dict(divergence_deg=beam.divergence_deg, temperature_k=beam.temperature_k))]
    out = []
    quiet = tof.copy(det_jitter=0.0)
    for label, kw in cases:
        b = replace(beam, **{**base, **kw})
        r = fly(quiet, b, sp, rng=np.random.default_rng(3), n_record=0)
        t, w = r.arrival(0)
        out.append((label, fwhm(t, w)))
    out.append(("Detector and timing jitter", 2.3548 * tof.det_jitter))
    r = fly(tof, beam, sp, rng=np.random.default_rng(3), n_record=0)
    st = r.stats()[0]
    out.append(("All together", st["fwhm"]))
    return out, st["t_mean"]


def resolution(tof, beam, species, n=1500, seed=4):
    """Resolution (t / 2 FWHM) of the first species."""
    r = fly(tof, beam, [replace(species[0], n=n)], rng=np.random.default_rng(seed), n_record=0)
    s = r.stats()[0]
    return s["resolution"] if np.isfinite(s["resolution"]) else 0.0


SCAN_PARAMS = {
    "Reflectron back plate (V)": ("v_r2", 1.0),
    "Reflectron stage 1 (V)": ("v_r1", 1.0),
    "Pusher pulse (V)": ("v_push", 1.0),
    "Puller pulse (V)": ("v_pull", 1.0),
    "Drift tube (V)": ("v_drift", 1.0),
    "Drift length (mm)": ("drift_length", 1e-3),
    "Detector distance (mm)": ("det_distance", 1e-3),
    "Reflectron stage 2 depth (mm)": ("r2_depth", 1e-3),
    "Pulse rise time (ns)": ("pulse.rise", 1e-9),
}


def with_param(tof, key, value):
    if key.startswith("pulse."):
        t = tof.copy()
        setattr(t.pulse, key.split(".", 1)[1], value)
        return t
    return tof.copy(**{key: value})


def scan(tof, beam, species, key, values, n=1500, progress=None):
    """Resolution and mean flight time of the first species over values of one setting."""
    R, T = [], []
    for j, v in enumerate(values):
        try:
            t = with_param(tof, key, v)
            r = fly(t, beam, [replace(species[0], n=n)], rng=np.random.default_rng(4), n_record=0)
            s = r.stats()[0]
            R.append(s["resolution"] if np.isfinite(s["resolution"]) else 0.0); T.append(s["t_mean"])
        except ValueError:
            R.append(np.nan); T.append(np.nan)
        if progress and progress((j + 1) / len(values)):
            break
    return np.array(R), np.array(T)


def tune(tof, beam, species, keys=("v_r2", "v_r1"), n=800, rounds=3, progress=None):
    """Adjust reflectron voltages for the best resolution of the first species
    (coordinate search, golden section on each voltage in turn). Returns (tof, history)."""
    keys = [k for k in keys if tof.reflectron == "two-stage" or k != "v_r1"]
    best = tof.copy()
    hist = []
    k_ion = nominal_energy(best, beam)
    total = rounds * len(keys)
    done = 0
    for rd in range(rounds):
        for key in keys:
            v0 = getattr(best, key)
            span = max(abs(k_ion), 100.0) * (0.25 / (rd + 1))
            f = lambda v: -resolution(with_param(best, key, v), beam, species, n=n)
            a, b = v0 - span, v0 + span
            g = (np.sqrt(5) - 1) / 2
            c, d = b - g * (b - a), a + g * (b - a)
            fc, fd = f(c), f(d)
            for _ in range(14):
                if fc < fd:
                    b, d, fd = d, c, fc
                    c = b - g * (b - a); fc = f(c)
                else:
                    a, c, fc = c, d, fd
                    d = a + g * (b - a); fd = f(d)
            v = 0.5 * (a + b)
            if -f(v) >= -f(v0):
                best = with_param(best, key, v)
            hist.append((key, getattr(best, key), resolution(best, beam, species, n=n)))
            done += 1
            if progress and progress(done / total):
                return best, hist
    return best, hist


def _pull_on(tof):
    """Puller grid potential while the pulse is high."""
    return (tof.v_rest if tof.v_rest_pull is None else tof.v_rest_pull) + tof.v_pull


def nominal_energy(tof, beam):
    """Kinetic energy (eV per charge) an ion at the beam centre gains: start potential - drift."""
    gp = tof.push_gap
    xc = gp / 2 if beam.x0 is None else beam.x0
    vb, vg = tof.v_rest + tof.v_push, _pull_on(tof)
    return (vb + (vg - vb) * xc / gp) - tof.v_drift


def suggest_reflectron(tof, beam):
    """First-order energy focusing of an ideal reflectron at the nominal energy K.
    Single-stage: the turning depth is a quarter of the field-free path (out and back),
    counted from the source's space-focus plane. Two-stage: stage 1 takes 2/3 of K (the usual Mamyrin split) and the back plate
    is set so the flight time does not change with energy. Returns (new OATof, note)."""
    K = nominal_energy(tof, beam)
    s, Ka = (1.0 if K >= 0 else -1.0), abs(K)
    t = tof.copy()
    note = ""
    L = tof.drift_length + tof.det_dist()
    D = wiley_mclaren_focus(tof, beam)          # the reflectron refocuses from the source's space focus
    if np.isfinite(D) and 0 < D < 0.8 * tof.drift_length:
        L -= D
    if Ka <= 0 or tof.reflectron == "none":
        return t, "No reflectron to set." if tof.reflectron == "none" else "The ions gain no energy."
    if tof.reflectron == "single-stage":
        p = L / 4
        if t.r2_depth < 1.15 * p:
            t.r2_depth = round(1.25 * p, 4)
            note = f"Reflectron depth increased to {t.r2_depth * 1e3:.0f} mm so the ions turn inside it. "
        t.v_r2 = tof.v_drift + s * Ka * t.r2_depth / p
        return t, note + f"Turning depth {p * 1e3:.1f} mm = a quarter of the {L * 1e3:.0f} mm path from the space focus."
    U1 = (2 / 3) * Ka
    t.v_r1 = tof.v_drift + s * U1
    d1, d2 = t.r1_depth, t.r2_depth

    def flight(k, U2):                       # time per sqrt(m/q), ideal grids
        v, v1 = np.sqrt(2 * k), np.sqrt(2 * (k - U1))
        return L / v + 2 * (v - v1) * d1 / U1 + 2 * v1 * d2 / U2

    def slope(U2):
        return flight(Ka * 1.001, U2) - flight(Ka * 0.999, U2)

    lo, hi = (Ka - U1) * 1.02, 100 * Ka
    flo, fhi = slope(lo), slope(hi)
    if flo * fhi > 0:
        return t, "No back-plate voltage gives energy focusing with these depths; try a deeper second stage."
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        fm = slope(mid)
        if fm * flo > 0:
            lo, flo = mid, fm
        else:
            hi = mid
    U2 = 0.5 * (lo + hi)
    t.v_r2 = t.v_r1 + s * U2
    pen = (Ka - U1) / U2 * d2
    return t, f"Stage 1 takes 2/3 of the {Ka:.0f} eV; ions turn {pen * 1e3:.1f} mm into stage 2."


def wiley_mclaren_focus(tof, beam):
    """Space-focus distance beyond the drift grid for the two-field source (first order),
    with the pulse treated as instant: D = 2 s0 k0^(3/2) (1 - d / (s0 (k0 + sqrt(k0)))), where
    s0 is the start position from the puller grid, d the acceleration gap and
    k0 = (s0 Es + d Ed) / (s0 Es)."""
    gp = tof.push_gap
    xc = gp / 2 if beam.x0 is None else beam.x0
    s0 = gp - xc
    es = (tof.v_rest + tof.v_push - _pull_on(tof)) / gp
    ed = (_pull_on(tof) - tof.v_drift) / tof.accel_gap
    d = tof.accel_gap
    if es == 0:
        return np.nan
    k0 = (s0 * es + d * ed) / (s0 * es)
    if k0 <= 0:
        return np.nan
    return 2 * s0 * k0 ** 1.5 * (1 - d / (s0 * (k0 + np.sqrt(k0))))
