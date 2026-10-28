"""ESI inlet: spray region -> inlet capillary -> free-jet expansion -> skimmer.

Everything is 1D along the axis x:
    x = -spray_gap  emitter tip; ions start `release` downstream, already desolvated
    x = 0           capillary entrance (atmosphere)
    x = L           capillary exit (choked, Mach 1)
    x = L + skim    skimmer / first lens

Gas
---
* Spray region: still air drawn into the capillary as a hemispherical sink flow,
  u = Q / (2 pi r^2), capped at the capillary entrance speed.
* Capillary: compressible flow with wall friction (Fanno flow), choked at the exit.
  The gas is taken to be at the capillary temperature at the entrance. The friction
  factor follows the Reynolds number (laminar 16/Re, turbulent Blasius 0.079 Re^-0.25).
* Free jet: Ashkenas-Sherman centreline Mach number for an under-expanded sonic jet,
  ended by the Mach disk at x_M = 0.67 D sqrt(P0/Pb).

Ions
----
Ions feel gas drag through their mobility and the local electric field:
    dv/dt = (u - v) / tau + (q/m) E,      tau = m K / q,   K = K0 * N0 / N
integrated with an exponential step that stays stable for any tau (tau is ~ns at
1 atm and grows by orders of magnitude in the jet).
At high E/N a constant mobility overstates the drift, so by default the drag uses a
hard-sphere correction: tau_eff = tau / sqrt(1 + (v - u)^2 / c^2), with c the mean
relative thermal speed. Drift then grows like sqrt(E) at high field, as in Wannier theory. Kinetic energy is (1/2) m v^2, and
the Wannier effective temperature T_eff = T + M_gas (v - u)^2 / (3k) tracks
collisional heating.
"""
import numpy as np
from dataclasses import dataclass
from .constants import AMU, E_CHARGE, K_B

N_LOSCHMIDT = 2.6867811e25
TORR = 133.322


@dataclass
class Gas:
    name: str = "N2"
    mass_amu: float = 28.0134
    gamma: float = 1.4
    mu_ref: float = 1.663e-5      # Pa s at T_ref (Sutherland)
    t_ref: float = 273.15
    sutherland: float = 107.0

    @property
    def R(self):
        return K_B / (self.mass_amu * AMU)

    def mu(self, T):
        return self.mu_ref * (T / self.t_ref) ** 1.5 * (self.t_ref + self.sutherland) / (T + self.sutherland)


@dataclass
class Inlet:
    # spray
    spray_gap: float = 5e-3        # emitter tip to capillary entrance, m
    tip_radius: float = 50e-6      # m
    v_spray: float = 3500.0        # emitter minus capillary voltage, V
    release: float = 1e-3          # distance from the tip where ions leave the droplets, m
    # capillary
    diameter: float = 0.5e-3       # m
    length: float = 60e-3          # m
    t_cap: float = 548.0           # capillary / gas temperature, K (275 C)
    p_atm: float = 760.0           # Torr
    t_atm: float = 298.0           # K
    # first vacuum stage
    p_back: float = 1.5            # Torr
    skim_dist: float = 3e-3        # capillary exit to skimmer, m
    v_exit: float = 30.0           # capillary minus skimmer voltage, V
    gas: Gas = None

    def __post_init__(self):
        self.gas = self.gas or Gas()
        self.solve_gas()

    # ---------- capillary: Fanno flow ----------
    @staticmethod
    def fanno(M, g):
        M2 = M * M
        return (1 - M2) / (g * M2) + (g + 1) / (2 * g) * np.log((g + 1) * M2 / (2 + (g - 1) * M2))

    def _m_from_fanno(self, target, g):
        """Subsonic Mach number with fanno(M) = target (target >= 0)."""
        target = np.atleast_1d(target)
        lo, hi = np.full(target.shape, 1e-4), np.ones(target.shape)
        for _ in range(60):
            mid = 0.5 * (lo + hi)
            big = self.fanno(mid, g) > target       # fanno decreases with M on (0, 1)
            lo = np.where(big, mid, lo); hi = np.where(big, hi, mid)
        return 0.5 * (lo + hi)

    def solve_gas(self, nx=600):
        gs = self.gas; g = gs.gamma; R = gs.R
        T0, P0, D, L = self.t_cap, self.p_atm * TORR, self.diameter, self.length
        f = 0.005
        for _ in range(50):
            M1 = self._m_from_fanno(4 * f * L / D, g)[0]
            T1 = T0 / (1 + 0.5 * (g - 1) * M1 ** 2)
            P1 = P0 * (T1 / T0) ** (g / (g - 1))
            G = P1 / (R * T1) * M1 * np.sqrt(g * R * T1)        # mass flux, kg/m^2/s
            Re = G * D / gs.mu(T0)
            f_new = 16 / Re if Re < 2300 else 0.079 * Re ** -0.25
            if abs(f_new - f) < 1e-9:
                break
            f = f_new
        self.friction, self.reynolds, self.mass_flux = f, Re, G
        self.mdot = G * np.pi * D ** 2 / 4
        rho_std = 101325 / (R * 273.15)
        self.flow_slm = self.mdot / rho_std * 60e3                # standard L/min
        x = np.linspace(0, L, nx)
        M = self._m_from_fanno(self.fanno(M1, g) - 4 * f * x / D, g)
        M[-1] = 1.0
        T = T0 / (1 + 0.5 * (g - 1) * M ** 2)
        u = M * np.sqrt(g * R * T)
        P = G / u * R * T
        self.cap = dict(x=x, M=M, T=T, u=u, P=P, N=P / (K_B * T))

        # ---------- free jet from the sonic exit ----------
        Ts, Ps = T[-1], P[-1]
        P0e = Ps * ((g + 1) / 2) ** (g / (g - 1))                 # stagnation pressure of the jet
        self.x_mach_disk = 0.67 * D * np.sqrt(P0e / (self.p_back * TORR))
        self._jet0 = (T0, P0e)
        xs = np.linspace(0, max(self.skim_dist, 1e-6), 800)
        Mj, Tj, uj, Pj = self.jet_centre(xs)
        self.jet = dict(x=L + xs, M=Mj, T=Tj, u=uj, P=Pj, N=Pj / (K_B * Tj))
        self.u_terminal = np.sqrt(2 * g / (g - 1) * R * T0)

        # ---------- spray region ----------
        xsp = np.linspace(-self.spray_gap, 0, 400)
        a = D / 2
        # sink flow u = Q / (2 pi r^2) far away, blended to the entrance speed at r = 0
        usp = u[0] * a ** 2 / (a ** 2 + 2 * xsp ** 2)
        Psp = np.full_like(xsp, P0)
        Tsp = np.full_like(xsp, self.t_atm)
        self.spray = dict(x=xsp, M=usp / np.sqrt(g * R * Tsp), T=Tsp, u=usp, P=Psp, N=Psp / (K_B * Tsp))

        # one table along the whole axis
        parts = [self.spray, self.cap, self.jet]
        self.axis = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        self.x_end = L + self.skim_dist

    # ---------- electric field ----------
    def efield(self, x):
        """Axial field, V/m. Spray: hyperbolic-tip falloff normalised to v_spray.
        Capillary: field-free. Jet: uniform v_exit over skim_dist."""
        E = np.zeros_like(x)
        s = x + self.spray_gap                        # distance from the tip
        sp = x < 0
        rt = self.tip_radius
        E[sp] = 2 * self.v_spray / ((rt + 2 * s[sp]) * np.log(1 + 2 * self.spray_gap / rt))
        jet = x > self.length
        E[jet] = self.v_exit / max(self.skim_dist, 1e-9)
        return E

    def gas_at(self, x):
        ax = self.axis
        return {k: np.interp(x, ax["x"], ax[k]) for k in ("u", "T", "N", "P", "M")}

    def jet_centre(self, xs):
        """Mach number, T, speed and pressure on the jet axis at distances xs (m, increasing) from the
        capillary exit: Ashkenas-Sherman, then a normal shock at the Mach disk and subsonic gas at the
        back pressure."""
        gs = self.gas; g = gs.gamma; R = gs.R
        T0, P0e = self._jet0
        D = self.diameter
        r = np.asarray(xs, float) / D
        near = 1.0 + 3.337 * r ** 2 - 1.541 * r ** 3
        A, x0 = 3.65, 0.40
        far_arg = np.maximum(r - x0, 1e-6) ** (g - 1)
        far = A * far_arg - 0.5 * (g + 1) / (g - 1) / (A * far_arg)
        w = np.clip((r - 0.7) / 0.5, 0, 1)
        Mj = np.maximum.accumulate(np.maximum((1 - w) * near + w * far, 1.0))
        Tj = T0 / (1 + 0.5 * (g - 1) * Mj ** 2)
        Pj = P0e * (Tj / T0) ** (g / (g - 1))
        uj = Mj * np.sqrt(g * R * Tj)
        past = np.asarray(xs) >= self.x_mach_disk
        if past.any():                              # normal shock, then subsonic at back pressure
            i = np.argmax(past)
            M_md = Mj[i]
            M2 = np.sqrt((1 + 0.5 * (g - 1) * M_md ** 2) / (g * M_md ** 2 - 0.5 * (g - 1)))
            T2 = T0 / (1 + 0.5 * (g - 1) * M2 ** 2)
            Mj[past], Tj[past] = M2, T2
            uj[past] = M2 * np.sqrt(g * R * T2)
            Pj[past] = self.p_back * TORR
        return Mj, Tj, uj, Pj

    # ---------- the flow off the axis (for pictures; the ions fly on the axis tables above) ----------
    def jet_radius(self, xi):
        """Radius of the jet boundary (barrel shock) at xi metres after the exit: from the exit radius
        out to 0.375 x_M (a barrel about 0.75 x_M across) at the Mach disk, then held."""
        a, xm = self.diameter / 2, self.x_mach_disk
        s = np.clip(np.asarray(xi, float) / xm, 0, 1)
        return a + (max(0.375 * xm, a) - a) * np.sin(0.5 * np.pi * s) ** 0.6

    def velocity_2d(self, x, r):
        """Gas velocity (u_x, u_r) in m/s at axial x and radius r (m; r may be negative, u_r follows its
        sign), on the inlet's axis coordinates. Spray region: sink flow into the capillary entrance.
        Capillary: the 1D mean speed with a laminar (parabolic) or turbulent (1/7 power) profile, zero in
        the wall. Jet: radial from the exit inside the barrel, with the axis speed at the same distance;
        past the Mach disk the subsonic core slows over a few x_M; still background gas outside."""
        x, r = np.broadcast_arrays(np.asarray(x, float), np.asarray(r, float))
        ux, ur = np.zeros(x.shape), np.zeros(x.shape)
        a, L = self.diameter / 2, self.length
        ar = np.abs(r)
        sp = x < 0
        if sp.any():
            rho = np.hypot(x[sp], r[sp])
            u0 = self.cap["u"][0]
            mag = u0 * a ** 2 / (a ** 2 + 2 * rho ** 2)
            ux[sp] = -mag * x[sp] / np.maximum(rho, 1e-12)
            ur[sp] = -mag * r[sp] / np.maximum(rho, 1e-12)
        cap = (x >= 0) & (x <= L) & (ar < a)
        if cap.any():
            um = np.interp(x[cap], self.cap["x"], self.cap["u"])
            s = ar[cap] / a
            prof = 2 * (1 - s ** 2) if self.reynolds < 2300 else (60 / 49) * (1 - s) ** (1 / 7)
            ux[cap] = um * prof
        jet = x > L
        if jet.any():
            xi, rr = x[jet] - L, r[jet]
            xm = self.x_mach_disk
            inside = np.abs(rr) <= self.jet_radius(xi)
            rho = np.hypot(xi, rr)
            before = xi < xm
            u = np.zeros(xi.shape)
            if before.any():
                u[before] = self.jet_centre(np.sort(rho[before]))[2][np.argsort(np.argsort(rho[before]))]
            after = ~before
            if after.any():
                u2 = self.jet_centre(np.array([0.0, xm]))[2][-1]
                u[after] = u2 * np.exp(-(xi[after] - xm) / (2 * xm))
            u = np.where(inside, u, 0.0)
            dirx = np.where(before, xi / np.maximum(rho, 1e-12), 1.0)
            dirr = np.where(before, rr / np.maximum(rho, 1e-12), 0.0)
            ux[jet], ur[jet] = u * dirx, u * dirr
        return ux, ur


@dataclass
class InletIon:
    name: str
    mz: float
    charge_state: int = 1
    K0: float = 1.0                 # cm^2/Vs

    @property
    def mass(self):
        return self.mz * self.charge_state * AMU


def fly_inlet(inlet, ions, n_rec=4000, high_field=True, progress=None):
    """Deterministic axial flight of each ion species. Returns per-species dict arrays."""
    q = np.array([i.charge_state * E_CHARGE for i in ions])
    m = np.array([i.mass for i in ions])
    K0 = np.array([i.K0 * 1e-4 for i in ions])
    mg = inlet.gas.mass_amu * AMU
    n = len(ions)
    x = np.full(n, -inlet.spray_gap + min(max(inlet.release, inlet.tip_radius), 0.9 * inlet.spray_gap))
    v = np.zeros(n)
    t = np.zeros(n)
    dx_max = inlet.diameter / 100
    rec = {k: [[] for _ in range(n)] for k in ("x", "t", "v", "u", "ke", "teff")}
    x_end = inlet.x_end
    span = x_end - x[0]
    steps = 0
    active = np.ones(n, bool)
    while active.any() and steps < 5_000_000:
        a = np.nonzero(active)[0]
        g = inlet.gas_at(x[a])
        E = inlet.efield(x[a])
        K = K0[a] * N_LOSCHMIDT / g["N"]
        tau = m[a] * K / q[a]
        if high_field:
            mu_red = m[a] * mg / (m[a] + mg)
            c = np.sqrt(8 * K_B * g["T"] / (np.pi * mu_red))
            h = np.sqrt(1 + ((v[a] - g["u"]) / c) ** 2)
            tau = tau / h
            K = K / h
        speed = np.maximum(np.abs(v[a]), np.abs(g["u"] + K * E))
        dt = dx_max / np.maximum(speed, 1.0)
        # exponential integrator for dv/dt = (u + K E - v) / tau
        v_ss = g["u"] + K * E
        decay = np.exp(-dt / tau)
        v_new = v_ss + (v[a] - v_ss) * decay
        # exact position update for the same linear ODE
        x[a] += v_ss * dt + (v[a] - v_ss) * tau * (1 - decay)
        v[a] = v_new
        t[a] += dt
        steps += 1
        for j, i in enumerate(a):
            rec["x"][i].append(x[i]); rec["t"][i].append(t[i]); rec["v"][i].append(v[i])
            rec["u"][i].append(g["u"][j])
            rec["ke"][i].append(0.5 * m[i] * v[i] ** 2 / E_CHARGE)
            rec["teff"][i].append(g["T"][j] + mg * (v[i] - g["u"][j]) ** 2 / (3 * K_B))
        active[a[x[a] >= x_end]] = False
        if progress and steps % 2000 == 0 and progress(float(np.min(x[active] - x[0]) / span) if active.any() else 1.0):
            break
    out = []
    for i in range(n):
        d = {k: np.array(rec[k][i]) for k in rec}
        # thin to n_rec points for plotting
        if len(d["x"]) > n_rec:
            idx = np.unique(np.linspace(0, len(d["x"]) - 1, n_rec).astype(int))
            d = {k: val[idx] for k, val in d.items()}
        d["ion"] = ions[i]
        d["v_final"], d["ke_final"], d["t_final"] = v[i], 0.5 * m[i] * v[i] ** 2 / E_CHARGE, t[i]
        out.append(d)
    return out


DEFAULT_IONS = [
    InletIon("m/z 200", 200, 1, 1.40),
    InletIon("m/z 500", 500, 1, 1.00),
    InletIon("m/z 1500 (3+)", 1500, 3, 1.10),
]
