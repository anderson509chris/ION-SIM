"""Measured and simulated drive waveforms: SPICE results and oscilloscope captures.

A Trace is a voltage record v(t) that a driver output can add to its voltage
(Output.wave, scaled by Output.wave_scale), so an electrode can be driven by what the
circuit model says or by what the scope saw on the real board, ringing and droop included.
Between samples the trace is interpolated linearly. It either repeats with its period
(the record length unless set) or holds its last value.

Sources:
  read_raw(path)       SPICE raw files: ngspice and LTspice, binary or ASCII, including
                       LTspice's float32 / "double" / compressed / fastaccess / .step files
  read_table(path)     text and CSV: LTspice "Export data as text", ngspice wrdata,
                       Rigol CSV saved from the scope (Start / Increment header or a time
                       column), or any table with a time column and voltage columns
  run_spice(netlist)   runs ngspice or LTspice in batch mode and reads its raw file
  scope.RigolScope     captures a channel live over LAN or USB (scope.py)

Because the field is linear, a trace costs one extra field part per driver output that
carries it, however long the record is.
"""
import io
import os
import re
import shutil
import subprocess
import tempfile
import numpy as np

MAX_POINTS = 20000      # default points kept on import (the field part does not care, the save file does)


class Trace:
    """A sampled voltage, t in s from the start of the record, v in V. t0 is the time of the
    first sample in the source's own clock (a scope record starts before its trigger, so
    t0 < 0 there and the trigger sits at t = -t0 in the trace)."""

    def __init__(self, name, t, v, source="", repeat=True, period=0.0, delay=0.0, t0=None):
        t, v = np.asarray(t, float).ravel(), np.asarray(v, float).ravel()
        if t.size != v.size or t.size < 2:
            raise ValueError(f"trace '{name}' needs at least two (time, voltage) samples")
        order = np.argsort(t, kind="stable")
        t, v = t[order], v[order]
        keep = np.r_[True, np.diff(t) > 0]              # drop repeated time points (SPICE breakpoints)
        self.t0 = float(t[0]) if t0 is None else float(t0)
        self.t, self.v = t[keep] - t[0], v[keep]
        self.name, self.source = name, source
        self.repeat, self.period, self.delay = bool(repeat), float(period), float(delay)

    # ---- the waveform ----
    def duration(self):
        return float(self.t[-1])

    def cycle(self):
        """Repeat period (s): the set period, else the record length (first to last sample)."""
        return self.period if self.period > 0 else self.duration()

    def sample_dt(self):
        return float(np.median(np.diff(self.t)))

    def level(self, t):
        """Voltage at times t (array or scalar), before the output's scale."""
        s = np.asarray(t, float) - self.delay
        if self.repeat:
            s = np.mod(s, self.cycle())                 # a period longer than the record holds its last value
        return np.interp(s, self.t, self.v)

    def time_step(self):
        """Longest flight step that still follows the trace: about a fifth of the full swing
        per step on its steepest edge (like a pulse's rise / 4), never finer than its samples."""
        dt = self.sample_dt()
        span = float(np.ptp(self.v))
        if span <= 0:
            return np.inf
        slope = np.max(np.abs(np.diff(self.v)) / np.diff(self.t))
        return max(dt, 0.2 * span / slope)

    def phasor(self, f):
        """Complex amplitude c of the component at frequency f (v ~ Re(c exp(2 pi i f t))),
        over the whole periods of f that fit in the record; 0 if none fit."""
        n = int(np.floor(self.duration() * f + 1e-9)) if f > 0 else 0
        if n < 1:
            return 0j
        m = int(min(max(64 * n, 4096), 2_000_000))
        t = np.arange(m) * (n / f / m)
        return complex(2 * np.mean(np.interp(t, self.t, self.v) * np.exp(-2j * np.pi * f * t)))

    def main_tone(self):
        """(frequency, 0-pk amplitude) of the strongest component apart from DC."""
        dt = self.sample_dt()
        m = int(min(len(self.t), 1 << 20))
        t = np.linspace(0, self.duration(), m)
        v = np.interp(t, self.t, self.v)
        spec = np.abs(np.fft.rfft((v - v.mean()) * np.hanning(m)))
        if len(spec) < 3:
            return 0.0, 0.0
        k = int(np.argmax(spec[1:]) + 1)
        f = k / (t[1] - t[0]) / m
        return float(f), float(abs(self.phasor(f))) if f * self.duration() >= 1 else float(np.ptp(v) / 2)

    def stats(self):
        return dict(points=len(self.t), duration=self.duration(), dt=self.sample_dt(),
                    vmin=float(self.v.min()), vmax=float(self.v.max()),
                    vpp=float(np.ptp(self.v)), vmean=float(self.v.mean()))

    # ---- editing ----
    def decimated(self, max_points=MAX_POINTS):
        """A copy with at most max_points samples, each the mean of a block (so noise and
        ringing faster than the new spacing average out rather than alias)."""
        n = len(self.t)
        if max_points <= 1 or n <= max_points:
            return self.copy()
        k = int(np.ceil(n / max_points))
        m = (n // k) * k
        t = self.t[:m].reshape(-1, k).mean(1)
        v = self.v[:m].reshape(-1, k).mean(1)
        return Trace(self.name, t, v, self.source, self.repeat, self.period, self.delay, self.t0 + t[0])

    def cropped(self, t0, t1):
        """The part of the record between t0 and t1 (s from its start), restarting at 0."""
        sel = (self.t >= t0) & (self.t <= t1)
        if sel.sum() < 2:
            raise ValueError("fewer than two samples in that time range")
        return Trace(self.name, self.t[sel], self.v[sel], self.source, self.repeat, self.period, self.delay,
                     self.t0 + self.t[sel][0])

    def from_trigger(self):
        """The record from its trigger (source time 0) on, if the trigger is inside it."""
        return self.cropped(-self.t0, self.duration()) if 0 < -self.t0 < self.duration() else self.copy()

    def copy(self, name=None):
        return Trace(name or self.name, self.t.copy(), self.v.copy(), self.source, self.repeat, self.period,
                     self.delay, self.t0)

    # ---- storage ----
    def to_dict(self):
        return dict(name=self.name, source=self.source, repeat=self.repeat, period=self.period, delay=self.delay,
                    t0=self.t0, t=[float(f"{x:.9g}") for x in self.t], v=[float(f"{x:.7g}") for x in self.v])

    @classmethod
    def from_dict(cls, d):
        return cls(d["name"], d["t"], d["v"], d.get("source", ""), d.get("repeat", True),
                   d.get("period", 0.0), d.get("delay", 0.0), d.get("t0", 0.0))

    def __repr__(self):
        s = self.stats()
        return (f"Trace('{self.name}', {s['points']} points over {s['duration'] * 1e6:.4g} µs, "
                f"{s['vmin']:.4g} to {s['vmax']:.4g} V)")


# ---------------------------------------------------------------------------------
# SPICE raw files

def _decode_header(data):
    """(header text, offset of the data block, encoding) of a raw file's bytes."""
    utf16 = data[:2] == b"\xff\xfe" or (len(data) > 1 and data[1:2] == b"\x00")
    enc = "utf-16-le" if utf16 else "latin-1"
    start = 2 if data[:2] == b"\xff\xfe" else 0
    for marker in ("Binary:\n", "Values:\n", "Binary:\r\n", "Values:\r\n"):
        m = marker.encode(enc)
        i = data.find(m, start)
        if i >= 0:
            return data[start:i + len(m)].decode(enc), i + len(m), marker.strip().rstrip(":")
    raise ValueError("not a SPICE raw file (no 'Binary:' or 'Values:' line)")


def _plots(data):
    """Split a raw file holding several plots (ngspice writes one per analysis)."""
    out, pos = [], 0
    while pos < len(data) and data[pos:].strip(b"\x00\r\n\t "):
        header, off, kind = _decode_header(data[pos:])
        info = _parse_header(header)
        info["kind"] = kind
        n, nv = info["points"], len(info["vars"])
        flags = info["flags"]
        if kind == "Binary":
            if "complex" in flags:
                size = n * nv * 16
            elif "double" in flags or not info["ltspice"]:
                size = n * nv * 8
            else:
                size = n * (8 + 4 * (nv - 1))
            out.append((info, data[pos + off:pos + off + size]))
            pos += off + size
        else:
            # ASCII: the block runs to the next "Title:" (or the end)
            rest = data[pos + off:]
            nxt = rest.find(b"Title:")
            body = rest if nxt < 0 else rest[:nxt]
            out.append((info, body))
            pos += off + len(body)
            if nxt < 0:
                break
    return out


def _parse_header(text):
    info = dict(title="", plotname="", flags="", points=0, vars=[], ltspice=False)
    lines = text.replace("\r", "").split("\n")
    i = 0
    while i < len(lines):
        ln = lines[i]
        key, _, val = ln.partition(":")
        k = key.strip().lower()
        if k == "title":
            info["title"] = val.strip()
            info["ltspice"] = "ltspice" in val.lower() or info["ltspice"]
        elif k == "plotname":
            info["plotname"] = val.strip()
        elif k == "flags":
            info["flags"] = val.strip().lower()
        elif k == "no. points":
            info["points"] = int(val.strip())
        elif k in ("command", "backannotation") and "ltspice" in val.lower():
            info["ltspice"] = True
        elif k == "offset":
            info["ltspice"] = True                    # only LTspice writes an Offset line
        elif k == "variables":
            i += 1
            while i < len(lines) and lines[i].startswith(("\t", " ")) and lines[i].strip():
                parts = lines[i].split()
                info["vars"].append((parts[1], parts[2] if len(parts) > 2 else ""))
                i += 1
            continue
        i += 1
    if not info["vars"]:
        raise ValueError("raw file lists no variables")
    return info


def _read_plot(info, body):
    n, nv, flags = info["points"], len(info["vars"]), info["flags"]
    if "complex" in flags:
        raise ValueError(f"'{info['plotname']}' is a complex (AC) analysis; a drive needs a transient (.tran) run")
    if info["kind"] == "Values":
        nums = []
        for tok in body.decode("latin-1").split():
            if "," in tok:                             # complex "re,im"
                raise ValueError("complex values; a drive needs a transient (.tran) run")
            try:
                nums.append(float(tok))
            except ValueError:
                pass
        arr = np.array(nums, float)
        rows = arr.size // (nv + 1)                   # each row starts with its index
        arr = arr[:rows * (nv + 1)].reshape(rows, nv + 1)[:, 1:]
    elif "double" in flags or not info["ltspice"]:
        arr = np.frombuffer(body, "<f8", n * nv)
        arr = arr.reshape(nv, n).T if "fastaccess" in flags else arr.reshape(n, nv)
    else:
        if "fastaccess" in flags:
            t = np.frombuffer(body, "<f8", n)
            rest = np.frombuffer(body, "<f4", n * (nv - 1), offset=8 * n).reshape(nv - 1, n).T
        else:
            rec = np.dtype([("t", "<f8"), ("v", "<f4", (nv - 1,))])
            a = np.frombuffer(body, rec, n)
            t, rest = a["t"], a["v"].reshape(n, nv - 1)
        arr = np.column_stack([t, rest.astype(float)])
    arr = np.array(arr, float)
    arr[:, 0] = np.abs(arr[:, 0])                     # LTspice marks compressed points with a negative time
    return arr


def read_raw(path, step=None):
    """Transient vectors of a SPICE raw file: {name: (t, v)}, names as the simulator wrote
    them (V(out), I(R1), ...). A .step run holds several runs one after another; pass step
    (0-based) to pick one, otherwise the first is returned (see raw_steps)."""
    with open(path, "rb") as f:
        data = f.read()
    plots = _plots(data)
    tran = [(i, b) for i, b in plots if i["vars"][0][0].lower() == "time"]
    if not tran:
        names = ", ".join(i["plotname"] for i, _ in plots) or "none"
        raise ValueError(f"no transient analysis in {os.path.basename(path)} (plots: {names}); "
                         "run a .tran to drive electrodes")
    info, body = tran[-1]
    arr = _read_plot(info, body)
    runs = _split_steps(arr[:, 0])
    a, b = runs[0 if step is None else step]
    names = [v[0] for v in info["vars"]]
    t = arr[a:b, 0]
    return {nm: (t, arr[a:b, k]) for k, nm in enumerate(names) if k > 0}


def raw_steps(path):
    """Number of runs in a raw file (more than one after an LTspice .step)."""
    with open(path, "rb") as f:
        plots = _plots(f.read())
    tran = [(i, b) for i, b in plots if i["vars"][0][0].lower() == "time"]
    return len(_split_steps(_read_plot(*tran[-1])[:, 0])) if tran else 0


def _split_steps(t):
    """[(start, stop)] of each run: time drops back to the start between stepped runs."""
    cuts = np.flatnonzero(np.diff(t) < 0) + 1
    edges = np.r_[0, cuts, len(t)]
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a >= 2]


# ---------------------------------------------------------------------------------
# text and CSV (LTspice export, ngspice wrdata, Rigol CSV, anything with a time column)

def _floats(row):
    out = []
    for c in row:
        try:
            out.append(float(c))
        except ValueError:
            out.append(np.nan)
    return out


def read_table(path):
    """Voltage columns of a text or CSV file: {name: (t, v)}.

    Understood layouts:
      a header row with a time column (time, Time(s), X, Second, ...) and one column per signal;
      Rigol CSV with 'X,CH1,...,Start,Increment' and a 'Sequence,Volt,<start>,<increment>' row;
      ngspice wrdata (no header, time / value column pairs);
      plain two-column numbers (time, voltage)."""
    with open(path, "r", encoding="utf-8-sig", errors="replace") as f:
        text = f.read()
    lines = [ln for ln in text.replace("\r", "").split("\n") if ln.strip()]
    if not lines:
        raise ValueError("empty file")
    sep = "," if lines[0].count(",") >= 1 or (len(lines) > 1 and lines[1].count(",") >= 1) else None
    split = (lambda s: [c.strip() for c in s.split(",")]) if sep else (lambda s: s.split("\t") if "\t" in s else s.split())
    rows = [split(ln) for ln in lines]

    def numeric(r):
        vals = _floats([c for c in r if c != ""])
        return bool(vals) and not np.isnan(vals).any()

    # header rows come before the first numeric row
    first = next((i for i, r in enumerate(rows) if numeric(r)), None)
    if first is None:
        raise ValueError("no numbers in the file")
    head = rows[:first]
    data = rows[first:]
    width = max(len(r) for r in data[:50])
    arr = np.array([(_floats(r) + [np.nan] * width)[:width] for r in data], float)

    # Rigol: X,CH1,CH2,Start,Increment / Sequence,Volt,Volt,<start>,<inc>
    if head and head[0] and head[0][0].strip().upper() == "X" and any("increment" in c.lower() for c in head[0]):
        hdr = head[0]
        names = [c for c in hdr[1:] if c and c.lower() not in ("start", "increment")]
        nums = _floats(head[1] if len(head) > 1 else [])
        tail = [x for x in nums if not np.isnan(x)]
        if len(tail) >= 2:
            start, inc = tail[-2], tail[-1]
        else:                                          # some firmware puts them on the first data row
            start, inc = arr[0, len(names) + 1], arr[0, len(names) + 2]
        t = start + arr[:, 0] * inc
        return {nm: (t, arr[:, k + 1]) for k, nm in enumerate(names) if np.isfinite(arr[:, k + 1]).any()}

    if head:
        hdr = head[-1]
        tcol = next((k for k, c in enumerate(hdr) if re.match(r"(?i)\s*(time|t|x|sec|second|s)\b", c)), 0)
        out = {}
        for k, nm in enumerate(hdr):
            if k == tcol or k >= arr.shape[1] or not nm:
                continue
            ok = np.isfinite(arr[:, tcol]) & np.isfinite(arr[:, k])
            if ok.sum() >= 2:
                out[nm] = (arr[ok, tcol], arr[ok, k])
        if out:
            return out
    # no header: ngspice wrdata writes (t, v) pairs side by side; otherwise column 0 is time
    if arr.shape[1] >= 4 and arr.shape[1] % 2 == 0 and np.allclose(arr[:, 0], arr[:, 2], equal_nan=True):
        return {f"v{k + 1}": (arr[:, 2 * k], arr[:, 2 * k + 1]) for k in range(arr.shape[1] // 2)}
    return {f"v{k}": (arr[:, 0], arr[:, k]) for k in range(1, arr.shape[1])}


def read_any(path, step=None):
    """{name: (t, v)} from a raw file or a table, picked by content."""
    with open(path, "rb") as f:
        head = f.read(4096).lstrip(b"\xff\xfe")
    if head.startswith((b"Title:", "Title:".encode("utf-16-le"))) or b"Plotname:" in head \
            or "Plotname:".encode("utf-16-le") in head:
        return read_raw(path, step)
    return read_table(path)


# ---------------------------------------------------------------------------------
# running a simulator

LTSPICE_PATHS = [
    r"C:\Program Files\ADI\LTspice\LTspice.exe",
    r"C:\Program Files\LTC\LTspiceXVII\XVIIx64.exe",
    os.path.expanduser(r"~\AppData\Local\Programs\ADI\LTspice\LTspice.exe"),
    "/Applications/LTspice.app/Contents/MacOS/LTspice",
]


def find_simulator(kind="auto"):
    """(kind, executable) of an installed simulator, or (kind, None)."""
    if kind in ("auto", "ngspice"):
        exe = shutil.which("ngspice") or shutil.which("ngspice_con")
        for p in (r"C:\Program Files\Spice64\bin\ngspice_con.exe", r"C:\Spice64\bin\ngspice_con.exe"):
            exe = exe or (p if os.path.exists(p) else None)
        if exe or kind == "ngspice":
            return "ngspice", exe
    exe = shutil.which("LTspice") or shutil.which("ltspice") or next((p for p in LTSPICE_PATHS if os.path.exists(p)), None)
    return "ltspice", exe


def run_spice(netlist, kind="auto", exe=None, timeout=600, workdir=None):
    """Run a netlist in batch mode and return the path of its raw file.

    ngspice: ngspice -b -r <out.raw> <netlist>  (the netlist needs a .tran line).
    LTspice: LTspice -b <netlist>  writes <netlist>.raw next to it (.net / .cir / .asc;
    an .asc schematic needs LTspice on Windows or macOS)."""
    kind, found = find_simulator(kind) if exe is None else (kind if kind != "auto" else
                                                             ("ltspice" if "ltspice" in os.path.basename(exe).lower()
                                                              or "xvii" in os.path.basename(exe).lower() else "ngspice"), exe)
    exe = exe or found
    if not exe:
        raise FileNotFoundError(f"{'LTspice' if kind == 'ltspice' else 'ngspice'} was not found; install it or "
                                "give the path to its executable")
    netlist = os.path.abspath(netlist)
    if kind == "ngspice":
        out_dir = workdir or tempfile.mkdtemp(prefix="ionsim_spice_")
        raw = os.path.join(out_dir, os.path.splitext(os.path.basename(netlist))[0] + ".raw")
        cmd = [exe, "-b", "-r", raw, netlist]
        cwd = os.path.dirname(netlist)
    else:
        raw = os.path.splitext(netlist)[0] + ".raw"
        cmd = [exe, "-b", netlist]
        if os.name == "nt" and netlist.lower().endswith(".asc"):
            cmd = [exe, "-Run", "-b", netlist]
        cwd = os.path.dirname(netlist)
    if os.path.exists(raw):
        os.remove(raw)
    p = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True, timeout=timeout)
    if not os.path.exists(raw) or os.path.getsize(raw) == 0:
        log = (p.stdout + "\n" + p.stderr).strip()
        lg = os.path.splitext(netlist)[0] + ".log"
        if not log and os.path.exists(lg):
            with open(lg, errors="replace") as f:
                log = f.read()
        raise RuntimeError(f"{kind} wrote no raw file (exit code {p.returncode}).\n{log[-2000:]}")
    return raw


def spice_traces(path, step=None):
    """{name: Trace} of every voltage in a SPICE raw or text result."""
    src = os.path.basename(path)
    return {nm: Trace(nm, t, v, f"SPICE: {src} {nm}" + (f" (step {step + 1})" if step else ""), repeat=True)
            for nm, (t, v) in read_any(path, step).items()}
