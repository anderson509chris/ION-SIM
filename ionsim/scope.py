"""Capture a channel from a Rigol oscilloscope and use it as a drive waveform.

Talks SCPI either over a plain LAN socket (port 5555, nothing to install) or through VISA
(pyvisa, for USB: "USB0::0x1AB1::0x04CE::DS1ZA...::INSTR", or "TCPIP::192.168.1.50::INSTR").
Works with the DS1000Z / MSO1000Z, DS2000, MSO5000, DS7000/MSO7000 and DHO800 / DHO900 /
DHO1000 / DHO4000 families, which share the :WAVeform commands:

    :WAV:SOUR CHAN1   :WAV:MODE NORM|RAW   :WAV:FORM BYTE|WORD   :WAV:PRE?   :WAV:DATA?

NORM reads what is on screen (about 1200 points on a DS1000Z, more on newer scopes); RAW
reads the whole acquisition memory and needs the scope stopped, so capture() can take a
single shot first. Volts are (code - yorigin - yreference) * yincrement and the time of
point i is (i - xreference) * xincrement + xorigin, so t = 0 is the trigger.
"""
import socket
import time
import numpy as np

from .waveforms import Trace

PORT = 5555
CHANNELS = ["CHAN1", "CHAN2", "CHAN3", "CHAN4"]


class SocketLink:
    """SCPI over a raw TCP socket (Rigol LAN port 5555)."""

    def __init__(self, host, port=PORT, timeout=10.0):
        self.sock = socket.create_connection((host, port), timeout=timeout)
        self.sock.settimeout(timeout)
        self._t = timeout
        self._buf = b""

    def write(self, cmd):
        self.sock.sendall(cmd.encode("ascii") + b"\n")

    def _read(self, n):
        while len(self._buf) < n:
            chunk = self.sock.recv(max(65536, n - len(self._buf)))
            if not chunk:
                raise ConnectionError("the scope closed the connection")
            self._buf += chunk
        out, self._buf = self._buf[:n], self._buf[n:]
        return out

    def _readline(self):
        while b"\n" not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("the scope closed the connection")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(b"\n")
        return line

    def query(self, cmd):
        self.write(cmd)
        return self._readline().decode("ascii", "replace").strip()

    def query_block(self, cmd):
        """IEEE 488.2 definite-length block: #<n><n digits of length><data>\\n."""
        self.write(cmd)
        while True:                                      # skip anything before the '#'
            c = self._read(1)
            if c == b"#":
                break
        nd = int(self._read(1))
        size = int(self._read(nd)) if nd else 0
        data = self._read(size)
        if self._buf[:1] == b"\n":
            self._buf = self._buf[1:]
        else:
            try:                                         # eat the terminator if it is still on its way
                self.sock.settimeout(0.2)
                self._read(1)
            except (socket.timeout, ConnectionError):
                pass
            finally:
                self.sock.settimeout(self._t)
        return data

    def close(self):
        try:
            self.sock.close()
        except OSError:
            pass


class VisaLink:
    """SCPI through pyvisa (USB, or LAN by VISA resource name)."""

    def __init__(self, resource, timeout=10.0):
        import pyvisa
        try:
            rm = pyvisa.ResourceManager()
        except (OSError, ValueError):
            rm = pyvisa.ResourceManager("@py")           # pure-Python backend (pyvisa-py)
        self.inst = rm.open_resource(resource)
        self.inst.timeout = int(timeout * 1000)
        self.inst.read_termination = "\n"
        self.inst.write_termination = "\n"
        self.inst.chunk_size = 1 << 20

    def write(self, cmd):
        self.inst.write(cmd)

    def query(self, cmd):
        return self.inst.query(cmd).strip()

    def query_block(self, cmd):
        return bytes(self.inst.query_binary_values(cmd, datatype="B", container=bytearray))

    def close(self):
        try:
            self.inst.close()
        except Exception:
            pass


def connect(address, timeout=10.0):
    """A link to a scope: an IP address or host name (LAN socket, optionally host:port), or a
    VISA resource name (anything with '::')."""
    address = address.strip()
    if "::" in address:
        return VisaLink(address, timeout)
    host, _, port = address.partition(":")
    return SocketLink(host, int(port) if port else PORT, timeout)


def list_resources():
    """VISA resource names of connected instruments ([] without pyvisa)."""
    try:
        import pyvisa
    except ImportError:
        return []
    for backend in ("", "@py"):
        try:
            return list(pyvisa.ResourceManager(backend).list_resources())
        except Exception:
            continue
    return []


class RigolScope:
    def __init__(self, link):
        self.link = link if not isinstance(link, str) else connect(link)
        self.idn = self.link.query("*IDN?")
        parts = [p.strip() for p in self.idn.split(",")]
        self.maker = parts[0] if parts else ""
        self.model = parts[1] if len(parts) > 1 else ""
        # 12-bit scopes (DHO) send 16-bit words; 8-bit scopes bytes
        self.word = self.model.upper().startswith("DHO")

    def close(self):
        self.link.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def channels(self):
        """Channels switched on."""
        on = []
        for ch in CHANNELS:
            try:
                if self.link.query(f":{ch}:DISP?").strip() in ("1", "ON"):
                    on.append(ch)
            except Exception:
                break
        return on

    def single(self, timeout=10.0):
        """Arm a single acquisition and wait for it to finish."""
        self.link.write(":SING")
        t_end = time.time() + timeout
        time.sleep(0.05)
        while time.time() < t_end:
            if self.link.query(":TRIG:STAT?").upper().startswith("STOP"):
                return True
            time.sleep(0.05)
        raise TimeoutError("the scope did not trigger; check the trigger level and source")

    def preamble(self):
        p = [float(x) for x in self.link.query(":WAV:PRE?").split(",")[:10]]
        keys = ("format", "type", "points", "count", "xinc", "xorigin", "xref", "yinc", "yorigin", "yref")
        return dict(zip(keys, p))

    def capture(self, channel="CHAN1", memory=False, single=False, stop=None, timeout=20.0, name=None):
        """One channel as a Trace (t = 0 at the trigger; Trace.t0 says where the record starts).

        memory: read the whole acquisition memory (RAW) instead of the screen (NORM).
        single: take a fresh single-shot acquisition first.
        stop: stop the scope before reading (default: when reading memory)."""
        L = self.link
        ch = channel.upper().replace("CH", "CHAN") if not channel.upper().startswith("CHAN") else channel.upper()
        if single:
            self.single(timeout)
        elif stop if stop is not None else memory:
            L.write(":STOP")
        fmt = "WORD" if self.word else "BYTE"
        L.write(f":WAV:SOUR {ch}")
        L.write(f":WAV:MODE {'RAW' if memory else 'NORM'}")
        L.write(f":WAV:FORM {fmt}")
        pre = self.preamble()
        n = int(pre["points"])
        width = 2 if fmt == "WORD" else 1
        chunk = 125_000 if width == 2 else 250_000
        codes = []
        if memory and n > chunk:
            for a in range(1, n + 1, chunk):
                L.write(f":WAV:STAR {a}")
                L.write(f":WAV:STOP {min(a + chunk - 1, n)}")
                codes.append(self._codes(L.query_block(":WAV:DATA?"), width))
            L.write(":WAV:STAR 1")
        else:
            if memory:
                L.write(":WAV:STAR 1"); L.write(f":WAV:STOP {n}")
            codes.append(self._codes(L.query_block(":WAV:DATA?"), width))
        code = np.concatenate(codes).astype(float)
        v = (code - pre["yorigin"] - pre["yref"]) * pre["yinc"]
        t = (np.arange(len(v)) - pre["xref"]) * pre["xinc"] + pre["xorigin"]
        label = name or f"{self.model} {ch}"
        src = (f"Rigol {self.model} {ch}, {len(v)} points at {pre['xinc'] * 1e9:.4g} ns "
               f"({'memory' if memory else 'screen'}), {time.strftime('%Y-%m-%d %H:%M')}")
        return Trace(label, t, v, src, repeat=True)

    @staticmethod
    def _codes(data, width):
        return np.frombuffer(data[:len(data) // width * width], "<u2" if width == 2 else "u1")


def capture(address, channels=("CHAN1",), **kw):
    """{channel: Trace} from a scope at an address (see connect)."""
    with RigolScope(connect(address)) as sc:
        return {ch: sc.capture(ch, **kw) for ch in channels}
