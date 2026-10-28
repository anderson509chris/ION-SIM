"""A pretend Rigol scope on a local socket, for testing the scope capture without one.
It answers the SCPI the capture uses and holds a ringing pulse on CHAN1 and a sine on CHAN2."""
import socket
import threading
import numpy as np


class FakeRigol:
    def __init__(self, model="DS1104Z", screen=1200, memory=600_000, xinc=1e-8):
        self.model, self.word = model, model.upper().startswith("DHO")
        self.screen, self.memory, self.xinc = screen, memory, xinc
        self.state = dict(src="CHAN1", mode="NORM", fmt="BYTE", start=1, stop=screen, trig="TD")
        self.srv = socket.socket(); self.srv.bind(("127.0.0.1", 0)); self.srv.listen(1)
        self.port = self.srv.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def signal(self, ch, t):
        if ch == "CHAN1":                       # 600 V pusher pulse with 2 MHz ringing on its edges
            on = (t > 0) & (t < 2e-6)
            ring = 40 * np.exp(-np.abs(t) / 2e-7) * np.sin(2 * np.pi * 2e6 * t) * (t > 0)
            return 600 * on + ring
        return 50 * np.sin(2 * np.pi * 1e6 * t)

    def _record(self):
        n = self.memory if self.state["mode"] == "RAW" else self.screen
        xinc = self.xinc * (self.memory / self.screen if self.state["mode"] != "RAW" else 1)
        xorigin = -n * xinc / 4                 # trigger a quarter into the record
        t = np.arange(n) * xinc + xorigin
        v = self.signal(self.state["src"], t)
        bits = 65535 if self.word else 255
        yinc = 2 * 800.0 / bits
        yref, yorig = (32768, 0) if self.word else (127, 0)
        code = np.clip(np.round(v / yinc + yorig + yref), 0, bits).astype("<u2" if self.word else "u1")
        return n, xinc, xorigin, yinc, yorig, yref, code

    def _serve(self):
        conn, _ = self.srv.accept()
        f = conn.makefile("rb")
        for line in f:
            cmd = line.decode().strip()
            u = cmd.upper()
            st = self.state
            if u == "*IDN?":
                conn.sendall(f"RIGOL TECHNOLOGIES,{self.model},DS1ZA000000001,00.04.05\n".encode())
            elif u.endswith(":DISP?"):
                conn.sendall(b"1\n" if u[1:6] in ("CHAN1", "CHAN2") else b"0\n")
            elif u.startswith(":WAV:SOUR"):
                st["src"] = cmd.split()[1]
            elif u.startswith(":WAV:MODE"):
                st["mode"] = cmd.split()[1]
            elif u.startswith(":WAV:FORM"):
                st["fmt"] = cmd.split()[1]
            elif u.startswith(":WAV:STAR"):
                st["start"] = int(cmd.split()[1])
            elif u.startswith(":WAV:STOP"):
                st["stop"] = int(cmd.split()[1])
            elif u in (":STOP", ":SING"):
                st["trig"] = "STOP"
            elif u == ":TRIG:STAT?":
                conn.sendall(f"{st['trig']}\n".encode())
            elif u == ":WAV:PRE?":
                n, xinc, xo, yinc, yo, yr, _ = self._record()
                fmt = 1 if st["fmt"] == "WORD" else 0
                conn.sendall(f"{fmt},{2 if st['mode'] == 'RAW' else 0},{n},1,{xinc:e},{xo:e},0,{yinc:e},{yo},{yr}\n".encode())
            elif u == ":WAV:DATA?":
                n, *_, code = self._record()
                if st["mode"] == "RAW":
                    code = code[st["start"] - 1:st["stop"]]
                data = code.tobytes()
                conn.sendall(f"#9{len(data):09d}".encode() + data + b"\n")
        conn.close()
