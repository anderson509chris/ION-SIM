"""The numbers behind whole-system sweeps and tolerance studies (the Sweeps tab runs the models).

A setting is varied either one at a time over a range (a sweep) or all at once at random about its
nominal value (a tolerance study, Monte Carlo). Each spread is given as ± (in % of the nominal value
or in the setting's own units) with a distribution:

  Gaussian   ± is 3 sigma (99.7 % of trials inside it)
  Uniform    anywhere inside ±, equally likely
  Ripple     nominal + ± sin(random phase): a supply ripple or a slow drift that is much slower than
             the ions' transit, so each ion packet sees it frozen at some point of its cycle
"""
import numpy as np

DISTRIBUTIONS = ["Gaussian (± = 3σ)", "Uniform within ±", "Ripple (± sin, random phase)"]
UNITS = ["%", "abs"]


def tol_abs(nominal, pm, unit):
    """± in the setting's own units."""
    return abs(nominal) * pm / 100.0 if unit == "%" else abs(pm)


def sigma(tol, dist):
    """Standard deviation of a spread of ±tol."""
    k = DISTRIBUTIONS.index(dist) if dist in DISTRIBUTIONS else 0
    return tol * (1 / 3, 1 / np.sqrt(3), 1 / np.sqrt(2))[k]


def sample(nominal, tol, dist, rng, n=None):
    k = DISTRIBUTIONS.index(dist) if dist in DISTRIBUTIONS else 0
    if k == 0:
        d = rng.standard_normal(n) / 3
    elif k == 1:
        d = rng.uniform(-1, 1, n)
    else:
        d = np.sin(2 * np.pi * rng.uniform(0, 1, n))
    return nominal + tol * d


def tolerance_trials(nominals, tols, dists, n, seed=0):
    """(n, n_settings) array of random settings, one row per trial."""
    rng = np.random.default_rng(seed)
    cols = [sample(x0, t, d, rng, n) for x0, t, d in zip(nominals, tols, dists)]
    return np.column_stack(cols) if cols else np.zeros((n, 0))


def sweep_values(lo, hi, points):
    return np.linspace(lo, hi, max(int(points), 2))


def local_slope(x, y, x0):
    """d y / d x at x0 from a sweep (the points around x0; a straight-line fit when x0 is outside)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if len(np.unique(x)) < 2:
        return np.nan
    o = np.argsort(x)
    x, y = x[o], y[o]
    x, i = np.unique(x, return_index=True)
    y = y[i]
    if not (x[0] <= x0 <= x[-1]) or len(x) < 3:
        return float(np.polyfit(x, y, 1)[0])
    return float(np.interp(x0, x, np.gradient(y, x)))


def contributions(X, y):
    """Linear sensitivity of a result to each setting from a tolerance study: a least-squares fit of
    y on the settings. Returns (contribution of each setting to the spread of y, in y's units with
    the sign of the slope, explained fraction R²). NaN when there are too few good trials."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    ok = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    X, y = X[ok], y[ok]
    m = X.shape[1]
    if len(y) < m + 3 or np.std(y) == 0:
        return np.full(m, np.nan), np.nan
    sd = X.std(axis=0)
    use = sd > 0
    Z = (X[:, use] - X[:, use].mean(axis=0)) / sd[use]
    A = np.column_stack([np.ones(len(y)), Z])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    out = np.zeros(m)
    out[use] = coef[1:]
    fit = A @ coef
    r2 = 1 - np.sum((y - fit) ** 2) / np.sum((y - y.mean()) ** 2)
    return out, float(r2)


def spread(v):
    """Summary of one result over the trials."""
    v = np.asarray(v, float)
    ok = v[np.isfinite(v)]
    if not len(ok):
        return {"n": 0, "mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "p05": np.nan, "p95": np.nan}
    return {"n": len(ok), "mean": float(ok.mean()), "std": float(ok.std(ddof=1)) if len(ok) > 1 else 0.0,
            "min": float(ok.min()), "max": float(ok.max()),
            "p05": float(np.percentile(ok, 5)), "p95": float(np.percentile(ok, 95))}


class SameRandom:
    """While on, every np.random.default_rng() made without a seed gets the seeds base, base + 1, ...
    counted from restart(). Restarted before each run, every run draws the same random ions (common
    random numbers), so the differences between runs come from the settings, not sampling noise."""

    def __init__(self, base=20240):
        import threading
        self.base, self.k, self._lock, self._orig = int(base), 0, threading.Lock(), None

    def restart(self):
        with self._lock:
            self.k = 0

    def _rng(self, seed=None):
        if seed is None:
            with self._lock:
                seed, self.k = self.base + self.k, self.k + 1
        return self._orig(seed)

    def on(self):
        if self._orig is None:
            self._orig = np.random.default_rng
            np.random.default_rng = self._rng

    def off(self):
        if self._orig is not None:
            np.random.default_rng = self._orig
            self._orig = None
