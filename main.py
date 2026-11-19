"""GTA VI trophy-rarity forecaster, v2 — for make.fun's
gta6-trophy-rarity-forecast-01.

Forecasts, for each of the game's 55 trophies, five quantiles of its
Exophase earn rate at the case target, scored by pinball loss on the
logit scale.

What the data does (from the kit's competition.yaml and NOTICE):
Exophase refreshes a game's counts on its own schedule, hours apart.
Between refreshes every read repeats the last refresh exactly. At a
refresh the tracked-player count jumps and every rate but the 100% one
falls: newly tracked players hold far fewer trophies. A forecaster
that just repeats the last read therefore misses by ~1.5 logit units
when a refresh lands inside its horizon.

Method (see NOTES-v2.md for the validation numbers behind each choice):

1. Collapse the history to refresh points. Consecutive identical reads
   carry no information; a refresh point is the page's true state.
2. Per trophy, fit three saturating adoption curves to the refresh
   points on the logit scale (log-time linear, log-logistic with a
   fitted ceiling, Weibull adoption). The Weibull fit uses a two-stage
   grid: a coarse (k, lam) grid, then local refinement around the best
   cell. The coarse grid alone left ~0.1 logit of median error on
   synthetic Weibull worlds; refinement recovers most of it.
3. Weight the three curves by rolling-origin backtest inside the
   history: fit each curve on refresh points up to an origin, score its
   pinball loss against a later refresh point, pool over trophies.
4. Joint hierarchical model, fitted in parallel and blended in only
   when it earns its place: earns ~ Binomial(players, p) with
   logit p[t,i] = alpha[i] + f(t) + delta*log(N[t]/N[0]), alpha[i]
   shrunk toward PAVA-ordered tier means (bronze > silver > gold >
   platinum in logit), f(t) an RW2-smoothed shared trend over refresh
   points, delta a shrunk dilution coefficient. It is blended with the
   curve ensemble only if its own expanding-window CV pinball beats
   the backtest-weighted curves' by at least 10% at the longest
   horizon the history supports (weight capped at 0.30); otherwise
   its weight is exactly zero. On every synthetic check run so far the
   joint model loses to the curves (parallel logit curves are the wrong
   shape when each trophy has its own adoption speed), so the gate
   keeps it out there — it exists for the real final history, where a
   refresh drops every trophy's rate together and a shared component
   may genuinely help.
5. Spread from the backtest residuals of the blended median, grown
   slowly (log-wise) with the horizon and capped.
6. Flat history (no movement, as on launch night) carries no trend
   signal, so the forecast falls back to a dilution prior: a refresh
   is expected, the tracked count multiplies (median jump observed in
   the history's own refreshes when any exist, else 5x), and newly
   tracked players hold almost nothing yet except the very first
   trophies, so rates fall by roughly the N factor, with a wide
   spread. A trophy already at 100% stays there.

Standard library only, no network, deterministic (two runs on the
same case write identical bytes), inside the 8 s / 1 CPU / 1 GiB
budget.
"""

import json
import math
import os
import sys
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()
Z = [_NORM.inv_cdf(t) for t in LEVELS]

RELEASE = "2026-11-19T00:00:00Z"  # the game's release day, 00:00 UTC.
# Tier order, rarest first; tier means in logit are non-decreasing along it.
TIER_ORDER = ["platinum", "gold", "silver", "bronze"]
TIER_IDX = {name: k for k, name in enumerate(TIER_ORDER)}


def clamp_pct(p):
    """A rate in percent, clamped to the range the scorer's logit uses."""
    return min(99.99, max(0.01, p))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(y):
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """x rounded to 4 decimals, half up, as JavaScript's Math.round does."""
    scaled = x * 10000
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000
    return int(value) if value.is_integer() else value


def parse_time(s):
    """ISO-8601 UTC timestamp -> seconds since the epoch (float)."""
    if s.endswith("Z"):
        s = s[:-1] + "+00:00"
    from datetime import datetime

    return datetime.fromisoformat(s).timestamp()


def hours_since_release(when):
    return (parse_time(when) - parse_time(RELEASE)) / 3600.0


# ---------------------------------------------------------------------------
# History: refresh points and per-trophy series.
# ---------------------------------------------------------------------------

def refresh_points(case):
    """The history collapsed to its distinct refreshes.

    Consecutive reads with the same players_tracked and the same rates
    are one refresh repeated; keep the first read of each run. A read
    whose rates are all null carries nothing and is dropped.
    """
    points = []
    prev_key = None
    for read in case.get("history", []):
        rates = read.get("rates") or []
        if not rates or all(r is None for r in rates):
            continue
        key = (read.get("players_tracked"), tuple(rates))
        if key == prev_key:
            continue
        prev_key = key
        points.append(read)
    return points


def trophy_series(case, points, idx):
    """[(t_hours, rate, players_tracked)] for one trophy, oldest first."""
    series = []
    for read in points:
        rates = read.get("rates") or []
        if idx >= len(rates):
            continue
        rate = rates[idx]
        if rate is None:
            continue
        n = read.get("players_tracked")
        series.append((hours_since_release(read["slot"]), float(rate), n))
    return series


def series_moved(series):
    """Largest absolute logit move inside a series (0 when flat)."""
    if len(series) < 2:
        return 0.0
    ys = [logit(p) for _, p, _ in series]
    return max(ys) - min(ys)


def _weights(times, now):
    """Recency weights: a read a day old counts half as much as a fresh one."""
    return [0.5 ** max(0.0, (now - t) / 24.0) for t in times]


def _wls_line(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def _thin(series, max_n=48):
    """Evenly thin a series to at most max_n points, keeping first/last."""
    if len(series) <= max_n:
        return series
    idx = sorted({round(i * (len(series) - 1) / (max_n - 1))
                  for i in range(max_n)})
    return [series[i] for i in idx]


# ---------------------------------------------------------------------------
# Curve fits. Each takes [(t_hours, rate, n)] and a target time in hours
# since release, and returns a median rate in percent (or None).
# ---------------------------------------------------------------------------

def fit_loglinear(series, target_t):
    """y = a + b*ln(1+t): the curve baseline's shape, all history weighted."""
    if len(series) < 2:
        return None
    now = series[-1][0]
    xs = [math.log1p(max(0.0, t)) for t, _, _ in series]
    ys = [logit(p) for _, p, _ in series]
    ws = _weights([t for t, _, _ in series], now)
    line = _wls_line(xs, ys, ws)
    if line is None:
        return None
    a, b = line
    return inv_logit(a + b * math.log1p(max(0.0, target_t)))


def fit_loglogistic(series, target_t):
    """p = Pmax * sigmoid(a + b*ln(1+t)): a saturating curve with a fitted
    ceiling. Pmax is profiled over a grid; (a, b) are least squares given it.
    """
    if len(series) < 3:
        return None
    now = series[-1][0]
    us = [math.log1p(max(0.0, t)) for t, _, _ in series]
    ps = [p for _, p, _ in series]
    ws = _weights([t for t, _, _ in series], now)
    pmax_seen = max(ps)
    best = None
    lo = min(99.9, pmax_seen * 1.05 + 0.05)
    for k in range(25):
        frac = k / 24.0
        pmax = lo + (99.9 - lo) * frac * frac
        ys = []
        ok = True
        for p in ps:
            r = p / pmax
            if r <= 0.001 or r >= 0.999:
                ok = False
                break
            ys.append(math.log(r / (1.0 - r)))
        if not ok:
            continue
        line = _wls_line(us, ys, ws)
        if line is None:
            continue
        a, b = line
        sse = 0.0
        for w, u, y in zip(ws, us, ys):
            sse += w * (a + b * u - y) ** 2
        if best is None or sse < best[0]:
            best = (sse, pmax, a, b)
    if best is None:
        return None
    _, pmax, a, b = best
    u_t = math.log1p(max(0.0, target_t))
    s = 1.0 / (1.0 + math.exp(-(a + b * u_t)))
    return min(99.99, pmax * s)


def _weibull_cell(ts, ps, ws, k, lam):
    """Best Pmax and SSE for one (k, lam) cell, or None if degenerate."""
    ss = []
    for t in ts:
        x = t / lam if lam > 0 else 0.0
        ss.append(1.0 - math.exp(-(x ** k)) if x > 0 else 0.0)
    num = sum(w * p * s for w, p, s in zip(ws, ps, ss))
    den = sum(w * s * s for w, s in zip(ws, ss))
    if den <= 1e-12:
        return None
    pmax = num / den
    if pmax <= 0 or pmax > 120.0:
        return None
    sse = 0.0
    for w, p, s in zip(ws, ps, ss):
        sse += w * (pmax * s - p) ** 2
    return (sse, pmax, k, lam)


def fit_weibull(series, target_t):
    """p = Pmax * (1 - exp(-(t/lam)^k)): a classic adoption curve.

    Stage 1 grids k and lam coarsely; stage 2 refines both around the
    best cell (lam in x1.15 steps, k in 0.05 steps, two rounds). The
    coarse grid alone (lam doubling) misplaces lam by up to ~40% and
    costs ~0.1 logit of median accuracy on synthetic Weibull worlds;
    refinement recovers most of it for ~2x the cells of one extra
    coarse level, not the 10x of a uniformly fine grid.
    """
    if len(series) < 3:
        return None
    ts = [max(0.0, t) for t, _, _ in series]
    ps = [p for _, p, _ in series]
    now = ts[-1]
    ws = _weights(ts, now)
    best = None
    for k in (0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2):
        for lam in (3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0):
            cell = _weibull_cell(ts, ps, ws, k, lam)
            if cell is not None and (best is None or cell[0] < best[0]):
                best = cell
    if best is None:
        return None
    for _ in range(2):
        _, _, bk, blam = best
        for dk in (-0.15, -0.1, -0.05, 0.0, 0.05, 0.1, 0.15):
            k = bk + dk
            if k < 0.2 or k > 3.0:
                continue
            for f in (0.66, 0.76, 0.87, 1.0, 1.15, 1.32, 1.52):
                cell = _weibull_cell(ts, ps, ws, k, blam * f)
                if cell is not None and cell[0] < best[0]:
                    best = cell
    _, pmax, k, lam = best
    x = max(0.0, target_t) / lam
    return min(99.99, pmax * (1.0 - math.exp(-(x ** k))))


MODELS = [
    ("loglinear", fit_loglinear),
    ("loglogistic", fit_loglogistic),
    ("weibull", fit_weibull),
]


def _fit_all_models(series, target_t):
    """Median of each ensemble member at target_t (None where it cannot)."""
    series = _thin(series)
    out = {}
    for name, fn in MODELS:
        try:
            out[name] = fn(series, target_t)
        except Exception:
            out[name] = None
    return out


# ---------------------------------------------------------------------------
# Flat-history fallback: the dilution prior.
# ---------------------------------------------------------------------------

def earliness(trophy):
    """How early in a playthrough a trophy is earned: 2 early, 1 mid, 0 late.

    From the trophy's own description: story chapters and first missions
    come first, side content next, grinds / collections / 100% last.
    """
    name = (trophy.get("name") or "").lower()
    desc = (trophy.get("description") or "").lower()
    text = name + " " + desc
    if trophy.get("tier") == "platinum":
        return 0
    if "chapter 1" in text or "gone fishing" in text:
        return 2
    if "breakfast run" in text or "chapter 2" in text:
        return 2
    if "chapter 3" in text or "try each type of activity" in text:
        return 1
    if "complete chapter" in text or "final mission" in text:
        return 0
    if "100% completion" in text or "250 stunts" in text:
        return 0
    if "collect" in text or "every location" in text or "every spot" in text:
        return 0
    if "complete" in text and ("mission" in text or "heist" in text):
        return 1
    return 1


# Newly tracked players' earn rate, relative to the currently tracked
# population's, by earliness: they have barely played yet, so they hold
# almost nothing except the very first trophies.
NEW_PLAYER_RELATIVE = {2: 0.15, 1: 0.05, 0: 0.02}


def n_multiplier_prior(points):
    """Tracked-count multiplier expected at the next refresh.

    The median jump across the history's own refreshes when any exist
    (launch night's one observed refresh was ~6x); a 5x prior when the
    history is flat and no jump has ever been seen.
    """
    ratios = []
    prev = None
    for read in points:
        n = read.get("players_tracked")
        if n and prev and n > prev:
            ratios.append(n / prev)
        if n:
            prev = n
    if not ratios:
        return 5.0
    ratios.sort()
    return min(8.0, max(1.5, ratios[len(ratios) // 2]))


def dilution_median(last_rate, trophy, mult):
    """Median rate after the expected refresh: earners grow a little
    (new players earn at a fraction g of the old rate) while the tracked
    count multiplies by m: f = (1 + g*(m-1)) / m. A trophy at 100% stays.
    """
    if last_rate >= 99.99:
        return 100.0
    g = NEW_PLAYER_RELATIVE[earliness(trophy)]
    f = (1.0 + g * (mult - 1.0)) / mult
    return min(99.99, max(0.01, last_rate * f))


# ---------------------------------------------------------------------------
# Joint hierarchical model: earns ~ Binomial(N, p),
# logit p[t,i] = alpha[i] + f(t) + delta * log(N[t]/N[0]).
# alpha[i] shrunk to PAVA-ordered tier means; f(t) RW2-smoothed.
# Fitted by alternating penalised weighted least squares on the logit
# scale, with binomial-style weights w ~ q(1-q) and recency weights.
# ---------------------------------------------------------------------------

def _pava_nonincreasing(vals):
    """Isotonic regression: return vals pooled to be non-increasing."""
    blocks = [[v, 1] for v in vals]
    i = 0
    while i < len(blocks) - 1:
        if blocks[i][0] < blocks[i + 1][0]:
            tot = blocks[i][0] * blocks[i][1] + blocks[i + 1][0] * blocks[i + 1][1]
            cnt = blocks[i][1] + blocks[i + 1][1]
            blocks[i:i + 2] = [[tot / cnt, cnt]]
            if i > 0:
                i -= 1
        else:
            i += 1
    out = []
    for v, c in blocks:
        out.extend([v] * c)
    return out


def _solve_linear(A, b):
    """Dense Gaussian elimination with partial pivoting (small systems)."""
    n = len(b)
    M = [row[:] + [b[i]] for i, row in enumerate(A)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(M[r][col]))
        if abs(M[piv][col]) < 1e-12:
            continue
        M[col], M[piv] = M[piv], M[col]
        pv = M[col][col]
        for r in range(n):
            if r != col and M[r][col] != 0.0:
                fac = M[r][col] / pv
                for c in range(col, n + 1):
                    M[r][c] -= fac * M[col][c]
    return [M[i][n] / M[i][i] if abs(M[i][i]) > 1e-12 else 0.0
            for i in range(n)]


def joint_fit(case, points, lam_rw2=8.0, lam_alpha=2.0, lam_delta=30.0):
    """Fit the joint model on refresh points up to the case at hand.

    Returns a dict with per-trophy intercepts, the shared trend f at
    each refresh time, the dilution coefficient, tier means and the
    residual scale, or None when the history cannot support it.
    """
    trophies = case["trophies"]
    J = len(points)
    I = len(trophies)
    if J < 4 or I < 2:
        return None
    if J > 40:
        # Evenly thin to 40 refresh points (keep first/last): the joint
        # fit solves a JxJ system per sweep, so an unthinned long final
        # history would blow the 8 s budget for no material gain.
        idx = sorted({round(i * (J - 1) / 39) for i in range(40)})
        points = [points[i] for i in idx]
        J = len(points)
    times = [hours_since_release(r["slot"]) for r in points]
    n0 = points[0].get("players_tracked") or 1
    dlog = []
    for r in points:
        n = r.get("players_tracked") or n0
        dlog.append(math.log(max(1.0, n) / max(1.0, n0)))
    Y = [[None] * J for _ in range(I)]
    W = [[0.0] * J for _ in range(I)]
    now = times[-1]
    for i in range(I):
        for j, read in enumerate(points):
            rates = read.get("rates") or []
            if i >= len(rates) or rates[i] is None:
                continue
            q = rates[i] / 100.0
            Y[i][j] = logit(rates[i])
            rec = 0.5 ** max(0.0, (now - times[j]) / 24.0)
            W[i][j] = rec * min(0.25, max(0.002, q * (1.0 - q))) / 0.25
    alpha = []
    for i in range(I):
        vals = [Y[i][j] for j in range(J) if Y[i][j] is not None]
        alpha.append(vals[-1] if vals else logit(1.0))
    f = [0.0] * J
    delta = 0.0
    tier_of = [TIER_IDX.get(t.get("tier"), 3) for t in trophies]
    mu = [0.0] * 4
    for _sweep in range(12):
        # Tier means, ordered rarest-first non-decreasing in logit
        # (platinum lowest, bronze highest): PAVA on the negated means.
        raw = []
        for k in range(4):
            vals = [alpha[i] for i in range(I) if tier_of[i] == k]
            raw.append(sum(vals) / len(vals) if vals else 0.0)
        mu = [-v for v in _pava_nonincreasing([-v for v in raw])]
        for i in range(I):
            num = lam_alpha * mu[tier_of[i]]
            den = lam_alpha
            for j in range(J):
                if Y[i][j] is None:
                    continue
                num += W[i][j] * (Y[i][j] - f[j] - delta * dlog[j])
                den += W[i][j]
            alpha[i] = num / den if den > 0 else mu[tier_of[i]]
        num = 0.0
        den = lam_delta
        for i in range(I):
            for j in range(J):
                if Y[i][j] is None:
                    continue
                r = Y[i][j] - alpha[i] - f[j]
                num += W[i][j] * r * dlog[j]
                den += W[i][j] * dlog[j] * dlog[j]
        delta = num / den if den > 0 else 0.0
        delta = max(-1.5, min(1.5, delta))
        A = [[0.0] * J for _ in range(J)]
        b = [0.0] * J
        for j in range(J):
            sw = 0.0
            sb = 0.0
            for i in range(I):
                if Y[i][j] is None:
                    continue
                sw += W[i][j]
                sb += W[i][j] * (Y[i][j] - alpha[i] - delta * dlog[j])
            A[j][j] += sw + 1e-6
            b[j] += sb
        for j in range(1, J - 1):
            idx = [j - 1, j, j + 1]
            co = [1.0, -2.0, 1.0]
            for x in range(3):
                for y in range(3):
                    A[idx[x]][idx[y]] += lam_rw2 * co[x] * co[y]
        f_new = _solve_linear(A, b)
        shift = sum(f_new) / J
        f = [v - shift for v in f_new]
        alpha = [a + shift for a in alpha]
    res = []
    for i in range(I):
        for j in range(J):
            if Y[i][j] is None:
                continue
            res.append(Y[i][j] - alpha[i] - f[j] - delta * dlog[j])
    if res:
        res.sort()
        med = res[len(res) // 2]
        devs = sorted(abs(r - med) for r in res)
        sigma_res = 1.4826 * devs[len(devs) // 2]
    else:
        sigma_res = 0.3
    return {"alpha": alpha, "f": f, "times": times, "delta": delta,
            "mu": mu, "sigma_res": max(0.05, sigma_res),
            "dlog": dlog, "n0": n0}


def joint_forecast_median(jfit, case, points, target_t):
    """Per-trophy median logit at the target from a fitted joint model.

    The shared trend is extrapolated with a damped slope (RW2 keeps
    going straight; damping keeps a 4.5-day extrapolation honest), and
    the tracked count at the target comes from a saturating N(t) fit.
    """
    times = jfit["times"]
    f = jfit["f"]
    J = len(times)
    if J >= 3:
        slope = (f[-1] - f[-3]) / max(1.0, times[-1] - times[-3])
    elif J >= 2:
        slope = (f[-1] - f[-2]) / max(1.0, times[-1] - times[-2])
    else:
        slope = 0.0
    horizon = max(0.0, target_t - times[-1])
    damp = math.exp(-horizon / 96.0)
    f_t = f[-1] + slope * horizon * damp
    f_t = max(f[-1] - 1.5, min(f[-1] + 1.5, f_t))
    ns = [(times[j], points[j].get("players_tracked") or 0) for j in range(J)]
    best = None
    for tau in (12.0, 24.0, 48.0, 96.0, 192.0):
        n_first = ns[0][1] or 0
        num = 0.0
        den = 0.0
        for t, n in ns:
            x = max(0.0, t) / (max(0.0, t) + tau) if t > 0 else 0.0
            num += x * (n - n_first)
            den += x * x
        if den <= 1e-12:
            continue
        ninf = num / den
        if ninf <= 0:
            continue
        sse = 0.0
        for t, n in ns:
            x = max(0.0, t) / (max(0.0, t) + tau) if t > 0 else 0.0
            sse += (n_first + ninf * x - n) ** 2
        if best is None or sse < best[0]:
            best = (sse, ninf, tau, n_first)
    if best is not None:
        _, ninf, tau, n_first = best
        x = max(0.0, target_t) / (max(0.0, target_t) + tau)
        n_t = n_first + ninf * x
        d_t = math.log(max(1.0, n_t) / max(1.0, jfit["n0"]))
    else:
        d_t = jfit["dlog"][-1]
    d_t = max(jfit["dlog"][-1], min(jfit["dlog"][-1] + 2.0, d_t))
    return [jfit["alpha"][i] + f_t + jfit["delta"] * d_t
            for i in range(len(case["trophies"]))]


# ---------------------------------------------------------------------------
# Pinball loss and rolling-origin backtests.
# ---------------------------------------------------------------------------

def pinball_loss(actual_pct, quantiles_pct):
    """Mean pinball loss over the five levels, on the logit scale."""
    y = logit(actual_pct)
    total = 0.0
    for tau, q in zip(LEVELS, quantiles_pct):
        d = y - logit(q)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def _origin_reads(points, horizon_h):
    """Up to 8 origin refresh points, spread over the history."""
    if len(points) < 5:
        return []
    if len(points) > 12:
        step = (len(points) - 4) / 8.0
        idx = sorted({min(len(points) - 2, 3 + int(i * step))
                      for i in range(8)})
        return [points[i] for i in idx]
    return points[3:-1]


def backtest(case, points, target_t, horizon_h):
    """Rolling-origin evaluation of the curve ensemble on refresh points.

    Each curve is fit on points up to an origin and scored (pinball,
    spread 0.5 logit) against the first later point at least eval_h
    ahead, or the last point. Returns (weights, sigma_eval, eval_h):
    weights per curve pooled over trophies, sigma_eval the robust
    residual scale of the weighted ensemble median. Empty when the
    history cannot support it.
    """
    if len(points) < 4:
        return {}, None, None
    origins = _origin_reads(points, horizon_h)
    trophies = case["trophies"]
    eval_h = min(horizon_h, 24.0)
    losses = {}
    trials = []
    for origin in origins:
        o_t = hours_since_release(origin["slot"])
        upto = [r for r in points if hours_since_release(r["slot"]) <= o_t]
        later = [r for r in points if hours_since_release(r["slot"]) > o_t]
        if not later:
            continue
        eval_read = None
        for r in later:
            if hours_since_release(r["slot"]) - o_t >= eval_h:
                eval_read = r
                break
        if eval_read is None:
            eval_read = later[-1]
        e_t = hours_since_release(eval_read["slot"])
        eval_rates = eval_read.get("rates") or []
        for idx, trophy in enumerate(trophies):
            if idx >= len(eval_rates):
                continue
            actual = eval_rates[idx]
            if actual is None:
                continue
            series = trophy_series(case, upto, idx)
            if len(series) < 2:
                continue
            fits = _fit_all_models(series, e_t)
            medians = [v for v in fits.values() if v is not None]
            if not medians:
                continue
            for name, med in fits.items():
                if med is None:
                    continue
                q = [inv_logit(logit(med) + z * 0.5) for z in Z]
                losses.setdefault(name, []).append(pinball_loss(actual, q))
            trials.append((actual, fits))
    if not losses or not trials:
        return {}, None, None
    mean_loss = {k: sum(v) / len(v) for k, v in losses.items()}
    best = min(mean_loss.values())
    weights = {}
    total = 0.0
    for k, v in mean_loss.items():
        w = 1.0 / max(0.05, (v / best) ** 2)
        weights[k] = w
        total += w
    weights = {k: w / total for k, w in weights.items()}
    residuals = []
    for actual, fits in trials:
        wsum = sum(weights.get(n, 0.0) for n in fits if fits[n] is not None)
        if wsum <= 0:
            continue
        avg = sum(weights.get(n, 0.0) * logit(m)
                  for n, m in fits.items() if m is not None) / wsum
        residuals.append(logit(actual) - avg)
    if not residuals:
        return weights, None, eval_h
    residuals.sort()
    med = residuals[len(residuals) // 2]
    devs = sorted(abs(r - med) for r in residuals)
    mad = devs[len(devs) // 2]
    sigma_eval = 1.4826 * mad
    return weights, sigma_eval, eval_h


def joint_cv_weight(case, points, horizon_h, curve_weights):
    """The joint model's blend weight, earned (or not) by CV.

    Same origins as backtest(), but the joint model and the
    backtest-WEIGHTED curve ensemble are scored head to head at the
    longest horizon the history supports. The comparison must use the
    weighted ensemble — the equal-weight average is far weaker and
    would let the joint model in on worlds where it still loses to the
    forecast actually shipped. The joint model enters only if it beats
    the weighted curves by at least 10% there, with weight
    proportional to the win and capped at 0.30; otherwise the weight
    is exactly 0. Returns 0.0 when the history cannot support the
    comparison.
    """
    if len(points) < 6:
        return 0.0
    origins = _origin_reads(points, horizon_h)
    if len(origins) < 2:
        return 0.0
    eval_h = min(horizon_h, 24.0)
    lj, lc = [], []
    for origin in origins:
        o_t = hours_since_release(origin["slot"])
        upto = [r for r in points if hours_since_release(r["slot"]) <= o_t]
        later = [r for r in points if hours_since_release(r["slot"]) > o_t]
        if not later:
            continue
        eval_read = None
        for r in later:
            if hours_since_release(r["slot"]) - o_t >= eval_h:
                eval_read = r
                break
        if eval_read is None:
            eval_read = later[-1]
        e_t = hours_since_release(eval_read["slot"])
        eval_rates = eval_read.get("rates") or []
        try:
            jf = joint_fit(case, upto)
            jmed = joint_forecast_median(jf, case, upto, e_t) if jf else None
        except Exception:
            jmed = None
        if jmed is None:
            continue
        for idx in range(len(case["trophies"])):
            if idx >= len(eval_rates) or eval_rates[idx] is None:
                continue
            actual = eval_rates[idx]
            series = trophy_series(case, upto, idx)
            if len(series) < 2:
                continue
            fits = _fit_all_models(series, e_t)
            wsum = sum(curve_weights.get(n, 0.0) for n, v in fits.items()
                       if v is not None)
            if wsum <= 0:
                continue
            c_logit = sum(curve_weights.get(n, 0.0) * logit(v)
                          for n, v in fits.items() if v is not None) / wsum
            qj = [inv_logit(jmed[idx] + z * 0.5) for z in Z]
            qc = [inv_logit(c_logit + z * 0.5) for z in Z]
            lj.append(pinball_loss(actual, qj))
            lc.append(pinball_loss(actual, qc))
    if len(lj) < 10:
        return 0.0
    mj = sum(lj) / len(lj)
    mc = sum(lc) / len(lc)
    if mj >= 0.9 * mc:
        return 0.0
    return max(0.0, min(0.30, 0.30 * (mc - mj) / max(0.05, mc)))


# ---------------------------------------------------------------------------
# Forecast assembly.
# ---------------------------------------------------------------------------

def forecast_trophy(case, points, idx, trophy, weights, target_t,
                    horizon_h, sigma_floor, joint_median, w_joint):
    """(median_pct, sigma_logit) for one trophy."""
    series = trophy_series(case, points, idx)
    last_rate = series[-1][1] if series else None
    if last_rate is None:
        return 1.0, 1.2
    if last_rate >= 99.99:
        return 100.0, 0.05
    if len(points) < 2 or series_moved(series) < 0.02:
        return dilution_median(last_rate, trophy,
                               n_multiplier_prior(points)), max(1.0, sigma_floor)
    fits = _fit_all_models(series, target_t)
    medians = [(weights.get(name, 0.0), med)
               for name, med in fits.items() if med is not None]
    if not medians:
        return last_rate, sigma_floor
    wsum = sum(w for w, _ in medians)
    avg_logit = sum(w * logit(m) for w, m in medians) / wsum
    if joint_median is not None and w_joint > 0.0:
        avg_logit = (1.0 - w_joint) * avg_logit + w_joint * joint_median[idx]
    median = inv_logit(avg_logit)
    spread_members = 0.0
    if len(medians) > 1:
        var = sum(w * (logit(m) - avg_logit) ** 2 for w, m in medians) / wsum
        spread_members = math.sqrt(max(0.0, var))
    sigma = math.sqrt(sigma_floor ** 2 + spread_members ** 2)
    return median, sigma


def forecast_case(case):
    points = refresh_points(case)
    target_t = hours_since_release(case["target"])
    at_t = hours_since_release(case["at"])
    horizon_h = max(0.5, (target_t - at_t))
    horizon_d = horizon_h / 24.0
    weights, sigma_eval, eval_h = backtest(case, points, target_t, horizon_h)
    if not weights:
        weights = {"loglinear": 0.35, "loglogistic": 0.35, "weibull": 0.3}
    w_joint = 0.0
    joint_median = None
    if len(points) >= 6:
        try:
            w_joint = joint_cv_weight(case, points, horizon_h, weights)
            if w_joint > 0.0:
                jf = joint_fit(case, points)
                if jf is not None:
                    joint_median = joint_forecast_median(jf, case, points,
                                                          target_t)
        except Exception:
            w_joint = 0.0
            joint_median = None
    if sigma_eval is None:
        sigma_floor = 0.9 * (min(horizon_d, 4.0) / 0.0625) ** 0.25
        sigma_floor = min(sigma_floor, 1.6)
    else:
        ratio = max(1.0, horizon_h / max(eval_h, 1.0))
        grow = 1.0 + 0.10 * math.log2(ratio)
        sigma_floor = min(sigma_eval * min(grow, 1.2), 0.9)
        sigma_floor = max(sigma_floor, 0.25)
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        median, sigma = forecast_trophy(case, points, idx, trophy, weights,
                                        target_t, horizon_h, sigma_floor,
                                        joint_median, w_joint)
        sigma = max(0.05, min(2.0, sigma))
        qs = []
        for z in Z:
            q = inv_logit(logit(median) + z * sigma)
            qs.append(round4(min(99.99, max(0.01, q))))
        for i in range(1, len(qs)):
            if qs[i] < qs[i - 1]:
                qs[i] = qs[i - 1]
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        if case.get("levels") != LEVELS:
            raise ValueError(f"{path.name} asks for levels {case.get('levels')}, not {LEVELS}")
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
