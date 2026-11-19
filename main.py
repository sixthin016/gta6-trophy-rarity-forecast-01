"""Forecast GTA VI trophy earn rates on Exophase, one week after launch.

Entry for make.fun's gta6-trophy-rarity-forecast-01 (program/v1, Python
3.13, standard library only). For every trophy it writes five quantiles
(5/25/50/75/95%) of the earn rate at the case's target, scored by pinball
loss on the logit scale.

What the data looks like, and why the forecaster is shaped this way:

* Exophase does not update its counts continuously. It refreshes a game's
  page on its own schedule, hours apart; between refreshes every read is a
  byte-for-byte repeat of the last refresh. At a refresh the tracked-player
  count jumps (launch night: 638 -> 3,868) and almost every rate falls,
  because the newly tracked players have earned far fewer trophies.
  Fitting curves to the raw reads would overweight stale repeats, so the
  history is first collapsed to its distinct refresh points.
* A trophy's rate is earners / tracked, so dilution is already inside the
  rates a refresh prints. Three saturating adoption curves are fitted to
  the refresh points per trophy - logit-linear in log time, a
  log-logistic with a fitted ceiling, and a Weibull adoption curve - and
  combined with weights chosen by rolling-origin backtest inside the
  history itself. (Fitting earners and tracked counts separately was
  considered and rejected: two extrapolations compound their errors.)
* The spread comes from the backtest residuals of the weighted ensemble,
  grown slowly with the horizon and capped; it has to cover refresh-timing
  uncertainty as well as trend uncertainty.
* A flat history (launch night: every read identical) carries no trend
  signal at all. The forecast then falls back to a dilution prior: a
  refresh is expected before the target, the tracked count multiplies, and
  newly tracked players hold almost nothing yet except the very first
  trophies, so rates fall by roughly the player-count factor, with a wide
  spread. A trophy already at 100% stays at 100%.

Deterministic: two runs on the same case write identical bytes. No
network, no third-party packages, well inside 8 s / 1 CPU / 1 GiB.
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
_NORM = NormalDist()
ZS = [_NORM.inv_cdf(t) for t in LEVELS]

# Release day, 00:00 UTC: the origin of the log-time axis.
RELEASE_EPOCH = datetime(2026, 11, 19, tzinfo=timezone.utc).timestamp()

# A read half a day old counts half as much as a fresh one (recency decay).
HALF_LIFE_HOURS = 24.0


# ---------------------------------------------------------------------------
# Small numeric helpers (percent <-> logit, rounding, time).
# ---------------------------------------------------------------------------

def clip(pct):
    """Clamp a percent into the range the scorer's logit is defined on."""
    return 99.99 if pct > 99.99 else (0.01 if pct < 0.01 else pct)


def to_logit(pct):
    p = clip(pct)
    return math.log(p / (100.0 - p))


def from_logit(y):
    """Inverse logit, written to stay finite for extreme y."""
    if y >= 0:
        return 100.0 / (1.0 + math.exp(-y))
    e = math.exp(y)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, like JavaScript's Math.round.

    Whole numbers come back as int so json.dumps writes 100, not 100.0.
    """
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if value == int(value) else value


def to_epoch(iso):
    return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()


def hours_since_release(iso):
    return (to_epoch(iso) - RELEASE_EPOCH) / 3600.0


# ---------------------------------------------------------------------------
# History: refresh points and per-trophy series.
# ---------------------------------------------------------------------------

def refresh_points(history):
    """Collapse the reads to their distinct refreshes.

    Consecutive reads with the same player count and the same rates are
    one refresh repeated; keep the first read of each run. A read with no
    usable rate at all is dropped.
    """
    points = []
    seen = None
    for read in history or []:
        rates = read.get("rates") or []
        if not rates or all(r is None for r in rates):
            continue
        key = (read.get("players_tracked"), tuple(rates))
        if key == seen:
            continue
        seen = key
        points.append(read)
    return points


def trophy_series(points, idx):
    """[(hours_since_release, rate_pct)] for one trophy, oldest first."""
    out = []
    for read in points:
        rates = read.get("rates") or []
        if idx >= len(rates):
            continue
        rate = rates[idx]
        if rate is None:
            continue
        out.append((hours_since_release(read["slot"]), float(rate)))
    return out


def total_logit_move(series):
    if len(series) < 2:
        return 0.0
    vals = [to_logit(p) for _, p in series]
    return max(vals) - min(vals)


# ---------------------------------------------------------------------------
# Weighted least squares line (shared by the curve fits).
# ---------------------------------------------------------------------------

def recency_weights(times):
    latest = times[-1]
    return [0.5 ** max(0.0, (latest - t) / HALF_LIFE_HOURS) for t in times]


def wls_line(xs, ys, ws):
    """Weighted least squares for y ~ a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return None
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    slope = sxy / sxx
    return my - slope * mx, slope


def thin(series, cap=40):
    """Evenly thin a long series, keeping its first and last points.

    The final history can hold hundreds of refresh points; the fits are
    O(points x grid), so thinning keeps the run far inside the budget.
    """
    if len(series) <= cap:
        return series
    keep = sorted({round(i * (len(series) - 1) / (cap - 1)) for i in range(cap)})
    return [series[i] for i in keep]


# ---------------------------------------------------------------------------
# The three ensemble members. Each takes a per-trophy series and a target
# time (hours since release) and returns a median rate in percent, or None
# when the series cannot support that fit.
# ---------------------------------------------------------------------------

def fit_log_time(series, target_h):
    """logit(p) linear in ln(1 + t): the fitted-curve baseline's shape."""
    if len(series) < 2:
        return None
    xs = [math.log1p(max(0.0, t)) for t, _ in series]
    ys = [to_logit(p) for _, p in series]
    line = wls_line(xs, ys, recency_weights([t for t, _ in series]))
    if line is None:
        return None
    a, b = line
    return from_logit(a + b * math.log1p(max(0.0, target_h)))


def fit_ceiling_logistic(series, target_h):
    """p = C / (1 + exp(-(a + b*ln(1+t)))) with the ceiling C profiled.

    For each candidate ceiling the transformed rates are linear in log
    time; the ceiling with the lowest weighted residual wins.
    """
    if len(series) < 3:
        return None
    us = [math.log1p(max(0.0, t)) for t, _ in series]
    ps = [p for _, p in series]
    ws = recency_weights([t for t, _ in series])
    floor = min(99.9, max(ps) * 1.05 + 0.05)
    best = None
    for step in range(25):
        frac = step / 24.0
        ceiling = floor + (99.9 - floor) * frac * frac
        ys = []
        usable = True
        for p in ps:
            r = p / ceiling
            if r <= 0.001 or r >= 0.999:
                usable = False
                break
            ys.append(math.log(r / (1.0 - r)))
        if not usable:
            continue
        line = wls_line(us, ys, ws)
        if line is None:
            continue
        a, b = line
        sse = sum(w * (a + b * u - y) ** 2 for w, u, y in zip(ws, us, ys))
        if best is None or sse < best[0]:
            best = (sse, ceiling, a, b)
    if best is None:
        return None
    _, ceiling, a, b = best
    u_target = math.log1p(max(0.0, target_h))
    return min(99.99, ceiling / (1.0 + math.exp(-(a + b * u_target))))


def fit_weibull_adoption(series, target_h):
    """p = C * (1 - exp(-(t/lambda)^k)): classic adoption curve.

    Shape k and scale lambda are gridded; the ceiling C is the weighted
    least-squares scale for that (k, lambda). Time runs from release.
    """
    if len(series) < 3:
        return None
    ts = [max(0.0, t) for t, _ in series]
    ps = [p for _, p in series]
    ws = recency_weights(ts)
    best = None
    for k in (0.4, 0.6, 0.8, 1.0, 1.3, 1.7, 2.2):
        for lam in (3.0, 6.0, 12.0, 24.0, 48.0, 96.0, 192.0, 384.0):
            shape = [(1.0 - math.exp(-((t / lam) ** k))) if t > 0 else 0.0
                     for t in ts]
            num = sum(w * p * s for w, p, s in zip(ws, ps, shape))
            den = sum(w * s * s for w, s in zip(ws, shape))
            if den <= 1e-12:
                continue
            ceiling = num / den
            if ceiling <= 0 or ceiling > 120.0:
                continue
            sse = sum(w * (ceiling * s - p) ** 2 for w, p, s in zip(ws, ps, shape))
            if best is None or sse < best[0]:
                best = (sse, ceiling, k, lam)
    if best is None:
        return None
    _, ceiling, k, lam = best
    x = max(0.0, target_h) / lam
    return min(99.99, ceiling * (1.0 - math.exp(-(x ** k))))


FITTERS = (
    ("log_time", fit_log_time),
    ("ceiling_logistic", fit_ceiling_logistic),
    ("weibull", fit_weibull_adoption),
)


def fit_all(series, target_h):
    """{member name: median rate} for every member that can fit."""
    series = thin(series)
    out = {}
    for name, fn in FITTERS:
        try:
            value = fn(series, target_h)
        except Exception:
            value = None
        if value is not None:
            out[name] = value
    return out


# ---------------------------------------------------------------------------
# Flat-history fallback: the dilution prior.
# ---------------------------------------------------------------------------

def earliness(trophy):
    """2 = earned very early in a playthrough, 1 = mid, 0 = late.

    Read off the trophy's own text: the first story chapters and opening
    missions come first, side content next, grinds and collections last.
    """
    text = ((trophy.get("name") or "") + " " +
            (trophy.get("description") or "")).lower()
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


# What fraction of the current rate a newly tracked player holds, by
# earliness: they have barely played yet, so almost nothing except the
# very first trophies.
NEW_PLAYER_SHARE = {2: 0.15, 1: 0.05, 0: 0.02}
# How much the tracked count is expected to multiply at the refresh
# before the target (launch night's refresh: 638 -> 3,868, about 6x).
EXPECTED_N_GROWTH = 5.0


def dilution_median(last_rate, trophy):
    """Median rate after the expected refresh.

    Earners grow a little (new players earn at share g of the old rate)
    while the tracked count multiplies by m, so the rate is scaled by
    (1 + g*(m-1)) / m. A trophy at 100% stays at 100%.
    """
    if last_rate >= 99.99:
        return 100.0
    g = NEW_PLAYER_SHARE[earliness(trophy)]
    m = EXPECTED_N_GROWTH
    factor = (1.0 + g * (m - 1.0)) / m
    return min(99.99, max(0.01, last_rate * factor))


# ---------------------------------------------------------------------------
# Rolling-origin backtest: ensemble weights and spread calibration.
# ---------------------------------------------------------------------------

def pinball(actual_pct, quantiles_pct):
    """Mean pinball loss over the five levels, on the logit scale."""
    y = to_logit(actual_pct)
    total = 0.0
    for tau, q in zip(LEVELS, quantiles_pct):
        d = y - to_logit(q)
        total += 2.0 * max(tau * d, (tau - 1.0) * d)
    return total / len(LEVELS)


def rolling_backtest(case, points, horizon_h):
    """Fit each member on refresh points up to an origin, score it against
    a later refresh, pool over trophies.

    Returns (weights, sigma, eval_horizon): weights per member, sigma the
    robust residual scale (logit units) of the weighted ensemble at the
    evaluation horizon, and that horizon in hours. Empty weights when the
    history cannot support a backtest.
    """
    if len(points) < 4:
        return {}, None, None
    if len(points) > 10:
        step = (len(points) - 3) / 8.0
        origin_idx = sorted({min(len(points) - 2, 2 + int(i * step))
                             for i in range(8)})
        origins = [points[i] for i in origin_idx]
    else:
        origins = points[2:-1]
    trophies = case["trophies"]
    eval_h = min(horizon_h, 24.0)
    losses = {}
    trials = []
    for origin in origins:
        origin_t = hours_since_release(origin["slot"])
        upto = [r for r in points if hours_since_release(r["slot"]) <= origin_t]
        later = [r for r in points if hours_since_release(r["slot"]) > origin_t]
        if not later:
            continue
        eval_read = None
        for read in later:
            if hours_since_release(read["slot"]) - origin_t >= eval_h:
                eval_read = read
                break
        if eval_read is None:
            eval_read = later[-1]
        eval_t = hours_since_release(eval_read["slot"])
        eval_rates = eval_read.get("rates") or []
        for idx, trophy in enumerate(trophies):
            if idx >= len(eval_rates):
                continue
            actual = eval_rates[idx]
            if actual is None:
                continue
            series = trophy_series(upto, idx)
            if len(series) < 2:
                continue
            fits = fit_all(series, eval_t)
            if not fits:
                continue
            for name, med in fits.items():
                q = [from_logit(to_logit(med) + z * 0.5) for z in ZS]
                losses.setdefault(name, []).append(pinball(actual, q))
            trials.append((actual, fits))
    if not losses or not trials:
        return {}, None, None
    mean_loss = {k: sum(v) / len(v) for k, v in losses.items()}
    best = min(mean_loss.values())
    weights = {}
    total = 0.0
    for name, loss in mean_loss.items():
        # A member twice as bad as the best still keeps some weight: the
        # ensemble is insurance against any one curve shape failing.
        w = 1.0 / max(0.05, (loss / best) ** 2)
        weights[name] = w
        total += w
    weights = {k: w / total for k, w in weights.items()}
    # Residuals of the weighted ensemble's own median: that is the
    # forecast whose spread sigma calibrates.
    residuals = []
    for actual, fits in trials:
        wsum = sum(weights.get(n, 0.0) for n in fits)
        if wsum <= 0:
            continue
        avg = sum(weights.get(n, 0.0) * to_logit(m)
                  for n, m in fits.items()) / wsum
        residuals.append(to_logit(actual) - avg)
    if not residuals:
        return weights, None, eval_h
    residuals.sort()
    med = residuals[len(residuals) // 2]
    devs = sorted(abs(r - med) for r in residuals)
    mad = devs[len(devs) // 2]
    return weights, 1.4826 * mad, eval_h


# ---------------------------------------------------------------------------
# Forecast assembly.
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {"log_time": 0.35, "ceiling_logistic": 0.35, "weibull": 0.30}


def forecast_one(case, points, idx, trophy, weights, target_h, horizon_h,
                 sigma_floor):
    """(median_pct, sigma_logit) for a single trophy."""
    series = trophy_series(points, idx)
    last_rate = series[-1][1] if series else None
    if last_rate is None:
        # No read ever listed this trophy: a mid-pack prior, wide spread.
        return 1.0, 1.2
    if last_rate >= 99.99:
        return 100.0, 0.05
    if len(points) < 2 or total_logit_move(series) < 0.02:
        # Flat history: no trend signal. Expect the refresh dilution.
        return dilution_median(last_rate, trophy), max(1.0, sigma_floor)
    fits = fit_all(series, target_h)
    usable = [(weights.get(name, 0.0), med)
              for name, med in fits.items()]
    if not usable:
        return last_rate, sigma_floor
    wsum = sum(w for w, _ in usable)
    avg_logit = sum(w * to_logit(m) for w, m in usable) / wsum
    median = from_logit(avg_logit)
    # Spread: the floor, plus the members' disagreement, grown with
    # the horizon. A weighted average of logits is used (not a median):
    # the members are all fits of the same data.
    spread_members = 0.0
    if len(usable) > 1:
        var = sum(w * (to_logit(m) - avg_logit) ** 2
                  for w, m in usable) / wsum
        spread_members = math.sqrt(max(0.0, var))
    sigma = math.sqrt(sigma_floor ** 2 + spread_members ** 2)
    return median, sigma


def spread_floor(sigma_eval, eval_h, horizon_h):
    """Logit spread floor for the forecast horizon.

    The backtest residual already measures this forecast's own errors at
    a day-ish horizon, and those errors grow only slowly with horizon
    once saturation binds (most of the residual is level noise and
    curve-shape bias, not a random walk), so grow it log-wise and cap it
    hard: a spread much wider than the real error costs pinball on every
    trophy.
    """
    horizon_d = horizon_h / 24.0
    if sigma_eval is None:
        floor = 0.9 * (min(horizon_d, 4.0) / 0.0625) ** 0.25
        return min(floor, 1.6)
    ratio = max(1.0, horizon_h / max(eval_h, 1.0))
    grow = 1.0 + 0.10 * math.log2(ratio)
    return max(min(sigma_eval * min(grow, 1.2), 0.9), 0.25)


def forecast_case(case):
    points = refresh_points(case.get("history", []))
    target_h = hours_since_release(case["target"])
    at_h = hours_since_release(case["at"])
    horizon_h = max(0.5, target_h - at_h)
    weights, sigma_eval, eval_h = rolling_backtest(case, points, horizon_h)
    if not weights:
        weights = dict(DEFAULT_WEIGHTS)
    floor = spread_floor(sigma_eval, eval_h or 1.0, horizon_h)
    rows = []
    for idx, trophy in enumerate(case["trophies"]):
        median, sigma = forecast_one(case, points, idx, trophy, weights,
                                     target_h, horizon_h, floor)
        sigma = max(0.05, min(2.0, sigma))
        qs = []
        for z in ZS:
            q = from_logit(to_logit(median) + z * sigma)
            qs.append(round4(min(99.99, max(0.01, q))))
        # Keep the quantiles non-decreasing after rounding.
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
            raise ValueError(
                f"{path.name} asks for levels {case.get('levels')}, "
                f"not {LEVELS}")
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
