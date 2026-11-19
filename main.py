"""Forecast Grand Theft Auto VI trophy earn rates on Exophase at a target instant.

Reads one case per *.in file (schema makefun/gta6-trophies-case/v1) and writes
one forecast per case: five quantiles (0.05/0.25/0.5/0.75/0.95) of each trophy's
earn rate at the case's target, scored by pinball loss on the logit scale.

Method (Python standard library only, fully deterministic):

1. Refresh points. Exophase refreshes a game's counts on its own schedule,
   hours apart; between refreshes every read is byte-identical. The history
   is therefore collapsed to its distinct states, each kept at the time it
   was first observed. Fitting curves to raw reads would overweight the
   flat stretches.

2. Adoption-curve ensemble. Per trophy, three curves are fitted to the
   refresh points in logit space against log time, and read at the target:
     - a log-linear fit of logit(rate) on ln(hours + s);
     - a log-logistic curve with a fitted ceiling C:
       rate = C / (1 + exp(-(a + b ln(hours + s))));
     - a Weibull curve with a fitted ceiling C:
       rate = C * (1 - exp(-exp(a + b ln(hours + s)))).
   The ceiling is picked from a small grid by least squares in logit
   space, and so is the shift s: adoption started some unknown time
   before the first read (early players), and the shift that
   straightens the history is the clock the curve really runs on.
   Every fit weights recent points most (24-hour half-life): the
   slope that matters for the target is the current one, not launch
   night's.
   No E/N decomposition is used: splitting earners and players into two
   separate extrapolations was tried and roughly doubled the error.

3. Ensemble weights from fit quality. Each model is scored by how
   well it fits each trophy's own history (mean squared logit error,
   pooled across trophies and per trophy, shrunk toward the pooled
   weights): a model that cannot bend to a saturating trophy's curve
   must not carry its forecast, and one that cannot follow a
   still-rising trophy must not carry that one either. The spread
   comes from the median absolute residual of the weighted ensemble
   at long rolling origins (forecasting the last point from 12+
   hours before it), grown with the log-distance from the last point
   to the target and widened, trophy by trophy, by how far the
   models disagree at the target, capped at 0.9 logit units.

4. Dilution prior for a flat history. When the whole history is a single
   refresh point there is nothing to fit, and the one refresh the house
   ever observed (638 -> 3,868 players tracked, every rate but the 100%
   one falling) says what happens next: a refresh is expected, the player
   base grows roughly sixfold, and the newcomers hold almost no trophy
   except the one everybody holds. The median is therefore the diluted
   rate: old earners plus a thin spread of new earners (about 0.032 extra
   trophies per newcomer, shared in proportion to the old rates), over
   the grown player base. The trophy everybody holds stays at ~100%.

5. Quantiles. A logit-normal band around the ensemble median:
   q_tau = inv_logit(logit(median) + z_tau * spread), z the standard
   normal quantiles, rounded to 4 decimals, clamped to [0.01, 99.99].
"""

import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
Z = [-1.645, -0.674, 0.0, 0.674, 1.645]

LO, HI = 0.01, 99.99

# Spread (logit units) and dilution-prior constants, tuned on the practice
# cut and on synthetic launch worlds; see tools/backtest.py.
SPREAD_FLOOR = 0.12
SPREAD_CAP = 0.9
FLAT_SPREAD = 0.6
NO_ORIGIN_SPREAD = 0.30
DILUTION_GROWTH = 6.0        # player-base growth across one expected refresh
EXTRA_PER_NEWCOMER = 0.032   # trophies a newcomer holds beyond the universal one
SLOPE_MIN, SLOPE_MAX = -2.5, 3.0
# Candidate shifts (hours) of the adoption clock before the first read.
# The first read already shows earned trophies, so play began before
# it -- by up to about half a day for the earliest time zones.
SHIFTS = (1.0, 6.0, 12.0, 24.0)
# How far per-trophy weights are shrunk toward the pooled weights.
SHRINK = 0.3

# Fallback medians (percent) when a trophy has no usable rate at all.
TIER_PRIOR = {"platinum": 0.5, "gold": 1.0, "silver": 2.0, "bronze": 3.0}


def clamp_pct(x):
    return min(HI, max(LO, x))


def logit(pct):
    p = clamp_pct(pct)
    return math.log(p / (100.0 - p))


def inv_logit(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up, as the house baseline's JS does."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(text):
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def ols(xs, ys, ws=None):
    """Least squares y ~ a + b x, optionally weighted. Returns (a, b)."""
    if ws is None:
        ws = [1.0] * len(xs)
    sw = sum(ws)
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return my, 0.0
    b = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys)) / sxx
    return my - b * mx, b


def recency_weights(series):
    """Per-point weights for a fit: recent points count most, with a
    24-hour half-life. The slope that matters for the target is the
    current one, not launch night's."""
    last_u = max(u for u, _ in series)
    return [0.5 ** ((last_u - u) / 24.0) for u, _ in series]


def refresh_points(case):
    """Collapse consecutive identical reads to one point each.

    Returns [(epoch_seconds, players_tracked, rates_tuple)] in time order,
    one entry per distinct page state, timed at its first observation.
    """
    points = []
    for read in case.get("history") or []:
        try:
            when = parse_time(read["at"]).timestamp()
        except (KeyError, TypeError, ValueError):
            continue
        rates = tuple(read.get("rates") or [])
        state = (read.get("players_tracked"), rates)
        if points and (points[-1][1], points[-1][2]) == state:
            continue
        points.append((when, read.get("players_tracked"), rates))
    return points


def trophy_series(points, index):
    """[(hours_since_first_point, rate)] for one trophy, skipping nulls."""
    series = []
    if not points:
        return series
    t0 = points[0][0]
    for when, _n, rates in points:
        if index < len(rates) and rates[index] is not None:
            series.append(((when - t0) / 3600.0, float(rates[index])))
    return series


# ---------------------------------------------------------------- models

def fit_loglinear(series):
    """logit(rate) ~ a + b ln(u + s). The shift s is fitted: adoption
    started some unknown time before the first read (early players),
    and the shift that straightens the history is the clock the
    curve is really running on. Returns predict(u) -> logit."""
    if len(series) < 2:
        return None
    ws = recency_weights(series)
    best = None
    for s in SHIFTS:
        xs = [math.log(u + s) for u, _ in series]
        ys = [logit(p) for _, p in series]
        a, b = ols(xs, ys, ws)
        b = min(SLOPE_MAX, max(SLOPE_MIN, b))
        sse = sum(w * (a + b * x - y) ** 2
                  for w, x, y in zip(ws, xs, ys))
        if best is None or sse < best[0]:
            best = (sse, a, b, s)
    if best is None:
        return None
    sse, a, b, s = best

    def predict(u):
        return a + b * math.log(u + s)

    return predict, sse / sum(ws)


def _fit_ceiling(series, to_y, from_y):
    """Generic ceiling-curve fit.

    to_y(p_frac, c_frac) maps a rate onto the linear scale the curve is
    straight on; from_y(y, c_frac) maps back to a rate fraction. Both
    the ceiling and the time shift are chosen from grids by least
    squares in logit space.
    """
    if len(series) < 2:
        return None
    top = max(p for _, p in series)
    factors = (1.05, 1.5, 2.0, 3.0, 5.0, 10.0)
    ceilings = sorted({min(HI, top * f) for f in factors})
    if top < 90.0:
        ceilings.append(HI)
    ws_all = recency_weights(series)
    best = None
    for s in SHIFTS:
        for ceil in ceilings:
            cf = ceil / 100.0
            kept, xs, ys, ws = [], [], [], []
            for (u, p), w in zip(series, ws_all):
                pf = p / 100.0
                if pf <= 0.0 or pf >= cf:
                    continue
                kept.append((u, p))
                xs.append(math.log(u + s))
                ys.append(to_y(pf, cf))
                ws.append(w)
            if len(xs) < 2:
                continue
            a, b = ols(xs, ys, ws)
            sse = 0.0
            for (u, p), x, w in zip(kept, xs, ws):
                fitted = from_y(a + b * x, cf) * 100.0
                sse += w * (logit(fitted) - logit(p)) ** 2
            mean_sse = sse / sum(ws)
            if best is None or mean_sse < best[0]:
                best = (mean_sse, a, b, cf, s)
    if best is None:
        return None
    mse, a, b, cf, s = best

    def predict(u):
        return logit(from_y(a + b * math.log(u + s), cf) * 100.0)

    return predict, mse


def fit_loglogistic(series):
    """rate = C / (1 + exp(-(a + b ln(u + s))))."""
    return _fit_ceiling(
        series,
        to_y=lambda pf, cf: math.log(pf / (cf - pf)),
        from_y=lambda y, cf: cf / (1.0 + math.exp(-max(-700.0, min(700.0, y)))),
    )


def fit_weibull(series):
    """rate = C * (1 - exp(-exp(a + b ln(u + s))))."""
    def to_y(pf, cf):
        return math.log(-math.log(1.0 - pf / cf))

    def from_y(y, cf):
        if y > 700:
            return cf
        if y < -700:
            return 0.0
        return cf * (1.0 - math.exp(-math.exp(y)))

    return _fit_ceiling(series, to_y, from_y)


MODELS = (("loglinear", fit_loglinear),
          ("loglogistic", fit_loglogistic),
          ("weibull", fit_weibull))


def _thin(series, limit=24):
    """Thin a long series to at most `limit` points, evenly spaced and
    keeping the first and last. Fitting cost grows with the number of
    points; past two dozen the shape is already settled, and a case
    whose every read differs (hundreds of refresh points) must still
    finish well inside its 8 seconds."""
    if len(series) <= limit:
        return series
    step = (len(series) - 1) / (limit - 1)
    keep = sorted({min(len(series) - 1, int(round(k * step)))
                   for k in range(limit)})
    return [series[i] for i in keep]


def fit_all(series):
    """{name: (predict, mse)} for every model that fits this trophy's
    series. mse is the mean squared logit error of the fit, the number
    the ensemble weights are built from."""
    series = _thin(series)
    fitted = {}
    for name, fitter in MODELS:
        try:
            got = fitter(series)
        except (ValueError, OverflowError, ZeroDivisionError):
            got = None
        if got is not None:
            fitted[name] = got
    return fitted


# ------------------------------------------------------- rolling origins

def _softmax_weights(mse_by_model):
    """Weights from mean squared fit errors: exp(-mse/scale), the scale
    the mean error, floored so that models which all fit within a
    tenth of a logit count as tied instead of splitting hairs."""
    if not mse_by_model:
        return {}
    scale = sum(mse_by_model.values()) / len(mse_by_model)
    scale = max(scale, 0.01)
    raw = {n: math.exp(-m / scale) for n, m in mse_by_model.items()}
    total = sum(raw.values())
    return {n: v / total for n, v in raw.items()}


def weights_from_fits(fits_by_trophy):
    """Ensemble weights from how well each model fits each trophy's
    own history: a model that cannot bend to a saturating trophy's
    curve must not carry its forecast, and one that cannot follow a
    still-rising trophy must not carry that one either. Pooled weights
    come from the mean fit error across trophies; each trophy's own
    weights are shrunk part-way toward them, since one trophy's fit
    is few points.

    Returns (pooled, combined): pooled {name: w}, combined one dict
    of weights per trophy, aligned with fits_by_trophy.
    """
    pooled_mse = {}
    for fits in fits_by_trophy:
        for name, (_fn, mse) in fits.items():
            pooled_mse.setdefault(name, []).append(mse)
    pooled_means = {n: sum(v) / len(v) for n, v in pooled_mse.items() if v}
    pooled = _softmax_weights(pooled_means)
    if not pooled:
        pooled = {name: 1.0 / len(MODELS) for name, _ in MODELS}
    combined = []
    for fits in fits_by_trophy:
        own = {n: mse for n, (_fn, mse) in fits.items()}
        pt = _softmax_weights(own)
        if not pt:
            combined.append(dict(pooled))
            continue
        names = set(pooled) | set(pt)
        combined.append({n: SHRINK * pooled.get(n, 0.0)
                         + (1.0 - SHRINK) * pt.get(n, 0.0) for n in names})
    return pooled, combined


def rolling_origin(series_by_trophy, combined):
    """Measure the ensemble's extrapolation error at long origins, for
    the spread: with the given per-trophy weights, forecast the last
    point from origins 12+ hours before it (recent origins only, the
    shape that matters), and take the median absolute logit error.

    Returns (spread0, bt_scale): the median absolute error and the
    median log-time distance the origins forecast over. Either may be
    None when there is no origin to score.
    """
    spans = [s[-1][0] - s[0][0] for s in series_by_trophy if len(s) >= 3]
    long_mode = bool(spans) and max(spans) >= 24.0
    abs_err = []
    horizons = []
    for ti, series in enumerate(series_by_trophy):
        if len(series) < 3:
            continue
        w = combined[ti] if combined else {}
        last_u, last_rate = series[-1]
        if long_mode:
            # Score only recent origins: the shape of the last day and a
            # half is what the target's extrapolation has to continue.
            # Early origins, where every model misses a delayed jump
            # alike, would drown that signal in shared error.
            cand = [j for j in range(1, len(series) - 1)
                    if last_u - series[j][0] >= 12.0
                    and series[j][0] >= last_u - 36.0]
            if len(cand) < 2:
                cand = [j for j in range(1, len(series) - 1)
                        if last_u - series[j][0] >= 12.0]
            if not cand:
                cand = list(range(1, len(series) - 1))
            jobs = [(j, last_u, last_rate) for j in cand]
        else:
            jobs = [(j, series[j + 1][0], series[j + 1][1])
                    for j in range(1, len(series) - 1)]
        if len(jobs) > 6:
            step_i = len(jobs) / 6.0
            jobs = [jobs[min(len(jobs) - 1, int(k * step_i))]
                    for k in range(6)]
        for j, u_t, actual_rate in jobs:
            fitted = fit_all(series[: j + 1][-48:])
            if not fitted:
                continue
            actual = logit(actual_rate)
            wsum = sum(w.get(name, 0.0) for name in fitted)
            if wsum <= 0:
                continue
            blend = 0.0
            usable = True
            for name, (fn, _mse) in fitted.items():
                try:
                    blend += w.get(name, 0.0) * fn(u_t)
                except (ValueError, OverflowError):
                    usable = False
                    break
            if not usable:
                continue
            abs_err.append(abs(blend / wsum - actual))
            horizons.append(math.log1p(u_t) - math.log1p(series[j][0]))
    if not abs_err:
        return None, None
    abs_err.sort()
    mid = len(abs_err) // 2
    spread0 = abs_err[mid] if len(abs_err) % 2 else 0.5 * (abs_err[mid - 1] + abs_err[mid])
    horizons.sort()
    bt_scale = horizons[len(horizons) // 2] if horizons else None
    return spread0, bt_scale


# ---------------------------------------------------------- dilution prior

def dilution_median(rates, players):
    """Median rates after one expected refresh of a flat history.

    The player base grows DILUTION_GROWTH-fold; newcomers hold the
    trophy everybody holds (at its current rate) and share
    EXTRA_PER_NEWCOMER further trophies each, in proportion to the old
    rates. Returns a list of percents aligned with `rates`.
    """
    known = [(i, r) for i, r in enumerate(rates) if r is not None]
    if not known:
        return [None] * len(rates)
    top_i, top_r = max(known, key=lambda kv: kv[1])
    others_total = sum(r for i, r in known if i != top_i)
    n_old = float(players) if players else 1.0
    n_new = n_old * (DILUTION_GROWTH - 1.0)
    n_tot = n_old * DILUTION_GROWTH
    out = []
    for i, r in enumerate(rates):
        if r is None:
            out.append(None)
            continue
        old_earners = r / 100.0 * n_old
        if i == top_i:
            new_earners = (top_r / 100.0) * n_new
        elif others_total > 0:
            new_earners = EXTRA_PER_NEWCOMER * n_new * r / others_total
        else:
            new_earners = 0.0
        out.append((old_earners + new_earners) / n_tot * 100.0)
    return out


# ---------------------------------------------------------------- forecast

def forecast_case(case):
    trophies = case["trophies"]
    points = refresh_points(case)
    horizon_min = float(case.get("horizon_minutes") or 0)
    horizon_days = horizon_min / 1440.0
    target_u = None
    if points:
        try:
            target_u = (parse_time(case["target"]).timestamp() - points[0][0]) / 3600.0
        except (KeyError, TypeError, ValueError):
            target_u = horizon_min / 60.0
    series_by_trophy = [trophy_series(points, i) for i in range(len(trophies))]

    # Flat history (or none): the dilution prior is the forecast.
    flat = len(points) <= 1
    medians = [None] * len(trophies)
    spreads = [None] * len(trophies)
    spread = FLAT_SPREAD * math.sqrt(1.0 + math.log1p(horizon_days))
    if flat and points:
        last_rates = list(points[-1][2])
        diluted = dilution_median(last_rates, points[-1][1])
        for i, value in enumerate(diluted):
            medians[i] = value if value is not None else TIER_PRIOR.get(
                trophies[i].get("tier"), 3.0)
    elif not points:
        for i, trophy in enumerate(trophies):
            medians[i] = TIER_PRIOR.get(trophy.get("tier"), 3.0)
        spread = SPREAD_CAP
    else:
        fits_by_trophy = [fit_all(series[-48:])
                          for series in series_by_trophy]
        pooled, combined = weights_from_fits(fits_by_trophy)
        spread0, bt_scale = rolling_origin(series_by_trophy, combined)
        if spread0 is None:
            spread0 = NO_ORIGIN_SPREAD
        # Log-distance from the last point to the target, in units of
        # the distance the backtest origins forecast over, grows the
        # spread like a random walk: the origins already measured how
        # far this ensemble misses when it extrapolates that far.
        if bt_scale is None or bt_scale <= 0:
            gaps = []
            for series in series_by_trophy:
                prev = None
                for u, _ in series:
                    x = math.log1p(u)
                    if prev is not None and x - prev > 1e-9:
                        gaps.append(x - prev)
                    prev = x
            gaps.sort()
            bt_scale = gaps[len(gaps) // 2] if gaps else 0.1
        step = max(bt_scale, 0.05)
        last_u = max((u for series in series_by_trophy for u, _ in series),
                     default=0.0)
        dx = max(0.0, math.log1p(target_u) - math.log1p(last_u))
        spread = spread0 * math.sqrt(1.0 + dx / step)
        base_spread = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))
        for i, series in enumerate(series_by_trophy):
            if not series:
                medians[i] = TIER_PRIOR.get(trophies[i].get("tier"), 3.0)
                spreads[i] = base_spread
                continue
            fitted = fits_by_trophy[i]
            if not fitted:
                medians[i] = series[-1][1]
                spreads[i] = base_spread
                continue
            w = combined[i]
            preds = {}
            for name, (fn, _mse) in fitted.items():
                try:
                    preds[name] = fn(target_u)
                except (ValueError, OverflowError):
                    continue
            pw = sum(w.get(n, 0.0) for n in preds)
            if not preds or pw <= 0:
                medians[i] = series[-1][1]
                spreads[i] = base_spread
                continue
            # Saturation guard: when both ceiling models agree the
            # trophy is nearly done rising (their target within 30%
            # of the last rate), the unbounded log-linear fit must
            # not run away with the median -- it has no ceiling to
            # stop at, and a saturating trophy is exactly where it
            # overshoots. Cap it just above the best ceiling call.
            ceil_rates = [inv_logit(preds[n]) for n in
                          ("loglogistic", "weibull") if n in preds]
            if ceil_rates and "loglinear" in preds:
                top_rate = max(ceil_rates)
                if top_rate <= 1.3 * max(series[-1][1], 0.01):
                    preds["loglinear"] = min(
                        preds["loglinear"], logit(top_rate) + 0.35)
            blend = sum(w.get(n, 0.0) * v
                        for n, v in preds.items()) / pw
            medians[i] = inv_logit(blend)
            # Where the models genuinely disagree at the target, the
            # honest spread is wider than the backtest residual alone.
            var = sum(w.get(n, 0.0) * (v - blend) ** 2
                      for n, v in preds.items()) / pw
            spreads[i] = min(SPREAD_CAP,
                             math.sqrt(base_spread ** 2 + var))

    rows = []
    for i, trophy in enumerate(trophies):
        median = medians[i]
        if median is None:
            median = TIER_PRIOR.get(trophy.get("tier"), 3.0)
        spread_i = spreads[i]
        if spread_i is None:
            spread_i = min(SPREAD_CAP, max(SPREAD_FLOOR, spread))
        centre = logit(median)
        qs = []
        prev = None
        for z in Z:
            q = round4(clamp_pct(inv_logit(centre + z * spread_i)))
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        rows.append({"id": trophy["id"], "q": qs})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        case = json.loads(path.read_text(encoding="utf-8"))
        forecast = forecast_case(case)
        out = OUTPUT_DIR / f"{path.stem}.out"
        out.write_text(json.dumps(forecast, separators=(",", ":")) + "\n",
                       encoding="utf-8")


if __name__ == "__main__":
    main()
