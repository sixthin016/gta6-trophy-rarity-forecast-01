#!/usr/bin/env python3
"""GTA VI trophy rarity forecast -- v2 (fresh rewrite).

Forecasts every trophy's Exophase earn rate at the case target as five
quantiles, scored by pinball loss on the logit scale.

Method (from the design notes, written from scratch in this session):
  * Collapse the history to *refresh points*: consecutive reads that are
    byte-identical in rates and players-tracked carry no new information
    (Exophase refreshes its counts hours apart, not continuously).
  * Per trophy, fit three curves in logit space on the refresh points:
    log-linear (logit rate linear in log time), log-logistic (a ceiling
    fit: logit(rate / ceiling) linear in log time, ceiling chosen by
    grid search), and Weibull (rate = C * (1 - exp(-(t/lam)^k)) fitted
    by a two-stage grid: coarse (k, lam) grid, then two rounds of local
    refinement, lam steps x1.15, k +-0.15). All three fits weight
    recent refresh points more (linear recency weights): the current
    regime is what a far-horizon extrapolation depends on.
  * Ensemble weights come from a rolling-origin backtest inside the
    history itself (fit on early refresh points, predict the next one).
  * Spread comes from the median residual of the weighted ensemble,
    growing log-wise with the horizon, capped at 0.9 logit.
  * Flat history (a single refresh point) falls back to a dilution
    prior: the tracked-player count is assumed to multiply by the
    median jump seen in the history itself, else 5x, and the new
    players are assumed to hold almost no trophies except the very
    earliest few.
  * A joint hierarchical binomial model (per-trophy intercepts shrunk
    to PAVA-ordered tier means, an RW2-smoothed common time trend,
    a shrunk dilution coefficient) is fitted in full, but it is only
    blended in when its expanding-window cross-validation beats the
    weighted curve ensemble by at least 10% (blend weight at most
    0.30); otherwise its weight is exactly 0.

Stdlib only, deterministic, no network.
"""

import glob
import json
import math
import os
import sys
from datetime import datetime, timezone

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
FLOOR = 0.01
CEIL = 99.99
TIER_ORDER = ["platinum", "gold", "silver", "bronze"]
# Tier prior rates (percent) used only when a trophy has no usable rate
# anywhere in the history.
TIER_PRIOR = {"platinum": 0.5, "gold": 1.0, "silver": 2.0, "bronze": 3.0}


# ---------------------------------------------------------------- helpers

def clamp_pct(x):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return None
    return min(CEIL, max(FLOOR, float(x)))


def logit_pct(p):
    p = clamp_pct(p)
    q = p / 100.0
    return math.log(q / (1.0 - q))


def inv_logit_pct(y):
    if y > 30.0:
        return CEIL
    if y < -30.0:
        return FLOOR
    return 100.0 / (1.0 + math.exp(-y))


def parse_time(s):
    if s is None:
        return None
    if isinstance(s, (int, float)):
        return datetime.fromtimestamp(s, tz=timezone.utc)
    txt = str(s).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    try:
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def ols(xs, ys):
    """Least squares y ~ a + b*x. Returns (a, b) or None."""
    n = len(xs)
    if n < 2:
        return None
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def wols(xs, ys, ws):
    """Weighted least squares y ~ a + b*x. Returns (a, b) or None."""
    sw = sum(ws)
    if sw <= 0 or len(xs) < 2:
        return None
    mx = sum(w * x for w, x in zip(ws, xs)) / sw
    my = sum(w * y for w, y in zip(ws, ys)) / sw
    sxx = sum(w * (x - mx) ** 2 for w, x in zip(ws, xs))
    if sxx <= 1e-12:
        return (my, 0.0)
    sxy = sum(w * (x - mx) * (y - my) for w, x, y in zip(ws, xs, ys))
    b = sxy / sxx
    return (my - b * mx, b)


def recency_weights(pts):
    """Linear recency weights: the most recent refresh point counts
    most. Recent reads carry the current regime, which is what an
    extrapolation to a far target actually depends on."""
    t_last = pts[-1][0]
    if t_last <= 0:
        return [1.0] * len(pts)
    return [t / t_last for t, _ in pts]


def median(vals):
    v = sorted(vals)
    n = len(v)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return v[n // 2]
    return 0.5 * (v[n // 2 - 1] + v[n // 2])


# ------------------------------------------------------------ refresh points

def refresh_points(case):
    """Collapse history to distinct states.

    Returns (points, t_target, horizon_days, n_first) where points is a
    list of dicts {t, n, rates} with t = 1 + hours since the first read,
    and rates the raw percents (None where missing).
    """
    hist = case.get("history") or []
    target = parse_time(case.get("target"))
    horizon_min = case.get("horizon_minutes") or 0
    points = []
    t0 = None
    for read in hist:
        at = parse_time(read.get("at") or read.get("slot"))
        if at is None:
            continue
        if t0 is None:
            t0 = at
        rates = read.get("rates") or []
        n = read.get("players_tracked")
        try:
            n = float(n) if n is not None else None
        except (TypeError, ValueError):
            n = None
        if points:
            prev = points[-1]
            if prev["n"] == n and prev["rates"] == list(rates):
                continue
        t = 1.0 + (at - t0).total_seconds() / 3600.0
        points.append({"t": t, "n": n, "rates": list(rates)})
    if target is not None and t0 is not None:
        t_target = 1.0 + (target - t0).total_seconds() / 3600.0
    else:
        t_target = 1.0 + horizon_min / 60.0
    horizon_days = max(horizon_min, 0.0) / 1440.0
    n_first = points[0]["n"] if points else None
    return points, t_target, horizon_days, n_first


def trophy_series(points, idx):
    """(t, rate) pairs for one trophy over the refresh points."""
    out = []
    for p in points:
        rates = p["rates"]
        if idx < len(rates):
            r = rates[idx]
            if r is not None:
                try:
                    r = float(r)
                except (TypeError, ValueError):
                    continue
                if 0.0 <= r <= 100.0:
                    out.append((p["t"], r))
    return out


def ceiling_candidates(last_rate):
    cands = []
    for m in (1.15, 1.5, 2.2, 3.5, 7.0, 14.0):
        c = last_rate * m
        if c < CEIL:
            cands.append(c)
    cands.append(CEIL)
    out = []
    for c in cands:
        if c > last_rate * 1.02 and all(abs(c - d) > 1e-9 for d in out):
            out.append(c)
    return out or [CEIL]


# ------------------------------------------------------------ curve models
# Each fit function takes pts = [(t, rate)] and returns a predictor
# t -> logit at that time. All are deterministic.

def fit_loglin(pts):
    """Returns a predictor t -> logit, or None."""
    if not pts:
        return None
    last = logit_pct(pts[-1][1])
    if len(pts) < 2:
        return lambda t: last
    xs = [math.log(t) for t, _ in pts]
    ys = [logit_pct(r) for _, r in pts]
    fit = wols(xs, ys, recency_weights(pts))
    if fit is None:
        return lambda t: last
    a, b = fit
    b = max(-1.5, min(4.0, b))

    def pred(t):
        return a + b * math.log(max(t, 1.0))
    return pred


def _loglog_fit(pts, ceiling):
    """Weighted OLS fit of logit(rate/C) on log(t); returns
    (weighted sse, a, b) or None."""
    xs, ys = [], []
    for t, r in pts:
        if r >= ceiling * 0.999:
            return None
        p = min(max(r / ceiling, 1e-4), 0.9999)
        ys.append(math.log(p / (1.0 - p)))
        xs.append(math.log(t))
    ws = recency_weights(pts)
    fit = wols(xs, ys, ws)
    if fit is None:
        return None
    a, b = fit
    b = max(-1.5, min(6.0, b))
    sse = 0.0
    for (t, r), w in zip(pts, ws):
        pred_p = ceiling / (1.0 + math.exp(-(a + b * math.log(t))))
        sse += w * (logit_pct(pred_p) - logit_pct(r)) ** 2
    return (sse, a, b)


def fit_loglogistic(pts):
    if not pts:
        return None
    last = logit_pct(pts[-1][1])
    if len(pts) < 2:
        return lambda t: last
    best = None
    for c in ceiling_candidates(pts[-1][1]):
        res = _loglog_fit(pts, c)
        if res is None:
            continue
        sse, a, b = res
        if best is None or sse < best[0]:
            best = (sse, c, a, b)
    if best is None:
        return fit_loglin(pts)
    _, c, a, b = best

    def pred(t):
        p = c / (1.0 + math.exp(-(a + b * math.log(max(t, 1.0)))))
        return logit_pct(p)
    return pred


def fit_weibull(pts):
    """Two-stage grid: a coarse (k, lam) grid with the ceiling fixed,
    then two rounds of local refinement (lam steps x1.15, k +-0.15)
    over (k, lam, ceiling) jointly. The coarse lam grid uses x1.5
    steps so the refinement window always covers the coarse winner's
    neighbourhood."""
    if not pts:
        return None
    last = logit_pct(pts[-1][1])
    if len(pts) < 2:
        return lambda t: last
    last_rate = pts[-1][1]
    ys = [logit_pct(r) for _, r in pts]
    lts = [math.log(t) for t, _ in pts]
    ws = recency_weights(pts)

    def sse(c, k, lam):
        log_lam = math.log(lam)
        tot = 0.0
        for lt, y, w in zip(lts, ys, ws):
            try:
                frac = 1.0 - math.exp(-math.exp(k * (lt - log_lam)))
            except OverflowError:
                frac = 1.0
            p = c * frac
            tot += w * (logit_pct(p) - y) ** 2
        return tot

    ks = (0.4, 0.6, 0.9, 1.3, 1.9, 2.8)
    lams = []
    lam = 4.0
    while lam <= 1200.0:
        lams.append(lam)
        lam *= 1.5
    c0 = min(CEIL, last_rate * 2.2)
    best = None  # (sse, ceiling, k, lam)
    for k in ks:
        for lm in lams:
            v = sse(c0, k, lm)
            if best is None or v < best[0]:
                best = (v, c0, k, lm)
    ceilings = ceiling_candidates(last_rate)
    for rnd in range(2):
        _, bc, bk, blam = best
        cand_lams = [blam / (1.15 ** 2), blam / 1.15, blam,
                     blam * 1.15, blam * (1.15 ** 2)]
        cand_ks = [bk - 0.15, bk - 0.075, bk, bk + 0.075, bk + 0.15]
        cand_cs = ceilings if rnd == 0 else [bc * 0.85, bc,
                                             min(CEIL, bc * 1.18)]
        for c in cand_cs:
            if c <= last_rate * 1.02:
                continue
            for k in cand_ks:
                if k <= 0.05:
                    continue
                for lm in cand_lams:
                    v = sse(c, k, lm)
                    if v < best[0]:
                        best = (v, c, k, lm)
    _, c, k, lam = best

    def pred(t):
        try:
            frac = 1.0 - math.exp(-((max(t, 1.0) / lam) ** k))
        except OverflowError:
            frac = 1.0
        return logit_pct(c * frac)
    return pred


MODEL_FITS = (("loglin", fit_loglin),
              ("loglog", fit_loglogistic),
              ("weibull", fit_weibull))


def fit_all_models(pts):
    """{name: predictor t -> logit} for every curve model."""
    fallback = logit_pct(pts[-1][1]) if pts else 0.0
    out = {}
    for name, fn in MODEL_FITS:
        try:
            f = fn(pts)
        except Exception:
            f = None
        if f is None:
            f = (lambda v: (lambda t: v))(fallback)
        out[name] = f
    return out


def predict_all(fits, t):
    out = {}
    for name, f in fits.items():
        try:
            v = f(t)
        except Exception:
            v = None
        if v is None or not math.isfinite(v):
            v = 0.0
        out[name] = v
    return out


def rolling_origin_weights(pts):
    """Weights from a rolling-origin backtest inside the history.

    Fit each model on pts[:j] and score it on pts[j] for the last few
    origins j. Weight is inverse mean squared logit error (floored and
    sharpened). Equal weights when there is nothing to backtest on.
    Returns (weights, best_origin_rmse): the RMSE of the best single
    model at the origins, an honest out-of-sample error scale.
    """
    n = len(pts)
    if n < 3:
        return ({name: 1.0 / len(MODEL_FITS) for name, _ in MODEL_FITS},
                None)
    errs = {name: [] for name, _ in MODEL_FITS}
    origins = [j for j in (n - 3, n - 2) if 2 <= j < n]
    if not origins:
        origins = [n - 1]
    for j in origins:
        train = pts[:j]
        t_next, r_next = pts[j]
        y_next = logit_pct(r_next)
        fits = fit_all_models(train)
        preds = predict_all(fits, t_next)
        for name in errs:
            errs[name].append((preds[name] - y_next) ** 2)
    weights = {}
    for name in errs:
        mse = sum(errs[name]) / len(errs[name]) if errs[name] else 1.0
        weights[name] = 1.0 / (max(mse, 0.005) + 0.005)
    tot = sum(weights.values())
    weights = {name: w / tot for name, w in weights.items()}
    # Cap any single weight at 0.7; redistribute the excess evenly.
    for name in weights:
        if weights[name] > 0.7:
            excess = weights[name] - 0.7
            weights[name] = 0.7
            others = [o for o in weights if o != name]
            for o in others:
                weights[o] += excess / len(others)
    best_rmse = min(
        math.sqrt(sum(v) / len(v)) for v in errs.values() if v)
    return weights, best_rmse


def ensemble_predict(fits, t_target, weights=None, equal=False):
    """(weighted median logit, per-model logits, weights)."""
    preds = predict_all(fits, t_target)
    if equal or weights is None:
        w = {name: 1.0 / len(MODEL_FITS) for name, _ in MODEL_FITS}
    else:
        w = weights
    med = sum(w[name] * preds[name] for name, _ in MODEL_FITS)
    return med, preds, w


def ensemble_spread(fits, pts, t_target, weights, horizon_days,
                    pooled_sigma, origin_rmse=None):
    """Spread from the median residual of the weighted ensemble.

    The scale is the larger of the in-sample residual median and the
    honest rolling-origin error, so a flexible fit cannot claim a
    spuriously narrow spread; it grows log-wise with the horizon and
    is capped at 0.9. Model disagreement widens it a little.
    """
    if len(pts) < 2:
        return min(0.9, 0.45 + 0.10 * math.log1p(horizon_days))
    resids = []
    for t, r in pts:
        med, _, _ = ensemble_predict(fits, t, weights)
        resids.append(abs(med - logit_pct(r)))
    sigma = median(resids) if resids else pooled_sigma
    if origin_rmse is not None:
        sigma = max(sigma, 0.6 * origin_rmse)
    if len(pts) < 3:
        sigma = pooled_sigma
    sigma = max(sigma, 0.10)
    _, preds, _ = ensemble_predict(fits, t_target, weights)
    disagree = max(preds.values()) - min(preds.values())
    growth = 1.0 + 0.35 * math.log1p(horizon_days)
    s = math.sqrt((sigma * growth) ** 2 + (0.35 * disagree) ** 2)
    return min(0.9, max(0.10, s))


# ------------------------------------------------------------ dilution prior

def dilution_multiplier(points):
    """Median jump of players-tracked across refresh points, else 5x."""
    jumps = []
    prev_n = None
    for p in points:
        n = p["n"]
        if n is None or n <= 0:
            continue
        if prev_n is not None and prev_n > 0 and n > prev_n:
            jumps.append(n / prev_n)
        prev_n = n
    if jumps:
        return median(jumps)
    return 5.0


def dilution_forecast(case, points, t_target, horizon_days):
    """Flat-history forecast: (median logit, spread) per trophy index.

    New players (the assumed extra (m-1)*N of them) are taken to hold
    almost no trophies: the trophy at 100% stays 100% (everyone holds
    it), the next three trophies by rate keep a 10% new-player earn
    rate, everything else gets none.
    """
    last = points[-1] if points else None
    n0 = last["n"] if last and last.get("n") else 638.0
    mult = dilution_multiplier(points)
    n1 = mult * n0
    trophies = case.get("trophies") or []
    last_rates = last["rates"] if last else []
    rates = []
    for i, tr in enumerate(trophies):
        r = last_rates[i] if i < len(last_rates) else None
        if r is None:
            r = TIER_PRIOR.get(tr.get("tier"), 3.0)
        rates.append(float(r))
    order = sorted(range(len(rates)), key=lambda i: -rates[i])
    earn_new = [0.0] * len(rates)
    seen_early = 0
    for i in order:
        if rates[i] >= 99.99:
            earn_new[i] = 1.0
        elif seen_early < 3:
            earn_new[i] = 0.10 * rates[i] / 100.0
            seen_early += 1
    spread = min(0.9, 0.50 + 0.10 * math.log1p(horizon_days))
    out = {}
    for i, tr in enumerate(trophies):
        e_old = rates[i] / 100.0 * n0
        e_new = earn_new[i] * (n1 - n0)
        p = 100.0 * (e_old + e_new) / n1 if n1 > 0 else rates[i]
        out[i] = (logit_pct(p), spread)
    return out


# ------------------------------------------------------- joint model (gated)

def _pava_increasing(vals, weights):
    """Isotonic regression (non-decreasing) by the pool-adjacent-violators
    algorithm. Returns the fitted values."""
    n = len(vals)
    levels = [[v, w, 1] for v, w in zip(vals, weights)]
    i = 0
    stack = []
    for v, w, c in levels:
        cur = [v, w, c]
        while stack and stack[-1][0] > cur[0]:
            pv, pw, pc = stack.pop()
            tot_w = pw + cur[1]
            cur = [(pv * pw + cur[0] * cur[1]) / tot_w, tot_w,
                   pc + cur[2]]
        stack.append(cur)
    out = []
    for v, w, c in stack:
        out.extend([v] * c)
    return out[:n]


def _solve_linear(a, b):
    """Gaussian elimination with partial pivoting (small dense system)."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            continue
        m[col], m[piv] = m[piv], m[col]
        pv = m[col][col]
        for r in range(n):
            if r != col and m[r][col] != 0.0:
                f = m[r][col] / pv
                for c in range(col, n + 1):
                    m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] if abs(m[i][i]) > 1e-12 else 0.0
            for i in range(n)]


def _rw2_smooth(raw, wts, lam):
    """Smooth a series with a second-difference (RW2) penalty."""
    n = len(raw)
    if n < 3:
        return list(raw)
    a = [[0.0] * n for _ in range(n)]
    b = [0.0] * n
    for i in range(n):
        a[i][i] += wts[i]
        b[i] += wts[i] * raw[i]
    for i in range(n - 2):
        for u, cu in ((i, 1.0), (i + 1, -2.0), (i + 2, 1.0)):
            for v, cv in ((i, 1.0), (i + 1, -2.0), (i + 2, 1.0)):
                a[u][v] += lam * cu * cv
    return _solve_linear(a, b)


def joint_fit(points, trophies, thin=20):
    """Hierarchical binomial joint model.

    earns ~ Binomial(N, p), logit p = alpha[i] + f(t) + delta*log(N/N0).
    alpha[i] shrunk to the PAVA-ordered tier mean (platinum < gold <
    silver < bronze in logit), f(t) RW2-smoothed, delta shrunk to 0.
    Returns a predictor (t_target, n_target) -> {trophy idx: logit},
    or None when the history is too thin.
    """
    pts = points
    if len(pts) > thin:
        step = len(pts) / float(thin)
        keep = [pts[int(i * step)] for i in range(thin)]
        if keep[-1] is not pts[-1]:
            keep.append(pts[-1])
        pts = keep
    if len(pts) < 4:
        return None
    n_tr = len(trophies)
    n0 = pts[0]["n"] or 638.0
    if not n0 or n0 <= 0:
        n0 = 638.0
    # Smoothed empirical logits and binomial weights.
    y = []   # y[i][j]
    w = []
    for i in range(n_tr):
        yi, wi = [], []
        for p in pts:
            rates = p["rates"]
            r = rates[i] if i < len(rates) else None
            n = p["n"] or n0
            if r is None or n <= 0:
                yi.append(None)
                wi.append(0.0)
                continue
            e = min(max(r / 100.0 * n, 0.0), n)
            pc = (e + 0.5) / (n + 1.0)
            yi.append(math.log(pc / (1.0 - pc)))
            wi.append(n * pc * (1.0 - pc))
        y.append(yi)
        w.append(wi)
    if sum(sum(row) for row in w) <= 0:
        return None
    tiers = [tr.get("tier", "bronze") for tr in trophies]
    log_n = []
    for p in pts:
        n = p["n"] or n0
        log_n.append(math.log(max(n, 1.0) / n0))
    times = [p["t"] for p in pts]
    alpha = [0.0] * n_tr
    for i in range(n_tr):
        num = sum(wi * yi for wi, yi in zip(w[i], y[i]) if yi is not None)
        den = sum(wi for wi, yi in zip(w[i], y[i]) if yi is not None)
        alpha[i] = num / den if den > 0 else logit_pct(
            TIER_PRIOR.get(tiers[i], 3.0))
    f = [0.0] * len(pts)
    delta = 0.0
    for _ in range(12):
        # alpha[i]: weighted mean of residuals, shrunk to tier mean.
        raw = []
        for i in range(n_tr):
            num = 0.0
            den = 0.0
            for j in range(len(pts)):
                if y[i][j] is None:
                    continue
                num += w[i][j] * (y[i][j] - f[j] - delta * log_n[j])
                den += w[i][j]
            raw.append(num / den if den > 0 else alpha[i])
        tier_mean = {}
        for tier in TIER_ORDER:
            idxs = [i for i in range(n_tr) if tiers[i] == tier]
            if idxs:
                tier_mean[tier] = sum(raw[i] for i in idxs) / len(idxs)
        present = [t for t in TIER_ORDER if t in tier_mean]
        if len(present) >= 2:
            vals = [tier_mean[t] for t in present]
            wts = [sum(1 for i in range(n_tr) if tiers[i] == t)
                   for t in present]
            fitted = _pava_increasing(vals, wts)
            for t, v in zip(present, fitted):
                tier_mean[t] = v
        for i in range(n_tr):
            tm = tier_mean.get(tiers[i], raw[i])
            den = sum(w[i][j] for j in range(len(pts))
                      if y[i][j] is not None)
            alpha[i] = (den * raw[i] + 60.0 * tm) / (den + 60.0)
        # f[j]: weighted mean of residuals, RW2-smoothed.
        raw_f = []
        w_f = []
        for j in range(len(pts)):
            num = 0.0
            den = 0.0
            for i in range(n_tr):
                if y[i][j] is None:
                    continue
                num += w[i][j] * (y[i][j] - alpha[i] - delta * log_n[j])
                den += w[i][j]
            raw_f.append(num / den if den > 0 else f[j])
            w_f.append(max(den, 1.0))
        f = _rw2_smooth(raw_f, w_f, 2.0)
        # delta: weighted regression of residuals on log(N/N0), shrunk.
        num = 0.0
        den = 0.0
        for i in range(n_tr):
            for j in range(len(pts)):
                if y[i][j] is None:
                    continue
                r = y[i][j] - alpha[i] - f[j]
                num += w[i][j] * log_n[j] * r
                den += w[i][j] * log_n[j] ** 2
        delta = num / (den + 200.0) if den > 0 else 0.0
        delta = max(-1.0, min(1.0, delta))
    # Extrapolation helpers.
    def predict(t_target, n_target):
        # f: extend the last slope, damped by half.
        if len(f) >= 3:
            slope = (f[-1] - f[-3]) / max(times[-1] - times[-3], 1e-9)
        elif len(f) >= 2:
            slope = (f[-1] - f[-2]) / max(times[-1] - times[-2], 1e-9)
        else:
            slope = 0.0
        slope = max(-0.05, min(0.05, slope))
        f_t = f[-1] + 0.5 * slope * max(t_target - times[-1], 0.0)
        ln_n = math.log(max(n_target, 1.0) / n0)
        return {i: alpha[i] + f_t + delta * ln_n for i in range(n_tr)}
    return predict


def _extrapolated_n(points, horizon_days):
    """Damped log-linear extrapolation of players-tracked."""
    ns = [(p["t"], p["n"]) for p in points if p["n"] and p["n"] > 0]
    if len(ns) < 2:
        return (points[-1]["n"] or 638.0) if points else 638.0
    xs = [t for t, _ in ns]
    ys = [math.log(n) for _, n in ns]
    fit = ols(xs, ys)
    if fit is None:
        return ns[-1][1]
    a, b = fit
    b = max(0.0, min(0.05, b))
    t_last = ns[-1][0]
    horizon_h = horizon_days * 24.0
    return math.exp(a + b * (t_last + 0.5 * horizon_h))


def _cv_loss_ensemble(points, trophies, cut):
    """Median-only logit MAE of the weighted ensemble trained on the
    points before `cut`, predicting the first two points at/after it."""
    train = [p for p in points if p["t"] < cut]
    test = [p for p in points if p["t"] >= cut]
    if len(train) < 3 or not test:
        return None
    errs = []
    for tp in test[:2]:
        for i in range(len(trophies)):
            pts = trophy_series(train, i)
            if len(pts) < 2:
                continue
            rates = tp["rates"]
            if i >= len(rates) or rates[i] is None:
                continue
            w, _ = rolling_origin_weights(pts)
            fits = fit_all_models(pts)
            med, _, _ = ensemble_predict(fits, tp["t"], w)
            errs.append(abs(med - logit_pct(rates[i])))
    return sum(errs) / len(errs) if errs else None


def _cv_loss_joint(points, trophies, cut):
    train = [p for p in points if p["t"] < cut]
    test = [p for p in points if p["t"] >= cut]
    if len(train) < 4 or not test:
        return None
    pred_fn = joint_fit(train, trophies)
    if pred_fn is None:
        return None
    errs = []
    for tp in test[:2]:
        n_t = tp["n"] or (train[-1]["n"] or 638.0)
        preds = pred_fn(tp["t"], n_t)
        for i in range(len(trophies)):
            rates = tp["rates"]
            if i >= len(rates) or rates[i] is None:
                continue
            errs.append(abs(preds[i] - logit_pct(rates[i])))
    return sum(errs) / len(errs) if errs else None


def joint_gate_weight(points, trophies, horizon_days):
    """0.30 if the joint model's expanding-window CV beats the weighted
    curve ensemble by >=10%, else exactly 0.0."""
    if len(points) < 6:
        return 0.0
    ts = [p["t"] for p in points]
    span = ts[-1] - ts[0]
    if span <= 0:
        return 0.0
    cut = ts[0] + 0.65 * span
    e = _cv_loss_ensemble(points, trophies, cut)
    j = _cv_loss_joint(points, trophies, cut)
    if e is None or j is None or e <= 0:
        return 0.0
    if j < 0.90 * e:
        return 0.30
    return 0.0


# ---------------------------------------------------------------- forecast

def forecast_case(case, joint_weight_override=None, equal_weights=False):
    """Returns the list of {"id": n, "q": [q05..q95]} rows for one case.

    joint_weight_override: None = use the CV gate; a float forces that
    blend weight (test/ablation use only). equal_weights: use the
    equal-weight curve ensemble instead of the backtest weights.
    """
    trophies = case.get("trophies") or []
    points, t_target, horizon_days, _ = refresh_points(case)
    rows = []
    if not points:
        # No history at all: tier priors with a wide spread.
        for tr in trophies:
            m = logit_pct(TIER_PRIOR.get(tr.get("tier"), 3.0))
            rows.append((tr.get("id"), m, 0.9))
    elif len(points) == 1:
        prior = dilution_forecast(case, points, t_target, horizon_days)
        for i, tr in enumerate(trophies):
            m, s = prior[i]
            rows.append((tr.get("id"), m, s))
    else:
        # Thin very long histories for the curve fits (18 refresh
        # points already pin down a 3-parameter curve).
        fit_points = points
        if len(points) > 18:
            step = len(points) / 18.0
            fit_points = [points[int(k * step)] for k in range(18)]
            if fit_points[-1] is not points[-1]:
                fit_points.append(points[-1])
        # Pooled residual scale across trophies (for thin series).
        pooled_res = []
        pooled_origin = []
        per_trophy = {}
        for i, tr in enumerate(trophies):
            pts = trophy_series(fit_points, i)
            if len(pts) < 2:
                continue
            fits = fit_all_models(pts)
            w, origin_rmse = rolling_origin_weights(pts)
            for t, r in pts:
                med, _, _ = ensemble_predict(fits, t, w)
                pooled_res.append(abs(med - logit_pct(r)))
            if origin_rmse is not None:
                pooled_origin.append(origin_rmse)
            per_trophy[i] = (pts, fits, w, origin_rmse)
        pooled_sigma = median(pooled_res) if pooled_res else 0.15
        if pooled_origin:
            pooled_sigma = max(pooled_sigma,
                               0.6 * median(pooled_origin))
        # Joint model, gated by expanding-window CV.
        if joint_weight_override is not None:
            w_joint = joint_weight_override
        else:
            w_joint = joint_gate_weight(points, trophies, horizon_days)
        joint_pred = None
        if w_joint and w_joint > 0.0:
            fn = joint_fit(points, trophies)
            if fn is not None:
                n_t = _extrapolated_n(points, horizon_days)
                joint_pred = fn(t_target, n_t)
            else:
                w_joint = 0.0
        for i, tr in enumerate(trophies):
            if i in per_trophy:
                pts, fits, w, origin_rmse = per_trophy[i]
                if equal_weights:
                    med, _, _ = ensemble_predict(fits, t_target, None,
                                                  equal=True)
                else:
                    med, _, _ = ensemble_predict(fits, t_target, w)
                s = ensemble_spread(fits, pts, t_target, w, horizon_days,
                                    pooled_sigma, origin_rmse)
            else:
                # No usable series: tier prior.
                med = logit_pct(TIER_PRIOR.get(tr.get("tier"), 3.0))
                s = 0.9
            if joint_pred is not None:
                med = (1.0 - w_joint) * med + w_joint * joint_pred[i]
            rows.append((tr.get("id"), med, s))
    out = []
    for tid, m, s in rows:
        qs = []
        prev = None
        for z in Z:
            q = clamp_pct(inv_logit_pct(m + z * s))
            q = round(q, 4)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        out.append({"id": tid, "q": qs})
    return out


def persistence_rows(case):
    """Fallback forecast: last rate, house-baseline random-walk spread."""
    trophies = case.get("trophies") or []
    points, _, horizon_days, _ = refresh_points(case)
    last = points[-1] if points else None
    d = max(horizon_days, 0.0)
    s = 0.5 * math.sqrt(d) if d > 0 else 0.1
    out = []
    for i, tr in enumerate(trophies):
        r = None
        if last and i < len(last["rates"]):
            r = last["rates"][i]
        if r is None:
            r = TIER_PRIOR.get(tr.get("tier"), 3.0)
        m = logit_pct(r)
        qs = []
        prev = None
        for z in Z:
            q = round(clamp_pct(inv_logit_pct(m + z * s)), 4)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        out.append({"id": tr.get("id"), "q": qs})
    return out


def main(argv=None):
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output")
    os.makedirs(out_dir, exist_ok=True)
    paths = sorted(glob.glob(os.path.join(in_dir, "*.in")))
    for path in paths:
        name = os.path.splitext(os.path.basename(path))[0]
        out_path = os.path.join(out_dir, name + ".out")
        try:
            with open(path, "r", encoding="utf-8") as fh:
                case = json.load(fh)
            rows = forecast_case(case)
        except Exception:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    case = json.load(fh)
                rows = persistence_rows(case)
            except Exception:
                rows = []
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump({"trophies": rows}, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main())
