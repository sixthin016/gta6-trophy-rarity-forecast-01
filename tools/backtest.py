"""Local validation harness for the trophy forecaster (NOT part of the entry).

Everything runs offline against known or constructed truths, because the
kit ships no scorer:

1. Practice-proxy: competition.yaml publishes the practice answers'
   aggregates (refresh 638 -> 3,868 players tracked, 943 -> 4,276
   trophies earned, every rate but the 100% one fell). Score this
   forecaster and the persistence baseline against four per-trophy truths
   that are all consistent with those aggregates. The forecaster must
   beat persistence (published loss 1.51) under every variant, not one.

2. Synthetic final: generate launch histories with refresh-step dynamics
   (reads repeat the last refresh for hours, then jump), per-trophy
   adoption curves and a growing tracked count; cut at 59.5 h (the real
   deadline), target at 168 h (Nov 26). Score this forecaster against
   persistence and the log-time-linear curve baseline over several random
   worlds, in three adoption families (Weibull, log-logistic, delayed
   sigmoid). This is the closest available stand-in for the final, whose
   true history is not recorded yet.

3. Edge cases: degenerate inputs must still produce a valid output.
   Determinism: two runs on the same case write identical bytes.

Usage: python3 tools/backtest.py
"""

import copy
import json
import math
import random
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as M

KIT = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
           "gta6-trophy-rarity-forecast-01-kit/input")


def score(out, truth):
    """Mean pinball loss of an output dict against {trophy_id: rate}."""
    rows = {r["id"]: r["q"] for r in out["trophies"]}
    return sum(M.pinball(truth[i], rows[i]) for i in truth) / len(truth)


def persistence(case):
    """The starter: median = last rate, sigma = 0.5*sqrt(horizon_days)."""
    last = case["history"][-1]
    spread = 0.5 * math.sqrt(case["horizon_minutes"] / 1440.0)
    rows = []
    for i, t in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = M.to_logit(rate if rate is not None else 0)
        qs = [M.round4(min(99.99, max(0.01, M.from_logit(lg + z * spread))))
              for z in M.ZS]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def curve_baseline(case):
    """The fitted-curve baseline: log-time linear on the last 24 h of
    refresh points, spread = random walk + fit residual combined."""
    points = M.refresh_points(case["history"])
    last = points[-1]
    d = case["horizon_minutes"] / 1440.0
    target_h = M.hours_since_release(case["target"])
    rows = []
    for i, t in enumerate(case["trophies"]):
        series = M.trophy_series(points, i)
        recent = [s for s in series if series[-1][0] - s[0] <= 24.0]
        med = None
        resid = 0.3
        if len(recent) >= 3:
            med = M.fit_log_time(recent, target_h)
            if med is not None:
                errs = []
                for tt, p in recent:
                    fitted = M.fit_log_time(recent, tt)
                    errs.append(abs(M.to_logit(p) - M.to_logit(fitted or p)))
                resid = sum(errs) / len(errs) if errs else 0.3
        if med is None:
            med = last["rates"][i]
        spread = math.sqrt((0.5 * math.sqrt(d)) ** 2 + resid ** 2)
        qs = [M.round4(min(99.99, max(0.01,
                        M.from_logit(M.to_logit(med) + z * spread))))
              for z in M.ZS]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


# ---------------------------------------------------------------------------
# 1. Practice-proxy check
# ---------------------------------------------------------------------------

def practice_truths():
    """Four plausible per-trophy refreshed truths, all consistent with the
    published aggregates: N 638 -> 3,868, total earned 943 -> 4,276,
    trophy 2 stays 100%. Each variant is rescaled so its total is 4,276.
    """
    case = json.loads((KIT / "p00.in").read_text())
    last = case["history"][-1]["rates"]
    trophies = case["trophies"]
    total_new = 4276
    n_new = 3868

    def rescale(rates):
        fixed = sum(r for r in rates.values() if r >= 99.99)
        rest = sum(r for r in rates.values() if r < 99.99)
        want_rest = total_new / n_new * 100.0 - fixed
        f = want_rest / rest if rest > 0 else 1.0
        return {i: (r if r >= 99.99 else r * f) for i, r in rates.items()}

    variants = {}
    variants["A uniform x0.22"] = rescale(
        {t["id"]: (100.0 if i == 1 else last[i] * 0.22)
         for i, t in enumerate(trophies)})
    fac = {2: 0.45, 1: 0.22, 0: 0.12}
    variants["B earliness-graded"] = rescale(
        {t["id"]: (100.0 if i == 1 else last[i] * fac[M.earliness(t)])
         for i, t in enumerate(trophies)})
    early = {3, 4, 8, 2}
    variants["C early-concentrated"] = rescale(
        {t["id"]: (100.0 if i == 1 else
                   last[i] * (0.8 if t["id"] in early else 0.05))
         for i, t in enumerate(trophies)})
    rng = random.Random(7)
    variants["D lognormal noise"] = rescale(
        {t["id"]: (100.0 if i == 1 else
                   max(0.01, last[i] * 0.22 * math.exp(rng.gauss(0, 0.7))))
         for i, t in enumerate(trophies)})
    return case, variants


def check_practice():
    case, variants = practice_truths()
    mine = M.forecast_case(case)
    base = persistence(case)
    print("PRACTICE-PROXY (published aggregates; persistence loss is 1.51)")
    for name, truth in variants.items():
        lm = score(mine, truth)
        lb = score(base, truth)
        print(f"  {name:24s} mine={lm:.3f}  persistence={lb:.3f}  "
              f"improvement={(lb - lm) / lb * 100:.0f}%")
        assert lm < lb, f"lost to persistence on {name}"
        assert lm < 1.0, f"practice loss too high on {name}: {lm}"
    print("  -> practice check PASSED (beats persistence on all variants)")


# ---------------------------------------------------------------------------
# 2. Synthetic final check
# ---------------------------------------------------------------------------

def _iso(t_h):
    base = datetime(2026, 11, 19, tzinfo=timezone.utc)
    return (base + timedelta(hours=t_h)).strftime("%Y-%m-%dT%H:%M:%SZ")


def synth_world(rng, family="weibull"):
    """One synthetic launch: refresh-step history to the 59.5 h cut,
    truth at 168 h. `family` picks the true adoption shape."""
    trophies = json.loads((KIT / "p00.in").read_text())["trophies"]
    params = {}
    for t in trophies:
        tier = t["tier"]
        if tier == "platinum":
            pmax = rng.uniform(0.3, 2.0)
        elif tier == "gold":
            pmax = rng.uniform(1.0, 8.0)
        elif tier == "silver":
            pmax = rng.uniform(4.0, 25.0)
        else:
            pmax = rng.uniform(5.0, 70.0)
        if family == "weibull":
            params[t["id"]] = (pmax, rng.uniform(24.0, 220.0),
                               rng.uniform(0.7, 1.5))
        elif family == "loglogistic":
            params[t["id"]] = (pmax, rng.uniform(-3.0, 0.5),
                               rng.uniform(0.8, 2.2))
        else:  # delayed sigmoid in log time
            params[t["id"]] = (pmax, rng.uniform(12.0, 120.0),
                               rng.uniform(0.3, 0.9))

    def true_rate(tid, t_h):
        pmax = params[tid][0]
        if t_h <= 0:
            return 0.01
        if family == "weibull":
            _, lam, k = params[tid]
            return pmax * (1.0 - math.exp(-((t_h / lam) ** k)))
        if family == "loglogistic":
            _, a, b = params[tid]
            return pmax / (1.0 + math.exp(-(a + b * math.log1p(t_h))))
        _, t50, s = params[tid]
        return pmax / (1.0 + math.exp(-(math.log(max(1.0, t_h)) -
                                        math.log(t50)) / s))

    def n_true(t_h):
        # Tracked players: fast early growth, saturating (~50k by day 7).
        return 50000.0 * max(0.0, t_h) / (max(0.0, t_h) + 72.0) + 600.0

    refreshes = []
    t = -0.2
    while t < 59.5:
        refreshes.append(t)
        t += rng.uniform(2.0, 7.0)
    refresh_values = {}
    for rt in refreshes:
        n = int(n_true(rt))
        rates = []
        for tr in trophies:
            p = true_rate(tr["id"], rt)
            rates.append(round(max(0.01, p) + rng.gauss(0, 0.005), 2))
        refresh_values[rt] = (n, rates,
                              int(sum(r / 100.0 * n for r in rates)))
    history = []
    slot = -0.17
    ri = 0
    while slot < 59.5:
        while ri + 1 < len(refreshes) and refreshes[ri + 1] <= slot:
            ri += 1
        rt = refreshes[ri]
        n, rates, earned = refresh_values[rt]
        history.append({"slot": _iso(slot), "at": _iso(slot),
                        "players_tracked": n, "trophies_earned": earned,
                        "rates": list(rates)})
        slot += 1.0 / 6.0
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "f00",
        "at": "2026-11-21T11:30:00Z",
        "horizon_minutes": 6510,
        "target": "2026-11-26T00:00:00Z",
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": {"name": "Grand Theft Auto VI", "platform": "PS5",
                 "release_date": "2026-11-19",
                 "source": "https://www.exophase.com/game/"
                           "grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies,
        "history": history,
        "notes": "synthetic",
    }
    truth = {tr["id"]: max(0.01, true_rate(tr["id"], 168.0))
             for tr in trophies}
    return case, truth


def check_synthetic(worlds=10):
    print(f"SYNTHETIC FINAL ({worlds} worlds, cut 59.5 h -> target 168 h)")
    lm_all, lb_all, lc_all = [], [], []
    for w in range(worlds):
        rng = random.Random(1000 + w)
        case, truth = synth_world(rng)
        lm = score(M.forecast_case(case), truth)
        lb = score(persistence(case), truth)
        lc = score(curve_baseline(case), truth)
        lm_all.append(lm)
        lb_all.append(lb)
        lc_all.append(lc)
        print(f"  world {w}: mine={lm:.3f} persistence={lb:.3f} "
              f"curve={lc:.3f}")
    m = lambda v: sum(v) / len(v)
    print(f"  MEAN mine={m(lm_all):.3f} persistence={m(lb_all):.3f} "
          f"curve={m(lc_all):.3f}")
    print(f"  mine beats persistence in "
          f"{sum(a < b for a, b in zip(lm_all, lb_all))}/{worlds} worlds; "
          f"beats curve in {sum(a < c for a, c in zip(lm_all, lc_all))}/{worlds}")
    assert m(lm_all) < m(lb_all), "mine does not beat persistence on average"
    assert m(lm_all) < m(lc_all), "mine does not beat curve baseline on average"
    print("  -> synthetic check PASSED")


def check_other_families(worlds=5):
    """Same final, but the true adoption shape is NOT Weibull: the
    ensemble must win on shapes it was not tuned to."""
    for family in ("loglogistic", "delayed"):
        lm_all, lc_all, lb_all = [], [], []
        for w in range(worlds):
            rng = random.Random(2000 + w)
            case, truth = synth_world(rng, family=family)
            lm_all.append(score(M.forecast_case(case), truth))
            lc_all.append(score(curve_baseline(case), truth))
            lb_all.append(score(persistence(case), truth))
        m = lambda v: sum(v) / len(v)
        print(f"  {family:12s} mine={m(lm_all):.3f} persistence={m(lb_all):.3f} "
              f"curve={m(lc_all):.3f}")
        assert m(lm_all) < m(lc_all), f"mine loses to curve on {family}"
        assert m(lm_all) < m(lb_all), f"mine loses to persistence on {family}"
    print("  -> other-families check PASSED")


# ---------------------------------------------------------------------------
# 3. Edge cases + determinism
# ---------------------------------------------------------------------------

def check_edge_cases():
    base = json.loads((KIT / "p00.in").read_text())

    def run(case, label):
        out = M.forecast_case(copy.deepcopy(case))
        assert set(out.keys()) == {"trophies"} and len(out["trophies"]) == 55
        for row in out["trophies"]:
            q = row["q"]
            assert len(q) == 5 and all(0 <= v <= 100 for v in q)
            assert all(q[i] <= q[i + 1] for i in range(4))
        print(f"  {label}: valid")

    c = copy.deepcopy(base)
    c["history"] = c["history"][:1]
    run(c, "single read")
    c = copy.deepcopy(base)
    c["history"] = []
    run(c, "empty history")
    c = copy.deepcopy(base)
    for r in c["history"]:
        r["rates"] = [None if i % 3 == 0 else v
                      for i, v in enumerate(r["rates"])]
    run(c, "a third of rates null")
    c = copy.deepcopy(base)
    for r in c["history"]:
        r["rates"] = [0.01 if v is not None else None for v in r["rates"]]
    run(c, "all rates at the floor")
    print("  -> edge-case check PASSED")


def check_determinism_and_runtime():
    """Two runs on the same case must write identical bytes, fast."""
    case = json.loads((KIT / "p00.in").read_text())
    t0 = time.time()
    out1 = M.forecast_case(copy.deepcopy(case))
    t1 = time.time()
    out2 = M.forecast_case(copy.deepcopy(case))
    t2 = time.time()
    assert json.dumps(out1) == json.dumps(out2), "non-deterministic output"
    size = len(json.dumps(out1, separators=(",", ":")))
    print(f"  practice case: {t1 - t0:.2f}s then {t2 - t1:.2f}s, "
          f"byte-identical, output {size} bytes (< 65536)")
    rng = random.Random(1000)
    case, _ = synth_world(rng)
    t0 = time.time()
    out = M.forecast_case(case)
    t1 = time.time()
    size = len(json.dumps(out, separators=(",", ":")))
    print(f"  synthetic final: {t1 - t0:.2f}s, output {size} bytes")
    assert t1 - t0 < 8.0, "too slow for the 8 s cap"
    print("  -> determinism + runtime check PASSED")


if __name__ == "__main__":
    check_practice()
    check_synthetic()
    check_other_families()
    check_edge_cases()
    check_determinism_and_runtime()
    print("ALL CHECKS PASSED")
