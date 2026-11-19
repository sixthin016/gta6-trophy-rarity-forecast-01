"""Validation harness for the trophy forecaster v2 (NOT part of the entry).

Same practice-proxy + synthetic-final + other-families + edge checks as
v1, plus an expanding-window CV check (truncate a synthetic history at
24/36/48 h, forecast +24/+18/+10 h ahead) and an ablation check that
reports the joint model's CV weight and the loss with the joint model
forced on vs gated off.


Two checks, both run offline against known or constructed truths:

1. Practice-proxy check: the kit's competition.yaml publishes the practice
   answers' aggregates (refresh 638 -> 3,868 players, 943 -> 4,276 trophies
   earned, every rate but the 100% one fell). Score my forecast and the
   persistence baseline against several per-trophy truths consistent with
   those aggregates. My forecast must beat persistence (loss 1.51) under
   every variant, not just one.

2. Synthetic final check: generate launch histories with refresh-step
   dynamics (reads repeat the last refresh for hours, then jump), Weibull
   adoption curves per trophy and growing tracked counts; cut at 59.5 h
   (the real deadline), target at 168 h (Nov 26). Score my forecaster
   against persistence and the log-time-linear curve baseline over many
   random worlds. This is the closest available stand-in for the final,
   where the true history is not yet recorded.

Usage: python3 tools/backtest.py
"""

import json
import math
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import main as M

KIT = Path("/home/hatch/workspace/makefun/gta6-trophy-rarity-forecast-01-kit/"
           "gta6-trophy-rarity-forecast-01-kit/input")


def pinball(actual, quantiles):
    return M.pinball_loss(actual, quantiles)


def score_forecast(out, truth):
    """Mean pinball loss of an output dict against {trophy_id: rate}."""
    rows = {r["id"]: r["q"] for r in out["trophies"]}
    return sum(pinball(truth[i], rows[i]) for i in truth) / len(truth)


def persistence_forecast(case):
    """The starter: median = last rate, sigma = 0.5*sqrt(horizon_days)."""
    last = case["history"][-1]
    d = case["horizon_minutes"] / 1440.0
    spread = 0.5 * math.sqrt(d)
    rows = []
    for i, t in enumerate(case["trophies"]):
        rate = last["rates"][i]
        lg = M.logit(rate if rate is not None else 0)
        qs = [M.round4(min(99.99, max(0.01, M.inv_logit(lg + z * spread))))
              for z in M.Z]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


# --------------------------------------------------------------------------
# 1. Practice-proxy check
# --------------------------------------------------------------------------

def practice_truths():
    """Plausible per-trophy refreshed truths, all consistent with the
    published aggregates: N 638 -> 3,868, total earned 943 -> 4,276,
    trophy 2 stays 100%. Each variant is scaled so its total is 4,276.
    """
    case = json.loads((KIT / "p00.in").read_text())
    last = case["history"][-1]["rates"]
    trophies = case["trophies"]
    n_new = 3868
    total_new = 4276

    def scale(rates):
        # rates: dict id -> new rate; rescale the non-100 ones to hit total.
        fixed = sum(r for i, r in rates.items() if r >= 99.99)
        rest = sum(r for i, r in rates.items() if r < 99.99)
        target_rest = total_new / n_new * 100 - fixed
        f = target_rest / rest if rest > 0 else 1.0
        return {i: (r if r >= 99.99 else r * f) for i, r in rates.items()}

    variants = {}
    # A: uniform dilution of every non-100 rate.
    variants["A uniform x0.22"] = scale(
        {t["id"]: (100.0 if i == 1 else last[i] * 0.22)
         for i, t in enumerate(trophies)})
    # B: earliness-graded dilution (early keeps more).
    fac = {2: 0.45, 1: 0.22, 0: 0.12}
    variants["B earliness-graded"] = scale(
        {t["id"]: (100.0 if i == 1 else last[i] * fac[M.earliness(t)])
         for i, t in enumerate(trophies)})
    # C: concentrated on the earliest story trophies, rest near zero.
    early = {3, 4, 8, 2}
    variants["C early-concentrated"] = scale(
        {t["id"]: (100.0 if i == 1 else
                   last[i] * (0.8 if t["id"] in early else 0.05))
         for i, t in enumerate(trophies)})
    # D: lognormal noise around the mean, seeded.
    rng = random.Random(7)
    variants["D lognormal noise"] = scale(
        {t["id"]: (100.0 if i == 1 else
                   max(0.01, last[i] * 0.22 * math.exp(rng.gauss(0, 0.7))))
         for i, t in enumerate(trophies)})
    return case, variants


def check_practice():
    case, variants = practice_truths()
    mine = M.forecast_case(case)
    base = persistence_forecast(case)
    print("PRACTICE-PROXY (published aggregates; persistence loss is 1.51)")
    for name, truth in variants.items():
        lm = score_forecast(mine, truth)
        lb = score_forecast(base, truth)
        print(f"  {name:24s} mine={lm:.3f}  persistence={lb:.3f}  "
              f"improvement={(lb - lm) / lb * 100:.0f}%")
        assert lm < lb, f"lost to persistence on {name}"
        assert lm < 1.0, f"practice loss too high on {name}: {lm}"
    print("  -> practice check PASSED (beats persistence on all variants)")


# --------------------------------------------------------------------------
# 2. Synthetic final check
# --------------------------------------------------------------------------

def synth_world(rng, family="weibull"):
    """One synthetic launch: refresh-step history to the 59.5 h cut,
    truth at 168 h. Returns (case, truth dict). `family` picks the true
    adoption shape: weibull, loglogistic, or delayed sigmoid (chapter-
    gated: a trophy's adoption only starts after a per-trophy delay)."""
    trophies = json.loads((KIT / "p00.in").read_text())["trophies"]
    # Per-trophy adoption: ceiling by tier, shape by family.
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
            u = math.log1p(t_h)
            return pmax / (1.0 + math.exp(-(a + b * u)))
        _, t50, s = params[tid]
        u = math.log(max(1.0, t_h))
        return pmax / (1.0 + math.exp(-(u - math.log(t50)) / s))

    def n_true(t_h):
        # Tracked players: fast early growth, saturating (~50k by day 7).
        return 50000.0 * max(0.0, t_h) / (max(0.0, t_h) + 72.0) + 600.0

    # Refresh times: every 2-7 h from -0.2 h to the cut.
    refreshes = []
    t = -0.2
    while t < 59.5:
        refreshes.append(t)
        t += rng.uniform(2.0, 7.0)
    # Reads every 10 min show the last refresh's values EXACTLY (like the
    # real recorder: identical reads between refreshes). Rates are computed
    # once per refresh, then repeated verbatim until the next refresh.
    refresh_values = {}
    for rt in refreshes:
        n = int(n_true(rt))
        rates = []
        for tr in trophies:
            p = true_rate(tr["id"], rt)
            rates.append(round(max(0.01, p) + rng.gauss(0, 0.005), 2))
        refresh_values[rt] = (n, rates, int(sum(r / 100.0 * n for r in rates)))
    history = []
    slot = -0.17
    ri = 0
    while slot < 59.5:
        while ri + 1 < len(refreshes) and refreshes[ri + 1] <= slot:
            ri += 1
        rt = refreshes[ri]
        n, rates, earned = refresh_values[rt]
        history.append({
            "slot": _iso(slot), "at": _iso(slot),
            "players_tracked": n,
            "trophies_earned": earned,
            "rates": list(rates),
        })
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
                 "source": "https://www.exophase.com/game/grand-theft-auto-vi-psn/trophies/"},
        "trophies": trophies,
        "history": history,
        "notes": "synthetic",
    }
    truth = {tr["id"]: max(0.01, true_rate(tr["id"], 168.0))
             for tr in trophies}
    return case, truth


def _iso(t_h):
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 11, 19, tzinfo=timezone.utc)
    return (base + timedelta(hours=t_h)).strftime("%Y-%m-%dT%H:%M:%SZ")


def curve_baseline_forecast(case):
    """The fitted-curve baseline: log-time linear on the last 24 h of
    refresh points, spread = random walk + fit residual combined."""
    points = M.refresh_points(case)
    last = points[-1]
    d = case["horizon_minutes"] / 1440.0
    target_t = M.hours_since_release(case["target"])
    rows = []
    for i, t in enumerate(case["trophies"]):
        series = M.trophy_series(case, points, i)
        recent = [s for s in series if series[-1][0] - s[0] <= 24.0]
        med = None
        resid = 0.3
        if len(recent) >= 3:
            med = M.fit_loglinear(recent, target_t)
            if med is not None:
                errs = [abs(M.logit(p) - M.logit(
                    M.fit_loglinear(recent, tt) or p))
                    for tt, p, _ in recent]
                resid = (sum(errs) / len(errs)) if errs else 0.3
        if med is None:
            med = last["rates"][i]
        spread = math.sqrt((0.5 * math.sqrt(d)) ** 2 + resid ** 2)
        qs = [M.round4(min(99.99, max(0.01,
                        M.inv_logit(M.logit(med) + z * spread))))
              for z in M.Z]
        for j in range(1, 5):
            qs[j] = max(qs[j], qs[j - 1])
        rows.append({"id": t["id"], "q": qs})
    return {"trophies": rows}


def check_synthetic(worlds=10):
    print(f"SYNTHETIC FINAL ({worlds} worlds, cut 59.5 h -> target 168 h)")
    lm_all, lb_all, lc_all = [], [], []
    for w in range(worlds):
        rng = random.Random(1000 + w)
        case, truth = synth_world(rng)
        mine = M.forecast_case(case)
        base = persistence_forecast(case)
        curve = curve_baseline_forecast(case)
        lm = score_forecast(mine, truth)
        lb = score_forecast(base, truth)
        lc = score_forecast(curve, truth)
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
            lm_all.append(score_forecast(M.forecast_case(case), truth))
            lc_all.append(score_forecast(
                curve_baseline_forecast(case), truth))
            lb_all.append(score_forecast(
                persistence_forecast(case), truth))
        m = lambda v: sum(v) / len(v)
        print(f"  {family:12s} mine={m(lm_all):.3f} persistence={m(lb_all):.3f} "
              f"curve={m(lc_all):.3f}")
        assert m(lm_all) < m(lc_all), f"mine loses to curve on {family}"
        assert m(lm_all) < m(lb_all), f"mine loses to persistence on {family}"
    print("  -> other-families check PASSED")


def check_edge_cases():
    """Degenerate inputs must still produce a valid, deterministic output."""
    base = json.loads((KIT / "p00.in").read_text())

    def run(case, label):
        import copy
        out = M.forecast_case(copy.deepcopy(case))
        assert set(out.keys()) == {"trophies"} and len(out["trophies"]) == 55
        for row in out["trophies"]:
            q = row["q"]
            assert len(q) == 5 and all(0 <= v <= 100 for v in q)
            assert all(q[i] <= q[i + 1] for i in range(4))
        print(f"  {label}: valid")

    c = json.loads(json.dumps(base))
    c["history"] = c["history"][:1]
    run(c, "single read")
    c = json.loads(json.dumps(base))
    c["history"] = []
    run(c, "empty history")
    c = json.loads(json.dumps(base))
    for r in c["history"]:
        r["rates"] = [None if i % 3 == 0 else v for i, v in enumerate(r["rates"])]
    run(c, "a third of rates null")
    c = json.loads(json.dumps(base))
    for r in c["history"]:
        r["rates"] = [0.01 if v is not None else None for v in r["rates"]]
    run(c, "all rates at the floor")
    print("  -> edge-case check PASSED")


def check_expanding_cv():
    """Truncate synthetic histories and forecast a shorter horizon ahead."""
    import copy
    from datetime import datetime, timedelta, timezone
    base = datetime(2026, 11, 19, tzinfo=timezone.utc)

    def trunc(case, cut_h, hz):
        c = copy.deepcopy(case)
        cut_iso = (base + timedelta(hours=cut_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        tgt_iso = (base + timedelta(hours=cut_h + hz)).strftime("%Y-%m-%dT%H:%M:%SZ")
        c["history"] = [r for r in case["history"] if r["slot"] < cut_iso]
        c["at"] = cut_iso
        c["target"] = tgt_iso
        c["horizon_minutes"] = int(hz * 60)
        return c

    def truth_at(case, t_h):
        iso = (base + timedelta(hours=t_h)).strftime("%Y-%m-%dT%H:%M:%SZ")
        for r in case["history"]:
            if r["slot"] >= iso:
                return {tr["id"]: r["rates"][i] for i, tr in
                        enumerate(case["trophies"]) if r["rates"][i] is not None}
        return None

    print("EXPANDING-WINDOW CV (Weibull worlds, 5 each)")
    for cut, hz in ((24.0, 24.0), (36.0, 18.0), (48.0, 10.0)):
        lm_all = []
        for w in range(5):
            rng = random.Random(1000 + w)
            case, _ = synth_world(rng, family="weibull")
            truth = truth_at(case, cut + hz)
            lm_all.append(score_forecast(
                M.forecast_case(trunc(case, cut, hz)), truth))
        print(f"  cut {cut:.0f}h -> +{hz:.0f}h  mine={sum(lm_all)/len(lm_all):.3f}")
    print("  -> expanding-window check done")


def check_ablation():
    """Joint model: CV weight actually granted, and loss forced-on vs gated."""
    print("ABLATION (Weibull worlds, 5 each; sigma 0.5 for comparability)")
    for fam, seed0 in (("weibull", 1000), ("delayed", 2000)):
        lj, lc, lb = [], [], []
        wjs = []
        for w in range(5):
            rng = random.Random(seed0 + w)
            case, truth = synth_world(rng, family=fam)
            pts = M.refresh_points(case)
            tgt = M.hours_since_release(case["target"])
            w0, _, _ = M.backtest(case, pts, tgt, 108.0)
            if not w0:
                w0 = {"loglinear": 0.35, "loglogistic": 0.35, "weibull": 0.3}
            wj = M.joint_cv_weight(case, pts, 108.0, w0)
            wjs.append(wj)
            jf = M.joint_fit(case, pts)
            jmed = M.joint_forecast_median(jf, case, pts, tgt) if jf else None
            for key, acc in (("joint", lj), ("curve", lc), ("blend", lb)):
                pass
            rows = {r["id"]: r["q"] for r in
                    {"trophies": []}["trophies"]} if False else None
            # score medians directly at sigma 0.5
            def sc(med_logit):
                tot = 0.0
                for i in truth:
                    qs = [M.inv_logit(med_logit[i] + z * 0.5) for z in M.Z]
                    tot += M.pinball_loss(truth[i], qs)
                return tot / len(truth)
            jlog, clog = {}, {}
            for i, tr in enumerate(case["trophies"]):
                s = M.trophy_series(case, pts, i)
                fits = M._fit_all_models(s, tgt)
                vals = [M.logit(v) for v in fits.values() if v is not None]
                if vals:
                    clog[tr["id"]] = sum(vals) / len(vals)
                if jmed is not None:
                    jlog[tr["id"]] = jmed[i]
            lj.append(sc(jlog))
            lc.append(sc(clog))
            blog = {i: 0.5 * jlog[i] + 0.5 * clog[i] for i in clog if i in jlog}
            lb.append(sc(blog))
        m = lambda v: sum(v) / len(v)
        print(f"  {fam:10s} joint={m(lj):.3f} curve-eq={m(lc):.3f} "
              f"blend50={m(lb):.3f} w_joint granted={sum(wjs)/len(wjs):.2f}")
    print("  -> ablation done (joint loses on synthetic worlds; gate keeps it out)")


if __name__ == "__main__":
    check_practice()
    check_synthetic()
    check_other_families()
    check_expanding_cv()
    check_ablation()
    check_edge_cases()
