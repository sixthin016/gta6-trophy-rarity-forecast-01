#!/usr/bin/env python3
"""Local validation harness for the GTA VI trophy forecast (NOT part of
the entry). Replicates the pinball-on-logit score from competition.yaml
and checks the entry against:

  * practice-proxy truths: four synthetic answers for the two practice
    cases, all consistent with the public aggregates (Exophase refreshed
    638 -> 3,868 players between 04:00Z and 04:30Z on 2026-11-19 and
    every rate but the 100% one fell):
      A uniform      -- new players hold nothing (except the 100% one)
      B earliness    -- new players hold the earliest trophies at 30%
      C early-conc.  -- new players hold ONLY the top 12 trophies, fully
      D lognormal    -- A with per-trophy lognormal noise (sigma 0.45)
  * synthetic final worlds: histories generated from three curve
    families (Weibull x10, log-logistic x5, delayed-sigmoid x5), scored
    against the entry, the persistence baseline and the curve baseline.
  * expanding-window CV: cut the history at 24h / 36h / 48h and
    forecast +24h / +18h / +10h.
  * ablation: joint-only, equal-weight curves, 50/50 blend, weighted
    curves (the shipped configuration).
  * edge cases (1 read, empty history, null rates, all at floor) and
    determinism (two runs byte-identical).

Usage: python3 tools/backtest.py   (run from the repository root)
"""

import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
import main as entry  # noqa: E402  (the entry under test)

KIT_INPUT = os.environ.get(
    "KIT_INPUT_DIR",
    os.path.join(os.path.expanduser("~"), "workspace", "makefun",
                 "gta6-trophy-rarity-forecast-01-kit",
                 "gta6-trophy-rarity-forecast-01-kit", "input"))

Z = (-1.645, -0.674, 0.0, 0.674, 1.645)
RELEASE = datetime(2026, 11, 19, 0, 0, tzinfo=timezone.utc)
DEADLINE = datetime(2026, 11, 21, 11, 30, tzinfo=timezone.utc)
TARGET = datetime(2026, 11, 26, 0, 0, tzinfo=timezone.utc)


# ---------------------------------------------------------------- scoring

def logit_pct(p):
    p = min(99.99, max(0.01, float(p)))
    q = p / 100.0
    return math.log(q / (1.0 - q))


def pinball_loss(actual, rows):
    """Mean pinball loss over every (trophy, level)."""
    by_id = {r["id"]: r["q"] for r in rows}
    tot, n = 0.0, 0
    for tid, y in actual.items():
        qs = by_id[tid]
        ly = logit_pct(y)
        for tau, q in zip((0.05, 0.25, 0.5, 0.75, 0.95), qs):
            d = ly - logit_pct(q)
            tot += 2.0 * max(tau * d, (tau - 1.0) * d)
            n += 1
    return tot / n if n else float("nan")


def run_entry(cases):
    """Run main.py (subprocess) on {case_id: case} -> ({id: rows}, secs)."""
    with tempfile.TemporaryDirectory() as td:
        in_dir = os.path.join(td, "in")
        out_dir = os.path.join(td, "out")
        os.makedirs(in_dir)
        os.makedirs(out_dir)
        for cid, case in cases.items():
            with open(os.path.join(in_dir, cid + ".in"), "w") as fh:
                json.dump(case, fh)
        env = dict(os.environ)
        env["MAKEFUN_INPUT_DIR"] = in_dir
        env["MAKEFUN_OUTPUT_DIR"] = out_dir
        t0 = time.time()
        proc = subprocess.run(
            [sys.executable, os.path.join(ROOT, "main.py")],
            env=env, capture_output=True, text=True, timeout=120)
        secs = time.time() - t0
        if proc.returncode != 0:
            raise RuntimeError("main.py failed: " + proc.stderr[-2000:])
        out = {}
        for cid in cases:
            with open(os.path.join(out_dir, cid + ".out")) as fh:
                out[cid] = json.load(fh)["trophies"]
        return out, secs


# ---------------------------------------------------------------- baselines

def persistence_rows(case):
    rows = entry.persistence_rows(case)
    return rows


def curve_baseline_rows(case):
    """House curve baseline: per trophy OLS of logit rate on
    ln(1 + hours since release) over the last 24h of reads, read at the
    target; spread = random-walk + residual combined."""
    trophies = case.get("trophies") or []
    points, t_target, horizon_days, _ = entry.refresh_points(case)
    target = entry.parse_time(case.get("target"))
    t0 = entry.parse_time((case.get("history") or [{}])[0].get("at")) \
        if case.get("history") else None
    # Rebuild points on the release clock for the baseline's x-axis.
    hist = case.get("history") or []
    reads = []
    for read in hist:
        at = entry.parse_time(read.get("at") or read.get("slot"))
        if at is None:
            continue
        reads.append((at, read))
    if target is None or not reads:
        return persistence_rows(case)
    cutoff = target - timedelta(hours=24)
    # NOTE: baseline uses reads in the last 24h before the checkpoint.
    checkpoint = entry.parse_time(case.get("at"))
    recent = [(at, r) for at, r in reads if at >= checkpoint - timedelta(hours=24)]
    d = max(horizon_days, 0.0)
    rw = 0.5 * math.sqrt(d)
    out = []
    for i, tr in enumerate(trophies):
        pts = []
        for at, read in recent:
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                h = (at - RELEASE).total_seconds() / 3600.0
                pts.append((math.log1p(max(h, 0.0)), logit_pct(rates[i])))
        last_rate = None
        for at, read in reversed(reads):
            rates = read.get("rates") or []
            if i < len(rates) and rates[i] is not None:
                last_rate = rates[i]
                break
        if last_rate is None:
            last_rate = entry.TIER_PRIOR.get(tr.get("tier"), 3.0)
        if len(pts) < 3:
            m, s = logit_pct(last_rate), rw
        else:
            fit = entry.ols([x for x, _ in pts], [y for _, y in pts])
            a, b = fit
            h_t = (target - RELEASE).total_seconds() / 3600.0
            m = a + b * math.log1p(max(h_t, 0.0))
            resid = [abs(y - (a + b * x)) for x, y in pts]
            s = min(0.9, math.sqrt(rw ** 2 + (entry.median(resid) or 0.0) ** 2))
        qs = []
        prev = None
        for z in Z:
            q = round(entry.clamp_pct(entry.inv_logit_pct(m + z * s)), 4)
            if prev is not None and q < prev:
                q = prev
            qs.append(q)
            prev = q
        out.append({"id": tr.get("id"), "q": qs})
    return out


# ------------------------------------------------------------ practice proxy

def load_practice():
    cases = {}
    for cid in ("p00", "p01"):
        with open(os.path.join(KIT_INPUT, cid + ".in")) as fh:
            cases[cid] = json.load(fh)
    return cases


def practice_truths(case, variant, seed=7):
    """Synthetic answer rates for a practice case, per variant."""
    last = case["history"][-1]
    rates = last["rates"]
    n_old = last["players_tracked"] or 638
    n_new = 3868
    order = sorted(range(len(rates)),
                   key=lambda i: -(rates[i] if rates[i] is not None else 0))
    out = {}
    rng = random.Random(seed)
    for i, r in enumerate(rates):
        if r is None:
            r = 0.0
        if r >= 99.99:
            out[case["trophies"][i]["id"]] = 99.99
            continue
        e_old = r / 100.0 * n_old
        if variant == "A":
            g = 0.0
        elif variant == "B":
            g = 0.30 * r / 100.0 if i in order[:12] else 0.0
        elif variant == "C":
            g = r / 100.0 if i in order[:12] else 0.0
        elif variant == "D":
            g = 0.0
        else:
            raise ValueError(variant)
        y = 100.0 * (e_old + g * (n_new - n_old)) / n_new
        if variant == "D":
            y *= math.exp(rng.gauss(0.0, 0.45))
        out[case["trophies"][i]["id"]] = min(99.99, max(0.01, y))
    return out


# ------------------------------------------------------- synthetic worlds

def _draw_ceiling(rng, tid):
    if tid == 1:
        return rng.uniform(0.3, 1.5)
    if tid in (2,):
        return 100.0
    if rng.random() < 0.04:
        return rng.uniform(60.0, 95.0)
    return math.exp(rng.uniform(math.log(0.4), math.log(40.0)))


def make_world(family, seed):
    """A synthetic final-case world. Returns (case, answer {id: rate})."""
    rng = random.Random(seed)
    practice = load_practice()["p00"]
    trophies = practice["trophies"]
    params = {}
    for tr in trophies:
        tid = tr["id"]
        c = _draw_ceiling(rng, tid)
        if family == "weibull":
            params[tid] = (c, rng.uniform(0.7, 2.2), rng.uniform(15.0, 180.0))
        elif family == "loglog":
            params[tid] = (c, rng.uniform(0.8, 2.4), rng.uniform(8.0, 90.0))
        elif family == "delayed":
            params[tid] = (c, rng.uniform(20.0, 75.0), rng.uniform(8.0, 30.0))
        else:
            raise ValueError(family)

    def curve(tid, t_hours):
        c, a, b = params[tid]
        if tid == 2:
            return 100.0
        te = t_hours + 1.0  # adoption clock starts 1h before release
        if family == "weibull":
            k, lam = a, b
            return c * (1.0 - math.exp(-((max(te, 0.01)) / lam) ** k))
        if family == "loglog":
            k, t0 = a, b
            return c / (1.0 + (t0 / max(te, 0.01)) ** k)
        t0, s = a, b
        return c / (1.0 + math.exp(-(te - t0) / s))

    # Refreshes every 6h from t=0; the shown rate is the curve at the
    # last refresh at or before the read.
    refresh_times = [6.0 * k for k in range(0, 11)]  # 0..60h
    n_at = {}
    n = 638.0
    for rt in refresh_times:
        n_at[rt] = n
        n *= rng.uniform(1.3, 1.9)
    history = []
    slot = RELEASE - timedelta(minutes=10)
    while slot <= DEADLINE:
        t = (slot - RELEASE).total_seconds() / 3600.0
        rt = max([x for x in refresh_times if x <= t], default=None)
        if rt is None:
            rates = [round(curve(tr["id"], -0.17), 2) for tr in trophies]
            nn = 638
        else:
            rates = [round(curve(tr["id"], rt), 2) for tr in trophies]
            nn = int(round(n_at[rt]))
        earned = sum(int(round(r / 100.0 * nn)) for r in rates)
        history.append({
            "slot": slot.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "at": (slot + timedelta(seconds=30)).strftime(
                "%Y-%m-%dT%H:%M:%S.000Z"),
            "players_tracked": nn,
            "trophies_earned": earned,
            "rates": rates,
        })
        slot += timedelta(minutes=30)
    horizon = int((TARGET - DEADLINE).total_seconds() // 60)
    case = {
        "schema": "makefun/gta6-trophies-case/v1",
        "id": "f00",
        "at": DEADLINE.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "horizon_minutes": horizon,
        "target": TARGET.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "levels": [0.05, 0.25, 0.5, 0.75, 0.95],
        "game": practice["game"],
        "trophies": trophies,
        "history": history,
        "notes": "synthetic world family=%s seed=%d" % (family, seed),
    }
    t_ans = (TARGET - RELEASE).total_seconds() / 3600.0
    answer = {}
    for tr in trophies:
        y = curve(tr["id"], t_ans)
        answer[tr["id"]] = min(99.99, max(0.01, round(y, 2)))
    return case, answer


def truncate_case(case, cut_dt, target_dt):
    """History truncated to reads before cut_dt, forecasting target_dt."""
    hist = []
    for read in case["history"]:
        at = entry.parse_time(read.get("at") or read.get("slot"))
        if at is not None and at < cut_dt:
            hist.append(read)
    new = dict(case)
    new["history"] = hist
    new["at"] = cut_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    new["target"] = target_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    new["horizon_minutes"] = int((target_dt - cut_dt).total_seconds() // 60)
    return new


# ------------------------------------------------------------------- checks

def check_practice():
    cases = load_practice()
    rows, secs = run_entry(cases)
    print("practice: entry runtime %.2fs for 2 cases" % secs)
    means = {}
    for variant in ("A", "B", "C", "D"):
        losses = []
        for cid, case in cases.items():
            truth = practice_truths(case, variant)
            losses.append(pinball_loss(truth, rows[cid]))
        means[variant] = sum(losses) / len(losses)
        print("  practice-proxy %s: %.3f" % (variant, means[variant]))
    mean = sum(means.values()) / 4.0
    print("  practice-proxy mean: %.3f" % mean)
    return means, mean


def check_synthetic():
    results = {}
    worst_secs = 0.0
    for family, n_worlds, seeds in (
            ("weibull", 10, range(101, 111)),
            ("loglog", 5, range(201, 206)),
            ("delayed", 5, range(301, 306))):
        mine, pers, curve = [], [], []
        wins_p, wins_c = 0, 0
        for seed in seeds:
            case, answer = make_world(family, seed)
            rows, secs = run_entry({"f00": case})
            worst_secs = max(worst_secs, secs)
            lm = pinball_loss(answer, rows["f00"])
            lp = pinball_loss(answer, persistence_rows(case))
            lc = pinball_loss(answer, curve_baseline_rows(case))
            mine.append(lm)
            pers.append(lp)
            curve.append(lc)
            wins_p += 1 if lm < lp else 0
            wins_c += 1 if lm < lc else 0
        m = lambda v: sum(v) / len(v)
        results[family] = (m(mine), m(pers), m(curve), wins_p, wins_c)
        print("  synth %-8s mine %.3f | persistence %.3f | curve %.3f "
              "| wins %d/%d vs pers, %d/%d vs curve"
              % (family, m(mine), m(pers), m(curve),
                 wins_p, n_worlds, wins_c, n_worlds))
    print("  worst single-case entry runtime: %.2fs" % worst_secs)
    return results, worst_secs


def check_expanding_cv():
    print("  expanding-window CV (Weibull worlds, mean over 10):")
    cuts = [(24, 48), (36, 54), (48, 58)]
    acc = {c: [] for c in cuts}
    for seed in range(101, 111):
        case, answer_full = make_world("weibull", seed)
        for cut_h, tgt_h in cuts:
            cut_dt = RELEASE + timedelta(hours=cut_h)
            tgt_dt = RELEASE + timedelta(hours=tgt_h)
            sub = truncate_case(case, cut_dt, tgt_dt)
            rows, _ = run_entry({"f00": sub})
            ans = _answer_at("weibull", seed, tgt_h)
            acc[(cut_h, tgt_h)].append(pinball_loss(ans, rows["f00"]))
    out = {}
    for (cut_h, tgt_h), vals in acc.items():
        out[(cut_h, tgt_h)] = sum(vals) / len(vals)
        print("    cut %dh -> +%dh: %.3f"
              % (cut_h, tgt_h - cut_h, out[(cut_h, tgt_h)]))
    return out


def _answer_at(family, seed, t_hours):
    """Answer rates at an arbitrary hour, for the CV check."""
    rng = random.Random(seed)
    practice = load_practice()["p00"]
    trophies = practice["trophies"]
    params = {}
    for tr in trophies:
        tid = tr["id"]
        c = _draw_ceiling(rng, tid)
        if family == "weibull":
            params[tid] = (c, rng.uniform(0.7, 2.2), rng.uniform(15.0, 180.0))
        elif family == "loglog":
            params[tid] = (c, rng.uniform(0.8, 2.4), rng.uniform(8.0, 90.0))
        else:
            params[tid] = (c, rng.uniform(20.0, 75.0), rng.uniform(8.0, 30.0))
    ans = {}
    for tr in trophies:
        tid = tr["id"]
        c, a, b = params[tid]
        if tid == 2:
            ans[tid] = 99.99
            continue
        te = t_hours + 1.0
        if family == "weibull":
            y = c * (1.0 - math.exp(-((max(te, 0.01)) / b) ** a))
        elif family == "loglog":
            y = c / (1.0 + (b / max(te, 0.01)) ** a)
        else:
            y = c / (1.0 + math.exp(-(te - a) / b))
        ans[tid] = min(99.99, max(0.01, round(y, 2)))
    return ans


def check_ablation():
    print("  ablation (5 worlds/family, in-process):")
    for family, seeds in (("weibull", range(101, 106)),
                          ("delayed", range(301, 306))):
        acc = {"weighted": [], "joint": [], "equal": [], "blend50": []}
        for seed in seeds:
            case, answer = make_world(family, seed)
            acc["weighted"].append(pinball_loss(
                answer, entry.forecast_case(case)))
            acc["joint"].append(pinball_loss(
                answer, entry.forecast_case(case, joint_weight_override=1.0)))
            acc["equal"].append(pinball_loss(
                answer, entry.forecast_case(case, equal_weights=True)))
            acc["blend50"].append(pinball_loss(
                answer, entry.forecast_case(case, joint_weight_override=0.5)))
        m = lambda v: sum(v) / len(v)
        print("    %-8s weighted %.3f | joint-only %.3f | equal %.3f "
              "| blend50 %.3f"
              % (family, m(acc["weighted"]), m(acc["joint"]),
                 m(acc["equal"]), m(acc["blend50"])))


def _validate_rows(rows, n_trophies):
    assert isinstance(rows, list) and len(rows) == n_trophies, "row count"
    ids = [r["id"] for r in rows]
    assert len(set(ids)) == n_trophies, "duplicate ids"
    for r in rows:
        assert set(r.keys()) == {"id", "q"}, "row keys"
        q = r["q"]
        assert len(q) == 5, "five quantiles"
        for v in q:
            assert isinstance(v, (int, float)) and math.isfinite(v), "finite"
            assert 0.0 <= v <= 100.0, "range"
        assert all(q[i] <= q[i + 1] for i in range(4)), "non-decreasing"


def check_edge_cases():
    base = load_practice()["p00"]
    trophies = base["trophies"]

    def mkcase(history):
        c = dict(base)
        c["history"] = history
        return c

    one = [base["history"][0]]
    cases = {
        "edge1": mkcase(one),
        "edge2": mkcase([]),
        "edge3": mkcase([dict(base["history"][0],
                              rates=[None] * 55),
                         dict(base["history"][1])]),
        "edge4": mkcase([dict(base["history"][0],
                              rates=[0.01] * 55,
                              players_tracked=638,
                              trophies_earned=0)]),
    }
    rows, _ = run_entry(cases)
    for cid, r in rows.items():
        _validate_rows(r, len(trophies))
        print("  edge %s: valid (%d rows)" % (cid, len(r)))
    # Determinism: run a synthetic final case twice, compare bytes.
    case, _ = make_world("weibull", 101)
    with tempfile.TemporaryDirectory() as td:
        outs = []
        for k in range(2):
            in_dir = os.path.join(td, "in%d" % k)
            out_dir = os.path.join(td, "out%d" % k)
            os.makedirs(in_dir)
            os.makedirs(out_dir)
            with open(os.path.join(in_dir, "f00.in"), "w") as fh:
                json.dump(case, fh)
            env = dict(os.environ)
            env["MAKEFUN_INPUT_DIR"] = in_dir
            env["MAKEFUN_OUTPUT_DIR"] = out_dir
            subprocess.run([sys.executable, os.path.join(ROOT, "main.py")],
                           env=env, capture_output=True, timeout=120,
                           check=True)
            with open(os.path.join(out_dir, "f00.out"), "rb") as fh:
                outs.append(fh.read())
    assert outs[0] == outs[1], "non-deterministic output"
    print("  determinism: 2 runs byte-identical (%d bytes)" % len(outs[0]))


def main():
    t0 = time.time()
    print("== practice proxy ==")
    check_practice()
    print("== synthetic final ==")
    check_synthetic()
    print("== expanding-window CV ==")
    check_expanding_cv()
    print("== ablation ==")
    check_ablation()
    print("== edge cases + determinism ==")
    check_edge_cases()
    print("total backtest wall time: %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
