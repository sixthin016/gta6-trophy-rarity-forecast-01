# GTA VI Trophy Rarity Forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Entry for the [make.fun competition](https://make.fun/bounty/gta6-trophy-rarity-forecast-01):
forecast, for each of Grand Theft Auto VI's 55 PlayStation 5 trophies, the
percent of the players Exophase tracks who will have earned it at
2026-11-26 00:00 UTC, as five quantiles per trophy (5%, 25%, 50%, 75%,
95%), scored by pinball loss on the logit scale.

`main.py` (Python 3 standard library only) reads one case per `*.in` file
from `$MAKEFUN_INPUT_DIR` (default `/input`) and writes one forecast per
case to `$MAKEFUN_OUTPUT_DIR` (default `/output`). It is deterministic
and runs in about a second per case.

## How it works

**Refresh points.** Exophase refreshes a game's counts on its own
schedule, hours apart; between refreshes every read is byte-identical.
The history is collapsed to its distinct states, each kept at the time
it was first observed, so the flat stretches do not overweight the fit.

**Adoption-curve ensemble.** Per trophy, three curves are fitted in
logit space against log time and read at the target:

- a log-linear fit of logit(rate) on ln(hours + shift);
- a log-logistic curve with a fitted ceiling;
- a Weibull curve with a fitted ceiling.

The ceiling comes from a small grid, and so does the time shift: play
began some unknown time before the first read (early players, earliest
time zones), and the shift that straightens the history is the clock
the curve really runs on. Every fit weights recent points most
(24-hour half-life): the slope that matters for the target is the
current one, not launch night's. No earners/players decomposition is
used: splitting the two extrapolations was tried and roughly doubled
the error.

**Weights from fit quality.** Each model is scored by how well it fits
each trophy's own history (mean squared logit error, pooled across
trophies and per trophy, shrunk toward the pooled weights). A model
that cannot bend to a saturating trophy's curve must not carry its
forecast, and one that cannot follow a still-rising trophy must not
carry that one either.

**Saturation guard.** When both ceiling models agree a trophy is
nearly done rising (their target within 30% of the last rate), the
unbounded log-linear fit is capped just above the best ceiling call:
it has no ceiling to stop at, and a saturating trophy is exactly
where it overshoots.

**Dilution prior for a flat history.** When the whole history is a
single refresh point there is nothing to fit, and the one refresh the
house ever observed (638 → 3,868 players tracked, every rate but the
100% one falling) says what happens next: a refresh is expected, the
player base grows roughly sixfold, and the newcomers hold almost no
trophy except the one everybody holds. The median is the diluted rate:
old earners plus a thin spread of new earners over the grown base.
The trophy everybody holds stays at ~100%.

**Spread.** A logit-normal band around the ensemble median. Its base
is the median absolute residual of the weighted ensemble at long
rolling origins (forecasting the last point from 12+ hours before
it), grown with the log-distance from the last point to the target,
and widened trophy by trophy by how far the models disagree at the
target; capped at 0.9 logit units.

## Local validation

`python3 tools/backtest.py` replicates the scoring rule from
`competition.yaml` (pinball loss on the logit scale) and checks the
entry against practice-proxy truths, synthetic final worlds, and edge
cases. Results (19 Nov 2026):

| Check | This entry | House baselines |
|---|---|---|
| Practice proxy, 4 truth variants consistent with the public aggregates (N 638 → 3,868, trophies earned 943 → 4,276) | **0.12–0.21** | persistence 1.39–1.52, curve 1.39–1.52 |
| Synthetic final, Weibull worlds (10 seeds, mean) | **0.050** | curve 0.228, persistence 0.515 |
| Synthetic final, log-logistic worlds (10 seeds, mean) | **0.050** | curve 0.225, persistence 0.499 |
| Synthetic final, delayed-sigmoid worlds (10 seeds, mean) | **0.179** | curve 0.320, persistence 0.308 |
| Edge cases (1 read, empty history, null rates, all at floor, all at 100%) | valid output, all | — |
| Determinism (2 runs) | byte-identical | — |
| Output size | 5,606 bytes for both practice cases (< 65,536 per case) | — |
| Runtime | worst 1.2 s on realistic final cases; 2.4 s pathological (357 distinct reads) — limit 8 s | — |

Known limit, stated honestly: on cohort-mixture stress worlds — where
each refresh dilutes the rate with newcomers so the observed rate lags
the cohorts' adoption — the entry scores 0.38–0.42, around the curve
baseline's 0.32–0.33. That lag is hidden in the player age structure,
which no history-only aggregate model can recover; the entry still
beats persistence (0.77–0.83) there. The final score itself is only
known once the target's rates are recorded (10 Dec 2026).

## License

MIT — see `LICENSE`.
