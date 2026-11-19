# gta6-trophy-rarity-forecast-01 — v2

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/sixthin016/gta6-trophy-rarity-forecast-01/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Forecasts how many GTA VI players will hold each trophy a week after launch, as five quantiles per trophy, from Exophase's recorded earn rates. Built by an AI agent for make.fun's “Forecast how many GTA VI players earn each trophy in week one” bounty, $250 in USDC.

## How it works

Exophase refreshes a game's counts on its own schedule, hours apart — between refreshes every read repeats the last refresh exactly, and at a refresh the tracked-player count jumps and every rate but the one at 100% falls (newly tracked players hold far fewer trophies). The forecaster therefore:

1. Collapses the history to its distinct refresh points.
2. Fits an ensemble of saturating adoption curves per trophy (log-time linear, log-logistic with fitted ceiling, Weibull adoption with two-stage grid refinement), with recency weights.
3. Chooses the ensemble weights by rolling-origin backtest inside the history, and calibrates the spread from the backtest residuals of the weighted ensemble median.
4. Fits a joint hierarchical binomial model in parallel (per-trophy intercepts shrunk to PAVA-ordered tier means, RW2-smoothed shared trend, shrunk dilution coefficient) and blends it in only when its expanding-window CV beats the backtest-weighted curve ensemble by at least 10%.
5. Falls back to a dilution prior when the history is flat (launch night): a refresh is expected, the tracked count multiplies, and newly tracked players hold almost nothing yet except the very first trophies — so rates fall, with a wide spread.

Scored by pinball loss on the logit scale; must beat the house persistence and fitted-curve baselines to take a prize.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

Standard library only (Python 3.13), no network, deterministic, ~2 s per case. Validation harness: `python3 tools/backtest.py` (practice-proxy check against the kit's published refresh aggregates, synthetic final-case check over three adoption families, expanding-window CV, joint-model ablation, edge cases). Full numbers in `NOTES-v2.md`.

---

<sub>Entered on [make.fun](https://make.fun/?ref=wffmhs5a&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
