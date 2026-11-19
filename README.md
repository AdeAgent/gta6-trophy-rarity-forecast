# gta6-trophy-rarity-forecast

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AdeAgent/gta6-trophy-rarity-forecast/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/AdeAgent/gta6-trophy-rarity-forecast/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

Logit-trend forecaster for GTA VI trophy earn rates on Exophase: per-trophy least-squares trends in logit space against log-time, shrunk toward tier medians, with honest spreads for the 4.5-day horizon and a stale-page refresh adjustment. Pure Python 3.13 standard library, no network. For each of the 55 trophies it forecasts five quantiles of the earn rate at the target: a line of logit(rate) on ln(1 + hours since launch) fitted over the last 12 hours of reads, read at the target, with the spread combining the house random walk, the fit residual, slope uncertainty and a model-uncertainty term that grows with the extrapolation. When every read in a case's history is identical — the page has not refreshed at all — the target read is a refreshed page with many more players, so the median shifts down for the dilution and the spread widens; the 100% trophy stays 100%.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=5dn9fd7w&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
