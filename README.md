# nba-predictor

An end-to-end NBA game-prediction pipeline: ingest historical games from
`nba_api`, engineer leakage-safe features, train and calibrate a gradient
boosted classifier, compare model probabilities against live betting
markets, and serve it all from a Streamlit dashboard.

The project is built around two disciplines that most student sports-ML
projects skip: **strict no-leakage feature construction** (shift-then-roll,
time-based splits, state-as-of-date at inference) and **probability
calibration** (reliability diagrams, ECE, post-hoc isotonic / Platt
rescaling). A miscalibrated 66 % classifier is useless for the Phase-5
market-comparison layer; a calibrated one is the whole point.

---

## Headline results

Metrics on the held-out 2023-24 test season, models trained on 2016-17
through 2021-22 and tuned on 2022-23:

| Model                       | Accuracy | Log-loss | Brier  | AUC   | ECE   |
|-----------------------------|---------:|---------:|-------:|------:|------:|
| Home-team-always-wins       |  ~0.590  |     —    |    —   |   —   |   —   |
| LogReg (4 features)         |   0.665  |   0.606  |  0.210 | 0.732 | 0.046 |
| XGBoost (raw)               |   0.651  |   0.611  |  0.211 | 0.732 | 0.050 |
| XGBoost + Isotonic          |   0.652  |   0.657  |  0.215 |  …    | 0.055 |
| XGBoost + Platt             |   0.648  |   0.615  |  0.213 |  …    | 0.054 |

Interpretation: on this data, the raw XGBoost probabilities are already
reasonably well calibrated (ECE ≈ 0.05), so fitting a post-hoc calibrator
on a single validation season slightly *overfits* the calibration map —
a useful real finding. The machinery is in place for when the calibration
gap gets bigger (e.g. with a noisier model or smaller calibration set).

Walk-forward CV across seven test seasons (LogReg, 4 features, expanding
window starting at 3 seasons of warm-up):

| Metric     | Mean   | Std    |
|------------|-------:|-------:|
| Accuracy   | 0.649  | 0.020  |
| Log-loss   | 0.626  | 0.018  |
| Brier      | 0.218  | 0.008  |
| ROC-AUC    | 0.700  | 0.029  |

Any single-season test number is a noisy draw from this distribution.
Std ≈ 0.02 on accuracy means the 1.4 pp spread between "LogReg" and
"XGBoost" in the table above is within one standard error.

---

## Quickstart

```bash
# 1. Install
pip install -r requirements.txt

# 2. Configure DB connection
cp .env.example .env
# edit .env with your Postgres DSN

# 3. One-time: create tables and bootstrap historical data
python -m src.ingest.ingest_games           # ~10 seasons of team-game logs
python -m src.ingest.ingest_team_stats      # advanced stats (ORtg, DRtg, pace, ...)
python -m src.ingest.ingest_player_games    # per-player minutes for availability features

# 4. Build model-ready feature matrix
python -m src.pipeline.build_training_set
#   → data/processed/features.parquet

# 5. Train + evaluate
python -m src.models.baseline               # home-wins + minimal logreg
python -m src.models.xgboost_model --save v1
python -m src.models.tune --save v2 --trials 100     # Optuna
python -m src.models.walkforward            # multi-season sanity check
python -m src.models.calibration            # reliability diagram + Iso/Platt fit

# 6. Score today's games and compare to market
python -m src.pipeline.predict --save v3
python -m src.pipeline.market_compare

# 7. Dashboard
streamlit run dashboard/app.py
```

---

## Project structure

```
nba-predictor/
├── data/
│   ├── raw/              # CSV backups of ingested sources
│   ├── processed/        # features.parquet + calibration PNGs
│   └── models/           # v{N}.json + sidecar manifests + calibrators
├── notebooks/
│   ├── 01_explore.ipynb          # EDA
│   ├── 02_model_analysis.ipynb   # feature importance, error analysis
│   └── 03_monte_carlo.ipynb      # playoff / play-in / lottery sims
├── src/
│   ├── ingest/           # nba_api + The Odds API ingestion
│   ├── features/         # rolling, rest, Elo, situational, player, odds math
│   ├── models/           # baseline, XGBoost, Optuna tune, splits,
│   │                     # walk-forward, calibration, Monte Carlo
│   └── pipeline/         # build_training_set, predict, market_compare,
│                         # daily_update, backfill
├── dashboard/app.py      # Streamlit front-end (4 tabs)
├── tests/                # pytest suite, 45 tests
├── db.py                 # SQLAlchemy engine + Base
└── requirements.txt
```

---

## Data flow

```
nba_api / The Odds API
         │
         ▼
 Postgres tables:
   team_games, team_games_advanced,
   player_games, odds_snapshots
         │
         ▼
 src/pipeline/build_training_set.py
   ├─ rolling stats (5/10/20)     ← .shift then .rolling ← no leakage
   ├─ rest days + diffs
   ├─ situational (month, covid flag, season progress)
   ├─ player availability (top1/top3 rotation minutes)
   ├─ pivot to matchup grain
   └─ Elo (pre-game, season regression 0.25)
         │
         ▼
 data/processed/features.parquet   (~9 seasons × ~1100 games)
         │
         ▼
 src/models/*  — XGBoost, Optuna, calibration, walk-forward
         │
         ▼
 data/models/{v1,v2,v3}.json       → src/pipeline/predict.py
                                   → predictions table
                                   → src/pipeline/market_compare.py
                                   → dashboard/app.py
```

---

## Feature catalog

74 features at inference. All use only data strictly before the game
being predicted.

- **Team efficiency (rolling 5/10/20 games)**: Net Rating, ORtg, DRtg,
  Pace, TS %, eFG %, points scored, plus/minus
  ([src/features/rolling_stats.py](src/features/rolling_stats.py))
- **Rest & schedule**: home / away days rest, rest differential, back-to-back
  flags ([src/features/rest_features.py](src/features/rest_features.py))
- **Elo**: pre-game rating per team, Elo diff, season regression factor 0.25
  ([src/features/elo.py](src/features/elo.py))
- **Situational**: game number, season progress, month, COVID-season flag
  ([src/features/situational.py](src/features/situational.py))
- **Player availability**: top-1 / top-3 rotation active, rotation minutes
  share, avg rotation minutes
  ([src/features/player_features.py](src/features/player_features.py))

The feature selector in
[src/models/xgboost_model.py:47](src/models/xgboost_model.py#L47) is an
allow-list: named pre-game columns + any column ending in `_r5 / _r10 / _r20 /
_rotation_* / _top*_active_recent`. Adding a new rolling feature upstream
picks it up automatically; adding a raw box-score column does not.

---

## Backtest discipline

### No leakage

- [test_features.py](tests/test_features.py) asserts that perturbing
  game *t*'s raw stats does not change game *t*'s rolling feature. If this
  test ever regresses the model is compromised.
- `.shift(1)` before every `.rolling()`.
- Chronological split in
  [src/models/splits.py](src/models/splits.py); no random splits anywhere.
- At serving time,
  [build_upcoming_matchups](src/pipeline/predict.py#L154) computes state
  strictly from games with `game_date < as_of`.

### Walk-forward CV

Single splits are noisy. [src/models/walkforward.py](src/models/walkforward.py)
refits season-by-season and reports per-fold + aggregate metrics.

```bash
python -m src.models.walkforward --model xgb --mode expanding --initial-train 3
python -m src.models.walkforward --model xgb --mode rolling --window 3
```

Each fold carves the most recent training season off as an inner
early-stopping set so the outer test season is never touched during
training.

### Calibration

[src/models/calibration.py](src/models/calibration.py) measures reliability
(binned predicted vs actual) + Expected Calibration Error, and provides two
post-hoc calibrators:

- `IsotonicCalibrator` — non-parametric monotone step function; prefer when
  the calibration set is large.
- `PlattCalibrator` — sigmoid on logits, two parameters; use when the
  calibration set is small.

Both persist to plain JSON (no pickle) so they're readable across sklearn
versions. Generic load via `load_calibrator(path)` dispatches on the `method`
tag.

---

## Dashboard

`streamlit run dashboard/app.py` — four tabs:

1. **Today's Predictions** — model probability per scheduled game.
2. **Historical Accuracy** — rolling accuracy over the season + reliability
   curve on realized results.
3. **Edge Finder** — model probability vs devigged market consensus,
   with Kelly-fraction sizing.
4. **Elo Leaderboard** — current ratings and recent form per team.

All loaders are Streamlit-cached and NaN-safe against partial-season data.

---

## Models

| File                                                                | Role                                      |
|---------------------------------------------------------------------|-------------------------------------------|
| [baseline.py](src/models/baseline.py)                               | Home-wins floor + 4-feature LogReg        |
| [xgboost_model.py](src/models/xgboost_model.py)                     | XGBoost with early stopping               |
| [tune.py](src/models/tune.py)                                       | Optuna TPE search over 7 hyperparameters  |
| [walkforward.py](src/models/walkforward.py)                         | Expanding / rolling CV across seasons     |
| [calibration.py](src/models/calibration.py)                         | ECE + Isotonic / Platt recalibration      |
| [monte_carlo.py](src/models/monte_carlo.py)                         | Playoff / play-in / lottery sims from Elo |
| [evaluate.py](src/models/evaluate.py)                               | Accuracy / log-loss / Brier / AUC         |

Model artifacts live in `data/models/`: each version is a `{v}.json` model,
a `{v}.manifest.json` with the feature list and training metadata, and
optionally a `{v}.params.json` from Optuna.

---

## Tests

```bash
pytest -q
# 45 tests across leakage, rest + Elo math, odds/market arithmetic,
# player features, walk-forward splits, calibrators
```

The leakage tests are the most important; they are the invariant that
keeps every downstream metric honest.

---

## Finance layer: the honest result

The project includes a full betting backtest (`src/finance/`) against Kaggle closing moneylines (2007–2022, 19,820 rows). Three charts summarize the finding; the full visual set lives in [docs/FINANCE.md](docs/FINANCE.md).

![Model comparison across OOS folds](docs/assets/model_comparison.png)

![Equity curves — 0.25× Kelly, 2% min-edge](docs/assets/equity_curve_overlay.png)

![Meta-LR edge-bucket win rate](docs/assets/edge_bucket_meta_lr.png)

The headline finding:

| Model edge over closing line | # bets | Bet-side win rate |
|------------------------------|:------:|:-----------------:|
| 0 – 2 %                      | 459    | **43.6 %**        |
| 2 – 5 %                      | 672    | **43.8 %**        |
| 5 – 8 %                      | 668    | **43.1 %**        |
| 8 – 12 %                     | 685    | **36.8 %**        |
| 12 – 20 %                    | 699    | **38.1 %**        |
| 20 %+                        | 288    | **29.9 %**        |

The model is accurate (65 %) but has **no betting alpha**. The edge signal is perfectly inverted: the more confidently the model disagrees with the market, the more often the market is right. This is market efficiency behaving as expected — the closing line knows things our feature set cannot see (line movement, lineup news, sharp money).

The full analysis is in [docs/FINANCE.md](docs/FINANCE.md) and [notebooks/04_betting_backtest.ipynb](nba-predictor/notebooks/04_betting_backtest.ipynb). The infrastructure — walk-forward predictions, Kelly sizing with simultaneous-bet normalization, Sharpe / drawdown / Calmar on sparse bet-day returns, a bet-log audit table — is all there. It just confirms that a 65 %-accurate moneyline model does not beat the closing line with this feature set.

**Meta-model residual-alpha test** ([docs/FINANCE.md §8.1](docs/FINANCE.md)). A side-level meta-model (L2-regularized logistic regression primary, shallow XGBoost as robustness check) was trained via nested walk-forward to test whether combining `side_model_prob` and `side_market_prob` via logit-space features extracts residual edge. Benchmark: market-only. Result: market-only wins on log-loss (0.610 vs 0.612 LR / 0.624 XGB), meta-LR ROI −10.7 % with 95 % bootstrap CI `[−21.6 %, +0.5 %]`, negative in both test folds. **No residual alpha found** — granting the model the market itself as a feature still doesn't produce positive-EV signal on this feature set. The strongest possible form of the market-efficiency finding.

## What's next

Given the meta-model's negative result (closing above), the natural next research directions are feature additions, not modeling changes:

- **Line-movement features** — opening line, intraday line moves, closing-line value (CLV) as a predictor. Requires paid historical odds with timestamps, not the single-snapshot Kaggle closes used here.
- **Lineup-level availability** — not just top-3 rotation minutes but actual game-day starter announcements (NLP on injury reports, late-scratch detection).
- **Referee crew assignments** — crew-level scoring / foul-rate tendencies are public but not in our feature set.
- **Faster feature refresh** — re-score the model at T-30 min (after lineup release) rather than morning-of. Deployment-engineering problem.

Existing wish-list items not attempted:

- **LightGBM + stacking ensemble** — cheap accuracy bump; helps prediction, not necessarily alpha.
- **Point-spread / totals regression head** — the odds data already supports this.
- **Play-by-play ingestion** (`nba_api.stats.endpoints.playbyplayv2`) —
  unlocks live win-probability (Phase 4).
- **Injury-impact model** — quantify win-probability delta per player absence.
- **Public Streamlit Cloud deployment** — live-updating version of the dashboard.

---

## Disclaimer

This project is a research / analytics exercise. Sports outcomes are
inherently uncertain and no model guarantees profit. Nothing here is
betting advice.
