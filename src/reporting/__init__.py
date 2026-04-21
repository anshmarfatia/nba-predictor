"""Reporting / visualization layer.

Pure plotting functions (`charts.py`) + a CLI (`generate_figures.py`) that
loads the project's existing artifacts and emits static PNGs to
`docs/assets/`. No empirical computation is done here beyond what's
strictly necessary to arrange the charts — results are consumed from
walk-forward / meta-model / backtest pipelines, never recomputed with
different discipline.
"""
