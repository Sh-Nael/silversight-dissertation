"""Named model specifications: the single list of what can be run.

The CLI, the experiment configs and (later) the app all refer to models by these
names, so a model's construction is defined in exactly one place.
"""

from __future__ import annotations

from xaglab.eval.harness import ModelSpec
from xaglab.models.aimdg.forecaster import AimDG
from xaglab.models.baselines import EWMA, ARGarch, Climatology, PooledGBM, PooledLinear
from xaglab.models.neural.forecaster import TunedNeural

_REGISTRY: dict[str, ModelSpec] = {
    "climatology": ModelSpec(Climatology, {}, "climatology"),
    "ewma": ModelSpec(EWMA, {"lam": 0.94}, "ewma"),
    "ar_garch": ModelSpec(ARGarch, {}, "ar_garch"),
    "linear": ModelSpec(PooledLinear, {}, "linear"),
    "gbm": ModelSpec(PooledGBM, {}, "gbm"),
    # D-39: the flat models with the MACD and Bollinger block (silver only / every market)
    "linear_ta": ModelSpec(PooledLinear, {"extras": "silver"}, "linear_ta"),
    "gbm_ta": ModelSpec(PooledGBM, {"extras": "silver"}, "gbm_ta"),
    "linear_ta_all": ModelSpec(PooledLinear, {"extras": "all"}, "linear_ta_all"),
    "gbm_ta_all": ModelSpec(PooledGBM, {"extras": "all"}, "gbm_ta_all"),
    "lstm": ModelSpec(TunedNeural, {"cell": "lstm"}, "lstm"),
    "gru": ModelSpec(TunedNeural, {"cell": "gru"}, "gru"),
    "static_gnn": ModelSpec(TunedNeural, {"net": "static_gnn"}, "static_gnn"),
    "aimdg": ModelSpec(AimDG, {}, "aimdg"),
    # Phase 5 A4: recency-weighted training, half-life 2 years (504 rows), for both graph models
    "static_gnn_rw": ModelSpec(TunedNeural, {"net": "static_gnn", "half_life": 504.0}, "static_gnn_rw"),
    "aimdg_rw": ModelSpec(AimDG, {"half_life": 504.0}, "aimdg_rw"),
}

BASELINES = ("climatology", "ewma", "ar_garch", "linear", "gbm")


def get(name: str) -> ModelSpec:
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown model {name!r}; known: {sorted(_REGISTRY)}") from None


def register(spec: ModelSpec) -> None:
    _REGISTRY[spec.name] = spec


def names() -> list[str]:
    return sorted(_REGISTRY)
