"""Drop-in upgrades for a Metaculus forecasting bot: aggregation, calibration,
structured probability models, numeric CDF handling, logging and offline evaluation."""
from .aggregation import (
    aggregate_binary,
    aggregate_mc,
    blend_logodds,
    logit,
    logodds_spread,
    mix_cdfs,
    normalize_with_floor,
    sigmoid,
    spread,
)
from .calibration import (
    PlattParams,
    SegmentedCalibrator,
    cap,
    fit_platt,
    fit_platt_cv,
    mc_temperature_apply,
    mc_temperature_fit,
)
from .structured import (
    Scenario,
    enforce_monotone_deadlines,
    enforce_monotone_thresholds,
    hazard_prob_at_least_one,
    negation_gap,
    prob_exceeds,
    scenario_tree,
)

__all__ = [name for name in dir() if not name.startswith("_")]
