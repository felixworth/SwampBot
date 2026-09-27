"""Central configuration for the staged forecast-bot upgrade.

Phase 0 only loads and reports these settings. Forecasting behaviour remains on
the existing adaptive configuration until the phase that owns each flag is
implemented and explicitly enabled.
"""

UPGRADE = {
    # Feature flags. Calibration is identity-only while calibration.json is absent.
    "FF_RESOLUTION_SPEC": True,
    "FF_RESEARCH_V2": True,
    "FF_ENSEMBLE_V2": False,
    "FF_STRUCTURED_MODELS": True,
    "FF_TRIGGERS_V2": False,
    "FF_RESOLUTION_ROUND": False,
    "FF_NUMERIC_V2": True,
    "FF_MARKET_ANCHOR": True,
    "FF_CALIBRATION": True,
    "FF_NUMERIC_TAILS": True,
    "FF_LOGIT_SHRINK": True,
    "FF_CONDITIONAL_CAP": True,
    "FF_SEPARATE_REVIEWER": True,
    "FF_ROLE_ROTATION": True,
    "FF_NUMERIC_LOGISTIC": False,

    # Reliability and logging targets for later phases.
    "run_every_minutes": 20,
    "question_time_budget_s": 1800,
    "log_path": "logs/forecasts.jsonl",
    "research_dir": "logs/research",

    # Research v2.
    "research_max_queries": 10,
    "research_min_providers": 2,
    "cruxes_min": 3,
    "cruxes_max": 5,

    # Ensemble v2. Placeholders are deliberately inert while FF_ENSEMBLE_V2 is false.
    "forecaster_models": [
        "PROVIDER_A_MODEL",
        "PROVIDER_B_MODEL",
        "PROVIDER_C_MODEL",
    ],
    "samples_per_model": 2,
    "reasoning_effort": "high",

    # Structured probability models.
    "structured_weight": 0.5,
    "hazard_prior_events": 0.5,
    "hazard_multiplier_bounds": (1 / 3, 3.0),

    # Validation and disagreement triggers.
    "closed_bound_violation_max": 0.02,
    "trigger_trimmed_logodds_range": 0.8,
    "trigger_extreme_member": (0.03, 0.97),
    "trigger_moderate_median": (0.10, 0.90),
    "trigger_numeric_median_spread": 0.15,
    "target_trigger_rate": 0.25,

    # Triggered resolution round.
    "resolution_round_queries": 3,
    "reliance_shift_logodds": 0.5,

    # Aggregation and numeric targets.
    "binary_aggregation": "median",
    "mc_floor": 0.01,
    "numeric_percentiles": [0.025, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.975],
    "numeric_baseline_weight": 0.3,
    "numeric_drift_damping": 0.5,
    "numeric_min_history_changes": 20,
    "tail_widening_factor": 1.0,
    "market_blend_weight": 0.15,
    "STRUCTURED_MODE": "shadow",
    "STRUCTURED_BLEND_WEIGHT": 0.5,
    "bot_version": "2026-09-27-v3-shadow",
    "market_max_deadline_gap_days": 7,

    # Calibration and caps.
    "calibration_path": "data/calibration.json",
    "history_glob": "data/runs/**/*.jsonl",
    "outcomes_path": "data/outcomes.jsonl",
    "spend_floor_usd": 1.0,
    "shrink_prior_p": 0.35,
    "shrink_tau": 2.0,
    "calibration_min_resolved": 100,
    "binary_cap": (0.02, 0.98),
    "binary_cap_auto_resolved": (0.03, 0.97),
}


def feature_flags() -> dict[str, bool]:
    """Return a copy suitable for logs and bot metadata."""
    return {
        name: bool(value)
        for name, value in UPGRADE.items()
        if name.startswith("FF_")
    }
