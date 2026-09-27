# Changelog

## 2026-09-27

Bot version `2026-09-27-v3-shadow`.

- Binary forecasts now log a `stages` dict: median, after review, after shrinkage, after anchor, after calibration, and final.
- Review no longer fires only because every model is extreme on the same side.
- Market blend weight is 0.15. A market is still used only when the title, deadline, and volume match.
- The structured odds track is in shadow mode. It is logged and does not change the submitted forecast.
- Stage-pull "before" table: not yet run. It needs about 30 live binary questions in `data/runs`.
