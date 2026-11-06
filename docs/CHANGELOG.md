# Changelog

## 2026-10-20

Bot version `2026-10-20-v6-minibench-oss`.

- Forecast logs are no longer uploaded as public Actions artifacts, and per-forecast reasoning is no longer printed in the job log.
- MiniBench has its own workflow. The Fall job forecasts only the seasonal tournament. The MiniBench slug can be set with `BOT_MINIBENCH_ID`.
- The ensemble is unchanged: two forecasts per model.
- The repository includes an MIT licence.

## 2026-09-28

Bot version `2026-09-28-v5-routing`.

- The number of web-search queries follows the resolution archetype. A numeric question gets three; a deadline question gets one. The cap stays at five queries.
- A rate limit, token limit, or context-length error is a normal question failure. Only credit, spending-limit, and authentication errors stop the run.

Bot version `2026-09-28-v4-ensemble`.

- Each configured model now forecasts twice. Samples stay grouped by model if one of them fails.
- Review disagreement and the split-extreme check use the per-model mean, so the extra sample does not raise the trigger rate by itself.
- Research keeps one search query per crux, at most five queries. The first query uses the web-search model; the rest use AskNews.
- A coverage check lists open Fall questions that are still unforecast after two hours, and questions stuck after two failures. `BOT_RETRY_FAILED` can retry those ids on purpose.
- The tournament workflow schedule is in the file. Enable the workflow on GitHub before it starts running.

## 2026-09-27

Bot version `2026-09-27-v3-shadow`.

- Binary forecasts now log a `stages` dict: median, after review, after shrinkage, after anchor, after calibration, and final.
- Review no longer fires only because every model is extreme on the same side.
- Market blend weight is 0.15. A market is still used only when the title, deadline, and volume match.
- The structured odds track is in shadow mode. It is logged and does not change the submitted forecast.
- Stage-pull "before" table: not yet run. It needs about 30 live binary questions in `data/runs`.
