# Forecasting bot: logic summary (for research / next-steps review)

## Context

- A bot for the **Metaculus AI Forecasting Benchmark** (seasonal AIB tournament + MiniBench,
  plus the Metaculus Cup). It forecasts binary, multiple-choice, numeric/discrete and date
  questions. Scoring is log-score based: peer score / head-to-head against other bots and pros.
- It's built on Metaculus's template repo and the `forecasting-tools` SDK (pinned at 0.2.92).
  The SDK fetches questions, skips questions already forecast, submits forecasts and posts
  a private reasoning comment.
- The live strategy is in `adaptive_bot.py` (`AdaptiveForecastBot`, a subclass of the
  template's `SummerTemplateBot2026`). `run_adaptive.py` is the entry point.
- GitHub Actions runs it every 20 minutes on the tournament and MiniBench, with publishing on.
  The Metaculus Cup workflow still runs the original `main.py` template.
- Models come from env vars (all via OpenRouter / LiteLLM IDs):
  - `BOT_MODELS`: 1–3 forecasting models.
  - `BOT_RESEARCHER`: primary research route, typically an AskNews route.
  - `BOT_RESEARCH_FALLBACK`: default is a Gemini Flash model with `:online` web search.

## Pipeline per question

### 1. Research (resolution-first, "research v2")

Runs when a researcher is configured. Otherwise, or if any step fails, it falls back to step 1b.

1. **Resolution spec.** A cheap LLM turns the question text, resolution criteria and fine
   print into JSON. The JSON has:
   - question type and archetype
   - target and what makes it resolve YES
   - deadline, time remaining (in days), units, thresholds and bounds
   - resolution sources with URLs
   - whether the event has already happened
   - conditions still to be met
   - ambiguities
2. **Resolution facts.** The bot fetches every URL in the spec with a plain HTTP GET, strips
   the HTML and keeps the first 4,000 characters of each page.
3. **Cruxes.** An LLM lists 3–5 unknowns that would most change the forecast, with 1–2
   search queries each.
4. **Crux search.** Up to 10 queries in total. Queries alternate between the primary
   researcher (AskNews) and the fallback (web-search LLM).
   - At least 2 distinct providers must succeed, or the whole v2 path is abandoned.
   - Each response must contain substantive text and a source URL.
   - Reliability layer: 1 retry, then the fallback route; a circuit breaker opens after
     repeated failures; results are cached for 1 hour.
5. **Claims ledger.** An LLM extracts factual claims and tags each with:
   - source URL
   - published and event dates
   - which cruxes it bears on
   - direction (for/against)
   - an "independence group", so syndicated reports of one event count once
6. **Outside view.** Finds up to 3 similar questions by word overlap (Jaccard ≥ 0.25) in the
   bot's own local log of resolved questions.
7. **Brief.** All of the above is formatted as one markdown brief that includes a list of
   open uncertainties (cruxes with no answer, plus the spec's ambiguities).

**1b. Fallback research.** Two search calls run in parallel:
- (a) current state, the resolution source and recent developments
- (b) historical base rates, structural changes and contrary evidence

### 2. Independent forecasts (ensemble)

- Each model in `BOT_MODELS` gets one call, all run at once. Each model is given a
  different "role":
  1. reference class / base rate first
  2. causal mechanisms and bottlenecks first
  3. alternative scenarios and overlooked evidence
- The prompt tells the model to:
  - optimise for log score and read the exact resolution criteria
  - use the event horizon in the criteria, not the submission deadline
  - keep sourced facts separate from judgment
  - not invent likelihood ratios or double-count correlated evidence
  - not fall back to 50% by default
  - give predictive (not confidence) intervals for numeric questions
- Output is strict JSON with a rationale, baseline, key uncertainty, a `needs_review` flag,
  and one of:
  - binary: a probability
  - multiple choice: a probability for every option
  - numeric/date: 7 quantiles (10/20/40/50/60/80/90)
- Strict validation checks for: malformed JSON, missing fields, option probabilities that
  don't sum to 1, quantiles out of order, and values outside closed bounds. The model gets
  one "fix the format, not the judgment" repair retry. The SDK must also accept the result
  (for numeric questions, a full CDF build).
- If every model fails, nothing is submitted. There is no made-up fallback forecast.

### 3. Review trigger → evidence audit → one revision round

A review is **triggered** if any of these hold:
- a model set `needs_review`
- research failed or came back empty
- binary: models are ≥ 0.20 apart, or any model is ≤ 5% or ≥ 95%
- multiple choice: models are ≥ 0.20 apart on any option, or any option is ≥ 95%
- numeric: the spread of the medians is more than half the widest 10–90 interval, or an
  interval has width zero
- only part of the ensemble succeeded

When triggered (and at least 30 seconds of the question budget remain):
1. The last model acts as **reviewer**. It names the most important factual disagreement or
   shared unsupported assumption, and gives one search query. It doesn't give a probability.
2. The bot runs **one targeted search** with that query.
3. Each model sees its own original forecast plus the audit and new evidence, and may
   **revise once**. It is told it need not agree with the reviewer.

### 4. Aggregation and post-processing

- **Binary:**
  1. median of the model probabilities
  2. Platt scaling, only if `calibration.json` exists and says `applied: true`
  3. clip to 2%–98% (3%–97% on MiniBench)
- **Multiple choice:** per-option mean, a 1% floor per option (mixed with uniform), then
  optional temperature scaling from `calibration.json`.
- **Numeric/date:** a linear mixture of each model's full CDF on the SDK's grid, which keeps
  the disagreement between models. The SDK then builds the submission from that.

### 5. Submission and logging

- The SDK posts the forecast, then a private comment with every model's forecasts, the
  review notes, the audit and the final forecasts.
- Every run appends events to `logs/forecasts.jsonl`: research complete, reasoning complete
  (initial and final model forecasts, raw aggregate, submitted value), and failures. Each
  event records per-call cost, latency and errors.
- Research text is stored under content hashes. `logs/` is uploaded as a GitHub Actions
  artifact kept for 90 days, and is not committed to the repo.

### Limits

- Per call: 120 seconds. Per question: 600 seconds (capped at close time minus 2 minutes).
- At most 4 provider calls run at once. Max output: 3,000 tokens.

## Built but not wired into live forecasting

Their feature flags in `upgrade_config.py` are off, or no code calls them yet.

- `forecast_upgrades/structured.py`:
  - Gamma-Poisson hazard model for "at least one event by the deadline", with the LLM's
    rate adjustment clamped to between ⅓× and 3×
  - monotonicity fixes (PAVA) across related deadlines and thresholds
  - a negation-coherence check
  - scenario-tree probabilities
- `forecast_upgrades/numeric.py`:
  - percentiles → CDF with log-scale grids and logistic tails
  - step-size rules matching Metaculus validation
  - tail widening
- `forecast_upgrades/aggregation.py`: log-odds mean with optional extremizing, trimmed mean,
  and a log-odds blend with a market price.
- Planned flags that are off: `FF_ENSEMBLE_V2` (3 providers × 2 samples),
  `FF_STRUCTURED_MODELS`, `FF_TRIGGERS_V2` (log-odds spread triggers), `FF_RESOLUTION_ROUND`,
  `FF_NUMERIC_V2`, and `FF_MARKET_ANCHOR` (blend with a matched prediction market at weight
  0.3).
- Offline evaluation (`scripts/replay_and_fit.py`, `forecast_upgrades/evaluation.py`):
  - replays different aggregation, calibration and cap pipelines on resolved questions,
    with no new LLM calls
  - splits by time and compares pipelines with a paired bootstrap on head-to-head scores
  - fits Platt scaling with cross-validated shrinkage, and only after 100 or more resolved
    binary questions
  - `scripts/sync_resolutions.py` fills in outcomes from the Metaculus API

## Known gaps and weaknesses (found while reading the code)

1. **The calibration loop can't get its data yet.** Each GitHub run starts empty, and `logs/`
   exists only as per-run artifacts. The replay workflow checks out the repo, which has no
   logs, so `calibration.json` is never produced and calibration stays as it is. Separately,
   the live logger writes event records (`question_type`, `initial_candidates`, …), but the
   replay script expects a different schema (`qtype`, `submitted_at`, `members[stage]`).
2. **The "outside view" is always empty in production**, for the same reason: there's no
   log history on the runner.
3. **Research v2 needs 2 distinct providers.** If the fallback fails, or matches the primary,
   every question silently drops to the simpler two-search research.
4. **The ensemble is small.** 1–3 models, one sample each. A plain median of 2–3 numbers is
   noisy, and nothing is weighted by each model's track record.
5. **The fixed 2–98% cap** costs points on questions that are near-certain, such as ones
   already resolved in fact, or long-shots with little time left. Nothing yet decides caps
   by time remaining or whether the event has already happened.
6. **Nothing uses the community prediction or prediction markets as an anchor** yet.
7. **Numeric forecasts** rely on 7 quantiles from the LLM, with no tail widening, no
   baseline from historical data, and no drift model.
8. **Forecasts are never updated.** Tournament mode skips questions already forecast, so a
   forecast is never revised as news arrives or close time approaches.
9. **All thresholds are guesses, not fitted:** the 0.20 disagreement, the 5%/95% extremes,
   the 30-second minimum and the caps.
10. **Forecasts and comments aren't posted atomically.** A failed comment isn't repaired.

## Questions for the research AI

- Which of the unwired upgrades (structured hazard models, market anchoring, log-odds
  aggregation with extremizing, numeric tail handling) have the best evidence of improving
  log/peer scores in LLM forecasting tournaments?
- What's the best way to persist the forecast log between stateless GitHub Actions runs
  (a git branch, a gist, a cloud bucket, a GitHub release asset) so calibration and the
  outside view can work?
- What does current research say about:
  - ensemble size and diversity (more models vs. more samples per model)
  - median vs. mean vs. log-odds mean aggregation
  - extremizing LLM ensembles
- Are debate/review rounds worth their cost, or do they mostly pull models toward each
  other (convergence without accuracy gains)?
- How should research be structured for different question archetypes (scheduled events,
  "will X happen by date", numeric time series, elections)? Is it worth sending questions
  to specialised routines?
- Is updating forecasts during the question's life (re-forecasting open questions over time)
  allowed and worthwhile in the AIB tournament?
- What have the top bots from past AIB seasons (open-source write-ups) done that this
  pipeline doesn't?
