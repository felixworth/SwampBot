# Forecast bot codebase map

This map describes the active adaptive runner as of Phase 0. `main.py` remains
the rollback/template implementation; production automation now calls
`run_adaptive.py`.

## Entrypoints and ownership

- `run_adaptive.py`: active CLI. Selects testing, seasonal tournament, Cup and
  MiniBench targets; dry-run is the default and `--publish` enables writes.
- `adaptive_bot.py`: active strategy. Subclasses `SummerTemplateBot2026` and
  replaces research plus binary, multiple-choice, numeric and date forecasting.
- `forecast_logic.py`: provider-independent prompts, strict output validation,
  review triggers, one optional evidence search and one revision round.
- `forecast_logging.py`: failure-safe JSONL events, per-call cost/latency/error
  telemetry and content-addressed private research files.
- `main.py`: original `SummerTemplateBot2026` and inherited SDK integration.
- `main_with_no_framework.py`: standalone reference implementation, not used by
  the adaptive workflow.
- `upgrade_config.py`: central target configuration and feature flags. Phase 0
  exposes these flags in bot metadata but does not change forecasting behaviour.

## B1 - Fetch questions and skip existing forecasts

**Files/functions**

- `run_adaptive.run`: builds tournament targets and calls
  `ForecastBot.forecast_on_tournament`, or fetches a limited testing-area sample
  before calling `ForecastBot.forecast_questions`.
- Inherited `ForecastBot.forecast_on_tournament` and `forecast_questions` own
  orchestration and server-derived `already_forecasted` filtering.

**Inputs**

- CLI mode, `BOT_TOURNAMENT_ID`, `--include-minibench`, `--limit`.
- Authenticated question records returned by Metaculus.

**Outputs**

- Typed `MetaculusQuestion` objects passed to the forecasting pipeline.
- Previously forecasted questions are skipped in tournament mode.

**Metaculus API**

- Installed client: `forecasting_tools.MetaculusClient` version 0.2.92.
- Open tournament questions: `GET /api/posts/` through
  `get_all_open_questions_from_tournament`.
- Individual post details: `GET /api/posts/{post_id}/`.
- Auth check: `GET /api/users/me`.

The exact `already_forecasted` source field is normalized by
`forecasting-tools` into `MetaculusQuestion.already_forecasted`; Phase 1 must
confirm the raw API field before changing this path.

## B2 - Research

**Files/functions**

- `adaptive_bot.AdaptiveForecastBot.run_research`
- `adaptive_bot.AdaptiveForecastBot._search`

**Inputs**

- Full typed question converted by `_spec`, including criteria, fine print,
  bounds, units, dates and conditional context.
- `BOT_RESEARCHER` provider route.

**Process**

- Two concurrent lanes: current/resolution evidence and historical/base-rate
  evidence.
- Provider routing supports AskNews preconfigured routes,
  `smart-searcher/<model>`, a search-enabled LLM, or explicit `no_research`.
- Responses must contain substantive text and at least one source URL.
- Transient primary-provider failures retry once, then route through
  `BOT_RESEARCH_FALLBACK` (OpenRouter native online search by default).
- Authentication failures do not retry. Repeated transient failures open a
  five-minute circuit breaker so later calls go directly to the fallback.
- Successful responses are cached for one hour by normalized prompt and
  provider route. Cache files stay under ignored private `logs/research/cache`.
- Each lane shares the question deadline and call limiter.

**Output**

- One text research report with two labelled sections. Provider failures become
  explicit `RESEARCH_FAILED` markers rather than disappearing.

## B3 - Independent model forecasts

**Files/functions**

- `adaptive_bot.AdaptiveForecastBot.__init__`
- `forecast_logic.ForecastEngine.run`
- `forecast_logic.forecast_prompt`

**Inputs**

- One to three comma-separated LiteLLM IDs in `BOT_MODELS`.
- Question spec and shared research report.

**Process**

- One independent first-pass call per configured model, concurrently.
- The SDK's outer forecast count is forced to one so an entire ensemble runs
  exactly once per question.
- Calls share a four-call process limiter, per-call timeout and per-question
  deadline.

**Output**

- A list of valid candidate dictionaries plus failure notes.

## B4 - Typed model output

**Files/functions**

- `forecast_logic.output_schema`
- `forecast_logic.validate_forecast`

**Contracts**

- Binary: decimal `probability`.
- Multiple choice: exact option-name-to-probability mapping.
- Numeric/discrete/date: seven ordered quantiles.
- Every candidate includes rationale, baseline, key uncertainty and a
  `needs_review` flag.

Discrete questions inherit `NumericQuestion` in forecasting-tools 0.2.92 and
therefore use the numeric path.

## B5 - Validation and repair

**Files/functions**

- `forecast_logic.parse_json`, `number`, `date_number`, `validate_forecast`
- `ForecastEngine._candidate`
- `AdaptiveForecastBot._convert`

**Process**

- Rejects malformed/non-object JSON, duplicate keys, non-finite numbers,
  booleans as numbers, missing fields, invalid option sets/totals, unordered
  quantiles and violations of closed bounds.
- Permits one format/validity repair call without asking the model to change its
  underlying judgment.
- Converts through the installed SDK before acceptance, forcing numeric CDF
  construction and validation.

**Output**

- SDK-compatible prediction values or a recorded analyst failure.

## B6 - Review triggers

**Files/functions**

- `forecast_logic.review_reasons`
- `forecast_logic.ForecastEngine.run`

**Triggers**

- Analyst self-flag, failed/empty research, binary or option disagreement,
  extreme probabilities, numeric/date disagreement, collapsed intervals, or a
  partially failed ensemble.

**Output**

- Trigger reasons in `EngineResult.notes`; no review when disabled, untriggered
  or fewer than 30 seconds remain.

## B7 - Evidence audit and revision

**Files/functions**

- Review and revision section of `ForecastEngine.run`.

**Process**

- A reviewer identifies one unsupported assumption or factual gap without
  dictating a replacement probability.
- At most one targeted search is made.
- Each successful analyst independently sees its own initial forecast plus the
  audit and may revise once.
- Failed review/revision retains valid initial forecasts.

**Output**

- Initial and final candidate lists, audit text and failure/retention notes.

## B8 - Aggregation

**Files/functions**

- `adaptive_bot.AdaptiveForecastBot._combine`

**Process/output**

- Binary: arithmetic mean after the existing 1%-99% clip.
- Multiple choice: per-option arithmetic mean; the SDK normalizes.
- Numeric/discrete/date: linear mixture of full member CDFs on the common SDK
  grid, then `NumericDistribution.from_question`.

The installed SDK can re-interpolate or standardize numeric distributions near
bounds. Any Phase 6 work must verify behaviour against installed
forecasting-tools 0.2.92 before relying on rules documented for 0.3.1.

## B9 - Submit forecast and private comment

**Files/functions**

- Inherited `ForecastBot` report/submission methods.
- `run_adaptive.run` controls `publish_reports_to_metaculus`.
- `AdaptiveForecastBot._adaptive_forecast` creates the reasoning comment.

**Metaculus writes**

- Forecasts: `POST /api/questions/forecast/`
  - binary payload: `probability_yes`
  - multiple choice: `probability_yes_per_category`
  - numeric/discrete/date: `continuous_cdf`
- Private comment: `POST /api/comments/create/` with `is_private=True`.
- The SDK posts the forecast before the comment.

**Dry run**

- Dry-run is the default; `--dry-run` is an explicit alias.
- The complete pipeline still runs and reports are saved under
  `logs/adaptive/`, while the SDK suppresses forecast and comment POSTs.
- `bot.log_report_summary` logs final predictions and comment reasoning.

## Question representations

- `BinaryQuestion` -> float probability.
- `MultipleChoiceQuestion` -> `PredictedOptionList`.
- `NumericQuestion` -> `NumericDistribution`.
- `DiscreteQuestion` -> numeric path because it subclasses `NumericQuestion`.
- `DateQuestion` -> `NumericDistribution` with date values represented as UTC
  timestamps internally.
- `ConditionalQuestion` -> inherited framework dispatch over parent/child
  binary subquestions; adaptive leaf methods produce each prediction.

## Scheduling and concurrency

- `.github/workflows/run_bot_on_tournament.yaml` runs every 20 minutes and can
  also be manually dispatched.
- Its concurrency group prevents overlapping copies of that workflow.
- It runs `run_adaptive.py --mode tournament --include-minibench --publish`.
- `.github/workflows/test_adaptive.yaml` is manual and dry-run only.
- Model/search calls are limited to four concurrent provider calls per process.
- Current active defaults remain 120 seconds per call and 600 seconds per
  question; the Phase 1 target in `upgrade_config.py` is not active yet.

## Models, providers and secrets

- Secrets come from `.env` locally and GitHub Actions Secrets remotely.
- Required runtime credentials: `METACULUS_TOKEN` and a key for the configured
  model provider, normally `OPENROUTER_API_KEY`.
- Model roster: `BOT_MODELS`.
- Research route: `BOT_RESEARCHER`.
- Current adaptive controls: `BOT_ENABLE_REVIEW`, `BOT_DISAGREEMENT`,
  `BOT_CALL_TIMEOUT_SECONDS`, `BOT_QUESTION_BUDGET_SECONDS` and
  `BOT_MAX_OUTPUT_TOKENS`. `BOT_ENABLE_DURABLE_LOGGING` defaults to true.
- Research reliability controls include `BOT_RESEARCH_FALLBACK`,
  `BOT_RESEARCH_RETRY_COUNT`, `BOT_RESEARCH_RETRY_DELAY_SECONDS`,
  `BOT_RESEARCH_CACHE_TTL_SECONDS`, `BOT_RESEARCH_BREAKER_FAILURES` and
  `BOT_RESEARCH_BREAKER_SECONDS`.
- Upgrade feature flags are centralized in `upgrade_config.py`; no Phase 3-7
  behaviour is enabled in Phase 0.

## Durable telemetry

- `logs/forecasts.jsonl` receives append-only research, reasoning, final-report
  and failure events.
- `logs/research/<content-hash>.txt` stores each distinct research report once;
  review text is stored under `logs/research/reviews/`.
- Provider calls record model/route, stage, tracked cost, latency and error
  type. Final SDK reports separately record the framework's total estimated
  cost and latency.
- Logging and research persistence are wrapped so filesystem failures cannot
  fail a forecast.
- The adaptive test and production workflows upload `logs/` as private
  GitHub Actions artifacts retained for 90 days. Logs remain excluded from Git.

## Known mismatches and constraints

- The upgrade plan is stored one directory above this bot root.
- This local directory currently has no Git metadata, so the plan's
  phase-by-phase commit workflow is unavailable until source control is
  connected.
- The plan cites forecasting-tools 0.3.1 numeric rules; this project locks
  0.2.92.
- Existing tests use `unittest`; Phase 0 adds pytest as a declared runner while
  retaining those tests.
- GitHub-hosted runners are ephemeral. Phase 1 must select persistent storage
  before forecast logs can support resolution syncing or calibration.
- The current all-analysts-failed policy suppresses submission rather than
  fabricating a fallback. Phase 1 must define and test any replacement policy.
- Forecast submission and private-comment submission are not atomic. Phase 1
  must implement idempotent comment-only repair without re-forecasting.
