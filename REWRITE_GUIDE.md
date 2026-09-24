# Adaptive forecasting rewrite

This is an additive implementation of the proposed strategy, not evidence that it
outperforms the original template. Start with a dry run in the testing area and
compare prospective scores before using it as your prize bot.

## What changed

The original `main.py`, `main_with_no_framework.py`, `bot_helpers.py`, dependency
files, integrations, and original workflows are unchanged. Your zip contained an
identical nested checkout; this package contains one copy, without Git history.
Your original main script remains the rollback baseline.

New files:

- `forecast_logic.py`: deterministic output validation; independent model calls;
  review triggers; one factual-gap search and one revision round when triggered.
- `adaptive_bot.py`: subclass of `SummerTemplateBot2026`; research-provider routing,
  conversion into the existing SDK types, and equal-weight aggregation.
- `run_adaptive.py`: explicit new entry point. Defaults to the testing area with
  publication OFF. Uses the same MetaculusClient and ForecastBot methods.
- `tests/`: offline reasoning tests and integration tests using the actual SDK,
  with model, research and submission calls mocked.
- `.env.adaptive.example`: non-secret strategy settings to add to your existing `.env`.
- `.github/workflows/test_adaptive.yaml`: manual dry-run workflow for the testing area.

## Preserved Metaculus integration

Authentication remains `METACULUS_TOKEN` and the SDK's MetaculusClient.
Question fetching, group expansion, already-forecasted filtering, report creation,
private comment posting, and forecast publishing remain in ForecastBot/the SDK.
No new HTTP implementation or alternative token handling has been introduced.

Binary, multiple-choice, numeric/discrete and date forecasts return the original
SDK prediction types. Conditional dispatch and reaffirmation logic are inherited
from the template; their subquestions use the new forecasting methods.

The new runner uses one event loop. It makes the seasonal tournament explicit
because the supplied main script's banner and SDK defaults can point to an old
season. It does not assume which tournament you entered.

## Setup

1. Copy the new files into the ROOT of your existing repository, alongside main.py.
   Do not create another nested metac-bot-template directory.
2. Keep your existing API keys. No working credentials were supplied in the zip;
   the included `.env.template` contains placeholders only. Do not commit `.env`.
3. Add the settings from `.env.adaptive.example` to `.env`. Set `BOT_MODELS` to
   one, two or three real LiteLLM model IDs available to your account. For OpenRouter
   the format is `openrouter/<provider>/<model>`. Prefer two capable models from
   different providers for the ensemble experiment. The same model twice gives
   independent samples, NOT independent model families.
4. Set `BOT_RESEARCHER` to your configured search provider. Supported routes match
   the template: `asknews/news-summaries`, the template's AskNews deep-research names,
   `smart-searcher/<model>`, or a search-enabled LLM ID. You can leave it unset to
   inherit the SDK's provider selection. An ordinary chat model does not acquire
   web access just because the prompt asks for current news.
5. Install with the existing locked dependencies:

```bash
poetry install --no-interaction --no-root
poetry run python -m unittest discover -s tests -v
poetry run python run_adaptive.py --mode test_questions
```

The last command fetches testing-area questions and uses paid model/search calls,
but DOES NOT submit anything. It saves reports under `logs/adaptive/`.
The unit tests use mocked services and do not spend credits or submit forecasts.

To test the complete posting path in the testing area:

```bash
poetry run python run_adaptive.py --mode test_questions --publish
```

After testing and choosing the tournament, the production command is:

```bash
poetry run python run_adaptive.py --mode tournament --tournament-id YOUR_TOURNAMENT_SLUG --include-minibench --publish
```

Omit `--include-minibench` if you only want the named tournament. To run only
MiniBench, set the tournament ID to `minibench` and omit that flag.

Do not tune or rerun based on previews of your answers to live competition questions.
Use the testing area or other permitted development questions.

## GitHub Actions activation

Your existing scheduled workflows still execute the ORIGINAL bot. They were not
silently switched. The new `test_adaptive.yaml` is manual and cannot publish.

After testing, replace the command in the EXISTING tournament workflow with the
production command above. Add these entries to that step's existing `env` block:

```yaml
BOT_MODELS: ${{ vars.BOT_MODELS }}
BOT_RESEARCHER: ${{ vars.BOT_RESEARCHER }}
BOT_TOURNAMENT_ID: ${{ vars.BOT_TOURNAMENT_ID }}
ASKNEWS_API_KEY: ${{ secrets.ASKNEWS_API_KEY }}
```

Then the command may use `--mode tournament --include-minibench --publish` without
`--tournament-id`, because the runner reads BOT_TOURNAMENT_ID. Configure the three
repository variables under Settings > Secrets and variables > Actions > Variables.
Existing OpenRouter/Metaculus/search secrets remain in place. The ASKNEWS_API_KEY
entry supports newer AskNews credentials; the original ID/secret pair is retained.

Keep one scheduled production runner per bot account. Preserve the existing
workflow concurrency group. Do not run the old and new scripts concurrently.
The inherited already-forecasted filter is not a distributed transaction lock;
this rewrite does not implement cross-process exactly-once submission. Likewise,
a comment can fail after a forecast succeeds: inspect submission state rather
than blindly rerunning to change an answer.

## Forecast logic

1. Supply full criteria, fine print, dates, units, bounds and conditional context.
2. Run two research calls: current facts/resolution sources; historical evidence,
   trends and contradictory evidence. These are different research emphases, but
   can still return overlapping sources. Both reports are shared with analysts.
3. Ask each selected model for an independent JSON forecast. They cannot see one
   another's first forecast. Use baseline/reference-class and causal perspectives.
4. Validate JSON, finite numbers, exact option names, probability totals, quantile
   order and closed bounds; then validate using the actual SDK. Permit one format
   repair. Do not silently sort malformed quantiles or replace failures with 50%.
5. Trigger review on material disagreement (initial heuristic: 20 percentage
   points), near-certainty, numerical disagreement/collapsed intervals, a failed
   research lane/model or an analyst's weak-evidence flag. Agreement alone does
   not prevent review. Thresholds are not learned or proven optimal.
6. A reviewer identifies a consequential unsupported assumption and optionally
   requests ONE factual-gap search. It does not dictate a replacement probability.
7. Each successful analyst sees its own initial forecast plus the audit and may
   revise once. Failed revisions retain the initial forecast. A failed reviewer
   leaves all valid initial forecasts intact. All analyst failures stop submission.
8. Average binary probabilities/option vectors. Mix full numeric/date CDFs on the
   SDK grid, preserving between-model uncertainty rather than averaging medians.
9. Return one ReasonedPrediction to the existing SDK report/submission flow.

The template's binary 1–99% clipping is retained. Multiple-choice SDK clamping and
normalization, and numeric standardization, remain in effect. The raw analyst JSON
in comments is therefore not always identical to the final submitted distribution.
The SDK report summary shows the final distribution. Numeric mixtures are exact
on the declared CDF grid before SDK interpolation/standardization; the SDK can
change probabilities near the bounds when producing its final CDF.

The inherited framework is set to ONE research report and ONE prediction call:
otherwise its default five calls would run five entire ensembles.

## Cost and failure limits

With two analysts and no review: two research-provider calls and two forecast calls.
With review: up to one additional research-provider call, one reviewer call, and
two revision calls. Format repair can add one call per initial/revised forecast:
a maximum of nine analyst/reviewer calls for a two-analyst run. Search providers
may perform multiple underlying searches or retries inside one provider call.
Conditional containers can run the forecasting process for up to four subquestions.

Defaults: four simultaneous provider calls per process; 120 seconds per call;
600 seconds per question for research and reasoning; stop reasoning at least
120 seconds before the supplied question close time. The SDK still owns the final
submission; network latency and external API failures can prevent submission.
There is no guaranteed dollar cap: use your OpenRouter account limit and inspect
actual usage. SDK monetary estimates may not capture every search-provider cost.

The new runner disables the extra research-summary LLM call. Logs retain initial
forecasts, review decisions, final forecasts, models and configuration to support
later comparison.

## What is deliberately not claimed or implemented

- No fitted Bayesian model, learned factor weights, calibration curve, time-series
  data adapter or automatic evaluation dashboard. Bayesian reasoning here is a
  prompting discipline; the model must label subjective updates.
- No contrarian bonus or tournament-payout gaming.
- No claim that search-model citations have been independently verified. Prompts
  request dated sources; dedicated page retrieval and evidence extraction would
  be a separate improvement.
- No independent evidence silo per forecaster. The initial two research lanes are
  shared. Testing separate retrieval per model is a future experiment.
- No guarantee that review improves scores. Disable it with BOT_ENABLE_REVIEW=false
  and compare on the same prospective questions.
- No calibrated statistical uncertainty inferred from model confidence statements.
- Conditional support is inherited; general cross-question probabilistic coherence
  is not enforced by a joint probabilistic model.
- No live API or tournament submission was performed during this rewrite.

## Validation

Tested against forecasting-tools 0.2.92 (the version in your lockfile), on Python
3.12, using real SDK question/report/distribution classes and mocked services.
The tests cover successful and failed model calls, review/revision decisions,
format repair, invalid numeric output, time limits, distribution mixing, all main
question types, conditional dispatch, the inherited submission/comment contract
and previously-forecasted skipping. Use your own testing-area dry run to verify
credentials, model availability and provider search behavior.

Your lockfile is retained. Installing a different SDK version may change default
models, distributions or API behavior; rerun the tests when upgrading.
