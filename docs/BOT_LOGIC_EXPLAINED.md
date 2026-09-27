# How the bot thinks, and what's different about it

This describes the code on `main` as of commit `b1da270`. For each stage it covers:
- **what the bot does**
- **why it does it that way**
- **how it differs from a typical AIB tournament bot**

At the end, one section is an honest list of what is built but not yet switched on, or not
yet doing anything in practice.

A *typical* AIB bot (the template this repo started from) works like this:
1. Ask one news tool for a summary.
2. Paste the summary into 3–5 prompts.
3. Take the median.

This bot keeps that overall shape, but adds structure at each step, aimed at the two things
that most often lose log-score points:
- misreading how a question resolves
- being confidently wrong

---

## The pipeline in one picture

```
question
  │
  ├─ 1. Resolution spec          what exactly makes this YES? which source decides?
  ├─ 2. Read the resolution source directly
  ├─ 3. Cruxes                   the 3–5 unknowns that would move the forecast most
  ├─ 4. Crux searches            AskNews and a web-search LLM, alternating
  ├─ 5. Claims ledger            facts tagged by independence group (syndication de-duplicated)
  ├─ 6. Outside view             similar questions the bot already saw resolve
  │
  ├─ 7. Independent forecasts    different models × different reasoning roles
  ├─ 8. Strict validation        malformed output is repaired for format, never for judgment
  ├─ 9. Review only if triggered disagreement / extreme / flagged → blind audit → 1 search → revise
  │
  ├─ 10. Aggregate               binary: log-odds shrinkage · MC: floor · numeric: CDF mixture
  ├─ 11. Anchor                  blend with a matched, liquid Manifold market if one exists
  ├─ 12. Cap                     2–98% (3–97% MiniBench), derived from log-score break-even
  │
  └─ log every member forecast → sync outcomes → replay alternatives offline → fit calibration
```

---

## 1–2. Resolution first, news second

**What:**
- Before any news search, a cheap model call turns the question into a structured
  *resolution spec*. It covers:
  - what counts as YES, with the criteria quoted where precision matters
  - the deadline in UTC, units, thresholds and bounds
  - the named resolution sources, with URLs
  - whether the event has already happened
  - which conditions still have to be met
  - any ambiguities (listed, not guessed)
- The bot then fetches those source URLs directly, all at once. It keeps the sentences that
  carry numbers or substance, and marks a page `needs_search` if too little came back.

**Why:**
- Many bot losses come from answering a slightly different question than the one asked.
  Examples: the submission deadline mistaken for the event deadline, "announced" confused
  with "completed", or the wrong statistical series.
- Many questions are resolved by one specific page: a government statistics table, a
  leaderboard, an official register. What that page says today is often the single most
  informative fact, and news summaries tend to paraphrase it or miss it.

**What's different:**
- Most bots start with "search the news about X". This one works out the rules first, then
  reads the scoreboard, and only then looks at commentary.
- The spec is also passed to later steps: the cruxes, the forecasters' evidence and the
  structured track all see the same parsed criteria.

## 3–4. Crux-driven search instead of topic search

**What:**
- A model lists the 3–5 unknowns whose answers would move the forecast most.
- Each crux gets a precise question, why it matters, and 1–2 search queries.
- The queries run through two providers in turn:
  - **AskNews**, a news index
  - **a web-search LLM** (`gpt-6-sol:online`)
- Each finding must contain substantive text and at least one source URL, or it doesn't count.

**Why:**
- Searching for the topic returns whatever is most written about, which isn't necessarily
  what decides the question.
- Searching for *cruxes* spends the budget on the facts that matter for the forecast.
- Using two providers with different retrieval systems means one provider's blind spot or
  stale index is less likely to become the whole evidence base.

**What's different:**
- The research is organised around decisions: each crux is listed with its findings.
- Any crux left unanswered, and every ambiguity in the resolution spec, is surfaced
  explicitly in an "Open uncertainties" section. The forecasters see what is *not* known,
  not just what is.

## 5. Claims ledger with independence groups

**What:** A model extracts individual factual claims from the findings. Each claim is tagged
with:
- source URL
- published date and event date
- which cruxes it bears on
- direction (for or against)
- an **independence group**: reports of the *same underlying event or announcement* share a
  group and are merged, with a note like "4 reports of the same event".

**Why:**
- Syndicated news is the most common way LLM forecasters double-count evidence. One press
  release, republished by 12 outlets, *looks* like 12 confirmations.
- Tagging independence makes that explicit before the forecasters see it.

**What's different:** Few bots model evidence correlation at all. The same independence
groups feed the structured evidence update (stage 10b), where only the strongest item in
each group counts.

## 6. Outside view from the bot's own history

**What:**
- Every run's log is saved to a private data repo.
- The research step looks up past questions with similar titles (word overlap of at least
  25%) that have since resolved.
- It shows what the bot forecast on them and what actually happened.

**Why:** Past questions from the same series are among the best base rates available. A
recurring series ("Will X exceed Y in month Z?") is common on Metaculus, and the bot's own
record on the previous instances is directly relevant.

**What's different:** Most bots forget everything between runs. This one keeps a memory of
how its forecasts turned out and feeds it back into research.

---

## 7. Independent forecasts with role diversity

**What:**
- Each model in `BOT_MODELS` forecasts independently and at the same time. The models come
  from different providers (for example GPT-6 Sol Pro and Claude Opus 5.5).
- Each forecaster is given a different **reasoning role**:
  1. reference class and base rate first
  2. causal mechanisms and bottlenecks first
  3. alternative scenarios and overlooked evidence first
- Roles **rotate by question ID**, so a given model doesn't always take the same role.
- The shared prompt tells each forecaster to:
  - optimise for log score
  - use the event horizon in the criteria, not the submission deadline
  - keep sourced facts separate from judgment
  - not invent likelihood ratios or multiply correlated evidence
  - not fall back to 50% when information is missing
  - not force agreement or contrarianism
  - give *predictive* intervals for numeric questions, not confidence intervals
  - raise a `needs_review` flag when decisive facts conflict or rest on one weak source

**Why:**
- An ensemble only helps if its members make *different* mistakes.
- Two frontier models given the same prompt tend to fail the same way.
- Different providers plus different starting frames decorrelates errors more cheaply than
  adding more samples.
- Rotation stops a model's quirks from being tied to one role.

**What's different:** Most ensembles are the same prompt run N times or on N models. Giving
each forecaster a deliberately different reasoning frame is uncommon.

## 8. Strict validation, format-only repair, no invented forecasts

**What:**
- Output must be one JSON object with a rationale, baseline, key uncertainty, a
  `needs_review` flag and the forecast.
- The validator rejects:
  - duplicate keys
  - non-finite numbers
  - booleans in number fields
  - option names that don't match exactly
  - option probabilities that don't sum to 1
  - quantiles out of order
  - values outside closed bounds
- The forecast must also pass the SDK's own checks.
- A failed forecast gets **one** repair call, told to fix the *format and validity, not the
  judgment*.
- If every forecaster fails, one simpler single-model fallback runs. If that fails too, the
  bot **submits nothing**; it never makes up a number.

**Why:**
- Parsing errors silently turn into bad forecasts in many bots. For example, a regex grabs
  the wrong percentage, or sorted quantiles hide a model's confusion.
- Refusing to guess keeps the scored record clean.

## 9. Review only when it's needed, and the reviewer can't see the numbers

**What:** A review round runs only when something warrants it:
- the models disagree by 0.20 or more
- any forecast is at 5% or below, or 95% or above
- a model raised `needs_review`
- research failed
- the ensemble only partly succeeded
- numeric forecasts disagree, or one has an interval of zero width

When it runs:
1. **A blind auditor.** The reviewer sees each forecaster's rationale, baseline and key
   uncertainty, but **not their probabilities**. It must name the most important factual
   disagreement, or an unsupported assumption they share, and propose one search query.
   It is told not to give a probability and not to invent dissent.
2. **One targeted search** for that gap.
3. **Independent revision.** Each forecaster sees only its *own* original forecast, plus the
   audit and the new evidence. It may revise once, and is explicitly told it need not agree.

**Why:**
- Debate rounds between LLMs tend to produce *convergence*, not accuracy. Models anchor on
  each other's numbers and drift to the middle.
- This design keeps the useful part of debate: finding the factual crux and getting evidence
  on it.
- It avoids the harmful part: showing forecasters each other's numbers, and letting the
  critic set the answer.
- Triggering only when needed spends extra compute where the evidence is contested or the
  stakes are highest (extreme forecasts carry the biggest log-score losses).

**What's different:** Adaptive compute tied to disagreement, plus a reviewer who can't see
the numbers, is the most distinctive part of this bot's reasoning loop.

---

## 10. Aggregation: each question type gets its own treatment

### Binary: log-odds shrinkage in place of a plain median

**What:**
- The forecasts are averaged **in log-odds**, then pulled toward a prior of **35%** (roughly
  how often binary tournament questions resolve YES).
- How far they're pulled depends on how much the models disagree:

```
weight on the models = τ² / (τ² + s²/K)        τ = 2, s² = variance of the member logits
```

- Models that **agree** keep essentially their own average. For example, 88% and 92% give a
  weight of about 0.99.
- Models that **disagree** are pulled toward the base rate. For example, with 50% and 90%
  the weight is about 0.77, giving about **67%** instead of a plain median of 70%.
- Samples are averaged within each model first, so a model with more samples doesn't
  dominate.

**Why:** Disagreement is evidence that the question is hard, and on hard questions the base
rate is a better anchor than the midpoint of two guesses. This is a normal-normal
hierarchical model: the same maths as empirical-Bayes shrinkage in statistics, applied to
forecaster disagreement.

**What's different:** Almost all bots use a plain median or mean. Shrinking *more when the
forecasters disagree* is a principled step most don't take.

### 10b. Structured evidence update (code does the arithmetic)

**What (when active):** For questions the resolution spec classifies as a *deadline event*
or a *scheduled decision*, a baseline is computed **in code**:
- **Deadline hazard:** a Gamma-Poisson model. It takes a reference-class count (k events in
  T years) and the time remaining, and gives the probability of at least one event before
  the deadline. A half-event prior keeps zero-count classes above 0%, and the model's rate
  adjustment is clamped to between ⅓× and 3×.
- **Scenario tree:** the law of total probability over mutually exclusive scenarios. The
  code rejects scenario sets that don't sum to 1.

A model then lists situational factors as **odds multipliers** (between 0.25 and 4), each
tagged with an independence group. The code then:
- counts only the strongest factor in each group
- clips each factor at ×4
- clips the total shift at ×20
- blends the result 50/50 with the ensemble in log-odds

**Why:**
- LLMs are good at choosing a reference class and naming factors, and bad at arithmetic:
  they multiply correlated evidence and ignore how probability scales with time remaining.
- Here the LLM supplies the inputs and the code does the Bayesian maths.

**What's different:** This follows recent published work (the AuditForecast and LEAP
papers). Very few tournament bots separate judgment from calculation this way.

### Multiple choice: floor only the tiny options

**What:**
- Options are averaged per option.
- Any option below 1% is raised to exactly 1%, and that mass is taken proportionally from
  the options above it.

**Why:**
- The usual approach mixes in a uniform distribution, which takes mass from the favourite
  even when only one tail option needed lifting.
- The SDK also clamps options to 1% and rejects moves of more than 5%. Pre-flooring this
  way means the SDK changes nothing.

### Numeric and date: 11 quantiles and a true mixture

**What:**
- Each forecaster gives **11 quantiles, including the 1st and 99th percentiles**. It's told
  those should be values it would be surprised to see crossed 1 time in 100, not the
  question bounds.
- The bot builds each forecaster's CDF itself:
  - a smooth, monotone curve through the quantiles
  - tails extrapolated smoothly beyond them
  - handled on a log grid for log-scaled questions
  - with Metaculus's bound rules and per-step limits enforced, spilling any excess into
    neighbouring bins so a sharp forecast stays sharp
- The CDFs are then **mixed**: averaged point by point, not by averaging the quantiles.
- If the mixed curve fails validation, the SDK's own mixture is used instead.

**Why:**
- Averaging quantiles makes the result *narrower* than any member when the members disagree.
  That is exactly when it should be wider.
- A mixture keeps the disagreement as extra spread, and on numeric questions the tails are
  where log score is won or lost.
- Asking for 1st and 99th percentiles means the tails come from the model's judgment, not
  from an extrapolation formula.

**What's different:** Most bots ask for 5–7 percentiles and let the SDK interpolate. Very
few check their own curve against Metaculus's step rules before submitting.

## 11. Market anchor

**What:**
- For binary questions, the bot searches Manifold for a matching market.
- It accepts one only if:
  - the market is binary with a price between 0 and 1
  - the titles overlap by at least 45%
  - the market closes within 7 days of the question
  - the market has traded more than 1,000 in volume
- If one matches, the forecast is blended with the market price in log-odds, with the
  market at **30%**.

**Why:** A liquid, well-matched market is an independent, crowd-aggregated forecast. Even a
small weight on it tends to reduce large errors.

**What's different:** The strict match rules matter. A *wrong* match (a similar title with a
different deadline or criteria) is worse than none, so the bot would rather skip than guess.

## 12. Caps worked out from log-score break-even

**What:**
- Final binary forecasts are clipped to **2–98%**, or **3–97% on MiniBench**, whose
  questions are resolved by an LLM and so have more resolver error.
- There is code to widen this to 0.5–99.5% when *both* of these hold:
  - the resolution source itself shows the outcome
  - every forecaster is on that extreme side
- The widening is justified by a break-even calculation: going from 2% to 0.5% gains about
  1.5 points when right and loses about 139 when wrong. So it pays only if the true chance
  of being wrong is below about 1.1%.

**Why:** The caps stand for the chance that the *whole pipeline* is wrong: criteria misread,
stale news, or a resolver mistake. Calibration on past data can't see that. So caps are
applied last, after calibration.

---

## The learning loop

**What:**
- Every forecast is logged, including **every forecaster's individual number before and
  after review**, plus per-call cost, latency, errors and hashes of the research text.
- Logs are saved to a private data repo after each run.
- `sync_resolutions.py` fills in outcomes from Metaculus.
- `replay_and_fit.py` then **replays different post-processing pipelines** on questions that
  have resolved, **with no new LLM calls**. It compares:
  - median vs. mean vs. log-odds mean
  - before vs. after review
  - with or without caps and calibration
- The comparison splits by **time** (fit on older questions, score on newer ones) and uses a
  **paired bootstrap** on head-to-head scores. It never compares two leaderboard totals.
- Platt calibration is fitted only after **100 or more resolved binary questions**:
  - the fit is shrunk toward "no change", with the strength chosen by cross-validation
  - it is applied only if it beats "no change" on held-out data
- The code also allows a separate calibration offset per tournament (seasonal, MiniBench,
  Cup). Small sources borrow strength from the overall offset.

**Why:** Most choices in a forecasting pipeline are made on intuition, such as median vs.
mean or whether review helps. Logging each forecaster's number turns those into questions
you can test for free on your own resolved data, with proper paired statistics.

**What's different:** This is the part most bots lack entirely. They can't measure whether a
change helped, only whether the leaderboard moved, which is mostly noise over a season.

---

## Engineering choices that protect the score

- **Dry run by default.** Publishing needs an explicit `--publish`.
- **Skip after repeated failures.** A question that has failed twice is skipped, so a broken
  question can't burn money every 20 minutes.
- **Research reliability:**
  - 1 retry, then the fallback provider
  - a circuit breaker after repeated failures
  - a 1-hour cache of successful searches
  - failures written as explicit `RESEARCH_FAILED` markers, never silently dropped
- **Spend floor.** A run stops if the OpenRouter key's remaining limit is below $1.
- **Comment repair.** A missing private comment is re-posted from the log without
  forecasting again.
- **Retrieved text is treated as untrusted.** Every prompt marks the question and evidence as
  data, not instructions, as a guard against prompt injection from web pages.

---

## Honest status: built vs. actually working today

| Component | Status |
|---|---|
| Resolution spec, source fetch, cruxes, claims | **Live** |
| Role-diverse ensemble, validation, blind review | **Live** |
| Log-odds shrinkage for binary aggregation | **Live**. τ and the 35% prior are guesses until fitted from logs |
| MC floor, numeric 11-quantile CDF mixture | **Live** |
| Market anchor | **Live** (fixed in `b1da270` to work without the structured track) |
| Outside view from own history | **Live**, but empty until resolved questions build up in the data repo |
| Structured evidence update | **Effectively off.** It only runs when the spec includes `reference_class` counts or `scenarios`, and the spec prompt never asks for them |
| Wider caps when the outcome is verified at the source | **Not connected.** `conditional_cap` is never told that the outcome was verified at the source, so it always applies 2–98% |
| Calibration | **Identity until 100+ resolved binary questions** |
| Separate calibration per tournament, empirical numeric baselines, tail widening | **Written and tested, not called** |
| Ensemble v2 (more samples per model), log-odds disagreement triggers | **Flags off** |

**Cost as it last ran:** about $2.10 per question, itemized. About half of that is the five
web-search LLM calls, and about a quarter is the review round.

---

## In one paragraph

This bot treats forecasting as a pipeline of checks rather than one clever prompt:
1. It pins down exactly how the question resolves before reading any news.
2. It searches for the facts that would change the answer, not the topic.
3. It deduplicates correlated evidence.
4. It gets independent views from different models with different reasoning frames.
5. It spends extra effort only where they disagree or the stakes are extreme, using a
   reviewer who can't see their numbers.
6. It combines forecasts with methods that widen uncertainty when forecasters disagree.
7. It caps outputs using explicit log-score maths.
8. It logs everything so that each of these choices can be tested later against its own
   resolved record.
