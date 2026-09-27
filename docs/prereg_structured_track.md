# Structured odds-space forecasting with evidence independence groups: a pilot

Registered: 2026-09-27. The registering commit is the commit that adds this revised pre-registration, before the study's confirmatory sample begins.

For questions like "will X happen by date Y", the model supplies how often this kind of event happens and which specific facts shift the odds. Code does the maths, scaled to the time left and counting related evidence only once. Recent papers show this structure beats letting the model eyeball a number. The twist being tested, live and pre-registered, is treating related evidence as one piece rather than several.

This season is a pilot. At 60 resolved questions, using the spread from the synthetic check, a 95% interval is about ±8 head-to-head points per question. The structured track would need to beat the comparator by about 11 points per question to be detected reliably. That is larger than whole scaffolds usually add, so a head-to-head at 60 is likely to be inconclusive even if the idea works. The calibration slope is the more sensitive measure, and it is the new part of the study, so it is the primary test at 60.

## Pilot, first 60 resolved questions

P1 (primary): naive multiplication is more overconfident than grouping. Calibration slope of p_naive is below the slope of p_grouped.

P2 (secondary): p_grouped scores better than the after_review forecast. Mean paired head-to-head points > 0, and the 95% cluster-bootstrap interval excludes 0. This is reported, and a null result here is not treated as the idea failing.

P3 (secondary): a log-odds blend of the after_review forecast and p_grouped beats the after_review forecast.

Comparator: the after_review stage, not the final submitted forecast. Weaker shrinkage and source-verified caps, planned later, change the submitted number and do not change after_review. Each forecast also stores bot_version, so any later analysis can be split by version.

## Confirmatory continuation, first 150 resolved questions

The same three tests, run once, on the first 150 resolved questions with a valid structured forecast. That sample will run into the next season. No peeking at these confirmatory tests before 150. The 60-question pilot may be looked at when it completes; that look does not change the 150-question rules.

C1 (primary at 150): the head-to-head of p_grouped against after_review. Mean paired points > 0, and the 95% cluster-bootstrap interval excludes 0.

C2: calibration slope of p_naive is below the slope of p_grouped.

C3: a log-odds blend of after_review and p_grouped beats after_review.

Decision, only at 150: if C1's interval excludes 0, switch STRUCTURED_MODE to blend at the weight replay picks. Otherwise stay in shadow and write up the result either way. The pilot does not authorize that switch.

Sample clock: from 2026-09-27. Checking whether the track produced a number, and how often it failed, is allowed at any time. Looking at P1–P3 outcomes before 60, or C1–C3 outcomes before 150, is not.

Analysis: paired, clustered by question.

Exclusions: annulled questions. Questions where the track failed are excluded from the tests, and the failure rate is reported.
