# Structured odds-space forecasting with evidence independence groups

Registered: 2026-09-27. The registering commit is the commit that adds this file.

For questions like "will X happen by date Y", the model supplies how often this kind of event happens and which specific facts shift the odds. Code does the maths, scaled to the time left and counting related evidence only once. Recent papers show this structure beats letting the model eyeball a number. The twist being tested, live and pre-registered, is treating related evidence as one piece rather than several.

H1 (primary): on resolved binary deadline and decision questions, p_grouped scores better than the submitted forecast. Mean paired head-to-head points > 0, and the 95% cluster-bootstrap interval excludes 0.

H2: naive multiplication is more overconfident than grouping. Calibration slope of p_naive is below the slope of p_grouped.

H3: a log-odds blend of the submitted forecast and p_grouped beats the submitted forecast.

Sample: the first 60 resolved questions with a valid structured forecast, from 2026-09-27. No look at H1-H3 before then. Checking whether the track produced a number, and how often it failed, is allowed.

Why 60, not 150: the track only runs on deadline and scheduled-decision binaries. A Fall season is 300-400 questions of mixed types. Two archetypes out of six labels, and only on binaries, is unlikely to yield 150 resolved cases inside one season. 60 is the largest target that can still finish in this season. This number is fixed before the first shadow forecast.

Analysis: scripts/analyze_structured_shadow.py, paired and clustered by question.

Exclusions: annulled questions. Questions where the track failed are excluded from H1-H3, and the failure rate is reported.

Decision: if H1's interval excludes 0, switch STRUCTURED_MODE to blend at the weight replay picks. Otherwise stay in shadow and write up the result either way.
