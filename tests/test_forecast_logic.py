import asyncio
import json
import time
import unittest

from forecast_logic import ForecastEngine, QUANTILES, review_reasons, validate_forecast

SPEC = {'kind': 'binary', 'question_text': 'Will X happen?'}


def candidate(p=.6, review=False):
    return {'probability': p, 'rationale': 'Dated source supports the event.',
            'baseline': 'Reference class', 'key_uncertainty': 'Remaining prerequisite',
            'needs_review': review}


class ValidationTests(unittest.TestCase):
    def test_rejects_percentage_scale_nan_bool_and_missing_fields(self):
        for value in (60, float('nan'), True, '0.6'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                validate_forecast(json.dumps(candidate(value)), SPEC)
        with self.assertRaises(ValueError):
            validate_forecast('{"probability": 0.6}', SPEC)

    def test_rejects_duplicate_json_keys(self):
        with self.assertRaises(ValueError):
            validate_forecast('{"probability": 0.4, "probability": 0.6}', SPEC)

    def test_multiple_choice_exact_options_and_normalization(self):
        spec = {'kind': 'multiple_choice', 'options': ['A', 'B']}
        value = candidate()
        value['probabilities'] = {'A': .333, 'B': .6665}
        result = validate_forecast(json.dumps(value), spec)
        self.assertAlmostEqual(sum(result['probabilities'].values()), 1)
        for bad in ({'A': .5}, {'A': .5, 'B': .4}, {'A': .5, 'C': .5}):
            value['probabilities'] = bad
            with self.assertRaises(ValueError):
                validate_forecast(json.dumps(value), spec)

    def test_numeric_order_units_bounds(self):
        spec = {'kind': 'numeric', 'lower_bound': 0, 'upper_bound': 100,
                'open_lower_bound': False, 'open_upper_bound': False}
        value = candidate()
        value['quantiles'] = [{'percentile': q, 'value': q*100} for q in QUANTILES]
        validate_forecast(json.dumps(value), spec)
        value['quantiles'][4]['value'] = 99
        with self.assertRaises(ValueError):
            validate_forecast(json.dumps(value), spec)
        value['quantiles'][4]['value'] = 40
        value['quantiles'][-1]['value'] = 101
        with self.assertRaises(ValueError):
            validate_forecast(json.dumps(value), spec)
        spec['open_upper_bound'] = True
        validate_forecast(json.dumps(value), spec)

    def test_agreement_is_not_sufficient_to_skip_review(self):
        self.assertFalse(review_reasons([candidate(.97), candidate(.97)], SPEC, 'sources', .2))
        self.assertTrue(review_reasons([candidate(.97), candidate(.40)], SPEC, 'sources', .2))
        self.assertTrue(review_reasons([candidate(.6), candidate(.6)], SPEC, 'RESEARCH_FAILED', .2))
        self.assertFalse(review_reasons([candidate(.6), candidate(.65)], SPEC, 'sources', .2))


class EngineTests(unittest.IsolatedAsyncioTestCase):
    async def test_independent_first_pass_and_no_unnecessary_review(self):
        calls = []
        async def invoke(i, prompt):
            calls.append(prompt)
            self.assertNotIn('Your own original forecast:', prompt)
            return json.dumps(candidate(.6 + i*.05))
        async def search(prompt):
            self.fail('Unexpected follow-up research')
        result = await ForecastEngine(invoke, search).run(SPEC, 'sources', time.monotonic()+20)
        self.assertEqual(len(calls), 2)
        self.assertEqual(len(result.candidates), 2)

    async def test_failed_analyst_does_not_discard_valid_forecast(self):
        async def invoke(i, prompt):
            if i == 1: raise RuntimeError('provider failure')
            return json.dumps(candidate())
        async def search(prompt): return 'sources'
        result = await ForecastEngine(invoke, search, review_enabled=False).run(SPEC, 'sources', time.monotonic()+20)
        self.assertEqual(len(result.candidates), 1)
        self.assertTrue(any('failed' in n for n in result.notes))

    async def test_all_failed_raises_instead_of_inventing_probability(self):
        async def invoke(i, prompt): raise RuntimeError('provider failure')
        async def search(prompt): return ''
        with self.assertRaises(RuntimeError):
            await ForecastEngine(invoke, search).run(SPEC, 'sources', time.monotonic()+20)

    async def test_one_format_repair_only(self):
        calls = 0
        async def invoke(i, prompt):
            nonlocal calls
            calls += 1
            return 'not json' if calls == 1 else json.dumps(candidate())
        async def search(prompt): return ''
        result = await ForecastEngine(invoke, search, analyst_count=1, review_enabled=False).run(
            SPEC, 'sources', time.monotonic()+20)
        self.assertEqual(calls, 2)
        self.assertEqual(result.candidates[0]['probability'], .6)

    async def test_review_gap_search_and_independent_revision(self):
        searches = []
        async def invoke(i, prompt):
            if 'Audit these forecasts' in prompt:
                return json.dumps({'critique': 'Check prerequisite.', 'search_query': 'Has prerequisite occurred?'})
            if 'Your own original forecast:' in prompt:
                return json.dumps(candidate(.65 + i*.05))
            return json.dumps(candidate(.3 + i*.5))
        async def search(prompt):
            searches.append(prompt)
            return 'Primary source confirms prerequisite; dated URL.'
        result = await ForecastEngine(invoke, search).run(SPEC, 'sources', time.monotonic()+120)
        self.assertEqual(len(searches), 1)
        self.assertEqual(result.initial_candidates[0]['probability'], .3)
        self.assertEqual(result.candidates[0]['probability'], .65)
        self.assertTrue(result.review)

    async def test_failed_review_keeps_originals(self):
        async def invoke(i, prompt):
            if 'Audit these forecasts' in prompt: raise RuntimeError('review unavailable')
            return json.dumps(candidate(.3 + .5*i))
        async def search(prompt): return ''
        result = await ForecastEngine(invoke, search).run(SPEC, 'sources', time.monotonic()+120)
        self.assertEqual(result.candidates, result.initial_candidates)
        self.assertIn('Review failed', ' '.join(result.notes))

    async def test_timeout_and_expired_budget(self):
        async def invoke(i, prompt):
            await asyncio.sleep(.05)
            return json.dumps(candidate())
        async def search(prompt): return ''
        engine = ForecastEngine(invoke, search, call_timeout=.005)
        with self.assertRaises(RuntimeError):
            await engine.run(SPEC, 'sources', time.monotonic()+1)
        with self.assertRaises(RuntimeError):
            await engine.run(SPEC, 'sources', time.monotonic()-1)


if __name__ == '__main__':
    unittest.main()
