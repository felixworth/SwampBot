"""Offline tests using real forecasting-tools models and mocked model/search calls."""
import json
import os
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, AsyncMock

os.environ['LITELLM_LOCAL_MODEL_COST_MAP'] = 'True'

from adaptive_bot import AdaptiveForecastBot
from forecasting_tools import BinaryQuestion, MultipleChoiceQuestion, NumericQuestion, DateQuestion, ConditionalQuestion
from forecast_logic import QUANTILES


def payload(kind):
    data = {'rationale': 'Offline test evidence.', 'baseline': 'Test baseline',
            'key_uncertainty': 'Test uncertainty', 'needs_review': False}
    if kind == 'binary': data['probability'] = .6
    elif kind == 'multiple_choice': data['probabilities'] = {'A': .6, 'B': .4}
    elif kind == 'numeric': data['quantiles'] = [{'percentile': q, 'value': q*100} for q in QUANTILES]
    else:
        data['quantiles'] = [{'percentile': q, 'value': (datetime(2028, 1, 1, tzinfo=timezone.utc)
                                  + timedelta(days=q*365)).isoformat()} for q in QUANTILES]
    return data


class SDKTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.env = patch.dict(os.environ, {'BOT_MODELS': 'openrouter/openai/gpt-4o,openrouter/openai/gpt-4o',
                                         'BOT_ENABLE_REVIEW': 'false', 'BOT_RESEARCHER': 'no_research'})
        self.env.start()
        self.addCleanup(self.env.stop)

    def bot(self, kind):
        bot = AdaptiveForecastBot(publish_reports_to_metaculus=False,
                                  skip_previously_forecasted_questions=True, extra_metadata_in_explanation=True)
        bot._engine.invoke = AsyncMock(return_value=json.dumps(payload(kind)))
        bot._search = AsyncMock(return_value='Offline source, https://example.org, 2026-09-24')
        return bot

    async def test_real_sdk_binary_pipeline_and_no_publication(self):
        bot = self.bot('binary')
        q = BinaryQuestion(question_text='Test binary?', id_of_post=1, id_of_question=2)
        with patch.object(bot.metaculus_client, 'post_binary_question_prediction') as submit:
            report = await bot.forecast_question(q)
        self.assertAlmostEqual(report.prediction, .6)
        self.assertIn('Initial forecasts', report.explanation)
        submit.assert_not_called()
        self.assertEqual(bot._engine.invoke.await_count, 2)
        self.assertEqual(bot.predictions_per_research_report, 1)
        self.assertEqual(bot._search.await_count, 2)

    async def test_multiple_choice_full_pipeline(self):
        q = MultipleChoiceQuestion(question_text='Which?', options=['A', 'B'])
        report = await self.bot('multiple_choice').forecast_question(q)
        self.assertAlmostEqual(report.prediction.to_dict()['A'], .6)

    async def test_numeric_full_pipeline(self):
        q = NumericQuestion(question_text='How much?', lower_bound=0, upper_bound=100,
                            open_lower_bound=False, open_upper_bound=False, zero_point=None)
        report = await self.bot('numeric').forecast_question(q)
        cdf = report.prediction.get_cdf()
        self.assertEqual(len(cdf), q.cdf_size)
        self.assertTrue(all(a.percentile <= b.percentile for a,b in zip(cdf,cdf[1:])))

    async def test_date_full_pipeline(self):
        q = DateQuestion(question_text='When?', lower_bound=datetime(2028,1,1,tzinfo=timezone.utc),
                         upper_bound=datetime(2029,1,1,tzinfo=timezone.utc),
                         open_lower_bound=False, open_upper_bound=False, zero_point=None)
        report = await self.bot('date').forecast_question(q)
        self.assertTrue(report.prediction.is_date)

    async def test_previously_forecasted_is_skipped(self):
        bot = self.bot('binary')
        q = BinaryQuestion(question_text='Already done?', already_forecasted=True)
        self.assertEqual(await bot.forecast_questions([q]), [])
        bot._engine.invoke.assert_not_called()

    async def test_near_close_rejected_without_model_calls(self):
        bot = self.bot('binary')
        q = BinaryQuestion(question_text='Too late?', close_time=datetime.now(timezone.utc)+timedelta(seconds=30))
        result = await bot.forecast_questions([q], return_exceptions=True)
        self.assertIsInstance(result[0], BaseException)
        bot._engine.invoke.assert_not_called()

    async def test_original_submission_and_private_comment_contract(self):
        bot = self.bot('binary')
        bot.publish_reports_to_metaculus = True
        q = BinaryQuestion(question_text='Mock publication?', id_of_post=7, id_of_question=9)
        with patch.object(bot.metaculus_client, 'post_binary_question_prediction') as submit, \
             patch.object(bot.metaculus_client, 'post_question_comment') as comment:
            await bot.forecast_question(q)
        submit.assert_called_once_with(9, .6)
        comment.assert_called_once()
        self.assertEqual(comment.call_args.args[0], 7)
        # Unchanged SDK method defaults to private=True.
        import inspect
        from forecasting_tools import MetaculusClient
        self.assertTrue(inspect.signature(MetaculusClient.post_question_comment).parameters['is_private'].default)

    async def test_conditional_pipeline_retained(self):
        bot = self.bot('binary')
        q = ConditionalQuestion(question_text='Conditional?',
            parent=BinaryQuestion(question_text='Parent?'),
            child=BinaryQuestion(question_text='Child?'),
            question_yes=BinaryQuestion(question_text='Child if parent yes?', conditional_type='yes'),
            question_no=BinaryQuestion(question_text='Child if parent no?', conditional_type='no'))
        report = await bot.forecast_question(q)
        self.assertAlmostEqual(report.prediction.prediction_yes, .6)
        self.assertAlmostEqual(report.prediction.prediction_no, .6)

    def test_numeric_mixture_matches_average_cdf(self):
        bot = self.bot('numeric')
        q = NumericQuestion(question_text='How much?', lower_bound=0, upper_bound=100,
                            open_lower_bound=False, open_upper_bound=False, zero_point=None)
        a, b = payload('numeric'), payload('numeric')
        for point in a['quantiles']: point['value'] *= .6
        for point in b['quantiles']: point['value'] = 40 + .6*point['value']
        ca, cb = bot._convert(a,q).get_cdf(), bot._convert(b,q).get_cdf()
        # The SDK re-interpolates/standardizes near bounds on get_cdf().
        # Assert the intended mixture on its declared grid before that step.
        distribution = bot._combine([a,b],q)
        mixed = distribution.declared_percentiles
        self.assertEqual(len(distribution.get_cdf()), q.cdf_size)
        for x,y,z in zip(ca,cb,mixed):
            self.assertAlmostEqual(z.percentile, (x.percentile+y.percentile)/2, places=3)

    def test_original_metaculus_entry_points_are_inherited(self):
        from main import SummerTemplateBot2026
        for method in ('forecast_on_tournament', 'forecast_questions', '_run_individual_question',
                       '_run_forecast_on_conditional'):
            self.assertIs(getattr(AdaptiveForecastBot, method), getattr(SummerTemplateBot2026, method))


if __name__ == '__main__':
    unittest.main()
