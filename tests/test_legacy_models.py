import os
from unittest.mock import patch

import pytest

from bot_helpers import reject_legacy_models


def test_reject_legacy_models_blocks_gpt4o():
    with pytest.raises(ValueError, match="gpt-4o"):
        reject_legacy_models("openrouter/openai/gpt-4o")


def test_reject_legacy_models_allows_frontier_mix():
    reject_legacy_models(
        "openrouter/openai/gpt-6-sol-pro,openrouter/anthropic/claude-opus-5.5"
    )


def test_check_environment_rejects_gpt4o_when_set():
    from bot_helpers import check_environment

    with patch.dict(os.environ, {
        "BOT_MODELS": "openrouter/openai/gpt-4o",
        "METACULUS_TOKEN": "not-a-placeholder-token",
        "OPENROUTER_API_KEY": "sk-or-test",
    }):
        with pytest.raises(SystemExit):
            check_environment(strict=True)
