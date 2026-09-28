from unittest.mock import patch

import pytest

from grasp.configs import ModelConfig
from grasp.model.base import Model, Response, ToolCall


class _FakeStatusError(Exception):
    def __init__(self, status_code: int):
        super().__init__(f"status {status_code}")
        self.status_code = status_code


class _ScriptedModel(Model):
    """Raises the given exceptions in order, then returns a fixed response."""

    def __init__(self, config: ModelConfig, side_effects: list[Exception]):
        super().__init__(config)
        self.side_effects = list(side_effects)
        self.calls = 0

    def call(self, messages, fns, config=None) -> Response:
        self.calls += 1
        if self.side_effects:
            raise self.side_effects.pop(0)
        return Response(id="resp", message="done", tool_calls=[])


class TestModelRateLimitRetry:
    # regression: a fast, non-agentic model can fire off enough requests to
    # trip a shared backend's per-minute rate limit; the SDK client's own
    # num_retries (meant for brief connection blips) wasn't enough to ride
    # out a limit that stays in effect for tens of seconds - so Model.
    # __call__ retries specifically on HTTP 429 with a longer, configurable
    # backoff instead of failing the whole record immediately.

    def test_retries_on_429_then_succeeds(self):
        config = ModelConfig(rate_limit_max_retries=3, rate_limit_backoff_seconds=0.01)
        model = _ScriptedModel(config, [_FakeStatusError(429), _FakeStatusError(429)])
        response = model([], [])
        assert response.message == "done"
        assert model.calls == 3

    def test_gives_up_after_max_retries(self):
        config = ModelConfig(rate_limit_max_retries=2, rate_limit_backoff_seconds=0.01)
        model = _ScriptedModel(config, [_FakeStatusError(429)] * 5)
        with pytest.raises(_FakeStatusError):
            model([], [])
        assert model.calls == 3  # initial attempt + 2 retries

    def test_does_not_retry_non_429_errors(self):
        config = ModelConfig(rate_limit_max_retries=3, rate_limit_backoff_seconds=0.01)
        model = _ScriptedModel(config, [_FakeStatusError(500)])
        with pytest.raises(_FakeStatusError):
            model([], [])
        assert model.calls == 1

    def test_sleeps_the_configured_backoff(self):
        config = ModelConfig(rate_limit_max_retries=1, rate_limit_backoff_seconds=42.0)
        model = _ScriptedModel(config, [_FakeStatusError(429)])
        with patch("grasp.model.base.time.sleep") as mock_sleep:
            model([], [])
        mock_sleep.assert_called_once_with(42.0)


class TestResponseHash:
    # regression: Response.hash() used to mix free-text commentary into the
    # same hash as the tool calls, so core.generate()'s loop detection missed
    # a model repeating the exact same (often broken) tool call turn after
    # turn as long as its accompanying explanation text varied - which wasted
    # real time/tokens in a real run (see e.g. a qwen9b conversation that
    # repeated an identical malformed SPARQL query 7 times before
    # self-correcting, undetected, eventually contributing to a timeout).

    def _tool_call_response(self, message: str | None) -> Response:
        return Response(
            id="resp",
            message=message,
            tool_calls=[
                ToolCall(id="tc1", name="search", args={"query": "held in"}),
            ],
        )

    def test_same_tool_call_same_hash_despite_different_text(self):
        a = self._tool_call_response("Let me check the held-in relation.")
        b = self._tool_call_response("I'll now verify whether this holds.")
        assert a.hash() == b.hash()

    def test_different_tool_call_args_different_hash(self):
        a = self._tool_call_response("same text")
        b = Response(
            id="resp",
            message="same text",
            tool_calls=[
                ToolCall(id="tc1", name="search", args={"query": "written by"}),
            ],
        )
        assert a.hash() != b.hash()

    def test_different_tool_call_order_same_hash(self):
        # order of tool calls within a turn should not matter for detecting
        # a repeat of the same set of calls
        a = Response(
            id="resp",
            tool_calls=[
                ToolCall(id="tc1", name="search", args={"query": "x"}),
                ToolCall(id="tc2", name="search", args={"query": "y"}),
            ],
        )
        b = Response(
            id="resp",
            tool_calls=[
                ToolCall(id="tc2", name="search", args={"query": "y"}),
                ToolCall(id="tc1", name="search", args={"query": "x"}),
            ],
        )
        assert a.hash() == b.hash()

    def test_no_tool_calls_falls_back_to_message_text(self):
        a = Response(id="resp", message="Done.", tool_calls=[])
        b = Response(id="resp", message="Done.", tool_calls=[])
        c = Response(id="resp", message="Still working.", tool_calls=[])
        assert a.hash() == b.hash()
        assert a.hash() != c.hash()
