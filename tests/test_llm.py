from types import SimpleNamespace

from anthropic.types.beta import BetaMessage, BetaTextBlock, BetaThinkingBlock, BetaToolUseBlock, BetaUsage

from toolagent.llm import ClaudeClient


class FakeMessages:
    def __init__(self, response):
        self.response = response
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return self.response


def test_claude_client_request_and_response_shape():
    response = BetaMessage.model_construct(
        id="msg_1", type="message", role="assistant", model="claude-opus-5-5", stop_reason="tool_use",
        content=[
            BetaThinkingBlock(type="thinking", thinking="", signature="sig"),
            BetaTextBlock(type="text", text="Computing."),
            BetaToolUseBlock(type="tool_use", id="toolu_1", name="calculator", input={"expression": "1+1"}),
        ],
        usage=BetaUsage.model_construct(input_tokens=120, output_tokens=30, cache_read_input_tokens=None),
    )
    messages = FakeMessages(response)
    client = ClaudeClient(SimpleNamespace(beta=SimpleNamespace(messages=messages)))
    tools = [{"name": "calculator", "description": "d", "input_schema": {"type": "object"}, "strict": True}]
    turn = client.complete(system="sys", messages=[{"role": "user", "content": "hi"}], tools=tools)

    sent = messages.kwargs
    assert sent["model"] == "claude-opus-5-5" and sent["tools"] == tools and sent["system"] == "sys"
    assert sent["fallbacks"] == "default" and sent["betas"] == ["server-side-fallback-2026-07-01"]
    assert "tool_choice" not in sent  # forced tool use isn't supported on current models
    assert turn.stop_reason == "tool_use"
    assert turn.usage == {"input_tokens": 120, "output_tokens": 30, "cache_read_input_tokens": 0}
    # Blocks are serialised losslessly so they can be echoed back unchanged.
    assert turn.content == [
        {"type": "thinking", "thinking": "", "signature": "sig"},
        {"type": "text", "text": "Computing."},
        {"type": "tool_use", "id": "toolu_1", "name": "calculator", "input": {"expression": "1+1"}},
    ]
