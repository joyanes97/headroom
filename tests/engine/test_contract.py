from headroom.engine.contract import Flavor, Provider, RequestContext


def test_request_context_carries_fields():
    ctx = RequestContext(
        provider=Provider.ANTHROPIC,
        flavor=Flavor.MESSAGES,
        headers_view={"x-api-key": "redacted"},
        raw_body=b'{"model":"claude"}',
        session_key="abc123",
    )
    assert ctx.provider is Provider.ANTHROPIC
    assert ctx.flavor is Flavor.MESSAGES
    assert ctx.raw_body == b'{"model":"claude"}'
    assert ctx.session_key == "abc123"
    assert ctx.headers_view["x-api-key"] == "redacted"
