"""The proxy: re-routing without touching the client's code."""

import json
import threading
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from tierwise import (
    ENFORCE,
    SHADOW,
    JsonlSink,
    ModelMap,
    ProxyRouter,
    Router,
    Tier,
    conversation_key,
    serve,
    signals_from_payload,
)
from tierwise.proxy import ConversationState, _usage_from, estimate_tokens

MODELS = ModelMap({Tier.LOW: "claude-haiku-4-5-20251001",
                   Tier.MEDIUM: "claude-sonnet-5",
                   Tier.HIGH: "claude-opus-5"})


def request(model="claude-opus-5", text="tidy up the comment in a.py", system="agent",
            tools=None, content=None):
    return {
        "model": model,
        "max_tokens": 1024,
        "system": system,
        "messages": [{"role": "user", "content": content or text}],
        **({"tools": tools} if tools else {}),
    }


def proxy(mode=SHADOW, telemetry=None):
    router = Router(model_map=MODELS, telemetry=telemetry) if telemetry else Router(model_map=MODELS)
    return ProxyRouter(router=router, mode=mode, model_map=MODELS)


# -- reading a request -------------------------------------------------------


def test_openai_shaped_payload_routes_the_same_as_anthropic_shaped():
    """The proxy assumes nothing Anthropic-specific about the body.

    Codex and other OpenAI-compatible clients send `model` + `messages` with no
    top-level `system` field and a `role: "system"` message instead. Confirms
    tierwise serve is a genuine drop-in for those clients too, not just
    Anthropic's Messages API shape.
    """
    openai_payload = {
        "model": "gpt-5",
        "messages": [
            {"role": "system", "content": "You are a coding agent."},
            {"role": "user", "content": "rename the variable foo to bar in a.py"},
        ],
        "max_tokens": 512,
    }
    router_proxy = proxy()
    plan = router_proxy.plan(openai_payload)
    assert plan.decision.tier is not None

    applied = router_proxy.apply(openai_payload, plan)
    # Shadow mode: the client's own model name is untouched either way.
    assert applied["model"] == "gpt-5"


def test_it_counts_files_mentioned_in_the_request():
    read = signals_from_payload(request(text="update src/a.py and tests/b.py"))
    assert read.signals.file_count == 2


def test_it_counts_diff_lines():
    diff = "```\n+    added = 1\n+    also = 2\n-    gone = 3\n```"
    assert signals_from_payload(request(text=diff)).signals.lines_changed == 3


def test_diff_headers_are_not_counted_as_changes():
    assert signals_from_payload(request(text="--- a/x.py\n+++ b/x.py\n+one\n")).signals.lines_changed == 1


def test_tool_results_mean_existing_context_was_read():
    content = [{"type": "tool_result", "content": "file contents here"}]
    assert signals_from_payload(request(content=content)).signals.requires_context is True
    assert signals_from_payload(request(text="plain ask")).signals.requires_context is False


def test_ambiguity_is_never_invented():
    """A request does not say how underspecified the work was."""
    assert signals_from_payload(request()).signals.ambiguity == 0.0
    assert signals_from_payload(request()).signals.dependency_depth == 0


def test_context_size_is_estimated_from_the_whole_body():
    small = signals_from_payload(request(text="hi")).context_tokens
    large = signals_from_payload(request(text="x" * 40_000)).context_tokens
    assert large > small + 9_000
    assert estimate_tokens("a" * 400) == 100


def test_the_description_survives_for_the_log():
    read = signals_from_payload(request(text="rename sessionId everywhere"))
    assert "rename sessionId" in read.signals.description


def test_conversations_are_keyed_on_their_stable_prefix():
    """A later turn appends messages; the system prompt and first turn hold."""
    first = request(system="agent A", text="open the file")
    later = dict(first, messages=first["messages"] + [
        {"role": "assistant", "content": "opened"},
        {"role": "user", "content": "now rename it"},
    ])
    other = request(system="agent B", text="open the file")

    assert conversation_key(first) == conversation_key(later)
    assert conversation_key(first) != conversation_key(other)


# -- planning ----------------------------------------------------------------

def test_shadow_mode_never_changes_the_request():
    p = proxy(SHADOW)
    payload = request()
    plan = p.plan(payload)
    applied = p.apply(payload, plan)

    assert plan.enforced is False
    assert applied["model"] == "claude-opus-5"        # untouched
    assert plan.routed_model != "" and plan.tier is not None


def test_enforce_mode_rewrites_the_model():
    p = proxy(ENFORCE)
    payload = request()
    plan = p.plan(payload)
    applied = p.apply(payload, plan)

    if plan.changed:
        assert applied["model"] == plan.routed_model
    assert plan.enforced is True


def test_the_original_payload_is_not_mutated():
    p = proxy(ENFORCE)
    payload = request()
    p.apply(payload, p.plan(payload))
    assert payload["model"] == "claude-opus-5"


def test_the_requested_model_sets_the_incumbent_tier():
    p = proxy()
    assert p.tier_of("claude-opus-5") is Tier.HIGH
    assert p.tier_of("claude-haiku-4-5-20251001") is Tier.LOW
    assert p.tier_of("something-else") is None


def test_a_warm_cache_holds_a_downgrade_back():
    """The failure that makes naive proxies cost money."""
    p = proxy(ENFORCE)
    payload = request()
    key = conversation_key(payload)
    p.observe(key, {"cache_read_input_tokens": 120_000, "output_tokens": 400})

    plan = p.plan(payload, key)
    assert plan.decision.source.value == "cache_hold"
    assert plan.tier is Tier.HIGH
    assert plan.changed is False


def test_a_cold_conversation_routes_freely():
    p = proxy(ENFORCE)
    payload = request()
    assert p.plan(payload).decision.source.value != "cache_hold"


def test_observed_usage_replaces_the_estimate():
    p = proxy()
    p.observe("k", {"cache_read_input_tokens": 50_000, "output_tokens": 1_200})
    assert p.conversations["k"].cached_tokens == 50_000
    assert p.conversations["k"].last_output_tokens == 1_200


def test_a_cache_write_also_counts_as_cached():
    p = proxy()
    p.observe("k", {"cache_creation_input_tokens": 30_000})
    assert p.conversations["k"].cached_tokens == 30_000


def test_missing_usage_creates_no_state():
    """A response without usage teaches nothing, so it records nothing."""
    p = proxy()
    p.observe("k", None)
    assert "k" not in p.conversations
    p.observe("k", {})
    assert "k" not in p.conversations


def test_every_decision_reaches_the_log(tmp_path):
    log = tmp_path / "routing.jsonl"
    p = proxy(SHADOW, telemetry=JsonlSink(log))
    p.plan(request())
    events = [json.loads(line) for line in log.read_text().splitlines()]
    assert events[0]["event"] == "routing_decision"


def test_an_unknown_mode_is_rejected():
    with pytest.raises(ValueError):
        ProxyRouter(mode="maybe")


# -- usage extraction --------------------------------------------------------

def test_usage_is_read_from_a_json_response():
    body = json.dumps({"usage": {"output_tokens": 42}}).encode()
    assert _usage_from(body)["output_tokens"] == 42


def test_usage_is_read_from_a_streamed_response():
    stream = (
        b'event: message_start\n'
        b'data: {"message": {"usage": {"cache_read_input_tokens": 900}}}\n\n'
        b'event: message_delta\n'
        b'data: {"usage": {"output_tokens": 7}}\n\n'
    )
    usage = _usage_from(stream)
    assert usage["cache_read_input_tokens"] == 900
    assert usage["output_tokens"] == 7


def test_unparseable_bodies_yield_nothing():
    assert _usage_from(b"not json at all") is None


# -- end to end, against a fake upstream -------------------------------------

class _Echo(BaseHTTPRequestHandler):
    """Stands in for the provider: reports the model it was actually asked for."""

    def log_message(self, *args):
        return

    def do_POST(self):
        length = int(self.headers.get("content-length") or 0)
        sent = json.loads(self.rfile.read(length))
        body = json.dumps({
            "model_received": sent.get("model"),
            "auth_seen": self.headers.get("x-api-key"),
            "usage": {"cache_read_input_tokens": 0, "output_tokens": 11},
        }).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _Echo)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


def call_through(port, payload):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/messages",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json", "x-api-key": "sk-test"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=10) as response:
        return json.loads(response.read())


@pytest.fixture
def running():
    servers = []

    def start(mode):
        p = proxy(mode)
        server = serve(port=0, upstream=start.upstream, proxy=p, verbose=False)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        servers.append(server)
        return server.server_address[1], p

    yield start
    for server in servers:
        server.shutdown()
        server.server_close()


def test_shadow_mode_forwards_the_request_untouched(upstream, running):
    running.upstream = upstream
    port, _ = running(SHADOW)
    result = call_through(port, request())
    assert result["model_received"] == "claude-opus-5"


def test_enforce_mode_reaches_the_provider_with_the_routed_model(upstream, running):
    running.upstream = upstream
    port, p = running(ENFORCE)
    payload = request()
    expected = p.plan(dict(payload)).routed_model
    result = call_through(port, payload)
    assert result["model_received"] == expected


def test_credentials_are_relayed_not_stored(upstream, running):
    running.upstream = upstream
    port, _ = running(SHADOW)
    assert call_through(port, request())["auth_seen"] == "sk-test"


def test_usage_from_the_response_is_learned(upstream, running):
    running.upstream = upstream
    port, p = running(SHADOW)
    payload = request()
    call_through(port, payload)
    assert p.conversations[conversation_key(payload)].last_output_tokens == 11


def test_an_unreachable_upstream_returns_a_readable_error(running):
    running.upstream = "http://127.0.0.1:1"      # nothing listening
    port, _ = running(SHADOW)
    with pytest.raises(urllib.error.HTTPError) as excinfo:
        call_through(port, request())
    assert excinfo.value.code == 502
    assert b"upstream unreachable" in excinfo.value.read()


# -- size, the signal a request actually carries ------------------------------

def test_request_size_stands_in_for_change_size_when_there_is_no_diff():
    """Ignoring size sent the largest, most cross-cutting requests to the cheapest tier."""
    small = signals_from_payload(request(text="tidy the comment in src/util.py")).signals
    large = signals_from_payload(
        request(content=[{"type": "tool_result", "content": "x" * 48_000}])
    ).signals
    assert small.lines_changed < 5
    assert large.lines_changed > 1_000
    assert Router().route(large).tier is Tier.HIGH
    assert Router().route(small).tier is Tier.LOW


def test_a_description_only_request_goes_to_the_classifier():
    """All text and no structure is exactly what the heuristic cannot judge."""
    read = signals_from_payload(request(text="make the checkout flow less confusing"))
    assert read.signals.metadata["measured"] is False
    assert Router().route(read.signals).source.value == "llm"


def test_a_request_with_something_to_read_is_scored_on_it():
    read = signals_from_payload(request(text="patch it\n+    a = 1"))
    assert read.signals.metadata["measured"] is True
    assert Router().route(read.signals).source.value == "heuristic"


def test_a_diff_beats_the_size_estimate():
    """A diff states the size of the change outright; nothing needs estimating."""
    payload = request(text="apply this\n+    a = 1\n-    b = 2")
    assert signals_from_payload(payload).signals.lines_changed == 2


def test_directories_touched_stand_in_for_reach():
    spread = signals_from_payload(request(
        text="src/net/a.py, src/db/b.py, lib/c.py and app/d.py"
    )).signals
    assert spread.file_count == 4
    assert spread.dependency_depth == 3


def test_depth_is_never_inferred_without_real_paths():
    assert signals_from_payload(request(text="just do the thing")).signals.dependency_depth == 0


def test_tool_blocks_count_as_material_in_play():
    content = [{"type": "tool_result", "content": "a"}, {"type": "tool_use", "name": "read"}]
    assert signals_from_payload(request(content=content)).signals.file_count == 2
