"""Re-route without touching anyone's code.

There are three places a model choice can be intercepted:

1. In your own call, where you pass ``decision.model`` yourself. Works today,
   and needs you to be the one writing the call.
2. In the request path, between a client and the provider. Works with any
   client that accepts a base URL, and needs nothing from the client but a
   changed environment variable. This module.
3. In the client's settings, which choose a model per *session*, not per
   request -- too coarse to route with.

So: point a client at this and it routes. ``ANTHROPIC_BASE_URL=http://localhost:8787``
is the whole integration.

Two commitments make it safe to try:

**Shadow mode is the default.** It reads every request, decides what it would
have routed to, writes that to the decision log, and forwards the request
unchanged. Nothing about the client's behaviour changes. You get a report of
what routing *would* have done to work you already did, and you turn on
enforcement when the report convinces you -- or don't.

**It is cache-aware.** It tracks what each conversation has cached and refuses a
downgrade that would forfeit more than it saves, which is the failure that makes
naive routers cost money (see ``tierwise.pricing``).

Signals here are coarser than a caller can supply: a request carries no ticket
and no notion of how underspecified the work is. What it does carry is size, how
much has been read, and whether a diff is being produced -- which is most of what
the scorer weighs. ``ambiguity`` stays at zero rather than being invented.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .mapping import FallbackMap, ModelMap
from .models import RoutingDecision, TaskSignals, Tier
from .budget import BudgetState
from .pricing import SwitchContext, actual_cost
from .router import Router

#: Rough characters per token. Only ever used to compare magnitudes -- the real
#: counts come back from the provider in `usage` and replace these.
CHARS_PER_TOKEN = 4

#: Conservative stand-in for output size before we have seen any. max_tokens is
#: an upper bound, not an expectation, and using it would bias every decision
#: toward switching.
ASSUMED_OUTPUT_TOKENS = 800

#: Tokens per line of code, for turning request size into the scorer's size
#: signal. A caller supplying signals directly knows how many lines changed; a
#: proxy only knows how much material is in play, and a request carrying 12k
#: tokens is not a twenty-line task. This is an approximation and is labelled as
#: one -- but ignoring size entirely was worse: it sent the largest, most
#: cross-cutting requests to the cheapest tier.
TOKENS_PER_LINE = 10

SHADOW, ENFORCE = "shadow", "enforce"

_PATH = re.compile(r"[\w./-]+\.[A-Za-z]{1,6}\b")
_DIFF_LINE = re.compile(r"^[+-](?![+-])", re.MULTILINE)


def _walk_text(value: Any, out: list[str]) -> None:
    """Collect every string in a request body, whatever shape it arrives in."""
    if isinstance(value, str):
        out.append(value)
    elif isinstance(value, dict):
        for key, item in value.items():
            if key not in {"type", "role", "model"}:
                _walk_text(item, out)
    elif isinstance(value, list):
        for item in value:
            _walk_text(item, out)


def estimate_tokens(text: str) -> int:
    return len(text) // CHARS_PER_TOKEN


def _has_tool_results(payload: dict[str, Any]) -> bool:
    for message in payload.get("messages") or []:
        content = message.get("content")
        if isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in {"tool_result", "tool_use"}:
                    return True
    return False


@dataclass
class RequestSignals:
    signals: TaskSignals
    context_tokens: int
    requested_model: str


def signals_from_payload(payload: dict[str, Any]) -> RequestSignals:
    """Derive what a request can honestly tell us about the work it asks for."""
    collected: list[str] = []
    _walk_text(payload.get("system"), collected)
    _walk_text(payload.get("messages"), collected)
    _walk_text(payload.get("tools"), collected)
    text = "\n".join(collected)

    paths = {match.group(0) for match in _PATH.finditer(text)}
    diff_lines = len(_DIFF_LINE.findall(text))
    context_tokens = estimate_tokens(text)

    blocks_hint = payload.get("messages") or []

    # Distinct directories touched stands in for how far the work reaches. Only
    # counted when real paths were found -- never inferred from nothing.
    directories = {path.rsplit("/", 1)[0] for path in paths if "/" in path}
    depth = max(len(directories) - 1, 0)

    blocks = sum(
        1 for message in blocks_hint
        for block in (message.get("content") if isinstance(message.get("content"), list) else [])
        if isinstance(block, dict) and block.get("type") in {"tool_result", "tool_use"}
    )

    last: list[str] = []
    messages = payload.get("messages") or []
    if messages:
        _walk_text(messages[-1], last)
    description = " ".join(" ".join(last).split())[:200]

    # A diff states the size of the change outright. Without one, request size
    # stands in -- but only when something structural was actually found. Deriving
    # a size signal from text length alone would manufacture evidence out of a
    # long sentence, and a description-only request is precisely the case that
    # must reach the classifier rather than be scored and called simple.
    measured_structure = bool(diff_lines or paths or blocks)
    if diff_lines:
        size = diff_lines
    elif measured_structure:
        size = context_tokens // TOKENS_PER_LINE
    else:
        size = 0

    # "measured" tells the scorer that zeros here are readings, not absences --
    # but only when there was something to read. A request carrying no diff, no
    # paths and no tool blocks is all description, and the description is the
    # one thing the heuristic cannot see: those go to the classifier rather than
    # being scored 0.0 and called simple with confidence. Marking every request
    # measured would re-open exactly the defect has_evidence exists to close.
    measured = measured_structure

    return RequestSignals(
        signals=TaskSignals(
            description=description,
            file_count=max(len(paths), blocks, 1),
            lines_changed=size,
            dependency_depth=depth,
            requires_context=_has_tool_results(payload),
            # A request does not say how underspecified the work was. Left at
            # zero rather than guessed; the scorer weights it heavily.
            ambiguity=0.0,
            metadata={"measured": measured, "context_tokens": context_tokens,
                      "paths": sorted(paths)[:20]},
        ),
        context_tokens=context_tokens,
        requested_model=str(payload.get("model") or ""),
    )


@dataclass
class ConversationState:
    """What we have seen from one conversation, for the switching-cost model."""

    cached_tokens: int = 0
    last_output_tokens: int = 0
    calls: int = 0


@dataclass
class ProxyPlan:
    requested_model: str
    routed_model: str
    tier: Tier
    decision: RoutingDecision
    enforced: bool
    #: True when enforce mode was suppressed for this one request because a
    #: configured spend ceiling was already hit -- the decision itself is
    #: still computed and logged, it just never reaches the client.
    budget_blocked: bool = False

    @property
    def changed(self) -> bool:
        return self.routed_model != self.requested_model

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested_model": self.requested_model,
            "routed_model": self.routed_model,
            "tier": self.tier.value,
            "changed": self.changed,
            "enforced": self.enforced,
            "budget_blocked": self.budget_blocked,
            "source": self.decision.source.value,
        }


def conversation_key(payload: dict[str, Any]) -> str:
    """Identify a conversation by its stable prefix -- what the cache keys on."""
    head: list[str] = []
    _walk_text(payload.get("system"), head)
    messages = payload.get("messages") or []
    if messages:
        _walk_text(messages[0], head)
    return hashlib.sha256("\n".join(head)[:4000].encode("utf-8")).hexdigest()[:16]


class ProxyRouter:
    """The routing decision for one request, with no HTTP in sight."""

    def __init__(
        self,
        router: Optional[Router] = None,
        mode: str = SHADOW,
        model_map: Optional[ModelMap] = None,
        budget: Optional[BudgetState] = None,
        fallback_map: Optional[FallbackMap] = None,
    ) -> None:
        if mode not in {SHADOW, ENFORCE}:
            raise ValueError(f"mode must be {SHADOW!r} or {ENFORCE!r}")
        self.mode = mode
        self.model_map = model_map or (router.model_map if router else ModelMap.resolve())
        self.router = router or Router(model_map=self.model_map)
        self.conversations: dict[str, ConversationState] = {}
        #: None means "no ceiling configured" -- every budget check below is
        #: then a no-op, so an unconfigured proxy pays nothing for carrying
        #: this around.
        self.budget = budget
        #: An empty/unset FallbackMap means "no fallback for any tier" --
        #: fallback_for() then always returns None, so an unconfigured proxy
        #: behaves exactly as it did before this existed.
        self.fallback_map = fallback_map or FallbackMap()

    def tier_of(self, model: str) -> Optional[Tier]:
        for tier, name in self.model_map.models.items():
            if name == model:
                return tier
        return None

    def fallback_for(self, model: str) -> Optional[str]:
        """The configured fallback model for whichever tier `model` belongs to.

        `model` may be a fallback model itself (already retried once) or a
        model outside the tier map entirely (e.g. a client's own requested
        model in shadow mode) -- both correctly resolve to no fallback, which
        is what stops a retry loop from ever forming.
        """
        return self.fallback_map.for_tier(self.tier_of(model))

    def plan(self, payload: dict[str, Any], key: Optional[str] = None) -> ProxyPlan:
        key = key or conversation_key(payload)
        state = self.conversations.setdefault(key, ConversationState())
        read = signals_from_payload(payload)

        incumbent = self.tier_of(read.requested_model)
        context = SwitchContext(
            cached_tokens=state.cached_tokens,
            expected_output_tokens=state.last_output_tokens or ASSUMED_OUTPUT_TOKENS,
            incumbent=incumbent,
            cache_warm=state.cached_tokens > 0,
        )

        decision = self.router.route(read.signals, session_id=key, step_index=state.calls,
                                     context=context)
        state.calls += 1

        budget_blocked = bool(self.budget is not None and self.budget.exceeded())
        enforced = self.mode == ENFORCE and not budget_blocked

        return ProxyPlan(
            requested_model=read.requested_model,
            routed_model=decision.model,
            tier=decision.tier,
            decision=decision,
            enforced=enforced,
            budget_blocked=budget_blocked,
        )

    def apply(self, payload: dict[str, Any], plan: ProxyPlan) -> dict[str, Any]:
        """Rewrite the model, or leave the request exactly as it arrived."""
        if plan.enforced and plan.changed:
            payload = dict(payload)
            payload["model"] = plan.routed_model
        return payload

    def observe(self, key: str, usage: Optional[dict[str, Any]],
                model: Optional[str] = None) -> None:
        """Learn the real token counts, which beat any estimate we made."""
        if not usage:
            return
        state = self.conversations.setdefault(key, ConversationState())
        cached = int(usage.get("cache_read_input_tokens") or 0)
        created = int(usage.get("cache_creation_input_tokens") or 0)
        if cached or created:
            state.cached_tokens = cached or created
        output = usage.get("output_tokens")
        if output:
            state.last_output_tokens = int(output)

        if self.budget is not None and self.budget.limit_usd is not None and model:
            cost = actual_cost(model, usage)
            if cost:
                self.budget.record_spend(cost)
                self.budget.save()


# ---------------------------------------------------------------------------
# The HTTP layer. Deliberately thin, and dependency-free: this is a local
# development proxy, not a production gateway.
# ---------------------------------------------------------------------------

import socketserver          # noqa: E402
import urllib.error          # noqa: E402
import urllib.request        # noqa: E402
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer  # noqa: E402

DEFAULT_UPSTREAM = "https://api.anthropic.com"
DEFAULT_PORT = 8787

#: Statuses that say "the provider or this specific model was unavailable,"
#: not "your request was bad" -- the only ones worth retrying against a
#: fallback. 529 is Anthropic's overloaded status; the rest are the usual
#: 5xx infra codes plus a request timeout.
_INFRA_ERROR_STATUSES = {408, 500, 502, 503, 504, 508, 529}

#: Hop-by-hop headers must not be relayed.
_SKIP_REQUEST_HEADERS = {"host", "content-length", "connection", "accept-encoding"}
_SKIP_RESPONSE_HEADERS = {"content-length", "transfer-encoding", "connection", "content-encoding"}


class _Handler(BaseHTTPRequestHandler):
    proxy: "ProxyRouter"
    upstream: str
    verbose: bool

    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # noqa: D102 - quieter default
        return

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("content-length") or 0)
        raw = self.rfile.read(length) if length else b""

        payload, plan, key, model_used = None, None, None, None
        try:
            payload = json.loads(raw)
        except (json.JSONDecodeError, UnicodeDecodeError):
            payload = None

        if isinstance(payload, dict) and payload.get("model"):
            key = conversation_key(payload)
            plan = self.proxy.plan(payload, key)
            payload = self.proxy.apply(payload, plan)
            raw = json.dumps(payload).encode("utf-8")
            model_used = payload.get("model")
            if self.verbose:
                mark = "->" if plan.changed and plan.enforced else ("~ " if plan.changed else "  ")
                if plan.budget_blocked:
                    mark = "$ "
                print(f"{mark} {plan.requested_model} -> {plan.routed_model} "
                      f"[{plan.decision.source.value}]"
                      + (" (budget ceiling hit -- not enforced)" if plan.budget_blocked else ""),
                      flush=True)

        self._relay(raw, plan, key, model_used)

    def _relay(self, body: bytes, plan: Optional[ProxyPlan], key: Optional[str],
               model_used: Optional[str] = None) -> None:
        headers = {k: v for k, v in self.headers.items()
                   if k.lower() not in _SKIP_REQUEST_HEADERS}
        request = urllib.request.Request(
            self.upstream.rstrip("/") + self.path, data=body, headers=headers, method="POST"
        )
        try:
            with urllib.request.urlopen(request) as response:
                self._begin(response.status, response.headers)
                self._pump(response, key, model_used)
                return
        except urllib.error.HTTPError as error:
            if self._retry_with_fallback(error, body, headers, key, model_used):
                return
            # No fallback available, or the fallback failed too: the error
            # belongs to the client unchanged, status and all.
            self._begin(error.code, error.headers)
            self._pump(error, key, model_used)
        except urllib.error.URLError as error:
            self._fail(502, f"upstream unreachable: {error.reason}")

    def _retry_with_fallback(self, error: urllib.error.HTTPError, body: bytes,
                              headers: dict, key: Optional[str],
                              model_used: Optional[str]) -> bool:
        """One retry, same tier, a different model -- only for infra failure.

        `model_used` failing with a 5xx-class status says nothing about
        whether the *tier* was right for the task; it says the provider or
        that specific model was unavailable. Retrying at the same tier
        against a configured fallback keeps that distinct from an
        escalation-worthy outcome, which is a judgement about task
        difficulty, not infrastructure. A non-infra status (4xx other than
        a timeout, or no fallback configured for this tier) is never
        retried -- this is deliberately narrow, not a general retry policy.
        """
        if error.code not in _INFRA_ERROR_STATUSES or not model_used:
            return False
        fallback = self.proxy.fallback_for(model_used)
        if not fallback:
            return False

        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return False
        if not isinstance(payload, dict):
            return False
        payload = dict(payload)
        payload["model"] = fallback
        retry_body = json.dumps(payload).encode("utf-8")

        if self.verbose:
            print(f"!! {model_used} returned {error.code} -- retrying once with "
                  f"fallback {fallback}", flush=True)

        retry_request = urllib.request.Request(
            self.upstream.rstrip("/") + self.path, data=retry_body, headers=headers,
            method="POST",
        )
        try:
            with urllib.request.urlopen(retry_request) as response:
                self._begin(response.status, response.headers)
                self._pump(response, key, fallback)
                return True
        except (urllib.error.HTTPError, urllib.error.URLError):
            # The fallback failed too -- fall through and surface the
            # *original* error to the client, not the fallback's.
            return False

    def _begin(self, status: int, headers) -> None:
        self.send_response(status)
        for name, value in headers.items():
            if name.lower() not in _SKIP_RESPONSE_HEADERS:
                self.send_header(name, value)
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

    def _pump(self, response, key: Optional[str], model_used: Optional[str] = None) -> None:
        """Stream the response through, keeping a copy only to read `usage`."""
        collected = bytearray()
        while True:
            chunk = response.read(8192)
            if not chunk:
                break
            if len(collected) < 1_000_000:
                collected.extend(chunk)
            self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
            self.wfile.flush()
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()
        if key:
            self.proxy.observe(key, _usage_from(bytes(collected)), model_used)

    def _fail(self, status: int, message: str) -> None:
        body = json.dumps({"error": {"type": "tierwise_proxy_error", "message": message}})
        encoded = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)


def _usage_from(body: bytes) -> Optional[dict[str, Any]]:
    """Pull `usage` out of a JSON response, or out of an SSE stream's events."""
    try:
        text = body.decode("utf-8", errors="ignore")
    except Exception:  # noqa: BLE001 - pragma: no cover
        return None

    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict) and isinstance(parsed.get("usage"), dict):
            return parsed["usage"]
    except json.JSONDecodeError:
        pass

    merged: dict[str, Any] = {}
    for line in text.splitlines():
        if not line.startswith("data:"):
            continue
        try:
            event = json.loads(line[5:].strip())
        except json.JSONDecodeError:
            continue
        for candidate in (event.get("usage"), (event.get("message") or {}).get("usage")):
            if isinstance(candidate, dict):
                merged.update(candidate)
    return merged or None


def serve(
    port: int = DEFAULT_PORT,
    upstream: str = DEFAULT_UPSTREAM,
    proxy: Optional[ProxyRouter] = None,
    verbose: bool = True,
    ready: Optional[Any] = None,
) -> socketserver.BaseServer:
    """Start the routing proxy. Returns the server; call shutdown() to stop."""
    handler = type("Handler", (_Handler,), {
        "proxy": proxy or ProxyRouter(),
        "upstream": upstream,
        "verbose": verbose,
    })
    server = ThreadingHTTPServer(("127.0.0.1", port), handler)
    server.daemon_threads = True
    if ready is not None:
        ready.set()
    return server
