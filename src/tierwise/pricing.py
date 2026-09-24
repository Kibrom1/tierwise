"""What a step costs, and what moving it costs.

Complexity is only half of a routing decision. The other half is what the move
itself costs, and until now nothing modelled it.

Prompt cache entries are tied to a model. Leaving a model forfeits its cache and
pays a cache *write* on the one you arrive at, so inside a long cached
conversation the cheap tier can easily cost more than the expensive one:

    100k cached context, one small step
      stay on Opus, cache hit    100k x $5 x 0.1   = $0.050
      drop to Haiku, cache miss  100k x $1 x 1.25  = $0.125

Dropping a tier is 2.5x *more* expensive there. It only pays once the output is
large enough for the per-token saving to outrun the cache penalty -- roughly
output > context x 0.0375 for that pair, about 3,750 tokens against a 100k
context. Most agent steps produce a fraction of that.

So: routing down pays when context is small relative to output, and loses when
context is large and warm. A router that ignores this will confidently spend
more money than doing nothing.

Prices are per million tokens, checked against Anthropic's pricing page in
February 2026. They will age; override them rather than trusting them.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

from .models import Tier

MTOK = 1_000_000


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token prices, plus the cache multipliers that apply to input."""

    input_per_mtok: float
    output_per_mtok: float
    #: Reading an existing cache entry, relative to base input.
    cache_read_multiplier: float = 0.10
    #: Writing a new one (5-minute TTL), relative to base input.
    cache_write_multiplier: float = 1.25


#: Checked February 2026. Add your own provider's models here, or pass a
#: price_for callable -- nothing in this module assumes Anthropic.
PRICES: dict[str, ModelPrice] = {
    "claude-haiku-4-5-20251001": ModelPrice(1.0, 5.0),
    "claude-haiku-4-5": ModelPrice(1.0, 5.0),
    "claude-sonnet-5": ModelPrice(2.0, 10.0),
    "claude-sonnet-4-5": ModelPrice(3.0, 15.0),
    "claude-opus-5": ModelPrice(5.0, 25.0),
    "claude-opus-4-5": ModelPrice(5.0, 25.0),
}


def price_for(model: str) -> Optional[ModelPrice]:
    return PRICES.get(model)


@dataclass
class SwitchContext:
    """What the caller knows about the request it is about to make.

    ``incumbent`` is the tier whose cache is currently warm -- usually the tier
    the previous step ran at. With no incumbent there is nothing to forfeit and
    no switching cost to weigh.
    """

    cached_tokens: int = 0
    expected_output_tokens: int = 0
    incumbent: Optional[Tier] = None
    cache_warm: bool = True

    def __post_init__(self) -> None:
        if isinstance(self.incumbent, str):
            self.incumbent = Tier(self.incumbent)
        if self.cached_tokens < 0 or self.expected_output_tokens < 0:
            raise ValueError("token counts must be >= 0")


def step_cost(price: ModelPrice, context_tokens: int, output_tokens: int,
              cache_hit: bool) -> float:
    """Cost of one request: the context at cache-read or cache-write rate, plus output."""
    multiplier = price.cache_read_multiplier if cache_hit else price.cache_write_multiplier
    return (context_tokens * price.input_per_mtok * multiplier
            + output_tokens * price.output_per_mtok) / MTOK


@dataclass
class SwitchVerdict:
    switch: bool
    reason: str
    incumbent_cost: Optional[float] = None
    candidate_cost: Optional[float] = None

    @property
    def saving(self) -> Optional[float]:
        if self.incumbent_cost is None or self.candidate_cost is None:
            return None
        return round(self.incumbent_cost - self.candidate_cost, 6)


def evaluate_switch(
    incumbent: Tier,
    candidate: Tier,
    context: SwitchContext,
    model_for: Callable[[Tier], str],
    prices: Optional[Callable[[str], Optional[ModelPrice]]] = None,
) -> SwitchVerdict:
    """Is moving from `incumbent` to `candidate` worth what the move costs?

    Only meaningful for a *downgrade*. Moving up is a correctness decision, and
    this module has no opinion on those -- a step that needs a better model
    needs it whatever the cache is doing.
    """
    lookup = prices or price_for

    if candidate is incumbent:
        return SwitchVerdict(False, "already at this tier")
    if candidate > incumbent:
        return SwitchVerdict(True, "upgrade: cost does not gate quality")
    if not context.cache_warm or context.cached_tokens <= 0:
        return SwitchVerdict(True, "no warm cache to forfeit")

    incumbent_price = lookup(model_for(incumbent))
    candidate_price = lookup(model_for(candidate))
    if incumbent_price is None or candidate_price is None:
        # Unknown prices: do not pretend to have done the arithmetic.
        return SwitchVerdict(True, "no price for one of these models; not gating")

    stay = step_cost(incumbent_price, context.cached_tokens,
                     context.expected_output_tokens, cache_hit=True)
    move = step_cost(candidate_price, context.cached_tokens,
                     context.expected_output_tokens, cache_hit=False)

    if move < stay:
        return SwitchVerdict(True, f"saves ${stay - move:.4f} after the cache write",
                             stay, move)
    return SwitchVerdict(
        False,
        f"cache write on {candidate.value} costs ${move - stay:.4f} more than "
        f"staying on {incumbent.value}",
        stay, move,
    )


def actual_cost(
    model: str,
    usage: dict,
    prices: Optional[Callable[[str], Optional[ModelPrice]]] = None,
) -> Optional[float]:
    """What a completed call actually cost, from the provider's own `usage`.

    Unlike `step_cost` (which assumes a call is either a full cache hit or a
    full cache write, for comparing two candidate tiers before the call),
    this reads the real breakdown a response reports -- base input, a cache
    read, a cache write, and output are billed at different rates and a real
    call can carry more than one. Returns None when the model's price is not
    known, rather than guessing.
    """
    lookup = prices or price_for
    price = lookup(model)
    if price is None:
        return None

    base_input = int(usage.get("input_tokens") or 0)
    cache_read = int(usage.get("cache_read_input_tokens") or 0)
    cache_write = int(usage.get("cache_creation_input_tokens") or 0)
    output = int(usage.get("output_tokens") or 0)

    return (
        base_input * price.input_per_mtok
        + cache_read * price.input_per_mtok * price.cache_read_multiplier
        + cache_write * price.input_per_mtok * price.cache_write_multiplier
        + output * price.output_per_mtok
    ) / MTOK


def breakeven_output_tokens(
    incumbent: Tier, candidate: Tier, cached_tokens: int,
    model_for: Callable[[Tier], str],
    prices: Optional[Callable[[str], Optional[ModelPrice]]] = None,
) -> Optional[float]:
    """Output size at which dropping to `candidate` starts paying.

    Useful for answering "is per-step routing worth it in my setup at all?"
    before wiring anything up.
    """
    lookup = prices or price_for
    hi, lo = lookup(model_for(incumbent)), lookup(model_for(candidate))
    if hi is None or lo is None:
        return None

    penalty = cached_tokens * (lo.input_per_mtok * lo.cache_write_multiplier
                               - hi.input_per_mtok * hi.cache_read_multiplier)
    per_token_saving = hi.output_per_mtok - lo.output_per_mtok
    if per_token_saving <= 0:
        return None
    if penalty <= 0:
        return 0.0
    return penalty / per_token_saving
