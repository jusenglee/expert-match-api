from __future__ import annotations

from typing import Any

# Keep planner and judge responses low-variance until runtime tuning is exposed.
CONSISTENCY_TEMPERATURE = 0.0
CONSISTENCY_TOP_P = 0.2
CONSISTENCY_REASONING_EFFORT = "low"
CONSISTENCY_INCLUDE_REASONING = False
CONSISTENCY_DISABLE_THINKING = True


def build_consistency_invoke_kwargs(
    *, max_tokens_hint: int | None = None, seed: int | None = None
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "temperature": CONSISTENCY_TEMPERATURE,
        "top_p": CONSISTENCY_TOP_P,
        "reasoning_effort": CONSISTENCY_REASONING_EFFORT,
        "include_reasoning": CONSISTENCY_INCLUDE_REASONING,
        "disable_thinking": CONSISTENCY_DISABLE_THINKING,
    }
    if max_tokens_hint is not None:
        kwargs["max_tokens_hint"] = max_tokens_hint
    if seed is not None:
        kwargs["seed"] = seed
    return kwargs
