"""Cost and latency evaluation models for LongCache qualification."""

from __future__ import annotations

from typing import Any

# Representative model pricing (USD per 1,000,000 tokens)
MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o": {
        "input_per_million": 2.50,
        "output_per_million": 10.00,
        "typical_latency_ms": 1200.0,
    },
    "claude-3-5-sonnet": {
        "input_per_million": 3.00,
        "output_per_million": 15.00,
        "typical_latency_ms": 1400.0,
    },
    "gpt-4o-mini": {
        "input_per_million": 0.15,
        "output_per_million": 0.60,
        "typical_latency_ms": 600.0,
    },
}


def compute_origin_cost(
    model: str = "gpt-4o", input_tokens: int = 500, output_tokens: int = 300
) -> float:
    """Compute estimated origin LLM cost for one request in USD."""
    pricing = MODEL_PRICING.get(model, MODEL_PRICING["gpt-4o"])
    input_cost = (input_tokens / 1_000_000.0) * pricing["input_per_million"]
    output_cost = (output_tokens / 1_000_000.0) * pricing["output_per_million"]
    return float(input_cost + output_cost)


def compute_net_cost(
    h_useful: float,
    model: str = "gpt-4o",
    c_infra_per_req: float = 0.00002,  # ~$20/mo Redis server across 1M requests
    input_tokens: int = 500,
    output_tokens: int = 300,
) -> dict[str, float]:
    """
    Compute net cost per request and savings percentage.

    Net Cost = (1 - h_useful) * C_origin + C_infra
    """
    c_origin = compute_origin_cost(model, input_tokens, output_tokens)
    net_cost = (1.0 - h_useful) * c_origin + c_infra_per_req
    savings_usd = c_origin - net_cost
    savings_pct = (savings_usd / c_origin) * 100.0 if c_origin > 0 else 0.0

    return {
        "origin_cost_per_req_usd": c_origin,
        "infra_cost_per_req_usd": c_infra_per_req,
        "net_cost_per_req_usd": net_cost,
        "savings_per_req_usd": savings_usd,
        "savings_percentage": savings_pct,
    }


def compute_net_latency(
    h_useful: float,
    l_hit_ms: float,
    l_miss_overhead_ms: float,
    l_origin_ms: float = 1200.0,
) -> float:
    """
    Compute net average latency experienced by callers.

    Net Latency = h_useful * L_hit + (1 - h_useful) * (L_miss_overhead + L_origin)
    """
    miss_rate = 1.0 - h_useful
    return float(h_useful * l_hit_ms + miss_rate * (l_miss_overhead_ms + l_origin_ms))


def compute_break_even_hit_rate(c_origin: float, c_infra: float) -> float:
    """Compute minimum useful hit rate to achieve financial break-even."""
    if c_origin <= 0:
        return 0.0
    return float(min(1.0, max(0.0, c_infra / c_origin)))


def generate_cost_analysis(
    h_useful: float = 0.50,
    l_hit_ms: float = 15.0,
    l_miss_overhead_ms: float = 2.5,
) -> dict[str, Any]:
    """Generate comprehensive cost savings analysis across supported model tiers."""
    analysis: dict[str, Any] = {}
    for model_name, info in MODEL_PRICING.items():
        cost_data = compute_net_cost(h_useful, model=model_name)
        net_lat = compute_net_latency(
            h_useful,
            l_hit_ms,
            l_miss_overhead_ms,
            l_origin_ms=info["typical_latency_ms"],
        )
        latency_reduction_pct = (
            ((info["typical_latency_ms"] - net_lat) / info["typical_latency_ms"]) * 100.0
            if info["typical_latency_ms"] > 0
            else 0.0
        )
        break_even = compute_break_even_hit_rate(
            cost_data["origin_cost_per_req_usd"],
            cost_data["infra_cost_per_req_usd"],
        )

        analysis[model_name] = {
            "model": model_name,
            "origin_cost_usd": cost_data["origin_cost_per_req_usd"],
            "net_cost_usd": cost_data["net_cost_per_req_usd"],
            "savings_usd_per_100k_reqs": cost_data["savings_per_req_usd"] * 100_000.0,
            "savings_percentage": cost_data["savings_percentage"],
            "baseline_latency_ms": info["typical_latency_ms"],
            "net_mean_latency_ms": net_lat,
            "latency_reduction_pct": latency_reduction_pct,
            "break_even_hit_rate": break_even,
        }
    return analysis
