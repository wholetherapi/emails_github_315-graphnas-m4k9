"""Search space definitions for GraphNAS."""

AGGREGATORS = ["mean", "sum", "max", "attention"]


def build_candidate_ops():
    """Return the candidate message-passing operations."""
    return [
        {"aggregator": agg, "hops": hops}
        for agg in AGGREGATORS
        for hops in (1, 2, 3)
    ]
