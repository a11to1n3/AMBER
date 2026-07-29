"""Counter-based cross-backend random tape.

Random values for semantic-attestation experiments must not depend on thread
index, event ordering, batching, or backend. Native NumPy/CuPy/FLAME streams
remain allowed only on a separate "native-idiom" performance track.
"""

from __future__ import annotations

import struct
from typing import Iterable, Tuple

# SplitMix64 constants (public domain / widely used counter-based mixer).
_GOLDEN = 0x9E3779B97F4A7C15
_M1 = 0xBF58476D1CE4E5B9
_M2 = 0x94D049BB133111EB
MASK64 = (1 << 64) - 1


def mix64(z: int) -> int:
    z = (z + _GOLDEN) & MASK64
    z = ((z ^ (z >> 30)) * _M1) & MASK64
    z = ((z ^ (z >> 27)) * _M2) & MASK64
    return (z ^ (z >> 31)) & MASK64


def pack_key(
    global_seed: int,
    step: int,
    event_type: int,
    agent_id: int,
    partner_id: int = 0,
    draw_index: int = 0,
) -> int:
    """Fold multi-field event key into one 64-bit counter."""
    x = int(global_seed) & MASK64
    for v in (step, event_type, agent_id, partner_id, draw_index):
        x = mix64(x ^ (int(v) & MASK64))
    return x


def rng64(
    global_seed: int,
    step: int,
    event_type: int,
    agent_id: int,
    partner_id: int = 0,
    draw_index: int = 0,
) -> int:
    return mix64(
        pack_key(global_seed, step, event_type, agent_id, partner_id, draw_index)
    )


def u01(
    global_seed: int,
    step: int,
    event_type: int,
    agent_id: int,
    partner_id: int = 0,
    draw_index: int = 0,
) -> float:
    """Map counter output to U(0,1) via top 53 mantissa bits."""
    u = rng64(global_seed, step, event_type, agent_id, partner_id, draw_index)
    return (u >> 11) * (1.0 / (1 << 53))


def int_range(
    low: int,
    high: int,
    global_seed: int,
    step: int,
    event_type: int,
    agent_id: int,
    partner_id: int = 0,
    draw_index: int = 0,
) -> int:
    """Integer uniform on [low, high) using rejection-free modular reduction.

    For semantic tests we accept slight non-uniformity at power-of-two
    boundaries in exchange for identical cross-language behaviour.
    """
    if high <= low:
        raise ValueError("high must be > low")
    span = high - low
    return low + (rng64(global_seed, step, event_type, agent_id, partner_id, draw_index) % span)


# Event-type tags (stable across workloads)
EVT_RECIPIENT = 1
EVT_DISPLACE_X = 2
EVT_DISPLACE_Y = 3
EVT_INFECTION = 4
EVT_RECOVERY = 5
EVT_PROPOSAL = 6
EVT_PRIORITY = 7
EVT_INIT = 8


def test_vectors() -> list[dict]:
    """Fixed vectors for cross-implementation regression."""
    cases = [
        (0, 0, EVT_RECIPIENT, 0, 0, 0),
        (42, 3, EVT_INFECTION, 7, 11, 0),
        (1, 10, EVT_DISPLACE_X, 99, 0, 2),
        (123456789, 50, EVT_PROPOSAL, 4, 0, 1),
    ]
    out = []
    for args in cases:
        out.append(
            {
                "args": list(args),
                "rng64": rng64(*args),
                "u01": u01(*args),
                "int_0_100": int_range(0, 100, *args),
            }
        )
    return out


if __name__ == "__main__":
    import json
    from pathlib import Path

    vec = test_vectors()
    path = Path(__file__).with_name("test_vectors.json")
    path.write_text(json.dumps(vec, indent=2))
    print(f"wrote {path}")
    for row in vec:
        print(row)
