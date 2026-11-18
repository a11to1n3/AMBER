"""Order-independent scalar random draws for AMBER activations."""

from hashlib import blake2b


DRAW_ALGORITHM = "blake2b-53-v1"


def keyed_uniform(seed: int, step: int, agent_id: int, event: str, index: int) -> float:
    """Map a draw key to an exactly representable uniform value in [0, 1)."""
    key = f"{seed}:{step}:{agent_id}:{index}:".encode("ascii") + event.encode("utf-8")
    digest = blake2b(key, digest_size=8, person=b"ambr-draw-v1").digest()
    return (int.from_bytes(digest, "big") >> 11) / (1 << 53)
