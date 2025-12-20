from __future__ import annotations

from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

import numpy as np


try:  # prefer gymnasium
    import gymnasium as _gym  # type: ignore
except Exception:  # noqa: BLE001
    _gym = None  # type: ignore

if _gym is None:
    try:
        import gym as _gym  # type: ignore
    except Exception:  # noqa: BLE001
        _gym = None  # type: ignore


def _require_gym():
    if _gym is None:
        raise RuntimeError("缺少 gym/gymnasium 依赖：RemoteEnv 需要 gym 的 spaces 来构造 observation_space/action_space。")
    return _gym


def _spaces():
    return _require_gym().spaces


class AnySpace(_spaces().Space):  # type: ignore[misc]
    """A permissive Space for values that don't have a clean gym space (e.g. prompt strings)."""

    def __init__(self, *, seed: Optional[int] = None) -> None:
        super().__init__(shape=(), dtype=object, seed=seed)

    def sample(self, mask: Any = None) -> Any:  # noqa: ANN401
        del mask
        return None

    def contains(self, x: Any) -> bool:  # noqa: ANN401
        return True


def space_from_spec(spec: Dict[str, Any]):
    """Build a gym Space from a JSON/msgpack-friendly spec dict."""
    spaces = _spaces()
    t = spec.get("type")
    if t == "Box":
        low_spec = spec["low"]
        high_spec = spec["high"]
        shape = tuple(spec["shape"])
        dtype = np.dtype(spec.get("dtype", "float32"))
        # gymnasium.Box behavior:
        # - if low/high are Python scalars, it can broadcast to `shape`
        # - if low/high are ndarrays, their shapes must match `shape`
        if isinstance(low_spec, (int, float)) and isinstance(high_spec, (int, float)):
            low = low_spec
            high = high_spec
        else:
            low = np.asarray(low_spec)
            high = np.asarray(high_spec)
        return spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
    if t == "Discrete":
        return spaces.Discrete(int(spec["n"]))
    if t == "MultiDiscrete":
        return spaces.MultiDiscrete(np.asarray(spec["nvec"], dtype=np.int64))
    if t == "MultiBinary":
        return spaces.MultiBinary(int(spec["n"]))
    if t == "Tuple":
        return spaces.Tuple(tuple(space_from_spec(s) for s in spec["spaces"]))
    if t == "Dict":
        return spaces.Dict({k: space_from_spec(v) for k, v in spec["spaces"].items()})
    if t == "Text":
        Text = getattr(spaces, "Text", None)
        if Text is None:
            return AnySpace()
        return Text(max_length=int(spec.get("max_length", 256)))
    if t == "Any":
        return AnySpace()
    raise ValueError(f"Unknown space spec type: {t}")


def libero_default_space_specs(*, resize_size: int, state_dim: int, prompt_max_length: int = 256) -> Tuple[Dict, Dict]:
    """Build observation/action space specs for the Libero env server output."""
    # observation fields produced by libero_env_server.py
    obs_spec = {
        "type": "Dict",
        "spaces": {
            "observation/image": {
                "type": "Box",
                "low": 0,
                "high": 255,
                "shape": [resize_size, resize_size, 3],
                "dtype": "uint8",
            },
            "observation/wrist_image": {
                "type": "Box",
                "low": 0,
                "high": 255,
                "shape": [resize_size, resize_size, 3],
                "dtype": "uint8",
            },
            "observation/state": {
                "type": "Box",
                "low": -np.inf,
                "high": np.inf,
                "shape": [state_dim],
                "dtype": "float32",
            },
            # prompt is a string; use Text if available, else Any.
            "prompt": {"type": "Text", "max_length": prompt_max_length},
        },
    }
    act_spec = {
        "type": "Box",
        "low": -1.0,
        "high": 1.0,
        "shape": [7],
        "dtype": "float32",
    }
    return obs_spec, act_spec


