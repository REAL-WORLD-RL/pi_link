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


def gym_space_to_spec(space: Any) -> Optional[Dict[str, Any]]:  # noqa: ANN401
    """Convert a gym.Space to a JSON-serializable spec dict.
    
    Supports both gym and gymnasium.spaces.
    
    Args:
        space: A gym.spaces.Space or gymnasium.spaces.Space instance
        
    Returns:
        Dict with "type" and other space-specific fields, or None if space is None
    """
    if space is None:
        return None
    
    # Get the space class name to handle both gym and gymnasium
    space_type = type(space).__name__
    
    # Box space
    if space_type == "Box":
        # Convert to Python types for serialization
        low = space.low
        high = space.high
        
        # If bounds are uniform, use scalar
        if np.all(low == low.flat[0]) and np.all(high == high.flat[0]):
            low = float(low.flat[0])
            high = float(high.flat[0])
        else:
            low = low.tolist()
            high = high.tolist()
        
        return {
            "type": "Box",
            "low": low,
            "high": high,
            "shape": list(space.shape),
            "dtype": str(space.dtype),
        }
    
    # Discrete space
    if space_type == "Discrete":
        return {
            "type": "Discrete",
            "n": int(space.n),
        }
    
    # MultiDiscrete space
    if space_type == "MultiDiscrete":
        return {
            "type": "MultiDiscrete",
            "nvec": space.nvec.tolist(),
        }
    
    # Dict space
    if space_type == "Dict":
        return {
            "type": "Dict",
            "spaces": {k: gym_space_to_spec(v) for k, v in space.spaces.items()},
        }
    
    # MultiBinary space
    if space_type == "MultiBinary":
        return {
            "type": "MultiBinary",
            "n": int(space.n),
        }
    
    # Tuple space
    if space_type == "Tuple":
        return {
            "type": "Tuple",
            "spaces": [gym_space_to_spec(s) for s in space.spaces],
        }
    
    # Text space
    if space_type == "Text":
        return {
            "type": "Text",
            "max_length": int(getattr(space, "max_length", 256)),
        }
    
    # Fallback: AnySpace or unknown
    return {"type": "Any"}


def libero_default_space_specs(*, resize_size: int, state_dim: int, prompt_max_length: int = 256) -> Tuple[Dict, Dict]:
    """Build observation/action space specs for the Libero env server output.
    
    This function creates gym.Space objects first, then converts them to specs.
    This approach is more robust and aligns with gym ecosystem conventions.
    
    Args:
        resize_size: Image dimensions after resizing
        state_dim: Dimension of the state vector
        prompt_max_length: Maximum length for text prompts
        
    Returns:
        Tuple of (observation_spec, action_spec) dicts suitable for serialization
    """
    spaces = _spaces()
    
    # Define observation space using gym.Space objects
    obs_space = spaces.Dict({
        "observation/image": spaces.Box(
            low=0,
            high=255,
            shape=(resize_size, resize_size, 3),
            dtype=np.uint8,
        ),
        "observation/wrist_image": spaces.Box(
            low=0,
            high=255,
            shape=(resize_size, resize_size, 3),
            dtype=np.uint8,
        ),
        "observation/state": spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(state_dim,),
            dtype=np.float32,
        ),
        # prompt is a string; use AnySpace for flexibility
        "prompt": AnySpace(),
    })
    
    # Define action space using gym.Space objects
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(7,),
        dtype=np.float32,
    )
    
    # Convert gym.Space objects to serializable specs
    obs_spec = gym_space_to_spec(obs_space)
    action_spec = gym_space_to_spec(action_space)
    
    return obs_spec, action_spec


