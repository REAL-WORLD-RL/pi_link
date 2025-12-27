from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Protocol, Tuple, Sequence

import numpy as np

from pi_link.obs_key_mapping import rename_dict_space_keys, rename_obs_dict_keys

import concurrent.futures as _futures
from collections.abc import Mapping as _Mapping

import gym


class EnvLike(Protocol):
    """Minimal env protocol for wrappers (works with RemoteEnv and gym envs)."""

    observation_space: Any
    action_space: Any

    def reset(self, **kwargs) -> Any: ...

    def step(self, action: Any) -> Any: ...

    def close(self) -> Any: ...


def _extract_obs_from_reset(reset_out: Any) -> Any:
    # gymnasium style: (obs, info)
    if isinstance(reset_out, tuple) and len(reset_out) == 2:
        return reset_out[0]
    return reset_out


def _extract_obs_from_step(step_out: Any) -> Any:
    if isinstance(step_out, tuple) and len(step_out) >= 1:
        return step_out[0]
    return None


def _maybe_clip_to_action_space(action: np.ndarray, action_space: Any) -> np.ndarray:
    low = getattr(action_space, "low", None)
    high = getattr(action_space, "high", None)
    if low is None or high is None:
        return action
    return np.clip(action, low, high)


def _to_gym_space(space: Any) -> Any:
    """Convert a gymnasium space (or gym-like space) into a gym space.

    SERL replay buffer code uses `gym.spaces.*` and checks `isinstance(x, gym.spaces.Space)`.
    If upstream env provides gymnasium spaces, we convert them here at the env boundary.
    """
    if space is None:
        return None
    if isinstance(space, gym.spaces.Space):
        return space

    # Dict-like
    spaces = getattr(space, "spaces", None)
    if isinstance(spaces, _Mapping):
        return gym.spaces.Dict({k: _to_gym_space(v) for k, v in spaces.items()})

    # Box-like
    if hasattr(space, "low") and hasattr(space, "high") and hasattr(space, "shape") and hasattr(space, "dtype"):
        low = np.asarray(space.low)
        high = np.asarray(space.high)
        return gym.spaces.Box(low=low, high=high, shape=tuple(space.shape), dtype=space.dtype)

    # Discrete-like
    if hasattr(space, "n") and isinstance(getattr(space, "n"), (int, np.integer)):
        return gym.spaces.Discrete(int(space.n))

    # MultiDiscrete-like
    if hasattr(space, "nvec"):
        return gym.spaces.MultiDiscrete(np.asarray(space.nvec, dtype=np.int64))

    # MultiBinary-like
    if getattr(space, "__class__", None) is not None and space.__class__.__name__ == "MultiBinary" and hasattr(space, "n"):
        return gym.spaces.MultiBinary(int(space.n))

    # Fallback: leave it as-is (may still fail downstream if stored in replay buffers).
    return space


def _add_leading_dim_to_value(
    v: Any,
    *,
    axis: int = 0,
    only_if_ndim: Optional[Tuple[int, ...]] = (3,),
) -> Any:
    """Best-effort: add a leading dimension to array-like observation values.

    - If v is a dict-like mapping, recursively processes values.
    - If v is numpy array (or array-like), np.expand_dims(v, axis).
    - If only_if_ndim is not None, only expand when v.ndim is in that set.

    This is intentionally conservative by default (only images: ndim==3).
    """
    if isinstance(v, dict):
        return {k: _add_leading_dim_to_value(vv, axis=axis, only_if_ndim=only_if_ndim) for k, vv in v.items()}
    if v is None:
        return None
    try:
        arr = np.asarray(v)
        if only_if_ndim is not None:
            if arr.ndim == 0 or arr.ndim not in only_if_ndim:
                return v
        return np.expand_dims(arr, axis=axis)
    except Exception:
        return v


def _add_leading_dim_to_box_space(space: Any, *, axis: int = 0) -> Any:
    """Add a leading dimension to a gym Box-like space."""
    if not isinstance(space, gym.spaces.Box):
        return space
    low = np.expand_dims(np.asarray(space.low), axis=axis)
    high = np.expand_dims(np.asarray(space.high), axis=axis)
    return gym.spaces.Box(low=low, high=high, dtype=space.dtype)


def _add_leading_dim_to_space(
    space: Any,
    *,
    axis: int = 0,
    keys: Optional[Sequence[str]] = None,
    skip_keys: Sequence[str] = ("state",),
    only_if_ndim: Optional[Tuple[int, ...]] = (3,),
) -> Any:
    """Recursively add a leading dim to selected Dict space entries."""
    if space is None:
        return None
    if isinstance(space, gym.spaces.Dict):
        new_spaces = {}
        for k, v in space.spaces.items():
            if k in skip_keys:
                new_spaces[k] = v
                continue
            if keys is not None and k not in keys:
                new_spaces[k] = v
                continue
            # Only apply to Box entries that match only_if_ndim (default: (H,W,C) images).
            if isinstance(v, gym.spaces.Box) and (only_if_ndim is None or len(v.shape) in only_if_ndim):
                new_spaces[k] = _add_leading_dim_to_box_space(v, axis=axis)
            else:
                # Recurse for nested Dicts; leave others untouched.
                new_spaces[k] = _add_leading_dim_to_space(
                    v, axis=axis, keys=None, skip_keys=(), only_if_ndim=only_if_ndim
                )
        return gym.spaces.Dict(new_spaces)
    return space


@dataclass
class AddLeadingDimObsWrapper:
    """Env wrapper: unconditionally add a leading (stack) dimension to observation values."""

    env: EnvLike
    axis: int = 0

    def __post_init__(self) -> None:
        obs_space = getattr(self.env, "observation_space", None)
        obs_space = _to_gym_space(obs_space)
        self.observation_space = _add_leading_dim_to_space(
            obs_space,
            axis=self.axis,
            keys=None,
            skip_keys=(),
            only_if_ndim=None,
        )
        self.action_space = _to_gym_space(getattr(self.env, "action_space", None))

    def reset(self, **kwargs) -> Any:
        out = self.env.reset(**kwargs)
        # gymnasium style: (obs, info)
        if isinstance(out, tuple) and len(out) == 2:
            obs, info = out
            return _add_leading_dim_to_value(obs, axis=self.axis, only_if_ndim=None), info
        return _add_leading_dim_to_value(out, axis=self.axis, only_if_ndim=None)

    def step(self, action: Any) -> Any:
        out = self.env.step(action)
        # gym style: (obs, reward, done, info)
        if isinstance(out, tuple) and len(out) == 4:
            obs, reward, done, info = out
            return _add_leading_dim_to_value(obs, axis=self.axis, only_if_ndim=None), reward, done, info
        # gymnasium style: (obs, reward, terminated, truncated, info)
        if isinstance(out, tuple) and len(out) == 5:
            obs, reward, terminated, truncated, info = out
            return _add_leading_dim_to_value(obs, axis=self.axis, only_if_ndim=None), reward, terminated, truncated, info
        return out

    def close(self) -> Any:
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


def _rename_reset_out(out: Any, mapping: Mapping[str, Any], strict: bool) -> Any:
    # gymnasium style: (obs, info)
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return rename_obs_dict_keys(obs, mapping, strict=strict), info
    # gym / RemoteEnv style: obs only
    return rename_obs_dict_keys(out, mapping, strict=strict)


def _rename_step_out(out: Any, mapping: Mapping[str, Any], strict: bool) -> Any:
    # gym style: (obs, reward, done, info)
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return rename_obs_dict_keys(obs, mapping, strict=strict), reward, done, info
    # gymnasium style: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return (
            rename_obs_dict_keys(obs, mapping, strict=strict),
            reward,
            terminated,
            truncated,
            info,
        )
    return out


@dataclass
class ObsKeyRemapWrapper:
    """Env wrapper: remap observation dict keys at the env boundary.

    - Updates `observation_space` (if it's a Dict-like space with `.spaces` Mapping)
    - Renames `obs` returned by `reset()` and `step()`

    This is an env-only wrapper: it doesn't touch any policy.
    """

    env: EnvLike
    mapping: Mapping[str, Any]
    strict: bool = False

    def __post_init__(self) -> None:
        # 1) Remap observation_space keys
        obs_space = getattr(self.env, "observation_space", None)
        remapped_obs_space = rename_dict_space_keys(obs_space, self.mapping, strict=self.strict)
        # 2) Convert to gym spaces so SERL replay buffers (gym-based) don't trip on gymnasium spaces
        self.observation_space = _to_gym_space(remapped_obs_space)

        act_space = getattr(self.env, "action_space", None)
        self.action_space = _to_gym_space(act_space)

    def reset(self, **kwargs) -> Any:
        return _rename_reset_out(self.env.reset(**kwargs), self.mapping, self.strict)

    def step(self, action: Any) -> Any:
        return _rename_step_out(self.env.step(action), self.mapping, self.strict)

    def close(self) -> Any:
        if hasattr(self.env, "close"):
            return self.env.close()
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


@dataclass
class AddPolicyActionWrapper:
    """Env wrapper: add a policy-produced action to the caller-provided action.

    Intended usage:
    - Wrap env with this wrapper (policy is evaluated on the wrapper's cached last obs)
    - Caller passes ONLY residual/delta action into `step(delta)`
    - Wrapper executes: final_action = policy_action(last_obs) + delta

    This is an env-only wrapper: it doesn't wrap/modify the policy object; it just calls it.
    Compatible with `pi_link.RemoteEnv` (duck-typed).
    """

    env: EnvLike
    base_policy: Any  # expects callable(obs)->action (e.g. RemotePolicy.step)
    clip_action: bool = False
    allow_none_delta: bool = True
    prefetch_base_action: bool = True

    def __post_init__(self) -> None:
        self.observation_space = getattr(self.env, "observation_space", None)
        self.action_space = getattr(self.env, "action_space", None)
        self._last_obs: Any = None
        self._base_executor: Optional[_futures.ThreadPoolExecutor] = None
        self._base_future: Optional[_futures.Future] = None
        self._base_future_obs: Any = None

        if self.prefetch_base_action:
            # Single worker is enough: we only ever want the latest base action.
            self._base_executor = _futures.ThreadPoolExecutor(max_workers=1)

    def _prefetch(self, obs: Any) -> None:
        """Kick off base_policy(obs) in the background."""
        if not self.prefetch_base_action or self._base_executor is None:
            return
        # If we already prefetched for this exact obs object, do nothing.
        if self._base_future is not None and self._base_future_obs is obs:
            return
        self._base_future_obs = obs
        self._base_future = self._base_executor.submit(self.base_policy, obs)

    def _get_base_action(self) -> np.ndarray:
        """Get base action for the current cached obs (possibly waiting for a prefetch)."""
        if self._last_obs is None:
            raise RuntimeError("AddPolicyActionWrapper: base action requested before reset().")

        if self.prefetch_base_action and self._base_future is not None and self._base_future_obs is self._last_obs:
            base_action = self._base_future.result()
        else:
            # Fallback: no prefetch available; do it synchronously.
            base_action = self.base_policy(self._last_obs)
        return np.asarray(base_action)

    def reset(self, **kwargs) -> Any:
        # If the base policy caches action chunks (e.g. RemotePolicy), clear it at episode start.
        if hasattr(self.base_policy, "reset"):
            try:
                self.base_policy.reset()
            except TypeError:
                # Some policies may define reset() with a different signature; ignore.
                pass
        out = self.env.reset(**kwargs)
        self._last_obs = _extract_obs_from_reset(out)
        # As soon as we have an obs, prefetch the base action for the *next* step call.
        self._prefetch(self._last_obs)
        return out

    def step(self, delta_action: Any = None) -> Any:
        if self._last_obs is None:
            raise RuntimeError(
                "AddPolicyActionWrapper.step() called before reset()."
            )

        if delta_action is None and not self.allow_none_delta:
            raise ValueError("delta_action is None but allow_none_delta=False")

        base_action = self._get_base_action()
        final_action = base_action + np.asarray(delta_action) if delta_action is not None else base_action
        # print('exec compose action: base action + delta action', base_action, delta_action, '->', final_action)

        if self.clip_action and self.action_space is not None:
            final_action = _maybe_clip_to_action_space(final_action, self.action_space)

        out = self.env.step(final_action)
        self._last_obs = _extract_obs_from_step(out)
        # If episode continues, immediately prefetch base action for the next step while
        # the caller computes/produces the next delta action.
        done = False
        if isinstance(out, tuple):
            if len(out) == 4:
                # (obs, reward, done, info)
                done = bool(out[2])
            elif len(out) == 5:
                # (obs, reward, terminated, truncated, info)
                done = bool(out[2] or out[3])
        if not done:
            self._prefetch(self._last_obs)
        else:
            self._base_future = None
            self._base_future_obs = None
        return out

    def close(self) -> Any:
        if hasattr(self.env, "close"):
            try:
                return self.env.close()
            finally:
                if self._base_executor is not None:
                    self._base_executor.shutdown(wait=False, cancel_futures=True)
                    self._base_executor = None
        return None

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)


