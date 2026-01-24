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


def _rename_reset_out(out: Any, mapping: Mapping[str, Any], strict: bool, filter_unmapped: bool = False) -> Any:
    # gymnasium style: (obs, info)
    if isinstance(out, tuple) and len(out) == 2:
        obs, info = out
        return rename_obs_dict_keys(obs, mapping, strict=strict, filter_unmapped=filter_unmapped), info
    # gym / RemoteEnv style: obs only
    return rename_obs_dict_keys(out, mapping, strict=strict, filter_unmapped=filter_unmapped)


def _rename_step_out(out: Any, mapping: Mapping[str, Any], strict: bool, filter_unmapped: bool = False) -> Any:
    # gym style: (obs, reward, done, info)
    if isinstance(out, tuple) and len(out) == 4:
        obs, reward, done, info = out
        return rename_obs_dict_keys(obs, mapping, strict=strict, filter_unmapped=filter_unmapped), reward, done, info
    # gymnasium style: (obs, reward, terminated, truncated, info)
    if isinstance(out, tuple) and len(out) == 5:
        obs, reward, terminated, truncated, info = out
        return (
            rename_obs_dict_keys(obs, mapping, strict=strict, filter_unmapped=filter_unmapped),
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
    filter_unmapped: bool = False

    def __post_init__(self) -> None:
        # 1) Remap observation_space keys
        obs_space = getattr(self.env, "observation_space", None)
        remapped_obs_space = rename_dict_space_keys(
            obs_space, self.mapping, strict=self.strict, filter_unmapped=self.filter_unmapped
        )
        # 2) Convert to gym spaces so SERL replay buffers (gym-based) don't trip on gymnasium spaces
        self.observation_space = _to_gym_space(remapped_obs_space)

        act_space = getattr(self.env, "action_space", None)
        self.action_space = _to_gym_space(act_space)

    def reset(self, **kwargs) -> Any:
        return _rename_reset_out(self.env.reset(**kwargs), self.mapping, self.strict, self.filter_unmapped)

    def step(self, action: Any) -> Any:
        return _rename_step_out(self.env.step(action), self.mapping, self.strict, self.filter_unmapped)

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
    obs_mapping: Optional[_Mapping[str, str]] = None  # 为 base_policy 映射 obs keys
    filter_unmapped_obs: bool = False  # 是否过滤未映射的 obs keys
    remove_time_dim_for_policy: bool = False  # 是否为 policy 移除时间维度 (去掉 T 维度)
    unbatch_obs_for_policy: bool = False  # 是否为 policy 移除 batch 维度 (只发送第一个样本)
    broadcast_policy_action: bool = False  # 是否将 policy 返回的单个 action 广播到所有 batch

    def __post_init__(self) -> None:
        self.observation_space = getattr(self.env, "observation_space", None)
        self.action_space = getattr(self.env, "action_space", None)
        self._last_obs: Any = None
        self._base_executor: Optional[_futures.ThreadPoolExecutor] = None
        self._base_future: Optional[_futures.Future] = None
        self._base_future_obs: Any = None
        self._batch_size: int = 1  # 记录 batch size，用于 broadcast action
        
        # Action buffer for action chunking
        self._action_buffer: Optional[np.ndarray] = None  # shape: (batch_size, action_horizon, action_dim)
        self._action_buffer_index: int = 0  # 当前应该取 buffer 中的第几个 action

        if self.prefetch_base_action:
            # Single worker is enough: we only ever want the latest base action.
            self._base_executor = _futures.ThreadPoolExecutor(max_workers=1)

    def _transform_obs_for_policy(self, obs: Any) -> Any:
        """Transform obs for base_policy if obs_mapping is provided."""
        if self.obs_mapping is None and not self.remove_time_dim_for_policy and not self.unbatch_obs_for_policy:
            return obs
        
        # 调试：打印转换前后的 obs keys
        if isinstance(obs, dict):
            print(f"🔍 [AddPolicyActionWrapper] 转换前 obs keys: {sorted(obs.keys())}")
            for k, v in obs.items():
                if hasattr(v, 'shape'):
                    print(f"    {k}: shape={v.shape}, dtype={v.dtype}")
            print(f"🔍 [AddPolicyActionWrapper] obs_mapping: {self.obs_mapping}")
            print(f"🔍 [AddPolicyActionWrapper] filter_unmapped_obs: {self.filter_unmapped_obs}")
            print(f"🔍 [AddPolicyActionWrapper] remove_time_dim_for_policy: {self.remove_time_dim_for_policy}")
            print(f"🔍 [AddPolicyActionWrapper] unbatch_obs_for_policy: {self.unbatch_obs_for_policy}")
        
        transformed_obs = obs
        
        # 步骤1: 如果配置了 obs_mapping，先进行 key 映射/过滤
        if self.obs_mapping is not None:
            transformed_obs = rename_obs_dict_keys(
                transformed_obs, 
                self.obs_mapping, 
                strict=False, 
                filter_unmapped=self.filter_unmapped_obs
            )
        
        # 步骤2: 如果需要，移除 batch 维度 (只取第一个样本)
        if self.unbatch_obs_for_policy:
            if isinstance(transformed_obs, dict):
                new_obs = {}
                for k, v in transformed_obs.items():
                    if isinstance(v, np.ndarray) and len(v.shape) > 0:
                        # 记录 batch size（从第一个数组key获取）
                        if self._batch_size == 1 and v.shape[0] > 1:
                            self._batch_size = v.shape[0]
                            print(f"🔍 [记录batch_size] batch_size={self._batch_size}")
                        # 取第一个样本: (B, ...) -> (...)
                        new_obs[k] = v[0]
                        print(f"🔍 [移除batch维度] key={k}, 原始shape={v.shape}, 新shape={new_obs[k].shape}")
                    else:
                        new_obs[k] = v
                transformed_obs = new_obs
            elif isinstance(transformed_obs, np.ndarray) and len(transformed_obs.shape) > 0:
                if self._batch_size == 1 and transformed_obs.shape[0] > 1:
                    self._batch_size = transformed_obs.shape[0]
                transformed_obs = transformed_obs[0]
        
        # 步骤3: 如果需要，移除时间维度 (squeeze 掉 size=1 的维度)
        if self.remove_time_dim_for_policy:
            if isinstance(transformed_obs, dict):
                new_obs = {}
                for k, v in transformed_obs.items():
                    if isinstance(v, np.ndarray):
                        print(f"🔍 [移除时间维度] key={k}, 原始 shape={v.shape}")
                        
                        # 策略：假设 shape 是 (B, T, ...) 或 (T, B, ...)
                        # 我们需要 squeeze 掉 T=1 的维度
                        
                        # 检查前两个维度哪个是1
                        if len(v.shape) >= 2:
                            if v.shape[0] == 1:
                                # shape 是 (1, B, ...) -> squeeze 第0维
                                new_v = np.squeeze(v, axis=0)
                                print(f"🔍 [移除时间维度] squeeze axis=0, 新 shape={new_v.shape}")
                                new_obs[k] = new_v
                            elif v.shape[1] == 1:
                                # shape 是 (B, 1, ...) -> squeeze 第1维
                                new_v = np.squeeze(v, axis=1)
                                print(f"🔍 [移除时间维度] squeeze axis=1, 新 shape={new_v.shape}")
                                new_obs[k] = new_v
                            else:
                                # 没有维度为1，保持不变
                                print(f"🔍 [移除时间维度] 没有维度为1，保持不变")
                                new_obs[k] = v
                        elif len(v.shape) == 1 and v.shape[0] == 1:
                            # 1D 数组且长度为1
                            new_obs[k] = np.squeeze(v)
                        else:
                            new_obs[k] = v
                    else:
                        new_obs[k] = v
                transformed_obs = new_obs
            elif isinstance(transformed_obs, np.ndarray):
                if 1 in transformed_obs.shape:
                    transformed_obs = np.squeeze(transformed_obs)
        
        # 调试：打印转换后的 obs keys（发送给 policy 的完整内容）
        print("=" * 80)
        print("🔍 [最终发送给 RemotePolicy 的 obs 内容]")
        if isinstance(transformed_obs, dict):
            print(f"obs keys: {sorted(transformed_obs.keys())}")
            for k, v in transformed_obs.items():
                if hasattr(v, 'shape'):
                    print(f"  {k}:")
                    print(f"    - type: {type(v)}")
                    print(f"    - shape: {v.shape}")
                    print(f"    - dtype: {v.dtype}")
                elif isinstance(v, str):
                    print(f"  {k}:")
                    print(f"    - type: {type(v)}")
                    print(f"    - value: {repr(v)}")
                elif isinstance(v, list):
                    print(f"  {k}:")
                    print(f"    - type: list, len={len(v)}")
                    if len(v) > 0:
                        print(f"    - first element: {repr(v[0])}")
                else:
                    print(f"  {k}:")
                    print(f"    - type: {type(v)}")
                    print(f"    - value: {repr(v)}")
        print("=" * 80)
        
        return transformed_obs

    def _prefetch(self, obs: Any) -> None:
        """Kick off base_policy(obs) in the background."""
        if not self.prefetch_base_action or self._base_executor is None:
            return
        
        # 如果 action buffer 还有足够的 action (>=2个)，不需要 prefetch
        if self._action_buffer is not None and self._action_buffer_index < self._action_buffer.shape[1] - 1:
            # buffer 还有至少2个action，不需要prefetch
            return
        
        # If we already prefetched for this exact obs object, do nothing.
        if self._base_future is not None and self._base_future_obs is obs:
            return
        self._base_future_obs = obs
        # 在提交给 policy 之前先转换 obs
        policy_obs = self._transform_obs_for_policy(obs)
        self._base_future = self._base_executor.submit(self.base_policy, policy_obs)

    def _get_base_action(self) -> np.ndarray:
        """Get base action for the current cached obs (possibly waiting for a prefetch)."""
        if self._last_obs is None:
            raise RuntimeError("AddPolicyActionWrapper: base action requested before reset().")

        # 检查 action buffer 是否还有可用的 action
        if self._action_buffer is not None and self._action_buffer_index < self._action_buffer.shape[1]:
            # 从 buffer 中取出当前 index 的 action
            action = self._action_buffer[:, self._action_buffer_index, :]  # shape: (batch_size, action_dim)
            self._action_buffer_index += 1
            print(f"🔍 [从 action buffer 取 action] index={self._action_buffer_index-1}, shape={action.shape}")
            return action
        
        # Action buffer 为空或用完了，需要调用 policy 获取新的 action chunk
        print("🔍 [action buffer 为空，调用 policy 获取新的 action chunk]")
        
        if self.prefetch_base_action and self._base_future is not None and self._base_future_obs is self._last_obs:
            base_action = self._base_future.result()
        else:
            # Fallback: no prefetch available; do it synchronously.
            # 在调用 policy 之前先转换 obs
            policy_obs = self._transform_obs_for_policy(self._last_obs)
            base_action = self.base_policy(policy_obs)
        
        # 调试：打印 policy 返回的 action
        print("=" * 80)
        print("🔍 [RemotePolicy 返回的 action]")
        print(f"  - type: {type(base_action)}")
        if hasattr(base_action, 'shape'):
            print(f"  - shape: {base_action.shape}")
            print(f"  - dtype: {base_action.dtype}")
        elif isinstance(base_action, (list, tuple)):
            print(f"  - length: {len(base_action)}")
            if len(base_action) > 0:
                print(f"  - first element type: {type(base_action[0])}")
                if hasattr(base_action[0], 'shape'):
                    print(f"  - first element shape: {base_action[0].shape}")
        print("=" * 80)
        
        base_action_array = np.asarray(base_action)
        
        # 检查 action 的 shape
        if len(base_action_array.shape) == 3:
            # shape: (batch_size, action_horizon, action_dim) - 这是 action chunking
            print(f"🔍 [检测到 action chunking] shape={base_action_array.shape}")
            self._action_buffer = base_action_array
            self._action_buffer_index = 0
            # 取第一个 action
            action = self._action_buffer[:, self._action_buffer_index, :]
            self._action_buffer_index += 1
            print(f"🔍 [初始化 action buffer 并取第一个 action] index=0, shape={action.shape}")
            return action
        elif len(base_action_array.shape) == 2:
            # shape: (batch_size, action_dim) - 单个 action，不需要 buffer
            print(f"🔍 [单个 action，不使用 buffer] shape={base_action_array.shape}")
            self._action_buffer = None
            self._action_buffer_index = 0
            return base_action_array
        else:
            # 其他情况，尝试广播或直接返回
            print(f"⚠️ [未知的 action shape] shape={base_action_array.shape}")
            # 如果需要，将单个 action 广播到所有 batch
            if self.broadcast_policy_action and self._batch_size > 1:
                print(f"🔍 [广播action] 原始shape={base_action_array.shape}, batch_size={self._batch_size}")
                # 如果 action 是 (action_dim,)，广播成 (batch_size, action_dim)
                if len(base_action_array.shape) == 1:
                    base_action_array = np.tile(base_action_array, (self._batch_size, 1))
                print(f"🔍 [广播action] 新shape={base_action_array.shape}")
            
            return base_action_array

    def reset(self, **kwargs) -> Any:
        out = self.env.reset(**kwargs)
        self._last_obs = _extract_obs_from_reset(out)
        
        # 清空 action buffer (reset 时需要重新预测)
        self._action_buffer = None
        self._action_buffer_index = 0
        print("🔍 [reset] 清空 action buffer")
        
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
        # final_action = base_action
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
                done_val = out[2]
                # 处理数组形式的 done (vectorized env)
                if isinstance(done_val, np.ndarray):
                    done = bool(done_val.any())  # 任何一个环境done就算done
                else:
                    done = bool(done_val)
            elif len(out) == 5:
                # (obs, reward, terminated, truncated, info)
                terminated = out[2]
                truncated = out[3]
                # 处理数组形式的 done (vectorized env)
                if isinstance(terminated, np.ndarray) or isinstance(truncated, np.ndarray):
                    done = bool(np.asarray(terminated).any() or np.asarray(truncated).any())
                else:
                    done = bool(terminated or truncated)
        if not done:
            self._prefetch(self._last_obs)
        else:
            # 关键修复：批量环境中任意一个 done，立即清空 action buffer
            # 因为 done 的环境会 auto-reset，旧的 action buffer 不再适用
            self._base_future = None
            self._base_future_obs = None
            self._action_buffer = None
            self._action_buffer_index = 0
            print("🔍 [done detected] 清空 action buffer，下次 step 将重新获取 action chunk")
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


