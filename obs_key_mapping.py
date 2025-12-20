from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Dict, MutableMapping, TypeVar

T = TypeVar("T")


def rename_obs_dict_keys(
    obs: T,
    mapping: Mapping[str, str],
    *,
    strict: bool = False,
) -> T:
    """重命名 observation dict 的 key，并返回一个新 dict（不修改输入）。

    - **obs 不是 dict/Mapping**：原样返回
    - **strict=False**：未出现在 mapping 的 key 保持不变
    - **strict=True**：遇到未出现在 mapping 的 key 直接报错
    - **冲突检测**：如果两个旧 key 映射到同一个新 key，会报错
    """
    if not isinstance(obs, Mapping):
        return obs

    mp: Dict[str, str] = dict(mapping)
    out: MutableMapping[str, Any] = {}
    for k, v in obs.items():
        if strict and k not in mp:
            raise KeyError(f"strict=True: key {k!r} not found in mapping")
        nk = mp.get(k, k)
        if nk in out:
            raise ValueError(f"key rename collision: {k!r} -> {nk!r} duplicates an existing key")
        if nk != False:
            out[nk] = v
    return out  # type: ignore[return-value]


def rename_dict_space_keys(
    space: T,
    mapping: Mapping[str, str],
    *,
    strict: bool = False,
) -> T:
    """重命名 Dict observation_space 的 key，并返回一个新 space（不修改输入）。

    兼容 gym / gymnasium：会沿用原始 space 的 class 来构造新的 Dict space，
    避免 gym 与 gymnasium 的 Space 混用导致的断言错误。

    - 如果 `space` 没有 `.spaces` 或 `.spaces` 不是 Mapping，则原样返回 `space`
    - strict 语义同 `rename_obs_dict_keys`
    """
    spaces = getattr(space, "spaces", None)
    if not isinstance(spaces, Mapping):
        return space

    mp: Dict[str, str] = dict(mapping)
    new_spaces: Dict[str, Any] = {}
    for k, v in spaces.items():
        if strict and k not in mp:
            raise KeyError(f"strict=True: key {k!r} not found in mapping")
        nk = mp.get(k, k)
        if nk in new_spaces:
            raise ValueError(f"space key rename collision: {k!r} -> {nk!r} duplicates an existing key")
        if nk != False:
            new_spaces[nk] = v

    return space.__class__(new_spaces)  # type: ignore[return-value]


