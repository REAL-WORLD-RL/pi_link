"""Tiny msgpack helpers with NumPy support (wire-compatible with OpenPI's msgpack_numpy).

We keep this local to avoid depending on internal openpi/openpi-client modules.
The encoding format is simple dictionaries that msgpack can carry across languages.
"""

from __future__ import annotations

import functools
from typing import Any

import msgpack
import numpy as np


def _pack_array(obj: Any) -> Any:
    # Disallow object/void/complex dtypes (same rationale as openpi_client.msgpack_numpy).
    if isinstance(obj, (np.ndarray, np.generic)) and obj.dtype.kind in ("V", "O", "c"):
        raise ValueError(f"Unsupported dtype: {obj.dtype}")

    if isinstance(obj, np.ndarray):
        return {
            b"__ndarray__": True,
            b"data": obj.tobytes(),
            b"dtype": obj.dtype.str,
            b"shape": obj.shape,
        }

    if isinstance(obj, np.generic):
        return {
            b"__npgeneric__": True,
            b"data": obj.item(),
            b"dtype": obj.dtype.str,
        }

    return obj


def _unpack_array(obj: Any) -> Any:
    if isinstance(obj, dict) and b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])

    if isinstance(obj, dict) and b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])

    return obj


Packer = functools.partial(msgpack.Packer, default=_pack_array, use_bin_type=True)
packb = functools.partial(msgpack.packb, default=_pack_array, use_bin_type=True)

Unpacker = functools.partial(msgpack.Unpacker, object_hook=_unpack_array, raw=False)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack_array, raw=False)


