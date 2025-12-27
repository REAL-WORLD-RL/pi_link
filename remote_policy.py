from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any
from typing import Callable
from typing import Deque
from typing import Iterable
import threading

import websockets.sync.client

from pi_link import msgpack_numpy


@dataclass(frozen=True)
class RemotePolicyResult:
    """Decoded raw response dict from the policy server."""

    raw: dict


class RemotePolicy:
    """Remote policy client with a minimal `step(obs)->action` interface.

    Wire protocol (compatible with OpenPI policy server):
    - Connect -> server sends one msgpack metadata frame
    - infer: client sends msgpack(obs) bytes; server replies msgpack(result) bytes
    - server may send a *text* frame on error; we raise RuntimeError in that case
    """

    def __init__(
        self,
        uri_or_host: str = "ws://127.0.0.1:8000",
        port: int | None = None,
        *,
        api_key: str | None = None,
        # Backward-compat: older callers used select_action(result)->single_action.
        # New behavior prefers select_actions(result)->action_chunk.
        select_action: Callable[[dict], Any] | None = None,
        select_actions: Callable[[dict], Any] | None = None,
        connect_timeout_s: float | None = 10.0,
    ) -> None:
        if uri_or_host.startswith("ws"):
            uri = uri_or_host
        else:
            uri = f"ws://{uri_or_host}"
        if port is not None:
            uri += f":{port}"

        self._uri = uri
        self._api_key = api_key
        self._packer = msgpack_numpy.Packer()
        if select_action is not None and select_actions is not None:
            raise ValueError("Pass only one of select_action or select_actions, not both.")
        if select_actions is not None:
            self._select_actions = select_actions
        elif select_action is not None:
            # Adapt single-action selector to a single-element chunk.
            self._select_actions = lambda result: [select_action(result)]
        else:
            self._select_actions = self._default_select_actions
        self._action_queue: Deque[Any] = deque()
        # NOTE: RemotePolicy can be used behind AddPolicyActionWrapper which may call
        # the policy in a background thread (prefetch). Serialize websocket IO + queue ops.
        self._lock = threading.Lock()

        headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
        self._ws = websockets.sync.client.connect(
            self._uri,
            compression=None,
            max_size=None,
            additional_headers=headers,
            open_timeout=connect_timeout_s,
        )
        meta = self._ws.recv()
        if isinstance(meta, str):
            raise RuntimeError(f"Policy server error during handshake:\n{meta}")
        self._metadata: dict = msgpack_numpy.unpackb(meta)

    @property
    def metadata(self) -> dict:
        return self._metadata

    def close(self) -> None:
        with self._lock:
            self._ws.close()

    def raw_step(self, obs: dict) -> RemotePolicyResult:
        with self._lock:
            data = self._packer.pack(obs)
            self._ws.send(data)
            resp = self._ws.recv()
            if isinstance(resp, str):
                raise RuntimeError(f"Policy server error:\n{resp}")
            decoded: dict = msgpack_numpy.unpackb(resp)
            return RemotePolicyResult(raw=decoded)

    def reset(self) -> None:
        """Clear any cached action chunk.

        Call this at the start of each episode (or whenever your obs stream is reset)
        to avoid executing stale queued actions.
        """
        with self._lock:
            self._action_queue.clear()

    def clear_queue(self) -> None:
        """Alias for reset()."""
        self.reset()

    @property
    def queued_actions(self) -> int:
        """Number of cached actions waiting to be executed locally."""
        with self._lock:
            return len(self._action_queue)

    def step(self, obs: dict) -> Any:
        """Return ONE action suitable to feed into env.step(action).

        If a previous remote inference returned an action chunk (sequence),
        we cache it locally and return actions one-by-one without re-contacting
        the server until the cache is exhausted.
        """
        with self._lock:
            if self._action_queue:
                return self._action_queue.popleft()

            # Queue is empty -> request next chunk from server.
            data = self._packer.pack(obs)
            self._ws.send(data)
            resp = self._ws.recv()
            if isinstance(resp, str):
                raise RuntimeError(f"Policy server error:\n{resp}")
            result: dict = msgpack_numpy.unpackb(resp)

            actions_obj = self._select_actions(result)
            actions = self._to_action_list(actions_obj)
            if not actions:
                raise RuntimeError(
                    "RemotePolicy got an empty action chunk from server; "
                    "expected non-empty 'actions' (or a non-empty selection)."
                )
            self._action_queue.extend(actions)
            return self._action_queue.popleft()

    @staticmethod
    def _default_select_actions(result: dict) -> Any:
        """Select the action chunk from the server result.

        Expected OpenPI output shape: {"actions": [T, ...], ...} where T is the horizon.
        """
        if "actions" in result:
            return result["actions"]
        if "action" in result:
            # Single action fallback.
            return [result["action"]]
        # Last-resort: treat the entire result as a single "action".
        return [result]

    @staticmethod
    def _to_action_list(actions_obj: Any) -> list[Any]:
        """Normalize various action container types to a Python list of per-step actions."""
        if actions_obj is None:
            return []

        # numpy/jax arrays: typically (T, action_dim)
        shape = getattr(actions_obj, "shape", None)
        ndim = getattr(actions_obj, "ndim", None)
        if shape is not None and ndim is not None:
            try:
                if int(ndim) == 0:
                    return [actions_obj]
                if int(ndim) == 1:
                    # already a single action vector
                    return [actions_obj]
                # treat first dimension as time/horizon
                t = int(shape[0])
                return [actions_obj[i] for i in range(t)]
            except Exception:
                # Fall through to iterable handling.
                pass

        if isinstance(actions_obj, (list, tuple)):
            return list(actions_obj)

        # Generic iterable (but avoid treating bytes/str as iterable of chars)
        if isinstance(actions_obj, (str, bytes)):
            return [actions_obj]
        if isinstance(actions_obj, Iterable):
            try:
                return list(actions_obj)
            except Exception:
                return [actions_obj]

        return [actions_obj]


