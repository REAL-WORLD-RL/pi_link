from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable

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
        select_action: Callable[[dict], Any] | None = None,
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
        self._select_action = select_action or self._default_select_action

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
        self._ws.close()

    def raw_step(self, obs: dict) -> RemotePolicyResult:
        data = self._packer.pack(obs)
        self._ws.send(data)
        resp = self._ws.recv()
        if isinstance(resp, str):
            raise RuntimeError(f"Policy server error:\n{resp}")
        decoded: dict = msgpack_numpy.unpackb(resp)
        return RemotePolicyResult(raw=decoded)

    def step(self, obs: dict) -> Any:
        """Return an action suitable to feed into env.step(action)."""
        return self._select_action(self.raw_step(obs).raw)

    @staticmethod
    def _default_select_action(result: dict) -> Any:
        # Common OpenPI output shape: {"actions": [T, ...], ...}
        if "actions" in result:
            actions = result["actions"]
            try:
                return actions[0]
            except Exception:
                return actions
        if "action" in result:
            return result["action"]
        return result


