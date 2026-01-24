from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from typing import Callable
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
        self._ws_lock = threading.Lock()  # 保护websocket的并发访问

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
        # 使用锁保证websocket的串行访问，避免并发冲突
        with self._ws_lock:
            data = self._packer.pack(obs)
            self._ws.send(data)
            resp = self._ws.recv()
            if isinstance(resp, str):
                raise RuntimeError(f"Policy server error:\n{resp}")
            decoded: dict = msgpack_numpy.unpackb(resp)
            
            # # 调试：打印policy server返回的原始数据
            # print("=" * 80)
            # print("🔍 [RemotePolicy.raw_step] Policy server 返回的原始数据:")
            # for k, v in decoded.items():
            #     if hasattr(v, 'shape'):
            #         print(f"  {k}: shape={v.shape}, dtype={v.dtype}")
            #     else:
            #         print(f"  {k}: type={type(v)}, value={v}")
            # print("=" * 80)
            
            return RemotePolicyResult(raw=decoded)

    def step(self, obs: dict) -> Any:
        """Return an action suitable to feed into env.step(action)."""
        raw_result = self.raw_step(obs).raw
        selected_action = self._select_action(raw_result)
        
        # # 调试：打印_select_action之后的结果
        # print("=" * 80)
        # print("🔍 [RemotePolicy.step] _select_action 之后的结果:")
        # if hasattr(selected_action, 'shape'):
        #     print(f"  shape={selected_action.shape}, dtype={selected_action.dtype}")
        # else:
        #     print(f"  type={type(selected_action)}")
        # print("=" * 80)
        
        return selected_action

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


