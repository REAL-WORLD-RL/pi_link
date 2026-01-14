from __future__ import annotations

from dataclasses import dataclass
import logging
import threading
import time
from typing import Any, Union

import numpy as np
import websockets.sync.client

from pi_link import msgpack_numpy
from pi_link.spaces import space_from_spec


@dataclass(frozen=True)
class RemoteEnvStep:
    """Step result supporting both single env (scalars) and vector env (arrays)."""
    obs: Any
    reward: Union[float, np.ndarray]  # scalar for single env, array for vec env
    terminated: Union[bool, np.ndarray]  # scalar for single env, array for vec env
    truncated: Union[bool, np.ndarray]  # scalar for single env, array for vec env
    info: dict

    @property
    def done(self) -> Union[bool, np.ndarray]:
        """Returns done flag(s). For vec env, returns boolean array."""
        if isinstance(self.terminated, np.ndarray) or isinstance(self.truncated, np.ndarray):
            return self.terminated | self.truncated
        return bool(self.terminated or self.truncated)


class RemoteEnv:
    """Remote environment client with gym-like `reset/step/close`.

    This speaks a tiny request/response protocol over websockets:
    - Connect -> server sends one msgpack metadata frame (dict)
    - Client sends msgpack dicts:
      - {"cmd":"reset","seed":int|None,"options":dict|None,"session_id":str|None}  (optional)
      - {"cmd":"step","action":Any,"session_id":str}  (if session protocol enabled)
      - {"cmd":"close_session","session_id":str}  (optional)
      - {"cmd":"ping","session_id":str} (heartbeat)
      - {"cmd":"close"}
    - Server replies msgpack dicts:
      - reset: {"obs":Any,"info":dict,"session_id":str} (if session protocol enabled)
      - step:  {"obs":Any,"reward":float,"done":bool,"info":dict,"session_id":str} (if session protocol enabled)
        (optionally "terminated"/"truncated")
    """

    def __init__(self, uri_or_host: str = "ws://127.0.0.1:9000", port: int | None = None) -> None:
        if uri_or_host.startswith("ws"):
            uri = uri_or_host
        else:
            uri = f"ws://{uri_or_host}"
        if port is not None:
            uri += f":{port}"

        self._uri = uri
        self._packer = msgpack_numpy.Packer()
        self._ws = websockets.sync.client.connect(self._uri, compression=None, max_size=None)
        
        # Lock for thread-safe socket access (needed for background heartbeat)
        self._ws_lock = threading.RLock()

        meta = self._ws.recv()
        if isinstance(meta, str):
            raise RuntimeError(f"Env server error during handshake:\n{meta}")
        self._metadata: dict = msgpack_numpy.unpackb(meta)
        self._request_id = 0

        # Session protocol (optional). If enabled, server routes requests by session_id.
        session_proto = self._metadata.get("session_protocol") or {}
        self._supports_sessions = bool(session_proto.get("enabled", False))
        self._session_id: str | None = None

        self.observation_space = None
        self.action_space = None
        # If server provides spaces in handshake metadata, construct real gym spaces.
        obs_spec = self._metadata.get("observation_space_spec")
        act_spec = self._metadata.get("action_space_spec")
        if isinstance(obs_spec, dict) and isinstance(act_spec, dict):
            self.observation_space = space_from_spec(obs_spec)
            self.action_space = space_from_spec(act_spec)
            
        # Heartbeat setup
        # Use a background thread to send pings if sessions are enabled
        self._heartbeat_thread: threading.Thread | None = None
        self._stop_heartbeat = threading.Event()
        
        # Determine heartbeat interval (default to 1/3 of idle timeout, or 10s)
        idle_s = float(session_proto.get("idle_timeout_s", 30.0))
        if idle_s <= 0:
            self._heartbeat_interval = 10.0
        else:
            self._heartbeat_interval = max(1.0, idle_s / 3.0)

    @property
    def metadata(self) -> dict:
        return self._metadata

    def close(self) -> None:
        self._stop_heartbeat_thread()
        try:
            # Best-effort release lease on server-side worker (if sessions enabled).
            try:
                self.close_session()
            except Exception:
                pass
            # Don't throw if socket is already closed
            try:
                self._send({"cmd": "close"})
            except Exception:
                pass
        finally:
            with self._ws_lock:
                self._ws.close()

    @property
    def session_id(self) -> str | None:
        return self._session_id

    def close_session(self) -> None:
        """Release the current session_id back to the server pool (if supported)."""
        if not self._supports_sessions or self._session_id is None:
            return
        resp = self._send({"cmd": "close_session", "session_id": self._session_id})
        if resp.get("ok"):
            self._session_id = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        new_session: bool = False,
        session_id: str | None = None,
        wait_for_capacity: bool = True,
        max_wait_s: float | None = None,
        poll_interval_s: float = 1.0,
    ) -> tuple[Any, dict]:
        """Gymnasium-style reset: returns (obs, info)."""
        return self.reset_with_info(
            seed=seed,
            options=options,
            new_session=new_session,
            session_id=session_id,
            wait_for_capacity=wait_for_capacity,
            max_wait_s=max_wait_s,
            poll_interval_s=poll_interval_s,
        )

    def reset_obs(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        new_session: bool = False,
        session_id: str | None = None,
        wait_for_capacity: bool = True,
        max_wait_s: float | None = None,
        poll_interval_s: float = 1.0,
    ) -> Any:
        """Legacy convenience: returns obs only."""
        obs, _info = self.reset_with_info(
            seed=seed,
            options=options,
            new_session=new_session,
            session_id=session_id,
            wait_for_capacity=wait_for_capacity,
            max_wait_s=max_wait_s,
            poll_interval_s=poll_interval_s,
        )
        return obs

    def reset_with_info(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
        new_session: bool = False,
        session_id: str | None = None,
        wait_for_capacity: bool = True,
        max_wait_s: float | None = None,
        poll_interval_s: float = 1.0,
    ) -> tuple[Any, dict]:
        if self._supports_sessions:
            # Default behavior:
            # - if we already have a session_id, reuse it (reset same env)
            # - unless new_session=True, in which case request a fresh env lease
            if new_session:
                requested_sid = None
            elif session_id is not None:
                requested_sid = session_id
            else:
                requested_sid = self._session_id
            start_s = time.time()
            last_log_s = 0.0
            while True:
                resp = self._send_raw({"cmd": "reset", "seed": seed, "options": options, "session_id": requested_sid})
                if "error" not in resp:
                    break
                err = resp.get("error") or {}
                code = err.get("code")
                if code != "capacity_full" or not wait_for_capacity:
                    raise RuntimeError(f"Env server error: {err}")
                now = time.time()
                waited = now - start_s
                if max_wait_s is not None and waited >= max_wait_s:
                    raise RuntimeError(f"Env server error: {err} (waited {waited:.1f}s)")
                # Log a helpful message occasionally to avoid spam.
                if now - last_log_s >= 2.0:
                    last_log_s = now
                    logging.warning(
                        "RemoteEnv: no free session capacity on server yet (capacity_full). "
                        "waiting %.1fs... (poll_interval_s=%.2f)",
                        waited,
                        poll_interval_s,
                    )
                time.sleep(max(0.05, float(poll_interval_s)))
            # Server returns the (possibly new) session_id.
            if isinstance(resp.get("session_id"), str):
                self._session_id = resp["session_id"]
                self._ensure_heartbeat()
        else:
            resp = self._send({"cmd": "reset", "seed": seed, "options": options})
        # Allow server to lazily provide space specs in reset response.
        if self.observation_space is None and isinstance(resp.get("observation_space_spec"), dict):
            self.observation_space = space_from_spec(resp["observation_space_spec"])
        if self.action_space is None and isinstance(resp.get("action_space_spec"), dict):
            self.action_space = space_from_spec(resp["action_space_spec"])
        return resp.get("obs"), resp.get("info") or {}

    def step(self, action: Any) -> tuple[Any, Union[float, np.ndarray], Union[bool, np.ndarray], Union[bool, np.ndarray], dict]:
        """Gymnasium-style step: returns (obs, reward, terminated, truncated, info).
        
        For vector envs, reward/terminated/truncated are numpy arrays.
        For single envs, they are scalars (float/bool).
        """
        s = self.step_struct(action)
        return s.obs, s.reward, s.terminated, s.truncated, s.info

    def step_legacy(self, action: Any) -> tuple[Any, Union[float, np.ndarray], Union[bool, np.ndarray], dict]:
        """Legacy convenience: returns (obs, reward, done, info).
        
        For vector envs, reward/done are numpy arrays.
        """
        s = self.step_struct(action)
        return s.obs, s.reward, s.done, s.info

    def step_struct(self, action: Any) -> RemoteEnvStep:
        if self._supports_sessions:
            if self._session_id is None:
                raise RuntimeError("RemoteEnv.step() requires an active session_id; call reset() first.")
            resp = self._send({"cmd": "step", "action": action, "session_id": self._session_id})
        else:
            resp = self._send({"cmd": "step", "action": action})
        obs = resp.get("obs")
        reward_raw = resp.get("reward", 0.0)
        # Strict protocol: require explicit gymnasium-style fields.
        if "terminated" not in resp or "truncated" not in resp:
            raise RuntimeError(
                "Env server protocol mismatch: step response must include boolean fields "
                "`terminated` and `truncated` (no implicit fallback from `done`). "
                f"Got keys={sorted(resp.keys())}"
            )
        terminated_raw = resp["terminated"]
        truncated_raw = resp["truncated"]
        info = resp.get("info") or {}
        
        # Support both single env (scalars) and vector env (arrays)
        if isinstance(reward_raw, np.ndarray):
            # Vector env: keep as arrays
            reward = reward_raw
            terminated = terminated_raw if isinstance(terminated_raw, np.ndarray) else np.asarray(terminated_raw)
            truncated = truncated_raw if isinstance(truncated_raw, np.ndarray) else np.asarray(truncated_raw)
        else:
            # Single env: convert to scalars
            reward = float(reward_raw)
            terminated = bool(terminated_raw)
            truncated = bool(truncated_raw)
        
        return RemoteEnvStep(
            obs=obs, reward=reward, terminated=terminated, truncated=truncated, info=info
        )

    def _send(self, msg: dict) -> dict:
        decoded = self._send_raw(msg)
        if "error" in decoded:
            err = decoded["error"]
            raise RuntimeError(f"Env server error: {err}")
        return decoded

    def _send_raw(self, msg: dict) -> dict:
        # Attach a request_id if the caller didn't supply one (useful for debugging).
        if "request_id" not in msg:
            self._request_id += 1
            msg["request_id"] = self._request_id
            
        with self._ws_lock:
            self._ws.send(self._packer.pack(msg))
            resp = self._ws.recv()
            
        if isinstance(resp, str):
            raise RuntimeError(f"Env server error:\n{resp}")
        return msgpack_numpy.unpackb(resp)

    def _ensure_heartbeat(self) -> None:
        """Start the background heartbeat thread if not already running."""
        if not self._supports_sessions:
            return
        if self._heartbeat_thread is not None and self._heartbeat_thread.is_alive():
            return
            
        self._stop_heartbeat.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True, name="RemoteEnv-Heartbeat")
        self._heartbeat_thread.start()
        
    def _stop_heartbeat_thread(self) -> None:
        if self._heartbeat_thread:
            self._stop_heartbeat.set()
            # We don't join() because this might be called during shutdown/cleanup 
            # and we don't want to block if the thread is sleeping.
            self._heartbeat_thread = None

    def _heartbeat_loop(self) -> None:
        """Background loop to send ping messages to keep session alive."""
        while not self._stop_heartbeat.is_set():
            try:
                if self._session_id is not None:
                    # We use _send_raw (which locks) but we ignore errors to avoid crashing the thread.
                    # We only want to keep the session alive on the server.
                    resp = self._send_raw({
                        "cmd": "ping", 
                        "session_id": self._session_id
                    })
                    # If server returns invalid_session, it means we timed out or were reaped.
                    if "error" in resp and resp["error"].get("code") == "invalid_session":
                         # Logging it might be noisy if we are in a 'closed' state, 
                         # but useful for debugging why a session died.
                         pass
            except Exception:
                # Connection might be closed, or network error. 
                # Just ignore in heartbeat loop; main thread will see error on next step.
                pass
            
            # Sleep in small chunks to allow faster shutdown
            sleep_time = self._heartbeat_interval
            step = 0.5
            start = time.time()
            while time.time() - start < sleep_time:
                if self._stop_heartbeat.is_set():
                    return
                time.sleep(step)
