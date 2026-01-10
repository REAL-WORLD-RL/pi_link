"""
Generic Gymnasium/Gym environment websocket server.

This module provides a reusable base class for serving any gym.Env over websockets
with the RemoteEnv protocol. It handles:
- Worker pool management (multiprocessing)
- Session management (session_id routing)
- Websocket protocol (reset/step/close/ping)
- Observation/action space inference

To use, subclass `GymEnvServer` and implement:
1. `create_env()` - factory to create your gym environment
2. `process_observation()` - transform gym obs to client format
3. `process_action()` - transform client action to gym format (optional)
"""

from __future__ import annotations

import asyncio
import logging
import multiprocessing as mp
import time
import uuid
from abc import ABC
from typing import Any, Dict, Optional, Tuple

import numpy as np
import websockets
import websockets.asyncio.server as _server

from pi_link import msgpack_numpy
from pi_link.spaces import gym_space_to_spec

# Try to import gymnasium first, fallback to gym
try:
    import gymnasium.spaces as gym
except ImportError:
    try:
        import gym.spaces as gym
    except ImportError:
        gym = None

logger = logging.getLogger(__name__)


class _WorkerCrashed(RuntimeError):
    pass


class _WorkerMethodFactory:
    """Picklable factory that calls a server method in the worker process.
    
    This avoids pickling the server instance (which has asyncio.Lock), by instead:
    1. Storing the server class and init kwargs (which are picklable)
    2. Re-instantiating the server in the worker process
    3. Calling the target method on the new instance
    """
    
    def __init__(self, server_class, server_init_kwargs, method_name, method_kwargs):
        self.server_class = server_class
        self.server_init_kwargs = server_init_kwargs
        self.method_name = method_name
        self.method_kwargs = method_kwargs
        self._server_instance = None
    
    def __call__(self, *args, **kwargs):
        # Lazy initialization: create server instance on first call
        if self._server_instance is None:
            # Create a minimal server instance just for calling methods
            # This happens in the worker process, so asyncio.Lock is fine here
            # Set _skip_worker_init=True to avoid starting workers/network in the worker process
            init_kwargs_with_flag = {**self.server_init_kwargs, "_skip_worker_init": True}
            self._server_instance = self.server_class(**init_kwargs_with_flag)
        
        # Get the method and call it
        method = getattr(self._server_instance, self.method_name)
        merged_kwargs = {**self.method_kwargs, **kwargs}
        return method(*args, **merged_kwargs)


def _now_s() -> float:
    return time.time()


def _worker_loop(
    *,
    env_factory: callable,
    env_kwargs: dict,
    obs_processor: callable,
    action_processor: callable,
    conn: Any,  # multiprocessing.connection.Connection
) -> None:
    """Worker process: owns one gym env and handles reset/step."""
    import os
    print(f"[worker] MUJOCO_GL at start: {os.environ.get('MUJOCO_GL', 'NOT SET')}", flush=True)
    
    # Create environment
    env = env_factory(**env_kwargs)
    
    current_obs: Optional[Any] = None
    episode_over = False
    steps_since_reset = 0
    
    def _do_reset(reset_seed: Optional[int], options: Optional[dict]) -> dict:
        nonlocal current_obs, episode_over, steps_since_reset
        
        if reset_seed is not None:
            env.seed(int(reset_seed)) if hasattr(env, 'seed') else None
        
        # Reset returns (obs, info) in gymnasium, obs in old gym
        result = env.reset()
        if isinstance(result, tuple):
            raw_obs, info = result
        else:
            raw_obs = result
            info = {}
        
        current_obs = raw_obs
        episode_over = False
        steps_since_reset = 0
        
        return {
            "obs": obs_processor(raw_obs),
            "info": info,
        }
    
    def _do_step(action: Any) -> Tuple[dict, float, bool, bool, dict]:
        nonlocal current_obs, episode_over, steps_since_reset
        
        if current_obs is None:
            raise RuntimeError("Environment not reset; call reset first.")
        if episode_over:
            raise RuntimeError("Episode already finished; call reset before step.")
        
        # Process action
        gym_action = action_processor(action)
        
        # Step returns different formats in gym vs gymnasium
        result = env.step(gym_action)
        
        if len(result) == 5:
            # Gymnasium: (obs, reward, terminated, truncated, info)
            raw_obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            # Old gym: (obs, reward, done, info)
            raw_obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected step return format: {len(result)} values")
        
        current_obs = raw_obs
        steps_since_reset += 1
        episode_over = bool(terminated or truncated)
        
        return (
            obs_processor(raw_obs),
            float(reward),
            bool(terminated),
            bool(truncated),
            info or {},
        )
    
    def _infer_specs() -> dict:
        """Infer observation and action space specs from processed observation.
        
        Returns spaces that match what the client actually receives/sends,
        not the raw env spaces.
        """
        nonlocal current_obs, episode_over, steps_since_reset
        
        # Reset to get initial observation
        result = env.reset()
        if isinstance(result, tuple):
            current_obs, _ = result
        else:
            current_obs = result
        
        episode_over = False
        steps_since_reset = 0
        
        # Process observation to get what client will actually receive
        processed_obs = obs_processor(current_obs)
        
        # Infer observation space from processed observation structure
        def _infer_space_from_value(value):
            """Infer gym.Space from an actual value."""
            if isinstance(value, dict):
                return gym.Dict({
                    k: _infer_space_from_value(v) for k, v in value.items()
                })
            elif isinstance(value, np.ndarray):
                return gym.Box(
                    low=-np.inf if np.issubdtype(value.dtype, np.floating) else np.iinfo(value.dtype).min,
                    high=np.inf if np.issubdtype(value.dtype, np.floating) else np.iinfo(value.dtype).max,
                    shape=value.shape,
                    dtype=value.dtype,
                )
            elif isinstance(value, (list, tuple)):
                arr = np.asarray(value)
                return gym.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=arr.shape,
                    dtype=arr.dtype,
                )
            elif isinstance(value, str):
                # String observations - use a placeholder Box
                from pi_link.spaces import AnySpace
                return AnySpace()
            else:
                # Unknown type - use AnySpace
                from pi_link.spaces import AnySpace
                return AnySpace()
        
        # Infer spaces from processed observation
        try:
            inferred_obs_space = _infer_space_from_value(processed_obs)
            obs_space_spec = gym_space_to_spec(inferred_obs_space)
        except Exception as e:
            logger.warning(f"Failed to infer observation space: {e}, using None")
            obs_space_spec = None
        
        # For action space, use raw env's action_space (actions are usually not processed as heavily)
        action_space = getattr(env, 'action_space', None)
        action_space_spec = gym_space_to_spec(action_space) if action_space else None
        
        return {
            "observation_space_spec": obs_space_spec,
            "action_space_spec": action_space_spec,
            "sample_obs": processed_obs,
        }
    
    # Main worker loop
    while True:
        req = conn.recv()
        if not isinstance(req, dict):
            conn.send({"error": {"code": "bad_request", "message": f"Expected dict, got {type(req)}"}})
            continue
        
        cmd = req.get("cmd")
        try:
            if cmd == "infer_specs":
                conn.send(_infer_specs())
                continue
            
            if cmd == "reset":
                conn.send(_do_reset(req.get("seed"), req.get("options")))
                continue
            
            if cmd == "step":
                obs, reward, terminated, truncated, info = _do_step(req.get("action"))
                conn.send({
                    "obs": obs,
                    "reward": reward,
                    "terminated": terminated,
                    "truncated": truncated,
                    "done": terminated or truncated,
                    "info": info,
                })
                continue
            
            if cmd == "close":
                if hasattr(env, 'close'):
                    env.close()
                conn.send({"ok": True})
                return
            
            conn.send({"error": {"code": "unknown_cmd", "message": f"Unknown cmd: {cmd}"}})
        
        except Exception as e:
            logger.exception(f"Worker error on cmd={cmd}")
            conn.send({
                "error": {
                    "code": "worker_error",
                    "message": str(e),
                    "type": type(e).__name__,
                }
            })


class GymEnvServer(ABC):
    """Base class for serving gym environments over websockets."""
    
    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8000,
        max_sessions: int = 1,
        session_idle_timeout_s: float = 30.0,
        env_kwargs: Optional[dict] = None,
        _skip_worker_init: bool = False,  # Internal flag for worker process
    ) -> None:
        """
        Args:
            host: Host to bind websocket server
            port: Port to bind websocket server
            max_sessions: Maximum concurrent environment sessions
            session_idle_timeout_s: Auto-release idle sessions after this timeout
            env_kwargs: Keyword arguments passed to create_env()
            _skip_worker_init: Internal flag, set True when creating worker-side instance
        """
        self._host = host
        self._port = port
        self._max_sessions = int(max_sessions)
        self._session_idle_timeout_s = float(session_idle_timeout_s)
        self._env_kwargs = env_kwargs or {}
        
        # Skip worker/network initialization if this is a worker-side instance
        if _skip_worker_init:
            return
        
        if self._max_sessions <= 0:
            raise ValueError("max_sessions must be > 0")
        
        self._mp = mp.get_context("spawn")
        
        # Worker pool
        self._workers: list[dict] = []
        self._free_workers: list[int] = []
        
        # Session management
        self._session_to_worker: Dict[str, int] = {}
        self._worker_to_session: Dict[int, Optional[str]] = {}
        self._last_used_s: Dict[str, float] = {}
        
        # Cached specs
        self._space_specs: Optional[dict] = None
        
        self._start_workers()
    
    # Note: Subclasses should override _get_worker_target() to provide their own worker loop.
    # These methods are only used by the default _worker_loop and can be left unimplemented
    # if a custom worker is used.
    def create_env(self, **kwargs) -> Any:
        """Create and return a gym environment instance.
        
        This will be called in each worker process (if using default worker loop).
        Subclasses that provide custom worker loops don't need to implement this.
        
        Args:
            **kwargs: Environment-specific configuration
        
        Returns:
            A gym.Env instance
        """
        raise NotImplementedError("Subclass must implement create_env() or provide custom worker loop")
    
    def process_observation(self, obs: Any) -> Any:
        """Transform gym observation to client format.
        
        Default implementation passes observation through unchanged.
        Override if you need custom observation processing (e.g., to match pi0 format).
        
        This will be called in each worker process (if using default worker loop).
        Subclasses that provide custom worker loops don't need to implement this.
        
        Args:
            obs: Raw observation from gym env
        
        Returns:
            Processed observation in client-expected format.
            By default, returns the raw observation unchanged.
        """
        return obs
    
    def process_action(self, action: Any) -> Any:
        """Transform client action to gym format.
        
        Default implementation passes action through unchanged.
        Override if you need custom action processing.
        
        Args:
            action: Action from client
        
        Returns:
            Action in gym-compatible format
        """
        return action
    
    def get_metadata(self) -> dict:
        """Return server metadata sent in handshake.
        
        Override to add custom metadata fields.
        """
        return {
            "kind": "gym_env_server",
            "max_sessions": self._max_sessions,
            "idle_timeout_s": self._session_idle_timeout_s,
        }
    
    def _start_workers(self) -> None:
        """Start worker pool.
        
        Default implementation uses the generic _worker_loop with create_env,
        process_observation, and process_action methods.
        
        Subclasses with special needs (like LIBERO) can override _get_worker_target()
        to provide a custom worker function.
        """
        for wid in range(self._max_sessions):
            parent_conn, child_conn = self._mp.Pipe(duplex=True)
            
            # Get worker target and kwargs (default or custom)
            worker_target, worker_kwargs = self._get_worker_target(child_conn, wid)
            
            proc = self._mp.Process(
                target=worker_target,
                kwargs=worker_kwargs,
                daemon=True,
                name=f"gym-env-worker-{wid}",
            )
            proc.start()
            
            self._workers.append({
                "proc": proc,
                "conn": parent_conn,
                "lock": asyncio.Lock(),
            })
            self._free_workers.append(wid)
            self._worker_to_session[wid] = None
    
    def _get_worker_target(self, child_conn, worker_id):
        """Get worker target function and kwargs.
        
        Default implementation returns the generic _worker_loop with a worker factory.
        The factory will call create_env, process_observation, process_action methods
        IN THE WORKER PROCESS (not in the main process).
        
        Subclasses with special initialization needs (like LIBERO) can override this
        to provide a custom worker loop.
        
        Returns:
            tuple: (worker_function, kwargs_dict)
        """
        # Prepare env kwargs with unique seed per worker
        worker_env_kwargs = dict(self._env_kwargs)
        if 'seed' in worker_env_kwargs:
            worker_env_kwargs['seed'] = worker_env_kwargs['seed'] + worker_id
        
        # Return generic _worker_loop with a factory that will:
        # 1. Import this module in the worker process
        # 2. Re-instantiate the server class (light copy, only for calling methods)
        # 3. Call create_env/process_observation/process_action IN the worker
        
        return _worker_loop, {
            "env_factory": _WorkerMethodFactory(
                server_class=self.__class__,
                server_init_kwargs=self._get_picklable_init_kwargs(),
                method_name="create_env",
                method_kwargs=worker_env_kwargs,
            ),
            "env_kwargs": {},  # Already in factory
            "obs_processor": _WorkerMethodFactory(
                server_class=self.__class__,
                server_init_kwargs=self._get_picklable_init_kwargs(),
                method_name="process_observation",
                method_kwargs={},
            ),
            "action_processor": _WorkerMethodFactory(
                server_class=self.__class__,
                server_init_kwargs=self._get_picklable_init_kwargs(),
                method_name="process_action",
                method_kwargs={},
            ),
            "conn": child_conn,
        }
    
    def _get_picklable_init_kwargs(self):
        """Get picklable init kwargs for re-creating server in worker.
        
        Only includes data needed for create_env/process_observation/process_action.
        Does NOT include network/worker pool stuff.
        
        Subclasses should override if they need custom parameters.
        """
        return {
            "host": "dummy",  # Not used in worker
            "port": 0,  # Not used in worker
            "max_sessions": 0,  # Not used in worker
            "session_idle_timeout_s": 0.0,  # Not used in worker
            "env_kwargs": self._env_kwargs,
        }
    
    def serve_forever(self) -> None:
        """Start server and block forever."""
        asyncio.run(self.run())
    
    async def run(self) -> None:
        """Async server main loop."""
        await self._ensure_space_specs()
        
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
        ) as server:
            logger.info(f"Gym env server listening on ws://{self._host}:{self._port}")
            logger.info(f"Max sessions: {self._max_sessions}")
            
            reaper_task = asyncio.create_task(self._reap_idle_sessions())
            try:
                await server.serve_forever()
            finally:
                reaper_task.cancel()
    
    async def _ensure_space_specs(self) -> None:
        """Infer and cache space specs from a worker."""
        if self._space_specs is not None:
            return
        
        resp = await self._call_worker(0, {"cmd": "infer_specs"})
        if "error" in resp:
            raise RuntimeError(f"Failed to infer specs: {resp['error']}")
        
        self._space_specs = resp
        logger.info("Space specs inferred successfully")
    
    async def _reap_idle_sessions(self) -> None:
        """Background task to auto-release idle sessions."""
        if self._session_idle_timeout_s <= 0:
            return
        
        while True:
            await asyncio.sleep(1.0)
            now = _now_s()
            
            for sid, last_used in list(self._last_used_s.items()):
                if now - last_used < self._session_idle_timeout_s:
                    continue
                
                wid = self._session_to_worker.get(sid)
                if wid is None:
                    self._last_used_s.pop(sid, None)
                    continue
                
                if self._workers[wid]["lock"].locked():
                    continue
                
                logger.info(f"Auto-releasing idle session_id={sid}")
                self._release_session(sid)
    
    def _release_session(self, session_id: str) -> None:
        """Release a session and free its worker."""
        wid = self._session_to_worker.pop(session_id, None)
        self._last_used_s.pop(session_id, None)
        
        if wid is None:
            return
        
        self._worker_to_session[wid] = None
        self._free_workers.append(wid)
    
    def _alloc_session(self, requested_session_id: Optional[str]) -> Tuple[str, int]:
        """Allocate a new session."""
        if not self._free_workers:
            raise RuntimeError("capacity_full")
        
        if requested_session_id and requested_session_id not in self._session_to_worker:
            sid = requested_session_id
        else:
            sid = uuid.uuid4().hex
        
        wid = self._free_workers.pop()
        self._session_to_worker[sid] = wid
        self._worker_to_session[wid] = sid
        self._last_used_s[sid] = _now_s()
        
        return sid, wid
    
    async def _call_worker(self, worker_id: int, msg: dict) -> dict:
        """Send message to worker and get response."""
        w = self._workers[worker_id]
        proc = w["proc"]
        
        if not proc.is_alive():
            raise _WorkerCrashed(
                f"worker {worker_id} is not alive (exitcode={proc.exitcode})"
            )
        
        async with w["lock"]:
            conn = w["conn"]
            
            def _sync_roundtrip() -> dict:
                conn.send(msg)
                return conn.recv()
            
            loop = asyncio.get_running_loop()
            return await loop.run_in_executor(None, _sync_roundtrip)
    
    def _err(
        self,
        *,
        code: str,
        message: str,
        details: Optional[dict] = None,
        request_id: Any = None,
    ) -> dict:
        """Create error response."""
        out: dict = {"error": {"code": code, "message": message}}
        if details:
            out["error"]["details"] = details
        if request_id is not None:
            out["request_id"] = request_id
        return out
    
    async def _handler(self, ws: _server.ServerConnection) -> None:
        """Handle websocket connection."""
        packer = msgpack_numpy.Packer()
        await self._ensure_space_specs()
        
        # Send handshake metadata
        metadata = self.get_metadata()
        if self._space_specs:
            metadata.update({
                "observation_space_spec": self._space_specs.get("observation_space_spec"),
                "action_space_spec": self._space_specs.get("action_space_spec"),
            })
        
        metadata["session_protocol"] = {
            "enabled": True,
            "max_sessions": self._max_sessions,
            "idle_timeout_s": self._session_idle_timeout_s,
        }
        
        await ws.send(packer.pack(metadata))
        
        # Main message loop
        while True:
            try:
                req_raw = await ws.recv()
                req = msgpack_numpy.unpackb(req_raw)
                
                if not isinstance(req, dict):
                    await ws.send(
                        packer.pack(
                            self._err(
                                code="bad_request",
                                message=f"Expected dict, got {type(req)}",
                            )
                        )
                    )
                    continue
                
                cmd = req.get("cmd")
                request_id = req.get("request_id")
                session_id = req.get("session_id")
                
                # Handle close
                if cmd == "close":
                    await ws.send(packer.pack({"ok": True, "request_id": request_id}))
                    await ws.close(
                        code=websockets.frames.CloseCode.NORMAL_CLOSURE,
                        reason="Closed by client.",
                    )
                    return
                
                # Handle close_session
                if cmd == "close_session":
                    if not session_id or str(session_id) not in self._session_to_worker:
                        await ws.send(
                            packer.pack(
                                self._err(
                                    code="invalid_session",
                                    message="close_session requires a valid session_id",
                                    details={"session_id": session_id},
                                    request_id=request_id,
                                )
                            )
                        )
                        continue
                    
                    self._release_session(str(session_id))
                    await ws.send(
                        packer.pack({
                            "ok": True,
                            "session_id": str(session_id),
                            "request_id": request_id,
                        })
                    )
                    continue
                
                # Handle ping (heartbeat)
                if cmd == "ping":
                    if session_id and str(session_id) in self._session_to_worker:
                        self._last_used_s[str(session_id)] = _now_s()
                        await ws.send(
                            packer.pack({
                                "ok": True,
                                "cmd": "pong",
                                "request_id": request_id,
                            })
                        )
                    else:
                        await ws.send(
                            packer.pack(
                                self._err(
                                    code="invalid_session",
                                    message="ping requires a valid active session_id",
                                    details={"session_id": session_id},
                                    request_id=request_id,
                                )
                            )
                        )
                    continue
                
                # Handle reset
                if cmd == "reset":
                    new_session = bool(req.get("new_session", False))
                    if new_session:
                        session_id = None
                    
                    if session_id and str(session_id) in self._session_to_worker:
                        sid = str(session_id)
                        wid = self._session_to_worker[sid]
                    else:
                        try:
                            sid, wid = self._alloc_session(
                                str(session_id) if session_id else None
                            )
                        except RuntimeError as e:
                            if str(e) == "capacity_full":
                                await ws.send(
                                    packer.pack(
                                        self._err(
                                            code="capacity_full",
                                            message="No free env workers; server at max_sessions",
                                            details={"max_sessions": self._max_sessions},
                                            request_id=request_id,
                                        )
                                    )
                                )
                                continue
                            raise
                    
                    self._last_used_s[sid] = _now_s()
                    
                    resp = await self._call_worker(
                        wid,
                        {
                            "cmd": "reset",
                            "seed": req.get("seed"),
                            "options": req.get("options"),
                        },
                    )
                    
                    if "error" in resp:
                        await ws.send(
                            packer.pack(
                                self._err(
                                    code="reset_failed",
                                    message=str(resp["error"]),
                                    request_id=request_id,
                                )
                            )
                        )
                        continue
                    
                    await ws.send(
                        packer.pack({
                            "session_id": sid,
                            "obs": resp.get("obs"),
                            "info": resp.get("info") or {},
                            "request_id": request_id,
                        })
                    )
                    continue
                
                # Handle step
                if cmd == "step":
                    if not session_id or str(session_id) not in self._session_to_worker:
                        await ws.send(
                            packer.pack(
                                self._err(
                                    code="invalid_session",
                                    message="step requires a valid session_id (call reset first)",
                                    details={"session_id": session_id},
                                    request_id=request_id,
                                )
                            )
                        )
                        continue
                    
                    sid = str(session_id)
                    wid = self._session_to_worker[sid]
                    self._last_used_s[sid] = _now_s()
                    
                    resp = await self._call_worker(
                        wid,
                        {"cmd": "step", "action": req.get("action")},
                    )
                    
                    if "error" in resp:
                        err = resp.get("error") or {}
                        await ws.send(
                            packer.pack({
                                "error": {
                                    "code": "step_failed",
                                    "message": err.get("message", str(err)),
                                    "details": {
                                        "session_id": sid,
                                        "worker_error": err,
                                    },
                                },
                                "request_id": request_id,
                            })
                        )
                        continue
                    
                    await ws.send(
                        packer.pack({
                            "session_id": sid,
                            "obs": resp.get("obs"),
                            "reward": float(resp.get("reward", 0.0)),
                            "terminated": bool(resp.get("terminated", False)),
                            "truncated": bool(resp.get("truncated", False)),
                            "done": bool(resp.get("done", False)),
                            "info": resp.get("info") or {},
                            "request_id": request_id,
                        })
                    )
                    continue
                
                # Unknown command
                await ws.send(
                    packer.pack(
                        self._err(
                            code="unknown_cmd",
                            message=f"Unknown cmd: {cmd}",
                            request_id=request_id,
                        )
                    )
                )
            
            except websockets.ConnectionClosed:
                return
            except Exception as e:
                logger.exception("Unexpected error in handler")
                try:
                    await ws.send(
                        packer.pack(
                            self._err(
                                code="server_error",
                                message=str(e),
                                details={"type": type(e).__name__},
                            )
                        )
                    )
                except Exception:
                    pass
                return

