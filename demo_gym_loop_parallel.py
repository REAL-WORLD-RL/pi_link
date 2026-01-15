from __future__ import annotations

import argparse
import pathlib
import socket
import sys
from tqdm import tqdm
import numpy as np

# Allow running as a script: `python pi_link/demo_gym_loop.py`
#
# Note: when launched via some debuggers (e.g. debugpy), `__package__` can be
# an empty string instead of None. Treat both as "run as a script" so we add
# the repo root to `sys.path` and imports like `pi_link.*` work.
if __package__ in (None, ""):
    sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from pi_link.remote_env import RemoteEnv
from pi_link.remote_policy import RemotePolicy


def _probe_healthz(uri: str) -> str:
    """Best-effort probe: if this is an OpenPI websocket server, /healthz returns HTTP 200."""
    s: socket.socket | None = None
    try:
        if not uri.startswith("ws://"):
            return "skip (only ws:// supported)"
        hostport = uri[len("ws://") :].split("/", 1)[0]
        if ":" not in hostport:
            return "skip (missing :port)"
        host, port_s = hostport.rsplit(":", 1)
        port = int(port_s)

        s = socket.socket()
        s.settimeout(1)
        s.connect((host, port))
        s.sendall(b"GET /healthz HTTP/1.1\r\nHost: localhost\r\nConnection: close\r\n\r\n")
        data = s.recv(200)
        return repr(data[:120])
    except Exception as e:  # noqa: BLE001
        return f"err: {type(e).__name__}: {e}"
    finally:
        try:
            if s is not None:
                s.close()
        except Exception:
            pass

class GymLoop:
    """Gym loop supporting both single env (scalars) and vector env (arrays)."""
    
    def __init__(self, policy, env):
        self.remote_policy = policy
        self.remote_env = env
        # If the env server is at capacity, block here until a session becomes available.
        self.obs, _info = self.remote_env.reset(wait_for_capacity=True)
        
        # Detect if this is a vector env by checking obs shape
        self._is_vec_env = self._detect_vec_env(self.obs)
        if self._is_vec_env:
            self.num_envs = self._get_num_envs(self.obs)
            self.done: np.ndarray | bool = np.zeros(self.num_envs, dtype=bool)
            self.total_reward: np.ndarray | float = np.zeros(self.num_envs, dtype=np.float32)
            self.episode_counts: np.ndarray | int = np.zeros(self.num_envs, dtype=np.int32)
        else:
            self.num_envs = 1
            self.done = False
            self.total_reward = 0.0
            self.episode_counts = 0
        
        self.steps = 0
        self.sampled_epoches = 0
    
    def _detect_vec_env(self, obs) -> bool:
        """Detect if observation is from a vector env."""
        if isinstance(obs, np.ndarray) and obs.ndim >= 2:
            # Vector env typically has shape (num_envs, ...)
            return True
        if isinstance(obs, dict):
            # Check first value in dict
            for v in obs.values():
                if isinstance(v, np.ndarray) and v.ndim >= 2:
                    return True
        return False
    
    def _get_num_envs(self, obs) -> int:
        """Get number of environments from observation."""
        if isinstance(obs, np.ndarray):
            return obs.shape[0]
        if isinstance(obs, dict):
            for v in obs.values():
                if isinstance(v, np.ndarray):
                    return v.shape[0]
        return 1
    
    def panallel_step(self):
        assert self._is_vec_env
        if self.remote_policy is not None:
            actions = self.remote_policy.step(self.obs)
        else:
            actions = self.remote_env.action_space.sample()
        self.obs, reward, terminated, truncated, info = self.remote_env.step(actions)
        self.steps += 1

    def step(self):
        if self.remote_policy is not None:
            action = self.remote_policy.step(self.obs)
        else:
            action = self.remote_env.action_space.sample()
        self.obs, reward, terminated, truncated, info = self.remote_env.step(action)
        self.steps += 1
        
        if self._is_vec_env:
            # Vector env: ManiSkill auto-resets done envs, no manual reset needed.
            # We just track episode completions by counting done flags.
            assert isinstance(self.total_reward, np.ndarray)
            assert isinstance(self.episode_counts, np.ndarray)
            total_reward_arr = self.total_reward  # type: np.ndarray
            episode_counts_arr = self.episode_counts  # type: np.ndarray
            
            done = terminated | truncated
            total_reward_arr += reward
            
            # Count completed episodes (auto-reset already happened in env.step)
            num_done = int(np.sum(done))
            if num_done > 0:
                avg_reward = float(np.mean(total_reward_arr[done]))
                print(f"[pi_link] step {self.steps}: {num_done} eps done, avg_reward={avg_reward:.3f}")
                # Reset reward tracking for envs that just finished (obs is already new episode)
                total_reward_arr[done] = 0.0
                episode_counts_arr[done] += 1
                self.sampled_epoches += num_done
        else:
            # Single env: need manual reset when done
            self.done = terminated or truncated
            self.total_reward += float(reward)
            print(f"[pi_link] step {self.steps}: reward={reward} terminated={terminated} truncated={truncated}")
            if self.done:
                self.log_summary()
                self.remote_env.reset(wait_for_capacity=True)  # manual reset required
                self.done = False
                self.total_reward = 0.0
                self.sampled_epoches += 1
            
    def log_summary(self):
        if self._is_vec_env:
            print(f"vec env summary: total_episodes={np.sum(self.episode_counts)}, total_steps={self.steps}")
        else:
            print("episode done, total_reward=", self.total_reward, "total_steps=", self.steps)

    def reset(self):
        self.obs, _info = self.remote_env.reset(wait_for_capacity=True)
        self.steps = 0
        if self._is_vec_env:
            self.done = np.zeros(self.num_envs, dtype=bool)
            self.total_reward = np.zeros(self.num_envs, dtype=np.float32)
        else:
            self.done = False
            self.total_reward = 0.0

    def close(self):
        # Keep this minimal; RemoteEnv.close() is called in main().
        pass

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RemoteEnv/RemotePolicy with a gym-like loop.")
    # In `examples/libero/compose_separated.yml`, host 9001 maps to openpi_server:8000.
    parser.add_argument("--policy_uri", default="ws://127.0.0.1:9001")
    # In `examples/libero/compose_separated.yml`, host 9000 maps to runtime:8000 (not necessarily a websocket env server).
    parser.add_argument("--env_uri", default="ws://127.0.0.1:9100")
    parser.add_argument("--max_steps", type=int, default=5000)
    args = parser.parse_args()
    
    total_epoches = 100
    total_envs = 1

    policy_uri = args.policy_uri
    env_uri = args.env_uri

    # remote_policy = RemotePolicy(policy_uri)
    remote_policy = None
    
    remote_env = RemoteEnv(env_uri)
    gym_loop = GymLoop(remote_policy, remote_env)
    for _ in tqdm(range(args.max_steps)):
        gym_loop.panallel_step()
    gym_loop.log_summary()
    gym_loop.close()
    remote_env.close()
    if remote_policy is not None:
        remote_policy.close()

if __name__ == "__main__":
    main()


