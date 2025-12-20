from __future__ import annotations

import argparse
import pathlib
import socket
import sys

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
    def __init__(self, policy, env):
        self.remote_policy = policy
        self.remote_env = env
        # If the env server is at capacity, block here until a session becomes available.
        self.obs, _info = self.remote_env.reset(wait_for_capacity=True)
        self.done = False
        self.total_reward = 0.0
        self.steps = 0
        self.sampled_epoches = 0

    def step(self):
        # action = self.remote_policy.step(self.obs)
        action = self.remote_env.action_space.sample()
        self.obs, reward, terminated, truncated, info = self.remote_env.step(action)
        self.done = terminated or truncated
        self.total_reward += reward
        self.steps += 1
        print(f"[pi_link] step {self.steps}: reward={reward} terminated={terminated} truncated={truncated} info={info}")
        if self.done:
            self.log_summary()
            self.remote_env.reset(wait_for_capacity=True)
            self.done = False
            self.total_reward = 0.0
            self.steps = 0
            self.sampled_epoches += 1
            
    def log_summary(self):
        print("episode done, total_reward=", self.total_reward, "total_steps=", self.steps)

    def reset(self):
        self.obs, _info = self.remote_env.reset(wait_for_capacity=True)
        self.done = False
        self.total_reward = 0.0
        self.steps = 0

    def close(self):
        # Keep this minimal; RemoteEnv.close() is called in main().
        pass

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RemoteEnv/RemotePolicy with a gym-like loop.")
    # In `examples/libero/compose_separated.yml`, host 9001 maps to openpi_server:8000.
    parser.add_argument("--policy_uri", default="ws://127.0.0.1:9001")
    # In `examples/libero/compose_separated.yml`, host 9000 maps to runtime:8000 (not necessarily a websocket env server).
    parser.add_argument("--env_uri", default="ws://127.0.0.1:9000")
    parser.add_argument("--max_steps", type=int, default=5)
    args = parser.parse_args()
    
    total_epoches = 100
    total_envs = 1

    policy_uri = args.policy_uri
    env_uri = args.env_uri

    remote_policy = RemotePolicy(policy_uri)
    
    remote_envs = [RemoteEnv(env_uri) for _ in range(total_envs)]
    gym_loops = [GymLoop(remote_policy, remote_env) for remote_env in remote_envs]
    
    while sum(gym_loop.sampled_epoches for gym_loop in gym_loops) < total_epoches:
        for gym_loop in gym_loops:
            gym_loop.step()
            # print('gym_loop.steps=', gym_loop.steps)
        # print("sampled_epoches=", sum(gym_loop.sampled_epoches for gym_loop in gym_loops))
        

    for gym_loop in gym_loops:
        gym_loop.close()
    for remote_env in remote_envs:
        remote_env.close()
    remote_policy.close()

if __name__ == "__main__":
    main()


