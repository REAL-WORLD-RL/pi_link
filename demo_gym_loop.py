from __future__ import annotations

import argparse
import pathlib
import socket
import sys
import os
from datetime import datetime
import imageio
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
    def __init__(self, policy, env, record_video=False, video_folder="./videos", 
                 video_prefix="episode", camera_key="image", fps=30):
        self.remote_policy = policy
        self.remote_env = env
        self.record_video = record_video
        self.video_folder = video_folder
        self.video_prefix = video_prefix
        self.camera_key = camera_key  # 用于从 obs 中提取图像的 key
        self.fps = fps
        
        # 如果需要录制视频，创建视频文件夹
        if self.record_video and not os.path.exists(self.video_folder):
            os.makedirs(self.video_folder)
        
        # If the env server is at capacity, block here until a session becomes available.
        self.obs, _info = self.remote_env.reset(wait_for_capacity=True)
        self.done = False
        self.total_reward = 0.0
        self.steps = 0
        self.sampled_epoches = 0
        self.frames = []  # 存储当前 episode 的帧
        self.episode_count = 0  # 用于视频文件命名
        
        # Action chunking 相关
        self.action_chunks = None  # 存储 action chunks
        self.chunk_idx = 0  # 当前使用的 chunk index

    def step(self, use_policy=True):
        """执行一步并可选择性地录制视频
        
        Args:
            use_policy: 如果为 True，使用 remote policy；否则使用随机动作
        """
        # 录制当前帧（在执行动作之前）
        if self.record_video:
            frame = self._get_frame_from_obs(self.obs)
            if frame is not None:
                self.frames.append(frame)
        
        # 获取动作
        if use_policy:
            # 如果没有 action chunks 或者已经用完，重新获取
            if self.action_chunks is None or self.chunk_idx >= self.action_chunks.shape[0]:
                raw_result = self.remote_policy.raw_step(self.obs)
                actions = raw_result.raw.get("actions")
                
                if actions is not None:
                    # actions 形状可能是 (batch, horizon, dim) 或 (horizon, dim)
                    if len(actions.shape) == 3:
                        # (batch, horizon, dim) -> 取第一个 batch
                        self.action_chunks = actions[0]  # (horizon, dim)
                    elif len(actions.shape) == 2:
                        # (horizon, dim)
                        self.action_chunks = actions
                    else:
                        # (dim,) 单个 action，不是 chunking
                        self.action_chunks = actions.reshape(1, -1)  # (1, dim)
                    
                    self.chunk_idx = 0
                    print(f"[Policy] 获取新的 action chunks: shape={self.action_chunks.shape}")
                else:
                    raise ValueError("Policy 返回的结果中没有 'actions' 字段")
            
            # 使用当前的 chunk
            action = self.action_chunks[self.chunk_idx]
            self.chunk_idx += 1
            print(f"[Policy] 使用 action chunk {self.chunk_idx}/{len(self.action_chunks)}")
        else:
            action = self.remote_env.action_space.sample()
        
        # 执行动作
        self.obs, reward, terminated, truncated, info = self.remote_env.step(action)
        self.done = terminated or truncated
        # self.done = terminated or self.steps >= 1000
        
        # 确保 reward 是标量（处理 numpy array）
        if isinstance(reward, np.ndarray):
            reward_scalar = float(reward.item()) if reward.size == 1 else float(reward.flatten()[0])
        else:
            reward_scalar = float(reward) if hasattr(reward, '__float__') else reward
        self.total_reward += reward_scalar
        self.steps += 1
        
        print(f"[pi_link] step {self.steps}: reward={reward_scalar:.4f} terminated={terminated} truncated={truncated}")
        
        if self.done:
            # 保存最后一帧
            if self.record_video:
                frame = self._get_frame_from_obs(self.obs)
                if frame is not None:
                    self.frames.append(frame)
                self._save_video()
            
            self.log_summary()
            self.obs, _info = self.remote_env.reset(wait_for_capacity=True)
            self.done = False
            self.total_reward = 0.0
            self.steps = 0
            self.sampled_epoches += 1
            self.frames = []  # 重置帧缓存
            
            # 重置 action chunks
            self.action_chunks = None
            self.chunk_idx = 0
            
    def _get_frame_from_obs(self, obs):
        """从观测中提取图像帧（支持多相机拼接）
        
        Args:
            obs: 环境返回的观测字典
            
        Returns:
            numpy array 形状为 (H, W, 3) 的图像帧，如果无法提取则返回 None
        """
        if not isinstance(obs, dict):
            return None
        
        # 对于 ManiSkill 环境，尝试拼接多个相机的图像
        base_camera_key = 'observation.images.base_camera'
        hand_camera_key = 'observation.images.hand_camera'
        
        if base_camera_key in obs and hand_camera_key in obs:
            base_frame = self._process_frame(obs[base_camera_key])
            hand_frame = self._process_frame(obs[hand_camera_key])
            
            if base_frame is not None and hand_frame is not None:
                # 检查两个图像的高度是否相同
                if base_frame.shape[0] != hand_frame.shape[0]:
                    print(f"[警告] 两个相机图像高度不匹配: base={base_frame.shape}, hand={hand_frame.shape}")
                    # 使用较小的高度
                    min_height = min(base_frame.shape[0], hand_frame.shape[0])
                    base_frame = base_frame[:min_height, :, :]
                    hand_frame = hand_frame[:min_height, :, :]
                
                # 横向拼接两个图像
                try:
                    combined_frame = np.concatenate([base_frame, hand_frame], axis=1)
                    if self.steps == 0:  # 只在第一步打印一次
                        print(f"[视频] 拼接两个相机图像: base_camera{base_frame.shape} + hand_camera{hand_frame.shape} = {combined_frame.shape}")
                    return combined_frame
                except Exception as e:
                    print(f"[错误] 拼接图像失败: {e}")
                    print(f"  base_frame.shape={base_frame.shape}, dtype={base_frame.dtype}")
                    print(f"  hand_frame.shape={hand_frame.shape}, dtype={hand_frame.dtype}")
                    # 拼接失败，使用第一个相机
                    return base_frame
        
        # 如果拼接失败，尝试单独使用其中一个相机
        if base_camera_key in obs:
            frame = self._process_frame(obs[base_camera_key])
            if frame is not None:
                if self.steps == 0:
                    print(f"[视频] 使用 base_camera: shape={frame.shape}")
                return frame
        
        if hand_camera_key in obs:
            frame = self._process_frame(obs[hand_camera_key])
            if frame is not None:
                if self.steps == 0:
                    print(f"[视频] 使用 hand_camera: shape={frame.shape}")
                return frame
        
        # 尝试用户指定的 key
        if self.camera_key in obs:
            frame = obs[self.camera_key]
            return self._process_frame(frame)
        
        # 如果上述都失败，尝试常见的图像 key
        common_image_keys = [
            'image', 
            'rgb', 
            'camera', 
            'pixels', 
            'observation', 
            'camera_0', 
            'front_camera'
        ]
        for key in common_image_keys:
            if key in obs:
                frame = obs[key]
                # 检查是否是图像（3维且最后一维是3或4）
                if hasattr(frame, 'shape') and len(frame.shape) == 3 and frame.shape[-1] in [3, 4]:
                    if self.steps == 0:
                        print(f"[视频] 自动检测到图像 key: '{key}' (shape={frame.shape})")
                    self.camera_key = key  # 更新 camera_key
                    return self._process_frame(frame)
        
        return None
    
    def _process_frame(self, frame):
        """处理帧数据，确保是正确的格式
        
        Args:
            frame: 原始帧数据
            
        Returns:
            处理后的 uint8 格式帧，形状为 (H, W, C)
        """
        if frame is None:
            return None
            
        # 确保帧是 numpy array
        if not isinstance(frame, np.ndarray):
            frame = np.array(frame)
        
        # 去除多余的 batch 维度
        # 如果形状是 (1, H, W, C) 或 (B, H, W, C)，取第一个
        if len(frame.shape) == 4:
            frame = frame[0]
        
        # 如果是 4 通道（RGBA），转换为 3 通道（RGB）
        if len(frame.shape) == 3 and frame.shape[-1] == 4:
            frame = frame[:, :, :3]
        
        # 确保帧是 uint8 类型
        if frame.dtype != np.uint8:
            # 如果是浮点数 [0, 1]，缩放到 [0, 255]
            if frame.max() <= 1.0:
                frame = (frame * 255).astype(np.uint8)
            else:
                frame = frame.astype(np.uint8)
        
        # 确保是 3 维 (H, W, C)
        if len(frame.shape) != 3 or frame.shape[2] not in [1, 3, 4]:
            print(f"[错误] 帧形状不正确: {frame.shape}")
            return None
        
        return frame
    
    def _save_video(self):
        """将收集的帧保存为视频文件"""
        if len(self.frames) == 0:
            print(f"[警告] Episode {self.episode_count} 没有帧可以保存")
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{self.video_prefix}_{self.episode_count:04d}_{timestamp}.mp4"
        filepath = os.path.join(self.video_folder, filename)
        
        try:
            imageio.mimsave(filepath, self.frames, fps=self.fps)
            print(f"[视频保存] 保存到: {filepath} ({len(self.frames)} 帧)")
        except Exception as e:
            print(f"[错误] 保存视频失败: {e}")
            # 调试信息
            first_frame = self.frames[0]
            print(f"  第一帧形状: {first_frame.shape}, dtype: {first_frame.dtype}")
            print(f"  总帧数: {len(self.frames)}")
        
        self.episode_count += 1
            
    def log_summary(self):
        print(f"[Episode {self.sampled_epoches + 1}] 总奖励: {self.total_reward:.4f}, 总步数: {self.steps}")

    def reset(self):
        self.obs, _info = self.remote_env.reset(wait_for_capacity=True)
        self.done = False
        self.total_reward = 0.0
        self.steps = 0
        self.frames = []
        self.action_chunks = None
        self.chunk_idx = 0

    def close(self):
        # Keep this minimal; RemoteEnv.close() is called in main().
        pass

def main() -> None:
    parser = argparse.ArgumentParser(description="Validate RemoteEnv/RemotePolicy with a gym-like loop.")
    # In `examples/libero/compose_separated.yml`, host 9001 maps to openpi_server:8000.
    parser.add_argument("--policy_uri", default="ws://127.0.0.1:8000")
    # In `examples/libero/compose_separated.yml`, host 9000 maps to runtime:8000 (not necessarily a websocket env server).
    parser.add_argument("--env_uri", default="ws://127.0.0.1:9010")
    parser.add_argument("--max_steps", type=int, default=5)
    parser.add_argument("--total_epoches", type=int, default=100, help="Total episodes to run")
    parser.add_argument("--total_envs", type=int, default=1, help="Number of parallel environments")
    parser.add_argument("--use_policy", action="store_true", help="Use remote policy instead of random actions")
    parser.add_argument("--record_video", action="store_true", help="Record video of episodes")
    parser.add_argument("--video_folder", default="./videos", help="Folder to save videos")
    parser.add_argument("--video_prefix", default="episode", help="Prefix for video filenames")
    parser.add_argument("--camera_key", default="image", help="Key to extract image from observation dict")
    parser.add_argument("--fps", type=int, default=30, help="FPS for saved videos")
    args = parser.parse_args()
    
    total_epoches = args.total_epoches
    total_envs = args.total_envs

    policy_uri = args.policy_uri
    env_uri = args.env_uri

    # 只有在使用 policy 时才连接 policy server
    remote_policy = None
    if args.use_policy:
        print(f"[初始化] 连接到 policy server: {policy_uri}")
        remote_policy = RemotePolicy(policy_uri)
        print(f"[初始化] Policy metadata: {remote_policy.metadata}")
    else:
        print("[初始化] 使用随机动作（未连接 policy server）")
    
    print(f"[初始化] 连接到 {total_envs} 个环境 server: {env_uri}")
    remote_envs = [RemoteEnv(env_uri) for _ in range(total_envs)]
    
    # 创建 GymLoop 实例
    gym_loops = [
        GymLoop(
            remote_policy, 
            remote_env,
            record_video=args.record_video,
            video_folder=args.video_folder,
            video_prefix=f"{args.video_prefix}_env{i}",
            camera_key=args.camera_key,
            fps=args.fps
        ) 
        for i, remote_env in enumerate(remote_envs)
    ]
    
    if args.record_video:
        print(f"[录制] 视频将保存到: {args.video_folder}")
    
    print(f"[开始] 运行 {total_epoches} 个 episodes...")
    
    # 主循环
    while sum(gym_loop.sampled_epoches for gym_loop in gym_loops) < total_epoches:
        for gym_loop in gym_loops:
            gym_loop.step(use_policy=args.use_policy)
    
    print(f"\n[完成] 总共完成 {sum(gym_loop.sampled_epoches for gym_loop in gym_loops)} 个 episodes")
    
    # 清理资源
    for gym_loop in gym_loops:
        gym_loop.close()
    for remote_env in remote_envs:
        remote_env.close()
    if remote_policy is not None:
        remote_policy.close()
    
    print("[退出] 所有资源已清理")

if __name__ == "__main__":
    main()


