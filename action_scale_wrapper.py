"""
Action Scaling Wrapper

当 demo 数据的 action 范围和策略输出范围不匹配时，
使用这个 wrapper 来缩放 action。

例如：
- 策略输出范围: [-1, 1]
- Demo action 范围: [-0.02, 0.04]
- 需要将策略输出缩放 0.04 倍（或更小）

用法：
    env = ActionScaleWrapper(env, action_scale=0.05)
    # 策略输出 1.0 → 实际发送 0.05
"""

try:
    import gymnasium as gym
except ImportError:
    import gym
import numpy as np
from typing import Optional, Union


class ActionScaleWrapper(gym.Wrapper):
    """
    缩放 action 的 wrapper
    
    action_to_env = action_from_policy * action_scale
    """
    
    def __init__(
        self, 
        env: gym.Env, 
        action_scale: Union[float, np.ndarray] = 1.0,
        clip_action: bool = True,
    ):
        """
        Args:
            env: 原始环境
            action_scale: 缩放因子（可以是标量或每维度不同的数组）
            clip_action: 是否 clip 缩放后的 action 到 action_space
        """
        super().__init__(env)
        
        if isinstance(action_scale, (int, float)):
            self.action_scale = float(action_scale)
        else:
            self.action_scale = np.array(action_scale)
        
        self.clip_action = clip_action
        
        # 注意：我们不修改 action_space，保持策略认为的范围是 [-1, 1]
        # 实际发送给环境的 action 会被缩放
        
    def step(self, action):
        # 缩放 action
        scaled_action = action * self.action_scale
        
        # 可选：clip 到 action space
        if self.clip_action:
            if hasattr(self.action_space, 'low') and hasattr(self.action_space, 'high'):
                scaled_action = np.clip(
                    scaled_action, 
                    self.action_space.low, 
                    self.action_space.high
                )
        
        return self.env.step(scaled_action)
    
    def __repr__(self):
        return f"ActionScaleWrapper({self.env}, scale={self.action_scale})"


class ActionScaleWrapperV2(gym.Wrapper):
    """
    更灵活的 action scaling wrapper
    
    支持将策略输出 [-1, 1] 映射到任意目标范围 [target_low, target_high]
    
    用法：
        # 将 [-1, 1] 映射到 [-0.05, 0.05]
        env = ActionScaleWrapperV2(env, target_low=-0.05, target_high=0.05)
    """
    
    def __init__(
        self,
        env: gym.Env,
        target_low: Union[float, np.ndarray] = -0.05,
        target_high: Union[float, np.ndarray] = 0.05,
        source_low: float = -1.0,
        source_high: float = 1.0,
    ):
        super().__init__(env)
        
        self.target_low = np.array(target_low)
        self.target_high = np.array(target_high)
        self.source_low = source_low
        self.source_high = source_high
        
        # 计算缩放参数
        # action_scaled = (action - source_low) / (source_high - source_low) 
        #                 * (target_high - target_low) + target_low
        self.source_range = source_high - source_low
        self.target_range = self.target_high - self.target_low
        
    def step(self, action):
        # 将 [source_low, source_high] 映射到 [target_low, target_high]
        normalized = (action - self.source_low) / self.source_range
        scaled_action = normalized * self.target_range + self.target_low
        
        return self.env.step(scaled_action)
    
    def __repr__(self):
        return f"ActionScaleWrapperV2({self.env}, target=[{self.target_low}, {self.target_high}])"


def estimate_action_scale_from_demo(demo_path: str, target_range: float = 0.8) -> float:
    """
    从 demo 文件估计合适的 action scale
    
    Args:
        demo_path: demo pkl 文件路径
        target_range: 目标范围（策略输出范围的占比）
    
    Returns:
        action_scale: 建议的缩放因子
    """
    import pickle
    
    with open(demo_path, 'rb') as f:
        data = pickle.load(f)
    
    all_actions = np.array([t['actions'] for t in data])
    max_abs_action = np.abs(all_actions).max()
    
    # 如果策略输出 target_range，实际 action 应该是 max_abs_action
    # action_scale = max_abs_action / target_range
    action_scale = max_abs_action / target_range
    
    print(f"Demo action 统计:")
    print(f"  Max |action| = {max_abs_action:.6f}")
    print(f"  建议 action_scale = {action_scale:.6f}")
    print(f"  策略输出 {target_range:.2f} → 实际 action {max_abs_action:.6f}")
    
    return action_scale
