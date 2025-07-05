from typing import Optional

import gymnasium as gym
import numpy as np
import gym_super_mario_bros
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT


class SuperMarioBrosEnv(gym.Env):
    """
    自定义 Super Mario Bros Gym 环境
    动作空间：SIMPLE_MOVEMENT（按 NES 键位预定义的动作集）
    观测空间：RGB 原始像素
    奖励：使用游戏实际得分增量
    """
    # 新写法
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, stage='SuperMarioBros-1-1-v0', fps=60):
        super().__init__()
        # 创建原始环境并只保留 SIMPLE_MOVEMENT 动作集
        env = gym_super_mario_bros.make(
            'SuperMarioBros-1-1-v0',
            render_mode='rgb_array',  # 或 'human'
            apply_api_compatibility=True
        )
        self.env = JoypadSpace(env, SIMPLE_MOVEMENT)

        self.action_space = gym.spaces.discrete.Discrete(self.env.action_space.n)

        # 观测空间：取原始像素（H, W, C）
        h, w, c = self.env.observation_space.shape
        self.observation_space = gym.spaces.box.Box(
            low=0, high=255, shape=(h, w, c), dtype=np.uint8
        )

        # 帧率控制
        self.fps = fps
        self.dt = int(1000 / fps)

        # 用于计算得分增量
        self.prev_score = 0

    def reset(self, *, seed=None, options=None):
        # 调用底层 reset，可能返回 (obs, info) 或 仅 obs
        result = self.env.reset()
        if isinstance(result, tuple) and len(result) == 2:
            obs, _info = result
        else:
            obs = result
        # 重置内部状态
        self.prev_score = 0
        return obs, {}

    def step(self, action):
        """
        执行动作，返回 5 元组以兼容新版 Gym：
        obs        : (H, W, C) uint8
        reward     : 本步总奖励
        terminated : bool（本回合是否自然结束，比如到达终点）
        truncated  : bool（是否被截断，比如超过最大步数）
        info       : 包含 'score'、'flag_get' 等游戏信息
        """
        # 1) 执行动作
        obs, raw_reward, terminated, truncated, info = self.env.step(action)

        # 2) 计算是否结束
        done = terminated or truncated

        # —— 得分增量奖励 ——
        current_score = info.get('score', 0)
        score_reward = current_score - self.prev_score
        self.prev_score = current_score

        # —— 生存奖励（可选） ——
        survival_reward = -0.001  # 也可以用小正值鼓励存活

        # —— 死亡惩罚 ——
        death_penalty = -10.0 if done and info.get('life', 1) == 0 else 0.0

        # —— 时间惩罚（让 agent 感到紧迫） ——
        # 每一步扣 0.05 分，步数越多累积扣分越严重
        time_penalty = -0.05

        # —— 通关奖励 ——
        # info['flag_get'] 为 True 表示到达终点，给一次性大额奖励
        goal_reward = 100.0 if info.get('flag_get', False) else 0.0

        # —— 汇总所有奖励 ——
        reward = score_reward + survival_reward + death_penalty + time_penalty + goal_reward

        # 3) 按 Gym API 返回 5 元组
        return obs, reward, terminated, truncated, info

    def render(self, mode="human"):
        # 底层生成一帧图像数组
        frame = self.env.render()  # 默认返回 rgb_array
        if mode == "human":
            # 直接显示
            import cv2
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("YourEnv", frame)
            cv2.waitKey(1)
        elif mode == "rgb_array":
            return frame

    def close(self):
        self.env.close()
