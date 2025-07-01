import copy
import os.path
import random
from collections import deque

import cv2
import gym
import numpy as np
import torch
import tqdm
from torch import nn


# 智能体
class QL_Agent:
    def __init__(self):
        pass


# 数据池
class DataPool:
    def __init__(self, max_len=2000):
        self.memery = []
        self.max_len = max_len

    def __len__(self):
        return len(self.memery)

    def append(self, data):
        self.memery.append(data)
        self.memery = self.memery[-self.max_len:]

    def sample(self, size=64):
        return random.sample(self.memery, size)


# 一个 CliffWalking 环境
class NasWapper(gym.Wrapper):
    def __init__(self, render_mode='rgb_array'):
        # 获得原始环境
        env = gym.make('CliffWalking-v0', render_mode=render_mode)
        super(NasWapper, self).__init__(env)
        # 把原始环境，包装成一个类
        self.env = env
        # 走的步数
        self.step_n = 0

    def reset(self):
        """
        重置环境
        :return:
        """
        state, _ = self.env.reset()
        # 初始化参数
        self.step_n = 0
        return state

    def step(self, action):
        """
        执行一个动作，返回奖励等参数
        :param action: 执行的动作
        :return:
        """
        state, reward, terminated, truncated, info = self.env.step(action)
        # 计算是否结束此次回合
        over = terminated or truncated

        return state, reward, over

    def show(self):
        # 渲染出来，用cv2，快一些
        img = self.env.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("env", img)
        cv2.waitKey(10)


def train():
    pass


def test():
    pass


def run_one_episode(is_train=True):
    pass


if __name__ == '__main__':
    pass
