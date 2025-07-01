import copy
import os.path
import random
from collections import deque
from time import sleep

import cv2
import gym
import numpy as np
import torch
import tqdm
from torch import nn


# 智能体
class QL_Agent:
    def __init__(self, train=True):
        self.pool = DataPool()
        self.env = NasWapper()
        self.train = train
        self.box_num = self.env.env.observation_space.n
        self.act_num = self.env.env.action_space.n
        self.Q_table = np.zeros((self.box_num, self.act_num))

    def reset(self):
        state = self.env.reset()
        return state

    def learn(self):
        datas = self.pool.sample(64)
        for data in datas:

            state, action, reward, next_state, over, next_action = data
            if over:
                target = reward
            else:
                target = reward + 0.9 * self.Q_table[next_state, next_action]
            self.Q_table[state, action] += 0.1 * (target - self.Q_table[state, action])

    def save(self):
        npy_file = './q_table_sarsa.npy'
        np.save(npy_file, self.Q_table)
        print(npy_file + ' saved.')

    def load(self):
        npy_file = './q_table_sarsa.npy'
        self.Q_table = np.load(npy_file)
        print(npy_file + ' loaded.')

    def sample_action(self, state, is_train):
        if np.random.uniform(0, 1) < 0.01 and is_train:
            action = np.random.randint(0, self.act_num)
        else:
            action = np.argmax(self.Q_table[state])
        return action


# 数据池
class DataPool:
    def __init__(self, max_len=20000):
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
        self.step_n += 1
        if self.step_n >= 200:
            over = True
        return state, reward, over

    def show(self):
        # 渲染出来，用cv2，快一些
        img = self.env.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow("env", img)
        cv2.waitKey(10)


def train():
    # 实例化一个agent
    agent = QL_Agent()
    agent.train = True
    des = tqdm.tqdm(range(4000))
    for i in des:
        for j in range(64):
            r, _ = run_one_episode(agent, is_train=True)
            des.set_description(f"episode:{i}, reward:{r}")
        agent.learn()
    agent.save()


def test():
    agent = QL_Agent()
    agent.train = False
    agent.load()
    run_one_episode(agent, is_train=False)


def run_one_episode(agent, is_train=True):
    # 初始化环境
    state = agent.reset()
    total_reward = 0
    total_step = 0
    over = False
    action = agent.sample_action(state, is_train)
    while not over:

        next_state, reward, over = agent.env.step(action)
        next_action = agent.sample_action(next_state, is_train)
        agent.pool.append([state, action, reward, next_state, over, next_action])
        total_reward += reward
        total_step += 1
        state = next_state
        action = next_action
        if not is_train:
            sleep(0.2)
            agent.env.show()
    return total_reward, total_step


if __name__ == '__main__':
    train()
    test()
