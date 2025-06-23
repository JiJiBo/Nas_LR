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


# 简单的模型
class Model(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_channel, 128),
            nn.SiLU(),
            nn.Linear(128, out_channel)
        )

    def forward(self, x):
        return self.net(x)


DEVICE = "cpu"

# 统计成功率的队列
SUCCESS_DEQUE = deque(maxlen=100)


class REINFORCE():
    def __init__(self, model):
        self.model = model
        self.model.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss = nn.MSELoss()

    def predict(self, obs):
        """
        用模型，根据环境，计算出动作
        :param obs: 环境
        :return:
        """
        obs = torch.FloatTensor(obs).reshape(1, -1).to(DEVICE)
        return self.model(obs)


# 一个记忆池，管理训练的数据
class MemeryPool:
    def __init__(self):
        # 一个队列，存放数据，最多存 max_size 条，当超出范围，就把最早的数据，删除
        self.s_pool = []
        self.a_pool = []
        self.r_pool = []

    def __len__(self):
        return len(self.s_pool)

    def append(self, s, a, r):
        self.s_pool.append(s)
        self.a_pool.append(a)
        self.r_pool.append(r)

    def clear(self):
        self.s_pool = []
        self.a_pool = []
        self.r_pool = []

    def compute_returns(self):
        G = []
        R = 0
        for r in reversed(self.r_pool):
            R = r + 0.99 * R
            G.insert(0, R)
        return torch.tensor(G).float().to(DEVICE)


# 一个 MountainCar 环境，为了能重塑奖励。所以，需要一个 wrapper
class NasWapper(gym.Wrapper):
    def __init__(self, render_mode='rgb_array'):
        # 获得原始环境
        env = gym.make('CartPole-v1', render_mode=render_mode)
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


# 智能体
class REINFORCE_Agent:
    def __init__(self, env, train=True):
        self.env = env
        self.train = train
        self.memory = MemeryPool()
        state_dim = env.env.observation_space.shape[0]
        action_dim = env.env.action_space.n
        self.model = REINFORCE(Model(state_dim, action_dim))
        self.load()

    def play(self, show=False):
        """
        玩一把游戏，一把就是一个episode
        :param show: 是否渲染画面
        :return:
        """
        state = self.env.reset()
        total_rewards = []
        over = False
        while not over:
            state = torch.tensor(state).to(DEVICE).unsqueeze(0)
            probs = self.model.predict(state)
            action_dist = torch.distributions.Categorical(torch.softmax(probs, dim=1))
            action = action_dist.sample()
            next_state, reward, over = self.env.step(action.item())
            total_rewards.append(reward)
            self.memory.append(state, action, reward)
            state = next_state
            if over:
                loss = 0
                if self.train:
                    G = self.memory.compute_returns()
                    for i in range(len(self.memory)):
                        s, a = self.memory.s_pool[i], self.memory.a_pool[i]
                        s, a = torch.tensor(s).float().to(DEVICE), torch.tensor(a).long().to(DEVICE)
                        pi = self.model.predict(s)
                        dist = torch.distributions.Categorical(torch.softmax(pi, dim=1))
                        log_prob = dist.log_prob(a)
                        loss -= log_prob * G[i]
                    self.model.optimizer.zero_grad()
                    loss.backward()
                    self.model.optimizer.step()
                self.memory.clear()
                break
            if show:
                self.env.show()
            if sum(total_rewards) >= 475:
                print("Solved!")
                break
        cv2.destroyAllWindows()
        return sum(total_rewards)

    def save(self):
        torch.save(self.model.model.state_dict(), "model.pkl")

    def load(self):
        if os.path.exists("model.pkl"):
            self.model.model.load_state_dict(torch.load("model.pkl"))


def getSuccessPer():
    # 获得 成功登顶的概率 0-1
    return np.sum(list(SUCCESS_DEQUE)) / len(SUCCESS_DEQUE)


def train():
    env = NasWapper()
    env.reset()
    # 创建智能体
    agent = REINFORCE_Agent(env)
    # 开始训练
    des = tqdm.trange(100)
    for i in des:
        agent.play(show=False)
        agent.save()
        if i % 10 == 0:
            test_rewards = [agent.play(show=False) for _ in range(100)]
            print(f"测试平均 reward：{np.mean(test_rewards):.2f}")
            print(getSuccessPer())
            SUCCESS_DEQUE.append(np.mean(test_rewards) > 475)
            des.set_description(f"成功率：{np.sum(list(SUCCESS_DEQUE)) / len(SUCCESS_DEQUE):.2f}")


def predict():
    env = NasWapper()
    env.reset()
    agent = REINFORCE_Agent(env)
    agent.load()
    agent.play(show=True)


if __name__ == '__main__':
    # train()
    predict()
