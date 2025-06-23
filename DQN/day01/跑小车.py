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
            nn.Linear(in_channel, 64),
            nn.SiLU(),
            nn.Linear(64, out_channel)
        )

    def forward(self, x):
        return self.net(x)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 统计成功率的队列
SUCCESS_DEQUE = deque(maxlen=100)


# DQN管理模型的类
class DQN():
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.model.to(DEVICE)
        self.target_model.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss = nn.MSELoss()

    def predict(self, obs):
        """
        用模型，根据环境，计算出动作
        :param obs: 环境
        :return:
        """
        obs = torch.FloatTensor(obs).reshape(1, -1).to(DEVICE)
        with torch.no_grad():
            return self.model(obs).argmax().item()

    def learn(self, datas):
        """
        学习一个批次的数据
        :param datas: 一个批次的数据
        :return:
        """
        # 分别是 环境参数、动作、奖励、下一个环境参数、是否结束
        obs, action, reward, next_obs, done = datas
        # 放到gpu上（提升不大）
        reward = reward.to(DEVICE)
        obs = obs.to(DEVICE)
        next_obs = next_obs.to(DEVICE)
        action = action.to(DEVICE)
        # 获得预测的奖励
        predict_Q = self.model(obs).gather(1, action)
        done = done.float().to(DEVICE)
        # 得到target的奖励
        with torch.no_grad():
            target = reward + (1 - done) * 0.99 * self.target_model(next_obs).max(1, keepdim=True)[0]
        # 走网络，进行梯度下降
        loss = self.loss(predict_Q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target(self):
        """
        将当前的模型的参数更新成目标模型的参数
        :return:
        """
        self.target_model.load_state_dict(self.model.state_dict())


# 一个记忆池，管理训练的数据
class MemeryPool:
    def __init__(self, max_size=2000):
        # 一个队列，存放数据，最多存 max_size 条，当超出范围，就把最早的数据，删除
        self.pool = deque(maxlen=max_size)

    def __len__(self):
        return len(self.pool)

    def append(self, data):
        # 添加数据
        self.pool.extend(data)

    def clear(self):
        self.pool.clear()

    def sample(self, batch_size=100):
        # 得到一批数据
        data = random.sample(self.pool, batch_size)
        state = torch.FloatTensor([d[0] for d in data])
        action = torch.LongTensor([d[1] for d in data]).reshape(-1, 1)
        reward = torch.FloatTensor([d[2] for d in data]).reshape(-1, 1)
        next_state = torch.FloatTensor([d[3] for d in data])
        done = torch.FloatTensor([d[4] for d in data]).reshape(-1, 1)
        return state, action, reward, next_state, done


# 一个 MountainCar 环境，为了能重塑奖励。所以，需要一个 wrapper
class NasWapper(gym.Wrapper):
    def __init__(self, render_mode='rgb_array'):
        # 获得原始环境
        env = gym.make('MountainCar-v0', render_mode=render_mode)
        super(NasWapper, self).__init__(env)
        # 把原始环境，包装成一个类
        self.env = env
        # 走的步数
        self.step_n = 0
        # 到达的最右侧距离
        self.max_position = None
        # 到达的最左侧距离
        self.min_position = None
        # 记录奖励
        self.true_reward = []
        # 记录重塑的奖励
        self.shape_reward = []
        # 当前的位置
        self.position = 0
        # 当前的速度
        self.velocity = 0

    def reset(self):
        """
        重置环境
        :return:
        """
        state, _ = self.env.reset()
        # 初始化参数
        self.step_n = 0
        self.true_reward = []
        self.shape_reward = []
        self.max_position = None
        self.min_position = None
        self.position = 0
        self.velocity = 0
        return state

    def step(self, action):
        """
        执行一个动作，返回奖励等参数
        :param action: 执行的动作
        :return:
        """
        state, reward, terminated, truncated, info = self.env.step(action)
        # 把真实的奖励记下来
        self.true_reward.append(reward)
        # 解包环境，得到位置和速度
        position, velocity = state
        self.position = position
        self.velocity = velocity
        # 计算是否结束此次回合
        over = terminated or truncated

        # 初始化 min/max
        if self.min_position is None:
            self.min_position = position
        if self.max_position is None:
            self.max_position = position

        # 初始化重塑的奖励
        shaped_reward = reward
        # 当小车刷新最右侧的记录，需要给一个大大的奖励
        if position > self.max_position:
            shaped_reward += abs(position - self.max_position) * 20
            # 更新最右侧的距离
            self.max_position = max(self.max_position, position)
        # 当小车刷新最左侧的记录，需要给一个小小的奖励
        if self.min_position > position:
            shaped_reward += abs(self.min_position - position) * 5
            # 更新最左侧的距离
            self.min_position = min(self.min_position, position)
        # 当小车，处于高速运转
        if abs(velocity) > 0.01:
            # 给一个小小的奖励
            # 速度的奖励不能太大
            shaped_reward += 3 * abs(velocity)
        # 此次回合结束
        if over:
            # 到达目标
            if position > 0.5:
                # 给一个大大的奖励
                shaped_reward += 100
                self.shape_reward.append(shaped_reward)
                print(f"跑到顶端了{np.mean(self.shape_reward)}")
                # 记录下来
                SUCCESS_DEQUE.append(1)
            else:
                # 没成功，也记下来
                SUCCESS_DEQUE.append(0)
                self.shape_reward.append(shaped_reward)
        else:
            self.shape_reward.append(shaped_reward)
        # 步数加一
        self.step_n += 1
        # 返回 状态，重塑奖励，是否结束，是否成功
        # over是此次游戏结束
        # terminated 是此次游戏成功导致的结束
        return state, shaped_reward, over, terminated

    def show(self):
        # 渲染出来，用cv2，快一些
        img = self.env.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(
            img,
            f"step_n: {self.step_n}, position: {self.position:.3f}, velocity: {self.velocity:.3f}, reward: {sum(self.true_reward):.2f}",
            (10, 30),  # 坐标
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
            0.5,  # 字体大小
            (255, 0, 0),  # 颜色（白色）
            1,  # 线宽
            cv2.LINE_AA  # 抗锯齿
        )
        cv2.putText(
            img,
            f"max_position: {self.max_position:.3f}, min_position: {self.min_position:.3f} ",
            (10, 60),  # 坐标
            cv2.FONT_HERSHEY_SIMPLEX,  # 字体
            0.5,  # 字体大小
            (255, 0, 0),  # 颜色（白色）
            1,  # 线宽
            cv2.LINE_AA  # 抗锯齿
        )
        cv2.imshow("env", img)
        cv2.waitKey(10)


# 智能体
class DQN_Agent:
    def __init__(self, env, train=True):
        # 环境
        self.env = env
        # 是否开启训练
        self.train = train
        # 记忆池
        self.memory = MemeryPool()
        # 模型管理
        self.model = DQN(Model(2, 3))
        # 加载模型
        self.load()
        # 探索相关的参数
        self.epsilon = 1  # 初始完全探索
        # 探索的衰减系数
        self.epsilon_decay = 0.9995
        # 没有用处，这个参数
        self.epsilon_mid = 0.5
        # 最小的探索概率
        self.epsilon_min = 0.1
        # 学习步数
        self.learn_steps = 0
        # 没有用处
        self.learn_mid_steps = 5000
        # 目标更新频率，这里是每 target_update_freq 步，把当前模型，更新到目标模型
        self.target_update_freq = 5

    def play(self, show=False):
        """
        玩一把游戏，一把就是一个episode
        :param show: 是否渲染画面
        :return:
        """
        # 初始化环境
        state = self.env.reset()
        # 游戏结束的标志字段
        over = False
        # 玩游戏的数据记录字段
        data = []
        # 奖励记录字段
        rewards = []
        # 当游戏没有结束
        while not over:
            # 根据模型预测一个动作
            action = self.model.predict(state)
            # 当触发随机动作事件
            if np.random.uniform(0, 1) < self.epsilon and self.train:
                # 随机一个动作
                action = self.env.action_space.sample()
            # 走一步
            next_state, reward, over, terminated = self.env.step(action)
            # 把奖励记下来
            rewards.append(reward)
            # 把数据记下来 ， ⚠️这里结束用的是terminated，而不是over
            data.append((state, action, reward, next_state, terminated))
            # 状态更新
            state = next_state
            if show:
                self.env.show()
        # 记录数据
        self.memory.append(data)
        cv2.destroyAllWindows()
        # 返回真实的奖励和重塑的奖励
        return sum(self.env.true_reward), sum(rewards)

    def learn(self):
        """
        学习一把
        :return:
        """
        # 数据足够多的话
        if self.train and len(self.memory) > 100:
            # 训练10把游戏
            for _ in range(10):
                # 训练
                self.model.learn(self.memory.sample())
                # 训练步数加1
                self.learn_steps += 1
                # 如果到了目标更新频率，更新目标模型
                if self.learn_steps % self.target_update_freq == 0:
                    self.model.sync_target()
        # 这个我写的复杂了，其实就是，每次学习，需要更新下探索概率
        if self.train:
            if self.learn_steps < self.learn_mid_steps:
                if self.epsilon > self.epsilon_mid:
                    self.epsilon *= self.epsilon_decay
            elif self.epsilon > self.epsilon_min:
                self.epsilon *= self.epsilon_decay

    def save(self):
        torch.save(self.model.model.state_dict(), "model.pkl")

    def load(self):
        if os.path.exists("model.pkl"):
            self.model.model.load_state_dict(torch.load("model.pkl"))


def getSuccessPer():
    # 获得 成功登顶的概率 0-1
    return np.sum(list(SUCCESS_DEQUE)) / len(SUCCESS_DEQUE)


if __name__ == '__main__':
    # 创建环境
    env = NasWapper()
    env.reset()
    # 创建智能体
    agent = DQN_Agent(env)
    # 开始训练
    des = tqdm.trange(10000)
    # 记录奖励
    count_rewards = []
    # 记录重塑奖励
    count_re_rewards = []
    for i in des:
        # 每次玩10把
        for j in range(10):
            # 每隔200个步骤，渲染一次画面
            rewards, re_reward = agent.play(show=j == 0 and i % 200 == 0)
            count_rewards.append(rewards)
            count_rewards = count_rewards[-100:]
            count_re_rewards.append(re_reward)
            count_re_rewards = count_re_rewards[-100:]
            des.set_postfix({
                # 成功率
                "ok_per": getSuccessPer(),
                # 奖励
                "rewards": rewards,
                # 重塑奖励
                "re_reward": re_reward,
                # 当前的探索率
                "epsilon": agent.epsilon,
                # 100个回合的平均奖励
                "100_mean_rewards": np.mean(count_rewards),
                # 100个回合的平均重塑奖励
                "100_mean_re_rewards": np.mean(count_re_rewards)
            })
        # 学习
        agent.learn()
        if i % 20 == 0:
            # 保存一次模型
            agent.save()
    agent.save()
    agent.train = False
    print("开始测试")
    for i in range(5):
        r, _ = agent.play(show=True)
        print(f"第{i + 1}次回报: {r}")
    test_rewards = [agent.play(show=False)[0] for _ in range(100)]
    print(f"测试平均 reward：{np.mean(test_rewards):.2f}")
