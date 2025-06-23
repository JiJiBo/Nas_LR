import copy
import math
import os.path
import random
import time
from collections import deque
import imageio
import cv2
import gym
import numpy as np
import torch
import tqdm
from torch import nn


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

SUCCESS_DEQUE = deque(maxlen=100)


class DQN():
    def __init__(self, model):
        self.model = model
        self.target_model = copy.deepcopy(model)
        self.model.to(DEVICE)
        self.target_model.to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)
        self.loss = nn.MSELoss()

    def predict(self, obs):
        obs = torch.FloatTensor(obs).reshape(1, -1).to(DEVICE)
        with torch.no_grad():
            return self.model(obs).argmax().item()

    def learn(self, datas):
        obs, action, reward, next_obs, done = datas
        reward = reward.to(DEVICE)
        obs = obs.to(DEVICE)
        next_obs = next_obs.to(DEVICE)
        action = action.to(DEVICE)
        predict_Q = self.model(obs).gather(1, action)
        done = done.float().to(DEVICE)
        with torch.no_grad():
            target = reward + (1 - done) * 0.99 * self.target_model(next_obs).max(1, keepdim=True)[0]
        loss = self.loss(predict_Q, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def sync_target(self):
        self.target_model.load_state_dict(self.model.state_dict())


class MemeryPool:
    def __init__(self, max_size=2000):
        self.pool = deque(maxlen=max_size)

    def __len__(self):
        return len(self.pool)

    def append(self, data):
        self.pool.extend(data)

    def clear(self):
        self.pool.clear()

    def sample(self, batch_size=100):
        data = random.sample(self.pool, batch_size)
        state = torch.FloatTensor([d[0] for d in data])
        action = torch.LongTensor([d[1] for d in data]).reshape(-1, 1)
        reward = torch.FloatTensor([d[2] for d in data]).reshape(-1, 1)
        next_state = torch.FloatTensor([d[3] for d in data])
        done = torch.FloatTensor([d[4] for d in data]).reshape(-1, 1)
        return state, action, reward, next_state, done


class NasWapper(gym.Wrapper):
    def __init__(self, render_mode='rgb_array'):
        env = gym.make('MountainCar-v0', render_mode=render_mode)
        super(NasWapper, self).__init__(env)
        self.env = env
        self.step_n = 0
        self.max_position = None
        self.min_position = None
        self.true_reward = []
        self.frames = []
        self.shape_reward = []
        self.position = 0
        self.velocity = 0

    def reset(self):
        state, _ = self.env.reset()
        self.step_n = 0
        self.true_reward = []
        self.frames = []
        self.shape_reward = []
        self.max_position = None
        self.min_position = None
        self.position = 0
        self.velocity = 0
        return state

    def step(self, action):
        state, reward, terminated, truncated, info = self.env.step(action)
        self.true_reward.append(reward)
        position, velocity = state
        self.position = position
        self.velocity = velocity
        over = terminated or truncated

        # 初始化 min/max
        if self.min_position is None:
            self.min_position = position
        if self.max_position is None:
            self.max_position = position

        shaped_reward = reward

        if position > self.max_position:
            shaped_reward += abs(position - self.max_position) * 20
            self.max_position = max(self.max_position, position)
        # elif position < self.min_position:
        #     shaped_reward += abs(self.min_position - position) * 5
        if self.min_position > position:
            shaped_reward += abs(self.min_position - position) * 5
            self.min_position = min(self.min_position, position)
        if abs(velocity) > 0.01:
            shaped_reward += 3 * abs(velocity)
        if over:
            if position > 0.5:
                shaped_reward += 100
                self.shape_reward.append(shaped_reward)
                print(f"跑到顶端了{np.mean(self.shape_reward)}")
                SUCCESS_DEQUE.append(1)
            else:
                SUCCESS_DEQUE.append(0)
                self.shape_reward.append(shaped_reward)
        else:
            self.shape_reward.append(shaped_reward)

        self.step_n += 1

        return state, shaped_reward, over, terminated

    def show(self):
        img = self.env.render()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.putText(
            img,
            f"step_n: {self.step_n}, position: {self.position:.3f}, velocity: {self.velocity:.3f}, reward: {sum(self.true_reward):.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
        )
        cv2.putText(
            img,
            f"max_position: {self.max_position:.3f}, min_position: {self.min_position:.3f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1, cv2.LINE_AA
        )
        cv2.imshow("env", img)
        cv2.waitKey(1)
        # 保存帧（BGR转RGB）
        self.frames.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def record_to_gif(self, output_dir="gifs"):
        os.makedirs(output_dir, exist_ok=True)
        timestamp = int(time.time())
        filepath = os.path.join(output_dir, f"{timestamp}.gif")
        imageio.mimsave(filepath, self.frames, fps=30)
        print(f"GIF 保存成功: {filepath}")

class DQN_Agent:
    def __init__(self, env, train=True):
        self.env = env
        self.train = train
        self.memory = MemeryPool()
        self.model = DQN(Model(2, 3))
        self.load()
        self.epsilon = 1  # 初始完全探索
        self.epsilon_decay = 0.9995
        self.epsilon_mid = 0.5
        self.epsilon_min = 0.1
        self.learn_steps = 0
        self.learn_mid_steps = 5000
        self.target_update_freq = 5

    def play(self, show=False):
        state = self.env.reset()
        over = False
        data = []
        rewards = []
        while not over:
            action = self.model.predict(state)
            if np.random.uniform(0, 1) < self.epsilon and self.train:
                action = self.env.action_space.sample()
            next_state, reward, over, terminated = self.env.step(action)
            rewards.append(reward)
            data.append((state, action, reward, next_state, terminated))
            state = next_state
            if show:
                self.env.show()
        self.memory.append(data)
        cv2.destroyAllWindows()

        return sum(self.env.true_reward), sum(rewards)

    def learn(self):
        if self.train and len(self.memory) > 100:
            for _ in range(10):  # 更快响应性训练
                self.model.learn(self.memory.sample())
                self.learn_steps += 1
                if self.learn_steps % self.target_update_freq == 0:
                    self.model.sync_target()
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
    return np.sum(list(SUCCESS_DEQUE)) / len(SUCCESS_DEQUE)


if __name__ == '__main__':
    env = NasWapper()
    env.reset()
    agent = DQN_Agent(env)
    des = tqdm.trange(10000)
    count_rewards = []
    count_re_rewards = []
    # # 初始化阶段：积累经验
    # while len(agent.memory) < 2000:
    #     agent.play()
    for i in des:
        for j in range(10):
            rewards, re_reward = agent.play(show=j == 0 and i % 200 == 0)
            count_rewards.append(rewards)
            count_rewards = count_rewards[-100:]
            count_re_rewards.append(re_reward)
            count_re_rewards = count_re_rewards[-100:]
            des.set_postfix({
                "ok_per": getSuccessPer(),
                "rewards": rewards,
                "re_reward": re_reward,
                "epsilon": agent.epsilon,
                "100_mean_rewards": np.mean(count_rewards),
                "100_mean_re_rewards": np.mean(count_re_rewards)
            })
        agent.learn()
        if i % 20 == 0:
            agent.save()
    agent.save()
    agent.train = False
    print("开始测试")
    for i in range(5):
        r, _ = agent.play(show=True)
        print(f"第{i + 1}次回报: {r}")
    test_rewards = [agent.play(show=False)[0] for _ in range(100)]
    print(f"测试平均 reward：{np.mean(test_rewards):.2f}")
