import gym
import numpy as np
import time


# 定义智能体
class SarsaAgent:
    def __init__(self, box_num, act_num, lr=0.1, lambada=0.9, gamma=0.01):
        # 获得动作空间
        self.act_num = act_num
        # 设置 学习率
        self.lr = lr
        # 设置 衰减率
        self.lambada = lambada
        # 设置动作随机性
        self.gamma = gamma
        # Q 表
        self.q_table = np.zeros((box_num, act_num))

    def sample(self, obs):
        """
        根据环境，获取价值最大的1个动作(有几率是随机的)
        :param obs: 环境参数
        :return: 动作
        """
        if np.random.uniform(0, 1) < self.gamma:
            # 随机动作
            # action = self.predict(obs)
            action = np.random.randint(0, self.act_num)
        else:
            # 价值最大的动作
            action = self.predict(obs)
        return action

    def learn(self, obs, action, reward, next_obs, next_action, done):
        """
        训练
        :param obs:当前的环境
        :param action: 当前的动作
        :param reward: 奖励
        :param next_obs: 下一步的环境
        :param next_action: 下一步的动作
        :param done: 是否结束
        """
        predict_Q = self.q_table[obs, action]
        # 获取当前的真实状态价值
        if done:
            target = reward
        else:
            target = reward + self.lambada * self.q_table[next_obs, next_action]
        # 更新当前的预测状态价值
        self.q_table[obs, action] += self.lr * (target - predict_Q)

    def predict(self, obs):
        """
        根据环境，获取价值最大的1个动作
        :param obs: 环境参数
        :return: 动作
        """
        Q_list = self.q_table[obs, :]
        max_Q = np.max(Q_list)
        action_list = []
        for i in range(self.act_num):
            if Q_list[i] == max_Q:
                action_list.append(i)
        return np.random.choice(action_list)

    # 保存Q表格数据到文件
    def save(self):
        npy_file = './q_table.npy'
        np.save(npy_file, self.q_table)
        print(npy_file + ' saved.')

    # 从文件中读取Q值到Q表格中
    def restore(self, npy_file='./q_table.npy'):
        self.q_table = np.load(npy_file)
        print(npy_file + ' loaded.')


def train_episode(agent, env, render=False):
    """
    训练agent
    :param agent: 智能体（sarsa）
    :param env: 环境
    :param render: 是否渲染画面
    :return:
    """
    # 总的步数
    total_step = 0
    # 总的价值
    total_reward = 0
    # 初始化环境
    obs, _ = env.reset()
    # 得到初始化的动作
    action = agent.sample(obs)
    while True:
        # 进行一步交互
        re = env.step(action)
        next_obs, reward, done, truncated, info = re
        # 得到下一步的动作
        next_action = agent.sample(next_obs)
        # 学习
        agent.learn(obs, action, reward, next_obs, next_action, done)
        # 更新
        obs = next_obs
        action = next_action
        # 统计
        total_step += 1
        total_reward += reward
        if render:
            time.sleep(1)
            env.render()
        if done:
            break
    return total_step, total_reward


def run_episode(agent, env, render=True):
    """
    测试agent
    :param agent: 智能体（sarsa）
    :param env: 环境
    :param render: 是否渲染画面
    :return:
    """
    # 总的步数
    total_step = 0
    # 总的价值
    total_reward = 0
    # 初始化环境
    obs, _ = env.reset()

    while True:
        # 得到当前动作
        action = agent.predict(obs)
        # 进行一步交互
        obs, reward, done, _, _ = env.step(action)
        print(obs)
        if render:
            time.sleep(0.3)
            env.render()
        if done:
            break
    return total_step, total_reward


if __name__ == '__main__':
    # 创建悬崖环境
    env = gym.make('FrozenLake-v1', is_slippery=False)
    agent = SarsaAgent(
        env.observation_space.n,
        env.action_space.n,
    )
    episodes = 10000
    for i in range(episodes):
        ep_steps, ep_reward = train_episode(agent, env)
        print(f" episode = {i}.  ep_reward = {ep_reward}, ep_steps = {ep_steps}")
    env = gym.make('FrozenLake-v1', is_slippery=False, render_mode="human")

    run_episode(agent, env)
