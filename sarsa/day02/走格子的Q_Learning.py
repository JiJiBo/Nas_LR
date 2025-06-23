import gym
import numpy as np
import time


class Q_LearningAgent:
    def __init__(self, box_num, act_num, lr=0.1, lambada=0.9, gamma=0.1):
        self.act_num = act_num
        self.lr = lr
        self.lambada = lambada
        self.gamma = gamma
        self.q_table = np.zeros((box_num, act_num))

    def sample(self, obs):
        if np.random.uniform(0, 1) < self.gamma:
            action = np.random.randint(0, self.act_num)
        else:
            action = self.predict(obs)
        return action

    def learn(self, obs, action, reward, next_obs, done):
        predict_Q = self.q_table[obs, action]
        if done:
            target = reward
        else:
            target = reward + self.lambada * np.max(self.q_table[next_obs])
        self.q_table[obs, action] += self.lr * (target - predict_Q)

    def predict(self, obs):
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
    total_reward = 0
    total_step = 0
    obs, _ = env.reset()
    while True:
        action = agent.sample(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        agent.learn(obs, action, reward, next_obs, done)
        obs = next_obs
        total_reward += reward
        total_step += 1
        if render:
            time.sleep(0.1)
            env.render()
        if done:
            break
    return total_reward, total_step


def test_episode(agent, env, render=True):
    total_reward = 0
    total_step = 0
    obs, _ = env.reset()
    while True:
        action = agent.predict(obs)
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        obs = next_obs
        total_reward += reward
        total_step += 1
        if render:
            time.sleep(0.1)
            env.render()
        if done:
            break
    return total_reward, total_step


if __name__ == '__main__':
    env = gym.make('FrozenLake-v1', is_slippery=False)
    box_num = env.observation_space.n
    act_num = env.action_space.n
    agent = Q_LearningAgent(box_num, act_num)
    for episode in range(20000):
        total_reward, total_step = train_episode(agent, env, render=False)
        print('episode:{} total_step:{} total_reward:{}'.format(episode, total_step, total_reward))
    print(agent.q_table)
    agent.save()
    env = gym.make('FrozenLake-v1', render_mode="human", is_slippery=False)
    test_episode(agent, env)
