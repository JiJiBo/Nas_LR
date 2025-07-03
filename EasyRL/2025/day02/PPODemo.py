import gym
import torch
import torch.nn as nn
from torch.distributions import Categorical

# 创建环境：CartPole-v1，是一个经典的控制问题
env = gym.make("CartPole-v1")

# 超参数设置
lr = 3e-4  # 学习率
gamma = 0.99  # 折扣因子
eps_clip = 0.2  # PPO中使用的clip范围
K_epochs = 4  # 每个episode中PPO的优化轮数

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 定义Actor-Critic网络，共享前面一部分的特征提取层
class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        # 共享网络部分（可以提取状态特征）
        self.shared = nn.Sequential(
            nn.Linear(4, 64),
            nn.Tanh(),  # 激活函数，非线性
        )
        # actor网络输出动作的概率分布
        self.actor = nn.Sequential(
            nn.Linear(64, 2),
            nn.Softmax(dim=-1)  # 输出为概率分布
        )
        # critic网络输出状态值V(s)
        self.critic = nn.Linear(64, 1)

    def forward(self, x):
        x = self.shared(x)  # 提取特征
        return self.actor(x), self.critic(x)  # 返回策略和状态值

    # 用于采样动作
    def act(self, state):
        state = torch.FloatTensor(state).to(device)
        probs, _ = self.forward(state)
        dist = Categorical(probs)  # 离散动作分布
        action = dist.sample()  # 采样动作
        return action.item(), dist.log_prob(action)  # 返回动作及其对数概率

    # 用于训练中的评估阶段
    def evaluate(self, states, actions):
        probs, values = self.forward(states)
        dist = Categorical(probs)
        return dist.log_prob(actions), dist.entropy(), values.squeeze()


# GAE优势函数计算（Generalized Advantage Estimation）
def compute_gae(rewards, values, next_value, dones, gamma=0.99, lam=0.95):
    values = values + [next_value]  # 把最后一个状态的值加入
    gae = 0
    returns = []
    # 从后往前计算每一步的优势值
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * (1 - dones[step]) - values[step]
        gae = delta + gamma * lam * (1 - dones[step]) * gae
        returns.insert(0, gae + values[step])  # 目标值等于GAE + V(s)
    return returns


# 初始化网络和优化器
model = ActorCritic().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

# 主训练循环
for episode in range(1000):
    # 每一轮采样的buffer
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []

    state, _ = env.reset()
    done = False
    ep_reward = 0

    # 与环境交互，采集一整条轨迹
    while not done:
        action, log_prob = model.act(state)
        value = model.forward(torch.FloatTensor(state).to(device))[1]

        next_state, reward, done, _, _ = env.step(action)  # 与环境交互，获取新状态和奖励

        # 保存轨迹数据
        states.append(torch.FloatTensor(state))
        actions.append(torch.tensor(action))
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob)
        values.append(value.item())

        state = next_state
        ep_reward += reward

    # 最后一个状态的值，用于GAE计算
    with torch.no_grad():
        next_value = model.forward(torch.FloatTensor(state).to(device))[1].item()

    # 计算GAE目标值与优势函数
    returns = compute_gae(rewards, values, next_value, dones)
    advantages = torch.FloatTensor(returns) - torch.FloatTensor(values)

    # 转换为 tensor 并移动到设备
    states = torch.stack(states).to(device)
    actions = torch.stack(actions).to(device)
    old_log_probs = torch.stack(log_probs).detach().to(device)
    returns = torch.FloatTensor(returns).to(device)

    # 优势归一化，加速训练稳定性
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    advantages = advantages.to(device)

    # PPO的优化阶段，重复多次优化策略
    for _ in range(K_epochs):
        # 当前策略下评估动作的log概率、熵和状态值
        log_probs, entropy, new_values = model.evaluate(states, actions)
        ratios = torch.exp(log_probs - old_log_probs)  # 新旧策略概率比

        # PPO目标函数
        surr1 = ratios * advantages
        surr2 = torch.clamp(ratios, 1 - eps_clip, 1 + eps_clip) * advantages

        actor_loss = -torch.min(surr1, surr2).mean()  # 策略损失（取最小以防策略更新过大）
        critic_loss = (returns - new_values).pow(2).mean()  # 值函数损失（均方误差）

        # 总损失 = actor_loss + critic_loss（乘以0.5） - entropy（鼓励探索）
        loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # 打印每轮奖励
    print(f"Episode {episode}, Reward: {ep_reward}")