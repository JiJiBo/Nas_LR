import numpy as np

from sarsa.day02.DQN_小车 import NasWapper, DQN_Agent, getSuccessPer

if __name__ == '__main__':
    env = NasWapper()
    env.reset()
    agent = DQN_Agent(env)
    agent.train = False
    print("开始测试")
    for i in range(5):
        r, _ = agent.play(show=True)
        agent.env.record_to_gif()
        print(f"第{i + 1}次回报: {r}")
    test_rewards = [agent.play(show=False)[0] for _ in range(100)]
    print(f"测试平均 reward：{np.mean(test_rewards):.2f}")
    print(getSuccessPer())
