import gym
from ple import PLE
from ple.games.flappybird import FlappyBird


class FlappyBirdEnv(gym.Env):
    def __init__(self):
        self.game = FlappyBird()
        self.p = PLE(self.game, fps=30, display_screen=False)
        self.p.init()
        # 离散动作：0=不跳，1=跳
        self.action_space = gym.spaces.Discrete(2)
        # 简化状态：下一管道水平/垂直距离 + 小鸟速度
        self.observation_space = gym.spaces.Box(
            low=0, high=1, shape=(3,), dtype=float
        )

    def reset(self):
        self.p.reset_game()
        return self._get_obs()

    def step(self, action):
        reward = self.p.act(1 if action == 1 else None)
        done = self.p.game.game_over()
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        bx, by = self.p.game.getBirdPos()  # 小鸟在画面中的 (x,y) 像素坐标
        px, py = self.p.game.getNextPipePos()  # 下一个水管顶端/底端的 (x,y) 坐标
        vy = self.p.game.getPlayerVelY()  # 小鸟当前的垂直速度（像素/帧）
        # 归一化
        return [(px - bx) / 288, (py - by) / 512, (vy + 10) / 20]
