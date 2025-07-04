import gym, pygame, numpy as np
from ple.games.flappybird import FlappyBird

class FlappyBirdGymEnv(gym.Env):
    metadata = {"render.modes": ["human", "rgb_array"]}

    def __init__(self, width=288, height=512, pipe_gap=100, fps=30):
        super().__init__()
        # —— 一定要先 init Pygame ——
        pygame.init()

        # 1. 游戏实例 + 屏幕
        self.game = FlappyBird(width, height, pipe_gap)
        self.game.rng    = np.random.RandomState()
        # 让 PLE 在真实窗口里绘制
        self.game.screen = pygame.display.set_mode((width, height))

        # 2. 定义空间
        self.action_space = gym.spaces.Discrete(2)
        low  = np.array([0, -np.inf, 0, 0, 0, 0, 0, 0], dtype=np.float32)
        high = np.array([height, np.inf, width, height, height, width, height, height], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        # 帧率 & 每步 dt（毫秒，int）
        self.fps = fps
        self.dt  = int(1000 / fps)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.game.init()
        obs = self._dict_to_obs(self.game.getGameState())
        return obs, {}

    def step(self, action):
        if action == 1:
            self.game.player.flap()
        self.game.step(self.dt)
        d = self.game.getGameState()
        obs    = self._dict_to_obs(d)
        reward = self.game.rewards.get("tick", 0.0)
        done   = self.game.game_over()
        info   = {"score": self.game.getScore()}
        return obs, reward, done, False, info

    def render(self, mode="human"):
        # 一定要先“泵”事件，否则窗口卡死不刷新
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

        if mode == "human":
            # 把 Surface 上的内容呈现到真实窗口
            pygame.display.flip()
        elif mode == "rgb_array":
            # 返回 H×W×3
            arr = pygame.surfarray.array3d(self.game.screen)
            return arr.transpose((1, 0, 2))

    def close(self):
        pygame.quit()

    def _dict_to_obs(self, d):
        return np.array([
            d["player_y"],
            d["player_vel"],
            d["next_pipe_dist_to_player"],
            d["next_pipe_top_y"],
            d["next_pipe_bottom_y"],
            d["next_next_pipe_dist_to_player"],
            d["next_next_pipe_top_y"],
            d["next_next_pipe_bottom_y"],
        ], dtype=np.float32)
