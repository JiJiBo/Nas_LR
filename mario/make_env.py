import gym_super_mario_bros
from gym.wrappers import GrayScaleObservation
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT

from stable_baselines3.common.atari_wrappers import WarpFrame, MaxAndSkipEnv


# —— 1. 环境构造函数 ——
def make_mario_env():
    # 1.1 原始环境
    env = gym_super_mario_bros.make('SuperMarioBros-1-1-v3')
    # 1.2 限定动作集，减少输出维度
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    # 1.3 跳帧：每 4 帧决策一次
    env = MaxAndSkipEnv(env, skip=4)
    # 1.4 灰度化并保留通道维度
    env = GrayScaleObservation(env, keep_dim=True)
    # 1.5 缩放到 84×84
    env = WarpFrame(env, width=84, height=84)
    return env
