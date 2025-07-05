import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from PLE.SB3CompatWrapper import SB3CompatWrapper
from mario.make_env import SuperMarioBrosEnv


def make_mario_env():
    return SuperMarioBrosEnv()


eval_env = DummyVecEnv([make_mario_env])
eval_env = VecFrameStack(eval_env, n_stack=4)
eval_env = VecTransposeImage(eval_env)

model = PPO.load("./logs/best_model/best_model.zip", env=eval_env)
test_env = make_mario_env()
obs = test_env.reset()
done = False
while not done:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    test_env.render()
print("测试结束，最终得分：", info.get("score", None))
