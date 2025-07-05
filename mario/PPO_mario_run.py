from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack

from mario.make_env import SuperMarioBrosEnv


# 1. 保持和训练时相同的 make_env
def make_mario_env():
    return SuperMarioBrosEnv()


# 2. 用 DummyVecEnv 包装—even for a single env
test_env = DummyVecEnv([make_mario_env])

# 3. HWC→CHW
# test_env = VecTransposeImage(test_env)

# 4. 堆叠 4 帧
test_env = VecFrameStack(test_env, n_stack=4)

# 5. reset() 会得到 shape = (1, 12, 240, 256)
obs = test_env.reset()
done = [False]
model = PPO(
    policy="CnnPolicy",
    env=test_env,
    verbose=1,
    tensorboard_log="./logs/",
    learning_rate=2.5e-4,
    n_steps=128,
    batch_size=64,
    n_epochs=4,
    gamma=0.99,
    clip_range=0.1
)
while not done[0]:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, infos = test_env.step(action)
    # VecEnv 没有直接 render，要下钻到第 0 个 env
    test_env.envs[0].render()

print("测试结束，最终得分：", infos[0].get("score", None))
