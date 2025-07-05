import os

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
# —— 3. 模型加载或新建 ——
model_path = "./logs/ppo_mario.zip"
if os.path.exists(model_path):
    model = PPO.load(
        model_path,
        env=test_env,
        batch_size=2048,
        tensorboard_log="./logs/"
    )
else:
    model = PPO(
        policy="CnnPolicy",
        env=test_env,
        verbose=1,
        tensorboard_log="./logs/",
        learning_rate=3e-4,
        n_steps=256,
        batch_size=2048,
        n_epochs=10,
        gamma=0.99,
        clip_range=0.2,
        ent_coef=0.01,
    )

while not done[0]:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, done, infos = test_env.step(action)
    # VecEnv 没有直接 render，要下钻到第 0 个 env
    test_env.envs[0].render()
    print("测试结束，最终得分：", rewards)
print("测试结束，最终得分：", infos[-1].get("score", None))
