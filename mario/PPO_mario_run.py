import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from mario.make_env import SuperMarioBrosEnv


def make_mario_env():
    return SuperMarioBrosEnv()


# 1. 包装环境（和训练时一致）
env = DummyVecEnv([make_mario_env])
env = VecTransposeImage(env)
env = VecFrameStack(env, n_stack=4)

# 2. 加载模型时用 custom_objects “挂钩” 无法反序列化的部分
model_path = "/Users/nas/Downloads/ppo_mario_checkpoint_14400_steps.zip"
custom_objects = {
    # 覆盖空间检测
    "observation_space": env.observation_space,
    "action_space": env.action_space,
    # 覆盖学习率调度器和剪切范围（用常数或简单 λ）
    "lr_schedule": lambda _: 3e-4,
    "clip_range": lambda _: 0.2,
}

model = PPO.load(
    model_path,
    env=env,
    device="cpu",
    custom_objects=custom_objects,
)

# 3. 评估
obs = env.reset()
done = [False]
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, infos = env.step(action)
    env.venv.envs[0].render()

print("最终得分：", infos[0].get("score"))
