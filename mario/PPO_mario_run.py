import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack
from mario.make_env import SuperMarioBrosEnv


# —— 1. 保持和训练时相同的 make_env
def make_mario_env():
    return SuperMarioBrosEnv()


# —— 2. 包装环境
env = DummyVecEnv([make_mario_env])
# 2.1 HWC → CHW
env = VecTransposeImage(env)
# 2.2 堆叠 4 帧
env = VecFrameStack(env, n_stack=4)

# —— 3. 重置
obs = env.reset()
done = [False]
score = None

# —— 4. 加载或新建模型
model_path = "/Users/nas/Downloads/ppo_mario_checkpoint_14400_steps.zip"
if os.path.exists(model_path):
    # 在 load 时传入 env 以验证空间一致性
    model = PPO.load(model_path, env=env)
else:
    model = PPO(
        policy="CnnPolicy",
        env=env,
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

# —— 5. 评估循环
while not done[0]:
    action, _states = model.predict(obs, deterministic=True)
    obs, rewards, dones, infos = env.step(action)
    done = dones
    score = infos[0].get("score", None)
    # 渲染底层单环境
    env.venv.envs[0].render()

# —— 6. 输出最终得分
print("本次测试最终得分：", score)
