import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from PLE.FlappyBirdEnv import FlappyBirdGymEnv
from PLE.SB3CompatWrapper import SB3CompatWrapper

# —— 1. 构造并行训练环境 ——
def make_env():
    return SB3CompatWrapper(FlappyBirdGymEnv())

train_env = DummyVecEnv([make_env for _ in range(8)])  # 8 个并行环境
eval_env = SB3CompatWrapper(FlappyBirdGymEnv())       # 单独评估环境

# —— 2. 新建或加载模型 ——
if os.path.exists("./logs/ppo_flappybird.zip"):
    model = PPO.load(
        "./logs/ppo_flappybird.zip",
        env=train_env,
        tensorboard_log="./logs/"
    )
else:
    model = PPO(
        "MlpPolicy",
        train_env,
        verbose=1,
        tensorboard_log="./logs/"
    )

# —— 3. 设置评估回调 ——
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path='./logs/best_model',
    log_path='./logs/eval',
    eval_freq=50_000,
    n_eval_episodes=10,
    deterministic=True
)

# —— 4. 开始训练 ——
model.learn(
    total_timesteps=5_000_000,
    callback=eval_callback,
    tb_log_name="PPO-FlappyBird",
)

# —— 5. 保存最终模型 ——
model.save("./logs/ppo_flappybird")

# —— 6. 测试 ——
model = PPO.load("./logs/ppo_flappybird.zip", env=eval_env)
obs, _ = eval_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = eval_env.step(action)
    eval_env.render("human")
print("测试结束，得分：", info.get("score"))
