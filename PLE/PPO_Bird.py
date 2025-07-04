import os
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback

from PLE.FlappyBirdEnv import FlappyBirdGymEnv
from PLE.SB3CompatWrapper import SB3CompatWrapper

# 1. 创建环境
vec_env = SB3CompatWrapper(FlappyBirdGymEnv())

# 2. 加载或新建模型
if os.path.exists("./logs/ppo_flappybird.zip"):
    # 注意：PPO.load 返回一个新模型，需要赋值给 model
    model = PPO.load("./logs/ppo_flappybird.zip", env=vec_env)
else:
    model = PPO("MlpPolicy", vec_env, verbose=1)

eval_callback = EvalCallback(
    vec_env,
    best_model_save_path='./logs/',
    log_path='./logs/',
    eval_freq=50_000,
    n_eval_episodes=10,

)

model.learn(total_timesteps=5_000_000, callback=eval_callback)

# 4. 保存
model.save("./logs/ppo_flappybird")

# 5. 测试时重新加载，并绑定环境
model = PPO.load("./logs/ppo_flappybird.zip", env=vec_env)

# 6. 交互式运行一集
obs, _ = vec_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = vec_env.step(action)
    vec_env.render("human")
print(f"本次测试结束，得分：{info.get('score', None)}")
