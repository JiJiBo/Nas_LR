from stable_baselines3 import PPO

from PLE.FlappyBirdEnv import FlappyBirdGymEnv
from PLE.SB3CompatWrapper import SB3CompatWrapper

vec_env = SB3CompatWrapper(FlappyBirdGymEnv())
model = PPO.load("./logs/best_model.zip", env=vec_env)

# 6. 交互式运行一集
obs, _ = vec_env.reset()
done = False
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = vec_env.step(action)
    vec_env.render("human")
print(f"本次测试结束，得分：{info.get('score', None)}")
