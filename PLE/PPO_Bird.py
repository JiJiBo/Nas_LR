import os.path

from stable_baselines3 import PPO

from PLE.FlappyBirdEnv import FlappyBirdGymEnv
from PLE.SB3CompatWrapper import SB3CompatWrapper

vec_env = SB3CompatWrapper(FlappyBirdGymEnv())
model = PPO("MlpPolicy", vec_env, verbose=1)
if os.path.exists("ppo_flappybird"):
    model.load("ppo_flappybird")
model.learn(total_timesteps=999999)
model.save("ppo_flappybird")

del model  # remove to demonstrate saving and loading

model = PPO.load("ppo_cartpole")

obs, _ = vec_env.reset()
while True:
    action, _states = model.predict(obs)
    obs, rewards, dones, _, info = vec_env.step(action)
    vec_env.render("human")
    if dones:
        break
