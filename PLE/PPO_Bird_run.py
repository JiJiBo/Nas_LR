from stable_baselines3 import PPO
from PLE.FlappyBirdEnv import FlappyBirdGymEnv
from PLE.SB3CompatWrapper import SB3CompatWrapper
import imageio
import numpy as np

# 创建环境和加载模型
env = SB3CompatWrapper(FlappyBirdGymEnv())
model = PPO.load("./logs/best_model/best_model.zip", env=env)

# 重置环境
obs, _ = env.reset()
done = False

# 用来存帧
frames = []

while not done:
    # 选择动作
    action, _states = model.predict(obs, deterministic=True)
    # 交互
    obs, reward, done, _, info = env.step(action)
    # 以 rgb_array 模式渲染并收集帧
    frame = env.render(mode="human")  # 应返回形状为 (H, W, 3) 的 numpy 数组
    frames.append(frame)
    print(len(frames))

# 保存为 GIF
# 这里 fps 可根据你的 env.dt 或者游戏帧率自行调整
imageio.mimsave("flappybird_test.gif", frames, fps=30)

print(f"本次测试结束，得分：{info.get('score', None)}，GIF 已保存为 flappybird_test.gif")
