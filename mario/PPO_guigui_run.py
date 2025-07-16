import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage, VecFrameStack, VecMonitor

from mario.guigui_env import ContraEnv


def make_mario_env():
    return ContraEnv()


# 1. 包装环境（和训练时一致）
env = DummyVecEnv([make_mario_env])
# env = DummyVecEnv([make_mario_env for _ in range(12)])
env = VecMonitor(env)

env = VecFrameStack(env, n_stack=4)
env = VecTransposeImage(env)

model_path = r"C:\Users\12700\Downloads\model_step_20000.zip"

model = PPO.load(model_path, env=env, batch_size=2048)

# 3. 评估
frames = []
obs = env.reset()
done = [False]
isObs = False
while not done[0]:
    action, _ = model.predict(obs, deterministic=True)
    obs, rewards, done, infos = env.step(action)
    # import cv2
    import cv2

    #
    # print(obs.shape)
    if isObs:
        cv2.imshow("YourEnv", obs[0][0])
    else:
        # cv2.waitKey(1)
        frame = env.venv.envs[0].render(mode="rgb_array")
        # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        # print(frame.shape)
        frames.append(frame.copy())

        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imshow("YourEnv", frame)
    cv2.waitKey(40)

print(len(frames))
# 4. 把帧列表保存成 GIF
#    fps 参数控制帧率，数值越大，动图播放越快
imageio.mimsave('episode.gif', frames, fps=30)

print("最终得分：", infos[0].get("score"))
