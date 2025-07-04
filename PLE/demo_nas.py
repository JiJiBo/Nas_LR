from time import sleep

from PLE.FlappyBirdEnv import FlappyBirdGymEnv

if __name__ == '__main__':
    env = FlappyBirdGymEnv()
    obs, _ = env.reset()
    for i in range(300):
        # 选 0（不跳）或 1（跳），这里全用 0
        obs, r, done, _, info = env.step(1)
        # 渲染 human 模式
        env.render("human")
        # 每帧等待一下，保持约 fps
        sleep(1.0 / env.fps)
        print(f'reward {r}')
        if done:
            print("Game Over, score=", info["score"])
            obs, _ = env.reset()
    env.close()
