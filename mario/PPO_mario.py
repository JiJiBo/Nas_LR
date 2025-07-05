import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecFrameStack,
    VecTransposeImage,
)

from mario.make_env import SuperMarioBrosEnv


def make_mario_env():
    env = SuperMarioBrosEnv()
    return Monitor(env)


def train_mario():
    # —— 1. 并行训练环境 ——
    # 创建 16 个实例，并在最外层堆栈 4 帧
    train_env = DummyVecEnv([make_mario_env for _ in range(16)])
    train_env = VecMonitor(train_env)
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecTransposeImage(train_env)

    # —— 2. 评估环境 ——
    eval_env = DummyVecEnv([make_mario_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # —— 3. 模型加载或新建 ——
    model_path = "./logs/ppo_mario.zip"
    if os.path.exists(model_path):
        model = PPO.load(
            model_path,
            env=train_env,
            device="cuda",
            batch_size=4096,
            tensorboard_log="./logs/"
        )
    else:
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=3e-4,
            n_steps=256,
            batch_size=4096,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.01,
            device="cuda",
        )

    # —— 4. 设置评估回调 ——
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=100_000,
        n_eval_episodes=5,
        deterministic=True
    )

    # —— 5. 开始训练 ——
    model.learn(
        total_timesteps=5_000_000,
        callback=eval_callback,
        tb_log_name="PPO-Mario-16env"
    )

    # —— 6. 保存最终模型 ——
    model.save(model_path)

    # —— 7. 测试 ——
    test_env = SuperMarioBrosEnv()
    obs = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
    print("测试结束，最终得分：", info.get("score"))


if __name__ == "__main__":
    train_mario()
