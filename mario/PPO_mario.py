import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack, VecTransposeImage

from PLE.SB3CompatWrapper import SB3CompatWrapper
from mario.make_env import SuperMarioBrosEnv


def make_mario_env():
    return SuperMarioBrosEnv()


def trian_mario():
    # —— 2. 并行训练环境 ——
    # 创建 8 个实例，并在最外层堆栈 4 帧
    train_env = DummyVecEnv([make_mario_env for _ in range(8)])
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecTransposeImage(train_env)

    # —— 3. 评估环境 ——
    # 单实例，n_stack 和转置同训练
    eval_env = DummyVecEnv([make_mario_env])
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # —— 4. 模型加载或新建 ——
    model_path = "./logs/ppo_mario.zip"
    if os.path.exists(model_path):
        model = PPO.load(
            model_path,
            env=train_env,
            device="cuda",
            tensorboard_log="./logs/"
        )
    else:
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log="./logs/",
            learning_rate=2.5e-4,
            n_steps=128,
            batch_size=64,
            n_epochs=4,
            gamma=0.99,
            device="cuda",
            clip_range=0.1
        )

    # —— 5. 设置评估回调 ——
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path='./logs/best_model',
        log_path='./logs/eval',
        eval_freq=100_000,  # 每 10 万步评估一次
        n_eval_episodes=5,  # 评估 5 条回合
        deterministic=True
    )

    # —— 6. 开始训练 ——
    model.learn(
        total_timesteps=5_000_000,
        callback=eval_callback,
        tb_log_name="PPO-Mario"
    )

    # —— 7. 保存最终模型 ——
    model.save(model_path)

    # —— 8. 测试 ——
    # 为了方便 demo，这里直接用一个纯粹的单体环境来跑
    test_env = make_mario_env()
    obs = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
    print("测试结束，最终得分：", info.get("score", None))
