import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    EvalCallback,
    CheckpointCallback,
    BaseCallback,
)
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecMonitor,
    VecFrameStack,
    VecTransposeImage,
)

from mario.make_env import SuperMarioBrosEnv

class EntropyDecayCallback(BaseCallback):
    def __init__(self, start_coef: float, end_coef: float, decay_steps: int, verbose=0):
        super().__init__(verbose)
        self.start_coef = start_coef
        self.end_coef = end_coef
        self.decay_steps = decay_steps

    def _on_step(self) -> bool:
        frac = min(1.0, self.num_timesteps / self.decay_steps)
        new_coef = self.start_coef + frac * (self.end_coef - self.start_coef)
        self.model.ent_coef = new_coef
        return True

def make_mario_env():
    env = SuperMarioBrosEnv()
    return Monitor(env)

def train_mario():
    # —— 1. 环境准备 ——
    train_env = DummyVecEnv([make_mario_env for _ in range(16)])
    train_env = VecMonitor(train_env)
    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecTransposeImage(train_env)

    eval_env = DummyVecEnv([make_mario_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)

    # —— 2. 模型加载或新建 ——
    model_path = "./logs/ppo_mario.zip"
    if os.path.exists(model_path):
        model = PPO.load(
            model_path,
            env=train_env,
            device="cuda",
            batch_size=2048,
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
            batch_size=2048,
            n_epochs=10,
            gamma=0.99,
            clip_range=0.2,
            ent_coef=0.1,
            device="cuda",
        )

    # —— 3. 各类 Callback ——
    # 评估 Callback（每 100k 步评估一次）
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./logs/best_model",
        log_path="./logs/eval",
        eval_freq=100_000,
        n_eval_episodes=5,
        deterministic=True
    )
    # 探索衰减 Callback
    entropy_decay_cb = EntropyDecayCallback(
        start_coef=0.1,
        end_coef=0.005,
        decay_steps=2_000_000
    )
    # 检查点 Callback（每 10k 步保存一次）
    checkpoint_cb = CheckpointCallback(
        save_freq=10_000,                      # 每 10000 env step 保存一次
        save_path="./logs/checkpoints_10_000/",       # 保存目录
        name_prefix="ppo_mario_checkpoint"     # 文件名前缀
    )

    # —— 4. 训练 ——
    model.learn(
        total_timesteps=5_000_000,
        callback=[checkpoint_cb, eval_callback, entropy_decay_cb],
        tb_log_name="PPO-Mario-16env"
    )

    # —— 5. 保存最终模型 ——
    model.save(model_path)

    # —— 6. 测试 ——
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
