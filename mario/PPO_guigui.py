import os
import time

from gym.wrappers import GrayScaleObservation
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import MaxAndSkipEnv
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
    VecTransposeImage, SubprocVecEnv,
)

from mario.guigui_env import ContraEnv


class CustomCheckpointCallback(BaseCallback):
    def __init__(self, save_freq: int, save_path: str, verbose=0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = f"{self.save_path}/model_step_{self.n_calls}"
            self.model.save(model_path)
            if self.verbose > 0:
                print(f"模型已保存到 {model_path}")
        return True


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


def make_contra_env():
    env = ContraEnv()
    return Monitor(env)


def train_contra():
    # —— 1. 环境准备 ——
    train_env = SubprocVecEnv([make_contra_env for _ in range(24)])
    train_env = VecMonitor(train_env)

    train_env = VecFrameStack(train_env, n_stack=4)
    train_env = VecTransposeImage(train_env)

    eval_env = DummyVecEnv([make_contra_env])
    eval_env = VecMonitor(eval_env)
    eval_env = VecFrameStack(eval_env, n_stack=4)
    eval_env = VecTransposeImage(eval_env)
    time_str = time.strftime("%Y%m%d-%H%M%S")
    tensorboard_log = f"/root/tf-logs/time_{time_str}"
    # —— 2. 模型加载或新建 ——
    model_path = "checkpoints/model_step_30000.zip"
    if os.path.exists(model_path):
        print("will load",model_path)
        model = PPO.load(
            model_path,
            env=train_env,
            batch_size=2048,
            tensorboard_log=tensorboard_log
        )
    else:
        model = PPO(
            policy="CnnPolicy",
            env=train_env,
            verbose=1,
            tensorboard_log=tensorboard_log,

            # —— 采样与并行相关 ——
            n_steps=2048,  # 每个环境 rollout 步数，显存足够建议取 2048
            batch_size=4096,  # 越大越平滑，推荐 2048~8192（显存足够时取4096）
            n_epochs=8,  # 可适当提高epoch，加快收敛，建议 4~8
            learning_rate=5e-4,  # 学习率可以略微提高，加速前期收敛，5e-4 或 3e-4
            gamma=0.99,  # 折扣因子，经典设定
            gae_lambda=0.95,  # GAE参数，经典设定

            # —— PPO核心参数 ——
            clip_range=0.1,  # 初期 0.1，可随epoch线性衰减至0.05
            ent_coef=0.02,  # 适当增加探索（0.01~0.05），2D射击类建议 0.02
            vf_coef=0.5,  # 价值损失系数，默认
            max_grad_norm=0.5,  # 梯度裁剪，0.5 保守稳定
            target_kl=0.015,  # KL目标，0.01~0.02，更快响应策略崩溃

            # —— 多环境并行 ——
            seed=42,  # 固定随机种子
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

    rollout_save_freq = 100_000
    checkpoint_cb = CheckpointCallback(
        save_freq=rollout_save_freq,
        save_path="../autodl-tmp/checkpoints_3/",  # 保存目录
        name_prefix="ppo_contra_checkpoint"  # 文件名前缀
    )

    custom_checkpoint_cb = CustomCheckpointCallback(
        save_freq=10000,
        save_path="./checkpoints_dir",  # 保存目录
        verbose=1
    )

    # —— 4. 训练 ——
    model.learn(
        total_timesteps=4_000_000,
        callback=[checkpoint_cb, eval_callback, custom_checkpoint_cb],
        tb_log_name="PPO-contra-16env",
        progress_bar=True
    )

    # —— 5. 保存最终模型 ——
    model.save(model_path)

    # —— 6. 测试 ——
    test_env = ContraEnv()
    obs = test_env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = test_env.step(action)
        test_env.render()
    print("测试结束，最终得分：", info.get("score"))


if __name__ == "__main__":
    train_contra()
