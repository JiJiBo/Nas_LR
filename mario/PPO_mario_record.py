#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import platform
import subprocess
import time

import cv2
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import (
    DummyVecEnv,
    VecTransposeImage,
    VecFrameStack,
    VecMonitor,
)

from mario.make_env import SuperMarioBrosEnv

def make_mario_env():
    return SuperMarioBrosEnv()

def build_ffmpeg_cmd(output_path: str):
    """
    根据当前操作系统，构造 FFmpeg 录制命令：
    - Linux: x11grab + PulseAudio
    - macOS: AVFoundation
    - Windows: gdigrab + dshow
    """
    system = platform.system()
    # 录制参数：分辨率、帧率，可按需修改
    width, height, fps = 800, 600, 30

    if system == "Linux":
        return [
            "ffmpeg",
            "-y",
            "-f", "x11grab",
            "-video_size", f"{width}x{height}",
            "-framerate", str(fps),
            "-i", os.environ.get("DISPLAY", ":0.0"),
            "-f", "pulse",
            "-ac", "2",
            "-i", "default",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            output_path,
        ]
    elif system == "Darwin":
        # 先用 `ffmpeg -f avfoundation -list_devices true -i ""` 查索引
        screen_index = "1"    # 根据你的 `ffmpeg -list_devices` 结果调整
        audio_index = "0"
        return [
            "ffmpeg",
            "-y",
            "-f", "avfoundation",
            "-framerate", str(fps),
            "-video_size", f"{width}x{height}",
            "-i", f"{screen_index}:{audio_index}",
            "-c:v", "libx264", "-preset", "ultrafast",
            "-c:a", "aac",
            output_path,
        ]
    elif system == "Windows":
        # 先用 `ffmpeg -list_devices true -f dshow -i dummy` 查设备名称
        audio_device = "Stereo Mix (Realtek High Definition Audio)"
        return [
            "ffmpeg",
            "-y",
            "-f", "gdigrab",
            "-framerate", str(fps),
            "-i", "desktop",
            "-f", "dshow",
            "-i", f"audio={audio_device}",
            "-c:v", "libx264", "-preset", "superfast",
            "-c:a", "aac",
            output_path,
        ]
    else:
        raise RuntimeError(f"Unsupported OS: {system}")

def main():
    # 录制文件路径
    output_video = "mario_playback_with_audio.mp4"

    # 启动 FFmpeg 子进程
    ffmpeg_cmd = build_ffmpeg_cmd(output_video)
    print("启动录制：", " ".join(ffmpeg_cmd))
    ffmpeg_proc = subprocess.Popen(ffmpeg_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    try:
        # 1. 包装环境（与训练时一致）
        env = DummyVecEnv([make_mario_env])
        env = VecMonitor(env)
        env = VecFrameStack(env, n_stack=4)
        env = VecTransposeImage(env)

        # 2. 加载模型，并“挂钩” custom_objects
        model_path = "../moldes/best_v0_model.zip"
        custom_objects = {
            "observation_space": env.observation_space,
            "action_space":      env.action_space,
            "lr_schedule":       lambda _: 3e-4,
            "clip_range":        lambda _: 0.2,
        }
        model = PPO.load(model_path, env=env, custom_objects=custom_objects)

        # 3. 评估并渲染
        obs = env.reset()
        done = [False]
        frames = []
        while not done[0]:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, done, infos = env.step(action)

            # 渲染 RGB 数组
            rgb_frame = env.venv.envs[0].render(mode="rgb_array")
            frames.append(rgb_frame.copy())

            # 同时在 OpenCV 窗口中展示（可选）
            bgr = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow("Mario Playback", bgr)
            if cv2.waitKey(int(1000 / 30)) & 0xFF == ord('q'):
                break

        print(f"共渲染帧数: {len(frames)}")
        print("最终得分：", infos[0].get("score"))

    finally:
        # 停止录制
        print("结束录制，正在终止 FFmpeg...")
        ffmpeg_proc.terminate()
        ffmpeg_proc.wait()
        cv2.destroyAllWindows()

        # 可选：把帧保存为 GIF 备份
        gif_path = "episode_backup.gif"
        print("同时保存 GIF 备份：", gif_path)
        imageio.mimsave(gif_path, frames, fps=30)

    print("所有工作完成，输出文件：", output_video, "和", gif_path)

if __name__ == "__main__":
    main()
