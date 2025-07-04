import gym

class SB3CompatWrapper(gym.Wrapper):
    """
    把任何 reset()/step() 返回值兜底成 SB3 VecEnv 期待的格式：
      reset() → (obs, info)
      step()  → (obs, reward, done, info)
    """
    def reset(self, **kwargs):
        result = self.env.reset(**kwargs)
        # 如果是 tuple，就把第一个当 obs，最后一个当 info
        if isinstance(result, tuple):
            obs = result[0]
            info = result[-1] if isinstance(result[-1], dict) else {}
        else:
            obs, info = result, {}
        return obs, info

    def step(self, action):
        result = self.env.step(action)
        if not isinstance(result, tuple):
            raise ValueError(f"Unexpected step return: {result}")
        # Gymnasium API：5 元组 → 合并 terminated+truncated
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
            return obs, reward, terminated, truncated, info
        # 经典 Gym API：4 元组
        elif len(result) == 4:
            return result
        # 其它奇怪情况，兜底取值
        else:
            obs = result[0]
            reward = result[1] if len(result) > 1 else 0.0
            done   = result[2] if len(result) > 2 else False
            info   = result[-1] if isinstance(result[-1], dict) else {}
            return obs, reward, done,False, info
