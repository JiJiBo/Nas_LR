import random, matplotlib.pyplot as plt, matplotlib, platform
from collections import Counter, OrderedDict

# 中文显示适配
matplotlib.rcParams['axes.unicode_minus'] = False
system = platform.system()
if system == 'Windows':
    matplotlib.rcParams['font.family'] = 'SimHei'
elif system == 'Darwin':
    matplotlib.rcParams['font.family'] = 'Arial Unicode MS'
else:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'

# 状态与转移概率
states = ['空', '较空', '正常', '拥挤', '爆满']
transitions = {
    '空': {'空': 0.4, '较空': 0.4, '正常': 0.1, '拥挤': 0.1, '爆满': 0.0},
    '较空': {'空': 0.1, '较空': 0.3, '正常': 0.4, '拥挤': 0.2, '爆满': 0.0},
    '正常': {'较空': 0.1, '正常': 0.3, '拥挤': 0.4, '爆满': 0.2, '空': 0.0},
    '拥挤': {'正常': 0.1, '拥挤': 0.4, '爆满': 0.4, '较空': 0.1, '空': 0.0},
    '爆满': {'拥挤': 0.2, '爆满': 0.6, '正常': 0.2, '较空': 0.0, '空': 0.0},
}


# 模拟函数（一天内每10分钟采样）
def simulate_subway(start='较空', intervals=36):
    traj = [start]
    for _ in range(intervals - 1):
        next_state = random.choices(
            list(transitions[start].keys()),
            weights=list(transitions[start].values())
        )[0]
        traj.append(next_state)
        start = next_state
    return traj


# 蒙特卡洛：模拟 1000 辆车
all = [simulate_subway() for _ in range(1000)]
t_index = 18  # 180分钟后
counter = Counter(traj[t_index] for traj in all)
ordered_counter = OrderedDict((state, counter.get(state, 0)) for state in states)
# 可视化
plt.bar(ordered_counter.keys(), ordered_counter.values(), color='skyblue')
plt.title(f'第{(t_index + 1) * 10}分钟 地铁状态分布')
plt.xlabel('状态')
plt.ylabel('频数')
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 示例轨迹打印
print('示例轨迹：', ' → '.join(simulate_subway()))
