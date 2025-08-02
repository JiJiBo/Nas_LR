import platform
import random
import matplotlib.pyplot as plt
from collections import Counter
import matplotlib

# 设置中文字体，根据平台自动选择合适字体
system_name = platform.system()
if system_name == 'Windows':
    matplotlib.rcParams['font.family'] = 'SimHei'  # 黑体（Windows）
elif system_name == 'Darwin':
    matplotlib.rcParams['font.family'] = 'Arial Unicode MS'  # MacOS 字体
else:
    matplotlib.rcParams['font.family'] = 'DejaVu Sans'  # Linux 默认中文兼容字体

matplotlib.rcParams['axes.unicode_minus'] = False  # 正确显示负号
# 1. 定义状态集
states = ['睡觉', '工作', '吃饭', '玩手机']

# 2. 定义状态转移概率矩阵（字典表示）
# 每个状态对应一个字典，记录其转移到其他状态的概率
transition_matrix = {
    '睡觉': {'睡觉': 0.2, '工作': 0.6, '吃饭': 0.1, '玩手机': 0.1},
    '工作': {'睡觉': 0.1, '工作': 0.1, '吃饭': 0.3, '玩手机': 0.5},
    '吃饭': {'工作': 0.6, '吃饭': 0.2, '玩手机': 0.2, '睡觉': 0.0},
    '玩手机': {'工作': 0.3, '睡觉': 0.3, '吃饭': 0.2, '玩手机': 0.2},
}


# 3. 蒙特卡洛轨迹采样函数
def simulate_day(start_state='睡觉', hours=24):
    state = start_state
    trajectory = [state]
    for _ in range(hours - 1):
        next_state = random.choices(
            population=list(transition_matrix[state].keys()),
            weights=list(transition_matrix[state].values())
        )[0]
        trajectory.append(next_state)
        state = next_state
    return trajectory


# 4. 模拟多次
n_simulations = 1000
all_trajectories = [simulate_day() for _ in range(n_simulations)]

# 5. 分析某个小时的状态分布（比如第9小时）
hour = 9
state_at_hour = [traj[hour] for traj in all_trajectories]
counter = Counter(state_at_hour)

# 6. 可视化结果
plt.bar(counter.keys(), counter.values(), color='skyblue')
plt.title(f"第{hour + 1}小时状态分布 (共{n_simulations}天)")
plt.xlabel("状态")
plt.ylabel("频数")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.show()

# 7. 可选：输出一条完整轨迹
print("示例一天轨迹：", ' → '.join(simulate_day()))
