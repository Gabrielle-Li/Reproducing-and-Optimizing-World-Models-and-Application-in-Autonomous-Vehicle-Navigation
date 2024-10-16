import gym
from PIL import Image
import numpy as np
# 创建环境
env = gym.make('CarRacing-v2')

# 初始化环境
env.reset()

# 游戏循环
for i in range(5000):  # 运行1000步
    env.render()  # 渲染环境
    action = env.action_space.sample()  # 随机选择一个动作
    print(action)
    obs, reward, done, _, _ = env.step(action)  # 执行动作
    image = Image.fromarray(obs)  # 将 numpy 数组转换为 PIL 图像
    image.save(f'data/val/train_image{i}.jpg')  # 保存为 JPG 文件

    if done:
        env.reset()  # 如果游戏结束，重新开始

env.close()  # 关闭环境
