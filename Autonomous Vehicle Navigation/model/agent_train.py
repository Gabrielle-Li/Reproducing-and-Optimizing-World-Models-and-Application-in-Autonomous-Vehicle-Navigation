import os
import sys
sys.path.append("..")
import warnings
import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch
from model import agent, vision, memory
import Game
import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
TORCH_USE_CUDA_DSA = 1
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Trainer for PPO Module')
parser.add_argument('--device', default='0', type=str,
                    help='GPU')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='训练轮次')
parser.add_argument('--batch-size', default=24, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--actor-lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='初始学习率', dest='lr')
parser.add_argument('--criticr-lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='初始学习率', dest='lr')
parser.add_argument('--gamma', default=0.9, type=float, metavar='M',
                    help='折扣因子')
parser.add_argument('--n_hiddens', default=900, type=int, metavar='M',
                    help='折扣因子')
parser.add_argument('--embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--map-model-path', default='weights/map.pth', type=str,
                    help='地图特征提取模型位置')
parser.add_argument('--vision-model-path', default='weights/vision.pth', type=str,
                    help='视觉特征提取模型位置')
parser.add_argument('--memory-model-path', default='weights/memory.pth', type=str,
                    help='记忆模型位置')

args = parser.parse_args()

return_list = []  # 保存每个回合的return

# ----------------------------------------- #
# 环境加载
# ----------------------------------------- #

env = Game.Car_Environment
n_states = env.observation_space.shape[0] * env.observation_space.shape[1]  # 状态数
n_actions = env.action_space.n  # 动作数

# ----------------------------------------- #
# 模型构建
# ----------------------------------------- #

agent = agent.PPO(n_states=n_states,  # 状态数
                  n_hiddens=args.n_hiddens,  # 隐含层数
                  n_actions=n_actions,  # 动作数
                  actor_lr=args.actor_lr,  # 策略网络学习率
                  critic_lr=args.critic_lr,  # 价值网络学习率
                  lmbda=0.95,  # 优势函数的缩放因子
                  epochs=10,  # 一组序列训练的轮次
                  eps=0.2,  # PPO中截断范围的参数
                  gamma=args.gamma,  # 折扣因子
                  device=args.device
                  )

# ----------------------------------------- #
# 训练--回合更新 on_policy
# ----------------------------------------- #
def train(args):
    print('加载预训练模型......')
    model_map = vision.MapModule(args)
    model_map.load_state_dict(torch.load(args.map_model_path))
    model_map = model_map.cuda(args.device)
    model_map.eval()

    model_vision = vision.VisionModule(args)
    model_vision.load_state_dict(torch.load(args.vision_model_path))
    model_vision = model_vision.cuda(args.device)
    model_vision.eval()

    model_memory = memory.Memory(args)
    model_memory.load_state_dict(torch.load(args.memory_model_path))
    model_memory = model_memory.cuda(args.device)
    model_memory.eval()
    print('加载预训练构建模型：√')

    for i in range(args.epochs):

        state = env.reset()  # 环境重置
        done = False  # 任务完成的标记
        epoch_return = 0  # 累计每回合的reward

        # 构造数据集，保存每个回合的状态数据
        transition_dict = {
            'states': [],
            'actions': [],
            'next_states': [],
            'rewards': [],
            'dones': [],
        }
        with open('last_hidden_state.pkl', 'rb') as f:
            next_hidden = pickle.load(f)

        while not done:
            hidden = next_hidden
            action = agent.take_action(state)  # 动作选择
            action = torch.tensor(action, dtype=torch.float).view(1, -1).to(args.device)

            global_map, vision_map, reward, done, _ = env.play(action)
            map_features, _ = model_map(global_map)
            states, _ = model_vision(vision_map)

            vision_action = torch.cat([states, action], dim=-1)  #
            vision_action = vision_action.view(1, 1, -1)
            _, next_hidden = model_memory.infer(vision_action, hidden)
            next_state = torch.cat([states, next_hidden[0].squeeze(0)], dim=1)

            # 保存每个时刻的状态\动作\...
            transition_dict['states'].append(state)
            transition_dict['actions'].append(action)
            transition_dict['next_states'].append(next_state)
            transition_dict['rewards'].append(reward)
            transition_dict['dones'].append(done)
            # 更新状态
            state = next_state
            # 累计回合奖励
            epoch_return += reward

        # 保存每个回合的return
        return_list.append(epoch_return)
        # 模型训练
        agent.learn(transition_dict)

        # 打印回合信息
        print(f'iter:{i}, return:{np.mean(return_list[-10:])}')

    # -------------------------------------- #
    # 绘图
    # -------------------------------------- #

    plt.plot(return_list)
    plt.title('return')
    plt.show()


if __name__ == '__main__':
    train()
