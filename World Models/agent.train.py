import os
import sys

import cv2
import gym

sys.path.append("../..")
import warnings
import argparse
from torch.utils.data import DataLoader
import torch
from model import agent, mamba, vision
import numpy as np
from PIL import Image
from torchvision import transforms
from time import sleep
import random

os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
TORCH_USE_CUDA_DSA = 1
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Trainer for agent')
parser.add_argument('--device', default=0, type=int, metavar='N',
                    help='GPU')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='训练轮次')
parser.add_argument('--batch-size', default=1, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr', default=0.0001, type=float,
                    metavar='LR', help='初始学习率')
parser.add_argument('--actor-lr', default=0.0001, type=float,
                    metavar='LR', help='初始学习率')
parser.add_argument('--critic-lr', default=0.0001, type=float,
                    metavar='LR', help='初始学习率')
parser.add_argument('--gamma', default=0.9, type=float, metavar='M',
                    help='折扣因子')
parser.add_argument('--n_hiddens', default=512, type=int, metavar='M',
                    help='隐藏层尺寸')
parser.add_argument('--n-latents', default=512, type=str,
                    help='输入维度')
parser.add_argument('--n-actions', default=0, type=str,
                    help='动作维度')
parser.add_argument('--hidden-size', default=50, type=int, metavar='N',
                    help='隐藏层维度')
parser.add_argument('--view_embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--vision-model-path', default=r'weights/model_vision.pth', type=str,
                    help='视觉特征提取模型位置')
parser.add_argument('--memory-model-path', default=r'weights/model_mamba.pth', type=str,
                    help='记忆模型位置')

args = parser.parse_args()

return_list = []  # 保存每个回合的return

MAX_R = 1.

transform = transforms.Compose([

    transforms.ToTensor()
])


def set_seed(seed, env=None):
    if env is not None:
        env.seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def obs2tensor(obs):
    binary_road = obs2feature(obs)  # (10, 10)
    s = binary_road.flatten()
    s = torch.tensor(s.reshape([1, -1]), dtype=torch.float)
    obs = np.ascontiguousarray(obs)
    # obs = torch.tensor(obs, dtype=torch.float)
    obs = transform(obs).unsqueeze(0)
    return obs.to(args.device), s.to(args.device)


def obs2feature(s):
    upper_field = s[:84, 6:90]  # we crop side of screen as they carry little information
    img = cv2.cvtColor(upper_field, cv2.COLOR_RGB2GRAY)
    upper_field_bw = cv2.threshold(img, 120, 255, cv2.THRESH_BINARY)[1]
    upper_field_bw = cv2.resize(upper_field_bw, (10, 10), interpolation=cv2.INTER_NEAREST)  # re scaled to 7x7 pixels
    upper_field_bw = upper_field_bw.astype(np.float32) / 255
    return upper_field_bw


def train(agent, vae, memory, args, seed=9):
    file_path = 'output.txt'
    set_seed(seed)
    env = gym.make('CarRacing-v2')
    env.verbose = 0

    scores = [-100, ]
    running_means = []
    step = 0
    update_term = 100
    max_ep = 1000

    for ep in range(max_ep):
        obs = env.reset()
        score = 0.
        i = 0
        # next_hidden = [torch.zeros(1, 1, hp.rnn_hunits).to(device) for _ in range(2)]
        for _ in range(5):
            # env.render()
            agent.eval()
            next_obs, reward, _, done, _ = env.step(agent.possible_actions[-2])
            score += reward
        next_obs, next_s = obs2tensor(next_obs)
        # print(next_obs.shape)
        with torch.no_grad():
            next_latent_features, _, _ = vae(next_obs)

        while True:

            # env.render()
            obs = next_obs
            # s = next_s
            # hidden = next_hidden
            latent_features = next_latent_features
            mamba_current = memory(latent_features.unsqueeze(0))

            # state = torch.cat([latent_features, hidden[0].squeeze(0)], dim=1)
            state = torch.cat([latent_features, mamba_current.squeeze(0)], dim=1)

            agent.eval()

            action, p = agent.select_action(state)  # nparray, tensor

            next_obs, reward, _, done, _ = env.step(action.reshape([-1]))

            with torch.no_grad():
                next_obs, next_s = obs2tensor(next_obs)
                next_latent_features, _, _ = vae(next_obs)

            mamba_future = memory(next_latent_features.unsqueeze(0))

            # MDN-RNN about time t+1
            # with torch.no_grad():
            #     action = torch.tensor(action, dtype=torch.float).view(1, -1).to(args.device)
            #     vision_action = torch.cat([next_latent_features, action], dim=-1)  #
            #     vision_action = vision_action.view(1, 1, -1)
            #     _, _, _, next_hidden = rnn.infer(vision_action, hidden)  #

            # next_state = torch.cat([next_latent_features, next_hidden[0].squeeze(0)], dim=1)
            next_state = torch.cat([next_latent_features,mamba_future.squeeze(0)], dim=1)

            # Scores
            score += reward

            if done:

                reward_tensor = torch.tensor([reward / MAX_R], dtype=torch.float).to(args.device)
                agent.replay.push(state.data, p, reward_tensor, next_state.data)

                running_mean = np.mean(scores[-30:])
                # print('PID: {}, Ep: {}, Replays: {}, Running Mean: {:.2f}, Score: {:.2f}'.format(pid, ep,
                #                                                                                  len(agent.replay),
                #                                                                                  running_mean, score))
                print('Ep: {}, Replays: {}, Running Mean: {:.2f}, Score: {:.2f}'.format(ep,
                                                                                        len(agent.replay),
                                                                                        running_mean, score))
                content = 'Ep: {}, Replays: {}, Running Mean: {:.2f}, Score: {:.2f}\n'.format(ep, len(agent.replay), running_mean, score)
                with open(file_path, 'a+') as file:
                    file.write(content)

                scores.append(score)
                running_means.append(running_mean)

                agent.train()

                agent.update()

                break
            else:
                reward_tensor = torch.tensor([reward / MAX_R], dtype=torch.float).to(args.device)
                agent.replay.push(state.data, p, reward_tensor, next_state.data)

            if len(agent.replay) == update_term:
                agent.train()
                agent.update()

            i += 1
            step += 1
    #         agent.update()

    pdict = {
        'agent': agent,
        'scores': scores,
        'avgs': running_means,
        'step': step,
        'n_episodes': max_ep,
        'seed': seed,
        'update_term': update_term,
    }
    env.close()
    return pdict


if __name__ == '__main__':
    print('加载预训练模型......')

    model_vision = vision.VisionModule(args)
    model_vision.load_state_dict(torch.load(args.vision_model_path))
    model_vision = model_vision.cuda(args.device)
    model_vision.eval()

    model_memory = mamba.Mamba(mamba.ModelArgs())
    model_memory.load_state_dict(torch.load(args.memory_model_path))
    model_memory = model_memory.cuda(args.device)
    model_memory.eval()

    state_dims = 512+512

    A2C = agent.AAC(input_dims=state_dims, hidden_dims=512, lr=args.lr).cuda(args.device)

    print('加载预训练模型：√')

    train(A2C, model_vision, model_memory, args)
