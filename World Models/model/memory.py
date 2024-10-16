import torch
import torch.nn as nn

class Memory(nn.Module):
    def __init__(self, args):
        super(Memory, self).__init__()
        self.lstm = nn.LSTM(args.n_latents + args.n_actions, args.hidden_size, batch_first=True)
        self.fc = nn.Linear(args.hidden_size, args.n_latents)
        self.saved_states = []

    def forward(self, states):
        h, _ = self.lstm(states)
        y = self.fc(h)  # 取序列的最后一个时间步的隐藏状态作为输出
        return y

    def infer(self, states, hidden):
        h, next_hidden = self.lstm(states, hidden)  # return (out, hx, cx)
        y = self.fc(h)  # 取序列的最后一个时间步的隐藏状态作为输出
        return y, next_hidden

    def __call__(self, *args, **kwargs):
        y = self.forward(*args, **kwargs)
        self.saved_states.append(y)
        return y

    def get_saved_states(self):
        return self.saved_states