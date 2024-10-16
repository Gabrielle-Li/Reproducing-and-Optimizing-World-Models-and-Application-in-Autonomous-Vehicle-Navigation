import torch.nn as nn


class Memory(nn.Module):
    def __init__(self,args):
        super(Memory, self).__init__()
        self.lstm = nn.LSTM(args.n_latents + args.n_actions, args.hidden_size, batch_first=True)
        # target --> next latent (vision)
        self.fc = nn.Linear(args.hidden_size, args.n_latents)

    def forward(self, states):
        h, _ = self.lstm(states)
        y = self.fc(h[:, -1, :])  # 取序列的最后一个时间步的隐藏状态作为输出
        return y

    def infer(self, states, hidden):
        h, next_hidden = self.lstm(states, hidden)  # return (out, hx, cx)
        y = self.fc(h[:, -1, :])  # 取序列的最后一个时间步的隐藏状态作为输出
        return y, next_hidden

    def __call__(self, *args, **kwargs):
        # 调用forward方法，并保存最后一个隐藏状态
        y = self.forward(*args, **kwargs)
        self.saved_states.append(y)
        return y

    def get_saved_states(self):
        return self.saved_states


