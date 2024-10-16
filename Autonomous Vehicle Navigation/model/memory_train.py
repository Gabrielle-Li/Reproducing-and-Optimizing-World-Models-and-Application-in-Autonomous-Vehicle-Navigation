import sys
sys.path.append("..")
from model import memory_dataloader
from model import vision, memory
import tqdm
import torch
import argparse
import warnings
import os
from tqdm import tqdm
import pickle

os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
TORCH_USE_CUDA_DSA = 1

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Trainer for Memory Module')
parser.add_argument('--view_list', default='viewlist.txt', type=str,
                    help='局部训练集列表')
parser.add_argument('--view_val', default='viewval.txt', type=str,
                    help='局部验证集列表')
parser.add_argument('--device', default='0', type=str,
                    help='GPU')
parser.add_argument('--save_path', default='weights', type=str,
                    help='保存路径')
parser.add_argument('--epochs', default=50, type=int, metavar='N',
                    help='训练轮次')
parser.add_argument('--batch-size', default=24, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='初始学习率', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='动量的权值')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--n-latents', default=100, type=str,
                    help='输入维度')
parser.add_argument('--n-actions', default=4, type=str,
                    help='动作维度')
parser.add_argument('--hidden-size', default=50, type=int, metavar='N',
                    help='隐藏层维度')
parser.add_argument('--vision-model-path', default='weights/vision.pth', type=str,
                    help='实际模型位置')

args = parser.parse_args()


def train_memory(args):
    print('构建模型......')
    model_vision = vision.MapModule(args)
    model_vision.load_state_dict(torch.load(args.vision_model_path))
    model_vision = model_vision.cuda(args.device)
    model_vision.eval()

    model_memory = memory.Memory(args)
    model_memory = model_memory.cuda(args.device)

    print('构建模型：√')

    print('构建优化器......')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model_memory.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    print('构建优化器√')

    print('构建数据载入器......')
    memory_loader, _ = memory_dataloader.memory_loader(args)
    memory_val_loader, _ = memory_dataloader.memory_val_loader(args)
    print('构建数据载入器√')

    print('构建损失函数......')
    criterion = torch.nn.L1Loss()
    print('构建数损失函数√')

    print('开始训练：')
    train_loss_sum = 0
    count = 0
    min_val_loss = 999
    for epoch in range(args.start_epoch, args.epochs):

        model_memory.train()

        train_bar = tqdm(memory_loader, file=sys.stdout)
        for i, (img_current, img_future) in enumerate(train_bar):
            img_current = img_current.cuda(args.devide)
            img_future = img_future.cuda(args.devide)
            # 模型计算
            features_current, _ = model_vision(img_current)
            features_future, _ = model_vision(img_future)

            predict_future = model_memory(features_current)

            loss = criterion(predict_future, features_future)
            train_loss_sum += loss
            count += 1
            train_loss_avg = train_loss_sum/count


            # 计算梯度并执行求解器步骤
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_bar.desc = "train epoch[{}/{}] train_loss:{:.3f} ".format(epoch + 1, args.epochs,
                                                                      train_loss_avg, )

        model_memory.eval()
        j = 0
        val_loss_sum = 0
        with torch.no_grad():
            val_bar = tqdm(memory_val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_future = val_data
                current_features, _ = model_vision(val_images.cuda(args.device))
                future_features, _ = model_vision(val_future.cuda(args.device))


                val_loss = criterion(current_features, future_features)
                j += 1
                val_loss_sum += val_loss
                val_loss_avg = val_loss_sum/j
                val_bar.desc = "val epoch[{}/{}] val_loss:{:.3f} ".format(epoch + 1, args.epochs,
                                                                          val_loss_avg, )


        if val_loss_avg < min_val_loss:
            min_val_loss = val_loss_avg
            print("保存第{}轮的参数,val_loss_avg={}.".format(epoch + 1, val_loss_avg))
            model_save_path = "model_memory" + ".pth"
            model_save_path = os.path.join(args.save_path, model_save_path)
            torch.save(model_memory.state_dict(), model_save_path)

    # 保存训练后的最后一个隐藏状态
    last_hidden_state = model_memory.saved_states[-1]
    # 将最后一个隐藏状态保存到文件中
    with open('last_hidden_state.pkl', 'wb') as f:
        pickle.dump(last_hidden_state, f)


if __name__ == '__main__':



    train_memory(args)
