from model import mamba
from model import memory_dataloader,vision
from tqdm import tqdm
import torch
import sys
import os
import argparse
import warnings

os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
TORCH_USE_CUDA_DSA = 1

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Trainer for Memory Module')
parser.add_argument('--view_list', default='list/vision_memory.txt', type=str,
                    help='局部训练集列表')
parser.add_argument('--view_val', default='list/vision_memory_val.txt', type=str,
                    help='局部验证集列表')
parser.add_argument('--device', default=0, type=int, metavar='N',
                    help='GPU')
parser.add_argument('--save_path', default='weights', type=str,
                    help='保存路径')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='训练轮次')
parser.add_argument('--batch-size', default=24, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--view_embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='初始学习率', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='动量的权值')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('--vision-model-path', default='weights/model_vision.pth', type=str,
                    help='实际模型位置')

args = parser.parse_args()


def train_mamba(args):
    print('构建模型......')
    model_vision = vision.VisionModule(args)
    model_vision.load_state_dict(torch.load(args.vision_model_path))
    model_vision = model_vision.cuda(args.device)
    model_vision.eval()

    model = mamba.Mamba(mamba.ModelArgs())
    model.cuda(args.device)
    model.train()

    print('构建优化器......')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
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
    for epoch in range(0, args.epochs):

        model.train()

        train_bar = tqdm(memory_loader, file=sys.stdout)
        for i, (img_current, img_future) in enumerate(train_bar):
            img_current = img_current.cuda(args.device)
            img_future = img_future.cuda(args.device)
            # 模型计算
            features_current, _, _ = model_vision(img_current)
            features_future, _, _ = model_vision(img_future)


            features = features_current.unsqueeze(1)
            

            predict_future = model(features)
            

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

        model.eval()
        j = 0
        val_loss_sum = 0
        val_loss_avg = 0
        with torch.no_grad():
            val_bar = tqdm(memory_val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images, val_future = val_data
                current_features, _, _ = model_vision(val_images.cuda(args.device))
                future_features, _ , _v= model_vision(val_future.cuda(args.device))
                features = current_features.unsqueeze(0)
                predict_future = model(features)

                val_loss = criterion(predict_future, future_features)
                j += 1
                val_loss_sum += val_loss
                val_loss_avg = val_loss_sum/j
                val_bar.desc = "val epoch[{}/{}] val_loss:{:.3f} ".format(epoch + 1, args.epochs,
                                                                          val_loss_avg, )


        if val_loss_avg < min_val_loss:
            min_val_loss = val_loss_avg
            print("保存第{}轮的参数,val_loss_avg={}.".format(epoch + 1, val_loss_avg))
            model_save_path = "model_mamba" + ".pth"
            model_save_path = os.path.join(args.save_path, model_save_path)
            torch.save(model.state_dict(), model_save_path)



if __name__ == '__main__':



    train_mamba(args)