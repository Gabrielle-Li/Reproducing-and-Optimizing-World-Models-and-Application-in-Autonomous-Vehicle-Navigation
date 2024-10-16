import sys

sys.path.append("..")
from model import vision_dataloader
from model import vision
import tqdm
import torch
import argparse
import warnings
import os
from tqdm import tqdm

os.environ['CUDA_LAUNCH_BLOCKING'] = str(1)
TORCH_USE_CUDA_DSA = 1

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description='Trainer for Vision Module')
parser.add_argument('--map_list', default='list/map_list.txt', type=str,
                    help='全局训练集列表')
parser.add_argument('--map_val', default='list/map_val_list.txt', type=str,
                    help='全局验证集列表')
parser.add_argument('--view_list', default='list/vision_list.txt', type=str,
                    help='局部训练集列表')
parser.add_argument('--view_val', default='list/vision_val_list.txt', type=str,
                    help='局部验证集列表')
parser.add_argument('--device', default=0, type=int, metavar='N',
                    help='GPU')
parser.add_argument('--save_path', default='weights', type=str,
                    help='保存路径')
parser.add_argument('--epochs', default=30, type=int, metavar='N',
                    help='训练轮次')
parser.add_argument('--batch-size', default=2, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--view-embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--map-embedding-size', default=512, type=int,
                    help='The embedding feature size')
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='初始学习率', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='动量的权值')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
# parser.add_argument('--lr-drop-epoch', default=[30, 60], type=int, nargs='+',
#                     help='The learning rate drop epoch')
# parser.add_argument('--lr-drop-ratio', default=0.1, type=float,
#                     help='The learning rate drop ratio')

args = parser.parse_args()


def train_map(args):
    print('构建模型......')
    model = vision.MapModule(args)
    model = model.cuda(args.device)
    print('构建模型：√')

    print('构建优化器......')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    print('构建优化器√')

    print('构建数据载入器......')
    map_loader, _ = vision_dataloader.map_loader(args)
    map_val_loader, _ = vision_dataloader.map_val_loader(args)
    print('构建数据载入器√')

    print('构建损失函数......')
    # criterion = vision.ssim
    print('构建数损失函数√')

    print('开始训练：')
    min_val_loss = 999
    for epoch in range(0, args.epochs):
        train_loss_sum = 0
        count = 0
        
        model.train()
        train_bar = tqdm(map_loader, file=sys.stdout)
        for i, (img) in enumerate(train_bar):
            img = img.cuda(args.device)
            # 模型计算
            features, fake_img, ssim_loss = model(img)

            loss = torch.tensor(ssim_loss, requires_grad=True)
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
            val_bar = tqdm(map_val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images = val_data
                val_features, val_fake, val_loss = model(val_images.cuda(args.device))
                
                j += 1
                val_loss_sum += val_loss
                val_loss_avg = val_loss_sum/j
                val_bar.desc = "val epoch[{}/{}] val_loss:{:.3f} ".format(epoch + 1, args.epochs,
                                                                          val_loss_avg, )


        if val_loss_avg < min_val_loss:
            min_val_loss = val_loss_avg
            
            print("保存第{}轮的参数,val_loss_avg={}.".format(epoch + 1, val_loss_avg))
            model_save_path = "model_map" + ".pth"
            model_save_path = os.path.join(args.save_path, model_save_path)
            torch.save(model.state_dict(), model_save_path)

def train_vision(args):
    print('构建模型......')
    model = vision.VisionModule(args)
    model = model.cuda(args.device)
    print('构建模型：√')

    print('构建优化器......')
    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay)
    print('构建优化器√')

    print('构建数据载入器......')
    view_loader, _ = vision_dataloader.view_loader(args)
    view_val_loader, _ = vision_dataloader.view_val_loader(args)
    print('构建数据载入器√')

    print('构建损失函数......')
    # criterion = vision.ssim()
    print('构建数损失函数√')

    print('开始训练：')
    min_val_loss = 999
    for epoch in range(0, args.epochs):
        train_loss_sum = 0
        count = 0
        
        model.train()

        train_bar = tqdm(view_loader, file=sys.stdout)
        for i, (img) in enumerate(train_bar):
            img = img.cuda(args.device)
            # 模型计算
            features, fake_img, ssim_loss = model(img)
            loss = torch.tensor(ssim_loss, requires_grad=True)

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
            val_bar = tqdm(view_val_loader, file=sys.stdout)
            for val_data in val_bar:
                val_images = val_data
                val_features, val_fake, val_loss= model(val_images.cuda(args.device))
                j += 1
                val_loss_sum += val_loss
                val_loss_avg = val_loss_sum/j
                val_bar.desc = "val epoch[{}/{}] val_loss:{:.3f} ".format(epoch + 1, args.epochs,
                                                                          val_loss_avg, )

        
        if val_loss_avg < min_val_loss:
            min_val_loss = val_loss_avg
            print("保存第{}轮的参数,val_loss_avg={}.".format(epoch + 1, val_loss_avg))
            model_save_path = "model_vision" + ".pth"
            model_save_path = os.path.join(args.save_path, model_save_path)
            torch.save(model.state_dict(), model_save_path)

if __name__ == '__main__':


    #训练地图特征提取模型
    # train_map(args)
    # 训练视觉特征提取模型
    train_vision(args)
