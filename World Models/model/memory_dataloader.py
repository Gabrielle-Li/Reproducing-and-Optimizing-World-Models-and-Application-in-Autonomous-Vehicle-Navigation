import sys

sys.path.append("..")
from PIL import Image
import torch.utils.data as data
import torch
import torchvision.transforms as transforms


class ImgTrainDataset(data.Dataset):
    def __init__(self, img_file, transform=None):
        self.img_file = img_file  #存储训练图片路径的文本文件
        self.transform = transform
        self.init()
        

    def init(self):
        self.img_path = []
        self.future_path = []
        with open(self.img_file) as f:
            for line in f.readlines():
                data = line.strip().split(' ')
                self.img_path.append(data[0])  # 图片路径
                if len(data) > 1:
                    self.future_path.append(data[1])


    def __getitem__(self, index):  # 类实例被索引时自动调用
        if index < 0 or index >= len(self.img_path):
            print("Index out of range")
        im_name = self.img_path[index]
        future_name = self.future_path[index]
        img = Image.open(im_name)
        img = self.transform(img)

        future = Image.open(future_name)
        future = self.transform(future)

        return img, future

    def __len__(self):
        return len(self.img_path)


def memory_loader(args):  #地图训练集载入器
    img_trans = transforms.Compose([
        transforms.ToTensor()
    ])

    img_dataset = ImgTrainDataset(
        args.view_list,
        transform=img_trans
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        img_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_loader, len(img_dataset)


def memory_val_loader(args):  #地图验证集载入器
    img_trans = transforms.Compose([
        transforms.ToTensor()
    ])

    img_dataset = ImgTrainDataset(
        args.view_val,
        transform=img_trans
    )
    train_sampler = None
    train_loader = torch.utils.data.DataLoader(
        img_dataset,
        shuffle=(train_sampler is None),
        batch_size=args.batch_size,
        pin_memory=True,
        sampler=train_sampler,
        drop_last=(train_sampler is None))

    return train_loader, len(img_dataset)






