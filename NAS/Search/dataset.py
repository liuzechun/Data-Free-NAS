''' This code is modified from
Single Path One-Shot Neural Architecture Search with Uniform Sampling
(https://arxiv.org/abs/1904.00420) '''

import os
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
import cv2
import tarfile
import PIL
from PIL import Image
import tqdm
from utils import *

class OpencvResize(object):

    def __init__(self, size=256):
        self.size = size

    def __call__(self, img):
        assert isinstance(img, PIL.Image.Image)
        img = np.asarray(img) # (H,W,3) RGB
        img = img[:,:, ::-1] # 2 BGR
        img = np.ascontiguousarray(img)
        H, W, _ = img.shape
        target_size = (int(self.size/H * W + 0.5), self.size) if H < W else (self.size, int(self.size/W * H + 0.5))
        img = cv2.resize(img, target_size, interpolation=cv2.INTER_LINEAR)
        img = img[:,:, ::-1] # 2 RGB
        img = np.ascontiguousarray(img)
        img = Image.fromarray(img)
        return img

class ToBGRTensor(object):

    def __call__(self, img):
        assert isinstance(img, (np.ndarray, PIL.Image.Image))
        if isinstance(img, PIL.Image.Image):
            img = np.asarray(img)
        img = img[:,:, ::-1] # 2 BGR
        img = np.transpose(img, [2, 0, 1]) # 2 (3, H, W)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float()
        return img

class DataIterator(object):

    def __init__(self, dataloader):
        self.dataloader = dataloader
        self.iterator = enumerate(self.dataloader)

    def next(self):
        try:
            _, data = next(self.iterator)
        except Exception:
            self.iterator = enumerate(self.dataloader)
            _, data = next(self.iterator)
        return data[0], data[1]

train_dir = '/home/zechunliu/experiments/dataset/syn_data/split_for_search/train/train'
val_dir = '/home/zechunliu/experiments/dataset/syn_data/split_for_search/train/val'

assert os.path.exists(train_dir)
assert os.path.exists(val_dir)

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

# data augmentation
crop_scale = 0.08
lighting_param = 0.1
train_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(crop_scale, 1.0)),
    Lighting(lighting_param),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize])

train_dataset = datasets.ImageFolder(
    train_dir,
    transform=train_transforms)


# load validation data
valid_dataset = datasets.ImageFolder(
    val_dir,
    transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
)

def get_train_dataprovider(batch_size, *, num_workers, use_gpu):
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=use_gpu)
    train_dataprovider = DataIterator(train_loader)
    return train_dataprovider

def get_val_dataprovider(batch_size, *, num_workers, use_gpu):
    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=use_gpu
    )
    val_dataprovider = DataIterator(val_loader)
    return val_dataprovider


def main():
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)
    np.random.seed(0)


    print(len(train_dataset))
    print(train_dataset[np.random.randint(len(train_dataset))])


    print(len(valid_dataset))
    print(valid_dataset[np.random.randint(len(valid_dataset))])

    use_gpu = False
    train_batch_size = 128
    valid_batch_size = 200

    train_dataprovider = get_train_dataprovider(train_batch_size, use_gpu=use_gpu, num_workers=3)
    val_dataprovider = get_val_dataprovider(valid_batch_size, use_gpu=use_gpu, num_workers=2)

    train_data = train_dataprovider.next()
    val_data = val_dataprovider.next()

    print(train_data[0].mean().item())
    print(val_data[0].mean().item())

if __name__ == '__main__':
    main()
