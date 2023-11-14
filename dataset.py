import torchvision.datasets as datasets
import torchvision.transforms as T
from torch.utils.data import Dataset
import torch
import random
import numpy as np


def create_circular_mask(h, w, center=None, radius=None):
    if center is None:
        center = (int(w/2), int(h/2))
    if radius is None:
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y - center[1])**2)

    mask = dist_from_center <= radius
    mask = torch.from_numpy(mask)
    mask = mask.unsqueeze(0)  # 添加通道维度
    mask = mask.expand(3, -1, -1)  # 扩展遮罩以匹配图像的形状
    return mask

class MyDataset(Dataset):
    def __init__(self, data_dir, num_cls = 180, transform=None):
        super().__init__()
        self.num_cls = num_cls
        self.data_dir = data_dir
        self.transform = transform
        self.dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)
        self.pr = T.Compose([
                        T.ToTensor(),
                        T.RandomResizedCrop(224, scale=(0.4, 1.0), ratio=(0.8, 1.0)),
                    ])
        self.norm = T.Normalize(
                            mean = [0.485, 0.456, 0.406],
                            std = [0.229, 0.224, 0.225]
                        )
        self.mask = create_circular_mask(224, 224, radius=112)
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        l = random.random()
        img, _ = self.dataset[idx]
        img = self.pr(img)
        img = T.functional.rotate(img, int(l*360))
        img = torch.where(self.mask, img, torch.ones_like(img))
        img = self.norm(img)
        return img, int(l* self.num_cls)
    
if  __name__ == "__main__":
    train = MyDataset('./data')
    print(len(train))
    for i in range(10):
        img, label = train[i]
        new_img = T.ToPILImage()(img)
        new_img.show()
        print(label)