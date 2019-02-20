import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os
import os.path
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from pdb import set_trace as st

def default_loader(path):
    img = Image.open(path).convert('L')
    w, h = img.size
    if w != h:
        img = img.resize((130, 151))
    else:
        img = img.resize((144, 144))
    return img

def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label = line.strip().split(' ')
            imgList.append((imgPath, int(label)))
    return imgList

def default_transform():
    transform=transforms.Compose([ 
                transforms.RandomCrop(128),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
            ])
    return transform

def testPhase_transform():
    transform=transforms.Compose([ 
                transforms.CenterCrop(128),
                transforms.ToTensor(),
            ])
    return transform


class ImageList(data.Dataset):
    def __init__(self, root, fileList, testPahse=False, transform=default_transform, list_reader=default_list_reader, loader=default_loader):
        self.root      = root
        self.imgList   = list_reader(fileList)
        if testPahse:
            self.transform = testPhase_transform()
        else:
            self.transform = transform()
        self.loader    = loader

    def __getitem__(self, index):
        imgPath, target = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img = self.transform(img)
        return {'image': img, 'target': target}

    def __len__(self):
        return len(self.imgList)

    def name(self):
        return 'ImageListDataset'

    def initialize(self, opt):
        self.opt = opt
        pass


class ImageList_cross_view(data.Dataset):
    def initialize(self, opt):
        self.root      = opt.dataroot
        self.imgList0  = default_list_reader(opt.train_list)
        self.imgList1  = default_list_reader(opt.train_list_view1)
        self.len0 = len(self.imgList0)
        self.len1 = len(self.imgList1)
        if opt.isTrain:
            self.transform = default_transform()
        else:
            self.transform = testPhase_transform()
        self.loader    = default_loader
        pass

    def __getitem__(self, index):
        imgPath, target0 = self.imgList0[index % self.len0]
        img0 = self.loader(os.path.join(self.root, imgPath))
        imgPath, target1 = self.imgList1[index % self.len1]
        img1 = self.loader(os.path.join(self.root, imgPath))

        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return {'image0': img0, 'target0': target0,
                'image1': img1, 'target1': target1}

    def __len__(self):
        return max(self.len0, self.len1)

    def name(self):
        return 'CrossView_ImageListDataset'
