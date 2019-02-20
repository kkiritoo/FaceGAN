import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image


class SingleDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_A = os.path.join(opt.dataroot)

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)

        self.transform = get_transform(opt)

        self.A_nc = opt.input_nc

        if opt.YCbCr:
            self.YCbCr = True
        else:
            self.YCbCr = False

    def __getitem__(self, index):
        A_path = self.A_paths[index]
        if self.A_nc==1:
            A_img = Image.open(A_path).convert('L')
        else:
            if self.YCbCr:
                A_img = Image.open(A_path).convert('YCbCr')
            else:
                A_img = Image.open(A_path).convert('RGB')

        A_img = self.transform(A_img)

        return {'A': A_img, 'A_paths': A_path}

    def __len__(self):
        return len(self.A_paths)

    def name(self):
        return 'SingleImageDataset'
