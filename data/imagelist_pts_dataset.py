import os.path
import random
import torchvision.transforms as transforms
import torch
from data.base_dataset import BaseDataset
from data.image_folder import make_dataset
from PIL import Image
import numpy as np


class Imglist_Pts_Dataset(BaseDataset):
    def initialize(self, opt):
        # assert(opt.resize_or_crop == 'crop')

        self.opt = opt
        self.root = opt.dataroot
        self.color_mode = opt.color_mode
        self.list = opt.train_list
        self.is_train = opt.isTrain
        # train_list format:
        #   img0 img1 x1 y1 x2 y2...
        #   img0 img2 x1 y1 x2 y2...
        
        self.imgPtsList = self.default_list_pts_reader(self.list, bbox_flag=True)
        random.shuffle(self.imgPtsList)

        transform_list = [transforms.ToTensor()]
        if opt.input_normalize:
            transform_list.append(transforms.Normalize((0.5, 0.5, 0.5),
                                               (0.5, 0.5, 0.5)))

        self.transform = transforms.Compose(transform_list)
        self.debug_flag=False

    def __getitem__(self, index):
        A_imgPath, B_imgPath, Pts = self.imgPtsList[index]

        A_img = Image.open(A_imgPath).convert(self.color_mode)
        if self.is_train:
            B_img = Image.open(B_imgPath).convert(self.color_mode)

        w0 = A_img.size[0]
        h0 = A_img.size[1]
        c0 = Pts.shape[0]
        Heat_map = np.zeros((h0,w0,c0))
        sigma = 2
        for i in range(c0):
            Heat_map[:,:,i] = self.draw_gaussian(Heat_map[:,:,i], Pts[i,:], sigma)
            Heat_map[:,:,i] = Heat_map[:,:,i]*255


        # A_img = A_img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        # B_img = B_img.resize((self.opt.loadSize, self.opt.loadSize), Image.BICUBIC)
        # for i in range(c0):
        #     Heat_map[:,:,i] = Heat_map[:,:,i]

        A_img = self.transform(A_img)
        if self.is_train:
            B_img = self.transform(B_img)
        Heat_map = self.transform(Heat_map)

        #concat A_img and heat_map
        A_Pts_img = torch.cat((A_img, Heat_map),0)
        # print(type(A_img))
        # print(type(B_img))
        # print(A_img.size())
        # print(B_img.size())

        ##
        
        if self.is_train and self.debug_flag:
            print('--------A_img + Heat_map size:--------')
            print(A_Pts_img.size())
            print('----------------')
            fout=open("/home/lingxiao.song/slx_project/pytorch/cycleGAN/A_Heat_map.txt",'wb')
            cc = A_Pts_img.size(0)
            for i in range(cc):
                np.savetxt(fout,A_Pts_img[i].cpu().numpy(), delimiter=' ', fmt='%10.5f')
            fout.close()
            self.debug_flag=False
        ##
        
        w = A_img.size(2)
        h = A_img.size(1)
        if self.opt.isTrain:
            w_offset = random.randint(0, max(0, w - self.opt.fineSize - 1))
            h_offset = random.randint(0, max(0, h - self.opt.fineSize - 1))
        else:
            w_offset = int((self.opt.loadSize - self.opt.fineSize)/2)
            h_offset = w_offset

        A = A_img[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        A_Pts = A_Pts_img[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]
        if self.is_train:
            B = B_img[:, h_offset:h_offset + self.opt.fineSize,
               w_offset:w_offset + self.opt.fineSize]

        #flip image
        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            A_Pts = A_Pts.index_select(2, idx)
            if self.is_train:
                B = B.index_select(2, idx)

        if self.is_train:
            return {'A': A, 'A_Pts': A_Pts, 'B': B,
                'A_path': A_imgPath, 'B_path': B_imgPath}
        else:
            return {'A': A, 'A_Pts': A_Pts,
                'A_path': A_imgPath, 'B_path': B_imgPath}

    def __len__(self):
        return len(self.imgPtsList)

    def name(self):
        return 'ImglistPtsDataset'


    def draw_gaussian(self, img, pt, sigma):
        # Draw a 2D gaussian 
        # Adopted from https://github.com/anewell/pose-hg-train/blob/master/src/pypose/draw.py
        # img: numpy array

        # Check that any part of the gaussian is in-bounds
        ul = [int(pt[0] - 3 * sigma), int(pt[1] - 3 * sigma)]
        br = [int(pt[0] + 3 * sigma + 1), int(pt[1] + 3 * sigma + 1)]
        if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or
                br[0] < 0 or br[1] < 0):
            # If not, just return the image as is
            return img

        # Generate gaussian
        size = 6 * sigma + 1
        x = np.arange(0, size, 1, float)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], img.shape[1])
        img_y = max(0, ul[1]), min(br[1], img.shape[0])

        img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
        return img

    def default_list_pts_reader(self, fileList, bbox_flag=False):
        print("===>Read list from '{}'......".format(fileList))
        imgList = []
        with open(fileList, 'r') as file:
            for line in file.readlines():
                line_list = line.strip().split(' ')
                A_imgPath=line_list[0]
                B_imgPath=line_list[1]
                if bbox_flag:
                    pts = np.array(map(float,line_list[6:]))
                else:
                    pts = np.array(map(float,line_list[2:]))
                sp = pts.shape
                pts_count = sp[0]/2
                pts=pts.reshape(pts_count,2)

                A_path = os.path.join(self.root, A_imgPath)
                B_path = os.path.join(self.root, B_imgPath)
                imgList.append((A_path, B_path, pts))

        print("===>Read list from '{}' successed".format(fileList))
        return imgList
