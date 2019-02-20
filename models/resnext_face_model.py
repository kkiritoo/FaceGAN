import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data

import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys


class ResNeXt_Face_Model(BaseModel):
    def name(self):
        return 'ResNeXt_Face_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        assert(opt.dataset_mode == 'imagelist')

        #init input tensor
        nb = opt.batchSize
        size = opt.fineSize # W, H
        self.image = self.Tensor(nb, opt.input_nc, size, size)
        self.label = self.LongTensor(nb)

        # define networks 
        self.netF = networks.define_ResNeXt_F(opt.num_classes, size, opt.input_nc, opt.gpu_ids)
        print('---------- Networks initialized -------------')
        networks.print_network(self.netF)
        print('---------------------------------------------')
        
        #load networks
        if os.path.isfile(opt.F_weights):
            print("=> loading pretrained F model '{}'".format(opt.F_weights))
            checkpoint = torch.load(opt.F_weights)
            self.load_lightcnn_state_dict(self.netF, checkpoint['state_dict'])
            print('\n=> loaded pretrained F model from {}'.format(opt.F_weights))
        else:
            print("\n=> no pretrained F model found at '{}'".format(opt.F_weights))

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
            self.load_network(self.netF, 'F', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            # define loss functions
            self.criterion_cls = torch.nn.CrossEntropyLoss()
            # initialize optimizers
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

        self.save_feature_path = opt.save_feature_path

    def set_input(self, input):
        image = input['image']
        label = input['target']
        self.image.resize_(image.size()).copy_(image)
        self.label.resize_(label.size()).copy_(label)

    def forward(self):
        self.image_var = Variable(self.image)
        self.label_var = Variable(self.label)
        self.out = self.netF.forward(self.image_var)
        prec1, prec5 = self.accuracy(self.out.data, self.label, topk=(1,5))
        self.top1 = prec1[0]
        self.top5 = prec5[0]

    def backward_F(self):
        self.loss_cls = self.criterion_cls(self.out, self.label_var)
        self.loss_cls.backward()

    def optimize_parameters(self):
        #forward
        self.forward()

        #backward F
        self.optimizer_F.zero_grad()
        self.backward_F()
        self.optimizer_F.step()

    def get_current_errors(self):
        L_cls = self.loss_cls.data[0]
        return OrderedDict([('L_cls', L_cls),('top1', self.top1),('top5', self.top5)])

    def save(self, tag):
        self.save_network(self.netF, 'F', tag, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] = lr

        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def extract_feature(self):
        self.image_var = Variable(self.image)
        # print(self.image_var.size())
        self.feature = self.netF.extract_feature(self.image_var)
        return self.feature

    def save_features2txt(self):
        feature = self.extract_feature()
        # print(feature.size())
        np_features = features.data.cpu().numpy()
        self.save_features_text(self.save_feature_path, feature)






