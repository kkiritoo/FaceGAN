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
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class Discriminator(nn.Module):
    def __init__(self, input_size,output_size, num_hidden=128):
        super(Discriminator,self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, num_hidden),
            nn.Linear(num_hidden, output_size))

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = torch.nn.functional.sigmoid(x)
        return x


class CrossView_Model(BaseModel):
    def name(self):
        return 'CrossView_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        assert(opt.which_model_netG == 'lcnn' or opt.which_model_netG == 'lcnn_conv')
        assert(opt.which_model_netD == 'lcnn_feat_D')
        assert(opt.input_normalize == False)
        assert(opt.input_nc==1)
        assert(opt.output_nc==1)

        self.isTrain = opt.isTrain
        # define tensors
        self.input_A = self.Tensor(opt.batchSize, opt.input_nc,
                                   opt.fineSize, opt.fineSize)
        self.input_B = self.Tensor(opt.batchSize, opt.output_nc,
                                   opt.fineSize, opt.fineSize)
        self.label_A = self.LongTensor(opt.batchSize)
        self.label_B = self.LongTensor(opt.batchSize)


        # define networks
        print('---------- Define networks...')
        if opt.which_model_netG == 'lcnn':
            self.netF = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, opt.use_dropout, gpu_ids=self.gpu_ids)
        else:
            self.netF = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      opt.which_model_netG, opt.norm, opt.use_dropout, gpu_ids=self.gpu_ids)
            self.netF_fc1_A = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      'lcnn_fc1', opt.norm, opt.use_dropout, gpu_ids=self.gpu_ids)
            self.netF_fc1_B = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf,
                                      'lcnn_fc1', opt.norm, opt.use_dropout, gpu_ids=self.gpu_ids)

        if self.isTrain:
            if self.opt.share_FC2:
                self.netF_softmax = nn.Linear(256, opt.num_classes)
                if len(self.gpu_ids) > 0:
                    self.netF_softmax.cuda(device_id=self.gpu_ids[0])
            else:
                self.netF_softmax_A = nn.Linear(256, opt.num_classes)
                self.netF_softmax_B = nn.Linear(256, opt.num_classes)
                if len(self.gpu_ids) > 0:
                    self.netF_softmax_A.cuda(device_id=self.gpu_ids[0])
                    self.netF_softmax_B.cuda(device_id=self.gpu_ids[0])
            self.netD = networks.define_D(opt.input_nc, opt.ndf,
                                          opt.which_model_netD, gpu_ids=self.gpu_ids)
        print('---------- Define networks done.')

        print('---------- Networks initialized -------------')
        networks.print_network(self.netF)
        if self.isTrain:
            if self.opt.share_FC2:
                networks.print_network(self.netF_softmax)
            else:
                networks.print_network(self.netF_softmax_A)
                networks.print_network(self.netF_softmax_B)
            networks.print_network(self.netD)
        print('-----------------------------------------------')


        # load networks
        if self.isTrain:
            if opt.continue_train:
                print('------ Loading netF...')
                self.load_network(self.netF, 'F', opt.which_epoch)
                print('---------- Loading netF success.')

                if self.opt.share_FC2:
                    self.load_network(self.netF_softmax, 'F_s', opt.which_epoch)
                    print('---------- Loading netF_softmax success.')                   
                else:
                    self.load_network(self.netF_softmax_A, 'F_sA', opt.which_epoch)
                    print('---------- Loading netF_softmax_A success.')

                    self.load_network(self.netF_softmax_B, 'F_sB', opt.which_epoch)
                    print('---------- Loading netF_softmax_B success.')

            elif os.path.isfile(opt.F_weights):
                print("---------- loading pretrained F model from '{}'...".format(opt.F_weights))
                checkpoint = torch.load(opt.F_weights)
                self.load_lightcnn_state_dict(self.netF, checkpoint['state_dict'])
                print("---------- loading pretrained F model from '{}' success.".format(opt.F_weights))
                if self.opt.which_model_netG=='lcnn_conv':               
                    print("---------- loading pretrained F_fc1_A model from '{}'...".format(opt.F_weights))
                    self.load_lightcnn_state_dict(self.netF_fc1_A, checkpoint['state_dict'])
                    print("---------- loading pretrained F_fc1_A model from '{}' success.".format(opt.F_weights))
                    print("---------- loading pretrained F_fc1_B model from '{}'...".format(opt.F_weights))
                    self.load_lightcnn_state_dict(self.netF_fc1_B, checkpoint['state_dict'])
                    print("---------- loading pretrained F_fc1_B model from '{}' success.".format(opt.F_weights))
            else:
                print("---------- not specify pretrained F model.")
        else:
            print('------ Loading netF...')
            self.load_network(self.netF, 'F', opt.which_epoch)
            print('---------- Loading netF success.')
            if self.opt.which_model_netG=='lcnn_conv':
                print('------ Loading netF_fc1_A...')
                self.load_network(self.netF_fc1_A, 'F_fc1_A', opt.which_epoch)
                print('---------- Loading netF_fc1_A success.')
                print('------ Loading netF_fc1_B...')
                self.load_network(self.netF_fc1_B, 'F_fc1_B', opt.which_epoch)
                print('---------- Loading netF_fc1_B success.')



        if self.isTrain:
            print('------ Loading netD...')
            if opt.continue_train:
                self.load_network(self.netD, 'D', opt.which_epoch)
                print('---------- Loading netD success.')
            else:
                print("---------- not specify pretrained D model.")


        if self.isTrain:
            self.fake_B_pool = ImagePool(opt.pool_size)
            self.old_lr = opt.lr

            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionCLS = torch.nn.CrossEntropyLoss()
            self.criterionCentor = torch.nn.MSELoss()
            if self.opt.which_loss_cmd == 'cmd':
                self.criterionVAR = networks.CMD_Loss(K=self.opt.cmd_K, decay=self.opt.cmd_decay)
            elif self.opt.which_loss_cmd == 'cmd_k':
                self.criterionVAR = networks.CMD_K_Loss(K=self.opt.cmd_K)
            else:
                raise NotImplementedError('CMD loss name [%s] is not recognized' % self.opt.which_loss_cmd)

            #self.criterionInView = torch

            # define optimizers
            self.optimizer_F = torch.optim.Adam(self.netF.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.which_model_netG=='lcnn_conv':
                self.optimizer_F_fc1_A = torch.optim.Adam(self.netF_fc1_A.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_F_fc1_B = torch.optim.Adam(self.netF_fc1_B.parameters(),
                                                lr=opt.lr, betas=(opt.beta1, 0.999))
            if self.opt.share_FC2:
                self.optimizer_F_s = torch.optim.Adam(self.netF_softmax.parameters(),
                                                lr=10*opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_F_sA = torch.optim.Adam(self.netF_softmax_A.parameters(),
                                                    lr=10*opt.lr, betas=(opt.beta1, 0.999))
                self.optimizer_F_sB = torch.optim.Adam(self.netF_softmax_B.parameters(),
                                                    lr=10*opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(),
                                                lr=10*opt.lr, betas=(opt.beta1, 0.999))
        else:
            self.netF.eval()



    def set_input(self, input):
        input_A = input['image0']
        self.input_A.resize_(input_A.size()).copy_(input_A)
        input_B = input['image1']
        self.input_B.resize_(input_B.size()).copy_(input_B)
        label_A = input['target0']
        self.label_A.resize_(label_A.size()).copy_(label_A)
        label_B = input['target1']
        self.label_B.resize_(label_B.size()).copy_(label_B)

    def forward(self):
        self.A = Variable(self.input_A)
        self.B = Variable(self.input_B)
        self.var_label_A = Variable(self.label_A)
        self.var_label_B = Variable(self.label_B)

        # forward F
        self.feat_A = self.netF.forward(self.A)
        self.feat_B = self.netF.forward(self.B)
        if self.opt.which_model_netG=='lcnn_conv':
            self.feat_A = self.netF_fc1_A.forward(self.feat_A)
            self.feat_B = self.netF_fc1_B.forward(self.feat_B)

        # forward softmax
        if self.opt.share_FC2:
            self.out_A = self.netF_softmax.forward(self.feat_A) 
            self.out_B = self.netF_softmax.forward(self.feat_B) 
        else:
            self.out_A = self.netF_softmax_A.forward(self.feat_A) 
            self.out_B = self.netF_softmax_B.forward(self.feat_B) 
        

    # no backprop gradients
    def test(self):
        self.A = Variable(self.input_A)
        self.B = Variable(self.input_B)
        # forward F
        self.feat_A = self.netF.forward(self.A)
        self.feat_B = self.netF.forward(self.B)
        if self.opt.which_model_netG=='lcnn_conv':
            self.feat_A = self.netF_fc1_A.forward(self.feat_A)
            self.feat_B = self.netF_fc1_B.forward(self.feat_B)

    # get image paths
    # def get_image_paths(self):
    #     return self.B_Path

    def backward_D(self):
        # Fake
        # stop backprop to the generator by detaching fake_B
        fake_feat = self.fake_B_pool.query(self.feat_B)
        self.pred_fake = self.netD.forward(fake_feat.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, False)

        # Real
        real_feat = self.feat_A
        self.pred_real = self.netD.forward(real_feat)
        self.loss_D_real = self.criterionGAN(self.pred_real, True)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5

        self.loss_D.backward(retain_graph=True)

    def backward_F(self,epoch):

        # First, F(B) should fake the discriminator
        fake_feat = self.feat_B
        pred_fake = self.netD.forward(fake_feat)
        self.loss_F_GAN = self.criterionGAN(pred_fake, True) * self.opt.lambda_GAN

        # softmax_A, softmax_B
        self.loss_Cls_A = self.criterionCLS(self.out_A, self.var_label_A) * self.opt.lambda_Softmax
        self.loss_Cls_B = self.criterionCLS(self.out_B, self.var_label_B) * self.opt.lambda_Softmax

        # center loss
        mean_feat_A = self.feat_A.mean(0, keepdim=True)
        mean_feat_B = self.feat_B.mean(0, keepdim=True)
        mean_feat_A = mean_feat_A.repeat(self.opt.batchSize,1)
        mean_feat_B = mean_feat_B.repeat(self.opt.batchSize,1)
        # self.loss_Center_A = self.criterionCentor(self.feat_A, mean_feat_A.detach()) * self.opt.lambda_Center
        # self.loss_Center_B = self.criterionCentor(self.feat_B, mean_feat_B.detach()) * self.opt.lambda_Center
        self.loss_CMD = self.criterionVAR(self.feat_A, self.feat_B) * self.opt.lambda_Center

        self.loss_F = self.loss_F_GAN + self.loss_Cls_A + self.loss_Cls_B + self.loss_CMD #self.loss_Center_A + self.loss_Center_B 
        # self.loss_F = self.loss_F_GAN + self.loss_Cls_A + self.loss_Cls_B + self.loss_CMD2

        self.loss_F.backward()

    def optimize_parameters(self,epoch):
        self.forward()

        self.optimizer_D.zero_grad()
        self.backward_D()
        self.optimizer_D.step()

        self.optimizer_F.zero_grad()
        if self.opt.which_model_netG=='lcnn_conv':
            self.optimizer_F_fc1_A.zero_grad()
            self.optimizer_F_fc1_B.zero_grad()
        if self.opt.share_FC2:
            self.optimizer_F_s.zero_grad()
        else:
            self.optimizer_F_sA.zero_grad()
            self.optimizer_F_sB.zero_grad()

        self.backward_F(epoch)

        if epoch > 1:
            self.optimizer_F.step()
            if self.opt.which_model_netG=='lcnn_conv':
                self.optimizer_F_fc1_A.step()
                self.optimizer_F_fc1_B.step()
        if self.opt.share_FC2:
            self.optimizer_F_s.step()
        else:
            self.optimizer_F_sA.step()
            self.optimizer_F_sB.step()

    def get_current_errors(self,epoch):
        return OrderedDict([('F_GAN', self.loss_F_GAN.data[0]),
                            ('CLS_A', self.loss_Cls_A.data[0]),
                            ('CLS_B', self.loss_Cls_B.data[0]),
                            ('CMD', self.loss_CMD.data[0]),
                            # ('Center_A', self.loss_Center_A.data[0]),
                            # ('Center_B', self.loss_Center_B.data[0]),
                            ('D_real', self.loss_D_real.data[0]),
                            ('D_fake', self.loss_D_fake.data[0]),
                            ])

    def save(self, label):
        self.save_network(self.netF, 'F', label, self.gpu_ids)
        if self.opt.which_model_netG=='lcnn_conv':
            self.save_network(self.netF_fc1_A, 'F_fc1_A', label, self.gpu_ids)
            self.save_network(self.netF_fc1_B, 'F_fc1_B', label, self.gpu_ids)
        self.save_network(self.netD, 'D', label, self.gpu_ids)
        if self.opt.share_FC2:
            self.save_network(self.netF_softmax, 'F_s', label, self.gpu_ids)
        else:
            self.save_network(self.netF_softmax_A, 'F_sA', label, self.gpu_ids)
            self.save_network(self.netF_softmax_B, 'F_sB', label, self.gpu_ids)

    def update_learning_rate(self):
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_F.param_groups:
            param_group['lr'] = lr
        if self.opt.which_model_netG=='lcnn_conv':
            for param_group in self.optimizer_F_fc1_A.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_F_fc1_B.param_groups:
                param_group['lr'] = lr
        if self.opt.share_FC2:
            for param_group in self.optimizer_F_s.param_groups:
                param_group['lr'] = 10*lr
        else:
            for param_group in self.optimizer_F_sA.param_groups:
                param_group['lr'] = 10*lr
            for param_group in self.optimizer_F_sB.param_groups:
                param_group['lr'] = 10*lr
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = 10*lr
        print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

    def get_test_features(self):
        return OrderedDict([('A', self.feat_A),
                            ('B', self.feat_B),
                            ])


    def get_test_visuals(self):
        img_A = util.tensor2im(self.A.data,normalize=self.opt.input_normalize)
        img_B = util.tensor2im(self.B.data,normalize=self.opt.input_normalize)
        return OrderedDict([('img_A', img_A), ('img_B', img_B)])

    def print_netF_module(self):
        # conv1_filter = self.netF._modules['features.0.filter.weight']
        # print(conv1_filter)
        pass
