import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.autograd import Variable
import numpy as np
import math
import torch.nn.functional as F

from collections import namedtuple
###############################################################################
# Functions
###############################################################################


def weights_init(m):
    classname = m.__class__.__name__
    print(classname)
    if classname.find('Conv') != -1:
        print(m)
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') !=-1:
        n = (m.in_features + m.out_features) / 2 
        m.weight.data.normal_(0, math.sqrt(2./n))
        m.bias.data.zero_()


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm)
    return norm_layer


def define_G(input_nc, output_nc, ngf, which_model_netG, norm='batch', use_dropout=False, gpu_ids=[]):
    netG = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())

    if which_model_netG == 'resnet_9blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_6blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=6, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_4blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=4, gpu_ids=gpu_ids)
    elif which_model_netG == 'resnet_3blocks':
        netG = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=3, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_128':
        netG = UnetGenerator(input_nc, output_nc, 7, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'unet_256':
        netG = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout, gpu_ids=gpu_ids)
    elif which_model_netG == 'lcnn':
        netG = LightCnnFeatureGenerator(gpu_ids=gpu_ids)
    elif which_model_netG == 'lcnn_conv':
        netG = LightCnnNoFCFeatureGenerator(gpu_ids=gpu_ids)
    elif which_model_netG == 'lcnn_fc1':
        netG = LightCnnFC1Generator(gpu_ids=gpu_ids)
    else:
        raise NotImplementedError('Generator model name [%s] is not recognized' % which_model_netG)
        
    if len(gpu_ids) > 0:
        netG.cuda(device_id=gpu_ids[0])

    print_network(netG)
    print(netG.state_dict().keys())

    netG.apply(weights_init)
    return netG


def define_D(input_nc, ndf, which_model_netD,
             n_layers_D=3, norm='batch', use_sigmoid=False, gpu_ids=[]):
    netD = None
    use_gpu = len(gpu_ids) > 0
    norm_layer = get_norm_layer(norm_type=norm)

    if use_gpu:
        assert(torch.cuda.is_available())
    if which_model_netD == 'basic':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == '2_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers=2, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'n_layers':
        netD = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid, gpu_ids=gpu_ids)
    elif which_model_netD == 'lcnn_feat_D':
        netD = FC_Discriminator(input_size=256, output_size=1)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' %
                                  which_model_netD)
    if use_gpu:
        netD.cuda(device_id=gpu_ids[0])
    netD.apply(weights_init)
    return netD

def define_F(num_classes, gpu_ids=[], update_paras = False):
    netF = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())
    netF = LightCnnGenerator(num_classes,gpu_ids)

    if use_gpu:
        netF.cuda(device_id=gpu_ids[0])
    netF.apply(weights_init)

    if not update_paras:
        netF.eval()
    return netF

def define_ResNeXt_F(num_classes,input_size, input_channel, gpu_ids=[]):
    netF = None
    use_gpu = len(gpu_ids) > 0

    if use_gpu:
        assert(torch.cuda.is_available())
        #def __init__(self, input_size=224, in_nc=3, num_classes=1000, n_conv0=64, n_conv_stage1=256, base_width=4, rep_lists=[3,4,6,3], norm_layer=nn.BatchNorm2d, padding_type='zero', gpu_ids=[])
    netF = ResNeXtGenerator(input_size=input_size, in_nc=input_channel, num_classes=num_classes, n_conv0=64, n_conv_stage1=64, rep_lists=[2,3,4,2], gpu_ids=gpu_ids)

    if use_gpu:
        netF.cuda(device_id=gpu_ids[0])
    netF.apply(weights_init)
    return netF


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


##############################################################################
# Classes
##############################################################################


# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def get_target_tensor_new(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.Tensor(input.size()).fill_(self.real_label)
        else:
            target_tensor = self.Tensor(input.size()).fill_(self.fake_label)
        target_tensor = Variable(target_tensor, requires_grad=False)
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor_new(input, target_is_real)
        return self.loss(input, target_tensor)


# Defines the generator that consists of Resnet blocks between a few
# downsampling/upsampling operations.
# Code and idea originally from Justin Johnson's architecture.
# https://github.com/jcjohnson/fast-neural-style/
class ResnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, gpu_ids=[], padding_type='reflect'):
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        self.input_nc = input_nc
        self.output_nc = output_nc
        self.ngf = ngf
        self.gpu_ids = gpu_ids

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):
            mult = 2**i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2**n_downsampling
        for i in range(n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout)]

        for i in range(n_downsampling):
            mult = 2**(n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


# Defines the Unet generator.
# |num_downs|: number of downsamplings in UNet. For example,
# if |num_downs| == 7, image of size 128x128 will become of size 1x1
# at the bottleneck
class UnetGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, num_downs, ngf=64,
                 norm_layer=nn.BatchNorm2d, use_dropout=False, gpu_ids=[]):
        super(UnetGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        # currently support only input_nc == output_nc
        assert(input_nc == output_nc)

        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, norm_layer=norm_layer, innermost=True)
        for i in range(num_downs - 5):
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, submodule=unet_block, outermost=True, norm_layer=norm_layer)

        self.model = unet_block

    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the submodule with skip connection.
# X -------------------identity---------------------- X
#   |-- downsampling -- |submodule| -- upsampling --|
class UnetSkipConnectionBlock(nn.Module):
    def __init__(self, outer_nc, inner_nc,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost

        downconv = nn.Conv2d(outer_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:
            return torch.cat([self.model(x), x], 1)


# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False, gpu_ids=[]):
        super(NLayerDiscriminator, self).__init__()
        self.gpu_ids = gpu_ids

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)


# Defines the LightCNN generator.
class mfm(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, type=1):
        super(mfm, self).__init__()
        self.out_channels = out_channels
        if type == 1:
            self.filter = nn.Conv2d(in_channels, 2*out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        else:
            self.filter = nn.Linear(in_channels, 2*out_channels)

    def forward(self, input):
        x = self.filter(input)
        out = torch.split(x, self.out_channels, 1)
        return torch.max(out[0], out[1])

class group(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(group, self).__init__()
        self.conv_a = mfm(in_channels, in_channels, 1, 1, 0)
        self.conv   = mfm(in_channels, out_channels, kernel_size, stride, padding)

    def forward(self, input):
        x = self.conv_a(input)
        x = self.conv(x)
        return x

class LightCnnGenerator(nn.Module):
    def __init__(self, num_classes=99891, gpu_ids=[]):
        super(LightCnnGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)
        self.fc2 = nn.Linear(256, num_classes)


    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.features, input, self.gpu_ids)
            x = x.view(x.size(0), -1)
            feature = nn.parallel.data_parallel(self.fc1, x, self.gpu_ids)
            x = F.dropout(feature, training=self.training)
            out = nn.parallel.data_parallel(self.fc2, x, self.gpu_ids)
        else:
            x = self.features(input)
            x = x.view(x.size(0), -1)
            feature = self.fc1(x)
            x = F.dropout(feature, training=self.training)
            out = self.fc2(x)
        return out, feature


class LightCnnFeatureGenerator(nn.Module):
    def __init__(self, gpu_ids=[]):
        super(LightCnnFeatureGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        self.fc1 = mfm(8*8*128, 256, type=0)


    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            x = nn.parallel.data_parallel(self.features, input, self.gpu_ids)
            x = x.view(x.size(0), -1)
            feature = nn.parallel.data_parallel(self.fc1, x, self.gpu_ids)
            # feature = F.dropout(feature, training=self.training)
        else:
            x = self.features(input)
            x = x.view(x.size(0), -1)
            feature = self.fc1(x)
            # feature = F.dropout(feature, training=self.training)
        return feature

class LightCnnNoFCFeatureGenerator(nn.Module):
    # output conv features(feature map of the last conv layer) of the light cnn model
    def __init__(self, gpu_ids=[]):
        super(LightCnnNoFCFeatureGenerator, self).__init__()
        self.gpu_ids = gpu_ids

        self.features = nn.Sequential(
            mfm(1, 48, 5, 1, 2), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(48, 96, 3, 1, 1), 
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            group(96, 192, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True), 
            group(192, 128, 3, 1, 1),
            group(128, 128, 3, 1, 1),
            nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
            )
        # self.fc1 = mfm(8*8*128, 256, type=0)

    def forward(self, input):
        if len(self.gpu_ids) and isinstance(input.data, torch.cuda.FloatTensor):
            feature = nn.parallel.data_parallel(self.features, input, self.gpu_ids)
        else:
            feature = self.features(input)
        return feature

class LightCnnFC1Generator(nn.Module):
    # input is the feature map of conv layers
    def __init__(self, gpu_ids=[]):
        super(LightCnnFC1Generator, self).__init__()
        self.gpu_ids = gpu_ids
        self.fc1 = mfm(8*8*128, 256, type=0)

    def forward(self, x):
        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):
            x = x.view(x.size(0), -1)
            feature = nn.parallel.data_parallel(self.fc1, x, self.gpu_ids)
        else:
            x = x.view(x.size(0), -1)
            feature = self.fc1(x)
        return feature

#define the ResNext block
class ResNeXtBottleneck(nn.Module):
    """
    RexNeXt bottleneck type C (https://github.com/facebookresearch/ResNeXt/blob/master/models/resnext.lua)
    """
    def __init__(self, in_channels, out_channels, stride, cardinality, widen_factor):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            widen_factor: factor to reduce the input dimensionality before convolution.
        """
        super(ResNeXtBottleneck, self).__init__()
        D = cardinality * out_channels // widen_factor
        self.conv_reduce = nn.Conv2d(in_channels, D, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_reduce = nn.BatchNorm2d(D)
        self.conv_conv = nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn = nn.BatchNorm2d(D)
        self.conv_expand = nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn_expand = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut.add_module('shortcut_conv', nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False))
            self.shortcut.add_module('shortcut_bn', nn.BatchNorm2d(out_channels))

    def forward(self, x):
        bottleneck = self.conv_reduce.forward(x)
        bottleneck = F.relu(self.bn_reduce.forward(bottleneck), inplace=True)
        bottleneck = self.conv_conv.forward(bottleneck)
        bottleneck = F.relu(self.bn.forward(bottleneck), inplace=True)
        bottleneck = self.conv_expand.forward(bottleneck)
        bottleneck = self.bn_expand.forward(bottleneck)
        residual = self.shortcut.forward(x)
        return F.relu(residual + bottleneck, inplace=True)



#define the ResNext block
class ResNeXtBlock(nn.Module):

    def __init__(self, in_channels, out_channels, stride, cardinality, width, padding_type, norm_layer=torch.nn.BatchNorm2d):
        """ Constructor
        Args:
            in_channels: input channel dimensionality
            out_channels: output channel dimensionality
            stride: conv stride. Replaces pooling layer.
            cardinality: num of convolution groups.
            width: num of channels of each convolution groups.
            padding_type: 'reflect','replicate','zero'
            norm_layer: norm_layer types
        """
        super(ResNeXtBlock, self).__init__()
        self.conv_block = self.build_conv_block(in_channels, out_channels, stride, cardinality, width, padding_type, norm_layer)
        self.shortcut = self.build_shortcut(in_channels, out_channels, stride, norm_layer)

    def build_conv_block(self, in_channels, out_channels, stride, cardinality, width, padding_type, norm_layer=torch.nn.BatchNorm2d):
        D = cardinality * width

        #conv_reduce
        conv_block = [nn.Conv2d(in_channels, D, kernel_size=1, padding=0, bias=False), 
                        norm_layer(D),
                        nn.ReLU(True)]
        #conv_conv
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block +=[nn.Conv2d(D, D, kernel_size=3, stride=stride, padding=p, groups=cardinality, bias=False), 
                        norm_layer(D),
                        nn.ReLU(True)]
        #conv_expand
        conv_block += [nn.Conv2d(D, out_channels, kernel_size=1, stride=1, padding=0, bias=False), 
                        norm_layer(out_channels)]
        return nn.Sequential(*conv_block)

    def build_shortcut(self, in_channels, out_channels, stride, norm_layer=torch.nn.BatchNorm2d):
        shortcut = []
        if in_channels != out_channels or stride !=1:
            shortcut +=[nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
                        norm_layer(out_channels)]
        return nn.Sequential(*shortcut)

    def forward(self, x):
        # print('--ResNeXtBlock---')
        # print(x.size())
        out = self.conv_block(x) + self.shortcut(x)
        # print(out.size())

        return F.relu(out, inplace=True)


#define the ResNext network generator
class ResNeXtGenerator(nn.Module):
    def __init__(self, input_size=224, in_nc=3, num_classes=1000, n_conv0=64, n_conv_stage1=256, cardinality=32, base_width=4, rep_lists=[3,4,6,3], norm_layer=nn.BatchNorm2d, padding_type='zero', gpu_ids=[]):
        """ 
        Args:
            input_size: input width/height
            in_nc: input channel dimensionality
            num_classes: output num of classes
            n_conv0: num of filters of conv1, before the first ResNeXt block
            n_conv_stage1: num of output channel of the ResNeXt stage1
            cardinality: num of the convolution groups of the ResNext block
            base_width: if the base_width is 4, width of each ResNeXt block is 4,8,12...
            rep_lists: repetition times of ResNeXt block with each base_width.
                i.e: if rep_lists=[3,4,6,3], then the generated model is
                        -- 3 ResNext blocks with width = base_width, 
                        -- 4 ResNext blocks with width = base_width*2,
                        -- 6 ResNext blocks with width = base_width*3,
                        -- 3 ResNext blocks with width = base_width*4,
            norm_layer: norm_layer types
            padding_type: 'reflect','replicate','zero'
        """
        assert(len(rep_lists) > 0)
        super(ResNeXtGenerator, self).__init__()
        self.cardinality = cardinality
        self.padding_type = padding_type
        self.norm_layer = norm_layer
        self.gpu_ids = gpu_ids

        #conv0
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_nc, n_conv0, kernel_size=7, padding=0),
                 norm_layer(n_conv0),
                 nn.ReLU(True)]

        #resnext blocks
        t_width = base_width
        t_in_nc = n_conv0
        t_out_nc = n_conv_stage1
        for stage in range(len(rep_lists)):
            stage_name = 'stage_%d' % (stage+1)
            model += self.ResNeXt_stage(stage_name, t_in_nc, t_out_nc, block_depth=rep_lists[stage], pool_stride=2, width=t_width)
            if stage < len(rep_lists)-1:
                t_in_nc = t_out_nc
                t_out_nc *= 2
                t_width *=2


        self.model = nn.Sequential(*model)

        #global avg pool
        down_sampling = pow(2, len(rep_lists) + 1)
        global_pool_size = math.ceil(input_size / down_sampling)
        print('global_pool_size = %d' % int(global_pool_size))
        self.model.add_module('avg_pool', nn.AvgPool2d(kernel_size=int(global_pool_size), stride=int(global_pool_size)))

        #self.model = nn.Sequential(*model)

        # fc
        self.fc = nn.Linear(t_out_nc, num_classes)

    def ResNeXt_stage(self, name, in_nc, out_nc, block_depth, pool_stride, width):
        """ Stack n bottleneck modules where n is inferred from the depth of the network.
        Args:
            name: string name of the current block.
            in_nc: number of input channels
            out_nc: number of output channels
            block_depth: num of resnext bottlenecks
            pool_stride: factor to reduce the spatial dimensionality in the first bottleneck of the block.
        Returns: a Module consisting of n sequential bottlenecks.
        """
        block = nn.Sequential()
        for bottleneck in range(block_depth):
            name_ = '%s_bottleneck_%d' % (name, bottleneck)
            if bottleneck == 0:
                block.add_module(name_, ResNeXtBlock(in_nc, out_nc, stride=pool_stride, cardinality = self.cardinality, width=width, 
                                                          padding_type=self.padding_type, norm_layer = self.norm_layer))
            else:
                block.add_module(name_, ResNeXtBlock(out_nc, out_nc, stride=1, cardinality = self.cardinality, width=width, 
                                                          padding_type=self.padding_type, norm_layer = self.norm_layer))
        return block


    def forward(self,input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            # print(input.size())
            feature = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
            feature = feature.view(feature.size(0),-1)
            # print(feature.size())
            cls_score = nn.parallel.data_parallel(self.fc, feature, self.gpu_ids)
        else:
            feature = self.model(input)
            feature = feature.view(feature.size(0),-1)
            cls_score = self.fc(feature)
        return cls_score

    def extract_feature(self,input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            # print(input.size())
            feature = nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            feature = self.model(input)
        feature = feature.view(feature.size(0),-1)
        return feature


# https://discuss.pytorch.org/t/how-to-extract-features-of-an-image-from-a-trained-model/119/3
class ExtractFeatureNetwork(nn.Module):
    def __init__(self, model, layer_mapping):
        super(ExtractFeatureNetwork, self).__init__()
        self.model = model
        self.layer_name_mapping = layer_mapping
        self.feature_output = namedtuple('FeatureDic', layer_mapping.values())
    # layer_name_mapping = {
    #         '3': "relu1_2",
    #         '8': "relu2_2",
    #         '15': "relu3_3",
    #         '22': "relu4_3"
    #     }

    def forward(self, x):
        output = {}
        for name, module in self.model._modules.items():
            x = module(x)
            if name in self.layer_name_mapping:
                output[self.layer_name_mapping[name]] = x
        return self.feature_output(**output)


class FC_Discriminator(nn.Module):
    def __init__(self, input_size,output_size, num_hidden=128, gpu_ids=[]):
        super(FC_Discriminator,self).__init__()
        self.fc = nn.Sequential(nn.Linear(input_size, num_hidden),
            nn.Linear(num_hidden, output_size))
        self.gpu_ids = gpu_ids

    def forward(self, x):
        if len(self.gpu_ids) and isinstance(x.data, torch.cuda.FloatTensor):
            x = F.dropout(x, training=self.training)
            x = x.view(x.size(0), -1)
            x = nn.parallel.data_parallel(self.fc, x, self.gpu_ids)
            x = nn.parallel.data_parallel(torch.nn.functional.sigmoid, x, self.gpu_ids)
        else:
            x = F.dropout(x, training=self.training)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            x = torch.nn.functional.sigmoid(x)
        return x



class CMD_K_Loss(nn.Module):
    def __init__(self, K=1, p=2):
        super(CMD_K_Loss, self).__init__()
        assert(K>0)
        assert(p>0)
        self.K=K
        self.p=p

    def forward(self, inputA, inputB):
        N_A = inputA.size(0)
        N_B = inputB.size(0)

        meanA = inputA.mean(0, keepdim=True)
        meanB = inputB.mean(0, keepdim=True)

        assert(meanA.size()==meanB.size())

        if self.K==1:
            loss = meanA.sub(meanB).norm(self.p)
        else:
            meanA = meanA.expand_as(inputA)
            meanB = meanB.expand_as(inputB)

            distA = inputA.sub(meanA)
            distB = inputB.sub(meanB)

            CM_A = distA.pow(self.K).mean(0)
            CM_B = distB.pow(self.K).mean(0)

            loss = CM_A.sub(CM_B).norm(self.p)

        return loss

class CMD_Loss(nn.Module):
    def __init__(self, K=1, p=2, decay=1.0):
        super(CMD_Loss, self).__init__()
        assert(K>0)
        assert(p>0)
        assert(decay>0)
        self.K = K
        self.p = p
        self.decay = decay

    def forward(self, inputA, inputB):
        N_A = inputA.size(0)
        N_B = inputB.size(0)

        meanA = inputA.mean(0, keepdim=True)
        meanB = inputB.mean(0, keepdim=True)

        assert(meanA.size()==meanB.size())
        loss = meanA.sub(meanB).norm(self.p)

        if self.K>1:       
            meanA = meanA.expand_as(inputA)
            meanB = meanB.expand_as(inputB)
            distA = inputA.sub(meanA)
            distB = inputB.sub(meanB)
            for i in range(2,self.K+1): 
                CM_A = distA.pow(self.K).mean(0)
                CM_B = distB.pow(self.K).mean(0)
                loss_k = CM_A.sub(CM_B).norm(self.p)
                loss +=loss_k*pow(self.decay,i-1)

        return loss

        




