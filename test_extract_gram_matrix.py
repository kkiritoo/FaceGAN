import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.nn.functional as F
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import os
from collections import OrderedDict
from collections import namedtuple
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from models.base_model import BaseModel
from models import networks
from models.networks import LightCnnGenerator
from models.networks import ExtractFeatureNetwork
import sys
from util.util import gram_matrix

import cv2

def extract_gram_feature(model, transform, img_path):
	input = torch.zeros(1, 1, 128, 128)
	img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
	if img.shape==(144,144):
		img = img[8:136,8:136] 
	img = np.reshape(img, (128, 128, 1))
	img = transform(img)
	input[0,:,:,:] = img
	input_var = torch.autograd.Variable(input, volatile=True)
	features = model.feat_network(input_var)

	# for m in range(len(features)):
		# print(features._fields[m])

	gram_feat = [torch.autograd.Variable(gram_matrix(y).data, requires_grad=False) for y in features]
	return gram_feat

def style_loss(gram1, gram2):
	mse_loss = torch.nn.MSELoss()

	loss=0.0
	for m in range(len(gram1)):
		loss += mse_loss(gram1[m], gram2[m].expand_as(gram1[m]))
	return loss

def style_loss_list(gram1, gram2):
	mse_loss = torch.nn.MSELoss()

	loss=[]
	for m in range(len(gram1)):
		loss.append(mse_loss(gram1[m], gram2[m].expand_as(gram1[m])).data[0])
	return loss


bm = BaseModel()

#light cnn
bm.lcnn = LightCnnGenerator()
bm.lcnn.eval()
# checkpoint = torch.load('/home/lingxiao.song/slx_project/pytorch/light_cnn/lightCNN_6_checkpoint.pth.tar')
checkpoint = torch.load('/home/lingxiao.song/slx_project/pytorch/slx_code/save_model/casia_3/07.13/cross_view_test_20_checkpoint.pth.tar')
bm.load_lightcnn_state_dict(bm.lcnn, checkpoint['state_dict'])

layer_name_mapping={'0': 'conv1','2': 'conv2_2','4': 'conv3_2','6': 'conv4_2','7': 'conv5_2'}
bm.feat_network = ExtractFeatureNetwork(bm.lcnn.features, layer_name_mapping)
transform = transforms.Compose([transforms.ToTensor()])

#extract gram features
vis1= '/home/lingxiao.song/slx_project/python/s1_VIS.jpg'
vis2= '/home/lingxiao.song/slx_project/python/s2_VIS.jpg'
nir1= '/home/lingxiao.song/slx_project/python/s1_NIR.jpg'
nir2= '/home/lingxiao.song/slx_project/python/s2_NIR.jpg'

gram_vis1=extract_gram_feature(bm, transform, vis1)
gram_vis2=extract_gram_feature(bm, transform, vis2)
gram_nir1=extract_gram_feature(bm, transform, nir1)
gram_nir2=extract_gram_feature(bm, transform, nir2)



style_loss_vis2vis = style_loss_list(gram_vis1, gram_vis2)
style_loss_nir2nir = style_loss_list(gram_nir1, gram_nir2)
style_loss_vis2nir1 = style_loss_list(gram_vis1, gram_nir1)
style_loss_vis2nir2 = style_loss_list(gram_vis2, gram_nir2)
style_loss_vis2nir1_2 = style_loss_list(gram_vis1, gram_nir2)
style_loss_vis2nir2_1 = style_loss_list(gram_vis2, gram_nir1)

for m in range(len(style_loss_vis2nir1)):
	print('--%s--' % (layer_name_mapping.values()[m]))
	print('vis2vis: %.4e' %(style_loss_vis2vis[m]))
	print('nir2nir: %.4e' %(style_loss_nir2nir[m]))
	print('vis2nir1: %.4e' %(style_loss_vis2nir1[m]))
	print('vis2nir2: %.4e' %(style_loss_vis2nir2[m]))
	print('vis2nir1_2: %.4e' %(style_loss_vis2nir1_2[m]))
	print('vis2nir2_1: %.4e' %(style_loss_vis2nir2_1[m]))





