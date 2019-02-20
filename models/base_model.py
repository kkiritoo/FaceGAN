import os
import torch
from torch.nn.parameter import Parameter

import numpy as np


class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.LongTensor = torch.cuda.LongTensor if self.gpu_ids else torch.LongTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)
        if len(gpu_ids) and torch.cuda.is_available():
            network.cuda(device_id=gpu_ids[0])

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        save_path = os.path.join(self.save_dir, save_filename)
        network.load_state_dict(torch.load(save_path))

    def update_learning_rate():
        pass

    def load_state_dict_pretrained(self, model, pretrained_dict):
        model_dict = model.state_dict()
        print("--load_state_dict_pretrained: ")
        print("--   src keys:  --")
        print(pretrained_dict.keys())
        print('--   dst keys:  --')
        print(model_dict.keys())

        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # print('\nloading state_dict of pretrained model:')
        
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 
        # 3. load the new state dict
        model.load_state_dict(model_dict)

    def load_lightcnn_state_dict(self, model, pretrained_dict):
        model_dict = model.state_dict()
        print("--load_lightcnn_state_dict: ")
        print("--   src keys:  --")
        print(pretrained_dict.keys())
        print('--   dst keys:  --')
        print(model_dict.keys())

        for name, param in pretrained_dict.items():
            if name.find('fc2')!=-1:
                print('Ignoring layer {}'.format(name))
                continue;
            if name[7:] not in model_dict:
                print('Ignoring layer {}'.format(name))
                continue
            if isinstance(param, Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                print('Load dst layer {} <==== src lsyer{}'.format(name[7:],name))
                model_dict[name[7:]].copy_(param)
            except:
                print('While copying the parameter named {}, whose dimensions in the model are'
                      ' {} and whose dimensions in the checkpoint are {}, ...'.format(
                          name, model_dict[name[7:]].size(), param.size()))
                raise

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred    = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

    def save_features_text(self, save_file, features):
        fout = open(save_file, 'ab')
        np.savetxt(fout, features, delimiter=' ', fmt='%.7f')
        fout.close()
            
