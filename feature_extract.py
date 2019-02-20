import time
import os
from options.feature_extraction_options import ExtractFeaturesOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from util.visualizer import Visualizer
from pdb import set_trace as st
from util import html

opt = ExtractFeaturesOptions().parse()
opt.nThreads = 1   # test code only supports nThreads = 1
opt.serial_batches = True  # no shuffle
opt.no_flip = True  # no flip

# opt.batchSize = 1

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
model = create_model(opt)
visualizer = Visualizer(opt)

model.print_netF_module()

# test
count_batch = 0
for i, data in enumerate(dataset):
    # if i>=1:
        # break
    model.set_input(data)
    model.test()
    features = model.get_test_features()
    saveRoot = os.path.join(opt.save_root, opt.which_epoch)
    visualizer.save_features(features, saveRoot)
    # visuals = model.get_test_visuals()
    # visualizer.save_test_images(saveRoot, 'test_face', visuals, opt.color_mode)
    count_batch +=1
    print('process batch... %s' % count_batch)