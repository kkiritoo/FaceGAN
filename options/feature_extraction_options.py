from .base_options import BaseOptions


class ExtractFeaturesOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)

        self.parser.add_argument('--save_root', type=str, default='features', help='root folder of saving the feature files')
        self.parser.add_argument('--img_list', default='', type=str, metavar='listfile', help='list of face images for feature extraction (default: none).')
        self.parser.add_argument('-b', '--batch_size', default=10, type=int, metavar='N', help='mini-batch size (default: 10)')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.isTrain = False

