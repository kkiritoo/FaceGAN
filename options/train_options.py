from .base_options import BaseOptions


class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100, help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--niter', type=int, default=100, help='# of iter at starting learning rate')
        self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to linearly decay learning rate to zero')
        self.parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
        self.parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        #cycleGAN related options
        self.parser.add_argument('--no_lsgan', action='store_true', help='do *not* use least square GAN, if false, use vanilla GAN')
        self.parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        self.parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        self.parser.add_argument('--pool_size', type=int, default=50, help='the size of image buffer that stores previously generated images')
        self.parser.add_argument('--lambda_F', type=float, default=1.0, help='weight for feature L1 loss (realA -> fakeB, realB -> fakeA)')

        self.parser.add_argument('--epoch_F', type=int, default=50, help='# of iter at starting add face feature L1 Loss')
        # face classification related options
        
        self.parser.add_argument('--lambda_Softmax', type=float, default=1.0, help='weight for softmax loss')
        self.parser.add_argument('--lambda_Center', type=float, default=1.0, help='weight for center loss')
        self.parser.add_argument('--lambda_GAN', type=float, default=1.0, help='weight for GAN loss')
        self.parser.add_argument('--lambda_Y', type=float, default=1.0, help='weight for Y channel in YCbCr space')
        self.parser.add_argument('--which_loss_cmd', type=str, default='cmd', help='choose cmd loss function. [cmd | cmd_k]')
        self.parser.add_argument('--cmd_K', type=int, default=1, help='order K of the cmd loss')
        self.parser.add_argument('--cmd_decay', type=float, default=1.0, help='dacay of the cmd loss, only valid for cmd loss, not cmd_k loss')
        self.parser.add_argument('--share_FC2', action='store_true', help='share weights of the FC layer for softmax')


        self.isTrain = True

