
def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'cycle_gan':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_model import CycleGANModel
        model = CycleGANModel()
    elif opt.model == 'pix2pix':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_model import Pix2PixModel
        model = Pix2PixModel()
    elif opt.model == 'test':
        assert(opt.dataset_mode == 'single')
        from .test_model import TestModel
        model = TestModel()
    elif opt.model == 'cycle_gan_light_cnn':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_light_cnn_model import CycleGAN_Feat_Model
        model = CycleGAN_Feat_Model()
    elif opt.model == 'resnext_face':
        assert(opt.dataset_mode == 'imagelist')
        from .resnext_face_model import ResNeXt_Face_Model
        model = ResNeXt_Face_Model()
    elif opt.model == 'pix2pix_light_cnn':
        assert(opt.dataset_mode == 'aligned')
        from .pix2pix_lcnn_model import Pix2Pix_lightCnn_Model
        model = Pix2Pix_lightCnn_Model()
    elif opt.model == 'pts2face':
        assert(opt.dataset_mode == 'imglist_pts')
        from .pts2face_model import Pts2Face_Model
        model = Pts2Face_Model()
    elif opt.model == 'cross_view':
        assert(opt.dataset_mode == 'imagelist_cross_view')
        from .cross_view_model import CrossView_Model
        model = CrossView_Model()
    elif opt.model == 'cycle_gan_Y':
        assert(opt.dataset_mode == 'unaligned')
        from .cycle_gan_Y_model import CycleGAN_Y_Model
        model = CycleGAN_Y_Model()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model
