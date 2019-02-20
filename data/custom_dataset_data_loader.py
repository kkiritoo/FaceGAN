import torch.utils.data
from data.base_data_loader import BaseDataLoader


def CreateDataset(opt):
    dataset = None
    if opt.dataset_mode == 'aligned':
        from data.aligned_dataset import AlignedDataset
        dataset = AlignedDataset()
    elif opt.dataset_mode == 'unaligned':
        from data.unaligned_dataset import UnalignedDataset
        dataset = UnalignedDataset()
    elif opt.dataset_mode == 'single':
        from data.single_dataset import SingleDataset
        dataset = SingleDataset()
    elif  opt.dataset_mode == 'imagelist':
        from data.imagelist_dataset import ImageList
        if opt.isTrain:
            dataset = ImageList(root=opt.image_root, fileList=opt.train_list)
        else:
            dataset = ImageList(root=opt.image_root, fileList=opt.train_list, testPahse=True)
    elif  opt.dataset_mode == 'imagelist_cross_view':
        from data.imagelist_dataset import ImageList_cross_view
        dataset = ImageList_cross_view()
    elif opt.dataset_mode == 'imglist_pts':
        from data.imagelist_pts_dataset import Imglist_Pts_Dataset
        dataset = Imglist_Pts_Dataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.dataset_mode)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)
