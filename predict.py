from __future__ import print_function, division
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from ALD.dataLoader import Rescale, ToTensor, LandmarksDataset
from config import ConfigProfile
from ALD.train import val
from ALD.lossFunction import fusionLossFunc_improved
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

torch.multiprocessing.set_sharing_strategy('file_system')


def predict(_path):

    torch.cuda.empty_cache()
    config = ConfigProfile()

    print('STARTED')
    model_ft = torch.load(_path).cuda(config.use_gpu)

    print("image scale ", config.image_scale)
    print("GPU: ", config.use_gpu)

    model_ft = model_ft.cuda(config.use_gpu)
    criterion = fusionLossFunc_improved(config)

    optimizer_ft = optim.Adadelta(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1.0)

    transform_origin = torchvision.transforms.Compose([
        Rescale(config.image_scale),
        ToTensor()
    ])

    val_dataset = LandmarksDataset(csv_file=config.dataRoot + config.testSetCsv,
                                   root_dir=config.dataRoot + config.supervised_dataset_test,
                                   transform=transform_origin,
                                   landmarksNum=config.landmarkNum
                                   )
    print("Created datasets")

    val_dataloader = []

    val_dataloader_t = DataLoader(val_dataset, batch_size=config.batchSize,
                                  shuffle=False, num_workers=1)

    print("Created dataloaders")

    print('Pre-load val_dataloader_t')
    for data in tqdm(val_dataloader_t, total=len(val_dataloader_t)):
        val_dataloader.append(data)

    dataloaders = {'val': val_dataloader}
    print('dataLoaders completed')

    val(model_ft, dataloaders, criterion, optimizer_ft, config, 0, path_to_model.split('/')[0])


if __name__ == "__main__":

    path_to_model = 'models/model_19pkt_ald_our.pt'
    predict(path_to_model)