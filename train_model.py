from __future__ import print_function, division
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from ALD.dataLoader import Rescale, ToTensor, LandmarksDataset
from config import ConfigProfile
import models
from ALD.train import train_model
from ALD.lossFunction import fusionLossFunc_improved
import torch
from tqdm import tqdm
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

torch.multiprocessing.set_sharing_strategy('file_system')


def train(_model_name_to_save):

    torch.cuda.empty_cache()
    config = ConfigProfile()

    print('STARTED')
    model_ft = models.fusionVGG19(torchvision.models.vgg19_bn(pretrained=True), config).cuda(config.use_gpu)

    print("image scale ", config.image_scale)
    print("GPU: ", config.use_gpu)

    model_ft = model_ft.cuda(config.use_gpu)
    criterion = fusionLossFunc_improved(config)

    optimizer_ft = optim.Adadelta(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1.0)

    transform_origin = torchvision.transforms.Compose([
        Rescale(config.image_scale),
        ToTensor()
    ])

    for i in range(1):

        train_dataset_origin = LandmarksDataset(csv_file=config.dataRoot + config.trainingSetCsv,
                                                root_dir=config.dataRoot + config.supervised_dataset_train,
                                                transform=transform_origin,
                                                landmarksNum=config.landmarkNum
                                                )

        val_dataset = LandmarksDataset(csv_file=config.dataRoot + config.valSetCsv,
                                       root_dir=config.dataRoot + config.supervised_dataset_val,
                                       transform=transform_origin,
                                       landmarksNum=config.landmarkNum
                                       )
        print("Created datasets")

        train_dataloader = []
        val_dataloader = []

        train_dataloader_t = DataLoader(train_dataset_origin, batch_size=config.batchSize,
                                        shuffle=False, num_workers=1)

        val_dataloader_t = DataLoader(val_dataset, batch_size=config.batchSize,
                                      shuffle=False, num_workers=1)

        print("Created dataloaders")

        print('Pre-load train_dataloader_t')

        for data in tqdm(train_dataloader_t, total=len(train_dataloader_t)):
            train_dataloader.append(data)

        print('Pre-load val_dataloader_t')
        for data in tqdm(val_dataloader_t, total=len(val_dataloader_t)):
            val_dataloader.append(data)

        dataloaders = {'train': train_dataloader, 'val': val_dataloader}
        print('dataLoaders completed')

        train_model(model_ft, dataloaders, criterion, optimizer_ft, config, _model_name_to_save)


if __name__ == "__main__":

    model_name_to_save = 'sample_name'
    train(model_name_to_save)