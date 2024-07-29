from __future__ import print_function, division
import torch.optim as optim
import torchvision
from torch.utils.data import DataLoader
from dataLoader import Rescale, ToTensor, LandmarksDataset
import train
import lossFunction
import torch
import gc

torch.multiprocessing.set_sharing_strategy('file_system')






def main():

    torch.cuda.empty_cache()
    # config = ConfigProfile()

    print('STARTED')
    # model_ft = models.fusionVGG19(torchvision.models.vgg19_bn(pretrained=True), config).cuda(config.use_gpu)
    model_ft = torch.load('model_19pkt_ald.pt').cuda(config.use_gpu)

    print("image scale ", config.image_scale)
    print("GPU: ", config.use_gpu)

    model_ft = model_ft.cuda(config.use_gpu)
    criterion = lossFunction.fusionLossFunc_improved(config)

    optimizer_ft = optim.Adadelta(filter(lambda p: p.requires_grad, model_ft.parameters()), lr=1.0)

    file_to_save = open('publikacja_ald_original_83_2023.txt', 'a')

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

        val_dataset = LandmarksDataset(csv_file=config.dataRoot + config.testSetCsv,
                                       root_dir=config.dataRoot + config.supervised_dataset_test,
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

        gc.collect()
        torch.cuda.empty_cache()

        # pre-load all data into memory for efficient training

        print('Pre-load train_dataloader_t')
        idx_1 = 1
        idx_1_all = len(train_dataloader_t)
        for data in train_dataloader_t:
            if idx_1 % 10 == 0:
                print("{0} / {1}".format(idx_1, idx_1_all))
            train_dataloader.append(data)
            idx_1 += 1

        print('Pre-load val_dataloader_t')
        idx_2 = 1
        idx_2_all = len(val_dataloader_t)
        for data in val_dataloader_t:
            if idx_2 % 10 == 0:
                print("{0} / {1}".format(idx_2, idx_2_all))
            val_dataloader.append(data)
            idx_2 += 1

        print(len(train_dataloader), len(val_dataloader))

        dataloaders = {'train': train_dataloader, 'val': val_dataloader}

        print('dataLoaders completed')

        gc.collect()
        torch.cuda.empty_cache()

        train.train_model(model_ft, dataloaders, criterion, optimizer_ft, config, file_to_save)  # UCZENIE

        # train.val(model_ft, dataloaders, criterion, optimizer_ft, config, file_to_save, 0) # TESTOWANIE

        del train_dataloader, val_dataloader
        gc.collect()
        torch.cuda.empty_cache()

    file_to_save.close()


if __name__ == "__main__":
    main()