from __future__ import print_function, division
import torch
import time
import ALD.utils
from tqdm import tqdm
import gc


def train_model(model, dataloaders, criterion, optimizer, config, model_name):

    MODEL_DICT_PATH = f'{model_name}_{config.landmarkNum}_points_dict.pt'
    MODEL_PATH = f'{model_name}_{config.landmarkNum}_points_model.pt'
    LOG_PATH = f'log_train_{model_name}_{config.landmarkNum}_points.txt'

    since = time.time()
    test_epoch = 5
    for epoch in range(1, config.epochs+1):
        gc.collect()
        torch.cuda.empty_cache()
        train_dev = []
        epoch_start = time.time()
        for phase in ['train']:
            model.train(True)
            running_loss = 0.0

            lent = len(dataloaders[phase])
            pbar = tqdm(total=lent * config.batchSize)
            for ide in range(lent):
                data = dataloaders[phase][ide]

                inputs, labels = data['image'], data['landmarks']
                inputs = inputs.to(config.use_gpu)

                optimizer.zero_grad()
                heatmaps = model(inputs)

                loss = criterion(heatmaps[0].cpu(), labels.detach().cpu())

                loss.backward()
                optimizer.step()

                if epoch % test_epoch == 0:
                    predicted_landmarks = ALD.utils.regression_voting(heatmaps, config.R2).cuda(config.use_gpu)
                    dev = ALD.utils.calculate_deviation(predicted_landmarks.detach(), labels.cuda(config.use_gpu).detach())
                    train_dev.append(dev)
                    del dev

                running_loss += loss.item()
                del loss
                gc.collect()
                torch.cuda.empty_cache()

                pbar.update(config.batchSize)
            pbar.close()

            epoch_loss = running_loss / lent
            epoch_end = time.time()
            file_to_save = open(LOG_PATH, 'a')
            file_to_save.write('{} epoch: {} Loss: {}'.format(phase, epoch, epoch_loss))
            file_to_save.close()

        # validation
        if epoch % test_epoch == 0:
            train_dev = torch.stack(train_dev).squeeze() * config.spacing
            train_SDR, train_SD, train_MRE = ALD.utils.get_statistical_results(train_dev, config)
            a = torch.mean(train_MRE).detach().cpu().numpy()
            b = torch.mean(train_SD).detach().cpu().numpy()
            c = torch.mean((train_SDR), 0).detach().cpu().numpy()

            file_to_save = open(LOG_PATH, 'a')
            time_to_save = epoch_end - epoch_start
            file_to_save.write(
                "Epoch: {7} / time: {8} - train_MRE(SD): {0}({1}), SDR([1mm, 2mm, 2.5mm, 3mm, 4mm]): {2}, {3}, {4}, {5}, {6} \n".format(
                    a, b, c[0], c[1], c[2], c[3], c[4], epoch, time_to_save))
            file_to_save.close()
            torch.save(model.state_dict(), MODEL_DICT_PATH)
            torch.save(model, MODEL_PATH)

            del a, b, c, train_SDR, train_SD, train_MRE, train_dev
            gc.collect()
            torch.cuda.empty_cache()

            val(model, dataloaders, config, epoch, model_name)

        gc.collect()
        torch.cuda.empty_cache()

    time_elapsed = time.time() - since

    file_to_save = open(LOG_PATH, 'a')
    file_to_save.write('Training complete in {:.0f}m {:.0f}s \n'.format(time_elapsed // 60, time_elapsed % 60))
    file_to_save.close()


best_MRE = 10000
best_SDR = []
best_SD = 0


def val(model, dataloaders, config, epoch, model_name):
    since = time.time()

    test_dev = []

    LOG_PATH = f'log_val_{model_name}_{config.landmarkNum}_points.txt'
    COORDINATION_PATH = f'log_val_{model_name}_{config.landmarkNum}_points_epoch_{epoch}.txt'

    for phase in ['val']:
        model.train(False)
        lent = len(dataloaders[phase])
        pbar = tqdm(total=lent * config.batchSize)
        for ide in range(lent):
            data = dataloaders[phase][ide]

            inputs, labels = data['image'], data['landmarks']
            inputs = inputs.to(config.use_gpu)
            heatmaps = model(inputs)
            predicted_landmarks = ALD.utils.regression_voting(heatmaps, config.R2).to(config.use_gpu)
            dev = ALD.utils.calculate_deviation(predicted_landmarks.detach(), labels.to(config.use_gpu).detach())

            file_to_save = open(COORDINATION_PATH.format(epoch), 'a')
            for x, y, z in zip(predicted_landmarks.detach()[0], labels.to(config.use_gpu)[0], dev[0]):
                file_to_save.write("{0} {1} {2} {3} \n".format(
                    x.tolist(), y.tolist(), z.tolist() * 0.1, z.tolist() * 0.1 <= 2.5
                ))
            test_dev.append(dev)
            file_to_save.close()

            pbar.update(config.batchSize)
        pbar.close()

    # statistics

    test_dev = torch.stack(test_dev).squeeze() * config.spacing
    test_SDR, test_SD, test_MRE = ALD.utils.get_statistical_results(test_dev, config)

    a = torch.mean(test_MRE).detach().cpu().numpy()
    b = torch.mean(test_SD).detach().cpu().numpy()
    c = torch.mean((test_SDR), 0).detach().cpu().numpy()

    file_to_save = open(LOG_PATH, 'a')
    file_to_save.write(
        "test_MRE(SD): {0}({1}), SDR([1mm, 2mm, 2.5mm, 3mm, 4mm]): {2}, {3}, {4}, {5}, {6} \n".format(
            a, b, c[0], c[1], c[2], c[3], c[4]))
    file_to_save.close()

    global best_MRE
    global best_SD
    global best_SDR
    if best_MRE > torch.mean(test_MRE).detach().cpu().numpy():
        best_MRE = torch.mean(test_MRE).detach().cpu().numpy()
        best_SD = torch.mean(test_SD).detach().cpu().numpy()
        best_SDR = torch.mean((test_SDR), 0).detach().cpu().numpy()

    del a, b, c, test_dev, test_SDR, test_SD, test_MRE
    gc.collect()
    torch.cuda.empty_cache()

    time_elapsed = time.time() - since

    file_to_save = open(LOG_PATH, 'a')
    file_to_save.write('testing complete in {:.0f}m {:.0f}s\n'.format(
        time_elapsed // 60, time_elapsed % 60))
    file_to_save.close()