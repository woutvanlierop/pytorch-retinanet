import argparse
import collections

import numpy as np

import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, Augmenter, \
    Normalizer
from torch.utils.data import DataLoader

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
writer = SummaryWriter()


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_train', help='Path to file containing training annotations (see readme)')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')
    parser.add_argument('--model', help='Path to model.')

    parser.add_argument('--depth', help='Resnet depth, must be one of 18, 34, 50, 101, 152', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs', type=int, default=100)

    parser = parser.parse_args(args)

    # Create the data loaders
    if parser.dataset == 'coco':

        if parser.coco_path is None:
            raise ValueError('Must provide --coco_path when training on COCO,')

        dataset_train = CocoDataset(parser.coco_path, set_name='train2017',
                                    transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))
        dataset_val = CocoDataset(parser.coco_path, set_name='val2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))

    elif parser.dataset == 'csv':

        if parser.csv_train is None:
            raise ValueError('Must provide --csv_train when training on COCO,')

        if parser.csv_classes is None:
            raise ValueError('Must provide --csv_classes when training on COCO,')

        dataset_train = CSVDataset(train_file=parser.csv_train, class_list=parser.csv_classes,
                                   transform=transforms.Compose([Normalizer(), Augmenter(), Resizer()]))

        if parser.csv_val is None:
            dataset_val = None
            print('No validation annotations provided.')
        else:
            dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                     transform=transforms.Compose([Normalizer(), Resizer()]))

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler = AspectRatioBasedSampler(dataset_train, batch_size=2, drop_last=False, shuffle=True)
    dataloader_train = DataLoader(dataset_train, num_workers=5, collate_fn=collater, batch_sampler=sampler)

    if dataset_val is not None:
        sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False, shuffle=True)
        dataloader_val = DataLoader(dataset_val, num_workers=3, collate_fn=collater, batch_sampler=sampler_val)

    use_gpu = True
    retinanet = torch.load(parser.model)

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.training = True

    optimizer = optim.Adam(retinanet.parameters(), lr=1e-8)

    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=0, verbose=True)
    # scheduler = optim.lr_scheduler.CyclicLR(optimizer, 2.25e-6, 9e-6, step_size_up=2763,
    #                                         mode='triangular', cycle_momentum=False, last_epoch=- 1, verbose=True)

    # loss_hist = collections.deque(maxlen=500)

    retinanet.train()
    retinanet.module.freeze_bn()

    print('Num training images: {}'.format(len(dataset_train)))

    for epoch_num in range(parser.epochs):

        retinanet.train()

        epoch_loss = []
        epoch_class_loss = []
        epoch_regr_loss = []

        for iter_num, data in enumerate(dataloader_train):

            # Resetting gradients after each iter
            optimizer.zero_grad()

            # Forward
            classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot'].cuda().float()])

            # Calculating Loss
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()

            loss = classification_loss + regression_loss

            if bool(loss == 0):
                continue

            # Calculating Gradients
            loss.backward()

            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(retinanet.parameters(), 0.1)

            # # weights update
            # if ((iter_num + 1) % 1 == 0) or (iter_num + 1 == len(dataloader_train)):
            #     optimizer.step()
            #     optimizer.zero_grad()

            # Updating Weights
            optimizer.step()

            # Epoch Loss
            epoch_loss.append(float(loss))
            epoch_class_loss.append(float(classification_loss))
            epoch_regr_loss.append(float(regression_loss))

            # # Update the learning rate
            # if scheduler is not None:
            #     scheduler.step()

            writer.add_scalars(f'iteration_loss/all', {
                'total': loss,
                'classification': classification_loss,
                'regression': regression_loss
            }, iter_num + epoch_num * 12041)

            print(
                'Epoch: {}/{} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | Running '
                'loss: {:1.5f}'.format(
                    epoch_num + 1, parser.epochs, iter_num + 1, len(dataloader_train), float(classification_loss),
                    float(regression_loss), float(np.mean(epoch_loss))))

            del classification_loss
            del regression_loss

        epoch_loss_val = []
        epoch_class_loss_val = []
        epoch_regr_loss_val = []

        for iter_num, data in enumerate(dataloader_val):
            with torch.no_grad():
                # Forward
                classification_loss, regression_loss = retinanet(
                    [data['img'].cuda().float(), data['annot'].cuda().float()])

                # Calculating Loss
                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()
                loss = classification_loss + regression_loss

                # Epoch Loss
                epoch_loss_val.append(float(loss))
                epoch_class_loss_val.append(float(classification_loss))
                epoch_regr_loss_val.append(float(regression_loss))

                print(
                    'Epoch: {}/{} | Iteration: {}/{} | Classification loss: {:1.5f} | Regression loss: {:1.5f} | '
                    'Running loss: {:1.5f}'.format(
                        epoch_num + 1, parser.epochs, iter_num + 1, len(dataloader_val), float(classification_loss),
                        float(regression_loss), np.mean(epoch_loss_val)))

                del classification_loss
                del regression_loss

        # Update the learning rate
        if scheduler is not None:
            scheduler.step(np.mean(epoch_loss_val))

        writer.add_scalars(f'total_loss/all', {
            'training': np.mean(epoch_loss),
            'validation': np.mean(epoch_loss_val)
        }, epoch_num)

        writer.add_scalars(f'classification_loss/all', {
            'training': np.mean(epoch_class_loss),
            'validation': np.mean(epoch_class_loss_val)
        }, epoch_num)

        writer.add_scalars(f'regression_loss/all', {
            'training': np.mean(epoch_regr_loss),
            'validation': np.mean(epoch_regr_loss_val)
        }, epoch_num)

        # Save Model after each epoch
        torch.save(retinanet.module, '{}_retinanet_{}.pt'.format(parser.dataset, epoch_num))

        print(optimizer.param_groups[0]['lr'])

    torch.save(retinanet, 'model_final.pt')

    writer.flush()


if __name__ == '__main__':
    main()
