import pandas as pd

import numpy as np
import time
import argparse

import cv2

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, \
    UnNormalizer, Normalizer

assert torch.__version__.split('.')[0] == '1'

print('CUDA available: {}'.format(torch.cuda.is_available()))
# df = pd.read_csv('C:/Users/woutv/PycharmProjects/thesis/number_of_kernels.csv')

counts = []


def main(args=None):
    parser = argparse.ArgumentParser(description='Simple training script for training a RetinaNet network.')

    parser.add_argument('--dataset', help='Dataset type, must be one of csv or coco.')
    parser.add_argument('--coco_path', help='Path to COCO directory')
    parser.add_argument('--csv_classes', help='Path to file containing class list (see readme)')
    parser.add_argument('--csv_val', help='Path to file containing validation annotations (optional, see readme)')

    parser.add_argument('--model', help='Path to model (.pt) file.')

    parser = parser.parse_args(args)

    if parser.dataset == 'coco':
        dataset_val = CocoDataset(parser.coco_path, set_name='train2017',
                                  transform=transforms.Compose([Normalizer(), Resizer()]))
    elif parser.dataset == 'csv':
        dataset_val = CSVDataset(train_file=parser.csv_val, class_list=parser.csv_classes,
                                 transform=transforms.Compose([Normalizer(), Resizer()]))
    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    sampler_val = AspectRatioBasedSampler(dataset_val, batch_size=1, drop_last=False, shuffle=False)
    dataloader_val = DataLoader(dataset_val, num_workers=1, collate_fn=collater, batch_sampler=sampler_val)

    retinanet = torch.load(parser.model)

    use_gpu = True

    if use_gpu:
        if torch.cuda.is_available():
            retinanet = retinanet.cuda()

    if torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)

    retinanet.eval()

    unnormalize = UnNormalizer()

    def draw_caption(image, box, caption):

        b = np.array(box).astype(int)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2)
        cv2.putText(image, caption, (b[0], b[1] - 10), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1)

    box_sizes_all = []

    for idx, data in enumerate(dataloader_val):

        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.25)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            name = dataset_val.image_names[idx]
            img_2 = cv2.imread('C:/Users/woutv/PycharmProjects/thesis/one_image/green_images/' + name[59:])
            # img_2 = cv2.resize(img_2, (1544, 3752))
            count = 0
            box_sizes = []

            for j in range(idxs[0].shape[0]):
                bbox = transformed_anchors[idxs[0][j], :]
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])
                label_name = dataset_val.labels[int(classification[idxs[0][j]])]
                draw_caption(img, (x1, y1, x2, y2), label_name)
                draw_caption(img_2, (x1, y1, x2, y2), label_name)
                box_size = (x2 - x1) * (y2 - y1)
                box_sizes.append(box_size)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                cv2.rectangle(img_2, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=4)
                count = count + 1
            # print(label_name)

            print(dataset_val.image_names[idx])
            print(str(count))
            # print(box_size / len(range(idxs[0].shape[0])))
            # box_sizes.append(box_size / len(range(idxs[0].shape[0])))
            counts.append(count)
            box_sizes_all.append(box_sizes)

            # df.loc[len(df.index)] = [dataset_val.image_names[idx], count]
        # cv2.imwrite('C:/Users/woutv/PycharmProjects/thesis/one_image/2' + name[59:], img)
        cv2.imwrite('C:/Users/woutv/PycharmProjects/thesis/one_image/' + name[59:], img_2)
        # print(img.shape)

        # cv2.imshow('img', img)
        # cv2.waitKey(0)

        # df.to_csv(path_or_buf="C:/Users/woutv/PycharmProjects/thesis/number_of_detections.csv", index=False)
    # df = pd.DataFrame(dataset_val.image_names)
    # df['box_size'] = box_sizes
    # df['count'] = counts
    print(box_sizes_all)
    print(counts)
    # df.to_csv(path_or_buf="C:/Users/woutv/PycharmProjects/thesis/number_of_detections_test.csv", index=False)


if __name__ == '__main__':
    main()
