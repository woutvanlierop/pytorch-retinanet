from pathlib import Path

import numpy as np
import pandas as pd
import time
import argparse

import cv2

import torch
from matplotlib import pyplot as plt
from pascal import PascalVOC
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from retinanet.dataloader import CocoDataset, CSVDataset, collater, Resizer, AspectRatioBasedSampler, \
    UnNormalizer, Normalizer

assert torch.__version__.split('.')[0] == '1'
ds = Path("C:/Users/woutv/PycharmProjects/thesis/one_image/annotation_bigger/")
img_path = Path("C:/Users/woutv/PycharmProjects/thesis/one_image/images_bigger/")

print('CUDA available: {}'.format(torch.cuda.is_available()))
# df = pd.read_csv('C:/Users/woutv/PycharmProjects/thesis/number_of_kernels.csv')

counts = []
groundTruthCounts = []
smallboxsize_all = []
mediumboxsize_all = []
bigboxsize_all = []
smallboxcount_all = []
mediumboxcount_all = []
bigboxcount_all = []


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


    groundTruth_box_sizes_all = []

    min_axis_gt = []

    for idx, data in tqdm(enumerate(dataloader_val)):
        box_sizes_all = []
        min_axis = []
        with torch.no_grad():
            st = time.time()
            if torch.cuda.is_available():
                scores, classification, transformed_anchors = retinanet(data['img'].cuda().float())
            else:
                scores, classification, transformed_anchors = retinanet(data['img'].float())
            # print('Elapsed time: {}'.format(time.time() - st))
            idxs = np.where(scores.cpu() > 0.25)
            img = np.array(255 * unnormalize(data['img'][0, :, :, :])).copy()

            img[img < 0] = 0
            img[img > 255] = 255

            img = np.transpose(img, (1, 2, 0))

            img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2RGB)
            imagepath = dataset_val.image_names[idx]
            # xmlpath = "C:/Users/woutv/OneDrive - KU Leuven/thesisData/test_new_bigger/annots/" + imagepath[76:][:-4] + ".xml"
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
                box_size = (x2 - x1) * (y2 - y1)
                box_size = box_size*(0.06*0.06)
                box_size = round(box_size)
                box_sizes_all.append(box_size)
                min_axis.append(min((x2 - x1), (y2 - y1))*0.06)

                cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 0, 255), thickness=2)
                count = count + 1

            counts.append(count)
            # box_sizes_all.append(box_sizes)

            # get ground truth box distribution
            groundTruthCount = 0
            groundTruth_box_sizes = []

            # ann = PascalVOC.from_xml(xmlpath)
            # for obj in ann.objects:
            #     groundTruthCount = groundTruthCount + 1
            #     box_size = (obj.bndbox.xmax - obj.bndbox.xmin)*(obj.bndbox.ymax - obj.bndbox.ymin)
            #     box_size = box_size*(0.06*0.06)
            #     groundTruth_box_sizes_all.append(box_size)
            #     min_axis_gt.append(min(obj.bndbox.xmax - obj.bndbox.xmin, obj.bndbox.ymax - obj.bndbox.ymin)*0.06)

            groundTruthCounts.append(groundTruthCount)
            # groundTruth_box_sizes_all.append(groundTruth_box_sizes)
            # print(dataset_val.image_names[idx])
            # print(str(count))

        # df = pd.DataFrame(dataset_val.image_names)
        # df['count'] = counts
        # df['groundTruthcount'] = groundTruthCounts
        #
        # df2 = pd.DataFrame()
        # df2['box_size'] = box_sizes_all
        #
        # df3 = pd.DataFrame()
        # df3['groundTruth_box_size'] = groundTruth_box_sizes_all
        #
        # df4 = pd.DataFrame()
        # df4['min_axis'] = min_axis
        #
        # df5 = pd.DataFrame()
        # df5['min_axis_gt'] = min_axis_gt

        # bins = np.histogram(np.hstack((df4['min_axis'], df5['min_axis_gt'])), bins=20)[1]
        # plt.hist(df4['min_axis'], bins=bins, edgecolor='black', color='red', alpha=0.25)
        # plt.hist(df5['min_axis_gt'], bins=bins, edgecolor='black', color='green', alpha=0.25)
        # plt.title("Histogram for min axis.")
        # plt.xlabel("Min. axis in mm.")
        # plt.ylabel("Number of kernels.")
        # plt.show()

        smallboxsize = 0
        smallboxcount = 0
        mediumboxsize = 0
        mediumboxcount = 0
        bigboxsize = 0
        bigboxcount = 0

        for idx in range(len(box_sizes_all)):
            if min_axis[idx] < 1.18:
                smallboxsize = smallboxsize + box_sizes_all[idx]
                smallboxcount += 1
            if 1.18 <= min_axis[idx] < 4.75:
                mediumboxsize = mediumboxsize + box_sizes_all[idx]
                mediumboxcount += 1
            if min_axis[idx] >= 4.75:
                bigboxsize = bigboxsize + box_sizes_all[idx]
                bigboxcount += 1
        smallboxsize_all.append(smallboxsize)
        mediumboxsize_all.append(mediumboxsize)
        bigboxsize_all.append(bigboxsize)
        smallboxcount_all.append(smallboxcount)
        mediumboxcount_all.append(mediumboxcount)
        bigboxcount_all.append(bigboxcount)

        # print("small: " + str(smallboxsize))
        # print("medium: " + str(mediumboxsize))
        # print("big: " + str(bigboxsize))

        # smallboxsize_gt = 0
        # mediumboxsize_gt = 0
        # bigboxsize_gt = 0
        # for idx in range(len(groundTruth_box_sizes_all)):
        #     if min_axis_gt[idx] < 1.18:
        #         smallboxsize_gt = smallboxsize_gt + groundTruth_box_sizes_all[idx]
        #     if 1.18 <= min_axis_gt[idx] < 4.75:
        #         mediumboxsize_gt = mediumboxsize_gt + groundTruth_box_sizes_all[idx]
        #     if min_axis_gt[idx] >= 4.75:
        #         bigboxsize_gt = bigboxsize_gt + groundTruth_box_sizes_all[idx]
        #
        # print("groundtruth:")
        # print("small: " + str(smallboxsize_gt))
        # print("medium: " + str(mediumboxsize_gt))
        # print("big: " + str(bigboxsize_gt))

    # bins = np.histogram(np.hstack((df2['box_size'], df3['groundTruth_box_size'])), bins=20)[1]
    # plt.hist(df2['box_size'], bins=bins, edgecolor='black', color='red', alpha=0.25)
    # plt.hist(df3['groundTruth_box_size'], bins=bins, edgecolor='black', color='green', alpha=0.25)
    # plt.title("Histogram of box area.")
    # plt.xlabel("Area in mm.")
    # plt.ylabel("Number of kernels.")
    # plt.show()
    # plt.hist(df4['min_axis'], bins=[0, 1.18, 4.75, 20], edgecolor='black', color='red', alpha=0.25)
    # plt.hist(df5['min_axis_gt'], bins=[0, 1.18, 4.75, 20], edgecolor='black', color='green', alpha=0.25)
    # plt.title("Histogram of min. axis per sieve size")
    # plt.xlabel("Min. axis in mm.")
    # plt.ylabel("Number of kernels.")
    # plt.show()
    # print(box_sizes_all)
    # print(counts)
    # df.to_csv(path_or_buf="C:/Users/woutv/PycharmProjects/thesis/number_of_detections_test.csv", index=False)

    df_images_wo_annot = pd.DataFrame()
    df_images_wo_annot['image_names'] = dataset_val.image_names
    df_images_wo_annot['area_small'] = smallboxsize_all
    df_images_wo_annot['area_medium'] = mediumboxsize_all
    df_images_wo_annot['area_large'] = bigboxsize_all
    df_images_wo_annot['box_small'] = smallboxcount_all
    df_images_wo_annot['box_medium'] = mediumboxcount_all
    df_images_wo_annot['box_large'] = bigboxcount_all
    df_images_wo_annot.to_csv('result.csv', index=False)

if __name__ == '__main__':
    main()
