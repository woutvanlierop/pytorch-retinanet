import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

train_runs = [7, 9, 10, 13, 49, 53, 81, 89, 101, 102, 118, 124, 139, 147, 155, 164, 171, 175, 180, 192, 195, 200, 213,
              232, 246, 247, 253, 254, 255, 269, 277, 280, 304, 305, 307, 308, 330, 334, 335]
valid_runs = [12, 68, 98, 114, 117, 153, 157, 169, 212, 283, 323]
test_runs = [45, 63, 95, 224, 295]


def xml_to_csv(path):
    xml_list = []
    for number in train_runs:
        for xml_file in glob.glob(path + '/run' + str(number) + '/*0.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                bbx = member.find('bndbox')
                xmin = int(bbx.find('xmin').text)
                ymin = int(bbx.find('ymin').text)
                xmax = int(bbx.find('xmax').text)
                ymax = int(bbx.find('ymax').text)
                path_to_image = xml_file[:-3] + 'png'
                path_to_image = path_to_image.replace('\\', '/')

                value = (path_to_image,
                         xmin,
                         ymin,
                         xmax,
                         ymax,
                         "kernel")
                xml_list.append(value)
        column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv('train_labels.csv', index=False, header=False)
    xml_list = []
    for number in valid_runs:
        for xml_file in glob.glob(path + '/run' + str(number) + '/*0.xml'):
            single_list = []
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                bbx = member.find('bndbox')
                xmin = int(bbx.find('xmin').text)
                ymin = int(bbx.find('ymin').text)
                xmax = int(bbx.find('xmax').text)
                ymax = int(bbx.find('ymax').text)
                path_to_image = xml_file[:-3] + 'png'
                path_to_image = path_to_image.replace('\\', '/')

                value = (path_to_image,
                         xmin,
                         ymin,
                         xmax,
                         ymax,
                         "kernel")
                xml_list.append(value)
                single_list.append(value)
            column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
            single_df = pd.DataFrame(single_list, columns=column_name)
            single_df.to_csv(path_or_buf='C:/Users/woutv/PycharmProjects/thesis/valid_csv/' + str(path_to_image[63:-4]) + '.csv', index=False, header=False)
        column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv('valid_labels.csv', index=False, header=False)
    xml_list = []
    for number in test_runs:
        for xml_file in glob.glob(path + '/run' + str(number) + '/*0.xml'):
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for member in root.findall('object'):
                bbx = member.find('bndbox')
                xmin = int(bbx.find('xmin').text)
                ymin = int(bbx.find('ymin').text)
                xmax = int(bbx.find('xmax').text)
                ymax = int(bbx.find('ymax').text)
                path_to_image = xml_file[:-3] + 'png'
                path_to_image = path_to_image.replace('\\', '/')

                value = (path_to_image,
                         xmin,
                         ymin,
                         xmax,
                         ymax,
                         "kernel")
                xml_list.append(value)
        column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'class']
        xml_df = pd.DataFrame(xml_list, columns=column_name)
        xml_df.to_csv('test_labels.csv', index=False, header=False)


def main(path):
    image_path = os.path.join(path)
    xml_to_csv(image_path)
    print('Successfully converted xml to csv.')


main("C:/Users/woutv/OneDrive - KU Leuven/labeling Italy 2021")