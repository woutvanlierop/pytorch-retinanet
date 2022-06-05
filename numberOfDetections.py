import glob
import os


for file in glob.glob('C:/Users/woutv/PycharmProjects/thesis/valid_csv/*.csv'):
    file = file.replace('\\', '/')
    print(file)
    os.system("python visualize.py --dataset csv --csv_classes C:/Users/woutv/PycharmProjects/thesis/class_list.csv  --csv_val "+ file + " --model C:/Users/woutv/PycharmProjects/retinanet/csv_retinanet_39.pt")

