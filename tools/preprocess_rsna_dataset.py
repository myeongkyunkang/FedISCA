import os
import random

import pydicom as dcm
from PIL import Image
from tqdm import tqdm


def read_label(csv_path):
    data_list = []
    with open(csv_path) as f:
        f.readline()  # remove header
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line_split = line.strip().split(',')
            filename = f'{line_split[0]}.dcm'
            label = 0 if line_split[1] == 'Normal' else 1
            data_list.append((filename, label))
    return data_list


def save(label_list, dcm_dir, save_dir):
    for filename, label in tqdm(label_list):
        dcm_path = os.path.join(dcm_dir, filename)

        # read dcm
        ds = dcm.dcmread(dcm_path)
        img = Image.fromarray(ds.pixel_array, 'L').convert('RGB')
        img = img.resize(target_size, resample=Image.BILINEAR)

        img.save(os.path.join(save_dir, str(label), filename.replace('.dcm', '.png')))


if __name__ == '__main__':
    target_size = (224, 224)

    result_dir = './dataset/rsna'
    result_train_dir = os.path.join(result_dir, 'train')
    result_test_dir = os.path.join(result_dir, 'test')
    os.makedirs(os.path.join(result_train_dir, '0'))
    os.makedirs(os.path.join(result_train_dir, '1'))
    os.makedirs(os.path.join(result_test_dir, '0'))
    os.makedirs(os.path.join(result_test_dir, '1'))

    data_dir = './dataset/RSNA_Pneumonia'
    dcm_dir = os.path.join(data_dir, 'stage_2_train_images')
    csv_path = os.path.join(data_dir, 'stage_2_detailed_class_info.csv')

    data_label_list = read_label(csv_path)
    random.Random(1).shuffle(data_label_list)

    test_num = int(len(data_label_list) * 0.1)
    train_label_list = data_label_list[test_num:]
    test_label_list = data_label_list[:test_num]

    save(train_label_list, dcm_dir, result_train_dir)
    save(test_label_list, dcm_dir, result_test_dir)
