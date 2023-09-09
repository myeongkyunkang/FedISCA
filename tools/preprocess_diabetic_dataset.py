import os
from shutil import copy2


def read_label(csv_path):
    data_list = []
    with open(csv_path) as f:
        f.readline()  # remove header
        while True:
            line = f.readline()
            if len(line) == 0:
                break
            line_split = line.strip().split(',')
            data_list.append((f'{line_split[0]}.jpg', int(line_split[1])))
    return data_list


if __name__ == '__main__':
    result_dir = './dataset/diabetic2015'
    target_size = (224, 224)

    data_dir = './dataset/Resized 2015 & 2019 Blindness Detection Images'
    train_data_dir = os.path.join(data_dir, 'resized train 15_224')
    test_data_dir = os.path.join(data_dir, 'resized test 15_224')

    label_dir = os.path.join(data_dir, 'labels')
    train_label_path = os.path.join(label_dir, 'trainLabels15.csv')
    test_label_path = os.path.join(label_dir, 'testLabels15.csv')

    result_train_dir = os.path.join(result_dir, 'train')
    result_test_dir = os.path.join(result_dir, 'test')

    train_list = read_label(train_label_path)
    test_list = read_label(test_label_path)

    for filename, label in train_list:
        img_path = os.path.join(train_data_dir, filename)
        save_path = os.path.join(result_train_dir, str(label), filename)
        os.makedirs(os.path.join(result_train_dir, str(label)), exist_ok=True)

        if os.path.isfile(img_path):
            copy2(img_path, save_path)
        else:
            print('skip:', img_path)

    for filename, label in test_list:
        img_path = os.path.join(test_data_dir, filename)
        save_path = os.path.join(result_test_dir, str(label), filename)
        os.makedirs(os.path.join(result_test_dir, str(label)), exist_ok=True)

        if os.path.isfile(img_path):
            copy2(img_path, save_path)
        else:
            print('skip:', img_path)
