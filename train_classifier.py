import argparse
import os
import random

import medmnist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
from sklearn import metrics
from sklearn.utils import class_weight
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from dataset_isic2019 import FedIsic2019
from models import get_model_heter, get_model_heter_224
from models.resnet_cifar import ResNet18
from utils import DatasetSplit, ImageDataset, adjust_learning_rate, partition_data


def main(args):
    # set seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    if args.dataset in medmnist.INFO:
        info = medmnist.INFO[args.dataset]

        n_channels = info['n_channels']
        n_classes = len(info['label'])
        epochs = 100
        lr_step1 = 50
        lr_step2 = 75
        lr_init = 0.001

        DataClass = getattr(medmnist, info['python_class'])

        # check valid dataset
        if 'multi-class' != info['task']:
            raise ValueError("Invalid Task")

        # preprocessing
        aug_list = []
        if args.aug:
            aug_list.append(transforms.RandomCrop(28, padding=4))
            aug_list.append(transforms.RandomHorizontalFlip())
        preprocess_list = [transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]

        transform_train = transforms.Compose(aug_list + preprocess_list)
        transform_test = transforms.Compose(preprocess_list)

        # load the data
        med_train_data = DataClass(split='train', root=os.path.join(args.root, 'medmnist'))
        med_val_data = DataClass(split='val', root=os.path.join(args.root, 'medmnist'))

        data_images_merge = np.concatenate([med_train_data.imgs, med_val_data.imgs])
        data_targets_merge = np.concatenate([med_train_data.labels, med_val_data.labels])

        data_train = ImageDataset(data_images_merge, data_targets_merge, transform=transform_train)
        data_val = ImageDataset(data_images_merge, data_targets_merge, transform=transform_test)

        y_train = np.array(data_train.targets)

    elif args.dataset == 'isic2019':
        aug_list = []
        if args.aug:
            aug_list.append(transforms.RandomAffine(50, shear=0.1))
            aug_list.append(transforms.RandomResizedCrop(224))
            aug_list.append(transforms.RandomHorizontalFlip())
            aug_list.append(transforms.ColorJitter(brightness=0.15, contrast=0.1))
        preprocess_list = [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]

        transform_train = transforms.Compose(aug_list + preprocess_list)
        transform_test = transforms.Compose(preprocess_list)

        n_channels = 3
        n_classes = 8
        epochs = 100
        lr_step1 = 50
        lr_step2 = 75
        lr_init = 0.001
        args.num_users = 6
        print('reset num_users to', args.num_users)

    elif args.dataset == 'isic2019_merge':
        aug_list = []
        if args.aug:
            aug_list.append(transforms.RandomAffine(50, shear=0.1))
            aug_list.append(transforms.RandomResizedCrop(224))
            aug_list.append(transforms.RandomHorizontalFlip())
            aug_list.append(transforms.ColorJitter(brightness=0.15, contrast=0.1))
        preprocess_list = [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]

        transform_train = transforms.Compose(aug_list + preprocess_list)
        transform_test = transforms.Compose(preprocess_list)

        # set dataset
        data_train = FedIsic2019(center=-1, split='train', transform=transform_train, data_path=os.path.join(args.root, 'fed_isic2019'), val_rate=args.val_rate)
        data_val = FedIsic2019(center=-1, split='val', transform=transform_test, data_path=os.path.join(args.root, 'fed_isic2019'), val_rate=args.val_rate)

        n_channels = 3
        n_classes = 8
        epochs = 100
        lr_step1 = 50
        lr_step2 = 75
        lr_init = 0.001

        y_train = np.array(data_train.targets)

    elif args.dataset == 'diabetic2015':
        aug_list = []
        if args.aug:
            aug_list.append(transforms.RandomAffine(50))
            aug_list.append(transforms.RandomResizedCrop(224))
            aug_list.append(transforms.RandomHorizontalFlip())
            aug_list.append(transforms.ColorJitter(brightness=0.15, contrast=0.1))
        preprocess_list = [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]

        transform_train = transforms.Compose(aug_list + preprocess_list)
        transform_test = transforms.Compose(preprocess_list)

        data_train = ImageFolder(os.path.join(args.root, 'diabetic2015', 'train'), transform=transform_train)  # sorted
        data_val = ImageFolder(os.path.join(args.root, 'diabetic2015', 'train'), transform=transform_test)  # sorted

        n_channels = 3
        n_classes = 5
        epochs = 100
        lr_step1 = 50
        lr_step2 = 75
        lr_init = 0.001

        y_train = np.array(data_train.targets)

    elif args.dataset == 'rsna':
        aug_list = []
        if args.aug:
            aug_list.append(transforms.RandomResizedCrop(224))
            aug_list.append(transforms.RandomHorizontalFlip())
            aug_list.append(transforms.ColorJitter(brightness=0.15, contrast=0.1))
        preprocess_list = [transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])]

        transform_train = transforms.Compose(aug_list + preprocess_list)
        transform_test = transforms.Compose(preprocess_list)

        data_train = ImageFolder(os.path.join(args.root, 'rsna', 'train'), transform=transform_train)  # sorted
        data_val = ImageFolder(os.path.join(args.root, 'rsna', 'train'), transform=transform_test)  # sorted

        n_channels = 3
        n_classes = 2
        epochs = 100
        lr_step1 = 50
        lr_step2 = 75
        lr_init = 0.001

        y_train = np.array(data_train.targets)

    else:
        raise ValueError(f'Invalid Dataset: {args.dataset}')

    # ========================================
    # make partition
    if args.dataset != 'isic2019':
        user_groups = partition_data(y_train, n_classes, partition=args.partition, beta=args.beta, num_users=args.num_users)

        train_user_groups, val_user_groups = [], []
        for user_group in user_groups.values():
            train_user_groups.append(user_group[int(len(user_group) * args.val_rate):])
            val_user_groups.append(user_group[:int(len(user_group) * args.val_rate)])
    # ========================================

    for user_idx in range(args.num_users):
        output_dir = os.path.join(args.output_dir, f'{args.dataset}_{args.partition}_{args.num_users}_{args.beta}', f'client_{user_idx}')

        # make output dir
        os.makedirs(output_dir, exist_ok=True)
        if os.path.isfile(os.path.join(output_dir, 'val.csv')):
            os.remove(os.path.join(output_dir, 'val.csv'))

        # define model
        if args.dataset == 'isic2019':
            net = get_model_heter_224(user_idx, num_classes=n_classes).cuda() if args.model_heter else get_model_heter_224(0, num_classes=n_classes).cuda()

            # set dataset
            data_train_loader = DataLoader(
                FedIsic2019(center=user_idx, split='train', transform=transform_train, data_path=os.path.join(args.root, 'fed_isic2019'), val_rate=args.val_rate),
                batch_size=args.bs, shuffle=True, num_workers=8)
            data_val_loader = DataLoader(
                FedIsic2019(center=user_idx, split='val', transform=transform_test, data_path=os.path.join(args.root, 'fed_isic2019'), val_rate=args.val_rate),
                batch_size=args.bs, shuffle=True, num_workers=8)

            # calculate class_weights
            targets = data_train_loader.dataset.targets
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=list(range(0, n_classes)), y=np.array(targets + list(range(0, n_classes))))  # apply smoothing
            class_weights = torch.tensor(class_weights, dtype=torch.float)

        elif args.dataset in ['diabetic2015', 'isic2019_merge', 'rsna']:
            net = get_model_heter_224(user_idx, num_classes=n_classes).cuda() if args.model_heter else get_model_heter_224(0, num_classes=n_classes).cuda()

            # set dataset
            data_train_loader = DataLoader(DatasetSplit(data_train, train_user_groups[user_idx]), batch_size=args.bs, shuffle=True, num_workers=8)
            data_val_loader = DataLoader(DatasetSplit(data_val, val_user_groups[user_idx]), batch_size=args.bs, shuffle=True, num_workers=8)

            # calculate class_weights
            targets = data_train.targets
            class_weights = class_weight.compute_class_weight(class_weight='balanced', classes=list(range(0, n_classes)), y=np.array(targets + list(range(0, n_classes))))  # apply smoothing
            class_weights = torch.tensor(class_weights, dtype=torch.float)

        else:
            net = get_model_heter(user_idx, in_channels=n_channels, num_classes=n_classes).cuda() if args.model_heter else ResNet18(in_channels=n_channels, num_classes=n_classes).cuda()

            # set dataset
            data_train_loader = DataLoader(DatasetSplit(data_train, train_user_groups[user_idx]), batch_size=args.bs, shuffle=True, num_workers=8)
            data_val_loader = DataLoader(DatasetSplit(data_val, val_user_groups[user_idx]), batch_size=args.bs, shuffle=True, num_workers=8)
            class_weights = None

        # define loss and optim
        criterion = torch.nn.CrossEntropyLoss(weight=class_weights).cuda()
        optimizer = torch.optim.SGD(net.parameters(), lr=lr_init, momentum=0.9, weight_decay=5e-4)

        acc_best = 0
        for e in range(1, epochs + 1):
            # ========================================
            # Train
            adjust_learning_rate(optimizer, e, lr_init=lr_init, lr_step1=lr_step1, lr_step2=lr_step2)

            net.train()
            loss_list = []
            for i, (images, labels) in enumerate(data_train_loader):
                images, labels = Variable(images).cuda(), Variable(labels.flatten().long() if len(labels.shape) == 2 else labels.long()).cuda()
                optimizer.zero_grad()
                output = net(images)
                loss = criterion(output, labels)
                loss.backward()
                optimizer.step()
                loss_list.append(loss.data.item())
                if i == 1:
                    print('Train - Epoch %d, Batch: %d, Loss: %f' % (e, i, loss.data.item()))
            # ========================================

            # ========================================
            # Val
            net.eval()
            total_correct, num_samples = 0, 0
            avg_loss = 0.0
            gt_list, pred_list = [], []
            with torch.no_grad():
                for i, (images, labels) in enumerate(data_val_loader):
                    images, labels = Variable(images).cuda(), Variable(labels.flatten().long() if len(labels.shape) == 2 else labels.long()).cuda()
                    output = net(images)
                    avg_loss += criterion(output, labels).sum()
                    pred = output.data.max(1)[1]
                    total_correct += pred.eq(labels.data.view_as(pred)).sum()
                    num_samples += images.shape[0]
                    gt_list.extend(labels.tolist())
                    pred_list.extend(pred.tolist())

            avg_loss /= num_samples
            acc = float(total_correct) / num_samples
            b_acc = metrics.balanced_accuracy_score(gt_list, pred_list)
            print('Val Avg. Loss: %f, Accuracy: %f, Balanced Accuracy: %f' % (avg_loss.data.item(), acc, b_acc))

            # use balanced accuracy instead of accuracy
            if args.dataset in ['isic2019', 'diabetic2015']:
                acc = b_acc

            # write log
            with open(os.path.join(output_dir, 'val.csv'), 'at') as wf:
                wf.write('{},{:.4f}\n'.format(e, acc))
            # ========================================

            if acc_best < acc:
                acc_best = acc

                # save best
                torch.save(net, os.path.join(output_dir, 'best.pth'))

            # save last
            torch.save(net, os.path.join(output_dir, 'last.pth'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='train-teacher-network')

    # Basic model parameters.
    parser.add_argument('--dataset', default='bloodmnist', type=str)
    parser.add_argument('--output_dir', default='./pretrained_models', type=str)
    parser.add_argument('--root', default='./dataset', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--bs', default=128, type=int)
    parser.add_argument('--aug', action='store_true')
    parser.add_argument('--partition', default='dirichlet', type=str)
    parser.add_argument('--beta', default=0.6, type=float)
    parser.add_argument('--num_users', default=5, type=int)
    parser.add_argument('--val_rate', default=0.1, type=float)
    parser.add_argument('--model_heter', action='store_true')
    parser.add_argument('--pretrained', action='store_true')

    args = parser.parse_args()

    if args.model_heter:
        assert args.num_users == 5

    if args.dataset == 'diabetic2015':
        assert args.partition == 'iid'

    main(args)
