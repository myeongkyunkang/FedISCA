import argparse
import collections
import copy
import gc
import os
import random

import medmnist
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.hub import load_state_dict_from_url
from torch.utils.data import Dataset
from torchvision.models.resnet import resnet18

from models import get_model_heter, get_model_heter_224
from models.resnet_cifar import ResNet18
from utils import KLDiv, test, adjust_learning_rate, DeepInversionFeatureHook, Ensemble


def main(args):
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # make exp directory
    img_exp_descr = os.path.join(args.exp_descr, 'img')
    best_img_exp_descr = os.path.join(img_exp_descr, 'best')
    os.makedirs(best_img_exp_descr, exist_ok=True)
    if os.path.isfile(os.path.join(args.exp_descr, 'test.csv')):
        os.remove(os.path.join(args.exp_descr, 'test.csv'))

    if args.dataset in medmnist.INFO:
        info = medmnist.INFO[args.dataset]
        DataClass = getattr(medmnist, info['python_class'])

        # preprocessing
        transform_test = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

        # load the data
        data_test = DataClass(split='test', transform=transform_test, root=os.path.join(args.root, 'medmnist'))
        data_test_loader = torch.utils.data.DataLoader(dataset=data_test, batch_size=args.bs, shuffle=False, num_workers=8)

        input_size = 28
        n_channels = info['n_channels']
        n_classes = len(info['label'])

        epochs = 100
        lr_step1 = 50
        lr_step2 = 75
        lr_init = 0.001

    elif args.dataset == 'isic2019':
        from dataset_isic2019 import FedIsic2019

        # preprocessing
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

        data_test_loader = torch.utils.data.DataLoader(
            FedIsic2019(split='test', data_path=os.path.join(args.root, 'fed_isic2019'), transform=transform_test),
            batch_size=args.bs, shuffle=True, num_workers=8)

        input_size = 224
        n_channels = 3
        n_classes = 8

        epochs = 100
        lr_step1 = 50
        lr_step2 = 75
        lr_init = 0.001

    elif args.dataset == 'diabetic2015':
        from torchvision.datasets import ImageFolder

        # preprocessing
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

        data_test_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(args.root, 'diabetic2015', 'test'), transform=transform_test),
            batch_size=args.bs, shuffle=True, num_workers=8)

        input_size = 224
        n_channels = 3
        n_classes = 5

        epochs = 100
        lr_step1 = 50
        lr_step2 = 75
        lr_init = 0.001

    elif args.dataset == 'rsna':
        from torchvision.datasets import ImageFolder

        # preprocessing
        transform_test = transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[.5], std=[.5])])

        data_test_loader = torch.utils.data.DataLoader(
            ImageFolder(os.path.join(args.root, 'rsna', 'test'), transform=transform_test),
            batch_size=args.bs, shuffle=True, num_workers=8)

        input_size = 224
        n_channels = 3
        n_classes = 2

        epochs = 100
        lr_step1 = 50
        lr_step2 = 75
        lr_init = 0.001

    else:
        raise ValueError(f'Invalid Dataset: {args.dataset}')

    if os.path.isfile(args.teacher_weights):
        # define networks
        net_teacher = resnet18(num_classes=n_classes) if args.dataset in ['isic2019', 'diabetic2015', 'rsna'] else ResNet18(in_channels=n_channels, num_classes=n_classes)
        net_teacher = net_teacher.to(device)

        # load checkpoint
        checkpoint = torch.load(args.teacher_weights)
        net_teacher.load_state_dict(checkpoint.state_dict())
        net_teacher.eval()
    elif os.path.isdir(args.teacher_weights):
        model_list = []
        for client_dir in sorted(os.listdir(args.teacher_weights)):
            weight_path = os.path.join(args.teacher_weights, client_dir, 'best.pth')

            if not os.path.isfile(weight_path):
                continue

            # define networks
            if '_heter' in args.teacher_weights:  # TODO:
                print('load heterogeneous models')
                _get_model_heter = get_model_heter_224 if args.dataset in ['isic2019', 'diabetic2015', 'rsna'] else get_model_heter
                _net_teacher = _get_model_heter(int(client_dir.split('_')[-1]), in_channels=n_channels, num_classes=n_classes)
            else:
                _net_teacher = resnet18(num_classes=n_classes) if args.dataset in ['isic2019', 'diabetic2015', 'rsna'] else ResNet18(in_channels=n_channels, num_classes=n_classes)
            _net_teacher = _net_teacher.to(device)

            # load checkpoint
            checkpoint = torch.load(weight_path)
            _net_teacher.load_state_dict(checkpoint.state_dict())
            _net_teacher.eval()

            model_list.append(_net_teacher)

        if len(model_list) == 0:
            raise ValueError('Invalid weights:', args.teacher_weights)

        # ensemble models
        net_teacher = Ensemble(model_list)
    else:
        raise ValueError('Invalid weights:', args.teacher_weights)

    # copy teacher model
    net_teacher_noiseadapt = copy.deepcopy(net_teacher)
    net_teacher_noiseadapt = net_teacher_noiseadapt.to(device)
    net_teacher_noiseadapt.train()

    criterion = nn.CrossEntropyLoss()

    # Checking teacher accuracy
    print('==> Teacher validation')
    acc_teacher = test(net_teacher, data_test_loader, criterion, device)
    with open(os.path.join(args.exp_descr, 'test_teacher.csv'), 'wt') as wf:
        wf.write('{:.4f}\n'.format(acc_teacher))

    print("Starting model inversion")

    # placeholder for inputs
    inputs = torch.randn((args.bs, n_channels, input_size, input_size), requires_grad=True, device='cuda', dtype=torch.float)

    # target outputs to generate
    targets = torch.LongTensor(list(range(0, n_classes)) * (args.bs // n_classes) + list(range(0, args.bs % n_classes))).to('cuda')

    # define optimizer and loss
    optimizer_di = optim.Adam([inputs], lr=args.di_lr)

    # Create hooks for feature statistics catching
    loss_r_feature_layers = []
    for module in net_teacher.modules():
        if isinstance(module, nn.BatchNorm2d):
            loss_r_feature_layers.append(DeepInversionFeatureHook(module))

    # for classifier
    net_cls = resnet18(num_classes=n_classes) if args.dataset in ['isic2019', 'diabetic2015', 'rsna'] else ResNet18(in_channels=n_channels, num_classes=n_classes)
    net_cls = net_cls.to(device)
    if args.pretrained:
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth', progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        net_cls.load_state_dict(state_dict, strict=False)

    criterion_cls = KLDiv(T=args.T)
    optimizer_cls = torch.optim.SGD(net_cls.parameters(), lr=lr_init, momentum=0.9, weight_decay=5e-4)

    acc_best = 0
    for e in range(1, epochs + 1):
        # initialize gaussian inputs
        inputs.data = torch.randn((args.bs, n_channels, input_size, input_size), requires_grad=True, device='cuda')

        # ==============================
        # get_images
        best_cost = 1e6
        n_classes = targets.max().item() + 1

        optimizer_di.state = collections.defaultdict(dict)  # Reset state of optimizer

        image_list = []

        # empty cache
        torch.cuda.empty_cache()
        gc.collect()

        # setting up the range for jitter
        if inputs.shape[-1] > 128:
            lim_0, lim_1 = 30, 30
        else:
            lim_0, lim_1 = 2, 2

        for mi_idx in range(args.iters_mi):
            # apply random jitter offsets
            off1 = random.randint(-lim_0, lim_0)
            off2 = random.randint(-lim_1, lim_1)
            inputs_jit = torch.roll(inputs, shifts=(off1, off2), dims=(2, 3))

            # forward with jit images
            optimizer_di.zero_grad()
            net_teacher.zero_grad()
            outputs = net_teacher(inputs_jit)
            loss = criterion(outputs, targets)
            loss_target = loss.item()

            # apply total variation regularization
            diff1 = inputs_jit[:, :, :, :-1] - inputs_jit[:, :, :, 1:]
            diff2 = inputs_jit[:, :, :-1, :] - inputs_jit[:, :, 1:, :]
            diff3 = inputs_jit[:, :, 1:, :-1] - inputs_jit[:, :, :-1, 1:]
            diff4 = inputs_jit[:, :, :-1, :-1] - inputs_jit[:, :, 1:, 1:]
            loss_var = torch.norm(diff1) + torch.norm(diff2) + torch.norm(diff3) + torch.norm(diff4)
            loss = loss + args.di_var_scale * loss_var

            # R_feature loss
            loss_distr = sum([mod.r_feature for mod in loss_r_feature_layers])
            loss = loss + args.r_feature_weight * loss_distr  # best for noise before BN

            # l2 loss
            loss = loss + args.di_l2_scale * torch.norm(inputs_jit, 2)

            if mi_idx % args.log_freq == 0:
                print(f"It {mi_idx}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")
                vutils.save_image(inputs.data.clone(), '{}/output_{}_{}.png'.format(img_exp_descr, e, mi_idx), normalize=True, scale_each=True, nrow=n_classes)

            if best_cost > loss.item():
                best_cost = loss.item()
                best_inputs = inputs.data

            # backward pass
            loss.backward()
            optimizer_di.step()

            # append inputs
            image_list.append(inputs.detach().cpu().data)

        # save last
        print(f"It {args.iters_mi}\t Losses: total: {loss.item():3.3f},\ttarget: {loss_target:3.3f} \tR_feature_loss unscaled:\t {loss_distr.item():3.3f}")
        vutils.save_image(inputs.data.clone(), '{}/output_{}_{}.png'.format(img_exp_descr, e, args.iters_mi), normalize=True, scale_each=True, nrow=n_classes)

        # ==============================
        # evaluation
        outputs = net_teacher(best_inputs)
        _, predicted_teach = outputs.max(1)

        print('Teacher correct out of {}: {}, loss at {}'.format(args.bs, predicted_teach.eq(targets).sum().item(), criterion(outputs, targets).item()))

        vutils.save_image(best_inputs.clone(), '{}/output_{}.png'.format(best_img_exp_descr, e), normalize=True, scale_each=True, nrow=n_classes)

        # ==============================
        # train classifier
        print('==> Train classifier')
        adjust_learning_rate(optimizer_cls, e, lr_init=lr_init, lr_step1=lr_step1, lr_step2=lr_step2)

        # set train
        net_cls.train()
        net_teacher_noiseadapt.train()

        # update mean and std (from real to noise)
        for cls_i in range(len(image_list) - 1, -1, -1):
            with torch.no_grad():
                net_teacher_noiseadapt(image_list[cls_i].to(device))

        for cls_i in range(len(image_list)):
            cls_inputs = image_list[cls_i].to(device)

            optimizer_cls.zero_grad()

            # calculate alpha (0 -> 1)
            alpha = cls_i / len(image_list)

            # noise KD
            with torch.no_grad():
                # update mean and std (from real to noise)
                outputs_noise = net_teacher_noiseadapt(cls_inputs)

            # real KD
            with torch.no_grad():
                outputs = net_teacher(cls_inputs)
            outputs_cls = net_cls(cls_inputs)
            loss_cls_real = criterion_cls(outputs_cls, outputs.detach())
            loss_cls_noise = criterion_cls(outputs_cls, outputs_noise.detach())

            # emerge losses
            loss_cls = alpha * loss_cls_real + (1.0 - alpha) * loss_cls_noise

            loss_cls.backward()
            optimizer_cls.step()
        # ==============================

        # test classifier
        acc = test(net_cls, data_test_loader, criterion, device)

        # write log
        with open(os.path.join(args.exp_descr, 'test.csv'), 'at') as wf:
            wf.write('{},{:.4f}\n'.format(e, acc))

        # save best
        if acc_best < acc:
            acc_best = acc
            torch.save(net_cls, os.path.join(args.exp_descr, 'best.pth'))
            torch.save(net_teacher_noiseadapt, os.path.join(args.exp_descr, 'best_teacher_noiseadapt.pth'))

        # save last
        torch.save(net_cls, os.path.join(args.exp_descr, 'last.pth'))
        torch.save(net_teacher_noiseadapt, os.path.join(args.exp_descr, 'last_teacher_noiseadapt.pth'))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--bs', default=256, type=int, help='batch size')
    parser.add_argument('--iters_mi', default=500, type=int, help='number of iterations for model inversion')
    parser.add_argument('--di_lr', default=0.05, type=float, help='lr for deep inversion')
    parser.add_argument('--di_var_scale', default=2.5e-5, type=float, help='TV L2 regularization coefficient')
    parser.add_argument('--di_l2_scale', default=0.0, type=float, help='L2 regularization coefficient')
    parser.add_argument('--r_feature_weight', default=10, type=float, help='weight for BN regularization statistic')
    parser.add_argument('--exp_descr', default="result", type=str, help='name to be added to experiment name')
    parser.add_argument('--teacher_weights', default="./pretrained_models/bloodmnist_iid_5_0.6", type=str, help='path to load weights of the model')
    parser.add_argument('--dataset', default='bloodmnist', type=str)
    parser.add_argument('--root', default='./dataset', type=str)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--T', default=20, type=float)
    parser.add_argument('--log_freq', default=200, type=int)
    parser.add_argument('--pretrained', action='store_true')

    args = parser.parse_args()

    main(args)
