import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
from PIL import Image
from sklearn import metrics


class Ensemble(torch.nn.Module):
    def __init__(self, model_list):
        super(Ensemble, self).__init__()
        self.models = nn.ModuleList(model_list)

    def forward(self, x):
        logits_total = 0
        for i in range(len(self.models)):
            logits = self.models[i](x)
            logits_total += logits
        logits_e = logits_total / len(self.models)

        return logits_e


class DeepInversionFeatureHook():
    '''
    Implementation of the forward hook to track feature statistics and compute a loss on them.
    Will compute mean and variance, and will use l2 as a loss
    '''

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        # hook co compute deepinversion's feature distribution regularization
        nch = input[0].shape[1]

        mean = input[0].mean([0, 2, 3])
        var = input[0].permute(1, 0, 2, 3).contiguous().view([nch, -1]).var(1, unbiased=False)

        # forcing mean and variance to match between two distributions
        # other ways might work better, e.g. KL divergence
        r_feature = torch.norm(module.running_var.data.type(var.type()) - var, 2) + torch.norm(module.running_mean.data.type(var.type()) - mean, 2)

        self.r_feature = r_feature  # must have no output

    def close(self):
        self.hook.remove()


def test(net, testloader, criterion, device):
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    gt_list, pred_list = [], []
    img_size = 28

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.flatten().long().cuda() if len(targets.shape) == 2 else targets.long().cuda()
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            gt_list.extend(targets.tolist())
            pred_list.extend(predicted.tolist())
            img_size = inputs.shape[-1]

    acc = correct / total
    b_acc = metrics.balanced_accuracy_score(gt_list, pred_list)
    print('Loss: %.3f | Acc: %.3f%% (%d/%d), B. Acc: %.3f%%' % (test_loss / (batch_idx + 1), 100. * acc, correct, total, b_acc))

    # for isic2019, diabetic2015
    if img_size > 128:
        acc = b_acc

    return acc


class KLDiv(nn.Module):
    def __init__(self, T=1.0, reduction='batchmean'):
        super().__init__()
        self.T = T
        self.reduction = reduction

    def forward(self, logits, targets):
        return kldiv(logits, targets, T=self.T, reduction=self.reduction)


def kldiv(logits, targets, T=1.0, reduction='batchmean'):
    q = F.log_softmax(logits / T, dim=1)
    p = F.softmax(targets / T, dim=1)
    return F.kl_div(q, p, reduction=reduction) * (T * T)


def adjust_learning_rate(optimizer, epoch, lr_init=0.1, lr_step1=80, lr_step2=120):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    if epoch < lr_step1:
        lr = lr_init
    elif epoch < lr_step2:
        lr = lr_init * 0.1
    else:
        lr = lr_init * 0.1 * 0.1
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_cls_num_list(traindata_cls_counts, num_label):
    cls_num_list = []
    for key, val in traindata_cls_counts.items():
        temp = [0] * num_label
        for key_1, val_1 in val.items():
            temp[key_1] = val_1
        cls_num_list.append(temp)

    return cls_num_list


def record_net_data_stats(y_train, net_dataidx_map):
    net_cls_counts = {}

    for net_i, dataidx in net_dataidx_map.items():
        unq, unq_cnt = np.unique(y_train[dataidx], return_counts=True)
        tmp = {unq[i]: unq_cnt[i] for i in range(len(unq))}
        net_cls_counts[net_i] = tmp

    return net_cls_counts


def partition_data(y_train, num_label, partition, beta=0.4, num_users=5, debug=True):
    data_size = y_train.shape[0]

    if partition == "iid":
        idxs = np.random.permutation(data_size)
        batch_idxs = np.array_split(idxs, num_users)
        net_dataidx_map = {i: batch_idxs[i] for i in range(num_users)}

    elif partition == "dirichlet":
        min_size = 0
        min_require_size = 10
        net_dataidx_map = {}

        while min_size < min_require_size:
            idx_batch = [[] for _ in range(num_users)]
            for k in range(num_label):
                idx_k = np.where(y_train == k)[0]
                np.random.shuffle(idx_k)  # shuffle the label
                proportions = np.random.dirichlet(np.repeat(beta, num_users))
                proportions = np.array([p * (len(idx_j) < data_size / num_users) for p, idx_j in zip(proportions, idx_batch)])  # 0 or x
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
                idx_batch = [idx_j + idx.tolist() for idx_j, idx in zip(idx_batch, np.split(idx_k, proportions))]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_users):
            np.random.shuffle(idx_batch[j])
            net_dataidx_map[j] = idx_batch[j]

    if debug:
        train_data_cls_counts = record_net_data_stats(y_train, net_dataidx_map)
        print('Data statistics: %s' % str(train_data_cls_counts))

        train_cls_num_list = get_cls_num_list(train_data_cls_counts, num_label)
        print('Data number: %s' % str(train_cls_num_list))

    return net_dataidx_map


class DatasetSplit(torch.utils.data.Dataset):

    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label


class ImageDataset(torch.utils.data.Dataset):
    def __init__(self, images, targets, transform=None):
        self.images = images
        self.targets = targets
        self.transform = transform

    def __getitem__(self, idx):
        img = self.images[idx]
        img = Image.fromarray(img)
        target = self.targets[idx]
        if self.transform:
            img = self.transform(img)
        return img, target

    def __len__(self):
        return len(self.images)
