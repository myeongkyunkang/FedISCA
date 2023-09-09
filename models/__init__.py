from torch.hub import load_state_dict_from_url
from torchvision import models

from .resnet_cifar import ResNet18, ResNet34
from .vgg import vgg8_bn, vgg11_bn
from .wresnet import wrn_16_2


def get_model_heter(index, num_classes, in_channels=3):
    if index == 0:
        return ResNet18(in_channels=in_channels, num_classes=num_classes)
    elif index == 1:
        return ResNet34(in_channels=in_channels, num_classes=num_classes)
    elif index == 2:
        return wrn_16_2(in_channels=in_channels, num_classes=num_classes)
    elif index == 3:
        return vgg8_bn(in_channels=in_channels, num_classes=num_classes)
    elif index == 4:
        return vgg11_bn(in_channels=in_channels, num_classes=num_classes)

    print('WARNING Invalid Model Index:', index)
    return ResNet18(in_channels=in_channels, num_classes=num_classes)


def get_model_heter_224(index, num_classes, in_channels=3):
    if index in [0, 1, 2]:
        net = models.resnet18(num_classes=num_classes)
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth', progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        net.load_state_dict(state_dict, strict=False)
        return net
    elif index == 3:
        net = models.resnet34(num_classes=num_classes)
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet34-b627a593.pth', progress=True)
        del state_dict['fc.weight']
        del state_dict['fc.bias']
        net.load_state_dict(state_dict, strict=False)
        return net
    elif index == 4:
        net = models.vgg11_bn(num_classes=num_classes)
        state_dict = load_state_dict_from_url('https://download.pytorch.org/models/vgg11_bn-6002323d.pth', progress=True)
        del state_dict['classifier.6.weight']
        del state_dict['classifier.6.bias']
        net.load_state_dict(state_dict, strict=False)
        return net

    print('WARNING Invalid Model Index:', index)
    net = models.resnet18(num_classes=num_classes)
    state_dict = load_state_dict_from_url('https://download.pytorch.org/models/resnet18-f37072fd.pth', progress=True)
    del state_dict['fc.weight']
    del state_dict['fc.bias']
    net.load_state_dict(state_dict, strict=False)
    return net
