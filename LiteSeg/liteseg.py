import torch

# import liteseg_mobilenet as mobilenet
# import liteseg_shufflenet as shufflenet
from LiteSeg import liteseg_mobilenet as shufflenet
from LiteSeg import liteseg_mobilenet as mobilenet


def LiteSeg(num_class, backbone_network, pretrain_weight, is_train=False):
    if backbone_network.lower() == 'shufflenet':
        net = shufflenet.RT(n_classes=num_class, pretrained=is_train, PRETRAINED_WEIGHTS=pretrain_weight)
    elif backbone_network.lower() == 'mobilenet':
        net = mobilenet.RT(n_classes=num_class, pretrained=is_train, PRETRAINED_WEIGHTS=pretrain_weight)
    else:
        raise NotImplementedError

    print("Using LiteSeg with", backbone_network)

    return net


if __name__ == '__main__':
    model = LiteSeg(num_class=1, backbone_network='mobilenet', pretrain_weight=None, is_train=False)
    x = torch.Tensor(2, 5, 640, 640)
    y = model(x)
    print(y.shape)
