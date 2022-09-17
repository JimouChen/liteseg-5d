import numpy as np
from torch.utils.data import Dataset
import os
from torchvision import transforms
from torch.utils.data import DataLoader
import cv2
from skimage import io


class MyData(Dataset):
    def __init__(self, mask_path, img_path, img_size=(512, 512)):
        super(MyData, self).__init__()
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            # transforms.RandomResizedCrop(img_size),
            # transforms.Resize(img_size),
            # transforms.RandomHorizontalFlip()
        ])
        self.data_path = []
        self.mask_path = []
        for file in os.listdir(img_path):
            self.data_path.append(os.path.join(img_path, file))
        for file in os.listdir(mask_path):
            self.mask_path.append(os.path.join(mask_path, file))

    def __getitem__(self, index):
        img = io.imread(self.data_path[index])
        img = np.transpose(img[:5, :, :], (1, 2, 0))
        label = cv2.imread(self.mask_path[index], 0)
        data, label = np.array(img / 255, np.float32), np.array(label / 255, np.float32)
        data, label = self.transforms(img), self.transforms(label)

        return data, label

    def __len__(self):
        return len(self.data_path)


if __name__ == '__main__':
    from LiteSeg import liteseg
    from utils import get_parse
    import warnings

    warnings.filterwarnings('ignore')

    args = get_parse()
    model = liteseg.LiteSeg(num_class=1, backbone_network='mobilenet',
                            pretrain_weight=None, is_train=False)
    train_dataset = MyData(args.train_label, args.train_data, img_size=(640, 640))
    train_loader = DataLoader(train_dataset,
                              batch_size=4,
                              shuffle=True,
                              num_workers=4)
    for data, label in train_loader:
        print(data.shape, label.shape)
        print(type(data))
        out = model(data)
        print(out.shape)
        break
