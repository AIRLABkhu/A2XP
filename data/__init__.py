import os
from torch.utils.data import Dataset, ConcatDataset
from torchvision import datasets, transforms

_DATANAMES = ('svhn', 'mnist-m', 'syn', 'usps')
_TRAIN_REQUIREMENTS = {
    'svhn': 'train_32x32.mat',
    'mnist-m': 'MNIST-M',
    'syn': 'SYN',
    'usps': 'usps.bz2',
}
_TEST_REQUIREMENTS = {
    'svhn': 'test_32x32.mat',
    'mnist-m': 'MNIST-M',
    'syn': 'SYN',
    'usps': 'usps.t.bz2',
}


def list_datasets():
    return list(_DATANAMES)


def get_dataset(dataset: str, transform=None, target_trasnform=None):
    trainset = Digits(dataset, train=True, transform=transform, target_transform=target_trasnform)
    testset = Digits(dataset, train=False, transform=transform, target_transform=target_trasnform)
    return ConcatDataset([trainset, testset])


class Digits(Dataset):
    __DATADIR = os.path.dirname(__file__)
    def __init__(self, name: str, train: bool=True, input_size: int=224, transform=None, target_transform=None, root: str=__DATADIR):
        name = name.lower()
        requirements = _TRAIN_REQUIREMENTS if train else _TEST_REQUIREMENTS
        requirement_path = os.path.join(root, requirements[name])
        download = False
        if not os.path.exists(requirement_path):
            if name in {_DATANAMES[0], _DATANAMES[-1]}:
                download = True
            else:
                raise FileNotFoundError(f"'{name.upper()}' must be downloaded manually.")
            
        self.transform = transform
        self.target_transform = target_transform
        
        default_transform = [
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
            transforms.Resize(size=(input_size, input_size), interpolation=transforms.InterpolationMode.BILINEAR, antialias=True),
        ]
        if name == _DATANAMES[0]:
            split = 'train' if train else 'test'
            self.__dataset = datasets.SVHN(root=root, split=split, transform=transforms.Compose(default_transform), download=download)
            
        elif name == _DATANAMES[1]:
            split = 'training' if train else 'testing'
            root = os.path.join(root, _TRAIN_REQUIREMENTS[name], split)
            self.__dataset = datasets.ImageFolder(root=root, transform=transforms.Compose(default_transform))
            
        elif name == _DATANAMES[2]:
            split = 'imgs_train' if train else 'imgs_valid'
            root = os.path.join(root, _TRAIN_REQUIREMENTS[name], split)
            self.__dataset = datasets.ImageFolder(root=root, transform=transforms.Compose(default_transform))
            
        elif name == _DATANAMES[3]:
            default_transform.append(transforms.Lambda(lambda x: x.repeat(3, 1, 1)))
            self.__dataset = datasets.USPS(root=root, train=train, transform=transforms.Compose(default_transform), download=download)
        else:
            assert False
            
    def __len__(self):
        return len(self.__dataset)
    
    def __getitem__(self, i: int):
        img, label = self.__dataset[i]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            label = self.target_transform(label)
        return img, label
    
    @property
    def classes(self):
        return tuple(range(10))


if __name__ == '__main__':
    for dataname in _DATANAMES:
        dataset = Digits(dataname, train=False)
        img, target = dataset[1]
        print(dataname, img.shape, type(target))
    