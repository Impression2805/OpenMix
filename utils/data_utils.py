import numpy as np
import torch
from skimage.filters import gaussian as gblur
import torchvision.transforms as trn
from bisect import bisect_left
import random
from torchvision import datasets


class RandomImages(torch.utils.data.Dataset):
    def __init__(self, transform=None, exclude_cifar=True, data_num=10000):
        self.transform = transform

        self.data = np.load('/data/datasets/300K_random_images.npy').astype(np.uint8)

        # print(111111111111111111)

        if data_num != -1:
            all_id = list(range(len(self.data)))
            sample_id = random.sample(all_id, data_num)
            self.data = self.data[sample_id]

    def __getitem__(self, index):
        # id = self.id_sample[index]
        img = self.data[index]
        if self.transform is not None:
            img = self.transform(img)

        return img, 0, index  # 0 is the class

    def __len__(self):

        return len(self.data)


def build_dataset(args, dataset, mode="train", data_num=10000, origin_dataset=None):
    if origin_dataset is None:
        origin_dataset = dataset

    # mean and standard deviation of channels of CIFAR-10 images
    mean, std = get_dataset_normlize(origin_dataset)
    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                   trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset == 'cifar10':
        if mode == "train":
            data = CIFAR10(root='/data/datasets/cifar10_data/',
                                    download=True,
                                    dataset_type="train",
                                    transform=train_transform,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                    )
        else:
            data = CIFAR10(root='/data/datasets/cifar10_data/',
                                   download=True,
                                   dataset_type="test",
                                   transform=test_transform,
                                   noise_type=args.noise_type,
                                   noise_rate=args.noise_rate
                                   )
        num_classes = 10
    elif dataset == 'cifar100':
        if mode == "train":
            data = CIFAR100(root='./data/',
                                     download=True,
                                     dataset_type="train",
                                     transform=train_transform,
                                     noise_type=args.noise_type,
                                     noise_rate=args.noise_rate
                                     )
        else:
            data = CIFAR100(root='./data/',
                                    download=True,
                                    dataset_type="test",
                                    transform=test_transform,
                                    noise_type=args.noise_type,
                                    noise_rate=args.noise_rate
                                    )
        num_classes = 100

    elif dataset == "Textures":
        data = dset.ImageFolder(root="./data/dtd/images",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10

    elif dataset == "RandomImages":

        # from random_images_300 import RandomImages
        data = RandomImages(transform=trn.Compose(
            [trn.ToTensor(), trn.ToPILImage(), trn.RandomCrop(32, padding=4),
                trn.RandomHorizontalFlip(), trn.ToTensor(), trn.Normalize(mean, std)]), data_num=data_num)
        num_classes = None
    elif dataset == "SVHN":
        if mode == "train":
            data = svhn.SVHN(root='./data/svhn/', split="train",
                             transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
                             download=False)
        else:
            data = svhn.SVHN(root='./data/svhn/', split="test",
                             transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
                             download=False)
        num_classes = 10

    elif dataset == "Places365":
        data = dset.ImageFolder(root="./data/places365/test_subset",
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "LSUN-C":
        data = dset.ImageFolder(root="./data/LSUN_C",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "LSUN-R":
        data = dset.ImageFolder(root="./data/LSUN_R",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "iSUN":
        data = dset.ImageFolder(root="./data/iSUN",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10

    return data, num_classes


def get_dataset_normlize(dataset):
    if dataset == "cifar100":
        mean = (0.507, 0.487, 0.441)
        std = (0.267, 0.256, 0.276)
    else:
        mean = (0.492, 0.482, 0.446)
        std = (0.247, 0.244, 0.262)

    return mean, std


def build_ood_noise(noise_type, ood_num_examples, num_to_avg):
    if noise_type == "Gaussian":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.float32(np.clip(
            np.random.normal(size=(ood_num_examples * num_to_avg, 3, 32, 32), scale=0.5), -1, 1)))
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Rademacher":
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(np.random.binomial(
            n=1, p=0.5, size=(ood_num_examples * num_to_avg, 3, 32, 32)).astype(np.float32)) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    elif noise_type == "Blob":
        ood_data = np.float32(np.random.binomial(n=1, p=0.7, size=(ood_num_examples * num_to_avg, 32, 32, 3)))
        for i in range(ood_num_examples * num_to_avg):
            ood_data[i] = gblur(ood_data[i], sigma=1.5, multichannel=False)
            ood_data[i][ood_data[i] < 0.75] = 0.0
        dummy_targets = torch.ones(ood_num_examples * num_to_avg)
        ood_data = torch.from_numpy(ood_data.transpose((0, 3, 1, 2))) * 2 - 1
        ood_data = torch.utils.data.TensorDataset(ood_data, dummy_targets)
    return ood_data

