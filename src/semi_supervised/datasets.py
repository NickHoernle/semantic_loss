import numpy as np
from torchvision import transforms, datasets
import torch.utils.data as data
from torch.utils.data import Subset

class Joint(data.Dataset):
    def __init__(self, dataset1, dataset2):
        self.dataset1 = dataset1
        self.dataset2 = dataset2

    def __getitem__(self, index):
        return self.dataset1[index], self.dataset2[index]

    def __len__(self):
        return len(self.dataset1)


def x_u_split(labels, num_labelled, num_classes):
    label_per_class = num_labelled // num_classes
    labels = np.array(labels)
    labelled_idx = []
    unlabelled_idx = []
    for i in range(num_classes):
        idx = np.where(labels == i)[0]
        np.random.shuffle(idx)
        labelled_idx.extend(idx[:label_per_class])
        unlabelled_idx.extend(idx[label_per_class:])

    return labelled_idx, unlabelled_idx


def get_CIFAR10(dataroot, augment=True):

    image_shape = (32, 32, 3)
    num_classes = 10
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    test_transform = transforms.Compose([transforms.ToTensor(), normalize])

    if augment:
        transformations = [transforms.RandomCrop(size=32, padding=int(32 * 0.125), padding_mode='reflect'),
                           transforms.RandomHorizontalFlip()]
    else:
        transformations = []

    transformations.extend([transforms.ToTensor(), normalize])
    train_transform = transforms.Compose(transformations)

    train_dataset = datasets.CIFAR10(dataroot, train=True,
                                     transform=train_transform,
                                     download=True)

    test_dataset = datasets.CIFAR10(dataroot, train=False,
                                    transform=test_transform,
                                    download=False)

    return image_shape, num_classes, train_dataset, test_dataset


def get_semi_supervised(dataroot, num_labelled=4000, dataset="CIFAR10", augment=True):

    num_labelled = num_labelled
    image_shape, num_classes, train_dataset, test_dataset = get_CIFAR10(dataroot, augment=augment)

    num_unlabelled = len(train_dataset) - num_labelled

    td_targets = train_dataset.targets
    labelled_idxs, unlabelled_idxs = x_u_split(td_targets, num_labelled, num_classes)
    labelled_set, unlabelled_set = [Subset(train_dataset, labelled_idxs), Subset(train_dataset, unlabelled_idxs)]

    labelled_set = data.ConcatDataset([labelled_set for i in range(num_unlabelled // num_labelled + 1)])
    labelled_set, _ = data.random_split(labelled_set, [num_unlabelled, len(labelled_set) - num_unlabelled])

    return Joint(labelled_set, unlabelled_set), test_dataset, image_shape, num_classes