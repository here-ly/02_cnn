import torch
from torchvision import transforms
import torchvision.datasets
from torch.utils.data import DataLoader, Subset


CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)


def build_transforms(use_augmentation: bool = True):
    normalize = transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD)
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    if not use_augmentation:
        return transform_test, transform_test

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    return transform_train, transform_test


def denormalize_image(image_tensor: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(CIFAR10_MEAN, device=image_tensor.device).view(1, 3, 1, 1)
    std = torch.tensor(CIFAR10_STD, device=image_tensor.device).view(1, 3, 1, 1)
    image = image_tensor * std + mean
    return torch.clamp(image, 0.0, 1.0)


def build_datasets(data_root: str, use_augmentation: bool = True):
    transform_train, transform_test = build_transforms(use_augmentation)
    train_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=transform_train
    )
    test_dataset = torchvision.datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=transform_test
    )
    return train_dataset, test_dataset


def build_loaders(
    train_dataset,
    test_dataset,
    batch_size: int,
    test_mode: bool = False,
    test_train_size: int = 10000,
    test_test_size: int = 2000,
    num_workers: int = 0,
):
    train_ds = train_dataset
    test_ds = test_dataset
    if test_mode:
        train_ds = Subset(train_dataset, list(range(min(test_train_size, len(train_dataset)))))
        test_ds = Subset(test_dataset, list(range(min(test_test_size, len(test_dataset)))))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, test_loader
