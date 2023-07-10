from omegaconf import DictConfig, ValueNode
import torch
from torch.utils.data import Dataset
import os
from os import listdir
from os.path import isfile, join
from PIL import Image
import csv
from torchvision import transforms
from torchvision.datasets.cifar import CIFAR10 as cifar10_data
from torch.nn import functional as F
import numpy as np
from glob2 import glob
from skimage import io
from src.pl_data import normalizations


DATADIR = "data/"


def load_image(directory):
    return Image.open(directory).convert('L')


def invert(img):

    if img.ndim < 3:
        raise TypeError("Input image tensor should have at least 3 dimensions, but found {}".format(img.ndim))

    bound = torch.tensor(1 if img.is_floating_point() else 255, dtype=img.dtype, device=img.device)
    return bound - img


def colour(img, ch=0, num_ch=3):

    colimg = [torch.zeros_like(img)] * num_ch
    # colimg[ch] = img
    # Use beta distribution to push the mixture to ch 1 or ch 2
    if ch == 0:
        rand = torch.distributions.beta.Beta(0.5, 1.)
    elif ch == 1:
        rand = torch.distributions.beta.Beta(1., 0.5) 
    else:
        raise NotImplementedError("Only 2 channel images supported now.")
    rand = rand.sample()
    colimg[0] = img * rand
    colimg[1] = img * (1 - rand)
    return torch.cat(colimg)


class CIFAR10(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform

        self.dataset = cifar10_data(root=DATADIR, download=True)
        self.data_len = len(self.dataset)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        img, label = self.dataset[index]
        img = np.asarray(img)
        label = np.asarray(label)
        # Transpose shape from H,W,C to C,H,W
        img = img.transpose(2, 0, 1).astype(np.float32)
        # img = F.to_tensor(img)
        # label = F.to_tensor(label)
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class COR14(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.control = ["20CAG30"]
        self.disease = ["72CAG12"]

        # List all the files
        print("Globbing files for COR14, this may take a while...")
        self.c1 = glob(os.path.join(self.path, "**", "20CAG30", "*.tif"))
        self.c2 = glob(os.path.join(self.path, "**", "72CAG12", "*.tif"))
        min_files = min(len(self.c1), len(self.c2))
        print("Using {} files".format(min_files))
        self.files = self.c1[:min_files] + self.c2[:min_files]
        # self.files = glob(os.path.join(self.path, "**", "**", "*.tif"))
        self.files = np.asarray(self.files)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-2]
        if cell_line in self.control:
            label = 0
        elif cell_line in self.disease:
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [1, 2].".format(label))
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class JAK(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.control = "NN0005319"
        self.disease = "NN0005320"

        # List all the files
        print("Globbing files for JAK, this may take a while...")
        self.c1 = glob(os.path.join(self.path, "**", self.control, "Soma", "*.tif"))
        self.c2 = glob(os.path.join(self.path, "**", self.disease, "Soma", "*.tif"))
        min_files = min(len(self.c1), len(self.c2))
        print("Using {} files".format(min_files))
        self.files = self.c1[:min_files] + self.c2[:min_files]
        self.files = np.asarray(self.files)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel

        cell_line = fn.split(os.path.sep)[-3]
        if cell_line in self.control:
            label = 0
        elif cell_line in self.disease:
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [0, 1].".format(cell_line))
        return img, label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class SIMCLR_COR14(Dataset):
    def __init__(
        self, path: ValueNode, train: bool, cfg: DictConfig, transform, **kwargs
    ):
        super().__init__()
        self.cfg = cfg
        self.path = path
        self.train = train
        self.transform = transform
        self.maxval = 33000
        self.minval = 0
        self.denom = self.maxval - self.minval
        self.control = ["20CAG30", "20CAG44", "20CAG65"]
        self.disease = ["72CAG2", "72CAG4", "72CAG9", "72CAG12"]

        self.mu = 1086.6762200888888 / self.maxval
        self.sd = 2019.9389348809887 / self.maxval
        self.mu *= 255
        self.sd *= 255

        # List all the files
        print("Globbing files for COR14, this may take a while...")
        self.files = glob(os.path.join(self.path, "**", "**", "*.tif"))
        self.files = np.asarray(self.files)
        np.random.seed(42)
        shuffle_idx = np.random.permutation(len(self.files))
        self.files = self.files[shuffle_idx]
        self.data_len = len(self.files)

    def __len__(self) -> int:
        return self.data_len

    def __getitem__(self, index: int):
        fn = self.files[index]
        img = io.imread(fn, plugin='pil')
        img = img.astype(np.float32)
        img = (img - self.minval) / self.denom  # Normalize to [0, 1]
        # img = img[None].repeat(3, axis=0)  # Stupid but let's replicate 1->3 channel

        transform = SimCLRTrainDataTransform(
            input_height=200,
            # normalize=normalizations.COR14_normalization,
            gaussian_blur=False)
        img = Image.fromarray((img * 255).astype(np.uint8))
        (xi, xj) = transform(img)
        xi = xi.tile(3, 1, 1)
        xj = xj.tile(3, 1, 1)
        xi = (xi - self.mu) / self.sd
        xj = (xj - self.mu) / self.sd
        cell_line = fn.split(os.path.sep)[-2]
        if cell_line in self.control:
            label = 0
        elif cell_line in self.disease:
            label = 1
        else:
            raise RuntimeError("Found label={} but expecting labels in [1, 2].".format(label))
        return (xi, xj), label

    def __repr__(self) -> str:
        return f"MyDataset({self.name}, {self.path})"


class SimCLRTrainDataTransform:
    """Transforms for SimCLR.
    Transform::
        RandomResizedCrop(size=self.input_height)
        RandomHorizontalFlip()
        RandomApply([color_jitter], p=0.8)
        RandomGrayscale(p=0.2)
        GaussianBlur(kernel_size=int(0.1 * self.input_height))
        transforms.ToTensor()
    Example::
        from pl_bolts.models.self_supervised.simclr.transforms import SimCLRTrainDataTransform
        transform = SimCLRTrainDataTransform(input_height=32)
        x = sample()
        (xi, xj) = transform(x)
    """

    def __init__(
        self, input_height: int = 224, gaussian_blur: bool = False, jitter_strength: float = 1.0, normalize=None
    ) -> None:

        self.jitter_strength = jitter_strength
        self.input_height = input_height
        self.gaussian_blur = gaussian_blur
        self.normalize = normalize

        self.color_jitter = transforms.ColorJitter(
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.8 * self.jitter_strength,
            0.2 * self.jitter_strength,
        )

        data_transforms = [
            transforms.RandomResizedCrop(size=self.input_height),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply([self.color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
        ]

        if self.gaussian_blur:
            kernel_size = int(0.1 * self.input_height)
            if kernel_size % 2 == 0:
                kernel_size += 1

            data_transforms.append(transforms.GaussianBlur(kernel_size=kernel_size))

        data_transforms = transforms.Compose(data_transforms)

        if normalize is None:
            self.final_transform = transforms.ToTensor()
        else:
            self.final_transform = transforms.Compose([transforms.ToTensor(), normalize])

        self.train_transform = transforms.Compose([data_transforms, self.final_transform])

        # # add online train transform of the size of global view
        # self.online_transform = transforms.Compose(
        #     [transforms.RandomResizedCrop(self.input_height), transforms.RandomHorizontalFlip(), self.final_transform]
        # )

    def __call__(self, sample):
        transform = self.train_transform

        xi = transform(sample)
        xj = transform(sample)

        return xi, xj  # , self.online_transform(sample)
