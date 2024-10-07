import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import Subset
from typing import Dict, Any
from ..dataset import Dataset
from .utils import split_train_val_indices

__all__ = ['VietnameseImage']

class VietnameseImage(Dataset):
    def __init__(self, root, num_classes, image_size, val_ratio=None, extra_train_transforms=None):
        # Define the transform for the training and validation datasets
        train_transforms_pre = [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip()
        ]
        train_transforms_post = [
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        if extra_train_transforms is not None:
            if not isinstance(extra_train_transforms, list):
                extra_train_transforms = [extra_train_transforms]
            for ett in extra_train_transforms:
                if isinstance(ett, (transforms.LinearTransformation, transforms.Normalize, transforms.RandomErasing)):
                    train_transforms_post.append(ett)
                else:
                    train_transforms_pre.append(ett)
        train_transforms = transforms.Compose(train_transforms_pre + train_transforms_post)

        test_transforms = [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        test_transforms = transforms.Compose(test_transforms)

        # Load the dataset using ImageFolder
        train_dataset = ImageFolder(root=root + "/Train", transform=train_transforms)
        test_dataset = ImageFolder(root=root + "/Test", transform=test_transforms)
        
        if val_ratio is None:
            super().__init__(train=train_dataset, test=test_dataset)
            self.dataset_dict = {'train': train_dataset, 'test': test_dataset} 
        else:
            # Split the dataset into training and validation sets
            # train_indices, val_indices = split_train_val_indices(
            #     targets=[label for _, label in train_dataset.samples], 
            #     val_ratio=val_ratio, 
            #     num_classes=len(train_dataset.classes)
            # )
            # train_dataset = Subset(train_dataset, indices=train_indices)
            val_dataset = ImageFolder(root=root + "/Validate", transform=test_transforms)
            super().__init__(train=train_dataset, val=val_dataset, test=test_dataset)
            self.dataset_dict = {'train': train_dataset, 'test': test_dataset, 'val': val_dataset} 

    def __len__(self):
        return sum(len(self.dataset_dict[key]) for key in self.dataset_dict)

    def __getitem__(self, key):
        return self.dataset_dict[key]
