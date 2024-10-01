import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from ..dataset import Dataset

__all__ = ['VietnameseImage']

class VietnameseImage(Dataset):
    def __init__(self, root, num_classes, image_size, extra_train_transforms=None):

        """
        Initializes the VietnameseImage dataset for CIFAR-10.

        Args:
            root (str): Path to the dataset root.
            num_classes (int): Number of classes in the dataset.
            val_ratio (Optional[float]): Ratio for splitting training and validation sets (between 0 and 1).
            extra_train_transforms (Optional[Union[List[transforms.Transform], transforms.Transform]]): Additional training transformations.
        """
        mean = [0.5818713307380676, 0.514377236366272, 0.4011467397212982]
        std = [0.2084718942642212, 0.206680029630661, 0.228310227394104]

        # Define the transform for the training dataset
        train_transforms_pre = [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
            transforms.RandomRotation(degrees=(30, 70)),
        ]
        
        train_transforms_post = [
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ]
        
        # Append any extra transformations provided
        if extra_train_transforms is not None:
            if not isinstance(extra_train_transforms, list):
                extra_train_transforms = [extra_train_transforms]
            for ett in extra_train_transforms:
                if isinstance(ett, (transforms.Normalize, transforms.RandomErasing)):
                    train_transforms_post.append(ett)
                else:
                    train_transforms_pre.append(ett)
        
        train_transforms = transforms.Compose(train_transforms_pre + train_transforms_post)

        # Define the transform for the test dataset
        test_transforms = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std=[0.5, 0.5, 0.5]
            )
        ])

        # Load the dataset using ImageFolder
        train_dataset = ImageFolder(root=root + "/Train", transform=train_transforms)
        test_dataset = ImageFolder(root=root + "/Test", transform=test_transforms)
        val_dataset = ImageFolder(root=root + "/Validate", transform=test_transforms)

        super().__init__(train=train_dataset, val=val_dataset, test=test_dataset)
