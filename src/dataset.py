import cv2
import pandas as pd

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
import pytorch_lightning as pl


def collate_fn(batch):
    """
    Custom collate function to handle batches of images and targets.

    Args:
        batch (list): List of tuples (image, target).

    Returns:
        tuple: Tuple containing lists of images and targets.
    """
    return tuple(zip(*batch))


class DetectionDataset(torch.utils.data.Dataset):
    """
    Custom Dataset for object detection tasks compatible with PyTorch DataLoader.
    """

    def __init__(self, *, path: str, transform: A.Compose):
        """
        Initialize the dataset.

        Args:
            path (str): Path to the CSV file containing annotations.
            transform (A.Compose): Albumentations transformations to apply.
        """
        self.df = pd.read_csv(path)
        self.classes = self.create_classes()
        self.transform = transform

    def create_classes(self) -> dict:
        """
        Create a mapping from class names to indices.

        Returns:
            dict: Dictionary mapping class names to integer indices.
        """
        labels = self.df.label.unique().tolist()
        # Start indexing from 1. 0 is for background
        return {x: i + 1 for i, x in enumerate(sorted(labels))}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        """
        Retrieve the image and its corresponding target annotations.
        Use COCO dataset annotation format.
        """
        data = self.df.iloc[idx]
        # Read the info
        image = self.__load_image(data.filepath)
        label = self.classes[data.label]
        box = data[["x1", "y1", "x2", "y2"]].values
        # Ensure bounding boxes are valid
        x1, y1, x2, y2 = box
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = max(x1 + 1, x2)  # Ensure x2 > x1
        y2 = max(y1 + 1, y2)  # Ensure y2 > y1
        box = [x1, y1, x2, y2]

        area = (x2 - x1) * (y2 - y1)
        area = torch.as_tensor([area], dtype=torch.float32)
        labels = [label]
        iscrowd = torch.zeros((1,), dtype=torch.int64)

        # Compose
        target = {
            "boxes": torch.as_tensor([box], dtype=torch.float32),
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": torch.tensor([idx]),
        }

        # Transform
        sample = {"image": image, "bboxes": target["boxes"], "labels": labels}
        sample = self.transform(**sample)
        image = torch.as_tensor(sample["image"], dtype=torch.float)
        target["boxes"] = torch.as_tensor(sample["bboxes"], dtype=torch.float32)
        target["labels"] = torch.as_tensor(sample["labels"], dtype=torch.int64)
        return image, target

    def __load_image(self, path):
        """
        NOTE: cv2.imread returns BGR image
        """
        img = cv2.imread(path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {path}")
        return img / 255.0


class DetectionDataModule(pl.LightningDataModule):
    """
    PyTorch Lightning DataModule for object detection tasks.
    """

    def __init__(
        self, *, data_dir: str = "data", batch_size: int = 4, num_workers: int = 8
    ):
        """
        Initialize the DataModule.

        Args:
            data_dir (str): Directory containing the CSV annotations and images.
            batch_size (int): Batch size for DataLoaders.
            num_workers (int): Number of workers for DataLoaders.
        """

        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def make_transforms(self):
        """
        Define the data transformations.

        Returns:
            A.Compose: Albumentations Compose object with transformations.
        """
        return A.Compose(
            [ToTensorV2()],
            bbox_params=A.BboxParams(format="pascal_voc", label_fields=["labels"]),
        )

    def train_dataloader(self):
        self.train_dataset = DetectionDataset(
            path=f"{self.data_dir}/train.csv",
            transform=self.make_transforms(),
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
        )

    def val_dataloader(self):
        self.val_dataset = DetectionDataset(
            path=f"{self.data_dir}/valid.csv",
            transform=self.make_transforms(),
        )
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
            persistent_workers=True,
        )
