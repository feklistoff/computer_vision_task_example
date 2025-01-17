import argparse
import random
import uuid
from pathlib import Path
import logging

import cv2
import numpy as np
import pandas as pd
import albumentations as A
from tqdm import tqdm

# Define the target image shape
SHAPE = (1920, 1080)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def random_aug(*, img: np.ndarray, is_icon: bool = False) -> np.ndarray:
    """
    Apply random augmentations to the input image.

    Args:
        img (np.ndarray): The input image.
        is_icon (bool): Flag indicating if the image is an icon.

    Returns:
        np.ndarray: The augmented image.
    """
    h, w = img.shape[:2]
    width = w // 2
    height = h // 2
    transform = A.Compose(
        [
            A.RandomCrop(width=width, height=height, p=0.5),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(p=0.1),
            A.OpticalDistortion(p=0.5),
            A.GridDistortion(p=0.5),
            A.CoarseDropout(p=0.5),
            A.GaussNoise(var_limit=(10, 100), p=0.2),
        ]
    )
    if is_icon:
        transform = A.Compose(
            [
                A.HorizontalFlip(p=0.5),
                A.RandomBrightnessContrast(p=0.2),
                A.Rotate(limit=10, p=0.5),
                A.Affine(shear=(-10, 10), p=0.2),
                A.RandomScale(scale_limit=(0.1, 1.5), p=0.8),
            ]
        )
    # Augment image
    transformed = transform(image=img)
    return transformed["image"]


class Generator:
    """
    Class for generating synthetic training and validation data by overlaying icons onto backgrounds.
    """

    def __init__(self, *, background_dir: str, icons_dir: str, data_dir: str = "data"):
        """
        Initialize the data generator.

        Args:
            background_dir (str): Directory containing background images.
            icons_dir (str): Directory containing icon images.
            data_dir (str): Directory to save generated data and annotations.
        """
        base_dir = Path(__file__).parent.parent
        self.data_dir = Path(base_dir) / data_dir
        self.icons_dir = Path(base_dir) / icons_dir
        background_dir = Path(base_dir) / background_dir
        self.backgrounds = [str(p) for p in background_dir.iterdir() if p.is_file()]
        self.labels = [p.stem for p in self.icons_dir.iterdir() if p.is_file()]
        self.df = pd.DataFrame(columns=["filepath", "label", "x1", "y1", "x2", "y2"])
        logging.info(
            f"Initialized Generator with {len(self.backgrounds)} backgrounds and {len(self.labels)} icons"
        )

    def generate_data(self, *, num_img_per_label: int = 10) -> None:
        """
        Generate synthetic data for each label.

        Args:
            num_img_per_label (int): Number of images to generate per label.
        """
        logging.info(f"Starting data generation: {num_img_per_label} images per label")
        for label in self.labels:
            logging.info(f"Generating images for label: {label}")
            self.generate_images(label=label, num_img_per_label=num_img_per_label)
        logging.info(f"Data generation complete. Total images: {len(self.df)}")

    def generate_images(self, *, label: str, num_img_per_label: int) -> None:
        """
        Generate images by overlaying the specified icon onto random backgrounds.

        Args:
            label (str): The label of the icon.
            num_img_per_label (int): Number of images to generate for this label.
        """
        # Create directory to save synthetic images
        save_dir = self.data_dir / "synth_data"
        save_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Saving images to {save_dir}")

        icon_path = self.icons_dir / f"{label}.png"
        if not icon_path.exists():
            logging.error(f"Icon file not found: {icon_path}")
            return

        for i in tqdm(range(num_img_per_label), desc=f"Generating {label}", leave=True):
            # Select a random background
            back = cv2.imread(random.choice(self.backgrounds))
            if back is None:
                logging.warning("Failed to read background image. Skipping.")
                continue
            back = random_aug(img=back)
            back = cv2.resize(back, dsize=SHAPE)

            # Load and augment the icon
            icon = cv2.imread(str(icon_path), flags=cv2.IMREAD_UNCHANGED)
            if icon is None:
                logging.warning(f"Failed to read icon image: {icon_path}. Skipping.")
                continue
            icon = random_aug(img=icon, is_icon=True)

            # Handle alpha channel
            if icon.shape[2] == 4:
                alpha = icon[:, :, 3:] / 255.0
                icon_rgb = icon[:, :, :3]
            else:
                alpha = np.ones((icon.shape[0], icon.shape[1], 1), dtype=icon.dtype)
                icon_rgb = icon

            # Ensure the icon fits within the background
            y_offset = random.randint(0, back.shape[0] - icon.shape[0])
            x_offset = random.randint(0, back.shape[1] - icon.shape[1])
            y1, y2 = y_offset, y_offset + icon.shape[0]
            x1, x2 = x_offset, x_offset + icon.shape[1]

            # Clip coordinates to image dimensions
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(back.shape[1], x2)
            y2 = min(back.shape[0], y2)

            # Overlay the icon onto the background
            back[y1:y2, x1:x2] = back[y1:y2, x1:x2] * (1 - alpha) + icon_rgb * alpha

            # Save the synthetic image
            name = f"{label}_{str(uuid.uuid1())[:8]}.png"
            filepath = str(save_dir / name)
            cv2.imwrite(filepath, back)

            # Record annotation
            self.df.loc[len(self.df)] = {
                "filepath": filepath,
                "label": label,
                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,
            }

        # Save annotations after generating images for this label
        logging.info(f"Saving data.csv to {self.data_dir}")
        self.df.to_csv(self.data_dir / "data.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic Data Generator for Object Detection"
    )
    parser.add_argument(
        "--num", default=50, type=int, help="Number of images per label to generate"
    )
    parser.add_argument(
        "--icons",
        default="artifacts/icons",
        type=str,
        help="Directory containing icon images",
    )
    parser.add_argument(
        "--backgrounds",
        default="artifacts/backgrounds",
        type=str,
        help="Directory containing background images",
    )
    parser.add_argument(
        "--save_dir",
        default="data",
        type=str,
        help="Directory to save generated data and annotations",
    )
    args = parser.parse_args()

    # Initialize and run the data generator
    generator = Generator(
        background_dir=args.backgrounds, icons_dir=args.icons, data_dir=args.save_dir
    )
    generator.generate_data(num_img_per_label=args.num)
