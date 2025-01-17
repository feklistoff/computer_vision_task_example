import cv2
import numpy as np
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2

import torch

ID_TO_CLASSNAME = {1: "iron", 2: "brick", 3: "wood"}


def build_transfrom():
    """Transforms for model's input"""
    return A.Compose([ToTensorV2()])


def predict(
    *,
    image: np.ndarray,
    device: str,
    model: torch.nn.Module,
    inference_thresh: float = 0.5,
) -> tuple:
    """
    Perform prediction on the input image.

    Args:
        image (np.ndarray): Input image in RGB format.
        device (str): Device to run inference on.
        model (torch.nn.Module): The trained model.
        inference_thresh (float): Confidence threshold for predictions.

    Returns:
        list: List of predictions containing label, bounding box, and score.
    """
    # Prepare
    transform = build_transfrom()
    image = image / 255
    inputs = transform(image=image)
    inputs = torch.as_tensor(inputs["image"], dtype=torch.float)
    # Predict
    model.eval()
    inputs = inputs.to(device)[None, ...]
    preds = model(inputs)

    predictions = []
    for pred in preds:
        scores = pred["scores"].cpu().detach().numpy()
        labels = pred["labels"].cpu().detach().numpy()
        boxes = pred["boxes"].cpu().detach().numpy()

        for score, label, box in zip(scores, labels, boxes):
            if score < inference_thresh:
                continue
            label_name = ID_TO_CLASSNAME.get(label, "unknown")
            bbox = box.astype(int)
            predictions.append(
                {
                    "label": label_name,
                    "bbox": bbox,
                    "score": float(score),
                }
            )
    return predictions


def plot_predictions(
    img: np.ndarray,
    predictions: list,
    figsize: tuple = (10, 10),
    color: tuple = (0, 255, 0),
    thickness: int = 4,
):
    """
    Plot the predictions on the image.

    Args:
        img (np.ndarray): Original image in BGR format.
        predictions (list): List of prediction dictionaries.
        figsize (tuple, optional): Figure size. Defaults to (10, 10).
        color (tuple, optional): Bounding box color. Defaults to (0, 255, 0).
        thickness (int, optional): Bounding box thickness. Defaults to 2.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for pred in predictions:
        label = pred["label"]
        bbox = pred["bbox"]
        score = pred["score"]
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)
        cv2.putText(
            img,
            f"{label}: {score:.2f}",
            (bbox[0], bbox[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            color,
            2,
        )
    plt.figure(figsize=figsize)
    plt.imshow(img)
    plt.axis("off")
    plt.show()
