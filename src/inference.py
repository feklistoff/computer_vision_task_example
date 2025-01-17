import argparse
import cv2
import torch
from model import DetectionLightningModule
from utils import predict, plot_predictions


def load_model(
    *, device: str, path: str = "model/fasterrcnn_final.pth"
) -> torch.nn.Module:
    """
    Load the trained model from a checkpoint.

    Args:
        device (str): Device to load the model on ('cuda', 'cpu', etc.).
        path (str): Path to the model checkpoint.
        num_classes (int): Number of classes including background.

    Returns:
        torch.nn.Module: The loaded model.
    """
    # Initialize the model
    model_module = DetectionLightningModule()
    model = model_module.model
    # Load weights from the checkpoint
    if device in ["cuda", "mps"]:
        checkpoint = torch.load(path, weights_only=True)
    else:
        checkpoint = torch.load(
            path, map_location=torch.device(device), weights_only=True
        )
    state_dict = {
        k.partition("model.")[2]: v for k, v in checkpoint["state_dict"].items()
    }
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main(args):
    device = args.device if torch.cuda.is_available() else "cpu"
    model = load_model(device=device, path=args.model_path)

    # Read the input image
    img = cv2.imread(args.image)
    if img is None:
        raise FileNotFoundError(f"Image not found: {args.image}")

    # Perform prediction
    predictions = predict(
        image=img,
        device=device,
        model=model,
        inference_thresh=args.inference_thresh,
    )

    # Print predictions
    if not predictions:
        print("No objects detected with the specified confidence threshold.")
    else:
        for pred in predictions:
            print(
                f"Label: {pred['label']}, BBox: {pred['bbox']}, Score: {pred['score']:.2f}"
            )

    # Plot predictions
    plot_predictions(img, predictions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference for Object Detection Model")
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image"
    )
    parser.add_argument(
        "--model_path", type=str, required=True, help="Path to the model checkpoint"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to run inference on ('cuda', 'cpu', 'mps')",
    )
    parser.add_argument(
        "--inference_thresh",
        type=float,
        default=0.5,
        help="Confidence threshold for predictions",
    )

    args = parser.parse_args()
    main(args)
