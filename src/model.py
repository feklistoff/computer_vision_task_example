import pytorch_lightning as pl
import torch
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision


class DetectionLightningModule(pl.LightningModule):
    """
    PyTorch Lightning Module for Faster R-CNN object detection.
    """

    def __init__(self, num_classes: int = 4, lr: float = 0.001):
        """
        Initialize the DetectionLightningModule.

        Args:
            num_classes (int): Number of classes in the dataset.
            lr (float): Learning rate for the optimizer.
        """
        super().__init__()
        self.num_classes = num_classes
        self.lr = lr
        self.model = self.build_model()
        self.metric = MeanAveragePrecision(
            iou_thresholds=[0.5, 0.75, 0.95],
            box_format="xyxy",
            iou_type="bbox",
            class_metrics=False,
        )

    def build_model(self):
        """
        Build the Faster R-CNN model with pre-trained weights.

        Returns:
            torchvision.models.detection.FasterRCNN: The Faster R-CNN model.
        """
        # Build model and load pre-trained weights
        model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
            weights=torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        )
        # Modify last layer
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = (
            torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
                in_features, self.num_classes
            )
        )
        return model

    def forward(self, x):
        """
        Forward pass for the model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, targets = batch
        # Fasterrcnn takes both images and targets for training
        loss_dict = self.model(images, targets)
        loss = sum(loss_dict.values())
        self.log(
            "train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        return loss

    def validation_step(self, batch, batch_idx):
        images, targets = batch
        # Forward pass
        # Fasterrcnn takes only images for eval() mode
        preds = self.model(images)
        self.metric.update(preds, targets)

    def on_validation_epoch_end(self):
        mAP_metrics = self.metric.compute()
        # Remove not needed metrics
        mAP_metrics.pop("classes")
        self.log_dict(mAP_metrics, prog_bar=True, logger=True, sync_dist=True)
        # Reset metric for next epoch
        self.metric.reset()

    def configure_optimizers(self):
        return torch.optim.SGD(
            self.model.parameters(),
            lr=self.lr,
            momentum=0.9,
            weight_decay=0.005,
        )
