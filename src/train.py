import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers.tensorboard import TensorBoardLogger

from dataset import DetectionDataModule
from model import DetectionLightningModule


def split_data(
    data_path: str,
    train_path: str,
    valid_path: str,
    test_size: float = 0.2,
    random_seed: int = 358,
):
    """
    Split the dataset into training and validation sets.

    Args:
        data_path (str): Path to the original CSV file containing annotations.
        train_path (str): Path to save the training CSV.
        valid_path (str): Path to save the validation CSV.
        test_size (float, optional): Proportion of the dataset to include in the validation split. Defaults to 0.2.
        random_seed (int, optional): Random seed for reproducibility. Defaults to 358.
    """
    # Load the data
    data = pd.read_csv(data_path)
    # Perform stratified splitting
    train, valid = train_test_split(
        data, test_size=test_size, random_state=random_seed, stratify=data["label"]
    )
    # Display split information
    print(f"Number of train samples: {len(train)}, valid samples: {len(valid)}")
    print("-" * 50)
    print(f"  Train:\n{train.label.value_counts()}")
    print("-" * 50)
    print(f"  Valid:\n{valid.label.value_counts()}")
    # Save the splits
    train.to_csv(train_path, index=False)
    valid.to_csv(valid_path, index=False)


def main(args):
    """
    Main function to handle training and evaluation.
    """
    data_csv = f"{args.data_dir}/data.csv"
    # Check if data needs to be split
    train_csv = f"{args.data_dir}/train.csv"
    valid_csv = f"{args.data_dir}/valid.csv"

    # Check if data needs to be split
    if not Path(train_csv).exists() or not Path(valid_csv).exists():
        print("Splitting data into training and validation sets...")
        split_data(
            data_path=data_csv,
            train_path=train_csv,
            valid_path=valid_csv,
            test_size=args.test_size,
        )

    # Initialize DataModule
    data_module = DetectionDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Initialize Model
    model_module = DetectionLightningModule(lr=args.lr)

    # Define Callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="map",
        dirpath="model/",
        filename="fasterrcnn-{epoch:02d}-{map:.2f}",
        save_top_k=1,
        mode="max",
        verbose=True,
    )
    early_stop_callback = EarlyStopping(
        monitor="map",
        patience=5,
        mode="max",
    )

    # Initialize Trainer
    trainer = pl.Trainer(
        max_epochs=args.num_epochs,
        log_every_n_steps=1,
        callbacks=[checkpoint_callback, early_stop_callback],
        accelerator=args.device,
        logger=TensorBoardLogger(save_dir=Path(__file__).parent.parent / "logs/"),
    )

    if args.evaluate:
        if not args.model_path:
            raise ValueError("Model path must be provided for evaluation.")
        # Load the model from checkpoint
        model_module = DetectionLightningModule.load_from_checkpoint(
            checkpoint_path=args.model_path, lr=args.lr
        )
        # Run validation
        trainer.validate(model=model_module, datamodule=data_module)
    else:
        # Start training
        trainer.fit(model=model_module, datamodule=data_module)
        # Save the final model
        trainer.save_checkpoint("model/fasterrcnn_final.pth", weights_only=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Faster R-CNN for Object Detection"
    )
    parser.add_argument(
        "--data_dir", default="data", type=str, help="Directory containing data.csv"
    )
    parser.add_argument(
        "--batch_size", default=4, type=int, help="Batch size for training"
    )
    parser.add_argument(
        "--num_epochs", default=10, type=int, help="Number of training epochs"
    )
    parser.add_argument("--lr", default=0.0005, type=float, help="Learning rate")
    parser.add_argument(
        "--num_workers", default=8, type=int, help="Number of workers for DataLoader"
    )
    parser.add_argument(
        "--device",
        default="auto",
        type=str,
        help="Device to use ('auto', 'gpu', 'cpu', 'mps')",
    )
    parser.add_argument(
        "--test_size", default=0.2, type=float, help="Train/validation split ratio"
    )
    parser.add_argument(
        "--evaluate",
        action="store_true",
        help="Flag to run evaluation instead of training",
    )
    parser.add_argument(
        "--model_path", type=str, help="Path to the model checkpoint for evaluation"
    )

    args = parser.parse_args()
    main(args)
