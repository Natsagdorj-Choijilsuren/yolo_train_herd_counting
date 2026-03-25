import argparse
import ultralytics

from ultralytics import YOLO

# local imports
from src.data.yolo_datasets import AlbumentationYOLODataset

from src.utils.config_helpers import load_config


def parse_args() -> argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train a YOLO model with custom dataset and augmentations"
    )

    parser.add_argument(
        "--config",
        type=str,
        default="src/config/train_config.yaml",
        help="Path to the training configuration file",
    )

    parser.add_argument(
        "--load_type",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Type of weights to load: 'best' or 'last'",
    )

    args = parser.parse_args()
    return args


def train(config: dict, args: argparse.Namespace) -> None:

    # Load the YOLO model
    model = YOLO(config["training"]["model"])
    pretrain_epoch = int(config["training"]["epochs"] * 0.9)
    ultralytics.data.dataset.YOLODataset = lambda *args, **kwargs: (
        AlbumentationYOLODataset(*args, use_albumentations=True, **kwargs)
    )

    model.train(
        data=config["training"]["data"],
        epochs=pretrain_epoch,
        batch=config["training"]["batch_size"],
        imgsz=config["training"]["img_size"],
        workers=config["training"]["workers"],
        device=config["training"]["device"],
        name=config["training"]["stage1_name"],
        # use dataugmentation
        mosaic=config["augmentations"]["mosaic"],
        hsv_h=config["augmentations"]["hsv_h"],
        hsv_s=config["augmentations"]["hsv_s"],
        hsv_v=config["augmentations"]["hsv_v"],
        degrees=config["augmentations"]["degrees"],
        translate=config["augmentations"]["translate"],
        scale=config["augmentations"]["scale"],
        shear=config["augmentations"]["shear"],
        perspective=config["augmentations"]["perspective"],
        flipud=config["augmentations"]["flipud"],
        fliplr=config["augmentations"]["fliplr"],
        close_mosaic=config["augmentations"]["close_mosaic"],
    )

    # Fine-tune the model for the remaining epochs
    model_path = (
        f"runs/detect/{config['training']['stage1_name']}/weights/{args.load_type}.pt"
    )
    model = YOLO(model_path)  # Load the best or last model
    last_train_epoch = config["training"]["epochs"] - pretrain_epoch

    ultralytics.data.dataset.YOLODataset = lambda *args, **kwargs: (
        AlbumentationYOLODataset(*args, use_albumentations=False, **kwargs)
    )
    model.train(
        data=config["training"]["data"],
        epochs=last_train_epoch,
        batch=config["training"]["batch_size"],
        imgsz=config["training"]["img_size"],
        workers=config["training"]["workers"],
        device=config["training"]["device"],
        name=config["training"]["stage2_name"],
        augment=False,  # Disable augmentations during fine-tuning
    )


if __name__ == "__main__":
    args = parse_args()
    config = load_config(args.config)
    train(config, args)
