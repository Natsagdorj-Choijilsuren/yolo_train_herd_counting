import argparse 
import yaml 

from ultralytics import YOLO  
from pathlib import Path

#local imports 
from src.data.yolo_datasets import (
    AlbumentationYOLODataset,
    albumentations_transform
)

from src.utils.config_helpers import load_config 

def parse_args()->argparse.Namespace:

    parser = argparse.ArgumentParser(
        description="Train a YOLO model with custom dataset and augmentations")
    
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/train_config.yaml", 
        help="Path to the training configuration file"
    )

    parser.add_argument(
        "--load_type", 
        type=str, 
        cchoices=["best", "last"],
        default="best",
        help="Type of weights to load: 'best' or 'last'"
    )


    args = parser.parse_args()
    return args
def train(config: dict,
          args: argparse.Namespace)->None:
    # Load the YOLO model
    model = YOLO(config['training']['model'])  
    pretrain_epoch = config['training']['epochs']*0.9

    model.train(
        data=config['training']['data'],
        epochs=pretrain_epoch,
        batch_size=config['training']['batch_size'],
        imgsz=config['training']['img_size'],
        workers=config['training']['workers'],
        device=config['training']['device'],
        name=config['training']['name'],
        albu=albumentations_transform,
        # use dataugmentation 
        mosaic=config['augmentations'],
        hsv_h= config['augmentations']['hsv_h'],
        hsv_s= config['augmentations']['hsv_s'],
        hsv_v= config['augmentations']['hsv_v'],
        degrees= config['augmentations']['degrees'],
        translate= config['augmentations']['translate'],
        scale= config['augmentations']['scale'],
        shear= config['augmentations']['shear'],
        perspective= config['augmentations']['perspective'],
        flipud= config['augmentations']['flipud'],
        fliplr= config['augmentations']['fliplr'],
        copy_paste= config['augmentations']['copy_paste'],
        close_mosaic= config['augmentations']['close_mosaic']
    )

    # Fine-tune the model for the remaining epochs
    model_path = f"runs/train/{config['training']['name']}/weights/{args.load_type}.pt"
    model = YOLO(model_path)  # Load the best or last model
    last_train_epoch = config['training']['epochs'] - pretrain_epoch

    model.train(
        data=config['training']['data'],
        epochs=last_train_epoch,
        batch_size=config['training']['batch_size'],
        imgsz=config['training']['img_size'],
        workers=config['training']['workers'],
        device=config['training']['device'],
        name=config['training']['name'],
        albu=None, 
        augment=False # Disable augmentations during fine-tuning
    )

if __name__ == "__main__": 
    args = parse_args()
    config = load_config(args.config)
    train(config, args)


    