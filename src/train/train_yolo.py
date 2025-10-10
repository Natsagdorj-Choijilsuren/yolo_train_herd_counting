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
    return parser.parse_args() 

def train(config: dict):
    # Load the YOLO model
    model = YOLO(config['training']['model'])  
    
    # Set up the dataset with augmentations
    data_path = config['training']['data']
    if data_path is None:
        raise ValueError("Please provide a valid path to the custom dataset folder in the config file.")
    
    dataset = AlbumentationYOLODataset()
    
    # Train the model
    model.train(
        data=dataset,
        epochs=config['training']['epochs'],
        batch_size=config['training']['batch_size'],
        imgsz=config['training']['img_size'],
        device=config['training']['device'],
        name=config['training']['name']
    )

if __name__ == "__main__": 
    pass 