import albumentations as A
import numpy as np
from ultralytics import YOLO
from ultralytics.data import YOLODataset
from pathlib import Path

from typing import List, Optional, Tuple, Union


albumentations_transform = A.Compose(
    [
        # Weather augmentations
        A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
        A.RandomRain(rain_type="drizzle", p=0.2),
        A.RandomSunFlare(
            flare_roi=(0, 0, 1, 0.5),
            angle_lower=0,
            angle_upper=1,
            num_flare_circles_lower=1,
            num_flare_circles_upper=2,
            p=0.15,
        ),
        # Motion and blur
        A.OneOf(
            [
                A.MotionBlur(blur_limit=5, p=1.0),
                A.GaussianBlur(blur_limit=5, p=1.0),
                A.MedianBlur(blur_limit=5, p=1.0),
            ],
            p=0.3,
        ),
        # Lighting and color
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.RandomShadow(
            shadow_roi=(0, 0.5, 1, 1), num_shadows_lower=1, num_shadows_upper=2, p=0.3
        ),
        A.CLAHE(clip_limit=4.0, p=0.3),
        A.HueSaturationValue(
            hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5
        ),
        # Noise and quality
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
        A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
        # Occlusion simulation
        A.CoarseDropout(
            max_holes=8,
            max_height=32,
            max_width=32,
            min_holes=1,
            min_height=8,
            min_width=8,
            fill_value=0,
            p=0.3,
        ),
    ],
    bbox_params=A.BboxParams(
        format="yolo",
        label_fields=["class_labels"],
        min_visibility=0.3,  # Remove boxes with <30% visibility after augmentation
    ),
)


class AlbumentationYOLODataset(YOLODataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # augmenttation detection
        self.albumentations_transform = A.Compose(
            [
                # Weather augmentations
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, p=0.2),
                A.RandomRain(rain_type="drizzle", p=0.2),
                A.RandomSunFlare(
                    flare_roi=(0, 0, 1, 0.5),
                    angle_lower=0,
                    angle_upper=1,
                    num_flare_circles_lower=1,
                    num_flare_circles_upper=2,
                    p=0.15,
                ),
                # Motion and blur
                A.OneOf(
                    [
                        A.MotionBlur(blur_limit=5, p=1.0),
                        A.GaussianBlur(blur_limit=5, p=1.0),
                        A.MedianBlur(blur_limit=5, p=1.0),
                    ],
                    p=0.3,
                ),
                # Lighting and color
                A.RandomBrightnessContrast(
                    brightness_limit=0.2, contrast_limit=0.2, p=0.5
                ),
                A.RandomShadow(
                    shadow_roi=(0, 0.5, 1, 1),
                    num_shadows_lower=1,
                    num_shadows_upper=2,
                    p=0.3,
                ),
                A.CLAHE(clip_limit=4.0, p=0.3),
                A.HueSaturationValue(
                    hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5
                ),
                # Noise and quality
                A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
                A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5), p=0.2),
                A.ImageCompression(quality_lower=70, quality_upper=100, p=0.3),
                # Occlusion simulation
                A.CoarseDropout(
                    max_holes=8,
                    max_height=32,
                    max_width=32,
                    min_holes=1,
                    min_height=8,
                    min_width=8,
                    fill_value=0,
                    p=0.3,
                ),
            ],
            bbox_params=A.BboxParams(
                format="yolo",
                label_fields=["class_labels"],
                min_visibility=0.3,  # Remove boxes with <30% visibility after augmentation
            ),
        )

    def __getitem__(self, index):
        """Override to apply Albumentations"""
        # Get the base item from parent class
        item = super().__getitem__(index)

        # Extract image and labels
        img = item["img"]  # Shape: (C, H, W)

        # Convert from torch tensor to numpy if needed
        if hasattr(img, "numpy"):
            img = img.numpy()

        # Convert from (C, H, W) to (H, W, C) for Albumentations
        img = np.transpose(img, (1, 2, 0))

        # Convert from float [0, 1] to uint8 [0, 255] if needed
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).astype(np.uint8)

        # Get bounding boxes
        if "bboxes" in item and len(item["bboxes"]) > 0:
            bboxes = (
                item["bboxes"].numpy()
                if hasattr(item["bboxes"], "numpy")
                else item["bboxes"]
            )

            # Extract class labels
            if "cls" in item:
                class_labels = (
                    item["cls"].numpy()
                    if hasattr(item["cls"], "numpy")
                    else item["cls"]
                )
                class_labels = class_labels.astype(np.int32).tolist()
            else:
                class_labels = [0] * len(bboxes)

            # Apply Albumentations
            try:
                transformed = self.albumentations_transform(
                    image=img, bboxes=bboxes, class_labels=class_labels
                )

                img = transformed["image"]
                bboxes = np.array(transformed["bboxes"])
                class_labels = np.array(transformed["class_labels"])

                # Update item with transformed data
                item["bboxes"] = bboxes
                item["cls"] = class_labels

            except Exception as e:
                # If augmentation fails, use original
                print(f"Albumentations failed: {e}")
                pass
        else:
            # No bboxes, just transform image
            try:
                transformed = self.albumentations_transform(
                    image=img, bboxes=[], class_labels=[]
                )
                img = transformed["image"]
            except:
                pass

        # Convert back to (C, H, W) and float [0, 1]
        img = img.astype(np.float32) / 255.0
        img = np.transpose(img, (2, 0, 1))
        item["img"] = img

        return item


if __name__ == "__main__":
    pass
