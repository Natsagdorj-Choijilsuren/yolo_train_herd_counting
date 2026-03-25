import argparse

from ultralytics import YOLO


def export_trained_model_to_tflite(
    stage2_name: str,
    img_size: int,
    export_cfg: dict,
    default_weights: str = "best",
) -> str:
    weights_name = export_cfg.get("weights", default_weights)
    model_path = f"runs/detect/{stage2_name}/weights/{weights_name}.pt"

    model = YOLO(model_path)
    return model.export(
        format="tflite",
        imgsz=img_size,
        int8=export_cfg.get("int8", False),
        half=export_cfg.get("half", False),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export trained YOLO weights to TFLite")
    parser.add_argument(
        "--stage2-name",
        type=str,
        required=True,
        help="Run name under runs/detect containing the trained weights",
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Image size to use during export",
    )
    parser.add_argument(
        "--weights",
        type=str,
        choices=["best", "last"],
        default="best",
        help="Which trained checkpoint to export",
    )
    parser.add_argument(
        "--int8",
        action="store_true",
        help="Export an INT8 TFLite model",
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="Export an FP16 TFLite model",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_trained_model_to_tflite(
        stage2_name=args.stage2_name,
        img_size=args.imgsz,
        export_cfg={
            "weights": args.weights,
            "int8": args.int8,
            "half": args.half,
        },
    )
