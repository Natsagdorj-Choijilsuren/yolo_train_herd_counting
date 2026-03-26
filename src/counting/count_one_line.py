import argparse
import cv2
import numpy as np
import ultralytics

from ultralytics import YOLO
from typing import Tuple, List, Dict 

# local imports
from src.utils.count_helpers import get_sign

LINE_START = (100, 200)
LINE_END = (500, 200)


def get_args():

    parser = argparse.ArgumentParser(
        description="Count objects in a video stream using YOLOv8 segmentation model."
    )

    parser.add_argument(
        "--source",
        type=str,
        default="0",
        help="Video source (default is 0 for webcam).",
    )

    parser.add_argument("--dest", type=str, default="1", help="Result video file")

    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n-seg.pt",
        help="Path to the YOLOv8 segmentation model.",
    )

    parser.add_argument(
        "--conf", type=float, default=0.25, help="Confidence threshold for detections."
    )

    parser.add_argument(
        "--iou", type=float, default=0.45, help="IoU threshold for NMS."
    )

    return parser.parse_args()


def get_video_writer(
    cap: cv2.VideoCapture, output_path: str = "output.mp4"
) -> cv2.VideoWriter:

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h, fps = (
        int(cap.get(x))
        for x in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    out = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    return out


def track_count_video(args):

    global LINE_START, LINE_END

    model = YOLO(args.model)
    names = model.model.names

    tracks = []

    # Open the video source
    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    video_writer = get_video_writer(cap, output_path=args.dest)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Could not read frame.")
            break

        results = model.track(
            frame, conf=args.conf, iou=args.iou, imgsz=640, tracking="bytetrack.yaml"
        )

        if results[0].boxes is not None and results[0].boxes.id is not None:
            # Get the boxes (x, y, w, h), class IDs, track IDs, and confidences
            boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
            class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
            track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
            confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

            for box, class_id, track_id, conf in zip(
                boxes, class_ids, track_ids, confidences
            ):
                c = names[class_id]
                x1, y1, x2, y2 = box
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the box

                d = get_sign((cx, cy), LINE_START, LINE_END)

    cap.release()
    cv2.destroyAllWindows()


class HerdCounter:
    def __init__(self, args: argparse.Namespace) -> None:

        self.track_states = {}
        self.model = YOLO(args.model)
        self.names = self.model.model.names

        # dictionary count of each class and in and out
        self.count_dict = {
            self.names[i]: {"in": 0, "out": 0} for i, name in enumerate(self.names)
        }

        self.line_start = LINE_START
        self.line_end = LINE_END

    def count_video(self, args: argparse.Namespace) -> Dict[str, Dict[str, int]]:

        cap = cv2.VideoCapture(args.source)

        if not cap.isOpened():
            print(f"Error: Could not open video source {args.source}")
            return None

        video_writer = get_video_writer(cap, output_path=args.dest)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Could not read frame.")
                break

            results = self.model.track(
                frame,
                conf=args.conf,
                iou=args.iou,
                imgsz=640,
                tracking="bytetrack.yaml",
            )

            if results[0].boxes is not None and results[0].boxes.id is not None:
                boxes = results[0].boxes.xyxy.int().cpu().tolist()  # Bounding boxes
                class_ids = results[0].boxes.cls.int().cpu().tolist()  # Class IDs
                track_ids = results[0].boxes.id.int().cpu().tolist()  # Track IDs
                confidences = results[0].boxes.conf.cpu().tolist()  # Confidence score

                for box, class_id, track_id, conf in zip(
                    boxes, class_ids, track_ids, confidences
                ):
                    c = self.names[class_id]
                    x1, y1, x2, y2 = box
                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2  # Center of the box

                    d = get_sign((cx, cy), self.line_start, self.line_end)

                    if track_id not in self.track_states:
                        self.track_states[track_id] = d
                    else:
                        if self.track_states[track_id] != d and d != 0:
                            if d == 1:
                                self.count_dict[c]["in"] += 1
                            elif d == -1:
                                self.count_dict[c]["out"] += 1
                            self.track_states[track_id] = d

            # Draw the counting line
            cv2.line(frame, self.line_start, self.line_end, (0, 255, 255), 2)

            # Display counts on the frame
            y_offset = 30
            for class_name, counts in self.count_dict.items():
                text = f"{class_name} In: {counts['in']} Out: {counts['out']}"
                cv2.putText(
                    frame,
                    text,
                    (10, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2,
                )
                y_offset += 30

            video_writer.write(frame)

        return self.count_dict


if __name__ == "__main__":
    args = get_args()
    #herd_counter = HerdCounter(args)
    #count_dict = herd_counter.count_video(args)
    track_count_video(args)
    print("Final Counts:", count_dict)
